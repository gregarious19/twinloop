import time
import zipfile

import pytest
import SimpleITK as sitk
from fastapi.testclient import TestClient

from app import server
from twinloop.io import save_nifti
from twinloop.phantom import make_phantom, write_dicom_series
from twinloop.schema import CaseResult


@pytest.fixture(autouse=True)
def _no_anatomy(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("disabled in tests")

    monkeypatch.setattr("twinloop.anatomy.segment_anatomy", _raise)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TWINLOOP_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "no_results"))
    monkeypatch.setenv("TWINLOOP_NNUNET_TRAINER", "NoSuchTrainer")
    with TestClient(server.create_app()) as c:
        yield c


@pytest.fixture
def files(tmp_path):
    out = {}
    for kind in ("stenosis", "tube"):
        img, mask = make_phantom(kind)
        save_nifti(img, tmp_path / f"{kind}_img.nii.gz")
        save_nifti(mask, tmp_path / f"{kind}_mask.nii.gz")
        out[kind] = (tmp_path / f"{kind}_img.nii.gz", tmp_path / f"{kind}_mask.nii.gz")
    return out


def post(client, image, mask=None, **data):
    handles = {"image": (image.name, image.open("rb"))}
    if mask is not None:
        handles["mask"] = (mask.name, mask.open("rb"))
    try:
        return client.post("/api/cases", files=handles, data=data)
    finally:
        for _, fh in handles.values():
            fh.close()


def wait_final(client, case_id, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/cases/{case_id}").json()
        if status["status"] in ("done", "failed"):
            return status
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"status", "cuda", "model_available", "queue_length"}
    assert body["status"] == "ok"
    assert body["model_available"] is False


def test_upload_with_mask_end_to_end(client, files):
    img, mask = files["stenosis"]
    r = post(client, img, mask)
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued"
    status = wait_final(client, body["case_id"])
    assert status["status"] == "done", status
    assert status["has_mask"] is True
    res = client.get(f"/api/cases/{body['case_id']}/result")
    assert res.status_code == 200
    result = CaseResult.model_validate(res.json())
    assert len(result.lesions) == 1
    assert result.summary.min_ffr < 0.99
    assert result.input.has_mask is True


def test_aortic_pressure_is_used(client, files):
    img, mask = files["stenosis"]
    ids = {}
    for p in (70, 120):
        ids[p] = post(client, img, mask, aortic_pressure_mmhg=str(p)).json()["case_id"]
        assert wait_final(client, ids[p])["aortic_pressure_mmhg"] == p
    results = {p: client.get(f"/api/cases/{i}/result").json() for p, i in ids.items()}
    assert results[70]["params"]["solver"]["aortic_pressure_mmhg"] == 70
    # lower driving pressure -> lower flow in the same tree
    assert results[70]["trees"][0]["q_hyper_ml_min"] < results[120]["trees"][0]["q_hyper_ml_min"]


def test_dicom_zip_upload(client, files, tmp_path):
    img, _ = make_phantom("tube")
    dcm = tmp_path / "dcm"
    write_dicom_series(img, dcm)
    zpath = tmp_path / "series.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for f in sorted(dcm.iterdir()):
            zf.write(f, f"series/{f.name}")
    r = post(client, zpath, files["tube"][1])
    assert r.status_code == 202
    status = wait_final(client, r.json()["case_id"])
    assert status["status"] == "done", status
    assert status["filename"] == "series.zip"


def test_unsupported_file_type(client, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    r = post(client, bad)
    assert r.status_code == 400
    assert r.json() == {"detail": "Only .nii.gz, .nii or .zip files are supported"}


def test_pressure_out_of_range(client, files):
    for value in ("200", "59.9"):
        r = post(client, files["tube"][0], files["tube"][1], aortic_pressure_mmhg=value)
        assert r.status_code == 400
        assert r.json() == {"detail": "aortic_pressure_mmhg must be between 60 and 140"}


def test_no_mask_and_no_model_fails_clearly(client, files):
    r = post(client, files["tube"][0])
    assert r.status_code == 202
    status = wait_final(client, r.json()["case_id"])
    assert status["status"] == "failed"
    assert "Segmentation model not available" in status["error"]
    assert client.get(f"/api/cases/{r.json()['case_id']}/result").status_code == 409


def test_bad_mask_size_fails(client, files, tmp_path):
    small = tmp_path / "small_mask.nii.gz"
    save_nifti(sitk.Image(8, 8, 8, sitk.sitkUInt8), small)
    r = post(client, files["tube"][0], small)
    status = wait_final(client, r.json()["case_id"])
    assert status["status"] == "failed"
    assert status["error"] == "Mask shape does not match image"


@pytest.mark.parametrize("case_id", ["nope", "c1_abc", "c20990101000000_abcdef", "..%2Fetc"])
def test_unknown_case_id(client, case_id):
    for path in ("", "/result"):
        r = client.get(f"/api/cases/{case_id}{path}")
        assert r.status_code == 404


def test_result_not_ready_then_done(client, files):
    img, mask = files["stenosis"]
    case_id = post(client, img, mask).json()["case_id"]
    r = client.get(f"/api/cases/{case_id}/result")
    if r.status_code != 200:  # may race to done on a fast machine
        assert r.status_code == 409
        assert r.json() == {"detail": "Result not ready"}
    wait_final(client, case_id)
    assert client.get(f"/api/cases/{case_id}/result").status_code == 200


def test_list_newest_first(client, files):
    ids = []
    for _ in range(2):
        ids.append(post(client, *files["tube"]).json()["case_id"])
        time.sleep(1.1)  # case ids carry second resolution
    listed = [c["case_id"] for c in client.get("/api/cases").json()]
    assert listed == ids[::-1]
    wait_final(client, ids[-1])


def test_upload_size_limit(client, files, monkeypatch):
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 1000)
    r = post(client, *files["tube"])
    assert r.status_code == 413
    assert r.json() == {"detail": "File larger than 1 GB"}
    assert client.get("/api/cases").json() == []


def test_restart_marks_unfinished_cases_failed(tmp_path, monkeypatch):
    from twinloop.casefiles import initial_status, write_json_atomic

    monkeypatch.setenv("TWINLOOP_DATA", str(tmp_path / "data"))
    case = tmp_path / "data" / "runs" / "cases" / "c20260101000000_aaaaaa"
    case.mkdir(parents=True)
    write_json_atomic(case / "status.json", initial_status(case.name, "x.nii.gz", False, 90.0))
    with TestClient(server.create_app()) as c:
        status = c.get(f"/api/cases/{case.name}").json()
    assert status["status"] == "failed"
    assert status["error"] == "Interrupted by server restart"


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "TwinLoop" in r.text
