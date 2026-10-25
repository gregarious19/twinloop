import numpy as np
import pytest
import SimpleITK as sitk

from twinloop import anatomy
from twinloop.config_v2 import AnatomyParams

AP = AnatomyParams()


def _img():
    img = sitk.GetImageFromArray(np.zeros((100, 100, 100), np.float32))
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def _mask(arr):
    m = sitk.GetImageFromArray(arr.astype(np.uint8))
    m.SetSpacing((1.0, 1.0, 1.0))
    return m


def _block(z0, z1, y0, y1, x0, x1):
    a = np.zeros((100, 100, 100), np.uint8)
    a[z0:z1, y0:y1, x0:x1] = 1
    return a


AORTA = _block(0, 50, 10, 30, 10, 30)  # 20 mL
MYO = _block(0, 60, 40, 80, 40, 80)  # 96 mL


def test_highres_success(tmp_path, monkeypatch):
    def fake(image, workdir, task, timeout_s):
        assert task == "heartchambers_highres"
        return {
            "aorta": _mask(AORTA),
            "heart_myocardium": _mask(MYO),
            "heart_ventricle_left": _mask(_block(10, 30, 50, 60, 50, 60)),
        }

    monkeypatch.setattr(anatomy, "run_totalseg", fake)
    a = anatomy.segment_anatomy(_img(), tmp_path, AP)
    assert set(a.source.values()) == {"heartchambers_highres"}
    assert a.qc == {
        "aorta_present": True,
        "aorta_connected": True,
        "myo_present": True,
        "ostia_ok": False,
    }
    assert a.lv_cavity is not None
    assert a.myo_volume_ml == pytest.approx(96.0)
    assert a.aorta_volume_ml == pytest.approx(20.0)


def test_licence_error_falls_back_to_total(tmp_path, monkeypatch):
    def fake(image, workdir, task, timeout_s):
        if task == "heartchambers_highres":
            raise RuntimeError("licence")
        return {"aorta": _mask(AORTA)}

    monkeypatch.setattr(anatomy, "run_totalseg", fake)
    a = anatomy.segment_anatomy(_img(), tmp_path, AP)
    assert a.source["aorta"] == "total"
    assert a.myocardium is None and a.lv_cavity is None
    assert a.qc["aorta_present"] and not a.qc["myo_present"]
    assert any("licence" in m for m in a.qc_messages)


def test_two_piece_aorta_not_connected(tmp_path, monkeypatch):
    arr = _block(0, 70, 10, 30, 10, 30)  # 28 mL, 70 %
    arr |= _block(0, 30, 60, 80, 10, 30)  # 12 mL, 30 %
    monkeypatch.setattr(
        anatomy,
        "run_totalseg",
        lambda *a, **k: {"aorta": _mask(arr), "heart_myocardium": _mask(MYO)},
    )
    a = anatomy.segment_anatomy(_img(), tmp_path, AP)
    assert a.qc["aorta_present"] and not a.qc["aorta_connected"]


def test_small_aorta_not_present(tmp_path, monkeypatch):
    small = _block(0, 25, 10, 30, 10, 30)  # 10 mL
    monkeypatch.setattr(
        anatomy,
        "run_totalseg",
        lambda *a, **k: {"aorta": _mask(small), "heart_myocardium": _mask(MYO)},
    )
    a = anatomy.segment_anatomy(_img(), tmp_path, AP)
    assert not a.qc["aorta_present"]


def test_all_fail_gives_missing(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(anatomy, "run_totalseg", boom)
    a = anatomy.segment_anatomy(_img(), tmp_path, AP)
    assert a.aorta is None and set(a.source.values()) == {"missing"}
    assert not any(a.qc.values())


@pytest.mark.gpu
@pytest.mark.data
def test_real_case_anatomy(tmp_path):
    from twinloop.io import load_image
    from twinloop.paths import case_image_path
    from twinloop.v2split import load_roles

    n = min(c for c, r in load_roles().items() if r == "v2_train")
    a = anatomy.segment_anatomy(load_image(case_image_path(n)), tmp_path, AP)
    print("myocardium mL:", a.myo_volume_ml, a.source, a.qc_messages)
    # SPEC ISSUE (DECISIONS_V2 D-V2-S01c): aorta_connected is False on real cases whose FOV cuts
    # the arch (two large pieces), so it is reported, not asserted.
    print("qc:", a.qc)
    assert a.qc["aorta_present"] and a.qc["myo_present"]
