import json

import pytest

from twinloop.anatomy import Anatomy
from twinloop.config import PipelineParams
from twinloop.phantom import make_phantom
from twinloop.pipeline import V2Options, run_pipeline


def fake_anatomy(*args, **kwargs):
    qc = {"aorta_present": True, "aorta_connected": True, "myo_present": True, "ostia_ok": True}
    src = dict.fromkeys(("aorta", "myocardium", "lv_cavity"), "heartchambers_highres")
    return Anatomy(None, None, None, src, qc, [], 140.0, 80.0)


def run(tmp_path, v2, kind="stenosis"):
    img, mask = make_phantom(kind)
    return run_pipeline(img, mask, PipelineParams(), tmp_path, "c_v2", f"{kind}.nii.gz", v2=v2)


def test_auto_uses_myocardial_mass(tmp_path, monkeypatch):
    monkeypatch.setattr("twinloop.anatomy.segment_anatomy", fake_anatomy)
    res = run(tmp_path, V2Options(anatomy=True, repair=False, lumen=False, flow_mode="auto"))
    assert res.v2.flow_mode == "myocardial_mass"
    assert res.v2.myocardium_ml == 140
    assert res.v2.qc["myo_present"] is True
    assert (tmp_path / "result.json").exists() and not (tmp_path / "totalseg").exists()
    again = json.loads((tmp_path / "result.json").read_text())
    assert again["v2"]["flow_mode"] == "myocardial_mass"
    # 140 mL * 1.05 g/mL * 1.0 mL/min/g, single tree
    assert res.trees[0].q_rest_ml_min == pytest.approx(147.0, abs=0.1)


def test_anatomy_off_uses_diameter_law(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not run")

    monkeypatch.setattr("twinloop.anatomy.segment_anatomy", boom)
    res = run(tmp_path, V2Options(anatomy=False))
    assert res.v2.flow_mode == "diameter_law"
    assert res.v2.qc == {}


def test_anatomy_error_does_not_fail_case(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("licence missing")

    monkeypatch.setattr("twinloop.anatomy.segment_anatomy", boom)
    res = run(tmp_path, V2Options(anatomy=True))
    assert any(w.startswith("Anatomy unavailable:") for w in res.warnings)
    assert res.v2.flow_mode == "diameter_law"
    assert not any(res.v2.qc.values())


def test_v2_none_is_v1(tmp_path):
    res = run(tmp_path, None)
    assert res.v2 is None
    assert "v2" not in json.loads((tmp_path / "result.json").read_text())


def test_repair_without_models_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TWINLOOP_DATA", str(tmp_path / "empty"))
    monkeypatch.setattr("twinloop.anatomy.segment_anatomy", fake_anatomy)
    with pytest.raises(RuntimeError, match="v2 model not available"):
        run(tmp_path / "case", V2Options(anatomy=True, repair=True))
