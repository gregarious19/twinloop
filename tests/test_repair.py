import json
import sys

import joblib
import numpy as np
import pytest
import SimpleITK as sitk
import torch
from scipy import ndimage
from sklearn.dummy import DummyClassifier
from torch import nn

from twinloop.config_v2 import V2Params
from twinloop.gapfill import apply_gapfill, extract_patch
from twinloop.graph import build_graph
from twinloop.modelio import load_model
from twinloop.phantom import make_phantom
from twinloop.regions import ACTIONS, FEATURE_NAMES
from twinloop.repair import RulesDecider, run_repair

P = V2Params()


class OracleNet(nn.Module):
    """Returns +10 where the GT is 1 on the patch grid of the most recent set_reference call."""

    def __init__(self, gt: sitk.Image):
        super().__init__()
        self.gt = gt
        self.ref_mask = None

    def set_reference(self, ref: sitk.Image) -> None:
        res = sitk.Resample(self.gt, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
        self.ref_mask = torch.from_numpy((sitk.GetArrayFromImage(res) > 0).astype(np.float32))

    def forward(self, x):
        out = torch.where(self.ref_mask > 0, 10.0, -10.0)
        return out[None, None].repeat(x.shape[0], 1, 1, 1, 1)


class ConstNet(nn.Module):
    def forward(self, x):
        return torch.full((x.shape[0], 1, 64, 64, 64), -10.0)


def ncomp(m):
    return ndimage.label(sitk.GetArrayFromImage(m) > 0, structure=np.ones((3, 3, 3)))[1]


def gap_pair():
    img, damaged = make_phantom("two_trees_gap")
    _, gt = make_phantom("two_trees")
    return img, damaged, gt


def test_extract_patch_on_tube():
    img, gt = make_phantom("tube")
    seed = np.array([28.0, 28.0, 40.0])
    p = extract_patch(img, gt, seed, np.array([0, 0, 1.0]), 0.0, P.gapfill)
    x = p["x"]
    assert x.shape == (3, 64, 64, 64) and x.dtype == np.float32
    zi, yi, xi = np.unravel_index(np.argmax(x[2]), x[2].shape)
    assert all(abs(v - 31.5) <= 1.0 for v in (zi, yi, xi))
    assert x[0].min() >= -1.0 and x[0].max() <= 1.0
    assert x[1][32, 32, 32] == 1.0


def test_apply_gapfill_oracle_closes_gap():
    img, damaged, gt = gap_pair()
    g = build_graph(damaged, P.centerline, P.graph)
    end = min(g.endings, key=lambda e: abs(e.point_mm[2] - 38.0) + abs(e.point_mm[0] - 28.0))
    out, n = apply_gapfill(OracleNet(gt), img, damaged, end.point_mm, end.direction, 8.0,
                           P.gapfill, "cpu")  # fmt: skip
    assert n > 20
    assert ncomp(out) == ncomp(damaged) - 1


def test_apply_gapfill_nothing_found():
    img, damaged, _ = gap_pair()
    out, n = apply_gapfill(ConstNet(), img, damaged, [28, 28, 36.8], [0, 0, 1.0], 8.0,
                           P.gapfill, "cpu")  # fmt: skip
    assert n == 0 and out is damaged


def test_run_repair_with_oracle():
    img, damaged, gt = gap_pair()
    models = {"decider": RulesDecider(), "gapfill": OracleNet(gt)}
    out, actions, flags = run_repair(img, damaged, None, models, P, "cpu")
    assert ncomp(out) < ncomp(damaged)
    assert any(a.action == "extend" and a.result == "applied" for a in actions)
    models["gapfill"] = None
    _, actions2, _ = run_repair(img, damaged, None, models, P, "cpu")
    assert actions2 and all(a.result == "flagged" for a in actions2)
    assert any(a.action == "extend" for a in actions2)


def test_rules_decider_rows():
    x = np.zeros((2, len(FEATURE_NAMES)))
    x[0, FEATURE_NAMES.index("kind_ending")] = 1
    x[0, FEATURE_NAMES.index("is_abrupt")] = 1
    pr = RulesDecider().predict_proba(x)
    assert pr[0, ACTIONS.index("extend")] == 1.0 and pr[1, ACTIONS.index("keep")] == 1.0


def test_load_model_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("TWINLOOP_DATA", str(tmp_path))
    with pytest.raises(RuntimeError, match="v2 model not available"):
        load_model("gapfill", "cpu")
    d = tmp_path / "models" / "v2" / "decider"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"feature_names": ["wrong"]}))
    with pytest.raises(RuntimeError, match="decider feature names do not match"):
        load_model("decider", "cpu")
    (d / "config.json").write_text(json.dumps({"feature_names": FEATURE_NAMES}))
    clf = DummyClassifier().fit(np.zeros((2, 31)), [0, 1])
    joblib.dump(clf, d / "decider.joblib")
    assert load_model("decider", "cpu") is not None
    g = tmp_path / "models" / "v2" / "gapfill"
    g.mkdir()
    (g / "config.json").write_text(json.dumps({"arch": "GapfillNet", "arch_kwargs": {}}))
    import twinloop

    monkeypatch.delattr(twinloop, "nets", raising=False)
    monkeypatch.setitem(sys.modules, "twinloop.nets", None)
    with pytest.raises(RuntimeError, match="v2 model not available: twinloop.nets not found"):
        load_model("gapfill", "cpu")


def test_full_proba_pads_missing_classes():
    from sklearn.ensemble import HistGradientBoostingClassifier

    from twinloop.modelio import FullProba

    X = np.random.default_rng(0).normal(size=(60, 3))
    y = np.array([0, 1, 4] * 20)  # classes 2, 3, 5 never occur
    p = FullProba(HistGradientBoostingClassifier(max_iter=5).fit(X, y), 6).predict_proba(X[:4])
    assert p.shape == (4, 6) and np.allclose(p.sum(1), 1.0) and (p[:, [2, 3, 5]] == 0).all()
