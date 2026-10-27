import json
from pathlib import Path

import numpy as np

from twinloop.centerline import extract_trees
from twinloop.config import PipelineParams
from twinloop.phantom import make_phantom
from twinloop.pipeline import V2Options, run_pipeline
from twinloop.radius_smooth import smooth_graph, smooth_trees

GOLDEN = json.loads((Path(__file__).parent / "data" / "v3_golden_stenosis.json").read_text())


def _run(tmp_path, **kw):
    img, mask = make_phantom("stenosis")
    v2 = V2Options(anatomy=False, flow_mode="diameter_law", **kw)
    run_pipeline(img, mask, PipelineParams(), tmp_path, "c_test", "stenosis.nii.gz", v2=v2)
    r = json.loads((tmp_path / "result.json").read_text())
    return {"trees": r["trees"], "lesions": r["lesions"]}


def _golden():
    return json.loads(json.dumps(GOLDEN, sort_keys=True))


def _norm(d):
    return json.loads(json.dumps(d, sort_keys=True))


def test_default_equals_golden(tmp_path):
    assert _norm(_run(tmp_path)) == _golden()


def test_zero_equals_golden(tmp_path):
    assert _norm(_run(tmp_path, radius_smooth_mm=0.0)) == _golden()


def _min_radius(res):
    return min(r for t in res["trees"] for b in t["branches"] for r in _radii(b))


def _radii(b):
    for k in ("radius_mm", "radii_mm", "r_mm"):
        if k in b:
            return b[k]
    return [b["r_min_mm"]] if "r_min_mm" in b else []


def test_smoothing_does_not_lower_min_radius(tmp_path):
    out = _run(tmp_path, radius_smooth_mm=1.0)
    g = _golden()
    if _min_radius(g) > 0:
        assert _min_radius(out) >= _min_radius(g) - 1e-9


def test_smooth_trees_in_place_noop_at_zero():
    _, mask = make_phantom("stenosis")
    trees = extract_trees(mask, PipelineParams().centerline)
    before = [b.radius_mm.copy() for t in trees for b in t.branches.values()]
    smooth_trees(trees, 0.0)
    after = [b.radius_mm for t in trees for b in t.branches.values()]
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))
    smooth_trees(trees, 1.0)
    assert min(b.min() for b in [x.radius_mm for t in trees for x in t.branches.values()]) >= min(
        b.min() for b in before
    )
    assert smooth_graph is not None
