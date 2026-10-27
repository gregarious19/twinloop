from dataclasses import replace

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from twinloop.config import CenterlineParams
from twinloop.config_v2 import DamageParams, GraphParams
from twinloop.damage import damage_case
from twinloop.graph import build_graph
from twinloop.phantom import make_phantom

TYPES = ("gap", "truncate", "delete_branch", "shrink", "false_branch", "occlusion")


def run(kind, kind_phantom="y_tree", seed=0):
    img, gt = make_phantom(kind_phantom)
    g = build_graph(gt, CenterlineParams(), GraphParams())
    w = tuple(1.0 if t == kind else 0.0 for t in TYPES)
    dp = replace(DamageParams(), n_damages_per_case=1, type_weights=w)
    out = damage_case(img, gt, g, dp, np.random.default_rng(seed))
    return img, gt, g, out


def arr(m):
    return sitk.GetArrayFromImage(m) > 0


def ncomp(m):
    return ndimage.label(arr(m), structure=np.ones((3, 3, 3)))[1]


def test_gap():
    img, gt, g, (_, out, rec) = run("gap")
    assert len(rec) == 1 and rec[0].type == "gap" and rec[0].action == "extend"
    assert arr(out).sum() < arr(gt).sum()
    assert ncomp(out) > ncomp(gt)


def test_truncate_removes_beyond_seed():
    img, gt, g, (_, out, rec) = run("truncate", "tube")
    removed = np.argwhere(arr(gt) & ~arr(out))
    z_mm = removed[:, 0] * 0.4
    seed_z = rec[0].point_mm[2]
    assert len(removed) > 0 and z_mm.min() >= seed_z - 1.0


def test_delete_branch():
    img, gt, g, (_, out, rec) = run("delete_branch")
    assert rec[0].action == "add"
    assert len(build_graph(out, CenterlineParams(), GraphParams()).trees[0].branches) < 3


def test_shrink_reduces_radius():
    img, gt, g, (_, out, rec) = run("shrink", "tube")
    p = np.array(rec[0].point_mm)
    half = rec[0].length_mm / 2.0

    def width(m):
        idx = np.argwhere(arr(m))
        z = idx[:, 0] * 0.4
        sel = np.abs(z - p[2]) <= half * 0.8
        xy = np.stack([idx[sel, 2] * 0.35, idx[sel, 1] * 0.35], axis=1)
        return np.sqrt(((xy - 28.0) ** 2).sum(1)).max()

    assert width(out) <= 0.75 * width(gt)


def test_false_branch_adds_dark_voxels():
    img, gt, g, (_, out, rec) = run("false_branch")
    added = arr(out) & ~arr(gt)
    assert rec[0].action == "remove"
    assert added.sum() == rec[0].voxels_changed >= 50
    assert (sitk.GetArrayFromImage(img)[added] <= 150).all()
    assert arr(out)[arr(gt)].all()


def test_occlusion_edits_image_only_on_removed():
    img, gt, g, (img_out, out, rec) = run("occlusion", "tube")
    a, b = sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(img_out)
    removed = arr(gt) & ~arr(out)
    assert rec[0].action == "stop" and removed.any()
    assert (a[~removed] == b[~removed]).all()
    assert abs(b[removed].mean() - 50.0) < 10.0


def test_deterministic():
    _, _, _, (i1, m1, r1) = run("gap", seed=5)
    _, _, _, (i2, m2, r2) = run("gap", seed=5)
    assert (arr(m1) == arr(m2)).all() and r1 == r2


def test_six_damages_do_not_crash():
    img, gt = make_phantom("y_tree")
    g = build_graph(gt, CenterlineParams(), GraphParams())
    _, out, rec = damage_case(img, gt, g, DamageParams(), np.random.default_rng(1))
    assert len(rec) <= 6 and [r.damage_id for r in rec] == [f"d{i}" for i in range(len(rec))]


def test_thin_bias_picks_thin_branches():
    img, gt = make_phantom("two_trees")
    g = build_graph(gt, CenterlineParams(), GraphParams())
    radius = {
        b.branch_id: float(np.median(b.radius_mm)) for t in g.trees for b in t.branches.values()
    }
    assert max(radius.values()) > 1.3 > min(radius.values())  # one thick, one thin branch
    w = tuple(1.0 if t == "truncate" else 0.0 for t in TYPES)
    dp = replace(DamageParams(), n_damages_per_case=1, type_weights=w, thin_bias_prob=1.0)
    for seed in range(8):
        _, _, rec = damage_case(img, gt, g, dp, np.random.default_rng(seed))
        assert rec and all(radius[r.branch_id] <= 1.3 for r in rec)
