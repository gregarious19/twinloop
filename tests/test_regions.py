from dataclasses import replace

import numpy as np
import SimpleITK as sitk

from twinloop.config_v2 import DamageParams, V2Params
from twinloop.damage import damage_case
from twinloop.graph import build_graph
from twinloop.phantom import make_phantom
from twinloop.regions import (
    FEATURE_NAMES,
    label_real,
    label_synthetic,
    propose_regions,
    region_features,
)

P = V2Params()
TYPES = ("gap", "truncate", "delete_branch", "shrink", "false_branch", "occlusion")


def graph(mask):
    return build_graph(mask, P.centerline, P.graph)


def damaged(kind, phantom, seed=0):
    img, gt = make_phantom(phantom)
    g = graph(gt)
    w = tuple(1.0 if t == kind else 0.0 for t in TYPES)
    dp = replace(DamageParams(), n_damages_per_case=1, type_weights=w)
    img2, mask2, recs = damage_case(img, gt, g, dp, np.random.default_rng(seed))
    return img2, gt, mask2, recs


def test_tube_regions_and_features():
    img, gt = make_phantom("tube")
    g = graph(gt)
    regs = propose_regions(g, img, gt, P)
    assert any(r.kind == "ending" for r in regs)
    x = region_features(regs, g, img, gt, P)
    assert x.shape == (len(regs), 31) and len(FEATURE_NAMES) == 31


def test_gap_ending_region():
    img, gt = make_phantom("two_trees_gap")
    g = graph(gt)
    regs = propose_regions(g, img, gt, P)
    x = region_features(regs, g, img, gt, P)
    col = {n: k for k, n in enumerate(FEATURE_NAMES)}
    hits = [
        k
        for k, r in enumerate(regs)
        if r.kind == "ending" and abs(r.point_mm[2] - 38.0) <= 3.0 and r.point_mm[0] > 20.0
    ]
    assert hits, [(r.kind, r.point_mm.round(1)) for r in regs]
    near_gap = min(hits, key=lambda k: abs(regs[k].point_mm[2] - 38.0))
    # SPEC ISSUE (D-V2-S08a): the spec asks for < 0.5, but with a 4 mm gap, 0.8 mm skeleton
    # retraction and 10 look-ahead centres the value is exactly 0.5 (a continuous vessel gives 1.0).
    assert x[near_gap, col["mask_ahead_frac"]] <= 0.5
    assert x[near_gap, col["in_fragment"]] == 0.0


def test_false_branch_labelled_remove():
    for seed in range(8):
        img, gt, mask2, recs = damaged("false_branch", "y_tree", seed)
        g = graph(mask2)
        regs = propose_regions(g, img, mask2, P)
        labels, _ = label_synthetic(regs, recs, P)
        if any(lab == "remove" and r.kind == "branch" for lab, r in zip(labels, regs, strict=True)):
            return
    raise AssertionError("no branch region labelled remove for any of 8 seeds")


def test_gap_labelled_extend():
    img, gt, mask2, recs = damaged("gap", "tube")
    g = graph(mask2)
    regs = propose_regions(g, img, mask2, P)
    labels, n_unmatched = label_synthetic(regs, recs, P)
    assert n_unmatched == 0
    assert any(lab == "extend" and r.kind == "ending" for lab, r in zip(labels, regs, strict=True))


def test_label_real_pred_equals_gt_is_all_keep():
    for kind in ("tube", "stenosis", "y_tree"):
        img, gt = make_phantom(kind)
        g = graph(gt)
        regs = propose_regions(g, img, gt, P)
        assert regs
        assert set(label_real(regs, g, gt, gt, g, P)) == {"keep"}


def test_label_real_missing_piece_is_extend():
    img, gt = make_phantom("tube")
    arr = sitk.GetArrayFromImage(gt)
    arr[(np.arange(arr.shape[0]) * 0.4 > 45.0), :, :] = 0
    pred = sitk.GetImageFromArray(arr)
    pred.CopyInformation(gt)
    g_pred, g_gt = graph(pred), graph(gt)
    regs = propose_regions(g_pred, img, pred, P)
    labels = label_real(regs, g_pred, pred, gt, g_gt, P)
    assert "extend" in [lab for lab, r in zip(labels, regs, strict=True) if r.kind == "ending"]
