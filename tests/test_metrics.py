import numpy as np
import pytest
import SimpleITK as sitk

from twinloop.config import CenterlineParams, LesionParams
from twinloop.config_v2 import GraphParams
from twinloop.graph import build_graph
from twinloop.metrics import (
    branch_recall,
    cl_dice,
    dice,
    false_branch_rate,
    false_branches,
    stenosis_metrics,
)
from twinloop.phantom import _cylinder, _grid, make_phantom

CP, GP, LP = CenterlineParams(), GraphParams(), LesionParams()


def graph(mask):
    return build_graph(mask, CP, GP)


def arr(img):
    return sitk.GetArrayFromImage(img) > 0


def with_voxels(img, new_arr):
    out = sitk.GetImageFromArray(new_arr.astype(np.uint8))
    out.CopyInformation(img)
    return out


def test_dice_cldice_basic():
    _, tube = make_phantom("tube")
    a = arr(tube)
    assert dice(a, a) == 1.0 and cl_dice(a, a) == pytest.approx(1.0)
    _, two = make_phantom("two_trees")
    disjoint_p = a
    disjoint_g = arr(two) & ~a
    assert dice(disjoint_p, disjoint_g) == 0.0 and cl_dice(disjoint_p, disjoint_g) == 0.0
    empty = np.zeros_like(a)
    assert dice(empty, empty) == 1.0


def test_branch_recall_child_removed():
    _, gt = make_phantom("y_tree")
    grid = _grid()
    junction = np.array([28.0, 28.0, 35.0])
    tilt = np.deg2rad(30.0)
    end = junction + 35.0 * np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    # remove one child (beyond a 3 mm margin from the junction)
    child = _cylinder(grid, junction + 3.0 * (end - junction) / 35.0, end, 2.0)
    pred = with_voxels(gt, arr(gt) & ~child)
    g_gt = graph(gt)
    assert branch_recall(g_gt, gt) == 1.0
    assert branch_recall(g_gt, pred) == pytest.approx(2 / 3, abs=0.01)


def test_false_branch_rate():
    _, gt = make_phantom("tube")
    grid = _grid()
    side = _cylinder(grid, (28, 28, 40), (43, 28, 40), 1.0)
    pred = with_voxels(gt, arr(gt) | side)
    g_pred = graph(pred)
    bad = false_branches(g_pred, gt)
    assert len(bad) == 1
    b = next(b for t in g_pred.trees for b in t.branches.values() if b.branch_id == bad[0])
    assert b.points_mm[:, 0].mean() > 30.0
    assert false_branch_rate(g_pred, gt) > 0
    assert false_branch_rate(graph(gt), gt) == 0.0


def test_widened_rate_stenosis_vs_tube():
    _, sten = make_phantom("stenosis")
    _, tube = make_phantom("tube")
    g_gt, g_tube = graph(sten), graph(tube)
    m = stenosis_metrics(g_gt, g_tube, LP)
    assert m["n_lesions"] >= 1
    assert m["widened_rate"] == 1.0
    same = stenosis_metrics(g_gt, g_gt, LP)
    assert same["widened_rate"] == 0.0 and same["mean_abs_rmin_err_mm"] == pytest.approx(0.0)
