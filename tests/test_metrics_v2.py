import copy

import numpy as np
import pytest
import SimpleITK as sitk

from twinloop.centerline import extract_trees
from twinloop.config import CenterlineParams, LesionParams
from twinloop.config_v2 import GraphParams
from twinloop.graph import build_graph
from twinloop.lesions import detect_lesions, reference_radius
from twinloop.metrics import stenosis_errors, summarize_stenosis
from twinloop.phantom import SPACING_XYZ, _grid, _tube, make_phantom
from twinloop.radius import area_eq_radii

CP, GP, LP = CenterlineParams(), GraphParams(), LesionParams()


def _img(mask: np.ndarray) -> sitk.Image:
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    img.SetSpacing(SPACING_XYZ)
    return img


def _main_branch(trees):
    return max((b for t in trees for b in t.branches.values()), key=lambda b: b.s_mm[-1])


def test_area_eq_radii_cylinder():
    _, mask = make_phantom("tube")
    trees = extract_trees(mask, CP)
    assert area_eq_radii(trees, mask) >= 1
    assert np.median(_main_branch(trees).radius_mm) == pytest.approx(1.5, abs=0.1)


def test_area_eq_radii_elliptical_tube():
    x, y, _ = g = _grid()
    ell = ((x - 28.0) / 1.5) ** 2 + ((y - 28.0) / 0.75) ** 2 <= 1.0
    mask = _img(ell & (g[2] >= 10.0) & (g[2] <= 70.0))
    trees = extract_trees(mask, CP)
    # inscribed: EDT of voxel centres is biased high by up to a voxel (0.35 mm) on a thin tube
    assert 0.75 <= np.median(_main_branch(trees).radius_mm) <= 1.05
    area_eq_radii(trees, mask)
    assert np.median(_main_branch(trees).radius_mm) == pytest.approx(np.sqrt(1.5 * 0.75), abs=0.1)


def test_stenosis_errors_identity():
    _, mask = make_phantom("stenosis")
    g = build_graph(mask, CP, GP)
    rows = stenosis_errors(g, g, LP)
    assert len(rows) >= 1
    assert all(r["ds_err_pp"] == 0 and not r["missed"] for r in rows)
    s = summarize_stenosis(rows)
    assert s["down50"] == 0 and s["down70"] == 0


def _flattened(g_gt):
    g = copy.deepcopy(g_gt)
    for t in g.trees:
        for b in t.branches.values():
            ref = reference_radius(b.s_mm, b.radius_mm, LP)
            for les in detect_lesions(b, LP):
                sl = slice(les.start_idx, les.end_idx + 1)
                b.radius_mm[sl] = ref[sl]
    return g


def test_stenosis_errors_flattened_lesion_counts_as_down():
    x, y, z = _grid()
    deep = _tube((x, y, z), lambda zz: np.where((zz >= 37.0) & (zz <= 43.0), 0.4, 1.5))
    g_gt = build_graph(_img(deep), CP, GP)
    rows = stenosis_errors(g_gt, _flattened(g_gt), LP)
    assert len(rows) >= 1 and rows[0]["ds_gt"] >= 0.5
    assert all(r["ds_err_pp"] < 0 for r in rows)
    s = summarize_stenosis(rows)
    assert s["down50"] == s["down50_denom"] >= 1

    _, sten = make_phantom("stenosis")  # ds near 0.5: down50 follows ds_gt
    g2 = build_graph(sten, CP, GP)
    rows2 = stenosis_errors(g2, _flattened(g2), LP)
    assert all(r["ds_err_pp"] < 0 for r in rows2)
    assert summarize_stenosis(rows2)["down50"] == sum(r["ds_gt"] >= 0.5 for r in rows2)


def test_stenosis_errors_missed_when_no_pred_nearby():
    _, sten = make_phantom("stenosis")
    g = build_graph(sten, CP, GP)
    x, y, z = _grid()
    far = _tube((x - 6.0, y, z), lambda zz: 1.5)  # a tube 6 mm away from the stenosis phantom
    empty = build_graph(_img(far), CP, GP)
    rows = stenosis_errors(g, empty, LP)
    assert rows and all(r["missed"] and np.isnan(r["ds_err_pp"]) for r in rows)
    assert summarize_stenosis(rows)["down50"] == 0


def test_area_eq_junction_exclusion_y_tree():
    _, mask = make_phantom("y_tree")
    plain, fixed = extract_trees(mask, CP), extract_trees(mask, CP)
    area_eq_radii(plain, mask)
    area_eq_radii(fixed, mask, exclude_junctions=True)
    n_junction = 0
    for tp, tf in zip(plain, fixed, strict=True):
        for bp, bf in zip(tp.branches.values(), tf.branches.values(), strict=True):
            assert np.all(np.isfinite(bf.radius_mm))
            assert bf.radius_mm.max() <= bp.radius_mm.max() + 1e-6  # never adds area
            n_junction += int(bf.parent_id is not None or bool(bf.children))
    assert n_junction >= 2


def test_area_eq_junction_exclusion_no_junction_is_identity():
    _, mask = make_phantom("tube")
    a, b = extract_trees(mask, CP), extract_trees(mask, CP)
    area_eq_radii(a, mask)
    area_eq_radii(b, mask, exclude_junctions=True)
    assert np.array_equal(_main_branch(a).radius_mm, _main_branch(b).radius_mm)


def _severe_phantom():
    from twinloop.phantom import _tube

    mask = _tube(_grid(), lambda z: np.where((z >= 37.0) & (z <= 43.0), 0.4, 1.5))
    return build_graph(_img(mask), CP, GP)


def test_false_sig_lesions_phantoms():
    from twinloop.metrics import false_sig_lesions

    healthy = build_graph(make_phantom("tube")[1], CP, GP)
    severe = _severe_phantom()
    assert [les.ds for b in severe.trees[0].branches.values() for les in detect_lesions(b, LP)]
    assert (
        max(les.ds for b in severe.trees[0].branches.values() for les in detect_lesions(b, LP))
        >= 0.5
    )
    assert false_sig_lesions(healthy, healthy, LP) == 0
    assert false_sig_lesions(severe, severe, LP) == 0  # same lesion in GT and prediction
    assert false_sig_lesions(healthy, severe, LP) == 1  # lesion with no GT lesion nearby
    assert false_sig_lesions(severe, healthy, LP) == 0  # a missed lesion is not a false one
