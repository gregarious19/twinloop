import numpy as np
import pytest
import SimpleITK as sitk

from twinloop.anatomy import Anatomy, coverage_check, find_ostia, label_branches, lv_axis
from twinloop.centerline import Branch, VesselTree
from twinloop.config_v2 import AnatomyParams, GraphParams
from twinloop.graph import CoronaryGraph, endings_for_tree

AP, GP = AnatomyParams(), GraphParams()
N, SP = 160, 0.5


def make_branch(branch_id, parent, start, direction, length_mm, radius_mm):
    n = int(round(length_mm / 0.5)) + 1
    s = np.linspace(0.0, length_mm, n)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    pts = np.asarray(start, float) + s[:, None] * d
    return Branch(branch_id, parent, pts, np.full(n, float(radius_mm)), s, [])


def make_tree(tree_id, branches):
    tree = VesselTree(tree_id, {b.branch_id: b for b in branches}, branches[0].branch_id)
    for b in branches:
        if b.parent_id:
            tree.branches[b.parent_id].children.append(b.branch_id)
    return tree


def make_graph(trees, with_endings=True):
    return CoronaryGraph(
        trees=list(trees),
        fragments=[],
        endings=[e for t in trees for e in endings_for_tree(t, GP)] if with_endings else [],
        labels={b: "other" for t in trees for b in t.branches},
        tree_side={t.tree_id: "unknown" for t in trees},
        warnings=[],
    )


def grid_mm():
    zz, yy, xx = np.indices((N, N, N), dtype=float) * SP
    return xx, yy, zz


def to_img(arr):
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.SetSpacing((SP, SP, SP))
    return img


def aorta_img(zmin=0.0, zmax=80.0):
    x, y, z = grid_mm()
    return to_img(((x - 40) ** 2 + (y - 40) ** 2 <= 144.0) & (z >= zmin) & (z <= zmax))


def shell_img():
    x, y, z = grid_mm()
    r = lambda a, b, c: ((x - 40) / a) ** 2 + ((y - 40) / b) ** 2 + ((z - 40) / c) ** 2  # noqa: E731
    return to_img((r(20, 20, 40) <= 1) & (r(12, 12, 32) > 1))


def anat(aorta=None, myo=None):
    return Anatomy(
        aorta,
        myo,
        None,
        {},
        {
            "aorta_present": True,
            "myo_present": myo is not None,
            "aorta_connected": True,
            "ostia_ok": False,
        },
        [],
    )


def two_trees(left_x=52.0, right_x=28.0):
    lt = make_tree("tree_0", [make_branch("tree_0_b0", None, (left_x, 40, 40), (1, 0, 0), 30, 1.5)])
    rt = make_tree(
        "tree_1", [make_branch("tree_1_b0", None, (right_x, 40, 40), (-1, 0, 0), 30, 1.5)]
    )
    return make_graph([lt, rt])


def test_ostia_and_sides():
    g = two_trees()
    a = anat(aorta_img())
    d = find_ostia(g, a, AP)
    assert all(v is not None and v <= 1.0 for v in d.values())
    assert a.qc["ostia_ok"] is True
    label_branches(g, a, AP)
    assert g.tree_side == {"tree_0": "left", "tree_1": "right"}
    assert g.labels["tree_1_b0"] == "RCA"


def test_ostium_far_fails():
    g = two_trees(left_x=58.0)
    a = anat(aorta_img())
    d = find_ostia(g, a, AP)
    assert d["tree_0"] == pytest.approx(6.0, abs=1.0)
    assert a.qc["ostia_ok"] is False
    assert find_ostia(g, None, AP) == {"tree_0": None, "tree_1": None}


def test_single_tree_side_from_aorta():
    t = make_tree("tree_0", [make_branch("tree_0_b0", None, (52, 40, 40), (1, 0, 0), 30, 1.5)])
    g = make_graph([t])
    label_branches(g, anat(aorta_img()), AP)
    assert g.tree_side["tree_0"] == "left"
    g2 = make_graph([t])
    label_branches(g2, anat(None), AP)
    assert g2.tree_side["tree_0"] == "unknown"


def test_lv_axis_apex_far_from_aorta():
    aorta = aorta_img(65.0, 80.0)
    base, apex = lv_axis(shell_img(), aorta)
    assert apex[2] < 40.0 < base[2]
    assert lv_axis(shell_img(), None) is None


def left_tree(root_len=10.0):
    b = [
        make_branch("tree_0_b0", None, (52, 40, 60), (0, 0, -1), root_len, 2.0),
    ]
    end = b[0].points_mm[-1]
    b += [
        make_branch("tree_0_b1", "tree_0_b0", end, (0, 0, -1), 60, 1.5),
        make_branch("tree_0_b2", "tree_0_b0", end, (0, 1, 0), 40, 1.3),
        make_branch("tree_0_b3", "tree_0_b0", end, (1, 0, 0), 20, 0.8),
    ]
    return make_tree("tree_0", b)


def right_tree():
    return make_tree("tree_1", [make_branch("tree_1_b0", None, (28, 40, 50), (-1, 0, 0), 60, 1.5)])


def test_label_lm_lad_lcx():
    g = make_graph([left_tree(), right_tree()])
    label_branches(g, anat(aorta_img(65.0, 80.0), shell_img()), AP)
    assert g.labels == {
        "tree_0_b0": "LM",
        "tree_0_b1": "LAD",
        "tree_0_b2": "LCX",
        "tree_0_b3": "other",
        "tree_1_b0": "RCA",
    }
    assert g.warnings == []


def test_no_myocardium_warning():
    g = make_graph([left_tree(), right_tree()])
    label_branches(g, anat(aorta_img(65.0, 80.0), None), AP)
    assert "LAD/LCX not labelled: no myocardium" in g.warnings
    assert g.labels["tree_0_b0"] == "LM" and g.labels["tree_0_b1"] == "other"


def test_long_root_no_lm_flag():
    g = make_graph([left_tree(40.0), right_tree()])
    label_branches(g, anat(aorta_img(65.0, 80.0), shell_img()), AP)
    assert "LM not identified" in g.warnings
    assert set(g.labels[b] for b in g.labels if b.startswith("tree_0")) == {"other"}
    flags = coverage_check(g, AP)
    assert [f["kind"] for f in flags if f["kind"] == "missing_main_branch"] == [
        "missing_main_branch"
    ]


def test_coverage_rca_short_and_abrupt():
    lroot = make_branch("tree_0_b0", None, (52, 40, 60), (0, 0, -1), 10, 2.0)
    end = lroot.points_mm[-1]
    left = make_tree(
        "tree_0",
        [
            lroot,
            make_branch("tree_0_b1", "tree_0_b0", end, (0, 0, -1), 30, 1.5),
            make_branch("tree_0_b2", "tree_0_b0", end, (0, 1, 0), 30, 1.3),
        ],
    )
    right = make_tree(
        "tree_1", [make_branch("tree_1_b0", None, (28, 40, 50), (-1, 0, 0), 31.2, 1.4)]
    )
    g = make_graph([left, right], with_endings=False)
    g.endings = endings_for_tree(right, GP)
    label_branches(g, anat(aorta_img(65.0, 80.0), None), AP)
    flags = coverage_check(g, AP)
    assert [f["kind"] for f in flags] == ["short_main_vessel", "abrupt_end"]
    assert flags[0]["message"] == "RCA main path 31.2 mm < 50 mm"
    assert flags[1]["message"] == "Vessel ends abruptly (r_end 1.40 mm)"
    assert flags[0]["point_mm"] == flags[1]["point_mm"]


def test_coverage_single_tree():
    g = make_graph([right_tree()], with_endings=False)
    flags = coverage_check(g, AP)
    assert flags[0]["kind"] == "missing_tree"
    assert flags[0]["message"] == "Only 1 coronary tree(s) found"
