import time

import numpy as np
import pytest
import SimpleITK as sitk

from twinloop.centerline import extract_trees
from twinloop.config import CenterlineParams
from twinloop.phantom import make_phantom

P = CenterlineParams()


def mid80(x):
    n = len(x)
    return x[int(0.1 * n) : int(0.9 * n)]


def test_tube():
    _, mask = make_phantom("tube")
    trees = extract_trees(mask, P)
    assert len(trees) == 1
    tree = trees[0]
    assert len(tree.branches) == 1
    b = tree.branches[tree.root_branch_id]
    assert 54.0 <= b.s_mm[-1] <= 61.0
    assert np.median(mid80(b.radius_mm)) == pytest.approx(1.5, abs=0.15)
    z0 = b.points_mm[0, 2]
    assert z0 < 15 or z0 > 65
    assert b.s_mm[0] == 0.0


def test_y_tree():
    _, mask = make_phantom("y_tree")
    t0 = time.perf_counter()
    trees = extract_trees(mask, P)
    assert time.perf_counter() - t0 < 10
    assert len(trees) == 1
    tree = trees[0]
    assert len(tree.branches) == 3
    root = tree.branches[tree.root_branch_id]
    assert len(root.children) == 2
    assert np.median(root.radius_mm) == pytest.approx(2.0, abs=0.2)
    assert root.points_mm[0, 2] < 12  # widest end is the root
    for cid in root.children:  # continuity at the junction
        child = tree.branches[cid]
        assert child.parent_id == root.branch_id
        assert np.allclose(child.points_mm[0], root.points_mm[-1])


def test_bfs_ids():
    _, mask = make_phantom("y_tree")
    tree = extract_trees(mask, P)[0]
    assert list(tree.branches) == ["tree_0_b0", "tree_0_b1", "tree_0_b2"]
    assert tree.root_branch_id == "tree_0_b0"


def test_two_trees():
    _, mask = make_phantom("two_trees")
    trees = extract_trees(mask, P)
    assert [t.tree_id for t in trees] == ["tree_0", "tree_1"]
    assert trees[0].volume_mm3 > trees[1].volume_mm3


def test_max_trees_and_min_component():
    _, mask = make_phantom("two_trees")
    assert len(extract_trees(mask, CenterlineParams(max_trees=1))) == 1
    assert len(extract_trees(mask, CenterlineParams(min_component_mm3=300.0))) == 1


def test_empty_mask():
    _, mask = make_phantom("tube")
    empty = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
    empty.CopyInformation(mask)
    with pytest.raises(ValueError, match="No coronary tree found in mask"):
        extract_trees(empty, P)


def test_physical_coordinates_use_origin():
    _, mask = make_phantom("tube")
    shifted = sitk.Image(mask)
    shifted.SetOrigin((100.0, -50.0, 20.0))
    a = extract_trees(mask, P)[0].branches["tree_0_b0"].points_mm
    b = extract_trees(shifted, P)[0].branches["tree_0_b0"].points_mm
    assert np.allclose(b - a, [100.0, -50.0, 20.0], atol=1e-6)
