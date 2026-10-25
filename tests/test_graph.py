import json

import numpy as np
import pytest
import SimpleITK as sitk

from twinloop.config import CenterlineParams
from twinloop.config_v2 import GraphParams
from twinloop.graph import build_graph, graph_summary, main_path
from twinloop.phantom import make_phantom

CP, GP = CenterlineParams(), GraphParams()


def _graph(kind):
    return build_graph(make_phantom(kind)[1], CP, GP)


def _root(g, i=0):
    t = g.trees[i]
    return t.branches[t.root_branch_id].points_mm[0]


def test_tube_has_abrupt_distal_ending():
    g = _graph("tube")
    assert len(g.trees) == 1 and not g.fragments
    (e,) = g.endings
    assert e.kind == "abrupt"
    assert e.r_end_mm == pytest.approx(1.5, abs=0.3)
    assert np.linalg.norm(e.point_mm - _root(g)) > 40.0
    assert e.direction[2] * np.sign(e.point_mm[2] - _root(g)[2]) > 0.9


def test_taper_ending_is_taper():
    g = _graph("taper")
    far = max(g.endings, key=lambda e: np.linalg.norm(e.point_mm - _root(g)))
    assert far.kind == "taper"


def test_two_trees_gap_gives_fragment():
    g = _graph("two_trees_gap")
    assert len(g.trees) == 2
    assert len(g.fragments) == 1
    assert len(g.endings) >= 3


def test_y_tree_main_path_and_summary():
    g = _graph("y_tree")
    t = g.trees[0]
    path = main_path(t, t.root_branch_id)
    assert len(path) == 2 and path[0] == t.root_branch_id
    s = graph_summary(g)
    txt = json.dumps(s)
    vals = [v for e in s["endings"] for v in (*e["point_mm"], e["r_end_mm"], e["taper_ratio"])]
    assert vals and all(round(v, 2) == v for v in vals)
    assert s["n_trees"] == 1 and set(s["labels"].values()) == {"other"}
    assert txt


def test_empty_mask_message():
    img, _ = make_phantom("tube")
    empty = sitk.Image(img.GetSize(), sitk.sitkUInt8)
    empty.CopyInformation(img)
    with pytest.raises(ValueError, match="No coronary tree found in mask"):
        build_graph(empty, CP, GP)
