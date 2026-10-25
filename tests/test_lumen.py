import numpy as np
import pytest
import torch
from torch import nn

from twinloop.centerline import Branch, VesselTree
from twinloop.config_v2 import V2Params
from twinloop.graph import build_graph
from twinloop.lumen import refine_radii
from twinloop.phantom import make_phantom

P = V2Params()


class HuNet(nn.Module):
    """+10 where the de-normalised slice HU is > 200, else -10."""

    def forward(self, x):
        hu = x * 500.0 + 300.0
        return torch.where(hu > 200.0, 10.0, -10.0)


class EmptyNet(nn.Module):
    def forward(self, x):
        return torch.full_like(x, -10.0)


def stenosis_tree():
    img, mask = make_phantom("stenosis")
    g = build_graph(mask, P.centerline, P.graph)
    return img, g.trees


def main_branch(trees):
    return max(trees[0].branches.values(), key=lambda b: b.s_mm[-1])


def test_refine_recovers_stenosis_radii():
    img, trees = stenosis_tree()
    b = main_branch(trees)
    b.radius_mm = np.full_like(b.radius_mm, 3.0)  # wrong v1 radii
    assert refine_radii(trees, img, HuNet(), P.cpr, "cpu") == []
    assert b.radius_mm.min() == pytest.approx(0.75, abs=0.15)
    assert np.median(b.radius_mm) == pytest.approx(1.5, abs=0.15)


def test_empty_model_keeps_radii():
    img, trees = stenosis_tree()
    before = [b.radius_mm.copy() for b in trees[0].branches.values()]
    warnings = refine_radii(trees, img, EmptyNet(), P.cpr, "cpu")
    assert warnings
    for old, b in zip(before, trees[0].branches.values(), strict=True):
        assert np.array_equal(old, b.radius_mm)


def test_short_branch_unchanged():
    img, _ = stenosis_tree()
    s = np.arange(0, 8.01, 0.5)
    pts = np.stack([np.full_like(s, 28.0), np.full_like(s, 28.0), 20.0 + s], axis=1)
    b = Branch("tree_0_b0", None, pts, np.full_like(s, 5.0), s, [])
    tree = VesselTree("tree_0", {b.branch_id: b}, b.branch_id)
    refine_radii([tree], img, HuNet(), P.cpr, "cpu")
    assert np.all(b.radius_mm == 5.0)


def test_window_starts():
    from twinloop.lumen import _window_starts

    assert _window_starts(100, 128, 96) == [0]
    assert _window_starts(128, 128, 96) == [0]
    assert _window_starts(200, 128, 96) == [0, 72]
    assert _window_starts(300, 128, 96) == [0, 96, 172]
