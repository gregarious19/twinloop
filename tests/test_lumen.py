import numpy as np
import pytest
import torch
from torch import nn

from twinloop.centerline import Branch, VesselTree
from twinloop.config_v2 import V2Params
from twinloop.cpr import r_ins_2d_mm
from twinloop.graph import build_graph
from twinloop.lumen import refine_radii, slice_radii
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


def test_predict_logits_accepts_bfloat16_output():
    from twinloop.config_v2 import V2Params
    from twinloop.lumen import _predict_logits

    class Bf16Net(nn.Module):
        def forward(self, x):
            return torch.zeros_like(x, dtype=torch.bfloat16)

    out = _predict_logits(Bf16Net(), np.zeros((10, 48, 48), np.float32), V2Params().cpr, "cpu")
    assert out.shape == (10, 48, 48)


def _disc(cy, cx, rad):
    yy, xx = np.mgrid[:48, :48]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= rad**2).astype(float)


def _fake_prob():
    """Slices: 0 centre disc, 1 near off-centre disc (0.8 mm away), 2 far disc (1.6 mm), 3 centre
    disc, 4 empty."""
    return np.stack(
        [_disc(24, 24, 5), _disc(24, 24 + 6, 2), _disc(24, 24 + 10, 2), _disc(24, 24, 5)]
        + [np.zeros((48, 48))]
    )


def test_r_ins_2d_mm():
    assert r_ins_2d_mm(np.zeros((48, 48))) == 0.0
    assert r_ins_2d_mm(_disc(24, 24, 5)) == pytest.approx(1.0, abs=0.15)
    ell = np.zeros((48, 48))
    ell[20:28, 4:44] = 1  # 8 px thick bar: inscribed radius ~ 4 px = 0.8 mm
    assert r_ins_2d_mm(ell) == pytest.approx(0.8, abs=0.1)


def test_slice_radii_fallbacks():
    prob, r_v1 = _fake_prob(), np.full(5, 9.0)
    r, hit, used = slice_radii(prob, r_v1, P.cpr, "eq", "v1")
    assert hit.tolist() == [True, False, False, True, False]
    assert used.tolist() == [False, False, False, False, True]
    assert (
        r[4] == 9.0 and r[0] == pytest.approx(1.0, abs=0.15) and r[2] == pytest.approx(0.4, abs=0.1)
    )
    r, _, used = slice_radii(prob, r_v1, P.cpr, "eq", "nearest")
    assert used.tolist() == [False, False, True, False, True]
    assert r[1] == pytest.approx(0.4, abs=0.1) and r[2] == 9.0
    r, _, used = slice_radii(prob, r_v1, P.cpr, "eq", "interp")
    assert used.tolist() == [False, False, False, False, True]
    assert r[1] == pytest.approx(r[0], abs=1e-9) and r[2] == pytest.approx(r[0], abs=1e-9)
    assert r[4] == 9.0


def test_slice_radii_interp_is_linear():
    prob = np.stack([_disc(24, 24, 3), np.zeros((48, 48)), _disc(24, 24, 5)])
    r, hit, _ = slice_radii(prob, np.full(3, 9.0), P.cpr, "eq", "interp")
    assert not hit[1] and r[1] == pytest.approx(0.5 * (r[0] + r[2]))


def test_refine_radii_ins_option_and_stats():
    img, trees = stenosis_tree()
    b = main_branch(trees)
    b.radius_mm = np.full_like(b.radius_mm, 3.0)
    stats: dict = {}
    refine_radii(trees, img, HuNet(), P.cpr, "cpu", radius_out="ins", sigma=0.5, stats=stats)
    assert b.radius_mm.min() == pytest.approx(0.75, abs=0.2)
    assert np.median(b.radius_mm) == pytest.approx(1.5, abs=0.2)
    assert stats["n_slices"] > 0 and 0.0 <= stats["fallback_fraction"] <= 1.0


def test_refine_radii_defaults_equal_explicit_old_settings():
    img, t1 = stenosis_tree()
    _, t2 = stenosis_tree()
    refine_radii(t1, img, HuNet(), P.cpr, "cpu")
    refine_radii(t2, img, HuNet(), P.cpr, "cpu", radius_out="eq", fallback="v1", sigma=1.0)
    assert np.array_equal(main_branch(t1).radius_mm, main_branch(t2).radius_mm)


def test_refine_radii_rejects_bad_option():
    img, trees = stenosis_tree()
    with pytest.raises(ValueError):
        refine_radii(trees, img, HuNet(), P.cpr, "cpu", fallback="nope")
