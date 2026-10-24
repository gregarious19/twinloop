import numpy as np
import pytest

from twinloop.centerline import Branch, extract_trees
from twinloop.config import CenterlineParams, LesionParams
from twinloop.lesions import detect_lesions, reference_radius
from twinloop.phantom import make_phantom

P = LesionParams()


def make_branch(r, children=None, step=0.5):
    r = np.asarray(r, dtype=float)
    s = np.arange(len(r)) * step
    pts = np.stack([np.zeros_like(s), np.zeros_like(s), s], axis=1)
    return Branch("b0", None, pts, r, s, children or [])


def test_worked_example():
    s = np.arange(61) * 0.5
    r = np.where((s >= 12) & (s <= 18), 0.75, 1.5)
    lesions = detect_lesions(make_branch(r), P)
    assert len(lesions) == 1
    les = lesions[0]
    assert 24 <= les.min_idx <= 36
    assert les.ds == pytest.approx(0.50, abs=0.01)
    assert les.length_mm == pytest.approx(6.0, abs=0.5)
    assert les.r_ref_mm == pytest.approx(1.5)
    assert les.lesion_id == ""


def test_constant_radius_has_no_lesion():
    assert detect_lesions(make_branch(np.full(61, 1.5)), P) == []


def test_end_taper_is_excluded():
    s = np.arange(61) * 0.5
    r = np.where(s > 28, 1.5 - (s - 28) * 0.5, 1.5)  # last 2 mm taper to 0.5
    assert r[-1] == pytest.approx(0.5)
    assert detect_lesions(make_branch(r), P) == []


def test_start_of_branch_is_excluded():
    r = np.full(61, 1.5)
    r[:3] = 0.8  # s = 0 .. 1 mm
    assert detect_lesions(make_branch(r, children=["c"]), P) == []


def test_taper_at_end_of_non_terminal_branch_is_not_excluded():
    s = np.arange(61) * 0.5
    r = np.where(s > 26, 0.8, 1.5)
    assert len(detect_lesions(make_branch(r, children=["c"]), P)) == 1


def test_small_vessels_are_ignored():
    s = np.arange(61) * 0.5
    r = np.where((s >= 12) & (s <= 18), 0.375, 0.75)  # reference diameter 1.5 mm is the limit
    assert len(detect_lesions(make_branch(r), P)) == 1
    r = np.where((s >= 12) & (s <= 18), 0.3, 0.7)  # reference diameter 1.4 mm
    assert detect_lesions(make_branch(r), P) == []


def test_runs_close_together_merge():
    s = np.arange(81) * 0.5
    r = np.full(81, 1.5)
    r[(s >= 10) & (s <= 12)] = 0.8
    r[(s >= 13) & (s <= 15)] = 0.8  # gap of 1 mm
    assert len(detect_lesions(make_branch(r), P)) == 1
    r[(s >= 13) & (s <= 15)] = 1.5
    r[(s >= 16) & (s <= 18)] = 0.8  # gap of 4 mm
    assert len(detect_lesions(make_branch(r), P)) == 2


def test_short_dip_is_dropped():
    r = np.full(61, 1.5)
    r[30] = 0.8  # a single sample: length 0 < 1 mm
    assert detect_lesions(make_branch(r), P) == []


def test_reference_radius_short_branch_uses_whole_branch():
    ref = reference_radius(np.arange(7) * 0.5, np.array([1, 1, 1, 1, 1, 1, 0.2]), P)
    assert np.allclose(ref, ref[0])


def test_stenosis_phantom_end_to_end():
    _, mask = make_phantom("stenosis")
    trees = extract_trees(mask, CenterlineParams())
    assert len(trees) == 1
    found = [(b, les) for b in trees[0].branches.values() for les in detect_lesions(b, P)]
    assert len(found) == 1
    branch, les = found[0]
    assert 0.35 <= les.ds <= 0.60
    assert branch.points_mm[les.min_idx, 2] == pytest.approx(40.0, abs=2.0)
