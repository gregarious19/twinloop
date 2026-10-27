import numpy as np
import pytest
import SimpleITK as sitk

from twinloop.radius_smooth import blur_mask, smooth_radius


def test_single_sample_dip():
    s = np.arange(11) * 0.5
    r = np.full(11, 2.0)
    r[5] = 1.0
    r5 = smooth_radius(s, r, 0.5)
    exp = [2.0, 2.0, 1.9955, 1.946, 1.758, 1.6009, 1.758, 1.946, 1.9955, 2.0, 2.0]
    assert r5 == pytest.approx(exp, abs=1e-3)
    assert smooth_radius(s, r, 1.0)[5] == pytest.approx(1.7994, abs=1e-3)


def test_three_mm_narrowing_survives():
    s = np.arange(41) * 0.5
    r = np.full(41, 2.0)
    r[17:24] = 1.0
    for sig, v in ((0.5, 1.0), (1.0, 1.076), (2.0, 1.3793)):
        assert smooth_radius(s, r, sig)[20] == pytest.approx(v, abs=1e-3)


def test_constant_unchanged_and_zero_copies():
    s = np.arange(20) * 0.5
    r = np.full(20, 1.7)
    for sig in (0.5, 1.0, 2.0):
        assert smooth_radius(s, r, sig) == pytest.approx(r)
    out = smooth_radius(s, r, 0.0)
    assert out is not r and np.array_equal(out, r)


def test_blur_mask_geometry():
    a = np.zeros((20, 20, 20), np.uint8)
    a[5:15, 5:15, 5:15] = 1
    m = sitk.GetImageFromArray(a)
    m.SetSpacing((0.5, 0.6, 0.7))
    m.SetOrigin((1.0, 2.0, 3.0))
    b = blur_mask(m, 0.5)
    assert b.GetSpacing() == m.GetSpacing() and b.GetOrigin() == m.GetOrigin()
    assert b.GetDirection() == m.GetDirection()
    assert set(np.unique(sitk.GetArrayFromImage(b))) <= {0, 1}
