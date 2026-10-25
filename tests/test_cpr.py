import numpy as np
import pytest
import SimpleITK as sitk

from twinloop.config import CenterlineParams
from twinloop.config_v2 import CprParams, GraphParams
from twinloop.cpr import center_component, phys_to_index_zyx, r_eq_mm, rmf, straighten
from twinloop.graph import build_graph
from twinloop.phantom import make_phantom

CPR = CprParams()


def test_phys_to_index_roundtrip_flipped_direction():
    img = sitk.Image(20, 30, 40, sitk.sitkFloat32)
    img.SetSpacing((0.4, 0.5, 0.7))
    img.SetOrigin((10.0, -5.0, 3.0))
    img.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    rng = np.random.default_rng(0)
    cont = rng.uniform(0, 19, size=(25, 3))  # (x, y, z) continuous indices
    pts = np.array([img.TransformContinuousIndexToPhysicalPoint(tuple(c)) for c in cont])
    got = phys_to_index_zyx(pts, img)
    assert np.abs(got - cont[:, ::-1]).max() < 1e-6


def test_rmf_on_helix():
    t = np.arange(0, 60, 0.5)
    w = 0.3
    pts = np.stack([5 * np.cos(w * t), 5 * np.sin(w * t), 0.4 * t], axis=1)
    T, U, V = rmf(pts)
    assert np.abs((T * U).sum(1)).max() < 1e-6
    assert np.abs(np.linalg.norm(U, axis=1) - 1).max() < 1e-9
    assert np.abs(np.linalg.norm(V, axis=1) - 1).max() < 1e-9
    cosang = np.clip((U[:-1] * U[1:]).sum(1), -1, 1)
    assert np.degrees(np.arccos(cosang)).max() < 10.0


def _straightened(kind):
    img, mask = make_phantom(kind)
    g = build_graph(mask, CenterlineParams(), GraphParams())
    b = max(g.trees[0].branches.values(), key=lambda b: b.s_mm[-1])
    return straighten(img, mask, b, CPR)


def _r_eq(res):
    return np.array([r_eq_mm(center_component(m)) for m in res["mask"]])


def test_tube_straightened():
    res = _straightened("tube")
    r = _r_eq(res)
    assert r[len(r) // 2] == pytest.approx(1.5, abs=0.15)
    assert res["img"][len(r) // 2, 24, 24] == pytest.approx(400.0, abs=1.0)
    assert res["img"].shape[1:] == (48, 48) and res["mask"].dtype == np.uint8


def test_stenosis_min_radius():
    r = _r_eq(_straightened("stenosis"))
    assert r.min() == pytest.approx(0.75, abs=0.15)


def test_center_component_nearest_and_empty():
    m = np.zeros((48, 48), np.uint8)
    assert not center_component(m).any()
    m[14:17, 14:17] = 1  # nearest pixel (16, 16): distance ~11.3 from the centre
    m[40:44, 40:44] = 1  # nearest pixel (40, 40): distance ~22.6
    c = center_component(m)
    assert c[14, 14] and not c[40, 40]
    m[22:26, 22:26] = 1  # component containing the centre pixel wins
    assert center_component(m)[24, 24] and not center_component(m)[14, 14]
