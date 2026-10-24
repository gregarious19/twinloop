import numpy as np
import pytest

from twinloop.centerline import Branch, VesselTree
from twinloop.config import MMHG_TO_PA, SolverParams
from twinloop.lesions import Lesion
from twinloop.solver import (
    branch_pressure_drop,
    outlet_weights,
    resting_flow_m3s,
    solve_tree,
    turbulent_drop_pa,
)

P = SolverParams()


def straight_branch(branch_id, parent, start, direction, length_mm, radius_mm):
    n = int(round(length_mm / 0.5)) + 1
    s = np.arange(n) * 0.5
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    pts = np.asarray(start, float) + s[:, None] * d
    return Branch(branch_id, parent, pts, np.full(n, float(radius_mm)), s, [])


def make_tree(branches):
    tree = VesselTree("tree_0", {b.branch_id: b for b in branches}, branches[0].branch_id)
    for b in branches:
        if b.parent_id:
            tree.branches[b.parent_id].children.append(b.branch_id)
    return tree


def lesion_for(branch, start, end, r_min, r_ref=1.5):
    return Lesion(
        "L1",
        branch.branch_id,
        start,
        end,
        (start + end) // 2,
        r_min,
        r_ref,
        1 - r_min / r_ref,
        branch.s_mm[end] - branch.s_mm[start],
    )


def test_poiseuille_tube():
    b = straight_branch("b0", None, (0, 0, 0), (0, 0, 1), 50, 1.5)
    assert len(b.s_mm) == 101
    drop = branch_pressure_drop(b.s_mm, b.radius_mm, 1e-6, [], P)
    assert drop[-1] == pytest.approx(88.03, rel=0.005)


def test_young_tsai():
    assert turbulent_drop_pa(1e-6, 1.5, 0.75, P) == pytest.approx(145.11, rel=0.005)


def test_monotone():
    b = straight_branch("b0", None, (0, 0, 0), (0, 0, 1), 30, 1.5)
    r = b.radius_mm.copy()
    r[20:30] = 0.8
    les = lesion_for(b, 20, 30, 0.8)
    drop = branch_pressure_drop(b.s_mm, r, 1e-6, [les], P)
    assert drop[0] == 0.0
    assert np.all(np.diff(drop) >= 0)


def test_murray():
    root = straight_branch("b0", None, (0, 0, 0), (0, 0, 1), 20, 2.0)
    c1 = straight_branch("b1", "b0", (0, 0, 20), (1, 0, 1), 20, 1.5)
    c2 = straight_branch("b2", "b0", (0, 0, 20), (-1, 0, 1), 20, 1.0)
    w = outlet_weights(make_tree([root, c1, c2]), P)
    assert w["b1"] == pytest.approx(0.7714, abs=1e-3)
    assert w["b2"] == pytest.approx(0.2286, abs=1e-3)


def test_flow_law():
    b = straight_branch("b0", None, (0, 0, 0), (0, 0, 1), 50, 1.5)
    q = resting_flow_m3s(make_tree([b]), P)
    assert q * 60e6 == pytest.approx(23.55, rel=0.001)


def test_single_tube_coupled():
    b = straight_branch("b0", None, (0, 0, 0), (0, 0, 1), 50, 1.5)
    sol = solve_tree(make_tree([b]), {}, P)
    assert sol.converged
    assert sol.q_hyper_m3s * 60e6 == pytest.approx(96.96, rel=0.005)
    assert sol.ffr["b0"][-1] == pytest.approx(0.9881, abs=0.001)


def test_symmetric_y():
    root = straight_branch("b0", None, (0, 0, 0), (0, 0, 1), 30, 2.0)
    c1 = straight_branch("b1", "b0", (0, 0, 30), (1, 0, 1), 35, 1.5)
    c2 = straight_branch("b2", "b0", (0, 0, 30), (-1, 0, 1), 35, 1.5)
    sol = solve_tree(make_tree([root, c1, c2]), {}, P)
    assert sol.converged
    q1, q2, q0 = sol.flow_m3s["b1"], sol.flow_m3s["b2"], sol.flow_m3s["b0"]
    assert q1 == pytest.approx(q2, rel=1e-6)
    assert q0 == pytest.approx(q1 + q2, rel=1e-9)
    for f in sol.ffr.values():
        assert np.all(np.diff(f) <= 1e-12)
    for cid in ("b1", "b2"):
        assert sol.pressure_pa[cid][0] == pytest.approx(sol.pressure_pa["b0"][-1], abs=1e-9)
    assert sol.ffr["b0"][0] == 1.0
    assert MMHG_TO_PA * 90 == pytest.approx(sol.pressure_pa["b0"][0])


def test_stenosis_lowers_ffr():
    b = straight_branch("b0", None, (0, 0, 0), (0, 0, 1), 50, 1.5)
    base = solve_tree(make_tree([b]), {}, P).ffr["b0"][-1]
    b2 = straight_branch("b0", None, (0, 0, 0), (0, 0, 1), 50, 1.5)
    b2.radius_mm[40:52] = 0.75
    les = lesion_for(b2, 40, 52, 0.75)
    ffr = solve_tree(make_tree([b2]), {"b0": [les]}, P).ffr["b0"][-1]
    assert ffr < base
