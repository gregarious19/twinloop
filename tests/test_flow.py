import pytest
from test_solver import lesion_for, make_tree, straight_branch

from twinloop.anatomy import Anatomy
from twinloop.config import SolverParams
from twinloop.config_v2 import FlowParams
from twinloop.flow import (
    MYO_UNAVAILABLE_WARNING,
    NEEDS_MYO_ERROR,
    ffr_uncertainty_flags,
    min_ffr,
    tree_rest_flows,
)
from twinloop.solver import solve_tree

SP = SolverParams()


def anat(myo_ml=140.0, present=True):
    return Anatomy(None, None, None, {}, {"myo_present": present}, [], myo_ml, None)


def tree_with_r3(tree_id, r3_mm3):
    b = straight_branch(f"{tree_id}_b0", None, (0, 0, 0), (0, 0, 1), 20, r3_mm3 ** (1 / 3))
    t = make_tree([b])
    t.tree_id = tree_id
    return t


def test_worked_example():
    trees = [tree_with_r3("tree_0", 30.0), tree_with_r3("tree_1", 15.0)]
    q, mode = tree_rest_flows(trees, anat(140.0), SP, FlowParams(flow_mode="myocardial_mass"))
    assert mode == "myocardial_mass"
    to_ml_min = 1e6 * 60.0
    assert q["tree_0"] * to_ml_min == pytest.approx(98.0, abs=0.1)
    assert q["tree_1"] * to_ml_min == pytest.approx(49.0, abs=0.1)


def test_auto_without_myocardium_falls_back():
    trees = [tree_with_r3("tree_0", 30.0)]
    for a in (None, anat(present=False)):
        q, mode = tree_rest_flows(trees, a, SP, FlowParams(flow_mode="auto"))
        assert q is None and mode == "diameter_law"
    assert MYO_UNAVAILABLE_WARNING == "Myocardium unavailable; using diameter flow law"
    with pytest.raises(ValueError, match=NEEDS_MYO_ERROR):
        tree_rest_flows(trees, None, SP, FlowParams(flow_mode="myocardial_mass"))
    q, mode = tree_rest_flows(trees, anat(), SP, FlowParams(flow_mode="auto"))
    assert mode == "myocardial_mass"


def test_override_changes_solution():
    b = straight_branch("b0", None, (0, 0, 0), (0, 0, 1), 50, 1.5)
    t = make_tree([b])
    base = solve_tree(t, {}, SP)
    over = solve_tree(t, {}, SP, q_rest_override_m3s=2 * base.q_rest_m3s)
    assert over.q_rest_m3s == 2 * base.q_rest_m3s
    assert over.q_hyper_m3s > base.q_hyper_m3s


def stenosed_tube(r_min):
    b = straight_branch("b0", None, (0, 0, 0), (0, 0, 1), 50, 1.5)
    b.radius_mm[40:61] = r_min
    return make_tree([b]), {"b0": [lesion_for(b, 40, 60, r_min)]}


def test_uncertainty_flag_near_cutoff():
    t, les = stenosed_tube(0.55)
    assert 0.78 <= min_ffr(t, solve_tree(t, les, SP)) <= 0.82
    flags = ffr_uncertainty_flags([t], les, SP, None)
    assert [f["kind"] for f in flags] == ["ffr_uncertain"]
    assert flags[0]["tree_id"] == "tree_0"
    assert flags[0]["message"].startswith("FFR near 0.80 is sensitive to lumen size (min FFR ")


def test_no_flag_without_lesion_or_far_from_cutoff():
    b = straight_branch("b0", None, (0, 0, 0), (0, 0, 1), 50, 1.5)
    assert ffr_uncertainty_flags([make_tree([b])], {}, SP, None) == []
    t, les = stenosed_tube(0.8)
    assert ffr_uncertainty_flags([t], les, SP, None) == []
