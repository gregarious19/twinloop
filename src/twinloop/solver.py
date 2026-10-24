"""Steady-state hyperemic 1D hemodynamics on a vessel tree (SPEC §5.5). SI units internally."""

from dataclasses import dataclass

import numpy as np

from twinloop.centerline import Branch, VesselTree
from twinloop.config import MMHG_TO_PA, SolverParams
from twinloop.lesions import Lesion


def turbulent_drop_pa(q_m3s: float, r_ref_mm: float, r_min_mm: float, p: SolverParams) -> float:
    """Young-Tsai expansion loss for one lesion."""
    a0 = np.pi * (r_ref_mm * 1e-3) ** 2
    amin = np.pi * (r_min_mm * 1e-3) ** 2
    u0 = q_m3s / a0
    return float(p.young_tsai_kt * (p.density_kg_m3 / 2) * (a0 / amin - 1) ** 2 * u0**2)


def branch_pressure_drop(
    s_mm: np.ndarray, r_mm: np.ndarray, q_m3s: float, lesions: list[Lesion], p: SolverParams
) -> np.ndarray:
    """Cumulative pressure drop in Pa at each sample; out[0] = 0, non-decreasing."""
    s = np.asarray(s_mm, dtype=np.float64) * 1e-3
    r = np.asarray(r_mm, dtype=np.float64) * 1e-3
    inv4 = r**-4
    seg = 8.0 * p.viscosity_pa_s * np.diff(s) * q_m3s / np.pi * 0.5 * (inv4[:-1] + inv4[1:])
    out = np.concatenate([[0.0], np.cumsum(seg)])
    for les in lesions:
        out[les.end_idx :] += turbulent_drop_pa(q_m3s, les.r_ref_mm, les.r_min_mm, p)
    return out


def _outlet_ids(tree: VesselTree) -> list[str]:
    return [b.branch_id for b in tree.branches.values() if not b.children]


def _tail_median_radius(b: Branch, window_mm: float) -> float:
    mask = b.s_mm >= b.s_mm[-1] - window_mm
    return float(np.median(b.radius_mm[mask]))


def outlet_weights(tree: VesselTree, p: SolverParams) -> dict[str, float]:
    """Murray's-law resting flow share per outlet branch."""
    r3 = {
        oid: _tail_median_radius(tree.branches[oid], p.outlet_window_mm) ** 3
        for oid in _outlet_ids(tree)
    }
    total = sum(r3.values())
    return {oid: v / total for oid, v in r3.items()}


def resting_flow_m3s(tree: VesselTree, p: SolverParams) -> float:
    """Resting tree flow from the root diameter (flow-diameter scaling law)."""
    root = tree.branches[tree.root_branch_id]
    mask = root.s_mm <= p.root_window_mm
    d_root = 2.0 * float(np.median(root.radius_mm[mask]))
    q_ml_min = p.flow_law_coeff_ml_min * d_root**p.flow_law_exponent
    return q_ml_min * 1e-6 / 60.0


def downstream_outlets(tree: VesselTree) -> dict[str, list[str]]:
    """For every branch, the outlet branches at or below it."""
    result: dict[str, list[str]] = {}

    def visit(bid: str) -> list[str]:
        b = tree.branches[bid]
        outs = [bid] if not b.children else [o for c in b.children for o in visit(c)]
        result[bid] = outs
        return outs

    visit(tree.root_branch_id)
    return result


@dataclass
class TreeSolution:
    tree_id: str
    pressure_pa: dict[str, np.ndarray]
    ffr: dict[str, np.ndarray]
    flow_m3s: dict[str, float]
    q_rest_m3s: float
    q_hyper_m3s: float  # flow in root branch
    converged: bool
    iterations: int


def _bfs_order(tree: VesselTree) -> list[str]:
    order = [tree.root_branch_id]
    for bid in order:
        order.extend(tree.branches[bid].children)
    return order


def solve_tree(tree: VesselTree, lesions: dict[str, list[Lesion]], p: SolverParams) -> TreeSolution:
    pa = p.aortic_pressure_mmhg * MMHG_TO_PA
    pv = p.venous_pressure_mmhg * MMHG_TO_PA
    q_rest = resting_flow_m3s(tree, p)
    weights = outlet_weights(tree, p)
    resist = {
        oid: p.hyperemia_resistance_factor * (pa - pv) / (w * q_rest) for oid, w in weights.items()
    }
    down = downstream_outlets(tree)
    order = _bfs_order(tree)
    q_out = {oid: (pa - pv) / resist[oid] for oid in resist}

    def propagate(q_o: dict[str, float]):
        q_branch = {bid: sum(q_o[o] for o in down[bid]) for bid in order}
        pressure: dict[str, np.ndarray] = {}
        for bid in order:
            b = tree.branches[bid]
            start = pa if b.parent_id is None else pressure[b.parent_id][-1]
            drop = branch_pressure_drop(b.s_mm, b.radius_mm, q_branch[bid], lesions.get(bid, []), p)
            pressure[bid] = start - drop
        return q_branch, pressure

    converged = False
    iterations = 0
    for it in range(1, p.max_iter + 1):
        iterations = it
        q_branch, pressure = propagate(q_out)
        q_new = {oid: max((pressure[oid][-1] - pv) / resist[oid], 0.0) for oid in resist}
        total_old = sum(q_out.values())
        err = max(abs(q_new[o] - q_out[o]) for o in q_out) / total_old
        q_out = {o: p.relax * q_new[o] + (1 - p.relax) * q_out[o] for o in q_out}
        if err < p.tol:
            converged = True
            break

    q_branch, pressure = propagate(q_out)
    ffr = {bid: np.clip(pr / pa, 0.0, 1.0) for bid, pr in pressure.items()}
    return TreeSolution(
        tree_id=tree.tree_id,
        pressure_pa=pressure,
        ffr=ffr,
        flow_m3s=q_branch,
        q_rest_m3s=q_rest,
        q_hyper_m3s=q_branch[tree.root_branch_id],
        converged=converged,
        iterations=iterations,
    )
