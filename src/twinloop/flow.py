"""Resting-flow modes and the FFR uncertainty flag (SPEC_V2 §11)."""

import copy
import dataclasses

import numpy as np

from twinloop.anatomy import Anatomy
from twinloop.centerline import VesselTree
from twinloop.config import SolverParams
from twinloop.config_v2 import FlowParams
from twinloop.lesions import Lesion
from twinloop.solver import _outlet_ids, _tail_median_radius, solve_tree

MYO_UNAVAILABLE_WARNING = "Myocardium unavailable; using diameter flow law"
NEEDS_MYO_ERROR = "myocardial_mass flow needs a valid myocardium"
FFR_CUTOFF = 0.80


def tree_rest_flows(
    trees: list[VesselTree], anat: Anatomy | None, sp: SolverParams, fp: FlowParams
) -> tuple[dict[str, float] | None, str]:
    """({tree_id: resting flow m3/s} or None, mode actually used)."""
    mode = fp.flow_mode
    have_myo = (
        anat is not None and bool(anat.qc.get("myo_present")) and anat.myo_volume_ml is not None
    )
    if mode == "diameter_law":
        return None, "diameter_law"
    if mode == "auto":
        if not have_myo:
            return None, "diameter_law"  # caller adds MYO_UNAVAILABLE_WARNING
        mode = "myocardial_mass"
    if mode != "myocardial_mass":
        raise ValueError(f"Unknown flow mode: {fp.flow_mode}")
    if not have_myo:
        raise ValueError(NEEDS_MYO_ERROR)
    mass_g = anat.myo_volume_ml * fp.myo_density_g_ml
    q_total_ml_min = fp.rest_flow_ml_min_per_g * mass_g
    r3 = {
        t.tree_id: sum(_tail_median_radius(t.branches[o], sp.outlet_window_mm) ** 3
                       for o in _outlet_ids(t))
        for t in trees
    }  # fmt: skip
    total = sum(r3.values())
    return {tid: q_total_ml_min * v / total * 1e-6 / 60.0 for tid, v in r3.items()}, (
        "myocardial_mass"
    )


def min_ffr(tree: VesselTree, solution, min_diam_mm: float = 1.5) -> float:
    """Minimum FFR over samples with diameter >= min_diam_mm (nan if there are none)."""
    vals = [solution.ffr[bid][2.0 * b.radius_mm >= min_diam_mm] for bid, b in tree.branches.items()]
    vals = [v for v in vals if len(v)]
    return float(min(v.min() for v in vals)) if vals else float("nan")


def _scaled(tree: VesselTree, lesions: dict[str, list[Lesion]], scale: float):
    t2 = copy.deepcopy(tree)
    les2: dict[str, list[Lesion]] = {}
    for bid, ls in lesions.items():
        if bid not in t2.branches:
            continue
        new = []
        for les in ls:
            t2.branches[bid].radius_mm[les.start_idx : les.end_idx + 1] *= scale
            new.append(dataclasses.replace(les, r_min_mm=les.r_min_mm * scale))
        les2[bid] = new
    return t2, les2


def ffr_uncertainty_flags(
    trees: list[VesselTree],
    lesions_by_branch: dict[str, list[Lesion]],
    sp: SolverParams,
    q_overrides: dict[str, float] | None,
    factor: float = 0.10,
    min_diam_mm: float = 1.5,
) -> list[dict]:
    flags = []
    for t in trees:
        les = {b: ls for b, ls in lesions_by_branch.items() if b in t.branches and ls}
        if not les:
            continue
        q = None if q_overrides is None else q_overrides.get(t.tree_id)
        vals = []
        for scale in (1.0, 1.0 - factor, 1.0 + factor):
            tt, ll = _scaled(t, les, scale)
            vals.append(min_ffr(tt, solve_tree(tt, ll, sp, q_rest_override_m3s=q), min_diam_mm))
        if np.isnan(vals).any():
            continue
        sides = {v <= FFR_CUTOFF for v in vals}
        if len(sides) > 1:
            flags.append(
                {
                    "kind": "ffr_uncertain",
                    "tree_id": t.tree_id,
                    "message": "FFR near 0.80 is sensitive to lumen size "
                    f"(min FFR {min(vals):.2f}–{max(vals):.2f})",
                }
            )
    return flags
