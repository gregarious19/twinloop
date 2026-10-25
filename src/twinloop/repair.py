"""Repair loop: decide an action per region, apply it, repeat (SPEC_V2 §10)."""

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree

from twinloop.anatomy import Anatomy, coverage_check, find_ostia, label_branches
from twinloop.config_v2 import V2Params
from twinloop.cpr import index_zyx_to_phys
from twinloop.gapfill import apply_gapfill
from twinloop.graph import build_graph
from twinloop.regions import ACTIONS, FEATURE_NAMES, propose_regions, region_features

OFFSET_MM = {"extend": 8.0, "add": 8.0, "restore": 0.0}
MIN_SEPARATION_MM = 4.0


@dataclass
class RepairAction:
    iteration: int
    region_id: str
    kind: str
    action: str
    prob: float
    point_mm: list[float]
    voxels_changed: int
    result: str  # "applied" | "nothing_found" | "flagged" | "skipped"


class RulesDecider:
    """Extend abrupt endings, keep everything else."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((len(X), len(ACTIONS)))
        out[:, ACTIONS.index("keep")] = 1.0
        hit = (X[:, FEATURE_NAMES.index("kind_ending")] == 1) & (
            X[:, FEATURE_NAMES.index("is_abrupt")] == 1
        )
        out[hit, :] = 0.0
        out[hit, ACTIONS.index("extend")] = 1.0
        return out


def _append(actions: list, iteration: int, reg, action: str, prob: float, result: str, n: int):
    point = [float(v) for v in reg.point_mm]
    actions.append(RepairAction(iteration, reg.region_id, reg.kind, action, prob, point, n, result))


def _remove_branch(mask: sitk.Image, g, branch_id: str) -> tuple[sitk.Image, int]:
    """Delete the voxels of a branch (s >= 1 mm) and its descendants, per the §6 interval rule."""
    trees = [*g.trees, *g.fragments]
    tree = next(t for t in trees if branch_id in t.branches)
    ids, stack = {branch_id}, list(tree.branches[branch_id].children)
    while stack:
        c = stack.pop()
        ids.add(c)
        stack.extend(tree.branches[c].children)
    all_b = [b for t in trees for b in t.branches.values()]
    pts = np.concatenate([b.points_mm for b in all_b])
    owner = np.concatenate([[b.branch_id] * len(b.s_mm) for b in all_b])
    s_all = np.concatenate([b.s_mm for b in all_b])
    r_all = np.concatenate([b.radius_mm for b in all_b])
    arr = sitk.GetArrayFromImage(mask)
    vox = np.argwhere(arr > 0)
    dist, nn = cKDTree(pts).query(index_zyx_to_phys(vox, mask), workers=-1)
    sel = np.isin(owner[nn], list(ids)) & (dist <= 1.5 * r_all[nn] + 0.5)
    sel &= (s_all[nn] >= 1.0) | (owner[nn] != branch_id)
    n = int(sel.sum())
    if n == 0:
        return mask, 0
    arr[tuple(vox[sel].T)] = 0
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(mask)
    return out, n


def run_repair(
    image: sitk.Image,
    mask: sitk.Image,
    anat: Anatomy | None,
    models: dict,
    P: V2Params,
    device: str,
) -> tuple[sitk.Image, list[RepairAction], list[dict]]:
    actions: list[RepairAction] = []
    flags: list[dict] = []

    def add_flag(kind: str, reg, prob: float, message: str) -> None:
        for f in flags:
            if (
                f["kind"] == kind
                and np.linalg.norm(np.array(f["point_mm"]) - reg.point_mm) < MIN_SEPARATION_MM
            ):
                return
        flags.append(
            {
                "kind": kind,
                "region_id": reg.region_id,
                "point_mm": [float(v) for v in reg.point_mm],
                "prob": float(prob),
                "message": message,
            }
        )

    for iteration in range(1, P.repair.max_iterations + 1):
        g = build_graph(mask, P.centerline, P.graph)
        regions = propose_regions(g, image, mask, P)
        if not regions:
            break
        proba = models["decider"].predict_proba(region_features(regions, g, image, mask, P))
        changed_points: list[np.ndarray] = []
        total_changed = 0
        for reg, row in zip(regions, proba, strict=True):
            a = int(np.argmax(row))
            action, prob = ACTIONS[a], float(row[a])
            if action == "keep":
                continue

            def log(result: str, n: int = 0, reg=reg, action=action, prob=prob, it=iteration):
                _append(actions, it, reg, action, prob, result, n)

            if prob < P.repair.decider_min_prob:
                log("flagged")
                add_flag(
                    "uncertain_region", reg, prob, f"Uncertain region ({action}, p={prob:.2f})"
                )
                continue
            if action == "stop":
                log("flagged")
                add_flag("possible_occlusion", reg, prob, "Possible occlusion: vessel not extended")
                continue
            if any(np.linalg.norm(reg.point_mm - q) < MIN_SEPARATION_MM for q in changed_points):
                log("skipped")
                continue
            if action == "remove":
                mask, n = _remove_branch(mask, g, reg.branch_id)
            elif models.get("gapfill") is None:
                log("flagged")
                continue
            else:
                mask, n = apply_gapfill(
                    models["gapfill"], image, mask, reg.point_mm, reg.direction,
                    OFFSET_MM[action], P.gapfill, device,
                )  # fmt: skip
            if n > 0:
                changed_points.append(reg.point_mm)
                total_changed += n
            log("applied" if n > 0 else "nothing_found", n)
        if total_changed == 0:
            break

    if anat is not None:
        g = build_graph(mask, P.centerline, P.graph)
        find_ostia(g, anat, P.anatomy)
        label_branches(g, anat, P.anatomy)
        flags.extend(coverage_check(g, P.anatomy))
    return mask, actions, flags
