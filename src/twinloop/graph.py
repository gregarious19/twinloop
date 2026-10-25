"""Vessel graph: main trees, fragments and branch endings (SPEC_V2 §4)."""

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk

from twinloop.centerline import Branch, VesselTree, extract_trees
from twinloop.config import CenterlineParams
from twinloop.config_v2 import GraphParams


@dataclass
class Ending:
    tree_id: str
    branch_id: str
    point_mm: np.ndarray  # (3,) last point of the branch
    direction: np.ndarray  # (3,) unit vector pointing distally
    r_end_mm: float
    r_ref_mm: float
    taper_ratio: float
    kind: str  # "abrupt" | "taper"


@dataclass
class CoronaryGraph:
    trees: list[VesselTree]  # the 2 largest components ("main trees")
    fragments: list[VesselTree]  # smaller components (e.g. distal piece after a gap)
    endings: list[Ending]  # one per terminal branch of trees AND fragments
    labels: dict[str, str] = field(default_factory=dict)  # branch_id -> LM|LAD|LCX|RCA|other
    tree_side: dict[str, str] = field(default_factory=dict)  # tree_id -> left|right|unknown
    warnings: list[str] = field(default_factory=list)


def branch_length_mm(b: Branch) -> float:
    return float(b.s_mm[-1])


def _point_at(b: Branch, s: float) -> np.ndarray:
    return np.array([np.interp(s, b.s_mm, b.points_mm[:, k]) for k in range(3)])


def end_direction(b: Branch, window_mm: float) -> np.ndarray:
    """Unit vector pointing distally at the end of a branch."""
    length = branch_length_mm(b)
    p_last = b.points_mm[-1]
    if length < window_mm:
        d = p_last - b.points_mm[0]
    else:
        d = p_last - _point_at(b, length - window_mm)
    n = float(np.linalg.norm(d))
    return d / n if n > 0 else np.zeros(3)


def _make_ending(tree_id: str, b: Branch, gp: GraphParams) -> Ending:
    length = branch_length_mm(b)
    s, r = b.s_mm, b.radius_mm
    r_end = float(np.median(r[s >= length - gp.end_window_mm]))
    ref = (s >= length - gp.taper_ref_to_end_mm) & (s <= length - gp.taper_ref_from_end_mm)
    if length < gp.taper_ref_to_end_mm or not ref.any():
        r_ref = float(np.median(r))
    else:
        r_ref = float(np.median(r[ref]))
    ratio = r_end / r_ref if r_ref > 0 else 0.0
    abrupt = r_end >= gp.abrupt_min_radius_mm and ratio >= gp.abrupt_taper_ratio
    return Ending(
        tree_id=tree_id,
        branch_id=b.branch_id,
        point_mm=np.asarray(b.points_mm[-1], dtype=float),
        direction=end_direction(b, gp.direction_window_mm),
        r_end_mm=r_end,
        r_ref_mm=r_ref,
        taper_ratio=float(ratio),
        kind="abrupt" if abrupt else "taper",
    )


def endings_for_tree(tree: VesselTree, gp: GraphParams) -> list[Ending]:
    return [_make_ending(tree.tree_id, b, gp) for b in tree.branches.values() if not b.children]


def build_graph(mask: sitk.Image, cp: CenterlineParams, gp: GraphParams) -> CoronaryGraph:
    # extract_trees raises ValueError("No coronary tree found in mask"); let it propagate.
    all_t = extract_trees(mask, dataclasses.replace(cp, max_trees=gp.max_components))
    trees, fragments = all_t[:2], all_t[2:]
    endings = [
        _make_ending(t.tree_id, b, gp) for t in all_t for b in t.branches.values() if not b.children
    ]
    return CoronaryGraph(
        trees=trees,
        fragments=fragments,
        endings=endings,
        labels={b: "other" for t in all_t for b in t.branches},
        tree_side={t.tree_id: "unknown" for t in trees},
        warnings=[w for t in all_t for w in t.warnings],
    )


def main_path(tree: VesselTree, start_branch_id: str) -> list[str]:
    path = [start_branch_id]
    cur = tree.branches[start_branch_id]
    while cur.children:

        def first5_radius(cid: str) -> float:
            c = tree.branches[cid]
            sel = c.s_mm <= 5.0
            return float(np.median(c.radius_mm[sel] if sel.any() else c.radius_mm))

        best = max(cur.children, key=first5_radius)
        path.append(best)
        cur = tree.branches[best]
    return path


def _r2(x) -> float:
    return round(float(x), 2)


def _pt(p) -> list[float]:
    return [_r2(v) for v in p]


def graph_summary(g: CoronaryGraph) -> dict:
    return {
        "n_trees": len(g.trees),
        "n_fragments": len(g.fragments),
        "trees": [
            {
                "tree_id": t.tree_id,
                "side": g.tree_side.get(t.tree_id, "unknown"),
                "n_branches": len(t.branches),
                "total_length_mm": _r2(sum(branch_length_mm(b) for b in t.branches.values())),
                "root_point_mm": _pt(t.branches[t.root_branch_id].points_mm[0]),
            }
            for t in g.trees
        ],
        "labels": dict(g.labels),
        "endings": [
            {
                "branch_id": e.branch_id,
                "kind": e.kind,
                "r_end_mm": _r2(e.r_end_mm),
                "taper_ratio": _r2(e.taper_ratio),
                "point_mm": _pt(e.point_mm),
            }
            for e in g.endings
        ],
        "warnings": list(g.warnings),
    }
