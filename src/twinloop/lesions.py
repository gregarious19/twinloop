"""Stenosis detection (SPEC §5.4)."""

from dataclasses import dataclass

import numpy as np

from twinloop.centerline import Branch
from twinloop.config import LesionParams


@dataclass
class Lesion:
    lesion_id: str  # "L1", "L2", ... numbered per case in (tree, branch BFS, position) order
    branch_id: str
    start_idx: int
    end_idx: int
    min_idx: int  # inclusive indices into the branch arrays
    r_min_mm: float
    r_ref_mm: float
    ds: float  # 1 - r_min / r_ref
    length_mm: float  # s[end] - s[start]


def reference_radius(s_mm: np.ndarray, r_mm: np.ndarray, p: LesionParams) -> np.ndarray:
    """80th percentile of the radius within +-ref_window/2 of every sample."""
    s, r = np.asarray(s_mm, dtype=np.float64), np.asarray(r_mm, dtype=np.float64)
    half = p.ref_window_mm / 2.0
    if s[-1] - s[0] < half:
        return np.full(len(r), np.percentile(r, p.ref_percentile))
    lo = np.searchsorted(s, s - half, side="left")
    hi = np.searchsorted(s, s + half, side="right")
    return np.array([np.percentile(r[a:b], p.ref_percentile) for a, b in zip(lo, hi, strict=True)])


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Maximal contiguous runs of True as inclusive (start, end) index pairs."""
    edges = np.diff(np.concatenate([[0], flags.astype(np.int8), [0]]))
    return list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1, strict=True))


def detect_lesions(branch: Branch, p: LesionParams) -> list[Lesion]:
    s, r = branch.s_mm, branch.radius_mm
    ref = reference_radius(s, r, p)
    ds = 1.0 - r / ref

    excluded = (s - s[0]) < p.end_exclusion_mm
    if not branch.children:  # terminal: skeleton end taper
        excluded |= (s[-1] - s) < p.end_exclusion_mm
    runs = _runs((ds >= p.ds_threshold) & ~excluded)

    merged: list[list[int]] = []
    for a, b in runs:
        if merged and s[a] - s[merged[-1][1]] < p.merge_gap_mm:
            merged[-1][1] = int(b)
        else:
            merged.append([int(a), int(b)])

    lesions = []
    for a, b in merged:
        length = float(s[b] - s[a])
        if length < p.min_length_mm:
            continue
        k = a + int(np.argmin(r[a : b + 1]))
        if 2.0 * ref[k] < p.min_ref_diameter_mm:
            continue
        lesions.append(
            Lesion(
                "",
                branch.branch_id,
                a,
                b,
                k,
                float(r[k]),
                float(ref[k]),
                float(1.0 - r[k] / ref[k]),
                length,
            )
        )
    return lesions
