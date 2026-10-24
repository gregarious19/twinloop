"""Stenosis detection (SPEC §5.4). Detection is added in T03."""

from dataclasses import dataclass


@dataclass
class Lesion:
    lesion_id: str  # "L1", "L2", ... assigned per case by the pipeline
    branch_id: str
    start_idx: int
    end_idx: int
    min_idx: int  # inclusive indices into the branch arrays
    r_min_mm: float
    r_ref_mm: float
    ds: float  # 1 - r_min / r_ref
    length_mm: float  # s[end] - s[start]
