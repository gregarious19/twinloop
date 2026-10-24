"""Mask -> trees of branches (SPEC §5.3). Extraction is added in T03."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Branch:
    branch_id: str  # f"{tree_id}_b{k}", k in BFS order from root, root branch k=0
    parent_id: str | None
    points_mm: np.ndarray  # (N, 3) physical LPS coords, mm, ordered proximal -> distal
    radius_mm: np.ndarray  # (N,)
    s_mm: np.ndarray  # (N,) arc length from branch start, s[0] = 0
    children: list[str] = field(default_factory=list)


@dataclass
class VesselTree:
    tree_id: str  # "tree_0" (largest component by volume), "tree_1"
    branches: dict[str, Branch]
    root_branch_id: str
    volume_mm3: float = 0.0
    warnings: list[str] = field(default_factory=list)
