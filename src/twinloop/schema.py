"""Pydantic models of result.json (SPEC §6.3)."""

from pydantic import BaseModel

DISCLAIMER = (
    "Research prototype. FFR values are not validated against invasive measurements. "
    "Not for clinical use."
)


class BranchOut(BaseModel):
    branch_id: str
    parent_id: str | None
    children: list[str]
    points_mm: list[list[float]]  # [[x,y,z], ...] rounded to 0.01
    radius_mm: list[float]  # rounded to 0.001
    ffr: list[float]  # rounded to 0.001
    flow_ml_min: float  # rounded to 0.01


class TreeOut(BaseModel):
    tree_id: str
    root_branch_id: str
    volume_mm3: float
    q_rest_ml_min: float
    q_hyper_ml_min: float
    converged: bool
    iterations: int
    branches: list[BranchOut]


class LesionOut(BaseModel):
    lesion_id: str
    tree_id: str
    branch_id: str
    min_point_index: int
    position_mm: list[float]
    diameter_stenosis_pct: float
    r_min_mm: float
    r_ref_mm: float
    length_mm: float
    ffr_proximal: float
    ffr_distal: float
    delta_ffr: float


class Summary(BaseModel):
    min_ffr: float | None
    min_ffr_branch_id: str | None
    min_ffr_point_index: int | None
    n_lesions: int
    n_significant_branches: int


class InputInfo(BaseModel):
    filename: str
    size_xyz: list[int]
    spacing_mm: list[float]
    has_mask: bool


class V2Out(BaseModel):
    """Optional v2 block of result.json (SPEC_V2 §12)."""

    anatomy_source: dict[str, str]
    qc: dict[str, bool]
    myocardium_ml: float | None
    flow_mode: str
    radius_mode: str = "inscribed"  # added in S20; older result.json files lack it
    branch_labels: dict[str, str]
    repair_actions: list[dict]
    review_flags: list[dict]
    timings_s: dict[str, float]


class CaseResult(BaseModel):
    schema_version: str = "1.0"
    case_id: str
    created_at: str  # ISO 8601 UTC, e.g. "2026-10-24T13:15:01Z"
    input: InputInfo
    params: dict  # dataclasses.asdict(PipelineParams)
    trees: list[TreeOut]
    lesions: list[LesionOut]
    summary: Summary
    timings_s: dict[str, float]  # keys: load, segmentation, centerline, solver, total
    warnings: list[str]
    disclaimer: str
    v2: V2Out | None = None
