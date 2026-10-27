"""v2 parameters (SPEC_V2 §3, exact defaults)."""

from dataclasses import dataclass, field

from twinloop.config import CenterlineParams, LesionParams, SolverParams


@dataclass(frozen=True)
class GraphParams:
    end_window_mm: float = 2.0  # radius at end = median radius over last 2 mm
    taper_ref_from_end_mm: float = 5.0  # reference window: 5..10 mm from the end
    taper_ref_to_end_mm: float = 10.0
    abrupt_min_radius_mm: float = 0.8
    abrupt_taper_ratio: float = 0.75  # abrupt if r_end >= 0.8 and r_end/r_ref >= 0.75
    direction_window_mm: float = 3.0  # end direction = unit(p_last - p_at(L-3mm))
    max_components: int = 8  # components kept by build_graph (2 main + fragments)


@dataclass(frozen=True)
class AnatomyParams:
    totalseg_timeout_s: int = 900
    aorta_min_ml: float = 15.0
    aorta_max_ml: float = 400.0
    aorta_main_component_frac: float = 0.95
    myo_min_ml: float = 50.0
    myo_max_ml: float = 350.0
    ostium_max_dist_mm: float = 3.0
    lm_max_length_mm: float = 25.0
    min_main_len_lad_mm: float = 50.0
    min_main_len_lcx_mm: float = 25.0
    min_main_len_rca_mm: float = 50.0


@dataclass(frozen=True)
class DamageParams:
    n_damages_per_case: int = 6
    gap_len_mm: tuple[float, float] = (2.0, 8.0)
    shrink_len_mm: tuple[float, float] = (3.0, 8.0)
    shrink_factor: tuple[float, float] = (0.4, 0.7)
    false_len_mm: tuple[float, float] = (8.0, 25.0)
    false_radius_mm: tuple[float, float] = (0.8, 1.5)
    false_max_hu: float = 150.0
    occlusion_hu_mean: float = 50.0
    occlusion_hu_sd: float = 15.0
    min_branch_radius_mm: float = 0.8
    min_branch_len_mm: float = 15.0
    type_weights: tuple[float, ...] = (0.2, 0.2, 0.15, 0.2, 0.15, 0.1)
    # order: gap, truncate, delete_branch, shrink, false_branch, occlusion
    # Extra decider data (D-V2-S25): with this probability a gap/truncate/occlusion try is only
    # accepted where the vessel radius at the cut is <= thin_max_radius_mm (real first-pass
    # misses are thin). 0 = off; the default draws no extra random number.
    thin_bias_prob: float = 0.0
    thin_max_radius_mm: float = 1.3


@dataclass(frozen=True)
class RegionParams:
    probe_step_mm: float = 5.0
    probe_min_branch_radius_mm: float = 1.0
    probe_rays: int = 8
    probe_inner_mm: float = 1.0
    probe_outer_mm: float = 5.0  # added to local radius
    probe_min_hu: float = 250.0
    probe_nms_mm: float = 5.0
    probe_junction_exclusion_mm: float = 4.0
    min_terminal_branch_len_mm: float = 5.0
    match_radius_mm: float = 4.0  # damage record ↔ region matching


@dataclass(frozen=True)
class GapfillParams:
    patch_vox: int = 64
    patch_spacing_mm: float = 0.4
    forward_offset_mm: float = 8.0  # patch centre = seed + offset * direction (extend, add)
    seed_sigma_mm: float = 2.0
    hu_clip: tuple[float, float] = (-200.0, 800.0)
    hu_center: float = 300.0
    hu_scale: float = 500.0
    prob_threshold: float = 0.5
    min_new_voxels: int = 20


@dataclass(frozen=True)
class CprParams:
    step_mm: float = 0.5
    size_px: int = 48
    pixel_mm: float = 0.2
    min_branch_len_mm: float = 10.0
    window_slices: int = 128
    window_stride: int = 96


@dataclass(frozen=True)
class RepairParams:
    max_iterations: int = 3
    decider_min_prob: float = 0.6


@dataclass(frozen=True)
class FlowParams:
    flow_mode: str = "auto"  # "auto" | "diameter_law" | "myocardial_mass"
    myo_density_g_ml: float = 1.05
    rest_flow_ml_min_per_g: float = 1.0  # UNCERTAIN, verify in S09
    ffr_uncertainty_radius_factor: float = 0.10


@dataclass(frozen=True)
class V2Params:  # bundle passed to repair/eval code
    centerline: CenterlineParams = field(default_factory=CenterlineParams)  # v1 config.py
    lesion: LesionParams = field(default_factory=LesionParams)  # v1 config.py
    solver: SolverParams = field(default_factory=SolverParams)  # v1 config.py
    graph: GraphParams = field(default_factory=GraphParams)
    anatomy: AnatomyParams = field(default_factory=AnatomyParams)
    damage: DamageParams = field(default_factory=DamageParams)
    region: RegionParams = field(default_factory=RegionParams)
    gapfill: GapfillParams = field(default_factory=GapfillParams)
    cpr: CprParams = field(default_factory=CprParams)
    repair: RepairParams = field(default_factory=RepairParams)
    flow: FlowParams = field(default_factory=FlowParams)
