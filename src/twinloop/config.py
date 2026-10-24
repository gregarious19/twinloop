"""All pipeline parameters (SPEC §5.6)."""

from dataclasses import dataclass, field

MMHG_TO_PA = 133.322


@dataclass(frozen=True)
class CenterlineParams:
    min_component_mm3: float = 50.0
    max_trees: int = 2
    spur_min_mm: float = 3.0
    spur_radius_factor: float = 1.5
    prune_passes: int = 2
    resample_mm: float = 0.5
    position_sigma_mm: float = 1.0
    radius_sigma_mm: float = 0.75
    radius_half_voxel_correction: bool = False


@dataclass(frozen=True)
class LesionParams:
    ref_window_mm: float = 10.0
    ref_percentile: float = 80.0
    ds_threshold: float = 0.25
    min_length_mm: float = 1.0
    merge_gap_mm: float = 2.0
    min_ref_diameter_mm: float = 1.5
    end_exclusion_mm: float = 2.0


@dataclass(frozen=True)
class SolverParams:
    aortic_pressure_mmhg: float = 90.0
    venous_pressure_mmhg: float = 0.0
    viscosity_pa_s: float = 0.0035
    density_kg_m3: float = 1060.0
    hyperemia_resistance_factor: float = 0.24
    flow_law_coeff_ml_min: float = 1.43
    flow_law_exponent: float = 2.55
    root_window_mm: float = 5.0
    outlet_window_mm: float = 3.0
    young_tsai_kt: float = 1.52
    relax: float = 0.5
    tol: float = 1e-6
    max_iter: int = 1000


@dataclass(frozen=True)
class PipelineParams:
    centerline: CenterlineParams = field(default_factory=CenterlineParams)
    lesion: LesionParams = field(default_factory=LesionParams)
    solver: SolverParams = field(default_factory=SolverParams)
    min_display_diameter_mm: float = 1.5
