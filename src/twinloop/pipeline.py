"""Pipeline orchestration and CLI (SPEC §5.7): image (+mask) -> result.json."""

import argparse
import dataclasses
import os
import shutil
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import SimpleITK as sitk

from twinloop import anatomy, segmentation
from twinloop.anatomy import Anatomy, coverage_check, find_ostia, label_branches
from twinloop.casefiles import (
    initial_status,
    new_case_id,
    update_status,
    utc_now_iso,
    write_json_atomic,
)
from twinloop.centerline import VesselTree, extract_trees
from twinloop.config import PipelineParams
from twinloop.config_v2 import FlowParams, V2Params
from twinloop.flow import MYO_UNAVAILABLE_WARNING, ffr_uncertainty_flags, tree_rest_flows
from twinloop.graph import CoronaryGraph, endings_for_tree
from twinloop.io import load_image, load_mask, save_nifti
from twinloop.lesions import Lesion, detect_lesions
from twinloop.lumen import refine_radii
from twinloop.modelio import load_model
from twinloop.radius import area_eq_radii
from twinloop.repair import RulesDecider, run_repair
from twinloop.schema import (
    DISCLAIMER,
    BranchOut,
    CaseResult,
    InputInfo,
    LesionOut,
    Summary,
    TreeOut,
    V2Out,
)
from twinloop.solver import TreeSolution, solve_tree

SIGNIFICANT_FFR = 0.80
V2_DEFAULT_FLOW_MODES = ("auto", "diameter_law", "myocardial_mass")
M3S_TO_ML_MIN = 1e6 * 60.0


@dataclasses.dataclass
class V2Options:
    anatomy: bool = True
    repair: bool = False
    lumen: bool = False
    flow_mode: str = "auto"
    decider: str = "learned"  # "learned" | "rules" (rules: RulesDecider, used by the evaluation)
    radius_mode: Literal["inscribed", "area_eq"] = "inscribed"  # SPEC_V2 §13b
    lumen_model: str = "lumen"  # model folder under models/v2 (S23: "lumen_v3")
    lumen_radius_out: Literal["eq", "ins"] = "eq"  # defaults = old lumen behaviour
    lumen_fallback: Literal["v1", "nearest", "interp"] = "v1"
    lumen_sigma: float = 1.0


def _empty_anatomy(message: str) -> Anatomy:
    qc = {"aorta_present": False, "aorta_connected": False, "myo_present": False}
    return Anatomy(None, None, None, dict.fromkeys(("aorta", "myocardium", "lv_cavity"), "missing"),
                   {**qc, "ostia_ok": False}, [message])  # fmt: skip


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _round_list(a: np.ndarray, digits: int):
    return np.round(np.asarray(a, dtype=np.float64), digits).tolist()


def _lesions_for_tree(tree: VesselTree, params: PipelineParams) -> dict[str, list[Lesion]]:
    return {
        bid: sorted(detect_lesions(b, params.lesion), key=lambda les: les.min_idx)
        for bid, b in tree.branches.items()  # branches are stored in BFS order
    }


def _tree_out(tree: VesselTree, sol: TreeSolution) -> TreeOut:
    branches = [
        BranchOut(
            branch_id=bid,
            parent_id=b.parent_id,
            children=list(b.children),
            points_mm=_round_list(b.points_mm, 2),
            radius_mm=_round_list(b.radius_mm, 3),
            ffr=_round_list(sol.ffr[bid], 3),
            flow_ml_min=round(sol.flow_m3s[bid] * M3S_TO_ML_MIN, 2),
        )
        for bid, b in tree.branches.items()
    ]
    return TreeOut(
        tree_id=tree.tree_id,
        root_branch_id=tree.root_branch_id,
        volume_mm3=round(tree.volume_mm3, 1),
        q_rest_ml_min=round(sol.q_rest_m3s * M3S_TO_ML_MIN, 2),
        q_hyper_ml_min=round(sol.q_hyper_m3s * M3S_TO_ML_MIN, 2),
        converged=sol.converged,
        iterations=sol.iterations,
        branches=branches,
    )


def _lesion_out(tree: VesselTree, sol: TreeSolution, les: Lesion, lesion_id: str) -> LesionOut:
    b = tree.branches[les.branch_id]
    ffr = sol.ffr[les.branch_id]
    proximal = float(ffr[max(les.start_idx - 2, 0)])
    distal = float(ffr[min(les.end_idx + 20, len(ffr) - 1)])
    return LesionOut(
        lesion_id=lesion_id,
        tree_id=tree.tree_id,
        branch_id=les.branch_id,
        min_point_index=les.min_idx,
        position_mm=_round_list(b.points_mm[les.min_idx], 2),
        diameter_stenosis_pct=round(les.ds * 100.0, 1),
        r_min_mm=round(les.r_min_mm, 3),
        r_ref_mm=round(les.r_ref_mm, 3),
        length_mm=round(les.length_mm, 2),
        ffr_proximal=round(proximal, 3),
        ffr_distal=round(distal, 3),
        delta_ffr=round(proximal - distal, 3),
    )


def _summary(trees: list[VesselTree], sols: list[TreeSolution], n_lesions: int, p) -> Summary:
    min_ffr, where = None, (None, None)
    n_sig = 0
    for tree, sol in zip(trees, sols, strict=True):
        for bid, b in tree.branches.items():
            ffr = sol.ffr[bid]
            big = 2.0 * b.radius_mm >= p.min_display_diameter_mm
            if big.any():
                i = int(np.nonzero(big)[0][np.argmin(ffr[big])])
                if min_ffr is None or ffr[i] < min_ffr:
                    min_ffr, where = float(ffr[i]), (bid, i)
            if ffr[-1] <= SIGNIFICANT_FFR and big[-1]:
                n_sig += 1
    return Summary(
        min_ffr=None if min_ffr is None else round(min_ffr, 3),
        min_ffr_branch_id=where[0],
        min_ffr_point_index=where[1],
        n_lesions=n_lesions,
        n_significant_branches=n_sig,
    )


def run_pipeline(
    image: sitk.Image,
    mask: sitk.Image | None,
    params: PipelineParams,
    case_dir: Path,
    case_id: str,
    input_filename: str,
    load_s: float = 0.0,
    on_step: Callable[[str], None] | None = None,
    v2: V2Options | None = None,
) -> CaseResult:
    step = on_step or (lambda name: None)
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    created_at = utc_now_iso()
    has_mask = mask is not None
    t_start = time.perf_counter()

    P2 = V2Params()
    device = _device() if v2 is not None and (v2.repair or v2.lumen) else "cpu"
    models: dict = {}
    lumen_model = None
    lumen_stats: dict = {}
    if v2 is not None and v2.repair:  # fail early if a model is missing
        decider = RulesDecider() if v2.decider == "rules" else load_model("decider", device)
        models = {"decider": decider, "gapfill": load_model("gapfill", device)}
    if v2 is not None and v2.lumen:
        lumen_model = load_model(v2.lumen_model, device)

    seg_s = 0.0
    warnings: list[str] = []
    anat: Anatomy | None = None
    anatomy_failed = False
    anatomy_s = repair_s = lumen_s = radius_s = 0.0
    if mask is None or (v2 is not None and v2.anatomy):
        step("segmenting")
    if mask is None:
        t = time.perf_counter()
        mask = segmentation.segment_nnunet(image, case_dir / "nnunet")
        seg_s = time.perf_counter() - t
    if v2 is not None and v2.anatomy:
        t = time.perf_counter()
        workdir = case_dir / "totalseg"
        try:
            anat = anatomy.segment_anatomy(image, workdir, P2.anatomy)
        except Exception as exc:  # anatomy never fails a case
            anat = _empty_anatomy(str(exc))
            anatomy_failed = True
            warnings.append(f"Anatomy unavailable: {exc}")
        shutil.rmtree(workdir, ignore_errors=True)
        for name, m in (("aorta", anat.aorta), ("myocardium", anat.myocardium)):
            if m is not None:
                save_nifti(m, case_dir / f"{name}.nii.gz")
        anatomy_s = time.perf_counter() - t
        seg_s += anatomy_s

    step("extracting")
    t = time.perf_counter()
    repair_actions: list = []
    review_flags: list[dict] = []
    if v2 is not None and v2.repair:
        t_r = time.perf_counter()
        save_nifti(mask, case_dir / "mask_firstpass.nii.gz")
        mask, repair_actions, review_flags = run_repair(
            image, mask, anat if v2.anatomy else None, models, P2, device
        )
        repair_s = time.perf_counter() - t_r
    trees = extract_trees(mask, params.centerline)
    if v2 is not None and v2.radius_mode == "area_eq":
        t_a = time.perf_counter()
        area_eq_radii(trees, mask, P2.cpr, params.centerline)
        radius_s = time.perf_counter() - t_a
    if lumen_model is not None:
        t_l = time.perf_counter()
        warnings += refine_radii(
            trees,
            image,
            lumen_model,
            P2.cpr,
            device,
            radius_out=v2.lumen_radius_out,
            fallback=v2.lumen_fallback,
            sigma=v2.lumen_sigma,
            stats=lumen_stats,
        )
        lumen_s = time.perf_counter() - t_l
    lesion_maps = [_lesions_for_tree(tree, params) for tree in trees]
    centerline_s = time.perf_counter() - t

    step("solving")
    t = time.perf_counter()
    solver_params = params.solver
    q_rest, flow_mode_used = None, "diameter_law"
    if v2 is not None:
        fp = FlowParams(flow_mode=v2.flow_mode)
        q_rest, flow_mode_used = tree_rest_flows(trees, anat, solver_params, fp)
        if v2.flow_mode == "auto" and flow_mode_used == "diameter_law" and not anatomy_failed:
            if v2.anatomy:
                warnings.append(MYO_UNAVAILABLE_WARNING)
    sols = [
        solve_tree(
            tree,
            lm,
            solver_params,
            **({} if q_rest is None else {"q_rest_override_m3s": q_rest[tree.tree_id]}),
        )
        for tree, lm in zip(trees, lesion_maps, strict=True)
    ]
    solver_s = time.perf_counter() - t

    branch_labels: dict[str, str] = {}
    if v2 is not None:
        merged = {b: ls for lm in lesion_maps for b, ls in lm.items()}
        review_flags += ffr_uncertainty_flags(
            trees, merged, solver_params, q_rest, P2.flow.ffr_uncertainty_radius_factor
        )
        if v2.anatomy and anat is not None:
            g = CoronaryGraph(
                trees=trees,
                fragments=[],
                endings=[e for t_ in trees for e in endings_for_tree(t_, P2.graph)],
                labels={b: "other" for t_ in trees for b in t_.branches},
                tree_side={t_.tree_id: "unknown" for t_ in trees},
                warnings=[],
            )
            find_ostia(g, anat, P2.anatomy)
            label_branches(g, anat, P2.anatomy)
            branch_labels = dict(g.labels)
            if not v2.repair:  # run_repair already appended the coverage flags
                review_flags += coverage_check(g, P2.anatomy)

    warnings += [w for tree in trees for w in tree.warnings]
    warnings += [f"Solver did not converge for {s.tree_id}" for s in sols if not s.converged]

    lesions_out: list[LesionOut] = []
    for tree, sol, lm in zip(trees, sols, lesion_maps, strict=True):
        for les_list in lm.values():
            for les in les_list:
                lesions_out.append(_lesion_out(tree, sol, les, f"L{len(lesions_out) + 1}"))

    save_nifti(image, case_dir / "image.nii.gz")
    save_nifti(mask, case_dir / "mask.nii.gz")

    total = load_s + (time.perf_counter() - t_start)
    result = CaseResult(
        case_id=case_id,
        created_at=created_at,
        input=InputInfo(
            filename=input_filename,
            size_xyz=list(image.GetSize()),
            spacing_mm=[round(v, 4) for v in image.GetSpacing()],
            has_mask=has_mask,
        ),
        params=dataclasses.asdict(params),
        trees=[_tree_out(tree, sol) for tree, sol in zip(trees, sols, strict=True)],
        lesions=lesions_out,
        summary=_summary(trees, sols, len(lesions_out), params),
        timings_s={
            "load": round(load_s, 1),
            "segmentation": round(seg_s, 1),
            "centerline": round(centerline_s, 1),
            "solver": round(solver_s, 1),
            "total": round(total, 1),
        },
        warnings=warnings,
        disclaimer=DISCLAIMER,
        v2=None
        if v2 is None
        else V2Out(
            anatomy_source=anat.source if anat is not None else {},
            qc=anat.qc if anat is not None else {},
            myocardium_ml=None
            if anat is None or anat.myo_volume_ml is None
            else round(anat.myo_volume_ml, 1),
            flow_mode=flow_mode_used,
            radius_mode=v2.radius_mode,
            lumen_fallback_fraction=lumen_stats.get("fallback_fraction"),
            branch_labels=branch_labels,
            repair_actions=[dataclasses.asdict(a) for a in repair_actions],
            review_flags=review_flags,
            timings_s={
                "anatomy": round(anatomy_s, 1),
                "repair": round(repair_s, 1),
                "lumen": round(lumen_s, 1),
                "radius": round(radius_s, 1),
            },
        ),
    )
    dump = result.model_dump()
    if result.v2 is None:
        dump.pop("v2")  # v1 result.json stays byte-for-byte compatible
    write_json_atomic(case_dir / "result.json", dump)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the TwinLoop pipeline on one scan")
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--mask", type=Path)
    ap.add_argument("--aortic-pressure", type=float, default=90.0)
    ap.add_argument("--case-dir", type=Path)
    ap.add_argument("--case-id")
    ap.add_argument("--no-anatomy", action="store_true")
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--lumen", action="store_true")
    ap.add_argument("--radius-mode", choices=["inscribed", "area_eq"], default="inscribed")
    ap.add_argument("--flow-mode", choices=V2_DEFAULT_FLOW_MODES, default="auto")
    args = ap.parse_args(argv)

    case_id = args.case_id or new_case_id()
    data_root = Path(os.environ.get("TWINLOOP_DATA", "/data"))
    case_dir = args.case_dir or data_root / "runs" / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        case_dir / "status.json",
        initial_status(
            case_id, args.image.name, args.mask is not None, args.aortic_pressure, "loading"
        ),
    )
    params = PipelineParams(
        solver=dataclasses.replace(
            PipelineParams().solver, aortic_pressure_mmhg=args.aortic_pressure
        )
    )
    try:
        t = time.perf_counter()
        image = load_image(args.image)
        mask = load_mask(args.mask, image) if args.mask else None
        load_s = time.perf_counter() - t
        result = run_pipeline(
            image,
            mask,
            params,
            case_dir,
            case_id,
            args.image.name,
            load_s=load_s,
            on_step=lambda name: update_status(case_dir, status=name),
            v2=V2Options(
                anatomy=not args.no_anatomy,
                repair=args.repair,
                lumen=args.lumen,
                flow_mode=args.flow_mode,
                radius_mode=args.radius_mode,
            ),
        )
    except Exception as exc:
        with open(case_dir / "log.txt", "a") as log:
            log.write(traceback.format_exc())
        update_status(case_dir, status="failed", error=str(exc))
        print(f"FAILED {exc}")
        return 1
    update_status(case_dir, status="done", timings_s=result.timings_s)
    s = result.summary
    min_ffr = "none" if s.min_ffr is None else f"{s.min_ffr:.3f}"
    print(
        f"OK case_id={case_id} trees={len(result.trees)} lesions={s.n_lesions} "
        f"min_ffr={min_ffr} total_s={result.timings_s['total']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
