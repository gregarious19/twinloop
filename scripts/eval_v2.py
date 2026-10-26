"""v2 evaluation harness (SPEC_V2 §13): per (case, variant) metrics into a resumable CSV.

    python scripts/eval_v2.py --cases regression --variants gt_v1,pred_v1 --out reports/x.csv

Uses only v2_holdout / v2_realfail cases (never test-split).
Case dirs: <data>/v2/eval/<variant>/<id>.
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from twinloop import metrics
from twinloop.centerline import Branch, VesselTree
from twinloop.config import CenterlineParams, LesionParams, PipelineParams
from twinloop.config_v2 import GraphParams
from twinloop.graph import CoronaryGraph, build_graph
from twinloop.io import load_image, load_mask
from twinloop.paths import case_image_path, case_label_path, data_root, v2_dir
from twinloop.pipeline import V2Options, run_pipeline
from twinloop.v2split import load_roles

REPO = Path(__file__).resolve().parents[1]
COLUMNS = [
    "case_id", "variant", "status", "dice", "cl_dice", "branch_recall", "false_branch_rate",
    "n_lesions", "mean_abs_rmin_err_mm", "widened_rate", "min_ffr_tree0", "min_ffr_tree1",
    "n_review_flags", "n_actions", "t_total", "error",
]  # fmt: skip
VARIANTS = {
    "gt_v1": ("gt", None),
    "pred_v1": ("pred", None),
    "pred_v2_rules": ("pred", V2Options(repair=True, decider="rules")),
    "pred_v2_repair": ("pred", V2Options(repair=True)),
    "pred_v2_repair_lumen": ("pred", V2Options(repair=True, lumen=True)),
}


def firstpass_path(cid: int) -> Path:
    own = v2_dir() / "firstpass" / f"ImageCAS_{cid:04d}.nii.gz"  # scripts/predict_firstpass.py
    if own.exists():
        return own
    trainer = os.environ.get("TWINLOOP_NNUNET_TRAINER", "nnUNetTrainer")
    return (
        data_root() / "nnUNet_results/Dataset501_ImageCAS"
        / f"{trainer}__nnUNetPlans__3d_fullres/fold_0/validation/ImageCAS_{cid:04d}.nii.gz"
    )  # fmt: skip


def case_list(which: str) -> list[int]:
    if which == "regression":
        return [
            int(c)
            for c in json.loads((REPO / "reports/v2_regression_set.json").read_text())["case_ids"]
        ]
    roles = load_roles()
    return sorted(c for c, r in roles.items() if r == "v2_holdout")


def graph_from_result(result: dict) -> CoronaryGraph:
    """The trees the solver used (radii after lumen refinement), as a CoronaryGraph."""
    trees = []
    for t in result["trees"]:
        branches = {}
        for b in t["branches"]:
            pts = np.asarray(b["points_mm"], dtype=float)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1) if len(pts) > 1 else np.zeros(0)
            branches[b["branch_id"]] = Branch(
                b["branch_id"], b["parent_id"], pts, np.asarray(b["radius_mm"], dtype=float),
                np.concatenate([[0.0], np.cumsum(seg)]), list(b["children"]),
            )  # fmt: skip
        trees.append(VesselTree(t["tree_id"], branches, t["root_branch_id"], t["volume_mm3"]))
    return CoronaryGraph(trees=trees, fragments=[], endings=[])


def min_ffr(result: dict, tree_id: str) -> float | None:
    vals = [
        f
        for t in result["trees"]
        if t["tree_id"] == tree_id
        for b in t["branches"]
        for f, r in zip(b["ffr"], b["radius_mm"], strict=True)
        if 2 * r >= 1.5
    ]
    return min(vals) if vals else None


def eval_one(cid: int, variant: str, gt_cache: dict) -> dict:
    src, v2 = VARIANTS[variant]
    row = dict.fromkeys(COLUMNS, "")
    row.update(case_id=cid, variant=variant)
    t0 = time.perf_counter()
    try:
        image = load_image(case_image_path(cid))
        if cid not in gt_cache:
            gt = load_mask(case_label_path(cid), image)
            gt_cache.clear()
            gt_cache[cid] = (gt, build_graph(gt, CenterlineParams(), GraphParams()))
        gt, g_gt = gt_cache[cid]
        mask = gt if src == "gt" else load_mask(firstpass_path(cid), image)
        case_dir = v2_dir() / "eval" / variant / f"{cid:04d}"
        res = run_pipeline(
            image, mask, PipelineParams(), case_dir, f"eval_{cid:04d}", f"ImageCAS_{cid:04d}",
            v2=v2,
        )  # fmt: skip
        result = json.loads((case_dir / "result.json").read_text())
        final = load_mask(case_dir / "mask.nii.gz", image)
        g_final = build_graph(final, CenterlineParams(), GraphParams())
        a, b = sitk.GetArrayViewFromImage(final) > 0, sitk.GetArrayViewFromImage(gt) > 0
        st = metrics.stenosis_metrics(g_gt, graph_from_result(result), LesionParams())
        row.update(
            status="ok",
            dice=metrics.dice(a, b),
            cl_dice=metrics.cl_dice(a, b),
            branch_recall=metrics.branch_recall(g_gt, final),
            false_branch_rate=metrics.false_branch_rate(g_final, gt),
            n_lesions=st["n_lesions"],
            mean_abs_rmin_err_mm=st["mean_abs_rmin_err_mm"],
            widened_rate=st["widened_rate"],
            min_ffr_tree0=min_ffr(result, "tree_0"),
            min_ffr_tree1=min_ffr(result, "tree_1"),
            n_review_flags=len(res.v2.review_flags) if res.v2 else 0,
            n_actions=sum(a_["result"] == "applied" for a_ in res.v2.repair_actions)
            if res.v2
            else 0,
        )
    except Exception as exc:  # a failed case is recorded, never dropped
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}".replace("\n", " "))
        print(traceback.format_exc(), file=sys.stderr)
    row["t_total"] = round(time.perf_counter() - t0, 1)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", choices=["regression", "holdout"], required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", help="K/N: run only every N-th case, starting at K (0-based)")
    ap.add_argument("--skip-from", type=Path, help="CSV whose (case, variant) rows are not rerun")
    args = ap.parse_args(argv)
    variants = args.variants.split(",")
    for v in variants:
        if v not in VARIANTS:
            ap.error(f"unknown variant {v}")
    cases = case_list(args.cases)[: args.limit]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        cases = cases[k::n]
    out = args.out if args.out.is_absolute() else Path.cwd() / args.out
    done: set[tuple[int, str]] = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {(int(r["case_id"]), r["variant"]) for r in csv.DictReader(f)}
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, COLUMNS).writeheader()
    if args.skip_from and args.skip_from.exists():
        with open(args.skip_from, newline="") as f:
            done |= {(int(r["case_id"]), r["variant"]) for r in csv.DictReader(f)}
    gt_cache: dict = {}
    for cid in cases:
        for v in variants:
            if (cid, v) in done:
                continue
            row = eval_one(cid, v, gt_cache)
            with open(out, "a", newline="") as f:
                csv.DictWriter(f, COLUMNS).writerow(row)
            print(cid, v, row["status"], row["dice"], row["error"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
