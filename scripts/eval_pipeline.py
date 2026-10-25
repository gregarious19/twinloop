"""Pipeline consistency evaluation (SPEC §8): GT mask vs predicted mask through the full pipeline.

Measures how much segmentation error moves min-FFR. It does NOT measure FFR accuracy.
"""

import argparse
import os
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from twinloop.config import PipelineParams
from twinloop.io import load_image, load_mask
from twinloop.pipeline import run_pipeline

DATA = Path(os.environ.get("TWINLOOP_DATA", "/data"))
COLUMNS = [
    "case_id", "mask_source", "status", "n_trees", "n_lesions", "min_ffr_tree0", "min_ffr_tree1",
    "t_segmentation", "t_centerline", "t_solver", "t_total", "error",
]  # fmt: skip
SIGNIFICANT_FFR = 0.80
MATCH_MM = 15.0


def tree_stats(result, min_diam_mm: float) -> list[dict]:
    """Per tree: ostium (root branch first point), min FFR and 'significant' flag."""
    out = []
    for tree in result.trees:
        best, sig = None, False
        for b in tree.branches:
            ffr = np.array(b.ffr)
            big = 2.0 * np.array(b.radius_mm) >= min_diam_mm
            if big.any():
                v = float(ffr[big].min())
                best = v if best is None else min(best, v)
                sig = sig or v <= SIGNIFICANT_FFR
        root = next(b for b in tree.branches if b.branch_id == tree.root_branch_id)
        out.append({"ostium": np.array(root.points_mm[0]), "min_ffr": best, "sig": sig})
    return out


def run_case(image, mask, case_dir: Path, case_id: int, source: str, filename: str, params):
    row = dict.fromkeys(COLUMNS)
    row.update(case_id=case_id, mask_source=source)
    shutil.rmtree(case_dir, ignore_errors=True)
    try:
        res = run_pipeline(image, mask, params, case_dir, f"{case_id}_{source}", filename)
    except Exception as exc:
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "log.txt").write_text(traceback.format_exc())
        row.update(status="failed", error=str(exc))
        return row, None
    stats = tree_stats(res, params.min_display_diameter_mm)
    t = res.timings_s
    row.update(
        status="done",
        n_trees=len(res.trees),
        n_lesions=res.summary.n_lesions,
        min_ffr_tree0=stats[0]["min_ffr"] if len(stats) > 0 else None,
        min_ffr_tree1=stats[1]["min_ffr"] if len(stats) > 1 else None,
        t_segmentation=t["segmentation"],
        t_centerline=t["centerline"],
        t_solver=t["solver"],
        t_total=t["total"],
    )
    return row, stats


def match_trees(gt: list[dict], pred: list[dict]) -> tuple[list[tuple[dict, dict]], int]:
    pairs, unmatched = [], 0
    for g in gt:
        if not pred:
            unmatched += 1
            continue
        d = [float(np.linalg.norm(g["ostium"] - p["ostium"])) for p in pred]
        j = int(np.argmin(d))
        if d[j] < MATCH_MM:
            pairs.append((g, pred[j]))
        else:
            unmatched += 1
    return pairs, unmatched


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=DATA / "raw/imagecas/manifest.csv")
    args = ap.parse_args()

    m = pd.read_csv(args.manifest)
    m = m[m.split == args.split].sort_values("case_id").head(args.n)
    params = PipelineParams()
    rows, deltas, agree, n_unmatched, n_matched = [], [], [], 0, 0
    for r in m.itertuples():
        cid = int(r.case_id)
        image = load_image(Path(r.image_path))
        gt_mask = load_mask(Path(r.label_path), image)
        base = DATA / "runs" / "eval" / "pipeline"
        row_gt, st_gt = run_case(
            image, gt_mask, base / f"{cid}_gt", cid, "gt", Path(r.image_path).name, params
        )
        row_pr, st_pr = run_case(
            image, None, base / f"{cid}_pred", cid, "pred", Path(r.image_path).name, params
        )
        rows += [row_gt, row_pr]
        print(f"case {cid}: gt={row_gt['status']} pred={row_pr['status']}", flush=True)
        if st_gt is not None and st_pr is not None:
            pairs, unm = match_trees(st_gt, st_pr)
            n_unmatched += unm
            for g, p in pairs:
                n_matched += 1
                agree.append(g["sig"] == p["sig"])
                if g["min_ffr"] is not None and p["min_ffr"] is not None:
                    deltas.append(abs(g["min_ffr"] - p["min_ffr"]))

    df = pd.DataFrame(rows, columns=COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    n = len(m)
    for src in ("gt", "pred"):
        d = df[df.mask_source == src]
        ok = d[d.status == "done"]
        print(
            f"source={src} success={len(ok)}/{n} t_total_median={ok.t_total.median():.1f} "
            f"t_total_p90={_pct(ok.t_total.dropna(), 90):.1f}"
        )
    ok = df[(df.mask_source == "pred") & (df.status == "done")]
    print(
        f"t_segmentation_median={ok.t_segmentation.median():.1f} "
        f"t_segmentation_p90={_pct(ok.t_segmentation.dropna(), 90):.1f}"
    )
    print(
        f"delta_min_ffr median={_pct(deltas, 50):.3f} p90={_pct(deltas, 90):.3f} "
        f"n_trees={len(deltas)} matched={n_matched} unmatched={n_unmatched}"
    )
    if agree:
        print(f"significant_tree_agreement={100 * np.mean(agree):.1f}% (n={len(agree)})")


if __name__ == "__main__":
    main()
