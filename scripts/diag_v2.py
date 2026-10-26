"""S18 diagnostics: why pred_v1 stenoses look widened and why lumen refinement hurt.

    python scripts/diag_v2.py --out-dir reports            # run D1-D3 on the regression set
    python scripts/diag_v2.py --out-dir reports --summarize  # print the report tables

Analysis only: reads the S16 inputs (GT masks, first-pass masks, deployed lumen model) and
changes nothing in the library. Regression-set cases only.
"""

import argparse
import copy
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import cKDTree

from twinloop.centerline import VesselTree
from twinloop.config import CenterlineParams, LesionParams
from twinloop.config_v2 import GraphParams, V2Params
from twinloop.cpr import phys_to_index_zyx
from twinloop.graph import CoronaryGraph, build_graph
from twinloop.io import load_image, load_mask
from twinloop.lesions import detect_lesions, reference_radius
from twinloop.lumen import refine_radii
from twinloop.metrics import MIN_LENGTH_MM, MIN_RADIUS_MM, stenosis_metrics
from twinloop.modelio import load_model
from twinloop.paths import case_image_path, case_label_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_v2 import case_list, firstpass_path  # noqa: E402

LP = LesionParams()
MATCH_MM = 2.0
POINT_COLS = [
    "case_id", "branch_id", "point_idx", "kind", "r_gt", "d_mm", "r_pred", "signed_err",
    "same_branch", "r_predmask_at_gt", "signed_err_mask", "in_predmask", "r_gtmask_at_gt",
    "error",
]  # fmt: skip
LUMEN_COLS = [
    "case_id", "input", "n_lesions", "mean_abs_rmin_err_mm", "mean_signed_rmin_err_mm",
    "widened_rate", "n_no_lumen_warnings", "error",
]  # fmt: skip
INPUTS = ["pred_nolumen", "gt", "pred", "gt_shift0.5", "gt_shift1.0"]


def all_branches(g: CoronaryGraph):
    return [b for t in [*g.trees, *g.fragments] for b in t.branches.values()]


def radius_map(mask: sitk.Image) -> np.ndarray:
    """The pipeline's radius field (centerline._build_tree): EDT in mm, no half-voxel term."""
    a = sitk.GetArrayViewFromImage(mask) > 0
    sp = np.array(mask.GetSpacing())[::-1]
    return ndimage.distance_transform_edt(a, sampling=sp)


def read_at(arr: np.ndarray, pts_mm: np.ndarray, ref: sitk.Image) -> np.ndarray:
    idx = phys_to_index_zyx(pts_mm, ref)
    return ndimage.map_coordinates(arr, idx.T, order=1, mode="nearest")


def point_rows(cid: int, g_gt, g_pred, gt_mask, pred_mask) -> list[dict]:
    pred = all_branches(g_pred)
    ppts = np.concatenate([b.points_mm for b in pred])
    powner = [(b, i) for b in pred for i in range(len(b.s_mm))]
    ptree = cKDTree(ppts)
    gts = all_branches(g_gt)
    gtree = cKDTree(np.concatenate([b.points_mm for b in gts]))
    glabel = np.concatenate([[b.branch_id] * len(b.s_mm) for b in gts])
    nearest_gt = {}  # pred branch -> majority nearest GT branch over its points
    for b in pred:
        lab, cnt = np.unique(glabel[gtree.query(b.points_mm)[1]], return_counts=True)
        nearest_gt[b.branch_id] = lab[np.argmax(cnt)]
    edt_pred, edt_gt = radius_map(pred_mask), radius_map(gt_mask)
    pred_bin = (sitk.GetArrayViewFromImage(pred_mask) > 0).astype(float)
    rows = []
    for b in gts:
        if np.median(b.radius_mm) < MIN_RADIUS_MM or b.s_mm[-1] < MIN_LENGTH_MM:
            continue
        in_les = np.zeros(len(b.s_mm), bool)
        for les in detect_lesions(b, LP):
            in_les[les.start_idx : les.end_idx + 1] = True
        d, j = ptree.query(b.points_mm)
        r_pm = read_at(edt_pred, b.points_mm, pred_mask)
        r_gm = read_at(edt_gt, b.points_mm, gt_mask)
        inside = read_at(pred_bin, b.points_mm, pred_mask) > 0.5
        for k in range(len(b.s_mm)):
            pb, i = powner[j[k]]
            ok = d[k] <= MATCH_MM
            r_pred = float(pb.radius_mm[i]) if ok else np.nan
            rows.append({
                "case_id": cid, "branch_id": b.branch_id, "point_idx": k,
                "kind": "lesion" if in_les[k] else "healthy", "r_gt": float(b.radius_mm[k]),
                "d_mm": float(d[k]), "r_pred": r_pred, "signed_err": r_pred - b.radius_mm[k],
                "same_branch": bool(nearest_gt[pb.branch_id] == b.branch_id),
                "r_predmask_at_gt": float(r_pm[k]),
                "signed_err_mask": float(r_pm[k] - b.radius_mm[k]),
                "in_predmask": bool(inside[k]), "r_gtmask_at_gt": float(r_gm[k]), "error": "",
            })  # fmt: skip
    return rows


def rmin_stats(g_gt: CoronaryGraph, g_pred: CoronaryGraph) -> dict:
    """stenosis_metrics' matching rule, also returning the signed r_min error."""
    pred = all_branches(g_pred)
    pts = np.concatenate([b.points_mm for b in pred])
    owner = [(b, i) for b in pred for i in range(len(b.s_mm))]
    refs = {b.branch_id: reference_radius(b.s_mm, b.radius_mm, LP) for b in pred}
    signed: list[float] = []
    n = widened = 0
    for t in g_gt.trees:
        for b in t.branches.values():
            for les in detect_lesions(b, LP):
                n += 1
                d = np.linalg.norm(pts - b.points_mm[les.min_idx], axis=1)
                j = int(np.argmin(d))
                if d[j] > MATCH_MM:
                    widened += 1
                    continue
                pb, i = owner[j]
                r = float(pb.radius_mm[i])
                signed.append(r - les.r_min_mm)
                ref_r = float(refs[pb.branch_id][i])
                if (max(0.0, 1.0 - r / ref_r) if ref_r > 0 else 0.0) < 0.5 * les.ds:
                    widened += 1
    lib = stenosis_metrics(g_gt, g_pred, LP)
    mine = float(np.mean(np.abs(signed))) if signed else float("nan")
    assert n == lib["n_lesions"] and (
        np.isclose(mine, lib["mean_abs_rmin_err_mm"], equal_nan=True)
    ), "rmin_stats disagrees with metrics.stenosis_metrics"
    return {
        "n_lesions": n,
        "mean_abs_rmin_err_mm": mine,
        "mean_signed_rmin_err_mm": float(np.mean(signed)) if signed else float("nan"),
        "widened_rate": widened / n if n else float("nan"),
    }


def shifted(trees: list[VesselTree], offset_mm: float) -> list[VesselTree]:
    """Copy of trees with every branch moved `offset_mm` perpendicular to its local tangent,
    in one random direction per branch (seed 0, branch order of the trees)."""
    rng = np.random.default_rng(0)
    out = copy.deepcopy(trees)
    for t in out:
        for b in t.branches.values():
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            tan = np.gradient(b.points_mm, axis=0)
            tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
            perp = v - (tan @ v)[:, None] * tan
            nrm = np.linalg.norm(perp, axis=1, keepdims=True)
            perp = np.where(nrm > 1e-6, perp / np.maximum(nrm, 1e-9), 0.0)
            b.points_mm = b.points_mm + offset_mm * perp
    return out


def lumen_rows(cid: int, image, g_gt, g_pred, model, device) -> list[dict]:
    cp = V2Params().cpr
    variants = {
        "pred_nolumen": (copy.deepcopy(g_pred.trees), False),
        "gt": (copy.deepcopy(g_gt.trees), True),
        "pred": (copy.deepcopy(g_pred.trees), True),
        "gt_shift0.5": (shifted(g_gt.trees, 0.5), True),
        "gt_shift1.0": (shifted(g_gt.trees, 1.0), True),
    }
    rows = []
    for name in INPUTS:
        trees, refine = variants[name]
        row = dict.fromkeys(LUMEN_COLS, "")
        row.update(case_id=cid, input=name)
        try:
            warns = refine_radii(trees, image, model, cp, device) if refine else []
            row.update(rmin_stats(g_gt, CoronaryGraph(trees, [], [])))
            row["n_no_lumen_warnings"] = sum("found no lumen" in w for w in warns)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
            print(traceback.format_exc(), file=sys.stderr)
        rows.append(row)
    return rows


def run(out_dir: Path) -> None:
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = load_model("lumen", device)
    pts_rows: list[dict] = []
    lum_rows: list[dict] = []
    for cid in case_list("regression"):
        print("case", cid, flush=True)
        try:
            image = load_image(case_image_path(cid))
            gt = load_mask(case_label_path(cid), image)
            pm = load_mask(firstpass_path(cid), image)
            g_gt = build_graph(gt, CenterlineParams(), GraphParams())
            g_pred = build_graph(pm, CenterlineParams(), GraphParams())
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}".replace("\n", " ")
            print(traceback.format_exc(), file=sys.stderr)
            pts_rows.append({**dict.fromkeys(POINT_COLS, ""), "case_id": cid, "error": err})
            lum_rows += [
                {**dict.fromkeys(LUMEN_COLS, ""), "case_id": cid, "input": i, "error": err}
                for i in INPUTS
            ]
            continue
        try:
            pts_rows += point_rows(cid, g_gt, g_pred, gt, pm)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}".replace("\n", " ")
            print(traceback.format_exc(), file=sys.stderr)
            pts_rows.append({**dict.fromkeys(POINT_COLS, ""), "case_id": cid, "error": err})
        lum_rows += lumen_rows(cid, image, g_gt, g_pred, model, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pts_rows, columns=POINT_COLS).to_csv(out_dir / "v2_diag_points.csv", index=False)
    pd.DataFrame(lum_rows, columns=LUMEN_COLS).to_csv(out_dir / "v2_diag_lumen.csv", index=False)


def summarize(out_dir: Path) -> None:
    p = pd.read_csv(out_dir / "v2_diag_points.csv")
    p = p[p.error.isna()]
    print("## D1/D2 (per GT centerline point)")
    print("| kind | n | D1 mean signed | D1 median signed | D1 mean abs | frac d>2mm | "
          "frac same_branch False (d<=2) | D2 mean signed | D2 median signed | D2 mean abs | "
          "frac outside pred mask | control: mean(r_gtmask-r_gt) |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for kind, g in [("lesion", p[p.kind == "lesion"]), ("healthy", p[p.kind == "healthy"]),
                    ("all", p)]:  # fmt: skip
        m = g[g.d_mm <= MATCH_MM]
        e, em = m.signed_err, g.signed_err_mask
        print(f"| {kind} | {len(g)} | {e.mean():.3f} | {e.median():.3f} | {e.abs().mean():.3f} "
              f"| {(g.d_mm > MATCH_MM).mean():.3f} | {(~m.same_branch.astype(bool)).mean():.3f} "
              f"| {em.mean():.3f} | {em.median():.3f} | {em.abs().mean():.3f} "
              f"| {(~g.in_predmask.astype(bool)).mean():.3f} "
              f"| {(g.r_gtmask_at_gt - g.r_gt).mean():.3f} |")  # fmt: skip
    print("\n## D1 restricted to same_branch True points within 2 mm")
    s = p[(p.d_mm <= MATCH_MM) & p.same_branch.astype(bool)]
    for kind in ("lesion", "healthy"):
        e = s[s.kind == kind].signed_err
        print(f"{kind}: n={len(e)} mean {e.mean():.3f} median {e.median():.3f}")
    lum = pd.read_csv(out_dir / "v2_diag_lumen.csv")
    print("\n## D3 means per input (cases with >=1 lesion; errors dropped)")
    ok = lum[lum.error.isna()]
    print(ok.groupby("input")[["n_lesions", "mean_abs_rmin_err_mm", "mean_signed_rmin_err_mm",
                               "widened_rate", "n_no_lumen_warnings"]].mean().round(3)
          .loc[INPUTS].to_markdown())  # fmt: skip
    piv = ok.pivot(index="case_id", columns="input", values="mean_abs_rmin_err_mm")
    for other in ("pred", "gt_shift0.5", "gt_shift1.0"):
        d = piv["gt"] - piv[other]
        print(f"gt vs {other}: gt better in {(d < -1e-9).sum()}, worse in {(d > 1e-9).sum()}, "
              f"equal {(d.abs() <= 1e-9).sum()} of {d.notna().sum()}")  # fmt: skip
    print("gt vs pred_nolumen:",
          f"gt better {(piv['gt'] < piv['pred_nolumen'] - 1e-9).sum()} of {len(piv)}")  # fmt: skip
    print(
        "failed rows:",
        int(lum.error.notna().sum())
        + int(pd.read_csv(out_dir / "v2_diag_points.csv").error.notna().sum()),
    )  # noqa: E501


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize(a.out_dir) if a.summarize else run(a.out_dir)
