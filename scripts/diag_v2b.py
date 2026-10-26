"""S19 diagnostics: radius definition (inscribed vs area-equivalent) and widened-rate calibration.

    python scripts/diag_v2b.py --part radius --out reports/v2_diag2_radius.csv
    python scripts/diag_v2b.py --part calib  --out reports/v2_diag2_calib.csv
    python scripts/diag_v2b.py --summarize

Analysis only; regression-set cases only; no library code is modified.
"""

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import cKDTree

from twinloop import metrics
from twinloop.config import CenterlineParams, LesionParams, PipelineParams
from twinloop.config_v2 import GraphParams, V2Params
from twinloop.cpr import straighten_points
from twinloop.graph import CoronaryGraph, build_graph
from twinloop.io import load_image, load_mask
from twinloop.lesions import detect_lesions, reference_radius
from twinloop.lumen import refine_radii
from twinloop.metrics import MIN_LENGTH_MM, MIN_RADIUS_MM
from twinloop.modelio import load_model
from twinloop.paths import case_image_path, case_label_path, v2_dir
from twinloop.pipeline import run_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_v2 import all_branches, radius_map, read_at  # noqa: E402
from eval_v2 import case_list, firstpass_path, graph_from_result  # noqa: E402

LP = LesionParams()
CP = V2Params().cpr
MATCH_MM = 2.0
SIGMAS = [0.25, 0.5, 0.75, 1.0]
BINS = [(LP.ds_threshold, 0.5), (0.5, 0.7), (0.7, 1.0001)]
RADIUS_COLS = [
    "case_id", "branch_id", "point_idx", "kind", "ds_gt", "r_ins", "r_edt_at_pt", "r_eq",
    "r_eq_nn", "ecc", "is_lesion_min", "r_lumen_gtcl", "r_lumen_predcl", "error",
]  # fmt: skip
CALIB_COLS = [
    "case_id", "source", "dice", "cl_dice", "branch_recall", "false_branch_rate",
    "mean_abs_rmin_err_mm", "widened_rate", "n_lesions", "healthy_bias_mm", "lesion_bias_mm",
    *(f"n_les_bin{i}" for i in range(3)), *(f"n_wid_bin{i}" for i in range(3)), "error",
]  # fmt: skip
SOURCES = ["pred_v1", *(f"gt_s{s}" for s in SIGMAS)]


def req_profiles(image, gt, pts_mm) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per CPR slice: (s_mm, r_eq with the slice-centre component only, r_eq nearest component)."""
    res = straighten_points(image, gt, pts_mm, CP)
    c = CP.size_px // 2
    strict, nearest = [], []
    for m in res["mask"]:
        lab, n = ndimage.label(m > 0, structure=np.ones((3, 3), int))
        area = float((lab == lab[c, c]).sum()) if n and lab[c, c] > 0 else 0.0
        strict.append(np.sqrt(area * CP.pixel_mm**2 / np.pi))
        if n == 0:
            nearest.append(0.0)
        elif lab[c, c] > 0:
            nearest.append(strict[-1])
        else:
            yy, xx = np.nonzero(lab)
            j = int(np.argmin((yy - c) ** 2 + (xx - c) ** 2))
            nearest.append(np.sqrt((lab == lab[yy[j], xx[j]]).sum() * CP.pixel_mm**2 / np.pi))
    return res["s_mm"], np.array(strict), np.array(nearest)


def lumen_at_gt_lesions(g_gt, trees, model, image, device, same_geometry: bool) -> dict:
    """r_lumen at every GT lesion minimum: same branch/index for GT centerlines, nearest
    predicted point within 2 mm (the stenosis_metrics rule) for predicted ones."""
    trees = copy.deepcopy(trees)
    refine_radii(trees, image, model, CP, device)
    g = CoronaryGraph(trees, [], [])
    pred = all_branches(g)
    pts = np.concatenate([b.points_mm for b in pred])
    owner = [(b, i) for b in pred for i in range(len(b.s_mm))]
    out = {}
    for t in g_gt.trees:
        for b in t.branches.values():
            for les in detect_lesions(b, LP):
                if same_geometry:
                    r = float(g.trees[[x.tree_id for x in g.trees].index(t.tree_id)]
                              .branches[b.branch_id].radius_mm[les.min_idx])  # fmt: skip
                else:
                    d = np.linalg.norm(pts - b.points_mm[les.min_idx], axis=1)
                    j = int(np.argmin(d))
                    r = float(owner[j][0].radius_mm[owner[j][1]]) if d[j] <= MATCH_MM else np.nan
                out[(b.branch_id, les.min_idx)] = r
    return out


def radius_rows(cid: int, model, device) -> list[dict]:
    image = load_image(case_image_path(cid))
    gt = load_mask(case_label_path(cid), image)
    pm = load_mask(firstpass_path(cid), image)
    g_gt = build_graph(gt, CenterlineParams(), GraphParams())
    g_pred = build_graph(pm, CenterlineParams(), GraphParams())
    r_gtcl = lumen_at_gt_lesions(g_gt, g_gt.trees, model, image, device, True)
    r_predcl = lumen_at_gt_lesions(g_gt, g_pred.trees, model, image, device, False)
    edt = radius_map(gt)
    rows = []
    for b in all_branches(g_gt):
        if np.median(b.radius_mm) < MIN_RADIUS_MM or b.s_mm[-1] < MIN_LENGTH_MM:
            continue
        ds = np.full(len(b.s_mm), np.nan)
        kind = np.array(["healthy"] * len(b.s_mm), dtype=object)
        mins = set()
        for les in detect_lesions(b, LP):
            ds[les.start_idx : les.end_idx + 1] = les.ds
            kind[les.start_idx : les.end_idx + 1] = "lesion"
            mins.add(les.min_idx)
        s_k, strict, near = req_profiles(image, gt, b.points_mm)
        r_eq, r_eq_nn = np.interp(b.s_mm, s_k, strict), np.interp(b.s_mm, s_k, near)
        r_edt = read_at(edt, b.points_mm, gt)
        for k in range(len(b.s_mm)):
            ins = float(b.radius_mm[k])
            rows.append({
                "case_id": cid, "branch_id": b.branch_id, "point_idx": k, "kind": kind[k],
                "ds_gt": ds[k], "r_ins": ins, "r_edt_at_pt": float(r_edt[k]),
                "r_eq": float(r_eq[k]), "r_eq_nn": float(r_eq_nn[k]),
                "ecc": float(r_eq[k] / ins) if ins > 0 else np.nan,
                "is_lesion_min": k in mins,
                "r_lumen_gtcl": r_gtcl.get((b.branch_id, k), np.nan),
                "r_lumen_predcl": r_predcl.get((b.branch_id, k), np.nan), "error": "",
            })  # fmt: skip
    return rows


def lesion_records(g_gt, g_pred) -> list[tuple[float, bool]]:
    """(GT ds, widened) per GT lesion, with stenosis_metrics' rule."""
    pred = all_branches(g_pred)
    pts = np.concatenate([b.points_mm for b in pred])
    owner = [(b, i) for b in pred for i in range(len(b.s_mm))]
    refs = {b.branch_id: reference_radius(b.s_mm, b.radius_mm, LP) for b in pred}
    out = []
    for t in g_gt.trees:
        for b in t.branches.values():
            for les in detect_lesions(b, LP):
                d = np.linalg.norm(pts - b.points_mm[les.min_idx], axis=1)
                j = int(np.argmin(d))
                wid = True
                if d[j] <= MATCH_MM:
                    pb, i = owner[j]
                    r, ref_r = float(pb.radius_mm[i]), float(refs[pb.branch_id][i])
                    ds_pred = max(0.0, 1.0 - r / ref_r) if ref_r > 0 else 0.0
                    wid = ds_pred < 0.5 * les.ds
                out.append((les.ds, bool(wid)))
    return out


def radius_bias(g_gt, g_pred) -> tuple[float, float]:
    """Mean signed (r_pred - r_gt) over GT points matched within 2 mm: (healthy, lesion)."""
    pred = all_branches(g_pred)
    tree = cKDTree(np.concatenate([b.points_mm for b in pred]))
    rp = np.concatenate([b.radius_mm for b in pred])
    h, l_ = [], []
    for b in all_branches(g_gt):
        if np.median(b.radius_mm) < MIN_RADIUS_MM or b.s_mm[-1] < MIN_LENGTH_MM:
            continue
        in_les = np.zeros(len(b.s_mm), bool)
        for les in detect_lesions(b, LP):
            in_les[les.start_idx : les.end_idx + 1] = True
        d, j = tree.query(b.points_mm)
        e = rp[j] - b.radius_mm
        ok = d <= MATCH_MM
        h += list(e[ok & ~in_les])
        l_ += list(e[ok & in_les])
    return (float(np.mean(h)) if h else np.nan, float(np.mean(l_)) if l_ else np.nan)


def blurred(gt: sitk.Image, sigma_mm: float) -> sitk.Image:
    sp = np.array(gt.GetSpacing())[::-1]
    a = ndimage.gaussian_filter((sitk.GetArrayViewFromImage(gt) > 0).astype(np.float32),
                                sigma=sigma_mm / sp)  # fmt: skip
    out = sitk.GetImageFromArray((a > 0.5).astype(np.uint8))
    out.CopyInformation(gt)
    return out


def calib_row(cid: int, source: str, image, gt, g_gt, mask) -> dict:
    row = dict.fromkeys(CALIB_COLS, "")
    row.update(case_id=cid, source=source)
    try:
        case_dir = v2_dir() / "diag2" / source / f"{cid:04d}"
        run_pipeline(
            image, mask, PipelineParams(), case_dir, f"diag2_{cid:04d}", f"ImageCAS_{cid:04d}",
            v2=None,
        )  # fmt: skip
        result = json.loads((case_dir / "result.json").read_text())
        final = load_mask(case_dir / "mask.nii.gz", image)
        g_final = build_graph(final, CenterlineParams(), GraphParams())
        g_res = graph_from_result(result)
        a, b = sitk.GetArrayViewFromImage(final) > 0, sitk.GetArrayViewFromImage(gt) > 0
        st = metrics.stenosis_metrics(g_gt, g_res, LP)
        rec = lesion_records(g_gt, g_res)
        assert len(rec) == st["n_lesions"]
        assert not rec or np.isclose(np.mean([w for _, w in rec]), st["widened_rate"])
        hb, lb = radius_bias(g_gt, g_res)
        row.update(
            dice=metrics.dice(a, b), cl_dice=metrics.cl_dice(a, b),
            branch_recall=metrics.branch_recall(g_gt, final),
            false_branch_rate=metrics.false_branch_rate(g_final, gt),
            mean_abs_rmin_err_mm=st["mean_abs_rmin_err_mm"], widened_rate=st["widened_rate"],
            n_lesions=st["n_lesions"], healthy_bias_mm=hb, lesion_bias_mm=lb,
        )  # fmt: skip
        for i, (lo, hi) in enumerate(BINS):
            sel = [w for ds, w in rec if lo <= ds < hi]
            row[f"n_les_bin{i}"], row[f"n_wid_bin{i}"] = len(sel), int(sum(sel))
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
        print(traceback.format_exc(), file=sys.stderr)
    return row


def run(part: str, out: Path) -> None:
    rows: list[dict] = []
    if part == "radius":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model("lumen", device)
    for cid in case_list("regression"):
        print(part, "case", cid, flush=True)
        try:
            if part == "radius":
                rows += radius_rows(cid, model, device)
                continue
            image = load_image(case_image_path(cid))
            gt = load_mask(case_label_path(cid), image)
            g_gt = build_graph(gt, CenterlineParams(), GraphParams())
            for src in SOURCES:
                mask = load_mask(firstpass_path(cid), image) if src == "pred_v1" else blurred(
                    gt, float(src[4:])
                )  # fmt: skip
                rows.append(calib_row(cid, src, image, gt, g_gt, mask))
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}".replace("\n", " ")
            print(traceback.format_exc(), file=sys.stderr)
            if part == "radius":
                rows.append({**dict.fromkeys(RADIUS_COLS, ""), "case_id": cid, "error": err})
            else:
                rows += [
                    {**dict.fromkeys(CALIB_COLS, ""), "case_id": cid, "source": s, "error": err}
                    for s in SOURCES
                ]
    cols = RADIUS_COLS if part == "radius" else CALIB_COLS
    pd.DataFrame(rows, columns=cols).to_csv(out, index=False)


def _err(g, col, ref):
    return (g[col] - g[ref]).abs().mean(), (g[col] - g[ref]).mean()


def summarize() -> None:
    a = pd.read_csv("reports/v2_diag2_radius.csv")
    print("failed radius rows:", int(a.error.notna().sum()))
    a = a[a.error.isna()]
    a = a.assign(diff=a.r_eq - a.r_ins)
    print("## C1 by kind (r_ins > 0)")
    print(
        "| kind | n | mean r_eq-r_ins | median r_eq-r_ins | mean ecc | median ecc | frac r_eq=0 |"
    )
    print("|---|---|---|---|---|---|---|")
    for k, g in [("lesion", a[a.kind == "lesion"]), ("healthy", a[a.kind == "healthy"]),
                 ("lesion minimum", a[a.is_lesion_min.astype(bool)])]:  # fmt: skip
        g = g[g.r_ins > 0]
        d = g["diff"]
        print(
            f"| {k} | {len(g)} | {d.mean():.3f} | {d.median():.3f} | {g.ecc.mean():.3f} "
            f"| {g.ecc.median():.3f} | {(g.r_eq == 0).mean():.3f} |"
        )
    m = a[a.is_lesion_min.astype(bool)]
    print(f"\n## Lumen error at GT lesion minima (n={len(m)})")
    print(
        "| input | n valid | vs r_ins: mean abs | vs r_ins: signed | "
        "vs r_eq: mean abs | vs r_eq: signed | vs r_eq_nn: mean abs | vs r_eq_nn: signed |"
    )
    print("|---|---|---|---|---|---|---|---|")  # fmt: skip
    for name, col in [("(i) GT centerlines", "r_lumen_gtcl"), ("(ii) predicted centerlines",
                                                              "r_lumen_predcl")]:  # fmt: skip
        g = m[m[col].notna()]
        (ai, si), (ae, se), (an, sn) = (_err(g, col, r) for r in ("r_ins", "r_eq", "r_eq_nn"))
        print(f"| {name} | {len(g)} | {ai:.3f} | {si:+.3f} | {ae:.3f} | {se:+.3f} "
              f"| {an:.3f} | {sn:+.3f} |")  # fmt: skip
    c = pd.read_csv("reports/v2_diag2_calib.csv")
    print("\nfailed calib rows:", int(c.error.notna().sum()))
    c = c[c.error.isna()]
    cols = ["dice", "cl_dice", "branch_recall", "false_branch_rate", "mean_abs_rmin_err_mm",
            "widened_rate", "healthy_bias_mm", "lesion_bias_mm", "n_lesions"]  # fmt: skip
    print("\n## C2 means over cases")
    print(c.groupby("source")[cols].mean().round(3).loc[SOURCES].to_markdown())
    print("\n## widened by severity bin (pooled over cases)")
    print("| source | bin ds 0.25-0.5: widened/lesions | 0.5-0.7 | 0.7-1.0 |")
    print("|---|---|---|---|")
    for s in SOURCES:
        g = c[c.source == s]
        cells = [f"{int(g[f'n_wid_bin{i}'].sum())}/{int(g[f'n_les_bin{i}'].sum())} "
                 f"({g[f'n_wid_bin{i}'].sum() / max(1, g[f'n_les_bin{i}'].sum()):.2f})"
                 for i in range(3)]  # fmt: skip
        print(f"| {s} | " + " | ".join(cells) + " |")
    # sanity: pred_v1 here vs S16
    e = pd.read_csv("reports/v2_eval.csv")
    e = e[e.variant == "pred_v1"].set_index("case_id")
    p = c[c.source == "pred_v1"].set_index("case_id")
    dd, dw = (p.dice - e.dice).abs().max(), (p.widened_rate - e.widened_rate).abs().max()
    print(f"\nmax |pred_v1 here - S16| over cases: dice {dd:.2e} widened {dw:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["radius", "calib"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize() if a.summarize else run(a.part, a.out)
