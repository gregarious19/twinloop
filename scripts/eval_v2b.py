"""S20 re-score on the 20 regression cases: radius_mode x source, severity-binned stenosis error.

    python scripts/eval_v2b.py --run [--shard K/N]      # writes <data>/v2/eval_b/parts/*.csv
    python scripts/eval_v2b.py --merge                  # reports/v2b_cases.csv, v2b_lesions.csv
    python scripts/eval_v2b.py --report                 # reports/v2b_results.md

Exploratory (the regression set was seen in S16/S18/S19). Same cases and first-pass masks as S16.
"""

import argparse
import copy
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from twinloop import metrics
from twinloop.config import CenterlineParams, LesionParams, PipelineParams
from twinloop.config_v2 import GraphParams
from twinloop.graph import build_graph
from twinloop.io import load_image, load_mask
from twinloop.paths import case_image_path, case_label_path, v2_dir
from twinloop.pipeline import V2Options, run_pipeline
from twinloop.radius import area_eq_radii

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_v2b import blurred  # noqa: E402
from eval_v2 import case_list, firstpass_path, graph_from_result, min_ffr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LP = LesionParams()
BLUR_SIGMA_MM = 0.5
CASE_COLS = [
    "case_id", "source", "radius_mode", "dice", "cl_dice", "branch_recall", "false_branch_rate",
    "n_lesions_gt", "widened_rate", "min_ffr_tree0", "min_ffr_tree1", "status", "error",
]  # fmt: skip
LESION_COLS = [
    "case_id", "source", "radius_mode", "lesion_id", "ds_gt", "ds_pred", "ds_err_pp", "rmin_gt",
    "rmin_pred", "rmin_err_mm", "missed", "bin", "ds_lesion", "rmin_lesion", "error",
]  # fmt: skip
# (source, mask kind, repair, lumen, radius modes). Non-repair rows run without TotalSegmentator
# and with the diameter-law flow (as v1), so inscribed rows equal the v1 result; repair rows keep
# the S16 settings (anatomy on, flow "auto").
SOURCES = [
    ("gt", "gt", False, False, ("inscribed", "area_eq")),
    ("gt_blur0.5", "blur", False, False, ("inscribed", "area_eq")),
    ("pred_v1", "pred", False, False, ("inscribed", "area_eq")),
    ("pred_lumen", "pred", False, True, ("area_eq",)),
    ("pred_repair", "pred", True, False, ("area_eq",)),
    ("pred_repair_lumen", "pred", True, True, ("area_eq",)),
]
BINS = ["mild", "moderate", "severe", "sub"]
PARTS = v2_dir() / "eval_b" / "parts"


def v2_options(repair: bool, lumen: bool, mode: str) -> V2Options:
    if repair:
        return V2Options(repair=True, lumen=lumen, radius_mode=mode)
    return V2Options(anatomy=False, lumen=lumen, flow_mode="diameter_law", radius_mode=mode)


def eval_row(cid, source, mkind, repair, lumen, mode, image, gt, g_gts) -> tuple[dict, list[dict]]:
    row = dict.fromkeys(CASE_COLS, "")
    row.update(case_id=cid, source=source, radius_mode=mode)
    lesions: list[dict] = []
    try:
        mask = {"gt": lambda: gt, "blur": lambda: blurred(gt, BLUR_SIGMA_MM),
                "pred": lambda: load_mask(firstpass_path(cid), image)}[mkind]()  # fmt: skip
        case_dir = v2_dir() / "eval_b" / f"{source}_{mode}" / f"{cid:04d}"
        run_pipeline(
            image, mask, PipelineParams(), case_dir, f"evalb_{cid:04d}", f"ImageCAS_{cid:04d}",
            v2=v2_options(repair, lumen, mode),
        )  # fmt: skip
        result = json.loads((case_dir / "result.json").read_text())
        final = load_mask(case_dir / "mask.nii.gz", image)
        g_final = build_graph(final, CenterlineParams(), GraphParams())
        g_res, g_gt = graph_from_result(result), g_gts[mode]
        a, b = sitk.GetArrayViewFromImage(final) > 0, sitk.GetArrayViewFromImage(gt) > 0
        rows = metrics.stenosis_errors(g_gt, g_res, LP)
        st = metrics.stenosis_metrics(g_gt, g_res, LP)
        assert len(rows) == st["n_lesions"]
        for r in rows:
            lesions.append(
                {**r, "case_id": cid, "source": source, "radius_mode": mode, "error": ""}
            )
        row.update(
            dice=metrics.dice(a, b), cl_dice=metrics.cl_dice(a, b),
            branch_recall=metrics.branch_recall(g_gt, final),
            false_branch_rate=metrics.false_branch_rate(g_final, gt),
            n_lesions_gt=len(rows), widened_rate=st["widened_rate"],
            min_ffr_tree0=min_ffr(result, "tree_0"), min_ffr_tree1=min_ffr(result, "tree_1"),
            status="ok",
        )  # fmt: skip
    except Exception as exc:  # a failed row is recorded, never dropped
        err = f"{type(exc).__name__}: {exc}".replace("\n", " ")
        row.update(status="failed", error=err)
        lesions = [{**dict.fromkeys(LESION_COLS, ""), "case_id": cid, "source": source,
                    "radius_mode": mode, "error": err}]  # fmt: skip
        print(traceback.format_exc(), file=sys.stderr)
    return row, lesions


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {(int(r["case_id"]), r["source"], r["radius_mode"]) for r in csv.DictReader(f)}


def append(path: Path, cols: list[str], rows: list[dict]) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, cols)
        if new:
            w.writeheader()
        w.writerows(rows)


def run(shard: str | None) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    k, n = (int(x) for x in shard.split("/")) if shard else (0, 1)
    cases = case_list("regression")[k::n]
    fc, fl = PARTS / f"cases_{k}of{n}.csv", PARTS / f"lesions_{k}of{n}.csv"
    done = done_keys(fc)
    for cid in cases:
        todo = [(s, mk, rp, lu, m) for s, mk, rp, lu, ms in SOURCES for m in ms
                if (cid, s, m) not in done]  # fmt: skip
        if not todo:
            continue
        print("case", cid, flush=True)
        image = load_image(case_image_path(cid))
        gt = load_mask(case_label_path(cid), image)
        g_ins = build_graph(gt, CenterlineParams(), GraphParams())
        g_eq = copy.deepcopy(g_ins)
        area_eq_radii(g_eq.trees, gt)
        g_gts = {"inscribed": g_ins, "area_eq": g_eq}
        for s, mk, rp, lu, m in todo:
            row, les = eval_row(cid, s, mk, rp, lu, m, image, gt, g_gts)
            append(fc, CASE_COLS, [row])
            append(fl, LESION_COLS, les)
            print(cid, s, m, row["status"], row["dice"], row["error"], flush=True)


def merge() -> None:
    for kind, cols, out in (("cases", CASE_COLS, "v2b_cases.csv"),
                            ("lesions", LESION_COLS, "v2b_lesions.csv")):  # fmt: skip
        parts = sorted(PARTS.glob(f"{kind}_*of*.csv"))
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        df = df.sort_values(["source", "radius_mode", "case_id"], kind="stable")
        df[cols].to_csv(REPO / "reports" / out, index=False)
        print(out, len(df))


# ---------------------------------------------------------------- report


def _f(x, fmt="{:+.1f}") -> str:
    return "n/a — no matched lesions" if x is None or not np.isfinite(x) else fmt.format(x)


def _stats_cells(s: dict) -> list[str]:
    if s["n"] == 0:
        return ["0", "n/a — no lesions"] + ["n/a — no lesions"] * 5
    return [
        str(s["n"]), str(s["n_missed"]),
        _f(s["mean_ds_err_pp"]), _f(s["median_ds_err_pp"]), _f(s["mean_abs_ds_err_pp"], "{:.1f}"),
        _f(s["mean_rmin_err_mm"], "{:+.3f}"), _f(s["mean_abs_rmin_err_mm"], "{:.3f}"),
    ]  # fmt: skip


def _summary(les: pd.DataFrame, source: str, mode: str) -> dict:
    g = les[(les.source == source) & (les.radius_mode == mode) & les.lesion_id.notna()]
    rows = g.assign(missed=g.missed.astype(bool)).to_dict("records")
    return metrics.summarize_stenosis(rows)


def report() -> None:
    cases = pd.read_csv(REPO / "reports/v2b_cases.csv")
    les = pd.read_csv(REPO / "reports/v2b_lesions.csv")
    L: list[str] = ["# S20 — area-equivalent radius, stenosis error metric, re-score", ""]
    L += ["Exploratory re-score of the 20 regression cases (SPEC_V2 §13b). Generated by "
          "`scripts/eval_v2b.py --report`; no SUPPORTED/FALSIFIED verdicts, no default changed "
          "for repair or lumen.", ""]  # fmt: skip
    ok = cases[cases.status == "ok"]
    L += ["Rows: " + ", ".join(
        f"{s}/{m} {int((ok.source == s).__and__(ok.radius_mode == m).sum())}/"
        f"{int(((cases.source == s) & (cases.radius_mode == m)).sum())} ok"
        for s, _, _, _, ms in SOURCES for m in ms) + ".", ""]  # fmt: skip

    L += ["## 1. Stenosis error per source, radius mode and severity bin", "",
          "`ds_err_pp` = 100·(ds_pred − ds_gt) (positive = prediction more stenosed, "
          "negative = stenosis under-called). "
          "`rmin_err` = r_pred − r_gt (mm). Bins by ds_gt: mild [0.25,0.5), moderate [0.5,0.7), "
          "severe [0.7,1.0]; `sub` < 0.25.", ""]  # fmt: skip
    hdr = ["source", "mode", "bin", "n", "n_missed", "mean ds_err_pp", "median ds_err_pp",
           "mean abs ds_err_pp", "mean rmin_err", "mean abs rmin_err"]  # fmt: skip
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    sums = {}
    for s, _, _, _, ms in SOURCES:
        for m in ms:
            sm = sums[(s, m)] = _summary(les, s, m)
            for b in ["mild", "moderate", "severe", "moderate_severe"]:
                st = sm["moderate_severe"] if b == "moderate_severe" else sm["bins"][b]
                L.append(f"| {s} | {m} | {b.replace('_', '+')} | " + " | ".join(_stats_cells(st))
                         + " |")  # fmt: skip
            if sm["bins"]["sub"]["n"]:
                L.append(f"| {s} | {m} | sub | " + " | ".join(_stats_cells(sm["bins"]["sub"]))
                         + " |")  # fmt: skip
    L += ["", "Down-calls (GT lesion ≥ threshold, matched, prediction below it):", "",
          "| source | mode | down50 | down70 |", "|---|---|---|---|"]  # fmt: skip
    for (s, m), sm in sums.items():
        L.append(f"| {s} | {m} | {sm['down50']}/{sm['down50_denom']} | "
                 f"{sm['down70']}/{sm['down70_denom']} |")  # fmt: skip

    L += [
        "",
        "## 2. GT lesion counts per bin by radius mode (source `gt`)",
        "",
        "| mode | mild | moderate | severe | sub | total |",
        "|---|---|---|---|---|---|",
    ]
    for m in ("inscribed", "area_eq"):
        sm = sums[("gt", m)]["bins"]
        tot = sum(sm[b]["n"] for b in BINS)
        L.append(f"| {m} | " + " | ".join(str(sm[b]["n"]) for b in BINS) + f" | {tot} |")
    L += ["", "Counts differ because lesions are detected on the radius profile, and r_eq ≠ r_ins "
          "(different ds = 1 − r/r_ref at every point), so lesions cross the 0.25 / 0.5 / 0.7 "
          "thresholds. The jump in mild lesions under area_eq is large (about 3x); its cause was "
          "not investigated here (a noisier r_eq profile from CPR cross-sections near "
          "junctions or neighbouring vessels is one candidate), so lesion counts and bins are "
          "not comparable across modes.", ""]  # fmt: skip

    L += ["## 3. FFR effect of the radius switch (min FFR per case, area_eq − inscribed)", "",
          "| source | n cases | mean diff | median diff | min diff | max diff | cross 0.80 down "
          "(≥0.80 → <0.80) | cross 0.80 up |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    ffr_cols = ["min_ffr_tree0", "min_ffr_tree1"]
    for s in ("gt", "pred_v1"):
        c = ok[ok.source == s].assign(case_min=lambda d: d[ffr_cols].min(axis=1))
        a = c[c.radius_mode == "inscribed"].set_index("case_id").case_min
        b = c[c.radius_mode == "area_eq"].set_index("case_id").case_min
        d = (b - a).dropna()
        a, b = a.loc[d.index], b.loc[d.index]
        down, up = int(((a >= 0.8) & (b < 0.8)).sum()), int(((a < 0.8) & (b >= 0.8)).sum())
        if len(d) == 0:
            L.append(f"| {s} | 0 | " + " | ".join(["n/a — no paired cases"] * 6) + " |")
            continue
        L.append(f"| {s} | {len(d)} | {d.mean():+.4f} | {d.median():+.4f} | {d.min():+.4f} | "
                 f"{d.max():+.4f} | {down} | {up} |")  # fmt: skip

    def ms(src, mode, key):
        return sums[(src, mode)]["moderate_severe"][key]

    def d50(src, mode):
        sm = sums[(src, mode)]
        return f"{sm['down50']}/{sm['down50_denom']}"

    L += ["", "## 4. Observations (area_eq, moderate + severe lesions, numbers only)", ""]
    for a, b, what in (("pred_lumen", "pred_v1", "lumen vs pred_v1"),
                       ("pred_repair", "pred_v1", "repair vs pred_v1"),
                       ("pred_repair_lumen", "pred_v1", "repair+lumen vs pred_v1"),
                       ("pred_v1", "gt_blur0.5", "pred_v1 vs gt_blur0.5 floor")):  # fmt: skip
        L.append(
            f"- {what}: mean ds_err_pp {_f(ms(a, 'area_eq', 'mean_ds_err_pp'))} vs "
            f"{_f(ms(b, 'area_eq', 'mean_ds_err_pp'))}; mean abs ds_err_pp "
            f"{_f(ms(a, 'area_eq', 'mean_abs_ds_err_pp'), '{:.1f}')} vs "
            f"{_f(ms(b, 'area_eq', 'mean_abs_ds_err_pp'), '{:.1f}')}; mean abs rmin_err "
            f"{_f(ms(a, 'area_eq', 'mean_abs_rmin_err_mm'), '{:.3f}')} vs "
            f"{_f(ms(b, 'area_eq', 'mean_abs_rmin_err_mm'), '{:.3f}')} mm; down50 "
            f"{d50(a, 'area_eq')} vs {d50(b, 'area_eq')}; n_missed "
            f"{ms(a, 'area_eq', 'n_missed')} vs {ms(b, 'area_eq', 'n_missed')}."
        )  # fmt: skip
    L += ["", "## 5. Caveats", "",
          "- Exploratory: n = 20 cases, all seen in S16/S18/S19; confirmatory tests need a new "
          "pre-selected stenosis-enriched set.",
          "- Few severe lesions; bin statistics rest on small n (see the `n` columns).",
          "- The metrics are relative to GT-mask lesions, not to invasive angiography.",
          "- FFR values are unvalidated (no invasive-FFR data) and not clinically accurate.",
          "- Non-repair rows run without TotalSegmentator and with diameter-law flow; repair rows "
          "use anatomy + flow mode `auto` as in S16, so FFR differs in flow model between "
          "those groups.", ""]  # fmt: skip
    (REPO / "reports/v2b_results.md").write_text("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--shard")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.run:
        run(a.shard)
    if a.merge:
        merge()
    if a.report:
        report()
