"""S24: H3 (lumen v3) evaluation on the 30-case stenosis set (SPEC_V2 §13c). Run once.

    python scripts/eval_v3.py --run [--shard K/N]     # writes <data>/v2/eval_v3/parts/*.csv
    python scripts/eval_v3.py --merge                 # reports/v3_cases.csv, reports/v3_lesions.csv
    python scripts/eval_v3.py --report                # reports/H3_RESULTS.md

All rows: inscribed radii, no repair, no anatomy, v1 diameter-law flow. The lumen v3 inference
settings are read from reports/v3_lumen_frozen.json (frozen in S23 on validation cases).
"""

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from twinloop import metrics
from twinloop.config import CenterlineParams, LesionParams, PipelineParams
from twinloop.config_v2 import GraphParams
from twinloop.graph import build_graph
from twinloop.io import load_image, load_mask
from twinloop.paths import case_image_path, case_label_path, v2_dir
from twinloop.pipeline import V2Options, run_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_v2 import firstpass_path, graph_from_result, min_ffr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LP = LesionParams()
FROZEN = json.loads((REPO / "reports/v3_lumen_frozen.json").read_text())
CASE_COLS = [
    "case_id", "source", "dice", "n_lesions_gt", "false_sig", "min_ffr_tree0", "min_ffr_tree1",
    "min_ffr", "fallback_fraction", "status", "error",
]  # fmt: skip
LESION_COLS = [
    "case_id", "source", "lesion_id", "ds_gt", "ds_pred", "ds_err_pp", "rmin_gt", "rmin_pred",
    "rmin_err_mm", "missed", "bin", "ds_lesion", "rmin_lesion", "error",
]  # fmt: skip


def opts(**kw) -> V2Options:
    return V2Options(anatomy=False, flow_mode="diameter_law", radius_mode="inscribed", **kw)


# source -> (mask kind, V2Options)
SOURCES = {
    "gt_v1": ("gt", opts()),
    "pred_v1": ("pred", opts()),
    "pred_lumen_old": ("pred", opts(lumen=True)),
    "pred_lumen3": ("pred", opts(
        lumen=True, lumen_model="lumen_v3", lumen_radius_out=FROZEN["radius_out"],
        lumen_fallback=FROZEN["fallback"], lumen_sigma=FROZEN["sigma"])),
    "gt_lumen3": ("gt", opts(
        lumen=True, lumen_model="lumen_v3", lumen_radius_out=FROZEN["radius_out"],
        lumen_fallback=FROZEN["fallback"], lumen_sigma=FROZEN["sigma"])),
}  # fmt: skip
PARTS = v2_dir() / "eval_v3" / "parts"
CASES = json.loads((REPO / "reports/v3_stenosis_set.json").read_text())["case_ids"]


def eval_row(cid, source, image, gt, g_gt) -> tuple[dict, list[dict]]:
    mkind, v2 = SOURCES[source]
    row = dict.fromkeys(CASE_COLS, "")
    row.update(case_id=cid, source=source)
    lesions: list[dict] = []
    try:
        mask = gt if mkind == "gt" else load_mask(firstpass_path(cid), image)
        case_dir = v2_dir() / "eval_v3" / source / f"{cid:04d}"
        run_pipeline(
            image, mask, PipelineParams(), case_dir, f"evalv3_{cid:04d}", f"ImageCAS_{cid:04d}",
            v2=v2,
        )  # fmt: skip
        result = json.loads((case_dir / "result.json").read_text())
        final = load_mask(case_dir / "mask.nii.gz", image)
        g_res = graph_from_result(result)
        rows = metrics.stenosis_errors(g_gt, g_res, LP)
        lesions = [{**r, "case_id": cid, "source": source, "error": ""} for r in rows]
        f0, f1 = min_ffr(result, "tree_0"), min_ffr(result, "tree_1")
        row.update(
            dice=metrics.dice(final, gt),
            n_lesions_gt=len(rows), false_sig=metrics.false_sig_lesions(g_gt, g_res, LP),
            min_ffr_tree0=f0, min_ffr_tree1=f1,
            min_ffr=min([f for f in (f0, f1) if f is not None], default=""),
            fallback_fraction=(result.get("v2") or {}).get("lumen_fallback_fraction") or "",
            status="ok",
        )  # fmt: skip
    except Exception as exc:  # a failed row is recorded, never dropped
        err = f"{type(exc).__name__}: {exc}".replace("\n", " ")
        row.update(status="failed", error=err)
        lesions = [{**dict.fromkeys(LESION_COLS, ""), "case_id": cid, "source": source,
                    "error": err}]  # fmt: skip
        print(traceback.format_exc(), file=sys.stderr)
    return row, lesions


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
    fc, fl = PARTS / f"cases_{k}of{n}.csv", PARTS / f"lesions_{k}of{n}.csv"
    done = set()
    if fc.exists():
        with open(fc, newline="") as f:
            done = {(int(r["case_id"]), r["source"]) for r in csv.DictReader(f)}
    for cid in CASES[k::n]:
        todo = [s for s in SOURCES if (cid, s) not in done]
        if not todo:
            continue
        print("case", cid, flush=True)
        image = load_image(case_image_path(cid))
        gt = load_mask(case_label_path(cid), image)
        g_gt = build_graph(gt, CenterlineParams(), GraphParams())
        for s in todo:
            row, les = eval_row(cid, s, image, gt, g_gt)
            append(fc, CASE_COLS, [row])
            append(fl, LESION_COLS, les)
            print(cid, s, row["status"], row["dice"], row["error"], flush=True)


def merge() -> None:
    for kind, cols, out in (("cases", CASE_COLS, "v3_cases.csv"),
                            ("lesions", LESION_COLS, "v3_lesions.csv")):  # fmt: skip
        df = pd.concat([pd.read_csv(p) for p in sorted(PARTS.glob(f"{kind}_*of*.csv"))])
        df = df.sort_values(["source", "case_id"], kind="stable")
        df[cols].to_csv(REPO / "reports" / out, index=False)
        print(out, len(df))


# ---------------------------------------------------------------- report

SRC = list(SOURCES)
BIN_ROWS = ["mild", "moderate", "severe", "moderate_severe"]


def _f(x, fmt="{:+.1f}") -> str:
    return "n/a (no matched lesions)" if x is None or not np.isfinite(x) else fmt.format(x)


def _cells(s: dict) -> list[str]:
    if s["n"] == 0:
        return ["0"] + ["n/a (no lesions)"] * 6
    return [
        str(s["n"]), str(s["n_missed"]), _f(s["mean_ds_err_pp"]), _f(s["median_ds_err_pp"]),
        _f(s["mean_abs_ds_err_pp"], "{:.1f}"), _f(s["mean_rmin_err_mm"], "{:+.3f}"),
        _f(s["mean_abs_rmin_err_mm"], "{:.3f}"),
    ]  # fmt: skip


def report() -> None:
    cases = pd.read_csv(REPO / "reports/v3_cases.csv")
    les = pd.read_csv(REPO / "reports/v3_lesions.csv")
    ok_cases = {s: set(cases[(cases.source == s) & (cases.status == "ok")].case_id) for s in SRC}
    paired = ok_cases["pred_v1"] & ok_cases["pred_lumen3"]  # verdict uses cases ok in both

    def summ(src, restrict=None) -> dict:
        g = les[(les.source == src) & les.lesion_id.notna()]
        if restrict is not None:
            g = g[g.case_id.isin(restrict)]
        return metrics.summarize_stenosis(g.assign(missed=g.missed.astype(bool)).to_dict("records"))

    L = ["# H3 results — lumen v3 on the 30-case stenosis set (S24)", "",
         "Pre-registered in SPEC_V2 §13c (S21); run once; nothing changed after seeing results. "
         "Cases: `reports/v3_stenosis_set.json`. All rows: first-pass or GT mask as stated, "
         "inscribed radii, no repair, no anatomy, v1 diameter-law flow. `pred_lumen3` settings "
         f"(frozen in S23 on validation cases): radius_out={FROZEN['radius_out']}, "
         f"fallback={FROZEN['fallback']}, sigma={FROZEN['sigma']}.", ""]  # fmt: skip
    L += ["Rows ok / total: " + ", ".join(
        f"{s} {len(ok_cases[s])}/{int((cases.source == s).sum())}" for s in SRC) + ".",
        f"Verdict computed on the {len(paired)} cases where pred_v1 and pred_lumen3 both ran.",
        ""]  # fmt: skip

    L += ["## 1. Stenosis error per source and severity bin", "",
          "`ds_err_pp` = 100·(ds_pred − ds_gt), negative = stenosis under-called; "
          "`rmin_err` = r_pred − r_gt (mm). Bins by ds_gt (§13b). All ok cases of each source.",
          ""]  # fmt: skip
    hdr = ["source", "bin", "n", "n_missed", "mean ds_err_pp", "median ds_err_pp",
           "mean abs ds_err_pp", "mean rmin_err", "mean abs rmin_err"]  # fmt: skip
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    sums = {s: summ(s) for s in SRC}
    for s in SRC:
        for b in BIN_ROWS:
            st = sums[s]["moderate_severe"] if b == "moderate_severe" else sums[s]["bins"][b]
            L.append(f"| {s} | {b.replace('_', '+')} | " + " | ".join(_cells(st)) + " |")
    L += ["", "## 2. Down-calls and false significant lesions", "",
          "| source | down50 | down70 | false_sig lesions (total, ok cases) |",
          "|---|---|---|---|"]  # fmt: skip
    fs = {s: int(cases[(cases.source == s) & (cases.status == "ok")].false_sig.astype(float).sum())
          for s in SRC}  # fmt: skip
    for s in SRC:
        sm = sums[s]
        L.append(f"| {s} | {sm['down50']}/{sm['down50_denom']} | {sm['down70']}/"
                 f"{sm['down70_denom']} | {fs[s]} |")  # fmt: skip

    # ---- verdict (paired cases)
    a, b = summ("pred_lumen3", paired), summ("pred_v1", paired)
    d_primary = a["moderate_severe"]["mean_ds_err_pp"] - b["moderate_severe"]["mean_ds_err_pp"]
    fs_pair = {s: int(cases[(cases.source == s) & cases.case_id.isin(paired)].false_sig.astype(
        float).sum()) for s in ("pred_v1", "pred_lumen3")}  # fmt: skip
    d_fs = fs_pair["pred_lumen3"] - fs_pair["pred_v1"]
    d_mild = a["bins"]["mild"]["mean_abs_ds_err_pp"] - b["bins"]["mild"]["mean_abs_ds_err_pp"]
    conds = {
        "PRIMARY: mean ds_err_pp (moderate+severe) pred_lumen3 − pred_v1 >= +10 pp":
            (d_primary, np.isfinite(d_primary) and d_primary >= 10.0),
        "GUARD (a): down50 pred_lumen3 <= pred_v1":
            (a["down50"] - b["down50"], a["down50"] <= b["down50"]),
        "GUARD (b): false_sig_lesions increase <= 5":
            (d_fs, d_fs <= 5),
        "GUARD (c): mild-bin mean |ds_err_pp| increase <= 5 pp":
            (d_mild, np.isfinite(d_mild) and d_mild <= 5.0),
    }  # fmt: skip
    verdict = "SUPPORTED" if all(v[1] for v in conds.values()) else "FALSIFIED"
    L += ["", "## 3. H3 verdict", "", f"**H3 is {verdict}.**", "",
          "| condition | pred_lumen3 | pred_v1 | difference | holds |",
          "|---|---|---|---|---|"]  # fmt: skip
    vals = {
        0: (a["moderate_severe"]["mean_ds_err_pp"], b["moderate_severe"]["mean_ds_err_pp"]),
        1: (a["down50"], b["down50"]), 2: (fs_pair["pred_lumen3"], fs_pair["pred_v1"]),
        3: (a["bins"]["mild"]["mean_abs_ds_err_pp"], b["bins"]["mild"]["mean_abs_ds_err_pp"]),
    }  # fmt: skip
    for i, (name, (diff, holds)) in enumerate(conds.items()):
        L.append(f"| {name} | {_f(vals[i][0], '{:.2f}')} | {_f(vals[i][1], '{:.2f}')} | "
                 f"{_f(diff, '{:+.2f}')} | {'yes' if holds else 'NO'} |")  # fmt: skip

    # ---- secondary: FFR
    L += ["", "## 4. Secondary (no verdict): minimum FFR per case versus gt_v1", ""]
    piv = cases[cases.status == "ok"].pivot(index="case_id", columns="source", values="min_ffr")
    piv = piv.apply(pd.to_numeric, errors="coerce")
    pos = piv.index[piv["gt_v1"] < 0.80] if "gt_v1" in piv else []
    L += [f"Cases with gt_v1 min FFR < 0.80: {len(pos)} of {int(piv['gt_v1'].notna().sum())}.", "",
          "| source | n cases | mean abs diff vs gt_v1 | gt_v1 <0.80 also <0.80 | false calls "
          "(<0.80 where gt_v1 >= 0.80) |", "|---|---|---|---|---|"]  # fmt: skip
    for s in SRC:
        if s == "gt_v1" or s not in piv:
            continue
        both = piv[["gt_v1", s]].dropna()
        hit = int(((both.gt_v1 < 0.8) & (both[s] < 0.8)).sum())
        n_pos = int((both.gt_v1 < 0.8).sum())
        fc = int(((both.gt_v1 >= 0.8) & (both[s] < 0.8)).sum())
        L.append(f"| {s} | {len(both)} | {(both[s] - both.gt_v1).abs().mean():.3f} | "
                 f"{hit}/{n_pos} | {fc} |")  # fmt: skip
    ff = cases[(cases.source == "pred_lumen3") & (cases.status == "ok")].fallback_fraction
    ff = pd.to_numeric(ff, errors="coerce")
    L += ["", f"Lumen v3 fallback fraction (slices whose centre pixel is not in a predicted "
          f"component), pred_lumen3: mean {ff.mean():.3f}, max {ff.max():.3f}.", ""]  # fmt: skip
    L += ["## 5. Caveats", "",
          "- Lesions are defined on GT-mask radius profiles, not invasive angiography; FFR values "
          "are unvalidated (no invasive-FFR data) and not clinically accurate.",
          "- n = 30 cases, single centre (ImageCAS); case set enriched for GT stenoses by design.",
          "- `gt_lumen3` (GT mask + lumen v3) is an oracle for perfect centerlines, reported "
          "without a verdict; `pred_lumen_old` is the deployed lumen model, for reference.",
          "- Missed lesions (no predicted point within 2 mm) are excluded from error means and "
          "counted in `n_missed`; `down50` counts only matched lesions.", ""]  # fmt: skip
    (REPO / "reports/H3_RESULTS.md").write_text("\n".join(L))
    print("VERDICT", verdict, {k: v for k, v in conds.items()})


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
