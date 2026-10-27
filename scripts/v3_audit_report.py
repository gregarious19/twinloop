"""A1 audit report (SPEC_V3 §6.2): tables per sigma_s, subset (sel/dev/all) and ds bin.

python scripts/v3_audit_report.py --csv /data/v3/audit/lesions.csv [--out reports/x.md]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from twinloop.paths import resolve

BINS = ["mild", "moderate", "severe", "modplus"]
BOOLS = ["missed_b025", "survive_b025", "missed_b05", "survive_b05", "calcified"]
HALFMAX_IOU_MIN = 0.60


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(resolve(path), dtype={"bin": str, "error": str})
    for c in BOOLS:
        d[c] = d[c].map({"True": True, "False": False, True: True, False: False})
    return d


def subset(d: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "dev":
        return d[d.case_id % 10 == 3]
    if name == "sel":
        return d[d.case_id % 10 != 3]
    return d


def table(d: pd.DataFrame) -> str:
    lines = [
        "| σ_s | bin | n lesions | lesions/case | survive σ_b 0.25 | survive σ_b 0.5 | calcified "
        "| image-supported (non-calc.) | median image_ds | median label_area_ds "
        "| dist_junction < 3 mm |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    ncase = d[d.status != "failed"].groupby("sigma_s").case_id.nunique()
    les = d[d.status == "ok"]
    for sig in sorted(les.sigma_s.unique()):
        ls = les[les.sigma_s == sig]
        for b in BINS:
            x = ls[ls.bin.isin(["moderate", "severe"])] if b == "modplus" else ls[ls.bin == b]
            n = len(x)
            if n == 0:
                lines.append(f"| {sig:g} | {b} | 0 | 0 | | | | | | | |")
                continue
            nc = x[~x.calcified.astype(bool) & x.image_ds.notna()]
            sup = f"{(nc.image_ds >= 0.25).mean():.2f} ({len(nc)})" if len(nc) else "n/a"
            lines.append(
                f"| {sig:g} | {b} | {n} | {n / max(ncase.get(sig, 1), 1):.2f} "
                f"| {x.survive_b025.mean():.2f} | {x.survive_b05.mean():.2f} "
                f"| {x.calcified.mean():.2f} | {sup} | {x.image_ds.median():.3f} "
                f"| {x.label_area_ds.median():.3f} | {(x.dist_junction_mm < 3).mean():.2f} |"
            )
    return "\n".join(lines)


def halfmax_usable(d: pd.DataFrame) -> tuple[float, bool]:
    d0 = d[(d.status == "ok") & (d.sigma_s == 0)]
    med = float(d0.iou_ref.median()) if d0.iou_ref.notna().any() else float("nan")
    return med, bool(med >= HALFMAX_IOU_MIN)


def report(d: pd.DataFrame) -> str:
    out = ["# v3 A1 — lesion audit", ""]
    failed = d[d.status == "failed"]
    out.append(
        f"Cases: {d.case_id.nunique()} distinct, {len(failed)} failed "
        f"({', '.join(map(str, failed.case_id)) or 'none'}). Rows with status `nolesions` mark "
        "a case/σ_s with no GT lesion. Survival, image support: SPEC_V3 §4.4–4.5."
    )
    for name in ("sel", "dev", "all"):
        out += ["", f"## Subset: {name}", "", table(subset(d, name))]
    med, usable = halfmax_usable(d)
    out += ["", f"median iou_ref over all D0 lesions: {med:.3f}; halfmax_usable: {usable}"]
    if len(failed):
        out += ["", "Failures:", *[f"- {r.case_id}: {r.error}" for r in failed.itertuples()]]
    return "\n".join(out) + "\n"


RETENTION_MIN, COUNT_FRAC_MIN, IMAGE_SUPPORT_MIN = 0.80, 0.40, 0.50
REPO = Path(__file__).resolve().parents[1]


def definition_stats(d: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    """Per σ_s: modplus count, retention (survive σ_b 0.5), image-supported fraction."""
    x = subset(d, subset_name)
    x = x[(x.status == "ok") & x.bin.isin(["moderate", "severe"])]
    rows = []
    for sig in sorted(d.sigma_s.unique()):
        m = x[x.sigma_s == sig]
        nc = m[~m.calcified.astype(bool) & m.image_ds.notna()]
        rows.append(
            {
                "sigma_s": sig,
                "count": len(m),
                "retention": float(m.survive_b05.mean()) if len(m) else float("nan"),
                "image_supported_frac": float((nc.image_ds >= 0.25).mean())
                if len(nc)
                else float("nan"),
                "n_noncalc": len(nc),
            }
        )
    return pd.DataFrame(rows)


def select(d: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    sel_ids = subset(d, "sel").case_id.unique()
    assert (sel_ids % 10 != 3).all(), "selection must use sel cases only"
    st = definition_stats(d, "sel")
    n0 = int(st.loc[st.sigma_s == 0, "count"].iloc[0])
    st["qualifies"] = (st.retention >= RETENTION_MIN) & (st["count"] >= COUNT_FRAC_MIN * n0)
    _, usable = halfmax_usable(d)
    aligned = json.loads((REPO / "reports" / "v3_A2_alignment.json").read_text())["dataset_aligned"]
    out = {
        "sigma_s_mm": None,
        "retention": None,
        "count": None,
        "count_d0": n0,
        "image_supported_frac": None,
        "halfmax_usable": usable,
        "dataset_aligned": bool(aligned),
        "g1": "fail",
        "g1_reason": "",
    }
    q = st[st.qualifies]
    if q.empty:
        best = st.loc[st.retention.idxmax()]
        out["g1_reason"] = (
            f"no definition has retention >= {RETENTION_MIN} with count >= {COUNT_FRAC_MIN} * "
            f"count(D0) = {COUNT_FRAC_MIN * n0:.0f}; best retention {best.retention:.3f} at "
            f"sigma_s {best.sigma_s:g} (count {int(best['count'])})"
        )
        return out, st
    r = q.iloc[0]
    out.update(
        sigma_s_mm=float(r.sigma_s),
        retention=float(r.retention),
        count=int(r["count"]),
        image_supported_frac=None
        if pd.isna(r.image_supported_frac)
        else float(r.image_supported_frac),
    )
    if usable and not (r.image_supported_frac >= IMAGE_SUPPORT_MIN):
        out["sigma_s_mm"] = None  # G1 failed
        out["g1_reason"] = (
            f"sigma* = {r.sigma_s:g} qualifies on robustness but the image-supported fraction "
            f"{r.image_supported_frac:.3f} < {IMAGE_SUPPORT_MIN}"
        )
        return out, st
    out["g1"] = "pass"
    out["g1_reason"] = (
        f"sigma* = {r.sigma_s:g}: retention {r.retention:.3f}, count {int(r['count'])} of {n0}"
        + ("" if usable else "; G1 passed on robustness only (half-max not usable)")
    )
    return out, st


def selection_report(d: pd.DataFrame, res: dict, st: pd.DataFrame) -> str:
    dev = definition_stats(d, "dev")
    md = ["# v3 A3 — definition selection and Gate G1", ""]
    md.append(
        f"Qualifying: retention (modplus survival under σ_b 0.5 mm) >= {RETENTION_MIN} and "
        f"count >= {COUNT_FRAC_MIN} * count(D0). Selection uses sel cases only; dev is shown, "
        "never used."
    )
    for name, t in (("sel (selection)", st), ("dev (check only)", dev)):
        md += ["", f"## {name}", "", "| definition | σ_s | modplus count | retention | "
               "image-supported (non-calc.) | n non-calc. | qualifies |",
               "|---|---|---|---|---|---|---|"]  # fmt: skip
        n0 = int(t["count"].iloc[0])
        for k, r in enumerate(t.itertuples()):
            q = r.retention >= RETENTION_MIN and r.count >= COUNT_FRAC_MIN * n0
            md.append(
                f"| D{k} | {r.sigma_s:g} | {r.count} | {r.retention:.3f} | "
                f"{r.image_supported_frac:.3f} | {r.n_noncalc} | {'yes' if q else 'no'} |"
            )
    md.append("")
    md.append(f"halfmax_usable: {res['halfmax_usable']}; dataset_aligned: {res['dataset_aligned']}")
    md += ["", f"**Gate G1: {res['g1'].upper()}.** {res['g1_reason']}"]
    return "\n".join(md) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="reports/v3_A1_lesion_audit.md")
    ap.add_argument("--select", action="store_true", help="A3: choose σ* and evaluate Gate G1")
    a = ap.parse_args(argv)
    if a.select:
        d = load(a.csv)
        res, st = select(d)
        (REPO / "reports" / "v3_definition.json").write_text(json.dumps(res, indent=2) + "\n")
        (REPO / "reports" / "v3_A3_definition.md").write_text(selection_report(d, res, st))
        print(json.dumps(res, indent=2))
        return 0
    md = report(load(a.csv))
    out = resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
