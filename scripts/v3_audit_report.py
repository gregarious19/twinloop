"""A1 audit report (SPEC_V3 §6.2): tables per sigma_s, subset (sel/dev/all) and ds bin.

python scripts/v3_audit_report.py --csv /data/v3/audit/lesions.csv [--out reports/x.md]
"""

import argparse
import sys

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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="reports/v3_A1_lesion_audit.md")
    a = ap.parse_args(argv)
    md = report(load(a.csv))
    out = resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
