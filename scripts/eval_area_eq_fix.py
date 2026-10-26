"""S22: GT lesion counts per bin, inscribed vs area_eq before/after the junction exclusion.

Only the first 50 v2_train case ids, GT masks (no model, no test/holdout case).
"""

import argparse
import copy
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from twinloop.config import CenterlineParams, LesionParams
from twinloop.config_v2 import GraphParams
from twinloop.graph import build_graph
from twinloop.io import load_image, load_mask
from twinloop.lesions import detect_lesions
from twinloop.paths import case_image_path, case_label_path
from twinloop.radius import area_eq_radii
from twinloop.v2split import load_roles

MODES = ("inscribed", "area_eq", "area_eq_fixed")
BINS = ("mild", "moderate", "severe")


def bin_of(ds: float) -> str | None:
    if ds >= 0.7:
        return "severe"
    if ds >= 0.5:
        return "moderate"
    return "mild" if ds >= 0.25 else None


def count(trees, lp) -> dict[str, int]:
    out = dict.fromkeys(BINS, 0)
    for t in trees:
        for b in t.branches.values():
            for les in detect_lesions(b, lp):
                if (k := bin_of(les.ds)) is not None:
                    out[k] += 1
    return out


def one_case(cid: int) -> dict:
    cl, lp = CenterlineParams(), LesionParams()
    image = load_image(case_image_path(cid))
    gt = load_mask(case_label_path(cid), image)
    row = {"case_id": cid}
    try:
        g = build_graph(gt, cl, GraphParams())
        sets = {"inscribed": g.trees}
        a, f = copy.deepcopy(g.trees), copy.deepcopy(g.trees)
        area_eq_radii(a, gt, cl=cl)
        area_eq_radii(f, gt, cl=cl, exclude_junctions=True)
        sets |= {"area_eq": a, "area_eq_fixed": f}
        for m, trees in sets.items():
            for k, v in count(trees, lp).items():
                row[f"{m}_{k}"] = v
    except Exception as exc:  # kept as an error row
        row["error"] = repr(exc)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    ids = sorted(c for c, r in load_roles().items() if r == "v2_train")[:50]
    rows = Parallel(n_jobs=a.jobs)(delayed(one_case)(c) for c in ids)
    bad = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]
    tot = {m: {k: sum(r[f"{m}_{k}"] for r in ok) for k in BINS} for m in MODES}
    for m in MODES:
        tot[m]["total"] = sum(tot[m][k] for k in BINS)
    ins = tot["inscribed"]
    lines = [
        "# area_eq junction fix (S22)",
        "",
        f"First 50 `v2_train` case ids ({ids[0]}..{ids[-1]}), GT masks only; {len(ok)} scored, "
        f"{len(bad)} failed{': ' + str([(r['case_id'], r['error']) for r in bad]) if bad else ''}. "
        "Lesions: v1 `detect_lesions` on the GT graph (first two trees), binned by `Lesion.ds`: "
        "mild [0.25, 0.5), moderate [0.5, 0.7), severe >= 0.7. `area_eq_fixed` skips slices "
        "within 1.5 x local inscribed radius + 1.0 mm of a junction and interpolates across them.",
        "",
        "| radius | mild | moderate | severe | total | total / inscribed | severe / inscribed |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in MODES:
        t = tot[m]
        lines.append(
            f"| {m} | {t['mild']} | {t['moderate']} | {t['severe']} | {t['total']} | "
            f"{t['total'] / max(1, ins['total']):.2f} | {t['severe'] / max(1, ins['severe']):.2f} |"
        )
    lines += ["", "## Adoption criterion (S22; not applied, defaults unchanged)", ""]
    for m in ("area_eq", "area_eq_fixed"):
        r_tot, r_sev = (
            tot[m]["total"] / max(1, ins["total"]),
            tot[m]["severe"] / max(1, ins["severe"]),
        )
        ok_c = r_tot <= 1.25 and r_sev >= 0.5
        lines.append(
            f"- {m}: total {r_tot:.2f} x inscribed (need <= 1.25), severe {r_sev:.2f} x "
            f"inscribed (need >= 0.50) -> {'MEETS' if ok_c else 'DOES NOT MEET'} the criterion."
        )
    per_case = np.array([[r[f"{m}_{k}"] for m in MODES for k in BINS] for r in ok])
    changed = int(np.sum(per_case[:, 3:6].sum(1) != per_case[:, 6:9].sum(1)))
    lines += [
        "",
        f"Cases where the fix changed the lesion count vs unfixed area_eq: {changed} of {len(ok)}.",
        "",
        "Caveats: GT-mask lesions on 50 training cases; descriptive only; no clinical claim.",
    ]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
