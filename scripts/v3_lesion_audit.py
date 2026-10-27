"""A1 lesion audit (SPEC_V3 §6.2): GT stenoses under radius smoothing, blur survival, image support.

    python scripts/v3_lesion_audit.py --cases first3 --sigmas 0,1.0 --out /data/v3/audit/slice.csv

Resumable: cases already in the CSV are skipped. v2_train cases only.
"""

import argparse
import copy
import csv
import math
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from twinloop import metrics
from twinloop.config import CenterlineParams, LesionParams
from twinloop.config_v2 import GraphParams, V2Params
from twinloop.cpr import center_component, straighten
from twinloop.graph import build_graph
from twinloop.halfmax import _DIST_MM, auto_mask
from twinloop.io import load_image, load_mask
from twinloop.lesions import detect_lesions, reference_radius
from twinloop.paths import case_image_path, case_label_path, resolve
from twinloop.radius_smooth import blur_mask, smooth_graph
from twinloop.v2split import load_roles

COLUMNS = [
    "case_id", "sigma_s", "lesion_id", "branch_id", "ds_gt", "bin", "r_min_mm", "r_ref_mm",
    "length_mm", "dist_junction_mm", "ds_b025", "missed_b025", "survive_b025", "ds_b05",
    "missed_b05", "survive_b05", "image_ds", "label_area_ds", "calcified", "iou_ref", "status",
    "error",
]  # fmt: skip
BLURS = {"b025": 0.25, "b05": 0.5}
NAN = float("nan")


def case_ids(which: str) -> list[int]:
    roles = load_roles()
    train = sorted(c for c, r in roles.items() if r == "v2_train")
    if which == "first3":
        return train[:3]
    if which == "dev":
        return [c for c in train if c % 10 == 3]
    if which == "sel":
        return [c for c in train if c % 10 != 3]
    if which == "all":
        return train
    raise SystemExit(f"unknown --cases {which}")


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return ""
        return "inf" if math.isinf(v) else f"{v:.5g}"
    return v


class SliceCache:
    """Per-branch straightened image/GT mask with per-slice half-max and label areas."""

    def __init__(self, image, gt, cp):
        self.image, self.gt, self.cp, self._c = image, gt, cp, {}

    def get(self, b):
        if b.branch_id not in self._c:
            res = straighten(self.image, self.gt, b, self.cp)
            n = len(res["s_mm"])
            auto_a, lab_a, iou, calc = np.zeros(n), np.zeros(n), np.full(n, NAN), np.zeros(n, bool)
            for k in range(n):
                hu = res["img"][k]
                auto = auto_mask(hu)
                lab = center_component(res["mask"][k] > 0.5)
                auto_a[k], lab_a[k] = auto.sum(), lab.sum()
                u = (auto | lab).sum()
                if u:
                    iou[k] = (auto & lab).sum() / u
                calc[k] = bool((hu[_DIST_MM <= 2.0] >= 800.0).any())
            self._c[b.branch_id] = (res["s_mm"], auto_a, lab_a, iou, calc)
        return self._c[b.branch_id]


def _ds_from(a_min: float, a_ref: float) -> float:
    if not a_ref > 0 or math.isnan(a_min):
        return NAN
    return max(0.0, 1.0 - math.sqrt(a_min / a_ref))


def image_support(cache, b, les, lp) -> dict:
    s, auto_a, lab_a, iou, calc = cache.get(b)
    s_min = b.s_mm[les.min_idx]
    lo, hi = b.s_mm[les.start_idx], b.s_mm[les.end_idx]
    out = dict(image_ds=NAN, label_area_ds=NAN, calcified=False, iou_ref=NAN)
    mn = np.abs(s - s_min) <= 0.5
    if not mn.any():
        return out
    r_ref = reference_radius(b.s_mm, b.radius_mm, lp)
    ds_s = 1.0 - np.interp(s, b.s_mm, b.radius_mm) / np.interp(s, b.s_mm, r_ref)
    dist_out = np.where(s < lo, lo - s, np.where(s > hi, s - hi, -1.0))
    ref = (dist_out >= 3.0) & (dist_out <= 8.0) & (ds_s < 0.25)
    out["calcified"] = bool(calc[mn].any())
    if ref.any():
        out["image_ds"] = _ds_from(np.median(auto_a[mn]), np.median(auto_a[ref]))
        out["label_area_ds"] = _ds_from(np.median(lab_a[mn]), np.median(lab_a[ref]))
        out["iou_ref"] = float(np.nanmedian(iou[ref])) if np.isfinite(iou[ref]).any() else NAN
    return out


def dist_junction(b, les) -> float:
    s_min = b.s_mm[les.min_idx]
    d = []
    if b.parent_id is not None:
        d.append(s_min - b.s_mm[0])
    if b.children:
        d.append(b.s_mm[-1] - s_min)
    return float(min(d)) if d else float("inf")


def survival(row: dict) -> bool:
    if row["missed"]:
        return False
    return row["ds_pred"] >= (0.5 if row["ds_gt"] >= 0.5 else 0.25)


def audit_case(args) -> list[dict]:
    cid, sigmas = args
    t0 = time.perf_counter()
    try:
        image = load_image(case_image_path(cid))
        gt = load_mask(case_label_path(cid), image)
        cp, lp = CenterlineParams(), LesionParams()
        g_gt = build_graph(gt, cp, GraphParams())
        g_b = {}
        for key, sb in BLURS.items():
            try:
                g_b[key] = build_graph(blur_mask(gt, sb), cp, GraphParams())
            except ValueError:  # blur removed every tree: all lesions count as missed
                g_b[key] = None
        cache = SliceCache(image, gt, V2Params().cpr)
        rows = []
        for sig in sigmas:
            gs = copy.deepcopy(g_gt)
            smooth_graph(gs, sig)
            surv = {}
            for key, g in g_b.items():
                if g is None:
                    surv[key] = None
                    continue
                gb = copy.deepcopy(g)
                smooth_graph(gb, sig)
                surv[key] = metrics.stenosis_errors(gs, gb, lp)
            lesions = [(b, les) for t in gs.trees for b in t.branches.values()
                       for les in detect_lesions(b, lp)]  # fmt: skip
            base = dict.fromkeys(COLUMNS, "")
            base.update(case_id=cid, sigma_s=sig, status="ok")
            if not lesions:
                rows.append({**base, "status": "nolesions"})
                continue
            ref_rows = metrics.stenosis_errors(gs, gs, lp)  # window ds_gt / bin in lesion order
            for i, (b, les) in enumerate(lesions):
                rr = ref_rows[i]
                row = {
                    **base, "lesion_id": f"{b.branch_id}_p{les.min_idx}", "branch_id": b.branch_id,
                    "ds_gt": rr["ds_gt"], "bin": rr["bin"], "r_min_mm": les.r_min_mm,
                    "r_ref_mm": les.r_ref_mm, "length_mm": les.length_mm,
                    "dist_junction_mm": dist_junction(b, les),
                }  # fmt: skip
                for key in BLURS:
                    if surv[key] is None:
                        row.update(
                            {f"ds_{key}": NAN, f"missed_{key}": True, f"survive_{key}": False}
                        )
                    else:
                        r = surv[key][i]
                        assert r["lesion_id"] == row["lesion_id"]
                        row.update({f"ds_{key}": r["ds_pred"], f"missed_{key}": r["missed"],
                                    f"survive_{key}": survival(r)})  # fmt: skip
                row.update(image_support(cache, b, les, lp))
                rows.append(row)
        rows[0]["error"] = rows[0]["error"] or ""
        print(f"case {cid}: {len(rows)} rows, {time.perf_counter() - t0:.0f}s", flush=True)
        return rows
    except Exception as exc:
        traceback.print_exc()
        return [{**dict.fromkeys(COLUMNS, ""), "case_id": cid, "status": "failed",
                 "error": f"{type(exc).__name__}: {exc}"[:300]}]  # fmt: skip


def done_cases(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {int(r["case_id"]) for r in csv.DictReader(f) if r["case_id"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--sigmas", default="0,0.5,1.0,1.5,2.0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    sigmas = [float(x) for x in a.sigmas.split(",")]
    out = resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    todo = [c for c in case_ids(a.cases) if c not in done_cases(out)]
    print(f"{len(todo)} cases to run", flush=True)
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        if new:
            w.writeheader()
        jobs = [(c, sigmas) for c in todo]
        it = map(audit_case, jobs) if a.workers <= 1 else Pool(a.workers).imap_unordered(
            audit_case, jobs)  # fmt: skip
        for rows in it:
            for r in rows:
                w.writerow({k: _fmt(v) for k, v in r.items()})
            f.flush()


if __name__ == "__main__":
    sys.exit(main())
