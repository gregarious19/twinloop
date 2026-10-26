"""Evaluation metrics for masks, graphs and stenoses (SPEC_V2 §13).

Masks are numpy bool arrays in (z, y, x) order, or ``sitk.Image``. The branch-based metrics
map physical points to voxels, so they need the grid: pass a ``sitk.Image`` mask or ``ref=``.
"""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.morphology import skeletonize

from twinloop.config import LesionParams
from twinloop.graph import CoronaryGraph
from twinloop.lesions import detect_lesions, reference_radius

_STRUCT = np.ones((3, 3, 3), dtype=bool)
MIN_RADIUS_MM = 0.75
MIN_LENGTH_MM = 10.0


def _arr(m) -> np.ndarray:
    if isinstance(m, sitk.Image):
        return sitk.GetArrayViewFromImage(m) > 0
    return np.asarray(m, dtype=bool)


def dice(p, g) -> float:
    p, g = _arr(p), _arr(g)
    denom = int(p.sum()) + int(g.sum())
    if denom == 0:
        return 1.0
    return 2.0 * int((p & g).sum()) / denom


def cl_dice(p, g) -> float:
    p, g = _arr(p), _arr(g)
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    sp, sg = skeletonize(p) > 0, skeletonize(g) > 0
    t_prec = (sp & g).sum() / max(int(sp.sum()), 1)
    t_sens = (sg & p).sum() / max(int(sg.sum()), 1)
    if t_prec + t_sens == 0:
        return 0.0
    return float(2.0 * t_prec * t_sens / (t_prec + t_sens))


def _points_in(points_mm: np.ndarray, mask: np.ndarray, ref: sitk.Image) -> np.ndarray:
    d = np.array(ref.GetDirection()).reshape(3, 3)
    idx = np.rint(((points_mm - np.array(ref.GetOrigin())) @ d) / np.array(ref.GetSpacing()))
    idx = idx.astype(int)[:, ::-1]  # (z, y, x)
    ok = np.all((idx >= 0) & (idx < np.array(mask.shape)), axis=1)
    out = np.zeros(len(points_mm), dtype=bool)
    out[ok] = mask[tuple(idx[ok].T)]
    return out


def _filtered_branches(g: CoronaryGraph):
    for t in [*g.trees, *g.fragments]:
        for b in t.branches.values():
            if np.median(b.radius_mm) >= MIN_RADIUS_MM and b.s_mm[-1] >= MIN_LENGTH_MM:
                yield b


def _frac_inside(g: CoronaryGraph, mask, ref) -> dict[str, float]:
    ref = ref if ref is not None else mask
    if not isinstance(ref, sitk.Image):
        raise ValueError("a sitk.Image mask or ref= is needed to map points to voxels")
    dil = ndimage.binary_dilation(_arr(mask), structure=_STRUCT)
    return {
        b.branch_id: float(_points_in(b.points_mm, dil, ref).mean()) for b in _filtered_branches(g)
    }


def branch_recall(g_gt: CoronaryGraph, pred_mask, ref: sitk.Image | None = None) -> float:
    """Fraction of GT branches (median r >= 0.75 mm, length >= 10 mm) covered >= 80 % by pred."""
    fr = _frac_inside(g_gt, pred_mask, ref)
    return 1.0 if not fr else sum(v >= 0.8 for v in fr.values()) / len(fr)


def false_branches(g_pred: CoronaryGraph, gt_mask, ref: sitk.Image | None = None) -> list[str]:
    """Ids of pred branches (same filter) with < 20 % of their points inside the GT mask."""
    return [b for b, v in _frac_inside(g_pred, gt_mask, ref).items() if v < 0.2]


def false_branch_rate(g_pred: CoronaryGraph, gt_mask, ref: sitk.Image | None = None) -> float:
    n = sum(1 for _ in _filtered_branches(g_pred))
    return 0.0 if n == 0 else len(false_branches(g_pred, gt_mask, ref)) / n


def stenosis_metrics(g_gt: CoronaryGraph, g_pred: CoronaryGraph, lp: LesionParams) -> dict:
    """GT lesions vs the prediction at their narrowest point (nearest pred point within 2 mm)."""
    pred = [b for t in [*g_pred.trees, *g_pred.fragments] for b in t.branches.values()]
    pts = np.concatenate([b.points_mm for b in pred]) if pred else np.zeros((0, 3))
    owner = [(b, i) for b in pred for i in range(len(b.s_mm))]
    refs = {b.branch_id: reference_radius(b.s_mm, b.radius_mm, lp) for b in pred}
    errs: list[float] = []
    n = widened = 0
    for t in g_gt.trees:
        for b in t.branches.values():
            for les in detect_lesions(b, lp):
                n += 1
                p = b.points_mm[les.min_idx]
                if len(pts) == 0:
                    widened += 1
                    continue
                d = np.linalg.norm(pts - p, axis=1)
                j = int(np.argmin(d))
                if d[j] > 2.0:
                    widened += 1
                    continue
                pb, i = owner[j]
                r = float(pb.radius_mm[i])
                errs.append(abs(r - les.r_min_mm))
                ref_r = float(refs[pb.branch_id][i])
                ds_pred = max(0.0, 1.0 - r / ref_r) if ref_r > 0 else 0.0
                if ds_pred < 0.5 * les.ds:
                    widened += 1
    return {
        "n_lesions": n,
        "mean_abs_rmin_err_mm": float(np.mean(errs)) if errs else float("nan"),
        "widened_rate": widened / n if n else float("nan"),
    }


STENOSIS_BINS = (("mild", 0.25, 0.5), ("moderate", 0.5, 0.7), ("severe", 0.7, 1.0 + 1e-9))
MATCH_MM = 2.0
WINDOW_MM = 2.0


def _bin_of(ds: float) -> str:
    for name, lo, hi in STENOSIS_BINS:
        if lo <= ds < hi:
            return name
    return "sub"


def _window_ds(branch, ref: np.ndarray, i: int) -> tuple[float, float]:
    """(max ds, radius there) over the points within +-WINDOW_MM arc length of point i."""
    win = np.nonzero(np.abs(branch.s_mm - branch.s_mm[i]) <= WINDOW_MM)[0]
    ds_w = 1.0 - branch.radius_mm[win] / ref[win]
    k = int(np.argmax(ds_w))
    return float(ds_w[k]), float(branch.radius_mm[win[k]])


def stenosis_errors(g_gt: CoronaryGraph, g_pred: CoronaryGraph, lp: LesionParams) -> list[dict]:
    """One row per GT lesion: severity error of the prediction near its narrowest point (§13b)."""
    pred = [b for t in [*g_pred.trees, *g_pred.fragments] for b in t.branches.values()]
    pts = np.concatenate([b.points_mm for b in pred]) if pred else np.zeros((0, 3))
    owner = [(b, i) for b in pred for i in range(len(b.s_mm))]
    refs = {b.branch_id: reference_radius(b.s_mm, b.radius_mm, lp) for b in pred}
    nan = float("nan")
    rows = []
    for t in g_gt.trees:
        for b in t.branches.values():
            ref_gt = reference_radius(b.s_mm, b.radius_mm, lp)
            for les in detect_lesions(b, lp):
                ds_gt, rmin_gt = _window_ds(b, ref_gt, les.min_idx)  # same rule as the prediction
                row = {
                    "case_id": "", "lesion_id": f"{b.branch_id}_p{les.min_idx}", "ds_gt": ds_gt,
                    "ds_pred": nan, "ds_err_pp": nan, "rmin_gt": rmin_gt, "rmin_pred": nan,
                    "rmin_err_mm": nan, "missed": True, "bin": _bin_of(ds_gt),
                    "ds_lesion": les.ds, "rmin_lesion": les.r_min_mm,
                }  # fmt: skip
                if len(pts):
                    d = np.linalg.norm(pts - b.points_mm[les.min_idx], axis=1)
                    j = int(np.argmin(d))
                    if d[j] <= MATCH_MM:
                        pb, i = owner[j]
                        ds_pred, r_pred = _window_ds(pb, refs[pb.branch_id], i)
                        row.update(
                            ds_pred=ds_pred, ds_err_pp=100.0 * (ds_pred - ds_gt),
                            rmin_pred=r_pred, rmin_err_mm=r_pred - rmin_gt, missed=False,
                        )  # fmt: skip
                rows.append(row)
    return rows


def summarize_stenosis(rows: list[dict]) -> dict:
    """Per severity bin error statistics plus the down50 / down70 counts (SPEC_V2 §13b)."""

    def stats(sel: list[dict]) -> dict:
        m = [r for r in sel if not r["missed"]]
        e = np.array([r["ds_err_pp"] for r in m], dtype=float)
        x = np.array([r["rmin_err_mm"] for r in m], dtype=float)

        def f(fn, a):
            return float(fn(a)) if len(a) else float("nan")

        return {
            "n": len(sel), "n_missed": len(sel) - len(m),
            "mean_ds_err_pp": f(np.mean, e), "median_ds_err_pp": f(np.median, e),
            "mean_abs_ds_err_pp": f(np.mean, np.abs(e)),
            "mean_rmin_err_mm": f(np.mean, x), "mean_abs_rmin_err_mm": f(np.mean, np.abs(x)),
        }  # fmt: skip

    out: dict = {"bins": {n: stats([r for r in rows if r["bin"] == n]) for n in
                          ("mild", "moderate", "severe", "sub")}}  # fmt: skip
    out["moderate_severe"] = stats([r for r in rows if r["bin"] in ("moderate", "severe")])
    for thr in (50, 70):
        sev = [r for r in rows if r["ds_gt"] >= thr / 100.0]
        out[f"down{thr}"] = sum(1 for r in sev if not r["missed"] and r["ds_pred"] < thr / 100.0)
        out[f"down{thr}_denom"] = len(sev)
    return out
