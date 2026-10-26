"""Lumen radius refinement from straightened cross-sections (SPEC_V2 §9.4)."""

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

from twinloop.centerline import VesselTree
from twinloop.config_v2 import CprParams
from twinloop.cpr import center_component, r_eq_mm, r_ins_2d_mm, straighten

MIN_BRANCH_MM = 10.0
HU_CLIP = (-200.0, 800.0)
HU_CENTER, HU_SCALE = 300.0, 500.0


def _window_starts(n: int, size: int, stride: int) -> list[int]:
    if n <= size:
        return [0]
    starts = list(range(0, n - size, stride))
    if starts[-1] + size < n:
        starts.append(n - size)
    return starts


def _predict_logits(model, x: np.ndarray, cp: CprParams, device: str) -> np.ndarray:
    """x (T, 48, 48) normalised -> logits (T, 48, 48), overlapping windows averaged."""
    n = len(x)
    size = min(cp.window_slices, n)
    total = np.zeros(x.shape, dtype=np.float64)
    count = np.zeros(n, dtype=np.float64)
    with torch.no_grad():
        for s in _window_starts(n, cp.window_slices, cp.window_stride):
            inp = torch.from_numpy(x[s : s + size])[None, :, None].float().to(device)
            out = model(inp)[0, :, 0].float().cpu().numpy()
            total[s : s + size] += out
            count[s : s + size] += 1.0
    return total / count[:, None, None]


NEAREST_MAX_MM = 1.0  # fallback="nearest": accept an off-centre component within this distance


def slice_radii(
    prob: np.ndarray, r_v1: np.ndarray, cp: CprParams, radius_out: str, fallback: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-slice radius from probabilities (T, 48, 48).

    Returns (r, centre_hit, used_v1): ``centre_hit`` = the centre pixel lies in a predicted
    component; ``used_v1`` = the slice kept the v1 radius. fallback "v1": any component counts
    (the centre one, else the nearest one), v1 only if the slice is empty; "nearest": off-centre
    components count only within NEAREST_MAX_MM of the centre; "interp": only centre components
    count, other slices are interpolated linearly between the nearest slices that had one (v1
    beyond the first/last such slice).
    """

    def size(comp: np.ndarray) -> float:
        if radius_out == "eq":
            return r_eq_mm(comp, cp.pixel_mm)
        return r_ins_2d_mm(comp, cp.pixel_mm)

    n = len(prob)
    r = r_v1.copy()
    centre_hit = np.zeros(n, dtype=bool)
    used_v1 = np.ones(n, dtype=bool)
    c = prob.shape[1] // 2
    for k in range(n):
        comp = center_component(prob[k] > 0.5)
        if not comp.any():
            continue
        centre_hit[k] = bool(comp[c, c])
        if fallback == "interp" and not centre_hit[k]:
            continue
        if fallback == "nearest" and not centre_hit[k]:
            yy, xx = np.nonzero(comp)
            if np.sqrt(np.min((yy - c) ** 2 + (xx - c) ** 2)) * cp.pixel_mm > NEAREST_MAX_MM:
                continue
        r[k] = size(comp)
        used_v1[k] = False
    if fallback == "interp" and centre_hit.any():
        idx = np.nonzero(centre_hit)[0]
        inside = (np.arange(n) >= idx[0]) & (np.arange(n) <= idx[-1]) & ~centre_hit
        r[inside] = np.interp(np.nonzero(inside)[0], idx, r[idx])
        used_v1[inside] = False
    return r, centre_hit, used_v1


def refine_radii(
    trees: list[VesselTree],
    image: sitk.Image,
    model,
    cp: CprParams,
    device: str,
    *,
    radius_out: str = "eq",
    fallback: str = "v1",
    sigma: float = 1.0,
    stats: dict | None = None,
) -> list[str]:
    """Replace branch radii in place by CPR-based lumen radii; returns warnings.

    ``radius_out``: "eq" (area-equivalent, as trained) or "ins" (2D inscribed radius).
    ``fallback``/``sigma``: see ``slice_radii``; ``sigma`` is the Gaussian width in samples
    (0 = no smoothing). The defaults reproduce the original behaviour exactly. If ``stats`` is
    given it receives ``n_slices``, ``n_centre_miss`` (slices whose centre pixel is not in a
    predicted component) and ``n_v1`` (slices that kept the v1 radius) plus ``fallback_fraction``
    = n_centre_miss / n_slices.
    """
    if radius_out not in ("eq", "ins") or fallback not in ("v1", "nearest", "interp"):
        raise ValueError(f"bad radius_out/fallback: {radius_out!r}/{fallback!r}")
    warnings: list[str] = []
    tot = {"n_slices": 0, "n_centre_miss": 0, "n_v1": 0}
    for tree in trees:
        for b in tree.branches.values():
            if b.s_mm[-1] < MIN_BRANCH_MM:
                continue
            res = straighten(image, None, b, cp)
            x = ((np.clip(res["img"], *HU_CLIP) - HU_CENTER) / HU_SCALE).astype(np.float32)
            prob = 1.0 / (1.0 + np.exp(-_predict_logits(model, x, cp, device)))
            r_v1 = np.interp(res["s_mm"], b.s_mm, b.radius_mm)
            r, hit, used_v1 = slice_radii(prob, r_v1, cp, radius_out, fallback)
            found = int((~used_v1).sum())
            if found == 0:
                warnings.append(f"Lumen refinement found no lumen in {b.branch_id}; kept v1 radii")
                continue
            tot["n_slices"] += len(r)
            tot["n_centre_miss"] += int((~hit).sum())
            tot["n_v1"] += int(used_v1.sum())
            if sigma > 0:
                r = ndimage.gaussian_filter1d(r, sigma=sigma, mode="nearest")
            b.radius_mm = np.interp(b.s_mm, res["s_mm"], r)
    if stats is not None:
        stats.update(tot)
        stats["fallback_fraction"] = tot["n_centre_miss"] / max(1, tot["n_slices"])
    return warnings
