"""Lumen radius refinement from straightened cross-sections (SPEC_V2 §9.4)."""

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

from twinloop.centerline import VesselTree
from twinloop.config_v2 import CprParams
from twinloop.cpr import center_component, r_eq_mm, straighten

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
            out = model(inp)[0, :, 0].cpu().numpy()
            total[s : s + size] += out
            count[s : s + size] += 1.0
    return total / count[:, None, None]


def refine_radii(
    trees: list[VesselTree], image: sitk.Image, model, cp: CprParams, device: str
) -> list[str]:
    """Replace branch radii in place by CPR-based lumen radii; returns warnings."""
    warnings: list[str] = []
    for tree in trees:
        for b in tree.branches.values():
            if b.s_mm[-1] < MIN_BRANCH_MM:
                continue
            res = straighten(image, None, b, cp)
            x = ((np.clip(res["img"], *HU_CLIP) - HU_CENTER) / HU_SCALE).astype(np.float32)
            prob = 1.0 / (1.0 + np.exp(-_predict_logits(model, x, cp, device)))
            r_v1 = np.interp(res["s_mm"], b.s_mm, b.radius_mm)
            r = r_v1.copy()
            found = 0
            for k in range(len(r)):
                comp = center_component(prob[k] > 0.5)
                if comp.any():
                    r[k] = r_eq_mm(comp, cp.pixel_mm)
                    found += 1
            if found == 0:
                warnings.append(f"Lumen refinement found no lumen in {b.branch_id}; kept v1 radii")
                continue
            r = ndimage.gaussian_filter1d(r, sigma=1.0, mode="nearest")
            b.radius_mm = np.interp(b.s_mm, res["s_mm"], r)
    return warnings
