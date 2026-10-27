"""Radius-profile smoothing and mask blur (SPEC_V3 §4.1, §4.2)."""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from twinloop.centerline import VesselTree
from twinloop.graph import CoronaryGraph


def blur_mask(mask: sitk.Image, sigma_mm: float) -> sitk.Image:
    a = (sitk.GetArrayFromImage(mask) > 0).astype(np.float32)
    sp = mask.GetSpacing()  # (x, y, z)
    b = ndimage.gaussian_filter(a, sigma=(sigma_mm / sp[2], sigma_mm / sp[1], sigma_mm / sp[0]))
    out = sitk.GetImageFromArray((b > 0.5).astype(np.uint8))
    out.CopyInformation(mask)
    return out


def smooth_radius(s_mm: np.ndarray, radius_mm: np.ndarray, sigma_mm: float) -> np.ndarray:
    r = np.asarray(radius_mm, dtype=np.float64)
    if sigma_mm <= 0:
        return r.copy()
    s = np.asarray(s_mm, dtype=np.float64)
    d = s[:, None] - s[None, :]
    w = np.exp(-0.5 * (d / sigma_mm) ** 2) * (np.abs(d) <= 3 * sigma_mm + 1e-9)
    return (w @ r) / w.sum(axis=1)


def smooth_trees(trees: list[VesselTree], sigma_mm: float) -> None:
    if sigma_mm <= 0:
        return
    for tree in trees:
        for b in tree.branches.values():
            b.radius_mm = smooth_radius(b.s_mm, b.radius_mm, sigma_mm)


def smooth_graph(graph: CoronaryGraph, sigma_mm: float) -> None:
    smooth_trees(graph.trees, sigma_mm)
    smooth_trees(graph.fragments, sigma_mm)
