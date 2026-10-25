"""Curved planar reformation: straighten a vessel into 48x48 cross-sections (SPEC_V2 §9)."""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from twinloop.centerline import Branch
from twinloop.config_v2 import CprParams

_EIGHT = np.ones((3, 3), dtype=int)


def phys_to_index_zyx(points_mm: np.ndarray, img: sitk.Image) -> np.ndarray:
    """Continuous voxel indices (z, y, x) of physical points, shape (N, 3)."""
    d = np.array(img.GetDirection()).reshape(3, 3)
    m = d @ np.diag(img.GetSpacing())
    idx_xyz = np.linalg.solve(m, (np.asarray(points_mm, float) - np.array(img.GetOrigin())).T).T
    return idx_xyz[:, ::-1]


def rmf(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation-minimising frame by double reflection (Wang et al. 2008): T, U, V (N, 3)."""
    x = np.asarray(points, dtype=float)
    n = len(x)
    t = np.zeros_like(x)
    if n > 1:
        t[1:-1] = x[2:] - x[:-2]
        t[0], t[-1] = x[1] - x[0], x[-1] - x[-2]
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    e = np.zeros(3)
    e[int(np.argmin(np.abs(t[0])))] = 1.0
    u0 = np.cross(t[0], e)
    r = np.zeros_like(x)
    r[0] = u0 / np.linalg.norm(u0)
    for i in range(n - 1):
        v1 = x[i + 1] - x[i]
        c1 = float(v1 @ v1)
        if c1 < 1e-12:
            r[i + 1] = r[i]
            continue
        r_l = r[i] - (2.0 / c1) * (v1 @ r[i]) * v1
        t_l = t[i] - (2.0 / c1) * (v1 @ t[i]) * v1
        v2 = t[i + 1] - t_l
        c2 = float(v2 @ v2)
        r[i + 1] = r_l if c2 < 1e-12 else r_l - (2.0 / c2) * (v2 @ r_l) * v2
    return t, r, np.cross(t, r)


def straighten_points(
    image: sitk.Image, mask_or_none, points_mm: np.ndarray, cp: CprParams
) -> dict:
    pts = np.asarray(points_mm, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s_k = np.arange(0, int(np.floor(s[-1] / cp.step_mm)) + 1) * cp.step_mm
    x_k = np.stack([np.interp(s_k, s, pts[:, a]) for a in range(3)], axis=1)
    _, u, v = rmf(x_k)
    off = (np.arange(cp.size_px) - (cp.size_px - 1) / 2.0) * cp.pixel_mm
    p = (
        x_k[:, None, None, :]
        + off[None, :, None, None] * u[:, None, None, :]
        + off[None, None, :, None] * v[:, None, None, :]
    )
    idx = phys_to_index_zyx(p.reshape(-1, 3), image).T
    shape = (len(s_k), cp.size_px, cp.size_px)
    img = ndimage.map_coordinates(
        sitk.GetArrayViewFromImage(image).astype(np.float32, copy=False),
        idx,
        order=1,
        mode="nearest",
    ).reshape(shape)
    mask = None
    if mask_or_none is not None:
        mask = ndimage.map_coordinates(
            sitk.GetArrayViewFromImage(mask_or_none), idx, order=0, mode="nearest"
        ).reshape(shape)
        mask = (mask > 0).astype(np.uint8)
    return {"img": img.astype(np.float32), "mask": mask, "s_mm": s_k, "points_mm": x_k}


def straighten(image: sitk.Image, mask_or_none, branch: Branch, cp: CprParams) -> dict:
    return straighten_points(image, mask_or_none, branch.points_mm, cp)


def center_component(mask_slice: np.ndarray) -> np.ndarray:
    """8-connected component containing the centre pixel (24, 24), else the nearest one."""
    m = np.asarray(mask_slice) > 0
    lab, n = ndimage.label(m, structure=_EIGHT)
    if n == 0:
        return np.zeros_like(m)
    c = (m.shape[0] // 2, m.shape[1] // 2)
    if lab[c] > 0:
        return lab == lab[c]
    yy, xx = np.nonzero(m)
    j = int(np.argmin((yy - c[0]) ** 2 + (xx - c[1]) ** 2))
    return lab == lab[yy[j], xx[j]]


def r_eq_mm(lumen_slice: np.ndarray, pixel_mm: float = 0.2) -> float:
    return float(np.sqrt(np.count_nonzero(lumen_slice) * pixel_mm**2 / np.pi))
