"""Damage generator: corrupt ground-truth masks to make repair training data (SPEC_V2 §6)."""

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree

from twinloop.centerline import Branch, VesselTree
from twinloop.config_v2 import DamageParams
from twinloop.cpr import index_zyx_to_phys, rmf
from twinloop.graph import CoronaryGraph

TYPES = ("gap", "truncate", "delete_branch", "shrink", "false_branch", "occlusion")
ACTIONS = ("extend", "extend", "add", "restore", "remove", "stop")
MIN_SEED_DISTANCE_MM = 10.0
MAX_TRIES = 50
MIN_FALSE_VOXELS = 50


@dataclass
class DamageRecord:
    damage_id: str  # "d0".."d5"
    type: str  # gap|truncate|delete_branch|shrink|false_branch|occlusion
    action: str  # extend|extend|add|restore|remove|stop
    branch_id: str  # GT branch that was damaged (false_branch: the branch the tube attaches to)
    point_mm: list[float]  # seed point
    direction: list[float]  # unit vector
    length_mm: float
    voxels_changed: int


def _point_at(b: Branch, s: float) -> np.ndarray:
    return np.array([np.interp(s, b.s_mm, b.points_mm[:, k]) for k in range(3)])


def _tangent_at(b: Branch, s: float) -> np.ndarray:
    lo, hi = max(s - 1.0, 0.0), min(s + 1.0, float(b.s_mm[-1]))
    d = _point_at(b, hi) - _point_at(b, lo)
    n = np.linalg.norm(d)
    return d / n if n > 0 else np.array([0.0, 0.0, 1.0])


def _descendants(tree: VesselTree, bid: str) -> list[str]:
    out, stack = [], list(tree.branches[bid].children)
    while stack:
        c = stack.pop()
        out.append(c)
        stack.extend(tree.branches[c].children)
    return out


def _voxels_near_segment(img: sitk.Image, a: np.ndarray, b: np.ndarray, radius: float):
    """(z, y, x) index array of voxels whose centre is within `radius` of segment a-b."""
    size = np.array(img.GetSize())  # (x, y, z)
    corners = np.array(
        [img.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in p)) for p in (a, b)]
    )
    pad = np.ceil(radius / np.array(img.GetSpacing())).astype(int) + 2
    lo = np.maximum(np.floor(corners.min(0)).astype(int) - pad, 0)
    hi = np.minimum(np.ceil(corners.max(0)).astype(int) + pad, size - 1)
    if np.any(hi < lo):
        return np.zeros((0, 3), int)
    grid = np.stack(
        np.meshgrid(*[np.arange(lo[k], hi[k] + 1) for k in (2, 1, 0)], indexing="ij"), axis=-1
    ).reshape(-1, 3)  # (z, y, x)
    p = index_zyx_to_phys(grid, img)
    ab = b - a
    t = np.clip(((p - a) @ ab) / float(ab @ ab), 0.0, 1.0)
    dist = np.linalg.norm(p - (a + t[:, None] * ab), axis=1)
    return grid[dist <= radius]


def damage_case(
    image: sitk.Image,
    gt: sitk.Image,
    g: CoronaryGraph,
    dp: DamageParams,
    rng: np.random.Generator,
) -> tuple[sitk.Image, sitk.Image, list[DamageRecord]]:
    """Returns (image_out, mask_out, records); image_out differs from image only for occlusion."""
    mask = sitk.GetArrayFromImage(gt) > 0
    hu = sitk.GetArrayFromImage(image)
    hu_out = hu.copy()

    trees = [*g.trees, *g.fragments]
    branches: list[tuple[VesselTree, Branch]] = [(t, b) for t in trees for b in t.branches.values()]
    index_of = {b.branch_id: i for i, (_, b) in enumerate(branches)}
    pts = np.concatenate([b.points_mm for _, b in branches])
    owner = np.concatenate([np.full(len(b.s_mm), i) for i, (_, b) in enumerate(branches)])
    s_all = np.concatenate([b.s_mm for _, b in branches])
    r_all = np.concatenate([b.radius_mm for _, b in branches])

    vox = np.argwhere(mask)  # (N, 3) z, y, x of the original GT voxels
    dist, nn = cKDTree(pts).query(index_zyx_to_phys(vox, gt), workers=-1)
    v_owner, v_s = owner[nn], s_all[nn]
    v_near = dist <= 1.5 * r_all[nn] + 0.5

    def eligible(delete: bool) -> list[int]:
        out = []
        for i, (t, b) in enumerate(branches):
            if delete:
                ok = b.branch_id != t.root_branch_id and b.s_mm[-1] >= 10.0
            else:
                ok = b.s_mm[-1] >= dp.min_branch_len_mm
            if ok and np.median(b.radius_mm) >= dp.min_branch_radius_mm:
                out.append(i)
        return out

    def interval(bi: int, s0: float, s1: float) -> np.ndarray:
        return (v_owner == bi) & (v_s >= s0) & (v_s <= s1) & v_near

    def subtree(bi: int) -> np.ndarray:
        t, b = branches[bi]
        ids = {index_of[b.branch_id]} | {index_of[c] for c in _descendants(t, b.branch_id)}
        return np.isin(v_owner, list(ids)) & v_near

    def try_damage(kind: str):
        """One try: returns (record fields, removal selection or None) or None if rejected."""
        pool = eligible(kind == "delete_branch")
        if not pool:
            return None
        bi = pool[int(rng.integers(len(pool)))]
        _, b = branches[bi]
        length = float(b.s_mm[-1])
        if kind == "gap":
            ln = rng.uniform(*dp.gap_len_mm)
            if length - 5.0 - ln < 5.0:
                return None
            s = rng.uniform(5.0, length - 5.0 - ln)
            sel = interval(bi, s, s + ln)
            return b, _point_at(b, s), _tangent_at(b, s), ln, sel
        if kind in ("truncate", "occlusion"):
            s = rng.uniform(0.3 * length, 0.8 * length)
            t_, _ = branches[bi]
            sel = interval(bi, s, length)
            for c in _descendants(t_, b.branch_id):
                sel |= (v_owner == index_of[c]) & v_near
            return b, _point_at(b, s), _tangent_at(b, s), length - s, sel
        if kind == "delete_branch":
            t_, _ = branches[bi]
            sel = interval(bi, 1.0, length)
            for c in _descendants(t_, b.branch_id):
                sel |= (v_owner == index_of[c]) & v_near
            d = _point_at(b, min(2.0, length)) - b.points_mm[0]
            return b, b.points_mm[0], d / np.linalg.norm(d), length, sel
        if kind == "shrink":
            ln = rng.uniform(*dp.shrink_len_mm)
            if length - 5.0 - ln < 5.0:
                return None
            s = rng.uniform(5.0, length - 5.0 - ln)
            f = rng.uniform(*dp.shrink_factor)
            inside = interval(bi, s, s + ln)
            keep_r = f * r_all[nn]
            sel = inside & (dist > keep_r)
            mid = s + ln / 2.0
            return b, _point_at(b, mid), _tangent_at(b, mid), ln, sel
        # false_branch: handled by the caller (adds voxels)
        s = rng.uniform(5.0, length - 5.0)
        theta = np.deg2rad(rng.uniform(60.0, 120.0))
        phi = np.deg2rad(rng.uniform(0.0, 360.0))
        ln = rng.uniform(*dp.false_len_mm)
        radius = rng.uniform(*dp.false_radius_mm)
        T, U, V = rmf(b.points_mm)
        k = int(np.argmin(np.abs(b.s_mm - s)))
        d = np.cos(theta) * T[k] + np.sin(theta) * (np.cos(phi) * U[k] + np.sin(phi) * V[k])
        d /= np.linalg.norm(d)
        p0 = _point_at(b, s)
        cand = _voxels_near_segment(gt, p0, p0 + ln * d, radius)
        if len(cand):
            free = ~mask[tuple(cand.T)] & (hu[tuple(cand.T)] <= dp.false_max_hu)
            cand = cand[free]
        if len(cand) < MIN_FALSE_VOXELS:
            return None
        return b, p0 + 0.5 * ln * d, d, ln, cand

    records: list[DamageRecord] = []
    for _ in range(dp.n_damages_per_case):
        k = int(rng.choice(len(TYPES), p=np.asarray(dp.type_weights)))
        kind = TYPES[k]
        for _try in range(MAX_TRIES):
            res = try_damage(kind)
            if res is None:
                continue
            b, seed, direction, ln, sel = res
            if any(
                np.linalg.norm(seed - np.array(r.point_mm)) < MIN_SEED_DISTANCE_MM for r in records
            ):
                continue
            if kind == "false_branch":
                changed = sel
                mask[tuple(changed.T)] = True
            else:
                changed = vox[sel & mask[tuple(vox.T)]]
                if len(changed) == 0:
                    continue
                mask[tuple(changed.T)] = False
                if kind == "occlusion":
                    new = rng.normal(dp.occlusion_hu_mean, dp.occlusion_hu_sd, len(changed))
                    hu_out[tuple(changed.T)] = (
                        np.rint(new) if np.issubdtype(hu_out.dtype, np.integer) else new
                    ).astype(hu_out.dtype)
            records.append(
                DamageRecord(
                    damage_id=f"d{len(records)}",
                    type=kind,
                    action=ACTIONS[k],
                    branch_id=b.branch_id,
                    point_mm=[float(v) for v in seed],
                    direction=[float(v) for v in direction],
                    length_mm=float(ln),
                    voxels_changed=int(len(changed)),
                )
            )
            break

    mask_out = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask_out.CopyInformation(gt)
    image_out = sitk.GetImageFromArray(hu_out)
    image_out.CopyInformation(image)
    return image_out, mask_out, records
