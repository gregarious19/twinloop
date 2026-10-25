"""Region proposals, geometry/intensity features and labels for the decider (SPEC_V2 §7)."""

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from twinloop.centerline import Branch, VesselTree
from twinloop.config_v2 import V2Params
from twinloop.cpr import index_zyx_to_phys, phys_to_index_zyx, rmf
from twinloop.graph import CoronaryGraph
from twinloop.lesions import detect_lesions, reference_radius

ACTIONS = ["keep", "extend", "add", "remove", "restore", "stop"]
FEATURE_NAMES = [
    "kind_ending", "kind_probe", "kind_narrowing", "kind_branch", "r_local_mm",
    "r_ref_mm", "ds", "taper_ratio", "is_abrupt", "branch_len_mm", "dist_from_root_mm", "depth",
    "n_children", "is_terminal", "in_fragment",
] + [f"hu_ahead_{k}" for k in range(1, 11)] + [
    "mask_ahead_frac", "hu_lumen", "hu_shell", "hu_branch_mean", "calcium_frac",
]  # fmt: skip
FEATURE_NAMES.append("probe_ray_hu")
KINDS = ("ending", "probe", "narrowing", "branch")
_COMPATIBLE = {
    "extend": "ending",
    "add": "probe",
    "restore": "narrowing",
    "remove": "branch",
    "stop": "ending",
}
_RAY_ANGLES = np.deg2rad(np.arange(8) * 45.0)


@dataclass
class Region:
    region_id: str  # "r0".. in proposal order
    kind: str  # "ending" | "probe" | "narrowing" | "branch"
    tree_id: str
    branch_id: str
    point_mm: np.ndarray
    direction: np.ndarray
    idx: int  # sample index on the branch


# ------------------------------------------------------------------ helpers


def sample_hu(image_arr_zyx: np.ndarray, image: sitk.Image, points_mm: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points_mm, float))
    idx = phys_to_index_zyx(pts, image)
    return ndimage.map_coordinates(image_arr_zyx, idx.T, order=1, mode="nearest", output=np.float64)


def _in_mask(mask_arr: np.ndarray, image: sitk.Image, points_mm: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points_mm, float))
    idx = np.rint(phys_to_index_zyx(pts, image)).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(mask_arr.shape)), axis=1)
    out = np.zeros(len(pts), dtype=bool)
    out[ok] = mask_arr[tuple(idx[ok].T)]
    return out


def _perp_basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = np.zeros(3)
    e[int(np.argmin(np.abs(d)))] = 1.0
    u = np.cross(d, e)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def _tangent(b: Branch, i: int) -> np.ndarray:
    lo, hi = max(i - 1, 0), min(i + 1, len(b.s_mm) - 1)
    d = b.points_mm[hi] - b.points_mm[lo]
    n = np.linalg.norm(d)
    return d / n if n > 0 else np.array([0.0, 0.0, 1.0])


def _all_trees(g: CoronaryGraph) -> list[VesselTree]:
    return [*g.trees, *g.fragments]


def _ray_points(p: np.ndarray, d: np.ndarray, r: float) -> np.ndarray:
    dist = np.arange(r + 1.0, r + 5.0 + 1e-9, 0.5)
    return p[None, :] + dist[:, None] * d[None, :]


# ------------------------------------------------------------------ proposals


def propose_regions(
    g: CoronaryGraph, image: sitk.Image, mask: sitk.Image, P: V2Params
) -> list[Region]:
    rp = P.region
    img_arr = sitk.GetArrayViewFromImage(image)
    mask_arr = sitk.GetArrayViewFromImage(mask) > 0
    trees = _all_trees(g)
    branches = {b.branch_id: (t, b) for t in trees for b in t.branches.values()}
    regions: list[Region] = []

    def add(kind, tree_id, branch_id, point, direction, idx):
        regions.append(
            Region(f"r{len(regions)}", kind, tree_id, branch_id, np.asarray(point, float),
                   np.asarray(direction, float), int(idx))
        )  # fmt: skip

    for e in g.endings:
        _, b = branches[e.branch_id]
        add("ending", e.tree_id, e.branch_id, e.point_mm, e.direction, len(b.s_mm) - 1)

    for t in trees:
        for b in t.branches.values():
            for les in detect_lesions(b, P.lesion):
                add("narrowing", t.tree_id, b.branch_id, b.points_mm[les.min_idx],
                    _tangent(b, les.min_idx), les.min_idx)  # fmt: skip

    for t in trees:
        for b in t.branches.values():
            if not b.children and b.s_mm[-1] >= rp.min_terminal_branch_len_mm:
                i = len(b.s_mm) // 2
                add("branch", t.tree_id, b.branch_id, b.points_mm[i], _tangent(b, i), i)

    starts = [b.points_mm[0] for t in trees for b in t.branches.values() if b.parent_id is not None]
    kept: list[np.ndarray] = []
    for t in trees:
        for b in t.branches.values():
            if np.median(b.radius_mm) < rp.probe_min_branch_radius_mm:
                continue
            length = float(b.s_mm[-1])
            _, U, V = rmf(b.points_mm)
            for s in np.arange(0.0, length, rp.probe_step_mm):
                if s < 2.0 or s > length - 2.0:
                    continue
                i = int(np.argmin(np.abs(b.s_mm - s)))
                p, r = b.points_mm[i], float(b.radius_mm[i])
                best, best_hu = None, -np.inf
                for a in _RAY_ANGLES[: rp.probe_rays]:
                    d = np.cos(a) * U[i] + np.sin(a) * V[i]
                    pts = _ray_points(p, d, r)
                    hu = float(sample_hu(img_arr, image, pts).mean())
                    if hu >= rp.probe_min_hu and not _in_mask(mask_arr, image, pts).any():
                        if hu > best_hu:
                            best, best_hu = d, hu
                if best is None:
                    continue
                if any(np.linalg.norm(p - q) < rp.probe_junction_exclusion_mm for q in starts):
                    continue
                if any(np.linalg.norm(p - q) < rp.probe_nms_mm for q in kept):
                    continue
                kept.append(p)
                add("probe", t.tree_id, b.branch_id, p, best, i)
    return regions


# ------------------------------------------------------------------ features


def region_features(
    regions: list[Region], g: CoronaryGraph, image: sitk.Image, mask: sitk.Image, P: V2Params
) -> np.ndarray:
    n = len(regions)
    out = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    if n == 0:
        return out
    col = {name: k for k, name in enumerate(FEATURE_NAMES)}
    img_arr = sitk.GetArrayViewFromImage(image)
    mask_arr = sitk.GetArrayViewFromImage(mask) > 0
    sp = np.array(image.GetSpacing())[::-1]  # (z, y, x)
    shape = np.array(mask_arr.shape)

    # distance to the mask, on the mask bounding box expanded by 10 mm
    nz = np.argwhere(mask_arr)
    if len(nz):
        pad = np.ceil(10.0 / sp).astype(int)
        lo = np.maximum(nz.min(0) - pad, 0)
        hi = np.minimum(nz.max(0) + pad + 1, shape)
        crop = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
        edt = ndimage.distance_transform_edt(~mask_arr[crop], sampling=sp)
    else:
        lo = hi = np.zeros(3, int)
        edt = np.zeros((0, 0, 0))

    trees = {t.tree_id: t for t in _all_trees(g)}
    fragment_ids = {t.tree_id for t in g.fragments}
    branches = {b.branch_id: b for t in trees.values() for b in t.branches.values()}
    endings = {e.branch_id: e for e in g.endings}
    ref_cache: dict[str, np.ndarray] = {}

    def ref_radius(b: Branch) -> np.ndarray:
        if b.branch_id not in ref_cache:
            ref_cache[b.branch_id] = reference_radius(b.s_mm, b.radius_mm, P.lesion)
        return ref_cache[b.branch_id]

    ring = np.deg2rad(np.arange(8) * 45.0)
    ks = np.arange(1, 11, dtype=float)
    block_r = int(np.ceil(3.0 / sp.min())) + 1

    for n_i, reg in enumerate(regions):
        row = out[n_i]
        b = branches[reg.branch_id]
        t = trees[reg.tree_id]
        p, d, i = reg.point_mm, reg.direction, reg.idx
        row[col[f"kind_{reg.kind}"]] = 0.0
        for k in KINDS:
            row[col[f"kind_{k}"]] = 1.0 if k == reg.kind else 0.0
        r_local = float(b.radius_mm[i])
        r_ref = float(ref_radius(b)[i])
        row[col["r_local_mm"]] = r_local
        row[col["r_ref_mm"]] = r_ref
        row[col["ds"]] = 1.0 - r_local / r_ref if r_ref > 0 else np.nan
        e = endings.get(reg.branch_id) if reg.kind == "ending" else None
        row[col["taper_ratio"]] = e.taper_ratio if e is not None else np.nan
        row[col["is_abrupt"]] = 1.0 if e is not None and e.kind == "abrupt" else 0.0
        row[col["branch_len_mm"]] = float(b.s_mm[-1])
        depth, dist_root, anc = 0, 0.0, b
        while anc.parent_id is not None:
            anc = t.branches[anc.parent_id]
            depth += 1
            dist_root += float(anc.s_mm[-1])
        row[col["dist_from_root_mm"]] = dist_root + float(b.s_mm[i])
        row[col["depth"]] = float(depth)
        row[col["n_children"]] = float(len(b.children))
        row[col["is_terminal"]] = 0.0 if b.children else 1.0
        row[col["in_fragment"]] = 1.0 if reg.tree_id in fragment_ids else 0.0

        u, v = _perp_basis(d)
        centres = p[None, :] + ks[:, None] * d[None, :]
        ring_off = np.cos(ring)[:, None] * u[None, :] + np.sin(ring)[:, None] * v[None, :]
        pts = np.concatenate([centres, (centres[:, None, :] + ring_off[None]).reshape(-1, 3)])
        hu = sample_hu(img_arr, image, pts)
        hu_c, hu_r = hu[:10], hu[10:].reshape(10, 8)
        for k in range(10):
            row[col[f"hu_ahead_{k + 1}"]] = (hu_c[k] + hu_r[k].sum()) / 9.0
        row[col["mask_ahead_frac"]] = float(_in_mask(mask_arr, image, centres).mean())

        # local block of voxels within 3 mm of the point
        c = phys_to_index_zyx(p[None], image)[0]
        blo = np.maximum(np.floor(c).astype(int) - block_r, 0)
        bhi = np.minimum(np.ceil(c).astype(int) + block_r + 1, shape)
        if np.all(bhi > blo):
            grid = np.stack(
                np.meshgrid(*[np.arange(blo[k], bhi[k]) for k in range(3)], indexing="ij"), -1
            ).reshape(-1, 3)
            dist = np.linalg.norm(index_zyx_to_phys(grid, image) - p, axis=1)
            near3 = dist <= 3.0
            gi, gd = grid[near3], dist[near3]
            if len(gi):
                inm = mask_arr[tuple(gi.T)]
                hu_v = img_arr[tuple(gi.T)].astype(np.float64)
                row[col["calcium_frac"]] = float((hu_v >= 800.0).mean())
                sel = inm & (gd <= 2.0)
                if sel.any():
                    row[col["hu_lumen"]] = float(hu_v[sel].mean())
                if len(edt):
                    rel = gi - lo
                    ok = np.all((rel >= 0) & (rel < np.array(edt.shape)), axis=1)
                    e_v = np.full(len(gi), np.inf)
                    e_v[ok] = edt[tuple(rel[ok].T)]
                    shell = ~inm & (e_v >= 1.0) & (e_v <= 2.0)
                    if shell.any():
                        row[col["hu_shell"]] = float(hu_v[shell].mean())
        if reg.kind == "branch":
            row[col["hu_branch_mean"]] = float(sample_hu(img_arr, image, b.points_mm).mean())
        if reg.kind == "probe":
            row[col["probe_ray_hu"]] = float(
                sample_hu(img_arr, image, _ray_points(p, d, r_local)).mean()
            )
    return out


# ------------------------------------------------------------------ labels


def match_records(regions: list[Region], records) -> list[int | None]:
    """For each record the index of the matched region (None if unmatched)."""
    taken: set[int] = set()
    matches: list[int | None] = []
    for rec in records:
        kind = _COMPATIBLE[rec.action]
        best, best_d = None, 4.0
        for j, reg in enumerate(regions):
            if reg.kind != kind:
                continue
            dd = float(np.linalg.norm(reg.point_mm - np.asarray(rec.point_mm)))
            if dd <= best_d:
                best, best_d = j, dd
        if best is None or best in taken:
            matches.append(None)
        else:
            taken.add(best)
            matches.append(best)
    return matches


def label_synthetic(regions: list[Region], records, P: V2Params) -> tuple[list[str], int]:
    labels = ["keep"] * len(regions)
    matches = match_records(regions, records)
    for rec, j in zip(records, matches, strict=True):
        if j is not None:
            labels[j] = rec.action
    return labels, sum(m is None for m in matches)


def label_real(
    regions: list[Region],
    g_pred: CoronaryGraph,
    pred_mask: sitk.Image,
    gt_mask: sitk.Image,
    g_gt: CoronaryGraph,
    P: V2Params,
) -> list[str]:
    struct = np.ones((3, 3, 3), dtype=bool)
    pred_dil = ndimage.binary_dilation(sitk.GetArrayViewFromImage(pred_mask) > 0, structure=struct)
    gt_dil = ndimage.binary_dilation(sitk.GetArrayViewFromImage(gt_mask) > 0, structure=struct)
    gt_branches = [b for t in _all_trees(g_gt) for b in t.branches.values()]
    q = np.concatenate(
        [b.points_mm[b.radius_mm >= 0.75] for b in gt_branches] or [np.zeros((0, 3))]
    )
    q_ds = np.concatenate(
        [
            1.0
            - b.radius_mm[b.radius_mm >= 0.75]
            / reference_radius(b.s_mm, b.radius_mm, P.lesion)[b.radius_mm >= 0.75]
            for b in gt_branches
        ]
        or [np.zeros(0)]
    )
    outside = ~_in_mask(pred_dil, pred_mask, q) if len(q) else np.zeros(0, bool)
    pred_branches = {b.branch_id: b for t in _all_trees(g_pred) for b in t.branches.values()}
    cos35 = np.cos(np.deg2rad(35.0))
    labels = []
    for reg in regions:
        p, d = reg.point_mm, reg.direction
        lab = "keep"
        if reg.kind == "ending" and len(q):
            rel = q - p
            along = rel @ d
            norm = np.linalg.norm(rel, axis=1)
            ahead = (along > 0) & (along <= 10.0) & (along >= cos35 * norm)
            if int((ahead & outside).sum()) >= 4:
                lab = "extend"
        elif reg.kind == "probe" and len(q):
            near = np.linalg.norm(q - (p + 3.0 * d), axis=1) <= 6.0
            if int((near & outside).sum()) >= 4:
                lab = "add"
        elif reg.kind == "branch":
            b = pred_branches[reg.branch_id]
            if _in_mask(gt_dil, gt_mask, b.points_mm).mean() < 0.2:
                lab = "remove"
        elif reg.kind == "narrowing" and len(q):
            dd = np.linalg.norm(q - p, axis=1)
            j = int(np.argmin(dd))
            if dd[j] <= 2.0 and q_ds[j] < 0.15:
                lab = "restore"
        labels.append(lab)
    return labels
