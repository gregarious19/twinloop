"""TotalSegmentator anatomy: aorta, myocardium, LV cavity and quality checks (SPEC_V2 §5)."""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from twinloop.centerline import Branch, VesselTree
from twinloop.config_v2 import AnatomyParams, GraphParams
from twinloop.graph import (
    CoronaryGraph,
    branch_length_mm,
    end_direction,
    main_path,
)
from twinloop.io import save_nifti

HIGHRES = "heartchambers_highres"
_STRUCTURES = ("aorta", "myocardium", "lv_cavity")


@dataclass
class Anatomy:
    aorta: sitk.Image | None
    myocardium: sitk.Image | None
    lv_cavity: sitk.Image | None
    source: dict[str, str] = field(default_factory=dict)
    qc: dict[str, bool] = field(default_factory=dict)
    qc_messages: list[str] = field(default_factory=list)
    myo_volume_ml: float | None = None
    aorta_volume_ml: float | None = None


def _volume_ml(mask: sitk.Image) -> float:
    n = int(sitk.GetArrayViewFromImage(mask).astype(bool).sum())
    return n * float(np.prod(mask.GetSpacing())) / 1000.0


def _read_outputs(out_dir: Path, image: sitk.Image) -> dict[str, sitk.Image]:
    result: dict[str, sitk.Image] = {}
    for f in sorted(out_dir.glob("*.nii.gz")):
        m = sitk.ReadImage(str(f))
        if m.GetSize() == image.GetSize():
            m.CopyInformation(image)
        else:
            m = sitk.Resample(
                m, image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
            )
        result[f.name[: -len(".nii.gz")]] = sitk.Cast(m > 0, sitk.sitkUInt8)
    return result


def run_totalseg(
    image: sitk.Image, workdir: Path, task: str, timeout_s: int
) -> dict[str, sitk.Image]:
    """Run TotalSegmentator for one task; returns {class_name: UInt8 mask on the CCTA grid}."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ct = workdir / "ct.nii.gz"
    out_dir = workdir / task
    save_nifti(image, ct)
    cmd = ["TotalSegmentator", "-i", str(ct), "-o", str(out_dir), "-ta", task]
    if task == "total":
        cmd += ["--roi_subset", "aorta", "heart"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("TotalSegmentator timed out") from e
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        raise RuntimeError(f"TotalSegmentator failed ({task}): {tail}")
    return _read_outputs(out_dir, image)


def _aorta_connected(aorta: sitk.Image) -> bool:
    arr = sitk.GetArrayViewFromImage(aorta) > 0
    total = int(arr.sum())
    if total == 0:
        return False
    lab, n = ndimage.label(arr, structure=np.ones((3, 3, 3)))
    largest = int(np.bincount(lab.ravel())[1:].max()) if n else 0
    return largest / total >= 0.95 - 1e-12


def segment_anatomy(image: sitk.Image, workdir: Path, ap: AnatomyParams) -> Anatomy:
    workdir = Path(workdir)
    masks: dict[str, sitk.Image | None] = dict.fromkeys(_STRUCTURES)
    source = dict.fromkeys(_STRUCTURES, "missing")
    messages: list[str] = []

    try:
        hi = run_totalseg(image, workdir, HIGHRES, ap.totalseg_timeout_s)
        for key, name in (
            ("aorta", "aorta"),
            ("myocardium", "heart_myocardium"),
            ("lv_cavity", "heart_ventricle_left"),
        ):
            if name in hi:
                masks[key], source[key] = hi[name], HIGHRES
            else:
                messages.append(f"{HIGHRES} output has no '{name}'")
    except RuntimeError as e:
        messages.append(f"{HIGHRES} unavailable: {e}")

    if masks["aorta"] is None:
        try:
            tot = run_totalseg(image, workdir, "total", ap.totalseg_timeout_s)
            if "aorta" in tot:
                masks["aorta"], source["aorta"] = tot["aorta"], "total"
            else:
                messages.append("total output has no 'aorta'")
        except RuntimeError as e:
            messages.append(f"total unavailable: {e}")

    aorta, myo = masks["aorta"], masks["myocardium"]
    a_ml = _volume_ml(aorta) if aorta is not None else None
    m_ml = _volume_ml(myo) if myo is not None else None
    qc = {
        "aorta_present": a_ml is not None and ap.aorta_min_ml <= a_ml <= ap.aorta_max_ml,
        "aorta_connected": aorta is not None and _aorta_connected(aorta),
        "myo_present": m_ml is not None and ap.myo_min_ml <= m_ml <= ap.myo_max_ml,
        "ostia_ok": False,
    }
    if not qc["aorta_present"]:
        messages.append("Aorta missing or implausible volume")
    if not qc["myo_present"]:
        messages.append("Myocardium missing or implausible volume")
    return Anatomy(
        aorta=aorta,
        myocardium=myo,
        lv_cavity=masks["lv_cavity"],
        source=source,
        qc=qc,
        qc_messages=messages,
        myo_volume_ml=m_ml,
        aorta_volume_ml=a_ml,
    )


# ---------------------------------------------------------------- part 2: geometry (SPEC §5.3)


def _mask_points_mm(mask: sitk.Image) -> np.ndarray:
    """(N, 3) physical x, y, z of all foreground voxel centres."""
    idx = np.argwhere(sitk.GetArrayViewFromImage(mask) > 0)[:, ::-1]  # (x, y, z) indices
    d = np.array(mask.GetDirection()).reshape(3, 3)
    return np.array(mask.GetOrigin()) + (idx * np.array(mask.GetSpacing())) @ d.T


def find_ostia(
    g: CoronaryGraph, anat_or_aorta: "Anatomy | sitk.Image | None", ap: AnatomyParams
) -> dict[str, float | None]:
    """Distance (mm) from each main tree's root point to the nearest aorta voxel."""
    aorta = anat_or_aorta.aorta if isinstance(anat_or_aorta, Anatomy) else anat_or_aorta
    arr = None if aorta is None else sitk.GetArrayViewFromImage(aorta) > 0
    if arr is None or not arr.any():
        out: dict[str, float | None] = {t.tree_id: None for t in g.trees}
    else:
        dist = ndimage.distance_transform_edt(~arr, sampling=aorta.GetSpacing()[::-1])
        out = {}
        for t in g.trees:
            p = t.branches[t.root_branch_id].points_mm[0]
            ci = aorta.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in p))
            coords = np.array(ci[::-1]).reshape(3, 1)  # (z, y, x)
            out[t.tree_id] = float(
                ndimage.map_coordinates(dist, coords, order=1, mode="nearest")[0]
            )
    if isinstance(anat_or_aorta, Anatomy):
        vals = list(out.values())
        anat_or_aorta.qc["ostia_ok"] = len(g.trees) == 2 and all(
            v is not None and v <= ap.ostium_max_dist_mm for v in vals
        )
    return out


def lv_axis(myo: sitk.Image, aorta: sitk.Image | None) -> tuple[np.ndarray, np.ndarray] | None:
    """(base_point_mm, apex_point_mm) from the principal axis of the myocardium."""
    if aorta is None:
        return None
    pts = _mask_points_mm(myo)
    apts = _mask_points_mm(aorta)
    if len(pts) < 3 or len(apts) == 0:
        return None
    mean = pts.mean(axis=0)
    u = np.linalg.svd(pts - mean, full_matrices=False)[2][0]
    proj = (pts - mean) @ u
    ends = [mean + u * np.percentile(proj, q) for q in (1, 99)]
    c = apts.mean(axis=0)
    d = [np.linalg.norm(e - c) for e in ends]
    apex = int(np.argmax(d))
    return ends[1 - apex], ends[apex]


def _lm_rule(tree: VesselTree, ap: AnatomyParams) -> bool:
    root = tree.branches[tree.root_branch_id]
    return branch_length_mm(root) <= ap.lm_max_length_mm and len(root.children) >= 2


def _first5_radius(b: Branch) -> float:
    sel = b.s_mm <= 5.0
    return float(np.median(b.radius_mm[sel] if sel.any() else b.radius_mm))


def label_branches(g: CoronaryGraph, anat: Anatomy, ap: AnatomyParams) -> None:
    """Coarse labels LM / LAD / LCX / RCA / other (fills g.labels and g.tree_side)."""
    for b in g.labels:
        g.labels[b] = "other"

    def warn(msg: str) -> None:
        if msg not in g.warnings:
            g.warnings.append(msg)

    roots = {t.tree_id: t.branches[t.root_branch_id].points_mm[0] for t in g.trees}
    if len(g.trees) >= 2:
        order = sorted(g.trees, key=lambda t: roots[t.tree_id][0], reverse=True)
        g.tree_side = {t.tree_id: "unknown" for t in g.trees}
        g.tree_side[order[0].tree_id] = "left"
        g.tree_side[order[1].tree_id] = "right"
    else:
        g.tree_side = {t.tree_id: "unknown" for t in g.trees}
        if g.trees and anat.aorta is not None and (apts := _mask_points_mm(anat.aorta)).size:
            t = g.trees[0]
            left = roots[t.tree_id][0] > apts[:, 0].mean()
            g.tree_side[t.tree_id] = "left" if left else "right"

    axis = None
    if anat.myocardium is not None and anat.qc.get("myo_present"):
        axis = lv_axis(anat.myocardium, anat.aorta)

    for t in g.trees:
        side = g.tree_side[t.tree_id]
        if side == "right":
            for bid in main_path(t, t.root_branch_id):
                g.labels[bid] = "RCA"
        elif side == "left":
            if not _lm_rule(t, ap):
                warn("LM not identified")
                continue
            root = t.branches[t.root_branch_id]
            g.labels[root.branch_id] = "LM"
            if axis is None:
                warn("LAD/LCX not labelled: no myocardium")
                continue
            apex = axis[1]
            paths = {c: main_path(t, c) for c in root.children}
            lad = min(
                root.children,
                key=lambda c: np.linalg.norm(t.branches[paths[c][-1]].points_mm[-1] - apex),
            )
            for bid in paths[lad]:
                g.labels[bid] = "LAD"
            rest = [c for c in root.children if c != lad]
            if rest:
                lcx = max(rest, key=lambda c: _first5_radius(t.branches[c]))
                for bid in paths[lcx]:
                    g.labels[bid] = "LCX"


def _flag(kind, branch_id, point, direction, message) -> dict:
    return {
        "kind": kind,
        "branch_id": branch_id,
        "point_mm": None if point is None else [float(v) for v in point],
        "direction": None if direction is None else [float(v) for v in direction],
        "message": message,
    }


def coverage_check(g: CoronaryGraph, ap: AnatomyParams) -> list[dict]:
    flags: list[dict] = []
    gp = GraphParams()
    if len(g.trees) < 2:
        flags.append(
            _flag("missing_tree", None, None, None, f"Only {len(g.trees)} coronary tree(s) found")
        )
    for t in g.trees:
        if g.tree_side.get(t.tree_id) == "left" and not _lm_rule(t, ap):
            root = t.branches[t.root_branch_id]
            flags.append(
                _flag(
                    "missing_main_branch",
                    root.branch_id,
                    root.points_mm[-1],
                    end_direction(root, gp.direction_window_mm),
                    "Left main bifurcation not found",
                )
            )
    branches = {b.branch_id: b for t in g.trees for b in t.branches.values()}
    for name, min_len in (
        ("LAD", ap.min_main_len_lad_mm),
        ("LCX", ap.min_main_len_lcx_mm),
        ("RCA", ap.min_main_len_rca_mm),
    ):
        ids = [b for b in branches if g.labels.get(b) == name]
        if not ids:
            continue
        total = sum(branch_length_mm(branches[b]) for b in ids)
        if total < min_len:
            last = next(
                branches[b]
                for b in ids
                if not any(g.labels.get(c) == name for c in branches[b].children)
            )
            flags.append(
                _flag(
                    "short_main_vessel",
                    last.branch_id,
                    last.points_mm[-1],
                    end_direction(last, gp.direction_window_mm),
                    f"{name} main path {total:.1f} mm < {min_len:.0f} mm",
                )
            )
    for e in g.endings:
        if e.kind == "abrupt":
            flags.append(
                _flag(
                    "abrupt_end",
                    e.branch_id,
                    e.point_mm,
                    e.direction,
                    f"Vessel ends abruptly (r_end {e.r_end_mm:.2f} mm)",
                )
            )
    return flags
