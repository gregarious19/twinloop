"""TotalSegmentator anatomy: aorta, myocardium, LV cavity and quality checks (SPEC_V2 §5)."""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from twinloop.config_v2 import AnatomyParams
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
