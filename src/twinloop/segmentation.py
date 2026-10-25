"""Segmentation (SPEC §5.2): nnU-Net v2 subprocess wrapper and the threshold baseline."""

import os
import subprocess
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

NOT_AVAILABLE = "Segmentation model not available; provide a mask or finish training"


def _trainer(trainer: str | None) -> str:
    return trainer or os.environ.get("TWINLOOP_NNUNET_TRAINER") or "nnUNetTrainer"


def model_available(trainer: str | None = None) -> bool:
    results = Path(os.environ.get("nnUNet_results", "/data/nnUNet_results"))
    ckpt = (
        results
        / "Dataset501_ImageCAS"
        / f"{_trainer(trainer)}__nnUNetPlans__3d_fullres"
        / "fold_0"
        / "checkpoint_final.pth"
    )
    return ckpt.is_file()


def segment_nnunet(
    image: sitk.Image, workdir: Path, trainer: str | None = None, timeout_s: int = 900
) -> sitk.Image:
    trainer = _trainer(trainer)
    if not model_available(trainer):
        raise RuntimeError(NOT_AVAILABLE)
    workdir = Path(workdir)
    in_dir, out_dir = workdir / "in", workdir / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(in_dir / "case_0000.nii.gz"))
    cmd = [
        "nnUNetv2_predict",
        "-i", str(in_dir),
        "-o", str(out_dir),
        "-d", "501",
        "-c", "3d_fullres",
        "-f", "0",
        "-tr", trainer,
        "--disable_tta",
    ]  # fmt: skip
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Segmentation timed out") from exc
    out_file = out_dir / "case.nii.gz"
    if proc.returncode != 0 or not out_file.is_file():
        (workdir / "nnunet_log.txt").write_text(proc.stdout + "\n" + proc.stderr)
        raise RuntimeError(f"Segmentation failed (exit {proc.returncode})")
    mask = sitk.Cast(sitk.ReadImage(str(out_file)) > 0, sitk.sitkUInt8)
    mask.CopyInformation(image)
    return mask


def segment_threshold_baseline(image: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(image)
    cand = (arr >= 250) & (arr <= 700)
    labels, n = ndimage.label(cand, structure=np.ones((3, 3, 3)))
    if n > 0:
        sizes = np.bincount(labels.ravel())[1:]
        keep = np.argsort(sizes)[::-1][:2] + 1
        cand = np.isin(labels, keep)
    mask = sitk.GetImageFromArray(cand.astype(np.uint8))
    mask.CopyInformation(image)
    return mask
