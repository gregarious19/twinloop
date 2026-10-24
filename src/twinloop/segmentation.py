"""Segmentation (SPEC §5.2). The full nnU-Net wrapper and baseline arrive in T08."""

import os
from pathlib import Path

import SimpleITK as sitk

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
    raise RuntimeError(NOT_AVAILABLE)
