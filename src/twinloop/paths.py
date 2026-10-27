"""Data locations for v2 (SPEC_V2 §2.1)."""

import os
from pathlib import Path

_RAW = "nnUNet_raw/Dataset501_ImageCAS"


def data_root() -> Path:
    return Path(os.environ.get("TWINLOOP_DATA", "/data"))


def resolve(p: str | Path) -> Path:
    s = str(p)
    if s.startswith("/data/"):
        return data_root() / s[6:]
    return Path(p)


def _check(case_id: int) -> None:
    if case_id % 5 == 0:
        raise ValueError(f"Test-split case not allowed in v2: {case_id}")


def case_image_path(case_id: int) -> Path:
    _check(case_id)
    return data_root() / _RAW / "imagesTr" / f"ImageCAS_{case_id:04d}_0000.nii.gz"


def case_label_path(case_id: int) -> Path:
    _check(case_id)
    return data_root() / _RAW / "labelsTr" / f"ImageCAS_{case_id:04d}.nii.gz"


def models_dir() -> Path:
    return data_root() / "models" / "v2"


def v2_dir() -> Path:
    return data_root() / "v2"


def v3_dir() -> Path:
    return data_root() / "v3"
