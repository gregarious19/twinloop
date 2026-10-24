"""Image I/O (SPEC §5.1). Everything is reoriented to LPS on load.

numpy arrays from `sitk.GetArrayFromImage` are (z, y, x); `GetSpacing()` is (x, y, z).
"""

import tempfile
import zipfile
from pathlib import Path

import SimpleITK as sitk

UNSUPPORTED = "Only .nii.gz, .nii or .zip files are supported"


def _to_lps(img: sitk.Image) -> sitk.Image:
    return sitk.DICOMOrient(img, "LPS")


def load_dicom_zip(path: Path, tmp_dir: Path) -> sitk.Image:
    """Extract a zipped DICOM study and read the series with the most files."""
    with zipfile.ZipFile(path) as zf:
        zf.extractall(tmp_dir)
    best: tuple[int, tuple[str, ...]] = (0, ())
    for d in sorted({p.parent for p in Path(tmp_dir).rglob("*") if p.is_file()}):
        for sid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(d)) or ():
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(d), sid)
            if len(files) > best[0]:
                best = (len(files), tuple(files))
    if best[0] == 0:
        raise ValueError("No DICOM series found in zip")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(list(best[1]))
    return reader.Execute()


def load_image(path: Path) -> sitk.Image:
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".nii.gz", ".nii")):
        img = sitk.ReadImage(str(path))
    elif name.endswith(".zip"):
        with tempfile.TemporaryDirectory() as td:
            img = load_dicom_zip(path, Path(td))
    else:
        raise ValueError(UNSUPPORTED)
    if img.GetDimension() != 3:
        raise ValueError("Image must be 3D")
    img = _to_lps(img)
    if img.GetSize()[2] < 64:
        raise ValueError("Image too small: need at least 64 slices")
    return img


def load_mask(path: Path, reference: sitk.Image) -> sitk.Image:
    mask = _to_lps(sitk.ReadImage(str(path)))
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    if mask.GetSize() != reference.GetSize():
        raise ValueError("Mask shape does not match image")
    mask.CopyInformation(reference)
    return mask


def save_nifti(img: sitk.Image, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))
