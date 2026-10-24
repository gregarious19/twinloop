import re
import zipfile

import numpy as np
import pytest
import SimpleITK as sitk

from twinloop.io import load_dicom_zip, load_image, load_mask, save_nifti
from twinloop.phantom import make_phantom, write_dicom_series


def test_nifti_round_trip(tmp_path):
    img, _ = make_phantom("tube")
    save_nifti(img, tmp_path / "a.nii.gz")
    back = load_image(tmp_path / "a.nii.gz")
    assert np.array_equal(sitk.GetArrayFromImage(back), sitk.GetArrayFromImage(img))
    assert back.GetSpacing() == pytest.approx(img.GetSpacing())


def test_ras_image_is_reoriented_to_lps(tmp_path):
    img, _ = make_phantom("tube")
    ras = sitk.Image(img)
    ras.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    save_nifti(ras, tmp_path / "ras.nii.gz")
    back = load_image(tmp_path / "ras.nii.gz")
    assert back.GetDirection() == pytest.approx((1, 0, 0, 0, 1, 0, 0, 0, 1))
    assert back.GetSize() == img.GetSize()


def test_dicom_zip(tmp_path):
    img, _ = make_phantom("tube")
    dcm = tmp_path / "dcm"
    write_dicom_series(img, dcm)
    zpath = tmp_path / "series.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for f in sorted(dcm.iterdir()):
            zf.write(f, f"study/{f.name}")
    back = load_image(zpath)
    assert back.GetSize() == img.GetSize()
    assert back.GetSpacing() == pytest.approx(img.GetSpacing(), abs=1e-3)


def test_zip_without_dicom(tmp_path):
    zpath = tmp_path / "empty.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("readme.txt", "no images here")
    with pytest.raises(ValueError, match="No DICOM series found in zip"):
        load_dicom_zip(zpath, tmp_path / "x")


def test_unsupported_extension(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    msg = "Only .nii.gz, .nii or .zip files are supported"
    with pytest.raises(ValueError, match=re.escape(msg)):
        load_image(tmp_path / "a.txt")


def test_too_few_slices(tmp_path):
    save_nifti(sitk.Image(8, 8, 10, sitk.sitkInt16), tmp_path / "s.nii.gz")
    with pytest.raises(ValueError, match="Image too small: need at least 64 slices"):
        load_image(tmp_path / "s.nii.gz")


def test_not_3d(tmp_path):
    sitk.WriteImage(sitk.Image(8, 8, sitk.sitkInt16), str(tmp_path / "flat.nii.gz"))
    with pytest.raises(ValueError, match="Image must be 3D"):
        load_image(tmp_path / "flat.nii.gz")


def test_mask_shape_mismatch(tmp_path):
    img, mask = make_phantom("tube")
    small = sitk.Image(8, 8, 8, sitk.sitkUInt8)
    save_nifti(small, tmp_path / "m.nii.gz")
    with pytest.raises(ValueError, match="Mask shape does not match image"):
        load_mask(tmp_path / "m.nii.gz", img)


def test_mask_binarised_and_info_copied(tmp_path):
    img, mask = make_phantom("tube")
    save_nifti(sitk.Cast(mask, sitk.sitkUInt8) * 7, tmp_path / "m.nii.gz")
    m = load_mask(tmp_path / "m.nii.gz", img)
    assert set(np.unique(sitk.GetArrayFromImage(m))) == {0, 1}
    assert m.GetSpacing() == img.GetSpacing()
    assert m.GetOrigin() == img.GetOrigin()
