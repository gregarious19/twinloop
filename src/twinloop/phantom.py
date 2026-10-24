"""Synthetic test volumes (SPEC §6.9). Origin (0,0,0), identity direction, LPS."""

import argparse
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

SIZE_XYZ = (160, 160, 200)
SPACING_XYZ = (0.35, 0.35, 0.4)
VESSEL_HU = 400


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical (x, y, z) mm of every voxel centre, each shaped (z, y, x)."""
    zz, yy, xx = np.indices(SIZE_XYZ[::-1], dtype=np.float64)
    return xx * SPACING_XYZ[0], yy * SPACING_XYZ[1], zz * SPACING_XYZ[2]


def _cylinder(grid, a, b, radius: float) -> np.ndarray:
    """Voxels within `radius` of segment a-b, clipped to the segment length (flat caps)."""
    a = np.asarray(a, dtype=np.float64)
    d = np.asarray(b, dtype=np.float64) - a
    rel = [g - a[i] for i, g in enumerate(grid)]
    t = sum(rel[i] * d[i] for i in range(3)) / float(d @ d)
    perp2 = sum((rel[i] - t * d[i]) ** 2 for i in range(3))
    return (t >= 0.0) & (t <= 1.0) & (perp2 <= radius**2)


def _tube(grid, radius_of_z) -> np.ndarray:
    x, y, z = grid
    r2 = (x - 28.0) ** 2 + (y - 28.0) ** 2
    return (z >= 10.0) & (z <= 70.0) & (r2 <= radius_of_z(z) ** 2)


def make_phantom(kind: str) -> tuple[sitk.Image, sitk.Image]:
    grid = _grid()
    if kind == "tube":
        mask = _tube(grid, lambda z: 1.5)
    elif kind == "stenosis":
        mask = _tube(grid, lambda z: np.where((z >= 37.0) & (z <= 43.0), 0.75, 1.5))
    elif kind == "y_tree":
        tilt = np.deg2rad(30.0)
        junction = np.array([28.0, 28.0, 35.0])
        mask = _cylinder(grid, (28, 28, 5), junction, 2.0)
        for sign in (1.0, -1.0):
            end = junction + 35.0 * np.array([sign * np.sin(tilt), 0.0, np.cos(tilt)])
            mask |= _cylinder(grid, junction, end, 1.5)
    elif kind == "two_trees":
        mask = _tube(grid, lambda z: 1.5) | _cylinder(grid, (15, 40, 10), (15, 40, 50), 1.2)
    else:
        raise ValueError(f"Unknown phantom kind: {kind}")
    mask_img = sitk.GetImageFromArray(mask.astype(np.uint8))
    img = sitk.GetImageFromArray((VESSEL_HU * mask).astype(np.int16))
    for im in (img, mask_img):
        im.SetSpacing(SPACING_XYZ)
        im.SetOrigin((0.0, 0.0, 0.0))
        im.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return img, mask_img


def write_dicom_series(img: sitk.Image, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img16 = sitk.Cast(img, sitk.sitkInt16)
    w = sitk.ImageFileWriter()
    w.KeepOriginalImageUIDOn()
    sx, sy, sz = img16.GetSpacing()
    uid = "1.2.826.0.1.3680043.2.1125." + str(int(time.time()))
    for k in range(img16.GetDepth()):
        sl = img16[:, :, k]
        z = img16.TransformIndexToPhysicalPoint((0, 0, k))
        for tag, val in [
            ("0020|000e", uid + ".1"),
            ("0020|000d", uid),
            ("0008|0060", "CT"),
            ("0020|0032", "\\".join(map(str, z))),
            ("0020|0013", str(k + 1)),
            ("0028|0030", f"{sy}\\{sx}"),
            ("0018|0050", str(sz)),
            ("0020|0037", "1\\0\\0\\0\\1\\0"),
        ]:
            sl.SetMetaData(tag, val)
        w.SetFileName(str(out_dir / f"slice_{k:04d}.dcm"))
        w.Execute(sl)


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a phantom image + mask as NIfTI")
    ap.add_argument("--kind", required=True, choices=["tube", "stenosis", "y_tree", "two_trees"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    img, mask = make_phantom(args.kind)
    args.out.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(args.out / f"{args.kind}_img.nii.gz"))
    sitk.WriteImage(mask, str(args.out / f"{args.kind}_mask.nii.gz"))
    print(f"wrote {args.kind}_img.nii.gz and {args.kind}_mask.nii.gz to {args.out}")


if __name__ == "__main__":
    main()
