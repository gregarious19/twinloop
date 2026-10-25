"""Gap-filler inference: patch extraction and applying a model inside one patch (SPEC_V2 §8.1)."""

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

from twinloop.config_v2 import GapfillParams


def _reference_grid(centre_mm: np.ndarray, gp: GapfillParams) -> sitk.Image:
    n, sp = gp.patch_vox, gp.patch_spacing_mm
    ref = sitk.Image((n, n, n), sitk.sitkUInt8)
    ref.SetSpacing((sp, sp, sp))
    ref.SetOrigin(tuple(float(v) for v in centre_mm - sp * (n - 1) / 2.0))
    return ref


def extract_patch(
    image: sitk.Image,
    mask: sitk.Image,
    seed_mm,
    direction,
    offset_mm: float,
    gp: GapfillParams,
) -> dict:
    """{"x": float32 (3, 64, 64, 64) [normalised image, mask, seed heatmap], "ref": grid}."""
    seed = np.asarray(seed_mm, float)
    centre = seed + offset_mm * np.asarray(direction, float)
    ref = _reference_grid(centre, gp)
    hu = sitk.GetArrayFromImage(
        sitk.Resample(image, ref, sitk.Transform(), sitk.sitkLinear, -1000.0, sitk.sitkFloat32)
    )
    m = sitk.GetArrayFromImage(
        sitk.Resample(mask, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    )
    lo, hi = gp.hu_clip
    img_n = (np.clip(hu, lo, hi) - gp.hu_center) / gp.hu_scale
    ax = [
        ref.GetOrigin()[k] + gp.patch_spacing_mm * np.arange(gp.patch_vox) for k in range(3)
    ]  # x, y, z
    zz, yy, xx = np.meshgrid(ax[2], ax[1], ax[0], indexing="ij")
    d2 = (xx - seed[0]) ** 2 + (yy - seed[1]) ** 2 + (zz - seed[2]) ** 2
    heat = np.exp(-d2 / (2.0 * gp.seed_sigma_mm**2))
    x = np.stack([img_n, (m > 0).astype(np.float32), heat]).astype(np.float32)
    return {"x": x, "ref": ref}


def apply_gapfill(
    model,
    image: sitk.Image,
    mask: sitk.Image,
    seed_mm,
    direction,
    offset_mm: float,
    gp: GapfillParams,
    device: str,
) -> tuple[sitk.Image, int]:
    """Run `model` on one patch; returns (mask with new voxels, number of new voxels)."""
    p = extract_patch(image, mask, seed_mm, direction, offset_mm, gp)
    ref = p["ref"]
    if hasattr(model, "set_reference"):  # test hook only; real nets do not have it
        model.set_reference(ref)
    with torch.no_grad():
        logits = model(torch.from_numpy(p["x"])[None].to(device))
    pred = torch.sigmoid(logits)[0, 0].cpu().numpy() > gp.prob_threshold
    in_patch = p["x"][1] > 0.5
    new = pred & ~in_patch
    lab, _ = ndimage.label(new | in_patch, structure=np.ones((3, 3, 3)))
    touching = np.unique(lab[in_patch])
    new &= np.isin(lab, touching[touching > 0])
    if not new.any():
        return mask, 0

    # resample the new voxels back onto the CCTA grid, only inside the patch's bounding box
    new_img = sitk.GetImageFromArray(new.astype(np.uint8))
    new_img.CopyInformation(ref)
    n, sp = gp.patch_vox, gp.patch_spacing_mm
    o = np.array(ref.GetOrigin())
    lo_c, hi_c = o - 0.5 * sp, o + (n - 0.5) * sp
    corners = [np.where(np.array(c) == 0, lo_c, hi_c) for c in np.ndindex(2, 2, 2)]
    idx = np.array([mask.TransformPhysicalPointToContinuousIndex(tuple(c)) for c in corners])
    size = np.array(mask.GetSize())
    lo_i = np.clip(np.floor(idx.min(0)).astype(int) - 1, 0, size - 1)
    hi_i = np.clip(np.ceil(idx.max(0)).astype(int) + 1, 0, size - 1)
    roi = mask[lo_i[0] : hi_i[0] + 1, lo_i[1] : hi_i[1] + 1, lo_i[2] : hi_i[2] + 1]
    res = sitk.Resample(new_img, roi, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    new_roi = (sitk.GetArrayFromImage(res) > 0) & ~(sitk.GetArrayFromImage(roi) > 0)
    count = int(new_roi.sum())
    if count < gp.min_new_voxels:
        return mask, 0
    out = sitk.GetArrayFromImage(mask).copy()
    out[lo_i[2] : hi_i[2] + 1, lo_i[1] : hi_i[1] + 1, lo_i[0] : hi_i[0] + 1] |= new_roi.astype(
        out.dtype
    )
    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(mask)
    return out_img, count
