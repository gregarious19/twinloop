"""Segmentation evaluation (SPEC §8): Dice and centerline sensitivity per case."""

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.morphology import skeletonize

from twinloop.io import load_image, load_mask
from twinloop.segmentation import segment_threshold_baseline

DATA = Path(os.environ.get("TWINLOOP_DATA", "/data"))
RAW = DATA / "nnUNet_raw" / "Dataset501_ImageCAS"
COLUMNS = ["case_id", "method", "dice", "cl_sens", "pred_volume_mm3", "gt_volume_mm3", "infer_s"]


def metrics(pred: np.ndarray, gt: np.ndarray, spacing_xyz: tuple[float, ...]) -> dict:
    pred, gt = pred > 0, gt > 0
    denom = pred.sum() + gt.sum()
    dice = 2.0 * np.logical_and(pred, gt).sum() / denom if denom else 1.0
    skel = skeletonize(gt)
    cl_sens = float(pred[skel].mean()) if skel.any() else float("nan")
    vox = float(np.prod(spacing_xyz))
    return {
        "dice": float(dice),
        "cl_sens": cl_sens,
        "pred_volume_mm3": float(pred.sum() * vox),
        "gt_volume_mm3": float(gt.sum() * vox),
    }


def _load_gt(cid: int, sub: str) -> sitk.Image:
    return sitk.ReadImage(str(RAW / f"labels{sub}" / f"ImageCAS_{cid:04d}.nii.gz"))


def _predict_nnunet(cids: list[int], sub: str, split: str, limit: int | None) -> tuple[Path, float]:
    trainer = os.environ.get("TWINLOOP_NNUNET_TRAINER", "nnUNetTrainer")
    in_dir = RAW / f"images{sub}"
    tmp = None
    if limit is not None:
        tmp = DATA / "runs" / "eval" / f"nnunet_in_{split}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for cid in cids:
            name = f"ImageCAS_{cid:04d}_0000.nii.gz"
            shutil.copy(in_dir / name, tmp / name)
        in_dir = tmp
    out_dir = (
        DATA / "runs" / "eval" / ("nnunet_pred" if split == "test" else f"nnunet_pred_{split}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nnUNetv2_predict", "-i", str(in_dir), "-o", str(out_dir), "-d", "501",
        "-c", "3d_fullres", "-f", "0", "-tr", trainer, "--disable_tta",
    ]  # fmt: skip
    t = time.perf_counter()
    subprocess.run(cmd, check=True)
    elapsed = time.perf_counter() - t
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_dir, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["nnunet", "threshold"], required=True)
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--manifest", type=Path, default=DATA / "raw/imagecas/manifest.csv")
    args = ap.parse_args()

    m = pd.read_csv(args.manifest)
    m = m[m.split == args.split].sort_values("case_id")
    if args.limit:
        m = m.head(args.limit)
    sub = "Ts" if args.split == "test" else "Tr"
    cids = [int(c) for c in m.case_id]

    rows = []
    if args.method == "nnunet":
        pred_dir, elapsed = _predict_nnunet(cids, sub, args.split, args.limit)
        per_case = elapsed / len(cids)
        for cid in cids:
            pred = sitk.ReadImage(str(pred_dir / f"ImageCAS_{cid:04d}.nii.gz"))
            gt = _load_gt(cid, sub)
            r = metrics(sitk.GetArrayFromImage(pred), sitk.GetArrayFromImage(gt), gt.GetSpacing())
            rows.append({"case_id": cid, "method": "nnunet", **r, "infer_s": per_case})
    else:
        for r_ in m.itertuples():
            image = load_image(Path(r_.image_path))
            gt = load_mask(Path(r_.label_path), image)
            t = time.perf_counter()
            pred = segment_threshold_baseline(image)
            dt = time.perf_counter() - t
            r = metrics(sitk.GetArrayFromImage(pred), sitk.GetArrayFromImage(gt), gt.GetSpacing())
            rows.append({"case_id": int(r_.case_id), "method": "threshold", **r, "infer_s": dt})

    new = pd.DataFrame(rows, columns=COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        old = pd.read_csv(args.out)
        new = pd.concat([old[old.method != args.method], new], ignore_index=True)
    new.to_csv(args.out, index=False)
    mine = pd.DataFrame(rows)
    print(
        f"method={args.method} n={len(mine)} dice_mean={mine.dice.mean():.4f} "
        f"dice_sd={mine.dice.std():.4f} clsens_mean={mine.cl_sens.mean():.4f}"
    )


if __name__ == "__main__":
    main()
