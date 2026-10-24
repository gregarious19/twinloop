"""Convert the ImageCAS manifest into an nnU-Net v2 raw dataset (LPS, binary labels)."""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

from twinloop.io import load_image, load_mask, save_nifti


def _convert(args: tuple[int, str, str, str, Path]) -> int:
    cid, image_path, label_path, split, out = args
    sub = "Tr" if split == "train" else "Ts"
    img_out = out / f"images{sub}" / f"ImageCAS_{cid:04d}_0000.nii.gz"
    lab_out = out / f"labels{sub}" / f"ImageCAS_{cid:04d}.nii.gz"
    if img_out.exists() and lab_out.exists():
        return cid
    img = load_image(Path(image_path))
    lab = load_mask(Path(label_path), img)
    save_nifti(img, img_out.with_name(img_out.name + ".tmp.nii.gz"))
    save_nifti(sitk.Cast(lab, sitk.sitkUInt8), lab_out.with_name(lab_out.name + ".tmp.nii.gz"))
    # labels first so an interrupted case never looks complete
    lab_out.with_name(lab_out.name + ".tmp.nii.gz").rename(lab_out)
    img_out.with_name(img_out.name + ".tmp.nii.gz").rename(img_out)
    return cid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    m = pd.read_csv(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(int(r.case_id), r.image_path, r.label_path, r.split, args.out) for r in m.itertuples()]
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_convert, jobs), 1):
            if i % 50 == 0:
                print(f"{i}/{len(jobs)}", flush=True)

    n_train = int((m.split == "train").sum())
    meta = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "coronary": 1},
        "numTraining": n_train,
        "file_ending": ".nii.gz",
        "name": "ImageCAS",
    }
    (args.out / "dataset.json").write_text(json.dumps(meta, indent=2))
    print(f"CONVERT DONE train={n_train} test={len(m) - n_train}")


if __name__ == "__main__":
    main()
