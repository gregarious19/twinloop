"""Build the ImageCAS manifest (SPEC §4.2)."""

import argparse
import re
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from twinloop.io import load_image

COLUMNS = [
    "case_id",
    "image_path",
    "label_path",
    "split",
    "size_x",
    "size_y",
    "size_z",
    "spacing_x",
    "spacing_y",
    "spacing_z",
]


def _case_id(path: Path) -> int | None:
    m = re.search(r"\d+", path.name)
    return int(m.group()) if m else None


def _scan(root: Path) -> tuple[dict[int, Path], dict[int, Path], list[Path]]:
    images: dict[int, Path] = {}
    labels: dict[int, Path] = {}
    unpaired: list[Path] = []
    files = sorted(p for p in root.rglob("*") if p.name.endswith((".nii.gz", ".nii")))
    for p in files:
        if "(1)" in str(p):
            continue
        cid = _case_id(p)
        if cid is None:
            unpaired.append(p)
            continue
        target = labels if "label" in p.name.lower() else images
        if cid in target:
            unpaired.append(p)
        else:
            target[cid] = p
    return images, labels, unpaired


def _row(args: tuple[int, Path, Path]) -> dict:
    cid, img_path, lab_path = args
    img = load_image(img_path)
    size, sp = img.GetSize(), img.GetSpacing()
    return {
        "case_id": cid,
        "image_path": str(img_path),
        "label_path": str(lab_path),
        "split": "test" if cid % 5 == 0 else "train",
        "size_x": size[0],
        "size_y": size[1],
        "size_z": size[2],
        "spacing_x": sp[0],
        "spacing_y": sp[1],
        "spacing_z": sp[2],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    images, labels, unpaired = _scan(args.root)
    paired = sorted(set(images) & set(labels))
    unpaired += [images[c] for c in sorted(set(images) - set(labels))]
    unpaired += [labels[c] for c in sorted(set(labels) - set(images))]

    with Pool(8) as pool:
        rows = pool.map(_row, [(c, images[c], labels[c]) for c in paired])
    df = pd.DataFrame(rows, columns=COLUMNS).sort_values("case_id")
    df.to_csv(args.out, index=False)

    n_test = int((df.split == "test").sum())
    print(f"cases={len(df)} train={len(df) - n_test} test={n_test} unpaired={len(unpaired)}")
    for p in unpaired:
        print("unpaired:", p)


if __name__ == "__main__":
    main()
