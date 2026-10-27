"""Cross-fit segmenter setup (D-V2-S26): Dataset511 = v2_train only, 3 folds, symlinked data.

Builds /data/nnUNet_preprocessed/Dataset511_ImageCASxfit from Dataset501_ImageCAS without copying
data or touching Dataset501 (its splits_final.json defines the v2 case roles). Only v2_train
cases are linked, so test / v2_holdout / v2_realfail cases cannot be read by the xfit trainings.
Folds: v2_train ids shuffled with seed 0, split in 3 near-equal parts (val = one part).
"""

import json
import sys

import numpy as np

from twinloop.paths import data_root
from twinloop.v2split import load_roles

SRC, DST = "Dataset501_ImageCAS", "Dataset511_ImageCASxfit"
N_FOLDS = 3


def main() -> int:
    root = data_root() / "nnUNet_preprocessed"
    src, dst = root / SRC, root / DST
    ids = sorted(c for c, r in load_roles().items() if r == "v2_train")
    assert all(c % 5 != 0 for c in ids)
    perm = np.random.RandomState(0).permutation(len(ids))
    parts = np.array_split(perm, N_FOLDS)
    name = lambda c: f"ImageCAS_{c:04d}"  # noqa: E731
    splits = []
    for k in range(N_FOLDS):
        val = sorted(ids[i] for i in parts[k])
        train = sorted(set(ids) - set(val))
        splits.append({"train": [name(c) for c in train], "val": [name(c) for c in val]})

    (dst / "nnUNetPlans_3d_fullres").mkdir(parents=True, exist_ok=True)
    (dst / "gt_segmentations").mkdir(exist_ok=True)
    n = 0
    for c in ids:
        for sub, suffixes in (
            ("nnUNetPlans_3d_fullres", (".b2nd", "_seg.b2nd", ".pkl")),
            ("gt_segmentations", (".nii.gz",)),
        ):
            for suf in suffixes:
                f = src / sub / f"{name(c)}{suf}"
                link = dst / sub / f.name
                if not f.exists():
                    sys.exit(f"missing {f}")
                if not link.is_symlink():
                    link.symlink_to(f)
                    n += 1
    plans = json.loads((src / "nnUNetPlans.json").read_text())
    plans["dataset_name"] = DST
    (dst / "nnUNetPlans.json").write_text(json.dumps(plans, indent=2))
    for f in ("dataset.json", "dataset_fingerprint.json"):
        (dst / f).write_text((src / f).read_text())
    (dst / "splits_final.json").write_text(json.dumps(splits, indent=2))
    (dst / "folds_summary.json").write_text(
        json.dumps({"n_cases": len(ids), "val_ids": [s["val"] for s in splits]}, indent=1)
    )
    print(
        f"{DST}: {len(ids)} cases, new links {n}, fold val sizes {[len(s['val']) for s in splits]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
