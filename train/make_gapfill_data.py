"""Gap-filler training samples (SPEC_V2 §14 'gapfill')."""

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from train.shards import parse_args, run_shard
from twinloop.config_v2 import V2Params
from twinloop.damage import damage_case
from twinloop.gapfill import extract_patch
from twinloop.graph import build_graph
from twinloop.io import load_image, load_mask
from twinloop.paths import case_image_path, case_label_path

SEEDS = (0, 1)
OFFSET_MM = {"gap": 8.0, "truncate": 8.0, "delete_branch": 8.0, "shrink": 0.0}
P = V2Params()


def _target(mask: sitk.Image, ref: sitk.Image) -> np.ndarray:
    r = sitk.Resample(mask, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return (sitk.GetArrayFromImage(r) > 0).astype(np.uint8)


def _save(path: Path, x, y, case_id: int, typ: str) -> None:
    np.savez_compressed(path, x=x.astype(np.float16), y=y, case_id=case_id, type=typ)


def process_case(cid: int, out: Path) -> None:
    image = load_image(case_image_path(cid))
    gt = load_mask(case_label_path(cid), image)
    g0 = build_graph(gt, P.centerline, P.graph)
    gp = P.gapfill
    for seed in SEEDS:
        rng = np.random.default_rng(1000003 * seed + cid)
        img2, mask2, recs = damage_case(image, gt, g0, P.damage, rng)
        for r in recs:
            if r.type == "false_branch":
                continue
            if r.type == "occlusion":
                p = extract_patch(img2, mask2, r.point_mm, r.direction, 8.0, gp)
                y = _target(mask2, p["ref"])
            elif r.type in OFFSET_MM:
                p = extract_patch(img2, mask2, r.point_mm, r.direction, OFFSET_MM[r.type], gp)
                y = _target(gt, p["ref"])
            else:
                continue
            _save(out / f"{cid:04d}_s{seed}_{r.damage_id}.npz", p["x"], y, cid, r.type)
        tapers = [e for e in g0.endings if e.kind == "taper"]
        if tapers:
            pick = rng.choice(len(tapers), size=min(2, len(tapers)), replace=False)
            for j, k in enumerate(sorted(pick)):
                e = tapers[int(k)]
                p = extract_patch(image, gt, e.point_mm, e.direction, 8.0, gp)
                _save(
                    out / f"{cid:04d}_s{seed}_neg{j}.npz",
                    p["x"],
                    _target(gt, p["ref"]),
                    cid,
                    "taper_neg",
                )


if __name__ == "__main__":
    run_shard(parse_args("gap-filler dataset"), process_case)
