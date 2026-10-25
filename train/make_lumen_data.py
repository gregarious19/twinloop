"""Lumen-model training branches (SPEC_V2 §14 'lumen')."""

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from train.shards import parse_args, run_shard
from twinloop.config_v2 import V2Params
from twinloop.cpr import center_component, rmf, straighten_points
from twinloop.graph import build_graph
from twinloop.io import load_image, load_mask
from twinloop.lesions import reference_radius
from twinloop.paths import case_image_path, case_label_path

P = V2Params()
HU_CLIP = (-200.0, 800.0)
HU_CENTER, HU_SCALE = 300.0, 500.0


def process_case(cid: int, out: Path) -> None:
    image = load_image(case_image_path(cid))
    gt = load_mask(case_label_path(cid), image)
    g = build_graph(gt, P.centerline, P.graph)
    rng = np.random.default_rng(7 * cid + 1)
    for tree in [*g.trees, *g.fragments]:
        for b in tree.branches.values():
            if b.s_mm[-1] < 10.0:
                continue
            r_ref = reference_radius(b.s_mm, b.radius_mm, P.lesion)
            for version in ("clean", "jit"):
                pts = b.points_mm
                if version == "jit":
                    _, u, v = rmf(pts)
                    noise = rng.normal(0.0, 0.3, size=(len(pts), 2))
                    pts = pts + noise[:, :1] * u + noise[:, 1:] * v
                    pts = gaussian_filter1d(pts, sigma=2.0, axis=0, mode="nearest")
                res = straighten_points(image, gt, pts, P.cpr)
                img = ((np.clip(res["img"], *HU_CLIP) - HU_CENTER) / HU_SCALE).astype(np.float16)
                mask = np.stack([center_component(m) for m in res["mask"]]).astype(np.uint8)
                r = np.interp(res["s_mm"], b.s_mm, b.radius_mm)
                rr = np.interp(res["s_mm"], b.s_mm, r_ref)
                ds = (1.0 - r / rr).astype(np.float16)
                name = f"{cid:04d}_{b.branch_id}_{version}.npz"
                np.savez_compressed(
                    out / name, img=img, mask=mask, ds=ds, case_id=cid, jitter=version == "jit"
                )


if __name__ == "__main__":
    run_shard(parse_args("lumen dataset"), process_case)
