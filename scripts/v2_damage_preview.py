"""Damage preview on the smallest-id v2_train cases (S06)."""

import argparse
import collections
import json
import time
from dataclasses import asdict

import numpy as np

from twinloop.config import CenterlineParams
from twinloop.config_v2 import DamageParams, GraphParams
from twinloop.damage import damage_case
from twinloop.graph import build_graph
from twinloop.io import load_image, load_mask, save_nifti
from twinloop.paths import case_image_path, case_label_path, resolve
from twinloop.v2split import load_roles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ids = sorted(c for c, r in load_roles().items() if r == "v2_train")[: a.n]
    dp = DamageParams()
    out_dir = resolve("/data/v2/tmp/damage_preview")
    counts: collections.Counter = collections.Counter()
    skipped = 0
    for k, cid in enumerate(ids):
        t0 = time.time()
        image = load_image(case_image_path(cid))
        gt = load_mask(case_label_path(cid), image)
        g = build_graph(gt, CenterlineParams(), GraphParams())
        rng = np.random.default_rng(1000003 * a.seed + cid)
        _, mask_out, recs = damage_case(image, gt, g, dp, rng)
        counts.update(r.type for r in recs)
        skipped += dp.n_damages_per_case - len(recs)
        print(f"case {cid}: {[r.type for r in recs]} t={time.time() - t0:.0f}s", flush=True)
        if k < 2:
            save_nifti(mask_out, out_dir / f"{cid}_mask_damaged.nii.gz")
            (out_dir / f"{cid}_records.json").write_text(
                json.dumps([asdict(r) for r in recs], indent=1)
            )
    print("counts per type:", dict(counts))
    print("skipped damages:", skipped)


if __name__ == "__main__":
    main()
