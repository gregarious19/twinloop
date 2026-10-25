"""Decider training rows from synthetic damage (SPEC_V2 §14 'decider_synth')."""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from train.shards import parse_args, run_shard, shard_cases
from twinloop.config_v2 import V2Params
from twinloop.damage import damage_case
from twinloop.graph import build_graph
from twinloop.io import load_image, load_mask
from twinloop.paths import case_image_path, case_label_path
from twinloop.regions import FEATURE_NAMES, label_synthetic, propose_regions, region_features

SEEDS = (0, 1)
P = V2Params()


def process_case(cid: int, out: Path) -> None:
    image = load_image(case_image_path(cid))
    gt = load_mask(case_label_path(cid), image)
    g0 = build_graph(gt, P.centerline, P.graph)
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(1000003 * seed + cid)
        img2, mask2, recs = damage_case(image, gt, g0, P.damage, rng)
        try:
            g = build_graph(mask2, P.centerline, P.graph)
        except Exception:  # noqa: BLE001 - skip the sample if the graph cannot be built
            continue
        regs = propose_regions(g, img2, mask2, P)
        if not regs:
            continue
        x = region_features(regs, g, img2, mask2, P)
        labels, _ = label_synthetic(regs, recs, P)
        for k, r in enumerate(regs):
            rows.append(
                {
                    "case_id": cid,
                    "seed": seed,
                    "region_id": r.region_id,
                    "kind": r.kind,
                    **dict(zip(FEATURE_NAMES, x[k].tolist(), strict=True)),
                    "label": labels[k],
                }
            )
    cdir = out / "_cases"
    cdir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(cdir / f"{cid:04d}.parquet", index=False)


def main() -> None:
    a = parse_args("decider dataset")
    run_shard(a, process_case)
    cases = shard_cases("v2_train", a.shard, a.nshards)[: a.limit]
    from twinloop.paths import resolve

    out = resolve(a.out)
    parts = [
        pd.read_parquet(p) for c in cases if (p := out / "_cases" / f"{c:04d}.parquet").exists()
    ]
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(
            out / f"shard_{a.shard:02d}.parquet", index=False
        )
        print(f"wrote shard_{a.shard:02d}.parquet at {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
