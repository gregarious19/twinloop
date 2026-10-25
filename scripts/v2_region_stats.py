"""Design check for regions and synthetic labels on the smallest v2_train cases (S08)."""

import argparse
import collections
import time

import numpy as np
import pandas as pd

from twinloop.config_v2 import V2Params
from twinloop.damage import damage_case
from twinloop.graph import build_graph
from twinloop.io import load_image, load_mask
from twinloop.paths import case_image_path, case_label_path, resolve
from twinloop.regions import (
    FEATURE_NAMES,
    match_records,
    propose_regions,
    region_features,
)
from twinloop.v2split import load_roles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = V2Params()
    ids = sorted(c for c, r in load_roles().items() if r == "v2_train")[: a.n]
    matched: collections.Counter = collections.Counter()
    total: collections.Counter = collections.Counter()
    labels: collections.Counter = collections.Counter()
    rows, n_regions, secs = [], [], []
    for cid in ids:
        t0 = time.time()
        image = load_image(case_image_path(cid))
        gt = load_mask(case_label_path(cid), image)
        g0 = build_graph(gt, P.centerline, P.graph)
        rng = np.random.default_rng(1000003 * a.seed + cid)
        img2, mask2, recs = damage_case(image, gt, g0, P.damage, rng)
        g = build_graph(mask2, P.centerline, P.graph)
        regs = propose_regions(g, img2, mask2, P)
        x = region_features(regs, g, img2, mask2, P)
        matches = match_records(regs, recs)
        lab = ["keep"] * len(regs)
        for r, m in zip(recs, matches, strict=True):
            total[r.type] += 1
            if m is not None:
                matched[r.type] += 1
                lab[m] = r.action
        labels.update(lab)
        n_regions.append(len(regs))
        secs.append(time.time() - t0)
        for k, r in enumerate(regs):
            rows.append({"case_id": cid, "region_id": r.region_id, "kind": r.kind, "label": lab[k],
                         **dict(zip(FEATURE_NAMES, x[k], strict=True))})  # fmt: skip
        print(f"case {cid}: regions={len(regs)} records={len(recs)} t={secs[-1]:.0f}s", flush=True)
    out = resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    for t in total:
        print(f"{t}: matched {matched[t]}/{total[t]}")
    print("label counts:", dict(labels))
    print(f"mean regions per case: {np.mean(n_regions):.1f}; seconds per case: {np.mean(secs):.1f}")


if __name__ == "__main__":
    main()
