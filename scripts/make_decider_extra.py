"""Extra synthetic decider rows for the rare classes add / stop / extend (D-V2-S25).

Same schema as `decider_synth` (train/make_decider_data.py), written to its own folder.
Differences: damage seeds 100.. each use a class profile (type weights tilted to one rare class)
and a thin-vessel bias (real first-pass misses are thin: median r 1.04 mm vs 1.45 mm in the
synthetic extend rows). Only `v2_train` cases. `keep` rows are subsampled (other labels are kept).
"""

import argparse
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from twinloop.config_v2 import V2Params
from twinloop.damage import damage_case
from twinloop.graph import build_graph
from twinloop.io import load_image, load_mask
from twinloop.paths import case_image_path, case_label_path, resolve
from twinloop.regions import FEATURE_NAMES, label_synthetic, propose_regions, region_features
from twinloop.v2split import load_roles

# type order: gap, truncate, delete_branch, shrink, false_branch, occlusion
PROFILES = {
    "add": (0.05, 0.05, 0.70, 0.05, 0.05, 0.10),
    "stop": (0.05, 0.15, 0.05, 0.00, 0.00, 0.75),
    "extend": (0.35, 0.45, 0.05, 0.05, 0.00, 0.10),
}
SEED_PROFILE = {
    100: "add", 101: "stop", 102: "extend", 103: "add", 104: "stop", 105: "extend",
    106: "add", 107: "add",
}  # fmt: skip
# real add sites are on thick parent vessels (r ~2 mm), real extend endings are thin
THIN_BIAS = {"add": 0.0, "stop": 0.9, "extend": 0.9}
THIN_MAX_MM = 1.1  # real extend rows: median r_local 1.04 mm
P = V2Params()


def process_case(cid: int, seeds: list[int], keep_frac: float) -> tuple[int, list[dict] | str]:
    try:
        image = load_image(case_image_path(cid))
        gt = load_mask(case_label_path(cid), image)
        g0 = build_graph(gt, P.centerline, P.graph)
        rows: list[dict] = []
        for seed in seeds:
            dp = replace(
                P.damage,
                type_weights=PROFILES[SEED_PROFILE[seed]],
                thin_bias_prob=THIN_BIAS[SEED_PROFILE[seed]],
                thin_max_radius_mm=THIN_MAX_MM,
            )
            rng = np.random.default_rng(1000003 * seed + cid)
            img2, mask2, recs = damage_case(image, gt, g0, dp, rng)
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
                if labels[k] == "keep" and rng.random() >= keep_frac:
                    continue
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
        return cid, rows
    except Exception as e:  # noqa: BLE001 - one bad case must not stop the run
        return cid, f"{type(e).__name__}: {e}"[:300]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output folder (under the data root)")
    ap.add_argument("--seeds", type=int, nargs="+", default=sorted(SEED_PROFILE))
    ap.add_argument("--keep-frac", type=float, default=0.25)
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N v2_train cases only")
    a = ap.parse_args()
    unknown = set(a.seeds) - set(SEED_PROFILE)
    if unknown:
        sys.exit(f"unknown seeds {sorted(unknown)}; known: {sorted(SEED_PROFILE)}")

    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = sorted(c for c, r in load_roles().items() if r == "v2_train")[: a.limit]
    results = Parallel(n_jobs=a.nproc)(
        delayed(process_case)(c, a.seeds, a.keep_frac) for c in cases
    )
    rows, errors = [], []
    for cid, res in results:
        (errors.append((cid, res)) if isinstance(res, str) else rows.extend(res))
    df = pd.DataFrame(rows)
    df.to_parquet(out / "decider_extra.parquet", index=False)
    if errors:
        pd.DataFrame(errors, columns=["case_id", "error"]).to_csv(out / "errors.csv", index=False)
    print(f"cases={len(cases)} errors={len(errors)} rows={len(df)}", file=sys.stderr)
    if len(df):
        print(pd.crosstab(df.kind, df.label).to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
