"""Real-failure decider dataset (SPEC_V2 §14 'decider_real', S14): v2_realfail cases only.

First-pass masks from scripts/predict_firstpass.py (or the nnU-Net validation folder), labels from
`label_real` against GT. Rows follow decider_synth with seed = -1.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from twinloop.config_v2 import V2Params
from twinloop.graph import build_graph
from twinloop.io import load_image, load_mask
from twinloop.paths import case_image_path, case_label_path, resolve
from twinloop.regions import FEATURE_NAMES, label_real, propose_regions, region_features
from twinloop.v2split import load_roles

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_v2 import firstpass_path  # noqa: E402

P = V2Params()


def process_case(cid: int) -> tuple[int, list[dict] | str]:
    try:
        image = load_image(case_image_path(cid))
        gt = load_mask(case_label_path(cid), image)
        pred = load_mask(firstpass_path(cid), image)
        g_pred = build_graph(pred, P.centerline, P.graph)
        g_gt = build_graph(gt, P.centerline, P.graph)
        regs = propose_regions(g_pred, image, pred, P)
        if not regs:
            return cid, []
        x = region_features(regs, g_pred, image, pred, P)
        labels = label_real(regs, g_pred, pred, gt, g_gt, P)
    except Exception as exc:  # noqa: BLE001 - skip and count the case
        return cid, f"{type(exc).__name__}: {exc}"
    rows = [
        {"case_id": cid, "seed": -1, "region_id": r.region_id, "kind": r.kind,
         **dict(zip(FEATURE_NAMES, x[k].tolist(), strict=True)), "label": labels[k]}
        for k, r in enumerate(regs)
    ]  # fmt: skip
    return cid, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/v2/datasets/decider_real/decider_real.parquet")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    cases = sorted(c for c, r in load_roles().items() if r == "v2_realfail")[: a.limit]
    res = Parallel(n_jobs=a.jobs)(delayed(process_case)(c) for c in cases)
    rows, skipped, empty = [], [], 0
    for cid, r in res:
        if isinstance(r, str):
            skipped.append((cid, r))
        elif not r:
            empty += 1
        else:
            rows += r
    df = pd.DataFrame(rows)
    out = resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    used = df.case_id.nunique() if len(df) else 0
    print(f"cases used {used}, skipped {len(skipped)}, no regions {empty}, rows {len(df)}")
    print("labels:", df.label.value_counts().to_dict() if len(df) else {})
    for cid, err in skipped:
        print("skipped", cid, err)
    return 0


if __name__ == "__main__":
    sys.exit(main())
