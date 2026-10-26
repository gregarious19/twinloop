"""S23 step 4: freeze lumen-v3 inference settings on validation cases only.

Validation cases = v2_train ids with id % 10 == 3 (never v2_holdout / v2_realfail / test). They have
no first-pass mask, so the GT centerlines shifted 0.5 mm perpendicular (S18 rule) stand in for
real centerlines. radius_out is "ins". For each (fallback, sigma) the refined branch radius at every
GT lesion minimum (ds >= 0.25, v1 detect_lesions, inscribed GT graph) is compared with the GT graph
radius there: error = |r_refined - r_gt| in mm; selection = mean over moderate+ lesions (ds >= 0.5).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from twinloop.config_v2 import V2Params
from twinloop.cpr import straighten
from twinloop.graph import build_graph
from twinloop.io import load_image, load_mask
from twinloop.lesions import detect_lesions
from twinloop.lumen import HU_CENTER, HU_CLIP, HU_SCALE, MIN_BRANCH_MM, _predict_logits, slice_radii
from twinloop.nets import LumenNet
from twinloop.paths import case_image_path, case_label_path
from twinloop.v2split import load_roles

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_v2 import shifted  # noqa: E402

P = V2Params()
FALLBACKS = ("v1", "nearest", "interp")
SIGMAS = (0.0, 0.5, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--shift-mm", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    device = "cuda"
    model = LumenNet().to(device)
    model.load_state_dict(torch.load(a.weights, map_location=device, weights_only=True))
    model.eval()
    ids = sorted(c for c, r in load_roles().items() if r == "v2_train" and c % 10 == 3)
    if a.limit:
        ids = ids[: a.limit]
    rows, miss = [], {k: [0, 0] for k in FALLBACKS}
    for cid in ids:
        image = load_image(case_image_path(cid))
        gt = load_mask(case_label_path(cid), image)
        try:
            g = build_graph(gt, P.centerline, P.graph)
        except ValueError:
            continue
        sh = shifted(g.trees, a.shift_mm)
        for t_gt, t_sh in zip(g.trees, sh, strict=True):
            for bid, b_gt in t_gt.branches.items():
                b = t_sh.branches[bid]
                les = [x for x in detect_lesions(b_gt, P.lesion) if x.ds >= 0.25]
                if not les or b.s_mm[-1] < MIN_BRANCH_MM:
                    continue
                res = straighten(image, None, b, P.cpr)
                x = ((np.clip(res["img"], *HU_CLIP) - HU_CENTER) / HU_SCALE).astype(np.float32)
                prob = 1.0 / (1.0 + np.exp(-_predict_logits(model, x, P.cpr, device)))
                r_v1 = np.interp(res["s_mm"], b.s_mm, b.radius_mm)
                for fb in FALLBACKS:
                    r0, hit, used = slice_radii(prob, r_v1, P.cpr, "ins", fb)
                    miss[fb][0] += int((~hit).sum())
                    miss[fb][1] += len(hit)
                    for sg in SIGMAS:
                        r = ndimage.gaussian_filter1d(r0, sg, mode="nearest") if sg > 0 else r0
                        r_b = np.interp(b.s_mm, res["s_mm"], r)
                        for x_ in les:
                            rows.append(dict(
                                case_id=cid, branch_id=bid, lesion_id=x_.lesion_id, ds=x_.ds,
                                bin="modplus" if x_.ds >= 0.5 else "mild", fallback=fb, sigma=sg,
                                r_gt=b_gt.radius_mm[x_.min_idx], r_pred=r_b[x_.min_idx],
                                v1_fraction=float(used.mean()),
                            ))  # fmt: skip
        print(f"case {cid}: {len(rows)} rows", flush=True)
    df = pd.DataFrame(rows)
    df["abs_err"] = (df.r_pred - df.r_gt).abs()
    df["err"] = df.r_pred - df.r_gt
    grid = (
        df.groupby(["fallback", "sigma", "bin"])
        .agg(n=("abs_err", "size"), mean_abs_err=("abs_err", "mean"), mean_err=("err", "mean"))
        .reset_index()
    )
    grid.to_csv(a.out_csv, index=False)
    mp = grid[grid["bin"] == "modplus"].sort_values(["mean_abs_err", "sigma", "fallback"])
    best = mp.iloc[0]
    out = {
        "radius_out": "ins",
        "fallback": best.fallback,
        "sigma": float(best.sigma),
        "modplus_mean_abs_err_mm": float(best.mean_abs_err),
        "n_modplus": int(best.n),
        "cases": len(ids),
        "shift_mm": a.shift_mm,
        "centre_miss_fraction": {k: v[0] / max(1, v[1]) for k, v in miss.items()},
        "grid": json.loads(grid.to_json(orient="records")),
    }
    Path(a.out_json).write_text(json.dumps(out, indent=1))
    print(grid.to_string())
    print("CHOSEN:", out["fallback"], out["sigma"], out["modplus_mean_abs_err_mm"])


if __name__ == "__main__":
    main()
