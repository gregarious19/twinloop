"""FFR sensitivity to the resting-flow mode on the smallest v2_train cases (S09)."""

import argparse
import dataclasses
import tempfile
from pathlib import Path

import numpy as np

from twinloop.anatomy import segment_anatomy
from twinloop.centerline import extract_trees
from twinloop.config import PipelineParams
from twinloop.config_v2 import AnatomyParams, FlowParams
from twinloop.flow import min_ffr, tree_rest_flows
from twinloop.io import load_image, load_mask
from twinloop.lesions import detect_lesions
from twinloop.paths import case_image_path, case_label_path
from twinloop.solver import solve_tree
from twinloop.v2split import load_roles

OUT = Path(__file__).resolve().parents[1] / "reports" / "v2_sensitivity.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    pp = PipelineParams()
    ids = sorted(c for c, r in load_roles().items() if r == "v2_train")[: a.n]
    rows = []
    for cid in ids:
        image = load_image(case_image_path(cid))
        gt = load_mask(case_label_path(cid), image)
        with tempfile.TemporaryDirectory(dir="/data/v2/tmp") as td:
            anat = segment_anatomy(image, Path(td), AnatomyParams())
        trees = extract_trees(gt, pp.centerline)
        lesions = [
            {b.branch_id: detect_lesions(b, pp.lesion) for b in t.branches.values()} for t in trees
        ]
        myo = anat.myo_volume_ml
        for label, factor, mode in (
            ("diameter_law", 1.0, "diameter_law"),
            ("myocardial_mass", 1.0, "myocardial_mass"),
            ("mass x0.8", 0.8, "myocardial_mass"),
            ("mass x1.2", 1.2, "myocardial_mass"),
        ):
            scaled = dataclasses.replace(anat, myo_volume_ml=None if myo is None else myo * factor)
            try:
                q, used = tree_rest_flows(trees, scaled, pp.solver, FlowParams(flow_mode=mode))
            except ValueError as e:
                rows.append((cid, label, myo, f"n/a ({e})", "n/a"))
                continue
            vals = []
            for t, les in zip(trees, lesions, strict=True):
                sol = solve_tree(t, les, pp.solver, None if q is None else q[t.tree_id])
                vals.append(min_ffr(t, sol))
            vals += [float("nan")] * (2 - len(vals))
            rows.append((cid, label, myo if mode != "diameter_law" else None, *vals[:2]))
            print(cid, label, vals, flush=True)

    def f(v):
        return f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else str(v)

    spans = []
    for cid in ids:
        for k in (3, 4):
            v = [
                r[k] for r in rows if r[0] == cid and isinstance(r[k], float) and not np.isnan(r[k])
            ]
            if v:
                spans.append(max(v) - min(v))
    lines = [
        "# v2 FFR sensitivity to resting-flow mode",
        "",
        f"GT masks of the {len(ids)} smallest-id v2_train cases, highres anatomy, v1 centerlines "
        "and lesions. FFR values are unvalidated.",
        "",
        "| case | mode | myo_ml | min_ffr tree_0 | min_ffr tree_1 |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {c} | {m} | {'' if y is None else f'{y:.1f}'} | {f(a1)} | {f(a2)} |"
        for c, m, y, a1, a2 in rows
    ]
    lines += [
        "",
        f"Across modes the min FFR of a tree changes by up to {max(spans):.2f} "
        f"(median {np.median(spans):.2f}) within one case.",
        "",
        "The aorta position does not enter the 1D model, so an aortic-root shift has no FFR "
        "effect.",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
