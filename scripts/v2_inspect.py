"""Inspect one case: anatomy -> graph -> ostia -> labels -> coverage (S04)."""

import argparse
import json
import time

from twinloop.anatomy import (
    Anatomy,
    coverage_check,
    find_ostia,
    label_branches,
    segment_anatomy,
)
from twinloop.config import CenterlineParams
from twinloop.config_v2 import AnatomyParams, GraphParams
from twinloop.graph import build_graph, graph_summary
from twinloop.io import load_image, load_mask
from twinloop.paths import case_image_path, case_label_path, resolve


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--case-id", type=int, required=True)
    ap_.add_argument("--mask-path")
    ap_.add_argument("--no-anatomy", action="store_true")
    ap_.add_argument("--out", required=True)
    a = ap_.parse_args()

    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ap = AnatomyParams()
    image = load_image(case_image_path(a.case_id))
    mask_path = resolve(a.mask_path) if a.mask_path else case_label_path(a.case_id)
    mask = load_mask(mask_path, image)

    t0 = time.time()
    if a.no_anatomy:
        anat = Anatomy(None, None, None, {}, {}, ["anatomy skipped"])
        anat.qc = {"aorta_present": False, "aorta_connected": False, "myo_present": False}
    else:
        anat = segment_anatomy(image, out / f"{a.case_id}_ts", ap)
    t_anat = time.time() - t0

    t1 = time.time()
    g = build_graph(mask, CenterlineParams(), GraphParams())
    find_ostia(g, anat, ap)
    anat.qc.setdefault("ostia_ok", False)
    label_branches(g, anat, ap)
    flags = coverage_check(g, ap)
    t_graph = time.time() - t1

    summary = {
        "case_id": a.case_id,
        "graph": graph_summary(g),
        "qc": anat.qc,
        "qc_messages": anat.qc_messages,
        "myo_volume_ml": anat.myo_volume_ml,
        "aorta_volume_ml": anat.aorta_volume_ml,
        "coverage": flags,
        "timings_s": {"anatomy": round(t_anat, 1), "graph": round(t_graph, 1)},
    }
    (out / f"{a.case_id}.json").write_text(json.dumps(summary, indent=1))
    labels = sorted(set(g.labels.values()))
    print(
        f"OK case={a.case_id} trees={len(g.trees)} fragments={len(g.fragments)} "
        f"labels={labels} flags={len(flags)} t={t_anat + t_graph:.0f}"
    )


if __name__ == "__main__":
    main()
