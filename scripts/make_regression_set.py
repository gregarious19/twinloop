"""Create reports/v2_regression_set.json once (SPEC_V2 §13, S05); v2_holdout images and GT only."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from joblib import Parallel, delayed
from scipy import ndimage

from twinloop.config import CenterlineParams, LesionParams
from twinloop.config_v2 import GraphParams
from twinloop.graph import build_graph
from twinloop.io import load_image, load_mask
from twinloop.lesions import detect_lesions
from twinloop.paths import case_image_path, case_label_path
from twinloop.v2split import load_roles

OUT = Path(__file__).resolve().parents[1] / "reports" / "v2_regression_set.json"
RULE = (
    "From v2_holdout: calcium = count of voxels with HU >= 800 inside (GT dilated by 4 vox) minus "
    "GT; n_lesions = v1 detect_lesions on the GT graph. Take the 10 highest calcium, then the 5 "
    "highest n_lesions not yet chosen, then the 5 lowest remaining case ids "
    "(ties broken by lower case id)."
)


def case_stats(cid: int) -> dict:
    image = load_image(case_image_path(cid))
    gt = load_mask(case_label_path(cid), image)
    g = sitk.GetArrayViewFromImage(gt) > 0
    hu = sitk.GetArrayViewFromImage(image)
    ring = ndimage.binary_dilation(g, structure=np.ones((3, 3, 3)), iterations=4) & ~g
    calcium = int((hu[ring] >= 800).sum())
    try:
        graph = build_graph(gt, CenterlineParams(), GraphParams())
        lp = LesionParams()
        n_les = sum(len(detect_lesions(b, lp)) for t in graph.trees for b in t.branches.values())
    except ValueError:
        n_les = 0
    return {"case_id": cid, "calcium": calcium, "n_lesions": n_les}


def main() -> None:
    argparse.ArgumentParser().parse_args()
    if OUT.exists():
        print("regression set already exists")
        sys.exit(1)
    ids = sorted(c for c, r in load_roles().items() if r == "v2_holdout")
    stats = Parallel(n_jobs=6)(delayed(case_stats)(c) for c in ids)
    chosen = [s["case_id"] for s in sorted(stats, key=lambda s: (-s["calcium"], s["case_id"]))[:10]]
    rest = [s for s in stats if s["case_id"] not in chosen]
    for s in sorted(rest, key=lambda s: (-s["n_lesions"], s["case_id"]))[:5]:
        chosen.append(s["case_id"])
    rest = sorted(s["case_id"] for s in stats if s["case_id"] not in chosen)
    chosen += rest[:5]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"case_ids": chosen, "rule": RULE}, indent=1))
    print(
        "stats:",
        {s["case_id"]: (s["calcium"], s["n_lesions"]) for s in stats if s["case_id"] in chosen},
    )
    print("case_ids:", chosen)


if __name__ == "__main__":
    main()
