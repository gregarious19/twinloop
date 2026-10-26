"""Create reports/v3_stenosis_set.json once (SPEC_V2 §13c, S21); GT masks only, no predictions."""

import argparse
import json
import sys
from pathlib import Path

from joblib import Parallel, delayed

from twinloop.config import CenterlineParams, LesionParams
from twinloop.config_v2 import GraphParams
from twinloop.graph import build_graph
from twinloop.io import load_image, load_mask
from twinloop.lesions import detect_lesions
from twinloop.paths import case_image_path, case_label_path
from twinloop.v2split import load_roles

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "v3_stenosis_set.json"
REGRESSION = ROOT / "reports" / "v2_regression_set.json"
N_CASES = 30
RULE = (
    "Pool = all v2_holdout and v2_realfail cases minus the 20 cases of "
    "reports/v2_regression_set.json. For each pool case, build the GT graph (inscribed radii, v1 "
    "path) and run v1 detect_lesions; score = number of GT lesions with ds >= 0.5. Take the 30 "
    "highest scores (ties: more lesions with ds >= 0.7, then lower case id), skipping any case "
    "without a first-pass mask (S16 location). Uses GT only: no predictions or model outputs."
)


def case_stats(cid: int) -> dict:
    image = load_image(case_image_path(cid))
    gt = load_mask(case_label_path(cid), image)
    row = {"case_id": cid, "ds": []}
    try:
        graph = build_graph(gt, CenterlineParams(), GraphParams())
        lp = LesionParams()
        for t in graph.trees:
            for b in t.branches.values():
                row["ds"] += [float(les.ds) for les in detect_lesions(b, lp)]
    except ValueError as exc:
        row["error"] = str(exc)
    row["n50"] = sum(d >= 0.5 for d in row["ds"])
    row["n70"] = sum(d >= 0.7 for d in row["ds"])
    return row


def bin_of(ds: float) -> str:
    if ds >= 0.7:
        return "severe"
    return "moderate" if ds >= 0.5 else "mild"


def main() -> None:
    argparse.ArgumentParser().parse_args()
    if OUT.exists():
        print("stenosis set already exists")
        sys.exit(1)
    from eval_v2 import firstpass_path

    excl = set(json.loads(REGRESSION.read_text())["case_ids"])
    roles = load_roles()
    pool = sorted(
        c for c, r in roles.items() if r in ("v2_holdout", "v2_realfail") and c not in excl
    )
    assert all(c % 5 != 0 for c in pool)
    stats = Parallel(n_jobs=6)(delayed(case_stats)(c) for c in pool)
    ranked = sorted(stats, key=lambda s: (-s["n50"], -s["n70"], s["case_id"]))
    chosen = [s for s in ranked if firstpass_path(s["case_id"]).exists()][:N_CASES]
    bins = {"mild": 0, "moderate": 0, "severe": 0}
    for s in chosen:
        for d in s["ds"]:
            if d >= 0.25:
                bins[bin_of(d)] += 1
    ids = [s["case_id"] for s in chosen]
    OUT.write_text(json.dumps({"case_ids": ids, "rule": RULE, "lesions_per_bin": bins}, indent=1))
    print("pool", len(pool), "failed", [s["case_id"] for s in stats if "error" in s])
    print("scores:", {s["case_id"]: (s["n50"], s["n70"]) for s in chosen})
    print("case_ids:", ids)
    print("lesions_per_bin:", bins)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
