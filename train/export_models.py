"""Pick the deployed seed per model, export to models/v2/<name>/ and write the training report
(C07, SPEC_V2 §8.3 and §14)."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from twinloop.config_v2 import GapfillParams
from twinloop.paths import models_dir, v2_dir

# name -> (metrics key, higher is better, SPEC val_metric_name, arch, display)
MODELS = {
    "gapfill": ("best_val_new_dice", True, "new_voxel_dice", "GapfillNet", "Dice"),
    "lumen": ("best_val_req_err_mm", False, "req_err_mm", "LumenNet", "r_eq err mm"),
    "decider": ("val_macro_f1", True, "macro_f1", "HistGradientBoostingClassifier", "macro-F1"),
}
STEPS = {"gapfill": 40000, "lumen": 30000}


def load_runs(name: str) -> dict[int, dict]:
    runs = {}
    for d in sorted((v2_dir() / "runs" / name).glob("seed*")):
        mj = d / "metrics.json"
        if mj.is_file():
            runs[int(d.name[4:])] = json.loads(mj.read_text())
    return runs


def pick(name: str, runs: dict[int, dict]) -> int:
    key, higher, *_ = MODELS[name]
    ok = {s: m[key] for s, m in runs.items() if not m.get("killed") and key in m}
    if not ok:
        raise SystemExit(f"{name}: no finished seed")
    best = max(ok.values()) if higher else min(ok.values())
    return min(s for s, v in ok.items() if v == best)  # ties -> lowest seed


def export_net(name: str, seed: int, runs: dict, commit: str) -> dict:
    from twinloop import nets

    key, _, vname, arch, _ = MODELS[name]
    run = v2_dir() / "runs" / name / f"seed{seed}"
    out = models_dir() / name
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    torch.save(state, out / "model.pt")
    kwargs = {"channels": [16, 32, 64, 128]}
    gp = GapfillParams()
    params = (
        {
            "patch_vox": gp.patch_vox,
            "patch_spacing_mm": gp.patch_spacing_mm,
            "hu_clip": list(gp.hu_clip),
            "hu_center": gp.hu_center,
            "hu_scale": gp.hu_scale,
            "seed_sigma_mm": gp.seed_sigma_mm,
        }
        if name == "gapfill"
        else {"hu_center": 300.0, "hu_scale": 500.0, "window_slices": 128, "size_px": 48}
    )
    cfg = {
        "name": name, "arch": arch, "arch_kwargs": kwargs, "params": params, "seed": seed,
        "git_commit": commit, "trained_on": "v2_train synthetic", "val_metric_name": vname,
        "val_metric": runs[seed][key], "torch_version": torch.__version__,
    }  # fmt: skip
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "metrics.json").write_text(json.dumps(runs[seed], indent=2))
    model = getattr(nets, arch)(channels=tuple(kwargs["channels"]))
    model.load_state_dict(torch.load(out / "model.pt", map_location="cpu", weights_only=True))
    x = torch.zeros(1, 3, 64, 64, 64) if name == "gapfill" else torch.zeros(1, 8, 1, 48, 48)
    with torch.no_grad():
        y = model.eval()(x)
    assert torch.isfinite(y).all()
    return cfg


def export_decider(seed: int, runs: dict, commit: str) -> dict:
    import joblib
    import sklearn

    from twinloop.regions import ACTIONS, FEATURE_NAMES

    run = v2_dir() / "runs" / "decider" / f"seed{seed}"
    out = models_dir() / "decider"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "decider.joblib", out / "decider.joblib")
    cfg = {
        "name": "decider", "arch": MODELS["decider"][3], "feature_names": FEATURE_NAMES,
        "actions": list(ACTIONS), "seed": seed, "sklearn_version": sklearn.__version__,
        "val_metric_name": "macro_f1", "val_metric": runs[seed]["val_macro_f1"],
        "trained_on": "v2_train synthetic", "git_commit": commit,
        "torch_version": torch.__version__,
    }  # fmt: skip
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "metrics.json").write_text(json.dumps(runs[seed], indent=2))
    clf = joblib.load(out / "decider.joblib")
    p = clf.predict_proba(np.zeros((1, len(FEATURE_NAMES))))
    assert p.shape[0] == 1 and p.shape[1] <= len(ACTIONS)
    return cfg


def report(runs_all: dict, deployed: dict, date: str, commit: str) -> str:
    L = [
        "# TwinLoop v2 training report",
        f"Date: {date}. Commit: {commit}. Hardware: DGX Spark (all training ran on SPARK).",
        "Validation = held-out `v2_train` cases only; no holdout/realfail/test case used.",
        "",
        "| Model | Seed | Val metric | Steps | Hours | Deployed |",
        "|---|---|---|---|---|---|",
    ]
    for name, (key, _, _, _, disp) in MODELS.items():
        runs = runs_all[name]
        for s, m in sorted(runs.items()):
            val = f"{disp} {m[key]:.4f}" if key in m else "n/a"
            status = " (KILLED)" if m.get("killed") else ""
            steps = m.get("final_step", STEPS.get(name, "n/a"))
            hrs = f"{m['hours']:.2f}" if "hours" in m else "n/a"
            dep = "yes" if s == deployed[name] else "no"
            L.append(f"| {name} | {s} | {val}{status} | {steps} | {hrs} | {dep} |")
        v = [m[key] for m in runs.values() if key in m and not m.get("killed")]
        L.append(f"| {name} | mean ± SD | {np.mean(v):.4f} ± {np.std(v, ddof=1):.4f} | | | |")
    L += [
        "",
        "Hours are wall-clock as logged; the gap-filler seed 0 figure includes a pause "
        "(docker pause by the owner), so hours are not comparable between seeds.",
        "",
        f"## Decider per-class table (validation, deployed seed {deployed['decider']})",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    pc = runs_all["decider"][deployed["decider"]]["per_class"]
    for c in ["keep", "extend", "add", "remove", "restore", "stop"]:
        r = pc[c]
        L.append(
            f"| {c} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} "
            f"| {int(r['support'])} |"
        )
    L += [
        "",
        "`add` has 24 validation rows and very low precision; treat its predictions as weak.",
        "",
        "Caveats: one deployed seed per network; synthetic damage is not the same as real "
        "segmentation errors.",
    ]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--git-commit", default="unknown")
    ap.add_argument("--date", required=True)
    ap.add_argument("--report", default="reports/v2_training.md")
    a = ap.parse_args()
    runs_all = {n: load_runs(n) for n in MODELS}
    deployed = {n: pick(n, runs_all[n]) for n in MODELS}
    for n in ("gapfill", "lumen"):
        export_net(n, deployed[n], runs_all[n], a.git_commit)
    export_decider(deployed["decider"], runs_all["decider"], a.git_commit)
    Path(a.report).write_text(report(runs_all, deployed, a.date, a.git_commit))
    print("deployed seeds:", deployed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
