"""Export the best lumen-v3 seed to models/v2/lumen_v3/ and append the report section (S23)."""

import argparse
import json
import sys
from pathlib import Path

import torch

from twinloop.paths import models_dir, v2_dir

KEY = "best_val_modplus_rins_err_mm"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--git-commit", default="unknown")
    ap.add_argument("--frozen", default="reports/v3_lumen_frozen.json")
    ap.add_argument("--report", default="reports/v2_training.md")
    ap.add_argument("--detail", default="reports/v3_lumen_training.md")
    a = ap.parse_args()
    root = v2_dir() / "runs" / "lumen_v3"
    runs = {int(d.name[4:]): json.loads((d / "metrics.json").read_text())
            for d in sorted(root.glob("seed*")) if (d / "metrics.json").is_file()}  # fmt: skip
    ok = {s: m[KEY] for s, m in runs.items() if not m.get("killed") and KEY in m}
    best = min(ok.values())
    seed = min(s for s, v in ok.items() if v == best)
    frozen = json.loads(Path(a.frozen).read_text())
    out = models_dir() / "lumen_v3"
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(root / f"seed{seed}" / "best.pt", map_location="cpu", weights_only=True)
    torch.save(state, out / "model.pt")
    cfg = {
        "name": "lumen_v3", "arch": "LumenNet", "arch_kwargs": {"channels": [16, 32, 64, 128]},
        "params": {"hu_center": 300.0, "hu_scale": 500.0, "window_slices": 128, "size_px": 48},
        "seed": seed, "git_commit": a.git_commit, "trained_on": "v2_train synthetic",
        "val_metric_name": "modplus_rins_err_mm", "val_metric": best,
        "inference_settings": {k: frozen[k] for k in ("radius_out", "fallback", "sigma")},
        "torch_version": torch.__version__,
    }  # fmt: skip
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "metrics.json").write_text(json.dumps(runs[seed], indent=2))
    old = json.loads((root / "old_reference" / "old_model_validation.json").read_text())
    names = ("healthy", "mild", "modplus")
    L = ["", "## Lumen v3 (S23)", "",
         "Validation = `jit` branch files of the v2_train validation cases (case_id % 10 == 3); "
         "mean |r(pred) − r(GT)| in mm over slices with a GT lumen, per severity bin of the slice "
         "(healthy ds < 0.25, mild [0.25, 0.5), moderate+ ≥ 0.5). Model selection = moderate+ "
         "r_ins. Kill criterion (moderate+ r_ins > 0.5 mm at 25 % of steps) never triggered.", "",
         "| model | seed | best step | r_ins healthy | r_ins mild | r_ins mod+ | r_eq healthy | "
         "r_eq mild | r_eq mod+ | hours | deployed as lumen_v3 |", "|" + "---|" * 11]  # fmt: skip
    L.append("| old `lumen` (reference) | 2 | 30000 | " + " | ".join(
        f"{old[f'ins_{n}']:.3f}" for n in names) + " | " + " | ".join(
        f"{old[f'eq_{n}']:.3f}" for n in names) + " | 1.28 | n/a |")  # fmt: skip
    for s, m in sorted(runs.items()):
        if m.get("killed"):
            L.append(f"| lumen v3 | {s} | KILLED at {m['kill_step']} | | | | | | | "
                     f"{m['hours']:.2f} | no |")  # fmt: skip
            continue
        v = m["best_val"]
        L.append(f"| lumen v3 | {s} | {m['best_step']} | " + " | ".join(
            f"{v[f'ins_{n}']:.3f}" for n in names) + " | " + " | ".join(
            f"{v[f'eq_{n}']:.3f}" for n in names) + f" | {m['hours']:.2f} | "
            f"{'yes' if s == seed else 'no'} |")  # fmt: skip
    n = runs[seed]["best_val"]
    L += ["", f"Validation slices (healthy / mild / mod+): {n['n_healthy']} / {n['n_mild']} / "
          f"{n['n_modplus']}; the moderate+ figure rests on few slices from few lesions.", "",
          "Frozen inference settings (validation cases only, see "
          "`reports/v3_lumen_freeze_grid.csv`): "
          f"radius_out={frozen['radius_out']}, fallback={frozen['fallback']}, "
          f"sigma={frozen['sigma']}. Moderate+ |error| at GT lesion minima on GT centerlines "
          f"shifted {frozen['shift_mm']} mm: {frozen['modplus_mean_abs_err_mm']:.3f} mm "
          f"(n = {frozen['n_modplus']} lesions, {frozen['cases']} cases), always an "
          "over-estimate of the GT-graph radius. This is far above the slice-level error "
          "(~0.4 mm); the ruler differs (GT-graph radius is the 3-D EDT value).", ""]  # fmt: skip
    with open(a.report, "a") as f:
        f.write("\n".join(L) + "\n")
    with open(a.detail, "a") as f:
        f.write("\n".join(L) + "\n")
    print("exported seed", seed, best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
