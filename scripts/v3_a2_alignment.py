"""A2 (SPEC_V3 §6.1): are the stored lumen-dataset labels aligned with the half-max mask?

    python scripts/v3_a2_alignment.py

Writes reports/v3_A2_alignment.{md,json}. Nothing is retrained or changed.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

from twinloop.config import CenterlineParams, LesionParams
from twinloop.config_v2 import GraphParams, V2Params
from twinloop.cpr import center_component, straighten
from twinloop.graph import build_graph
from twinloop.halfmax import auto_mask
from twinloop.io import load_image, load_mask
from twinloop.lesions import reference_radius
from twinloop.paths import case_image_path, case_label_path, data_root
from twinloop.v2split import load_roles

REPO = Path(__file__).resolve().parents[1]
HU_CENTER, HU_SCALE = 300.0, 500.0
N_FILES, MAX_PER_FILE, ALIGNED_MIN = 200, 50, 0.60
TRANSFORMS = {
    "identity": lambda a: a,
    "flip_axis0": lambda a: a[:, ::-1, :],
    "flip_axis1": lambda a: a[:, :, ::-1],
    "transpose": lambda a: a.transpose(0, 2, 1),
    "rot90_k1": lambda a: np.rot90(a, 1, axes=(1, 2)),
    "rot90_k2": lambda a: np.rot90(a, 2, axes=(1, 2)),
    "rot90_k3": lambda a: np.rot90(a, 3, axes=(1, 2)),
    "transpose_flip_axis0": lambda a: a.transpose(0, 2, 1)[:, ::-1, :],
}


def iou_stack(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = (a & b).sum((1, 2))
    union = (a | b).sum((1, 2))
    out = np.full(len(a), np.nan)
    ok = union > 0  # two empty masks are skipped
    out[ok] = inter[ok] / union[ok]
    return out


def shifted(lab: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return ndimage.shift(lab.astype(np.uint8), (0, dy, dx), order=0, cval=0) > 0


def dataset_slices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = data_root() / "v2" / "datasets" / "lumen"
    files = sorted(f.name for f in d.glob("*.npz"))
    pick = np.random.default_rng(0).choice(files, N_FILES, replace=False)
    autos, labs, jit = [], [], []
    for name in pick:
        z = np.load(d / name)
        ds = z["ds"].astype(np.float32)
        healthy = np.nonzero(~(ds >= 0.25))[0][:MAX_PER_FILE]  # NaN counts as healthy
        hu = z["img"][healthy].astype(np.float32) * HU_SCALE + HU_CENTER
        autos += [auto_mask(s) for s in hu]
        labs += list(z["mask"][healthy] > 0)
        jit += [bool(z["jitter"])] * len(healthy)
    return np.array(autos), np.array(labs), np.array(jit)


def fresh_reference() -> tuple[float, int, int]:
    roles = load_roles()
    dev = sorted(c for c, r in roles.items() if r == "v2_train" and c % 10 == 3)[:5]
    cp, lp = V2Params().cpr, LesionParams()
    autos, labs, nb = [], [], 0
    for cid in dev:
        image = load_image(case_image_path(cid))
        gt = load_mask(case_label_path(cid), image)
        g = build_graph(gt, CenterlineParams(), GraphParams())
        branches = [b for t in g.trees for b in t.branches.values() if b.s_mm[-1] >= 10.0][:4]
        for b in branches:
            nb += 1
            res = straighten(image, gt, b, cp)
            ref = reference_radius(b.s_mm, b.radius_mm, lp)
            ds = 1.0 - np.interp(res["s_mm"], b.s_mm, b.radius_mm) / np.interp(
                res["s_mm"], b.s_mm, ref
            )
            for k in np.nonzero(~(ds >= 0.25))[0][:MAX_PER_FILE]:
                autos.append(auto_mask(res["img"][k]))
                labs.append(center_component(res["mask"][k] > 0.5))
    iou = iou_stack(np.array(autos), np.array(labs))
    return float(np.nanmedian(iou)), nb, len(autos)


def main() -> int:
    auto, lab, jit = dataset_slices()
    print(f"{len(auto)} healthy slices from {N_FILES} files", flush=True)
    results = []  # (median, transform, dy, dx, median_clean, median_jit)
    for name, fn in TRANSFORMS.items():
        t = np.ascontiguousarray(fn(lab))
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                iou = iou_stack(auto, shifted(t, dy, dx))
                results.append(
                    (
                        float(np.nanmedian(iou)),
                        name,
                        dy,
                        dx,
                        float(np.nanmedian(iou[~jit])),
                        float(np.nanmedian(iou[jit])),
                    )
                )
    ident = next(r for r in results if r[1] == "identity" and r[2] == 0 and r[3] == 0)
    best = max(results, key=lambda r: r[0])
    fresh, n_br, n_fs = fresh_reference()
    aligned = ident[0] >= ALIGNED_MIN
    if aligned:
        verdict = f"aligned (identity median IoU {ident[0]:.3f} >= {ALIGNED_MIN})"
    elif best[0] >= ALIGNED_MIN:
        verdict = (
            f"misaligned: best transform <{best[1]}, shift ({best[2]}, {best[3]})> "
            f"gives {best[0]:.3f}"
        )
    else:
        verdict = f"shape mismatch (identity {ident[0]:.3f}, best {best[0]:.3f} < {ALIGNED_MIN})"
    out = {
        "identity_median_iou": ident[0],
        "best_transform": best[1],
        "best_shift": [best[2], best[3]],
        "best_median_iou": best[0],
        "fresh_identity_median_iou": fresh,
        "dataset_aligned": bool(aligned),
    }
    (REPO / "reports" / "v3_A2_alignment.json").write_text(json.dumps(out, indent=2) + "\n")
    top = sorted(results, key=lambda r: -r[0])[:8]
    md = [
        "# v3 A2 — lumen dataset alignment",
        "",
        f"{len(auto)} healthy slices (ds < 0.25 or NaN, first {MAX_PER_FILE} per file) from "
        f"{N_FILES} files sampled with `default_rng(0)`; "
        f"{int((~jit).sum())} from clean files, {int(jit.sum())} from jit files. "
        "IoU(half-max `auto_mask`, T(stored label)).",
        "",
        f"**Verdict: {verdict}**",
        "",
        "| set | median IoU, identity / no shift | n |",
        "|---|---|---|",
        f"| dataset, all | {ident[0]:.3f} | {len(auto)} |",
        f"| dataset, clean | {ident[4]:.3f} | {int((~jit).sum())} |",
        f"| dataset, jit | {ident[5]:.3f} | {int(jit.sum())} |",
        f"| fresh CPR of GT ({n_br} branches, 5 dev cases) | {fresh:.3f} | {n_fs} |",
        "",
        "Best (transform, shift) combinations over 8 transforms × 7×7 shifts:",
        "",
        "| transform | shift (dy, dx) | median IoU | clean | jit |",
        "|---|---|---|---|---|",
        *[f"| {r[1]} | ({r[2]}, {r[3]}) | {r[0]:.3f} | {r[4]:.3f} | {r[5]:.3f} |" for r in top],
        "",
        "Nothing was retrained or changed.",
    ]
    (REPO / "reports" / "v3_A2_alignment.md").write_text("\n".join(md) + "\n")
    print(verdict)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
