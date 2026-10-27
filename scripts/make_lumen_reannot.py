"""S25: re-annotated lumen dataset = min(auto half-maximum mask, stored label mask).

A slice whose minimum is empty keeps the stored label mask (flag `fell_back`).
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

from twinloop.cpr import center_component

HU_CENTER, HU_SCALE = 300.0, 500.0
_YY, _XX = np.mgrid[:48, :48]
_DIST_MM = np.hypot(_YY - 23.5, _XX - 23.5) * 0.2


def auto_mask(hu: np.ndarray) -> np.ndarray:
    """Half-maximum region around the centre of one slice (HU)."""
    sm = ndimage.gaussian_filter(hu, 0.6)
    near = _DIST_MM <= 1.0
    peak = np.percentile(sm[near], 90)
    thr = (peak + np.percentile(sm, 15)) / 2.0
    lab, _ = ndimage.label(sm >= thr)
    k = lab[np.unravel_index(np.argmax(np.where(near, sm, -np.inf)), sm.shape)]
    return k > 0 and ndimage.binary_fill_holes(lab == k)


def process(args: tuple[Path, Path]) -> dict:
    src, out = args
    d = np.load(src)
    hu = d["img"].astype(np.float32) * HU_SCALE + HU_CENTER
    label = d["mask"] > 0
    auto = np.stack([auto_mask(s) for s in hu])
    new = np.stack([center_component(a & lb) for a, lb in zip(auto, label, strict=True)])
    fell_back = new.sum((1, 2)) == 0
    new[fell_back] = d["mask"][fell_back] > 0
    np.savez_compressed(
        out / src.name,
        img=d["img"],
        mask=new.astype(np.uint8),
        ds=d["ds"],
        case_id=d["case_id"],
        jitter=d["jitter"],
        mask_label=d["mask"],
        mask_auto=auto.astype(np.uint8),
        fell_back=fell_back,
    )
    return {
        "ds": d["ds"].astype(float),
        "a_label": label.sum((1, 2)),
        "a_auto": auto.sum((1, 2)),
        "a_new": new.sum((1, 2)),
        "fb": fell_back,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/v2/datasets/lumen")
    ap.add_argument("--out", default="/data/v2/datasets/lumen_reannot")
    ap.add_argument("--report", default="reports/v3_lumen_reannot.md")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.src).glob("*.npz"))
    with Pool(a.workers) as p:
        res = p.map(process, [(f, out) for f in files], chunksize=32)
    ds = np.concatenate([r["ds"] for r in res])
    al, au, nw, fb = (
        np.concatenate([r[k] for r in res]) for k in ("a_label", "a_auto", "a_new", "fb")
    )
    bins = [("healthy (ds<0.25)", -9, 0.25), ("mild", 0.25, 0.5), ("moderate", 0.5, 0.7)]
    bins.append(("severe", 0.7, 9))
    lines = [
        "# Re-annotated lumen dataset (S25)",
        "",
        f"{len(files)} files, {len(ds)} slices. Source `{a.src}`, output `{a.out}`.",
        "New mask = pixelwise minimum (AND) of the auto half-maximum mask and the stored label "
        "mask, reduced to its centre component; if that is empty the stored label mask is kept. "
        "Areas are in pixels (0.2 mm). Not validated against any reference; `ds` is copied "
        "from the original files.",
        "",
        "| bin | slices | mean label | mean auto | mean new | new/label | fell back to label |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, lo, hi in bins:
        s = (ds >= lo) & (ds < hi)
        lines.append(
            f"| {name} | {int(s.sum())} | {al[s].mean():.1f} | {au[s].mean():.1f} | "
            f"{nw[s].mean():.1f} | {nw[s].sum() / al[s].sum():.2f} | {int(fb[s].sum())} |"
        )
    lines.append(
        f"| all | {len(ds)} | {al.mean():.1f} | {au.mean():.1f} | {nw.mean():.1f} | "
        f"{nw.sum() / al.sum():.2f} | {int(fb.sum())} |"
    )
    Path(a.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
