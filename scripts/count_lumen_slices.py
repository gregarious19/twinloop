"""Count lumen training slices and lesion runs per severity bin (S23 step 1)."""

import argparse
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from train.train_lumen import list_files
from twinloop.paths import resolve

BINS = ("healthy", "mild", "moderate", "severe")
EDGES = (0.25, 0.5, 0.7)


def bin_idx(ds: float) -> int:
    return int(np.searchsorted(EDGES, ds, side="right"))


def count_file(f: Path) -> dict:
    ds = np.load(f)["ds"].astype(np.float32)
    ds = np.nan_to_num(ds, nan=0.0)
    out = {"jit": f.stem.endswith("_jit"), "slices": np.zeros(4, int), "runs": np.zeros(4, int)}
    b = np.searchsorted(EDGES, ds, side="right")
    out["slices"] += np.bincount(b, minlength=4)
    flag = ds >= EDGES[0]
    edges = np.diff(np.concatenate([[0], flag.astype(np.int8), [0]]))
    for a, z in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0], strict=True):
        out["runs"][bin_idx(float(ds[a:z].max()))] += 1  # a run is binned by its max ds
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/v2/datasets/lumen")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tr, va = list_files(resolve(a.data))
    lines = [
        "# Lumen v3 training (S23)",
        "",
        "## Training-set counts (train part of v2_train, before training)",
        "",
        f"Files: {len(tr)} train, {len(va)} validation. Bins by per-slice `ds` (inscribed): "
        "healthy < 0.25, mild [0.25, 0.5), moderate [0.5, 0.7), severe >= 0.7. A *run* is a "
        "maximal contiguous stretch of slices with ds >= 0.25 inside one file, binned by its "
        "maximum ds.",
        "",
        "| version | measure | " + " | ".join(BINS) + " |",
        "|---|---|" + "---|" * 4,
    ]
    res = Parallel(n_jobs=6)(delayed(count_file)(f) for f in tr)
    for name, sel in (("clean", False), ("jit", True), ("all", None)):
        rows = [r for r in res if sel is None or r["jit"] == sel]
        for key, label in (("slices", "slices"), ("runs", "lesion runs")):
            tot = np.sum([r[key] for r in rows], axis=0) if rows else np.zeros(4, int)
            cells = [
                f"{int(v)}" if not (key == "runs" and i == 0) else "n/a" for i, v in enumerate(tot)
            ]
            lines.append(f"| {name} | {label} | " + " | ".join(cells) + " |")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
