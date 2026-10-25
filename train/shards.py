"""Shard helpers shared by the dataset generators (SPEC_V2 §14, C03)."""

import os

# several generator processes share the CPU cores with the running nnU-Net training
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import csv  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402

from twinloop.paths import resolve  # noqa: E402
from twinloop.v2split import load_roles  # noqa: E402


def shard_cases(role: str = "v2_train", shard: int = 0, nshards: int = 20) -> list[int]:
    """Sorted ids of `role` with index % nshards == shard."""
    ids = sorted(c for c, r in load_roles().items() if r == role)
    return [c for k, c in enumerate(ids) if k % nshards == shard]


def parse_args(desc: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N cases")
    return ap.parse_args()


def run_shard(a: argparse.Namespace, process_case: Callable[[int, Path], None]) -> None:
    """Run `process_case(case_id, out_dir)` over the shard; resumable via per-case markers."""
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = shard_cases("v2_train", a.shard, a.nshards)
    if a.limit is not None:
        cases = cases[: a.limit]
    n_err = 0
    for cid in cases:
        marker = out / f"_case_{cid:04d}.done"
        if marker.exists():
            continue
        t0 = time.time()
        try:
            process_case(cid, out)
            marker.write_text("ok\n")
            print(f"case {cid} ok {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001 - one bad case must not stop the shard
            n_err += 1
            with open(out / f"errors_shard_{a.shard:02d}.csv", "a", newline="") as f:
                csv.writer(f).writerow([cid, f"{type(e).__name__}: {e}"[:300]])
            print(f"case {cid} ERROR {type(e).__name__}: {e}", flush=True)
    if a.limit is None:
        (out / f"_done_shard_{a.shard:02d}").write_text(f"cases={len(cases)} errors={n_err}\n")
    print(f"shard {a.shard} finished: cases={len(cases)} errors={n_err}", flush=True)
