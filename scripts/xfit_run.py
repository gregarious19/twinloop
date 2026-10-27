"""Cross-fit segmenter runs (D-V2-S26): train fold k on 2/3 of v2_train, predict the other third.

Per fold: nnUNetv2_train 511 3d_fullres <k> (EPOCHS epochs; nnU-Net writes checkpoint_latest at
epoch 50 and checkpoint_final at the end), a watcher copies each new checkpoint_latest to
checkpoint_ep<N>.pth, then every held-out case is predicted with each kept checkpoint into
<data>/v2/xfit/pred/fold<k>_ep<N>/. Resumable (continues training, skips finished predictions).
Usage: python scripts/xfit_run.py [--folds 0 1 2] [--trainer nnUNetTrainer_100epochs]
"""

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

from twinloop.paths import case_image_path, data_root, v2_dir

DS, DS_NAME = "511", "Dataset511_ImageCASxfit"


def results_dir(trainer: str, k: int) -> Path:
    return (
        data_root()
        / "nnUNet_results"
        / DS_NAME
        / f"{trainer}__nnUNetPlans__3d_fullres"
        / f"fold_{k}"
    )


def watch(folder: Path, stop: threading.Event) -> None:
    """Copy each new checkpoint_latest.pth to checkpoint_ep<epoch>.pth once it is fully written."""
    seen = 0.0
    latest = folder / "checkpoint_latest.pth"
    while not stop.wait(20):
        if not latest.exists() or latest.stat().st_mtime == seen:
            continue
        size = latest.stat().st_size
        time.sleep(10)
        if latest.stat().st_size != size:
            continue
        seen = latest.stat().st_mtime
        try:
            epoch = int(torch.load(latest, map_location="cpu", weights_only=False)["current_epoch"])
        except Exception as e:  # noqa: BLE001 - retry on the next poll
            print(f"watcher: could not read {latest}: {e}", flush=True)
            seen = 0.0
            continue
        shutil.copyfile(latest, folder / f"checkpoint_ep{epoch}.pth")
        print(f"watcher: saved checkpoint_ep{epoch}.pth", flush=True)


def train(trainer: str, k: int) -> None:
    folder = results_dir(trainer, k)
    if (folder / "checkpoint_final.pth").exists():
        print(f"fold {k}: training already finished", flush=True)
        return
    cmd = ["nnUNetv2_train", DS, "3d_fullres", str(k), "-tr", trainer]
    if (folder / "checkpoint_latest.pth").exists():
        cmd.append("--c")
    folder.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    t = threading.Thread(target=watch, args=(folder, stop), daemon=True)
    t.start()
    try:
        subprocess.run(cmd, check=True)
    finally:
        stop.set()
        t.join(timeout=60)


def predict(trainer: str, k: int) -> None:
    folder = results_dir(trainer, k)
    val = json.loads(
        (data_root() / "nnUNet_preprocessed" / DS_NAME / "splits_final.json").read_text()
    )[k]["val"]
    ids = [int(v.split("_")[1]) for v in val]
    assert all(c % 5 != 0 for c in ids)
    ckpts = sorted(folder.glob("checkpoint_ep*.pth")) + [folder / "checkpoint_final.pth"]
    for ck in ckpts:
        tag = "ep" + ck.stem.removeprefix("checkpoint_ep") if "_ep" in ck.stem else "final"
        out = v2_dir() / "xfit" / "pred" / f"fold{k}_{tag}"
        if out.exists() and all((out / f"ImageCAS_{c:04d}.nii.gz").exists() for c in ids):
            continue
        out.mkdir(parents=True, exist_ok=True)
        inp = v2_dir() / "tmp" / f"xfit_in_fold{k}"
        shutil.rmtree(inp, ignore_errors=True)
        inp.mkdir(parents=True)
        for c in ids:
            (inp / f"ImageCAS_{c:04d}_0000.nii.gz").symlink_to(case_image_path(c))
        cmd = ["nnUNetv2_predict", "-i", str(inp), "-o", str(out), "-d", DS, "-c", "3d_fullres",
               "-f", str(k), "-tr", trainer, "-chk", ck.name, "--disable_tta"]  # fmt: skip
        subprocess.run(cmd, check=True)
        print(f"fold {k} {tag}: predicted {len(ids)} cases", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--trainer", default="nnUNetTrainer_100epochs")
    a = ap.parse_args()
    for k in a.folds:
        train(a.trainer, k)
        predict(a.trainer, k)
    print("xfit done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
