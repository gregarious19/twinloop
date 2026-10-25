"""Train GapfillNet (SPEC_V2 §14 recipe). Resumable; kill criterion at 25 % of the steps."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train.common import (
    CsvLogger,
    cosine_lr,
    last_elapsed,
    load_ckpt,
    save_ckpt,
    set_seed,
    soft_dice_loss,
    split_case_ids,
    write_json,
)
from twinloop.nets import GapfillNet
from twinloop.paths import resolve

NEG_TYPES = ("occlusion", "taper_neg")
KILL_DICE = 0.30


class GapfillData(Dataset):
    def __init__(self, files: list[Path], augment: bool):
        self.files, self.augment = files, augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        x = d["x"].astype(np.float32)
        y = d["y"].astype(np.float32)[None]
        if self.augment:
            for ax in range(3):
                if np.random.rand() < 0.5:
                    x = np.flip(x, axis=ax + 1)
                    y = np.flip(y, axis=ax + 1)
            x[0] = x[0] * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05)
            x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
        return torch.from_numpy(x), torch.from_numpy(y), str(d["type"]) in NEG_TYPES


def list_files(data: Path):
    files = sorted(data.glob("*.npz"))
    tr_ids, va_ids = split_case_ids({int(f.name[:4]) for f in files})
    tr, va = set(tr_ids), set(va_ids)
    return [f for f in files if int(f.name[:4]) in tr], [f for f in files if int(f.name[:4]) in va]


def loss_fn(logits, y, x):
    inp = x[:, 1:2]
    w = 1.0 + 4.0 * (y != inp).float()
    bce = (F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none") * w).mean()
    return bce + soft_dice_loss(logits, y)


def new_voxel_dice(logits, y, x):
    """Per-sample dice(pred & ~inp, tgt & ~inp); both empty -> 1. Also pred & ~inp counts."""
    inp = x[:, 1:2] > 0.5
    pred = (logits > 0) & ~inp
    tgt = (y > 0.5) & ~inp
    dims = (1, 2, 3, 4)
    inter = (pred & tgt).sum(dims).float()
    den = pred.sum(dims).float() + tgt.sum(dims).float()
    dice = torch.where(den == 0, torch.ones_like(den), 2 * inter / den.clamp(min=1))
    return dice, pred.sum(dims).float()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    pos, neg = [], []
    for x, y, is_neg in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        dice, cnt = new_voxel_dice(logits.float(), y, x)
        for k in range(len(x)):
            (neg if is_neg[k] else pos).append((dice[k].item(), cnt[k].item()))
    model.train()
    d = float(np.mean([p[0] for p in pos])) if pos else float("nan")
    n = float(np.mean([q[1] for q in neg])) if neg else float("nan")
    return d, n


def best_step_from_log(path: Path, best: float) -> int:
    rows = [r for r in csv.DictReader(open(path)) if r["val_metric"]]
    hit = [int(r["step"]) for r in rows if abs(float(r["val_metric"]) - best) < 1e-5]
    return hit[-1] if hit else 0


def sanity(files, device, workers) -> int:
    pos = [f for f in files if str(np.load(f)["type"]) not in NEG_TYPES]
    sel = pos[:: max(1, len(pos) // 4)][:4]
    xs, ys, _ = zip(*[GapfillData([f], False)[0] for f in sel], strict=True)
    x, y = torch.stack(xs).to(device), torch.stack(ys).to(device)
    model = GapfillNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(1000):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = loss_fn(logits.float(), y, x)
        if step == 0:
            init = loss.item()
            print(f"init loss {init:.4f}", flush=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        dice, _ = new_voxel_dice(model(x).float(), y, x)
    d = float(dice.mean())
    ok = np.isfinite(init) and 0.5 <= init <= 3.0 and d > 0.8
    print(f"sanity: init loss {init:.4f} (need 0.5-3.0), final train new-voxel Dice {d:.4f} "
          f"(need > 0.8) -> {'PASS' if ok else 'FAIL'}")  # fmt: skip
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="time N steps, print s/step, exit")
    a = ap.parse_args()
    device = "cuda"
    set_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    tr_files, va_files = list_files(resolve(a.data))
    print(f"train samples {len(tr_files)}, val samples {len(va_files)}", flush=True)
    if a.sanity:
        return sanity(tr_files, device, a.workers)

    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mj = out / "metrics.json"
    if mj.exists() and json.loads(mj.read_text()).get("killed"):
        print("this seed was killed earlier (metrics.json); not resuming")
        return 3
    model = GapfillNet().to(device)
    base_lr, warmup = 3e-4, 500
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    tl = DataLoader(
        GapfillData(tr_files, True), batch_size=a.batch, shuffle=True, drop_last=True,
        num_workers=a.workers, pin_memory=True, persistent_workers=True,
    )  # fmt: skip
    vl = DataLoader(GapfillData(va_files, False), batch_size=16, num_workers=a.workers)
    step, best = 0, -1.0
    ck = out / "ckpt.pt"
    if ck.exists() and not a.bench:
        step, best = load_ckpt(ck, model, opt)
        print(f"resumed at step {step}, best {best:.4f}", flush=True)
    log_path = out / "train_log.csv"
    t_off = last_elapsed(log_path) if not a.bench else 0.0
    cols = ["step", "loss", "lr", "val_metric", "elapsed_s", "val_neg_newvox"]
    log = CsvLogger(log_path if not a.bench else out / "bench_log.csv", cols)
    kill_step = int(0.25 * a.steps)
    best_step = best_step_from_log(log_path, best) if step else 0
    t0, losses = time.time(), []
    model.train()
    while step < a.steps:
        for x, y, _ in tl:
            if step >= a.steps:
                break
            lr = cosine_lr(step, a.steps, warmup, base_lr)
            for gp in opt.param_groups:
                gp["lr"] = lr
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = loss_fn(logits.float(), y, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            losses.append(loss.item())
            elapsed = t_off + time.time() - t0
            if a.bench and step == a.bench:
                print(
                    f"BENCH {a.bench} steps: {(time.time() - t0) / a.bench:.3f} s/step", flush=True
                )
                return 0
            val = neg = None
            if step % 2000 == 0 or step == kill_step or step == a.steps:
                val, neg = validate(model, vl, device)
                if val > best:
                    best, best_step = val, step
                    torch.save(model.state_dict(), out / "best.pt")
            if step % 100 == 0 or val is not None:
                log.log(step=step, loss=f"{np.mean(losses):.5f}", lr=f"{lr:.3e}",
                        val_metric="" if val is None else f"{val:.5f}",
                        elapsed_s=f"{elapsed:.0f}",
                        val_neg_newvox="" if neg is None else f"{neg:.1f}")  # fmt: skip
                losses = []
                if step % 100 == 0:
                    print(f"step {step} lr {lr:.2e} val {val}", flush=True)
            if val is not None:
                save_ckpt(ck, model, opt, step, best)
                if step == kill_step and val < KILL_DICE and step < a.steps:
                    write_json(out / "metrics.json", {"killed": True, "kill_step": step,
                               "val_new_dice_at_kill": val, "best_val_new_dice": best,
                               "best_step": best_step, "final_step": step,
                               "hours": elapsed / 3600})  # fmt: skip
                    print(f"KILLED at step {step}: val new-voxel Dice {val:.3f} < {KILL_DICE}")
                    return 3
    write_json(out / "metrics.json", {"best_val_new_dice": best, "best_step": best_step,
               "final_step": step, "hours": (t_off + time.time() - t0) / 3600})  # fmt: skip
    print(f"done: best val new-voxel Dice {best:.4f} at step {best_step}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
