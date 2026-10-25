"""Train LumenNet (SPEC_V2 §14 recipe). Resumable; kill criterion at 25 % of the steps."""

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
from twinloop.config_v2 import CprParams
from twinloop.cpr import center_component, r_eq_mm
from twinloop.lumen import _predict_logits
from twinloop.nets import LumenNet
from twinloop.paths import resolve

KILL_ERR_MM = 0.5
WINDOW = 128
CP = CprParams()


class LumenData(Dataset):
    def __init__(self, files: list[Path], augment: bool, n_per_epoch: int | None = None):
        self.files, self.augment = files, augment
        self.n = n_per_epoch or len(files)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        d = np.load(self.files[i % len(self.files)])
        img, mask, ds = d["img"].astype(np.float32), d["mask"].astype(np.float32), d["ds"]
        n = len(img)
        s = np.random.randint(0, n - WINDOW + 1) if n > WINDOW else 0
        img, mask, ds = img[s : s + WINDOW], mask[s : s + WINDOW], ds[s : s + WINDOW]
        if self.augment:
            for ax in (1, 2):
                if np.random.rand() < 0.5:
                    img, mask = np.flip(img, axis=ax), np.flip(mask, axis=ax)
            k = np.random.randint(4)
            img, mask = np.rot90(img, k, axes=(1, 2)), np.rot90(mask, k, axes=(1, 2))
            img = img * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05)
        return (
            torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)),
            torch.from_numpy(ds.astype(np.float32)),
        )


def collate(batch):
    t = max(len(b[0]) for b in batch)
    B = len(batch)
    x = torch.zeros(B, t, 1, 48, 48)
    y = torch.zeros(B, t, 1, 48, 48)
    ds = torch.zeros(B, t)
    pad = torch.ones(B, t, dtype=torch.bool)
    for i, (img, mask, d) in enumerate(batch):
        n = len(img)
        x[i, :n, 0], y[i, :n, 0], ds[i, :n] = img, mask, d
        pad[i, :n] = False
    return x, y, ds, pad


def list_files(data: Path):
    files = sorted(data.glob("*.npz"))
    tr_ids, va_ids = split_case_ids({int(f.name[:4]) for f in files})
    tr, va = set(tr_ids), set(va_ids)
    return [f for f in files if int(f.name[:4]) in tr], [f for f in files if int(f.name[:4]) in va]


def loss_fn(logits, y, ds, pad):
    valid = (~pad)[:, :, None, None, None].float()
    w = (1.0 + 2.0 * (ds >= 0.25).float())[:, :, None, None, None] * valid
    bce = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
    bce = (bce * w).sum() / (w.sum() * 48 * 48).clamp(min=1.0)
    lg = torch.where(valid.bool(), logits.float(), torch.full_like(logits.float(), -20.0))
    return bce + soft_dice_loss(lg, y * valid)


@torch.no_grad()
def validate(model, files, device) -> float:
    model.eval()
    errs = []
    for f in files:
        d = np.load(f)
        x = d["img"].astype(np.float32)
        gt = d["mask"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = _predict_logits(model, x, CP, device)
        pred = logits > 0
        for k in range(len(x)):
            if not gt[k].any():
                continue
            r_gt = r_eq_mm(gt[k] > 0, CP.pixel_mm)
            r_pr = r_eq_mm(center_component(pred[k]), CP.pixel_mm)
            errs.append(abs(r_pr - r_gt))
    model.train()
    return float(np.mean(errs)) if errs else float("nan")


def best_step_from_log(path: Path, best: float) -> int:
    rows = [r for r in csv.DictReader(open(path)) if r["val_metric"]]
    hit = [int(r["step"]) for r in rows if abs(float(r["val_metric"]) - best) < 1e-5]
    return hit[-1] if hit else 0


def sanity(files, device) -> int:
    sel = files[:: max(1, len(files) // 4)][:4]
    ds_ = LumenData(sel, False)
    x, y, ds, pad = collate([ds_[i] for i in range(len(sel))])
    x, y, ds, pad = x.to(device), y.to(device), ds.to(device), pad.to(device)
    model = LumenNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(1000):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, pad_mask=pad)
        loss = loss_fn(logits.float(), y, ds, pad)
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
        p = (model(x, pad_mask=pad).float() > 0).float()
    valid = (~pad)[:, :, None, None, None].float()
    inter = (p * y * valid).sum()
    dice = float(2 * inter / ((p * valid).sum() + (y * valid).sum()).clamp(min=1))
    ok = np.isfinite(init) and 0.5 <= init <= 3.0 and dice > 0.8
    print(f"sanity: init loss {init:.4f} (need 0.5-3.0), final train Dice {dice:.4f} "
          f"(need > 0.8) -> {'PASS' if ok else 'FAIL'}")  # fmt: skip
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="time N steps, print s/step, exit")
    a = ap.parse_args()
    device = "cuda"
    set_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    tr_files, va_files = list_files(resolve(a.data))
    print(f"train branches {len(tr_files)}, val branches {len(va_files)}", flush=True)
    if a.sanity:
        return sanity(tr_files, device)

    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mj = out / "metrics.json"
    if mj.exists() and json.loads(mj.read_text()).get("killed"):
        print("this seed was killed earlier (metrics.json); not resuming")
        return 3
    model = LumenNet().to(device)
    base_lr, warmup = 3e-4, 500
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    tl = DataLoader(
        LumenData(tr_files, True), batch_size=a.batch, shuffle=True, drop_last=True,
        collate_fn=collate, num_workers=a.workers, pin_memory=True, persistent_workers=True,
    )  # fmt: skip
    step, best = 0, float("inf")
    ck = out / "ckpt.pt"
    if ck.exists() and not a.bench:
        step, best = load_ckpt(ck, model, opt)
        print(f"resumed at step {step}, best {best:.4f}", flush=True)
    log_path = out / "train_log.csv"
    t_off = last_elapsed(log_path) if not a.bench else 0.0
    cols = ["step", "loss", "lr", "val_metric", "elapsed_s"]
    log = CsvLogger(log_path if not a.bench else out / "bench_log.csv", cols)
    kill_step = int(0.25 * a.steps)
    best_step = best_step_from_log(log_path, best) if step else 0
    t0, losses = time.time(), []
    model.train()
    while step < a.steps:
        for x, y, ds, pad in tl:
            if step >= a.steps:
                break
            lr = cosine_lr(step, a.steps, warmup, base_lr)
            for gp in opt.param_groups:
                gp["lr"] = lr
            x, y, ds, pad = (v.to(device, non_blocking=True) for v in (x, y, ds, pad))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, pad_mask=pad)
            loss = loss_fn(logits.float(), y, ds, pad)
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
            val = None
            if step % 2000 == 0 or step == kill_step or step == a.steps:
                val = validate(model, va_files, device)
                if val < best:
                    best, best_step = val, step
                    torch.save(model.state_dict(), out / "best.pt")
            if step % 100 == 0 or val is not None:
                log.log(step=step, loss=f"{np.mean(losses):.5f}", lr=f"{lr:.3e}",
                        val_metric="" if val is None else f"{val:.5f}",
                        elapsed_s=f"{elapsed:.0f}")  # fmt: skip
                losses = []
                if step % 100 == 0:
                    print(f"step {step} lr {lr:.2e} val {val}", flush=True)
            if val is not None:
                save_ckpt(ck, model, opt, step, best)
                if step == kill_step and val > KILL_ERR_MM and step < a.steps:
                    write_json(mj, {"killed": True, "kill_step": step,
                               "val_req_err_mm_at_kill": val, "best_val_req_err_mm": best,
                               "best_step": best_step, "final_step": step,
                               "hours": elapsed / 3600})  # fmt: skip
                    print(f"KILLED at step {step}: val r_eq error {val:.3f} mm > {KILL_ERR_MM}")
                    return 3
    write_json(mj, {"best_val_req_err_mm": best, "best_step": best_step, "final_step": step,
               "hours": (t_off + time.time() - t0) / 3600})  # fmt: skip
    print(f"done: best val r_eq error {best:.4f} mm at step {best_step}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
