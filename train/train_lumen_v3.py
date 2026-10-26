"""Train LumenNet v3 (S23): oversampled narrow slices, loss weights, stratified validation.

Same architecture and data as train_lumen.py. Resumable; kill criterion at 25 % of the steps.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Dataset

from train.common import (
    CsvLogger,
    cosine_lr,
    last_elapsed,
    load_ckpt,
    save_ckpt,
    set_seed,
    soft_dice_loss,
    write_json,
)
from train.train_lumen import CP, WINDOW, collate, list_files
from twinloop.cpr import center_component, r_eq_mm, r_ins_2d_mm
from twinloop.lumen import _predict_logits
from twinloop.nets import LumenNet
from twinloop.paths import resolve

KILL_ERR_MM = 0.5
P_LESION_WINDOW = 0.5  # share of batch items that are lesion windows
P_MODERATE = 0.5  # share of lesion windows centred on a moderate+ slice (else mild)
W_HEALTHY, W_MILD, W_MODPLUS = 1.0, 3.0, 6.0
BIN_NAMES = ("healthy", "mild", "modplus")


def _lesion_slices(f: Path) -> tuple[np.ndarray, np.ndarray]:
    ds = np.nan_to_num(np.load(f)["ds"].astype(np.float32), nan=0.0)
    return np.nonzero((ds >= 0.25) & (ds < 0.5))[0], np.nonzero(ds >= 0.5)[0]


def build_index(files: list[Path]) -> dict[str, np.ndarray]:
    """(file index, slice index) pairs of every mild and moderate+ slice."""
    res = Parallel(n_jobs=6)(delayed(_lesion_slices)(f) for f in files)
    out = {}
    for name, j in (("mild", 0), ("modplus", 1)):
        out[name] = np.array(
            [(i, k) for i, r in enumerate(res) for k in r[j]], dtype=np.int64
        ).reshape(-1, 2)
    return out


class LumenDataV3(Dataset):
    def __init__(self, files: list[Path], index: dict[str, np.ndarray]):
        self.files, self.index = files, index

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fi, centre = i % len(self.files), None
        if np.random.rand() < P_LESION_WINDOW:
            cat = "modplus" if np.random.rand() < P_MODERATE else "mild"
            if len(self.index[cat]) == 0:
                cat = "mild" if cat == "modplus" else "modplus"
            if len(self.index[cat]):
                fi, centre = (
                    int(v) for v in self.index[cat][np.random.randint(len(self.index[cat]))]
                )
        d = np.load(self.files[fi])
        img, mask, ds = d["img"].astype(np.float32), d["mask"].astype(np.float32), d["ds"]
        n = len(img)
        if n <= WINDOW:
            s = 0
        elif centre is None:
            s = np.random.randint(0, n - WINDOW + 1)
        else:  # uniform over the window offsets that contain the chosen slice
            s = np.random.randint(max(0, centre - WINDOW + 1), min(centre, n - WINDOW) + 1)
        img, mask, ds = img[s : s + WINDOW], mask[s : s + WINDOW], ds[s : s + WINDOW]
        for ax in (1, 2):
            if np.random.rand() < 0.5:
                img, mask = np.flip(img, axis=ax), np.flip(mask, axis=ax)
        k = np.random.randint(4)
        img, mask = np.rot90(img, k, axes=(1, 2)), np.rot90(mask, k, axes=(1, 2))
        img = img * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05)
        return (
            torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)),
            torch.from_numpy(np.nan_to_num(ds.astype(np.float32), nan=0.0)),
        )


def loss_fn(logits, y, ds, pad):
    valid = (~pad)[:, :, None, None, None].float()
    w = torch.where(ds >= 0.5, W_MODPLUS, torch.where(ds >= 0.25, W_MILD, W_HEALTHY))
    w = w[:, :, None, None, None] * valid
    bce = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
    bce = (bce * w).sum() / (w.sum() * 48 * 48).clamp(min=1.0)
    lg = torch.where(valid.bool(), logits.float(), torch.full_like(logits.float(), -20.0))
    return bce + soft_dice_loss(lg, y * valid)


@torch.no_grad()
def validate(model, files, device) -> dict:
    """Mean |r(pred) - r(GT)| in mm per bin for r_ins_2d and r_eq (slices with a GT lumen)."""
    was_training = model.training
    model.eval()
    errs = {(m, b): [] for m in ("ins", "eq") for b in BIN_NAMES}
    for f in files:
        d = np.load(f)
        x = d["img"].astype(np.float32)
        gt = d["mask"]
        ds = np.nan_to_num(d["ds"].astype(np.float32), nan=0.0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = _predict_logits(model, x, CP, device)
        pred = logits > 0
        for k in np.nonzero(gt.reshape(len(gt), -1).any(axis=1))[0]:
            comp = center_component(pred[k])
            g = gt[k] > 0
            b = BIN_NAMES[2 if ds[k] >= 0.5 else 1 if ds[k] >= 0.25 else 0]
            errs["ins", b].append(abs(r_ins_2d_mm(comp, CP.pixel_mm) - r_ins_2d_mm(g, CP.pixel_mm)))
            errs["eq", b].append(abs(r_eq_mm(comp, CP.pixel_mm) - r_eq_mm(g, CP.pixel_mm)))
    if was_training:
        model.train()
    out = {}
    for (m, b), v in errs.items():
        out[f"{m}_{b}"] = float(np.mean(v)) if v else float("nan")
        out[f"n_{b}"] = len(v) if m == "ins" else out.get(f"n_{b}", 0)
    return out


def jit_only(files: list[Path]) -> list[Path]:
    return [f for f in files if f.stem.endswith("_jit")]


def best_step_from_log(path: Path, best: float) -> int:
    rows = [r for r in csv.DictReader(open(path)) if r["val_metric"]]
    hit = [int(r["step"]) for r in rows if abs(float(r["val_metric"]) - best) < 1e-5]
    return hit[-1] if hit else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--bench", type=int, default=0, help="time N steps, print s/step, exit")
    ap.add_argument("--eval-old", action="store_true", help="validate the deployed `lumen` model")
    a = ap.parse_args()
    device = "cuda"
    set_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    tr_files, va_files = list_files(resolve(a.data))
    va_files = jit_only(va_files)
    print(f"train branches {len(tr_files)}, val (jit) branches {len(va_files)}", flush=True)
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.eval_old:
        from twinloop.modelio import load_model

        t0 = time.time()
        res = validate(load_model("lumen", device), va_files, device)
        res["seconds"] = time.time() - t0
        write_json(out / "old_model_validation.json", res)
        print(json.dumps(res, indent=1))
        return 0

    mj = out / "metrics.json"
    if mj.exists() and json.loads(mj.read_text()).get("killed"):
        print("this seed was killed earlier (metrics.json); not resuming")
        return 3
    index = build_index(tr_files)
    print({k: len(v) for k, v in index.items()}, "lesion slices indexed", flush=True)
    model = LumenNet().to(device)
    base_lr, warmup = 3e-4, 500
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    tl = DataLoader(
        LumenDataV3(tr_files, index), batch_size=a.batch, shuffle=True, drop_last=True,
        collate_fn=collate, num_workers=a.workers, pin_memory=True, persistent_workers=True,
    )  # fmt: skip
    step, best = 0, float("inf")
    ck = out / "ckpt.pt"
    if ck.exists() and not a.bench:
        step, best = load_ckpt(ck, model, opt)
        print(f"resumed at step {step}, best {best:.4f}", flush=True)
    log_path = out / "train_log.csv"
    t_off = last_elapsed(log_path) if not a.bench else 0.0
    cols = ["step", "loss", "lr", "val_metric", "ins_healthy", "ins_mild", "eq_healthy",
            "eq_mild", "eq_modplus", "elapsed_s"]  # fmt: skip
    log = CsvLogger(log_path if not a.bench else out / "bench_log.csv", cols)
    kill_step = int(0.25 * a.steps)
    best_step = best_step_from_log(log_path, best) if step else 0
    best_res: dict = json.loads((out / "best_val.json").read_text()) if step and (
        out / "best_val.json").exists() else {}  # fmt: skip
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
                m = val["ins_modplus"]
                if np.isfinite(m) and m < best:
                    best, best_step, best_res = m, step, {**val, "step": step}
                    torch.save(model.state_dict(), out / "best.pt")
                    write_json(out / "best_val.json", best_res)
            if step % 100 == 0 or val is not None:
                log.log(step=step, loss=f"{np.mean(losses):.5f}", lr=f"{lr:.3e}",
                        val_metric="" if val is None else f"{val['ins_modplus']:.5f}",
                        ins_healthy="" if val is None else f"{val['ins_healthy']:.5f}",
                        ins_mild="" if val is None else f"{val['ins_mild']:.5f}",
                        eq_healthy="" if val is None else f"{val['eq_healthy']:.5f}",
                        eq_mild="" if val is None else f"{val['eq_mild']:.5f}",
                        eq_modplus="" if val is None else f"{val['eq_modplus']:.5f}",
                        elapsed_s=f"{elapsed:.0f}")  # fmt: skip
                losses = []
                if step % 100 == 0:
                    print(f"step {step} lr {lr:.2e} val {val}", flush=True)
            if val is not None:
                save_ckpt(ck, model, opt, step, best)
                if step == kill_step and not val["ins_modplus"] <= KILL_ERR_MM and step < a.steps:
                    write_json(mj, {"killed": True, "kill_step": step, "val_at_kill": val,
                               "best_val_modplus_rins_err_mm": best, "best_step": best_step,
                               "final_step": step, "hours": elapsed / 3600})  # fmt: skip
                    print(f"KILLED at step {step}: moderate+ r_ins error {val['ins_modplus']:.3f}")
                    return 3
    write_json(mj, {"best_val_modplus_rins_err_mm": best, "best_step": best_step,
               "best_val": best_res, "final_step": step,
               "hours": (t_off + time.time() - t0) / 3600})  # fmt: skip
    print(f"done: best moderate+ r_ins error {best:.4f} mm at step {best_step}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
