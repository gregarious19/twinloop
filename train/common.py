"""Shared training utilities (SPEC_V2 §14)."""

from __future__ import annotations

import csv
import math
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    """Linear warmup to base_lr over `warmup` steps, then cosine to 0 at `total`."""
    if step < warmup:
        return base_lr * step / warmup
    if step >= total:
        return 0.0
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * prog))


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits.float())
    t = target.float()
    dims = tuple(range(1, p.ndim))
    inter = (p * t).sum(dim=dims)
    dice = (2 * inter + eps) / (p.sum(dim=dims) + t.sum(dim=dims) + eps)
    return (1.0 - dice).mean()


class CsvLogger:
    def __init__(self, path: str | Path, columns: list[str]):
        self.path = Path(path)
        self.columns = columns
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._f = open(self.path, "a", newline="")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(columns)
            self._f.flush()

    def log(self, **row) -> None:
        self._w.writerow([row.get(c, "") for c in self.columns])
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def save_ckpt(path: str | Path, model, opt, step: int, best_metric: float) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best_metric},
        tmp,
    )
    os.replace(tmp, path)


def load_ckpt(path: str | Path, model, opt) -> tuple[int, float]:
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    return int(ck["step"]), float(ck["best"])


def split_case_ids(ids) -> tuple[list[int], list[int]]:
    """val = case_id % 10 == 3."""
    ids = [int(i) for i in ids]
    return [i for i in ids if i % 10 != 3], [i for i in ids if i % 10 == 3]


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"
