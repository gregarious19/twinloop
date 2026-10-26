"""Fine-tune the decider on synthetic + real-failure rows (C08, SPEC_V2 §14)."""

import argparse
import json
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold

from train.common import write_json
from twinloop.paths import models_dir, v2_dir
from twinloop.regions import ACTIONS, FEATURE_NAMES

REAL_BASE = 5.0


def weights(y: np.ndarray, real: np.ndarray) -> np.ndarray:
    labels, counts = np.unique(y, return_counts=True)
    bal = {int(c): len(y) / (len(labels) * n) for c, n in zip(labels, counts, strict=True)}
    return np.where(real, REAL_BASE, 1.0) * np.array([bal[int(c)] for c in y])


def fit(X, y, real) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, class_weight=None, random_state=0
    ).fit(X, y, sample_weight=weights(y, real))


def macro_f1(y_true, y_pred) -> float:
    return float(
        f1_score(y_true, y_pred, labels=sorted(set(y_true)), average="macro", zero_division=0)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--git-commit", default="unknown")
    a = ap.parse_args()
    ds = v2_dir() / "datasets"
    syn = pd.concat(
        [pd.read_parquet(p) for p in sorted((ds / "decider").glob("shard_*.parquet"))],
        ignore_index=True,
    )
    real = pd.read_parquet(ds / "decider_real" / "decider_real.parquet")
    idx = {k: i for i, k in enumerate(ACTIONS)}

    def xy(df):
        return df[FEATURE_NAMES].to_numpy(np.float64), df.label.map(idx).to_numpy().astype(int)

    Xs, ys = xy(syn)
    Xr, yr = xy(real)
    print(f"synthetic {len(syn)} rows, real {len(real)} rows / {real.case_id.nunique()} cases")
    folds, pooled_t, pooled_p = [], [], []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(Xr, yr, real.case_id)):
        X = np.vstack([Xs, Xr[tr]])
        y = np.concatenate([ys, yr[tr]])
        isreal = np.r_[np.zeros(len(ys), bool), np.ones(len(tr), bool)]
        p = fit(X, y, isreal).predict(Xr[te])
        folds.append(macro_f1(yr[te], p))
        pooled_t.append(yr[te])
        pooled_p.append(p)
        print(f"fold {k}: held-out real rows {len(te)}  macro-F1 {folds[-1]:.4f}", flush=True)
    t, p = np.concatenate(pooled_t), np.concatenate(pooled_p)
    rep = classification_report(
        t, p, labels=list(range(len(ACTIONS))), target_names=ACTIONS, output_dict=True,
        zero_division=0,
    )  # fmt: skip
    print(pd.DataFrame(rep).T.round(3).to_string())
    cv = {"fold_macro_f1": folds, "mean": float(np.mean(folds)), "sd": float(np.std(folds, ddof=1)),
          "pooled_macro_f1": macro_f1(t, p), "per_class": rep}  # fmt: skip
    print(f"CV macro-F1 {cv['mean']:.4f} ± {cv['sd']:.4f}")

    X = np.vstack([Xs, Xr])
    y = np.concatenate([ys, yr])
    clf = fit(X, y, np.r_[np.zeros(len(ys), bool), np.ones(len(yr), bool)])

    cur, old = models_dir() / "decider", models_dir() / "decider_synth_only"
    if not old.exists():
        shutil.move(cur, old)
    cur.mkdir(exist_ok=True)
    joblib.dump(clf, cur / "decider.joblib")
    cfg = json.loads((old / "config.json").read_text())
    cfg.update(
        seed=0,
        trained_on="v2_train synthetic + v2_realfail real",
        sklearn_version=sklearn.__version__,
        val_metric_name="macro_f1_real_5fold_cv", val_metric=cv["mean"], git_commit=a.git_commit,
        torch_version=torch.__version__, cv=cv,
    )  # fmt: skip
    (cur / "config.json").write_text(json.dumps(cfg, indent=2))
    write_json(cur / "metrics.json", cv)
    out = clf.predict_proba(np.zeros((1, len(FEATURE_NAMES))))
    print("loaded predict_proba shape", out.shape, "classes", list(clf.classes_))
    return 0


if __name__ == "__main__":
    sys.exit(main())
