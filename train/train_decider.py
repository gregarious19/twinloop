"""Train the repair decider on synthetic damage rows (SPEC_V2 §14 recipe)."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "6")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402

from train.common import set_seed, split_case_ids, write_json  # noqa: E402
from twinloop.paths import resolve  # noqa: E402
from twinloop.regions import ACTIONS, FEATURE_NAMES  # noqa: E402


def macro_f1(y_true, y_pred) -> float:
    labels = sorted(set(y_true))
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    set_seed(a.seed)
    data, out = resolve(a.data), resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = sorted(data.glob("shard_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    print(f"{len(parts)} shards, {len(df)} rows, {df.case_id.nunique()} cases", flush=True)
    _, va_ids = split_case_ids(df.case_id.unique())
    is_val = df.case_id.isin(va_ids).to_numpy()
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = df.label.map({k: i for i, k in enumerate(ACTIONS)}).to_numpy()
    assert not np.isnan(y.astype(float)).any(), "unknown label in data"
    y = y.astype(int)
    t0 = time.time()
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=a.seed
    ).fit(X[~is_val], y[~is_val])
    print(f"fit {time.time() - t0:.0f}s", flush=True)
    p_tr, p_va = clf.predict(X[~is_val]), clf.predict(X[is_val])
    f_tr, f_va = macro_f1(y[~is_val], p_tr), macro_f1(y[is_val], p_va)
    present = sorted(set(y[is_val]) | set(p_va))
    rep = classification_report(
        y[is_val], p_va, labels=list(range(len(ACTIONS))), target_names=ACTIONS,
        output_dict=True, zero_division=0,
    )  # fmt: skip
    cm = confusion_matrix(y[is_val], p_va, labels=list(range(len(ACTIONS))))
    print(f"train macro-F1 {f_tr:.4f}  val macro-F1 {f_va:.4f}  (labels in val: {present})")
    print(pd.DataFrame(rep).T.round(3).to_string())
    print("confusion (rows = true, cols = pred, order", ACTIONS, ")\n", cm)
    joblib.dump(clf, out / "decider.joblib")
    ok = f_tr > f_va and f_va > 1 / 6
    write_json(out / "metrics.json", {
        "train_macro_f1": f_tr, "val_macro_f1": f_va, "per_class": rep,
        "confusion": cm.tolist(), "classes_in_training": [ACTIONS[c] for c in clf.classes_],
        "n_train_rows": int((~is_val).sum()), "n_val_rows": int(is_val.sum()),
        "sanity_passed": bool(ok),
    })  # fmt: skip
    print("sanity (train F1 > val F1 and val F1 > 1/6):", "PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
