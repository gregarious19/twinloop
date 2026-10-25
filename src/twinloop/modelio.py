"""Load trained v2 models from `models_dir()/<name>/` (SPEC_V2 §8.3)."""

import json

from twinloop.paths import models_dir

NOT_AVAILABLE = "v2 model not available: "


class FullProba:
    """Wrap a classifier so predict_proba always has one column per action (ACTIONS order).

    A class that never occurred in training (e.g. `add`) gets probability 0; without this the
    columns of a 5-class model would shift against ACTIONS.
    """

    def __init__(self, clf, n_actions: int):
        self.clf, self.n_actions = clf, n_actions

    def predict_proba(self, X):
        import numpy as np

        p = self.clf.predict_proba(X)
        out = np.zeros((len(p), self.n_actions))
        out[:, np.asarray(self.clf.classes_, dtype=int)] = p
        return out


def load_model(name: str, device: str):
    folder = models_dir() / name
    cfg_path = folder / "config.json"
    if not cfg_path.is_file():
        raise RuntimeError(f"{NOT_AVAILABLE}{folder}")
    cfg = json.loads(cfg_path.read_text())

    if name == "decider":
        import joblib

        from twinloop.regions import FEATURE_NAMES

        if cfg.get("feature_names") != FEATURE_NAMES:
            raise RuntimeError("decider feature names do not match")
        path = folder / "decider.joblib"
        if not path.is_file():
            raise RuntimeError(f"{NOT_AVAILABLE}{path}")
        from twinloop.regions import ACTIONS

        return FullProba(joblib.load(path), len(ACTIONS))

    import torch

    try:
        from twinloop import nets
    except ImportError as e:
        raise RuntimeError(f"{NOT_AVAILABLE}twinloop.nets not found") from e
    if not (folder / "model.pt").is_file():
        raise RuntimeError(f"{NOT_AVAILABLE}{folder / 'model.pt'}")
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg["arch_kwargs"].items()}
    model = getattr(nets, cfg["arch"])(**kwargs)
    state = torch.load(folder / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)
