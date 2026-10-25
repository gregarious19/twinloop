"""v2 case roles from nnU-Net fold 0 (SPEC_V2 §2.2)."""

import json

from twinloop.paths import data_root

_SPLITS = "nnUNet_preprocessed/Dataset501_ImageCAS/splits_final.json"


def _case_id(entry: str) -> int:
    return int(entry.split("_")[1])


def load_roles() -> dict[int, str]:
    """Map case_id -> v2_train | v2_holdout | v2_realfail. Test-split cases never appear."""
    js = json.loads((data_root() / _SPLITS).read_text())
    fold0 = js[0]
    roles: dict[int, str] = {}
    for entry in fold0["train"]:
        cid = _case_id(entry)
        if cid % 5 != 0:
            roles[cid] = "v2_train"
    for entry in fold0["val"]:
        cid = _case_id(entry)
        if cid % 5 != 0:
            roles[cid] = "v2_holdout" if cid % 2 == 0 else "v2_realfail"
    return roles
