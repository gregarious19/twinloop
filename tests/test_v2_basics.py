import dataclasses
import json

import pytest

from twinloop import config_v2 as c2
from twinloop import paths
from twinloop.v2split import load_roles


def test_resolve_maps_data_prefix(monkeypatch):
    monkeypatch.setenv("TWINLOOP_DATA", "/tmp/d")
    assert str(paths.resolve("/data/x")) == "/tmp/d/x"
    assert str(paths.resolve("/other/x")) == "/other/x"


def test_test_split_rejected():
    for fn in (paths.case_image_path, paths.case_label_path):
        with pytest.raises(ValueError, match="Test-split case not allowed in v2: 10"):
            fn(10)
    assert paths.case_image_path(7).name == "ImageCAS_0007_0000.nii.gz"
    assert paths.case_label_path(7).name == "ImageCAS_0007.nii.gz"


def test_load_roles_fake_splits(tmp_path, monkeypatch):
    monkeypatch.setenv("TWINLOOP_DATA", str(tmp_path))
    d = tmp_path / "nnUNet_preprocessed" / "Dataset501_ImageCAS"
    d.mkdir(parents=True)
    fold0 = {
        "train": ["ImageCAS_0001", "ImageCAS_0002", "ImageCAS_0005"],
        "val": ["ImageCAS_0003", "ImageCAS_0004", "ImageCAS_0010"],
    }
    (d / "splits_final.json").write_text(json.dumps([fold0]))
    assert load_roles() == {1: "v2_train", 2: "v2_train", 3: "v2_realfail", 4: "v2_holdout"}


def test_config_frozen_with_spec_defaults():
    p = c2.V2Params()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.graph.end_window_mm = 1.0  # type: ignore[misc]
    assert p.graph.max_components == 8
    assert p.anatomy.myo_max_ml == 350.0
    assert p.damage.type_weights == (0.2, 0.2, 0.15, 0.2, 0.15, 0.1)
    assert p.gapfill.patch_vox == 64
    assert p.repair.decider_min_prob == 0.6
    assert p.flow.flow_mode == "auto"
