"""Mechanics check of segment_nnunet with the weak 5-epoch benchmark model (train case only)."""

import os
from pathlib import Path

import pandas as pd
import pytest
import SimpleITK as sitk

from twinloop.io import load_image
from twinloop.segmentation import model_available, segment_nnunet

MANIFEST = Path(os.environ.get("TWINLOOP_DATA", "/data")) / "raw/imagecas/manifest.csv"

pytestmark = [pytest.mark.gpu, pytest.mark.data]


def test_segment_first_train_case(tmp_path, monkeypatch):
    monkeypatch.setenv("TWINLOOP_NNUNET_TRAINER", "nnUNetTrainer_5epochs")
    if not model_available():
        pytest.skip("benchmark checkpoint missing")
    m = pd.read_csv(MANIFEST)
    row = m[m.split == "train"].sort_values("case_id").iloc[0]
    image = load_image(Path(row.image_path))
    mask = segment_nnunet(image, tmp_path)
    assert mask.GetSize() == image.GetSize()
    assert mask.GetPixelID() == sitk.sitkUInt8
    values = set(sitk.GetArrayViewFromImage(mask).ravel().tolist())
    assert values <= {0, 1}
