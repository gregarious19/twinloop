import math

import torch

from train.common import CsvLogger, cosine_lr, soft_dice_loss, split_case_ids
from twinloop.nets import GapfillNet, LumenNet


def test_gapfill_shape():
    net = GapfillNet().eval()
    with torch.no_grad():
        y = net(torch.zeros(1, 3, 64, 64, 64))
    assert y.shape == (1, 1, 64, 64, 64)


def test_lumen_shape_and_pad_mask():
    net = LumenNet().eval()
    x = torch.randn(2, 16, 1, 48, 48)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[1, 10:] = True
    with torch.no_grad():
        assert net(x).shape == (2, 16, 1, 48, 48)
        assert net(x, pad_mask=mask).shape == (2, 16, 1, 48, 48)


def test_soft_dice_perfect():
    t = (torch.rand(2, 1, 8, 8, 8) > 0.5).float()
    assert soft_dice_loss(t * 40 - 20, t).item() < 1e-3


def test_cosine_lr():
    assert cosine_lr(0, 100, 10, 1e-3) == 0.0
    assert math.isclose(cosine_lr(10, 100, 10, 1e-3), 1e-3)
    assert cosine_lr(100, 100, 10, 1e-3) == 0.0


def test_split_and_logger(tmp_path):
    tr, va = split_case_ids([3, 13, 4, 7, 23])
    assert va == [3, 13, 23] and tr == [4, 7]
    p = tmp_path / "l.csv"
    for _ in range(2):
        lg = CsvLogger(p, ["a", "b"])
        lg.log(a=1, b=2)
        lg.close()
    assert p.read_text().splitlines() == ["a,b", "1,2", "1,2"]
