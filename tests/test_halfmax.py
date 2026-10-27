import numpy as np

from twinloop.halfmax import _XX, _YY, auto_mask


def test_blank_slice_is_all_false():
    m = auto_mask(np.full((48, 48), -1000.0))
    assert m.shape == (48, 48) and m.dtype == bool and not m.any()


def test_disc_area():
    hu = np.where(np.hypot(_YY - 23.5, _XX - 23.5) <= 8, 500.0, 0.0)
    area = auto_mask(hu).sum()
    assert abs(area - (hu > 0).sum()) <= 0.15 * (hu > 0).sum()
