"""Half-maximum lumen mask on a CPR slice (SPEC_V3 §4.5, §9.3)."""

import numpy as np
from scipy import ndimage

_YY, _XX = np.mgrid[:48, :48]
_DIST_MM = np.hypot(_YY - 23.5, _XX - 23.5) * 0.2
HU_CLIP = (-200.0, 800.0)


def auto_mask(hu: np.ndarray) -> np.ndarray:
    """Half-maximum region around the centre of one 48x48 slice (HU); all-False if none."""
    hu = np.clip(np.asarray(hu, dtype=np.float32), *HU_CLIP)
    sm = ndimage.gaussian_filter(hu, 0.6)
    near = _DIST_MM <= 1.0
    peak = np.percentile(sm[near], 90)
    thr = (peak + np.percentile(sm, 15)) / 2.0
    lab, _ = ndimage.label(sm > thr)  # strict: a blank (constant) slice gives no region
    k = lab[np.unravel_index(np.argmax(np.where(near, sm, -np.inf)), sm.shape)]
    if k == 0:
        return np.zeros(sm.shape, dtype=bool)
    return ndimage.binary_fill_holes(lab == k)
