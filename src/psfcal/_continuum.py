"""Continuum estimation helper shared by the order-level deconvolution
driver and the RV cross-correlation routine."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import percentile_filter

CONT_MEDFILT_KSIZE = 501  # odd, >> line width


def smooth_continuum(y: np.ndarray, kernel_size: int = CONT_MEDFILT_KSIZE) -> np.ndarray:
    """
    Continuum estimate via wide median filter with reflect padding.
    Matches HARPS/ESPRESSO DRS boundary handling.
    """
    y = np.asarray(y, float)
    ks = int(kernel_size)
    if ks % 2 == 0:
        ks += 1

    cont = percentile_filter(y, percentile=50, size=ks, mode="reflect")

    good = cont > 0
    if np.any(good):
        cont[~good] = np.median(cont[good])
    else:
        cont[:] = np.median(y[y > 0]) if np.any(y > 0) else 1.0

    return cont
