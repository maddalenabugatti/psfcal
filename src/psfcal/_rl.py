"""Richardson-Lucy deconvolution core, with optional mild L2 damping."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from ._kernel import normalize_nonneg_sum1, reflect_pad

RL_ITERS_DEFAULT = 10
RL_EPS = 1e-6
L2_DAMP_DEFAULT = 3e-6


def rl_deconvolve_1d(
    y: np.ndarray,
    k: np.ndarray,
    n_iter: int = RL_ITERS_DEFAULT,
    l2_damp: float = L2_DAMP_DEFAULT,
    eps: float = RL_EPS,
) -> np.ndarray:
    """
    Richardson-Lucy deconvolution with optional mild L2 damping.

    Parameters
    ----------
    y : observed spectrum, non-negative.
    k : kernel, non-negative, sum=1, defined on integer pixels.
    """
    y = np.asarray(y, float)
    k = normalize_nonneg_sum1(k)

    pad = len(k) // 2
    y_pad = reflect_pad(np.maximum(y, 0.0), pad)

    x = np.clip(y_pad, eps, None)  # init guess
    k_flip = k[::-1]

    for _ in range(int(n_iter)):
        conv = fftconvolve(x, k, mode="same")
        ratio = y_pad / np.maximum(conv, eps)
        backproj = fftconvolve(ratio, k_flip, mode="same")
        x = x * backproj

        if l2_damp and l2_damp > 0:
            x = x / (1.0 + l2_damp * x)

        x = np.clip(x, 0.0, None)

    return x[pad:-pad] if pad > 0 else x
