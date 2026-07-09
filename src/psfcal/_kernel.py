"""
Shared low-level helpers for turning a PSF/IP (sampled on an arbitrary
offset grid, in pixels or in velocity/m-s) into a centred, discrete,
sum-normalised integer-pixel convolution kernel.

Ported from the (previously duplicated) helpers in
``deconvolve_spectra_lucy.py`` and ``deconvolve_spectra_lucy_rv.py``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

C_LIGHT_MPS = 299_792_458.0


def reflect_pad(x: np.ndarray, halfwidth: int) -> np.ndarray:
    """Mirror-pad a 1-D array by ``halfwidth`` samples on each side."""
    if halfwidth <= 0:
        return x
    x = np.asarray(x)
    if x.size == 1:
        return np.concatenate([np.repeat(x, halfwidth), x, np.repeat(x, halfwidth)])
    left = x[1: halfwidth + 1][::-1]
    right = x[-halfwidth - 1: -1][::-1]
    return np.concatenate([left, x, right])


def normalize_nonneg_sum1(x: np.ndarray) -> np.ndarray:
    """Clip negative values to zero and normalise so the array sums to 1."""
    x = np.maximum(np.asarray(x, float), 0.0)
    s = np.sum(x)
    return x / s if s > 0 else x


def cosine_window(n: int) -> np.ndarray:
    """Hann-like 0..1..0 window, used to taper overlap-add chunk edges."""
    idx = np.arange(n)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * (idx + 0.5) / n)


# ─────────────────────────────────────────────────────────────
#  Pixel-space PSF -> integer pixel kernel
# ─────────────────────────────────────────────────────────────

def centroid_center_psf(
    psf_x: np.ndarray, psf_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Centre a pixel-space PSF so its centroid is at 0 and area-normalise it
    (integral = 1). Returns ``(x_centered, y_norm, centroid_shift)``.
    """
    x = np.asarray(psf_x, float)
    y = np.clip(np.asarray(psf_y, float), 0.0, None)

    area = np.trapz(y, x)
    if not np.isfinite(area) or area <= 0:
        return x.copy(), y.copy(), 0.0

    centroid = np.trapz(x * y, x) / area
    x2 = x - centroid

    area2 = np.trapz(y, x2)
    if area2 > 0:
        y = y / area2

    return x2, y, float(centroid)


def psf_to_integer_pixel_kernel(
    psf_x: np.ndarray,
    psf_y: np.ndarray,
    pixel_radius: int = 7,
) -> np.ndarray:
    """
    Resample a pixel-space PSF (arbitrary offset grid) onto integer pixel
    offsets ``[-pixel_radius .. pixel_radius]``, centred on its centroid.

    Output is non-negative and sum-normalised to 1 (a discrete convolution
    kernel).
    """
    x, y, _ = centroid_center_psf(psf_x, psf_y)
    offs = np.arange(-pixel_radius, pixel_radius + 1, dtype=float)
    k = np.interp(offs, x, y, left=0.0, right=0.0)
    return normalize_nonneg_sum1(k)


# ─────────────────────────────────────────────────────────────
#  RV-space IP -> integer pixel kernel
# ─────────────────────────────────────────────────────────────

def centroid_center_ip_velocity(
    ip_v: np.ndarray, ip_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Centre an IP defined on a velocity grid so its centroid is at 0 m/s,
    and area-normalise it (integral = 1).
    Returns ``(v_centered, y_norm, centroid_shift_mps)``.
    """
    v = np.asarray(ip_v, float)
    y = np.clip(np.asarray(ip_y, float), 0.0, None)

    area = np.trapz(y, v)
    if not np.isfinite(area) or area <= 0:
        return v.copy(), y.copy(), 0.0

    centroid = np.trapz(v * y, v) / area
    v2 = v - centroid

    area2 = np.trapz(y, v2)
    if area2 > 0:
        y = y / area2

    return v2, y, float(centroid)


def local_mps_per_pixel(wave_order: np.ndarray, lo: int, hi: int) -> float:
    """
    Local dispersion, in m/s per pixel, for the chunk ``[lo, hi)`` of a
    per-order wavelength solution ``wave_order`` (length n_pix).

    Uses the endpoint-to-endpoint slope dlambda/dpixel across the chunk
    (robust to a single noisy/duplicated sample at an edge) evaluated at
    the chunk's central wavelength, then converts to velocity via
    dv/v = dlambda/lambda -> dv = c * dlambda/lambda.
    """
    n_pix = wave_order.size
    idx_lo = int(np.clip(lo, 0, n_pix - 1))
    idx_hi = int(np.clip(hi - 1, 0, n_pix - 1))

    if idx_hi <= idx_lo:
        ic = int(np.clip((lo + hi) // 2, 1, n_pix - 2))
        dlam_dpix = (wave_order[ic + 1] - wave_order[ic - 1]) / 2.0
        lam_c = wave_order[ic]
    else:
        dlam_dpix = (wave_order[idx_hi] - wave_order[idx_lo]) / (idx_hi - idx_lo)
        lam_c = wave_order[(idx_lo + idx_hi) // 2]

    if not np.isfinite(lam_c) or lam_c <= 0 or not np.isfinite(dlam_dpix):
        return np.nan

    return C_LIGHT_MPS * dlam_dpix / lam_c


def rv_ip_to_integer_pixel_kernel(
    ip_v: np.ndarray,
    ip_y: np.ndarray,
    mps_per_pix: float,
    pixel_radius: int = 7,
) -> np.ndarray:
    """
    Convert an IP sampled on an arbitrary velocity grid (m/s) into a
    discrete pixel kernel defined at integer offsets ``[-R..R]``, using the
    local dispersion ``mps_per_pix`` (m/s/pixel) to rescale the velocity
    axis into a pixel-offset axis.

    Output kernel is non-negative and sum-normalised to 1.
    """
    if not np.isfinite(mps_per_pix) or mps_per_pix == 0:
        # can't map velocity to pixels here; fall back to a delta (no smoothing)
        k = np.zeros(2 * pixel_radius + 1, dtype=float)
        k[pixel_radius] = 1.0
        return k

    v, y, _ = centroid_center_ip_velocity(ip_v, ip_y)
    x_pix = v / mps_per_pix

    offs = np.arange(-pixel_radius, pixel_radius + 1, dtype=float)
    k = np.interp(offs, x_pix, y, left=0.0, right=0.0)
    return normalize_nonneg_sum1(k)
