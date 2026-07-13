"""
Items 6, 7 — Forward-convolve a chunk's spectrum with a PSF (pixel- or
RV-space). This is the forward operator that Richardson-Lucy deconvolution
(:mod:`psfcal.deconvolve`) approximately inverts, so it mirrors that
module's kernel-preparation exactly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import fftconvolve

from . import _kernel

PIXEL_RADIUS_DEFAULT = 7


def convolve_chunk_pixel(
    flux,
    psf_x,
    psf_y,
    *,
    pixel_radius: int = PIXEL_RADIUS_DEFAULT,
    recenter: bool = True,
) -> np.ndarray:
    """
    Forward-convolve a chunk's flux with a pixel-space PSF.

    ``recenter`` : if True (default), shift ``psf_x``/``psf_y`` to their
    own centroid before building the discrete kernel; set False to use
    the PSF exactly as given (e.g. computed with
    ``compute_psf_chunk_pixel(..., recenter=False)``).
    """
    flux = np.asarray(flux, float)

    if not np.isfinite(psf_y).any() or np.nanmax(psf_y) <= 0:
        k_pix = np.zeros(2 * pixel_radius + 1, dtype=float)
        k_pix[pixel_radius] = 1.0
    else:
        k_pix = _kernel.psf_to_integer_pixel_kernel(psf_x, psf_y, pixel_radius=pixel_radius, recenter=recenter)

    pad = len(k_pix) // 2
    flux_pad = _kernel.reflect_pad(flux, pad)
    conv_pad = fftconvolve(flux_pad, k_pix, mode="same")
    return conv_pad[pad:-pad] if pad > 0 else conv_pad


def convolve_chunk_rv(
    flux,
    ip_v,
    ip_y,
    *,
    wave_chunk=None,
    mps_per_pix: Optional[float] = None,
    pixel_radius: int = PIXEL_RADIUS_DEFAULT,
    recenter: bool = True,
) -> np.ndarray:
    """
    Forward-convolve a chunk's flux with an RV-space IP.

    Either pass ``mps_per_pix`` directly, or ``wave_chunk`` — the local
    wavelength solution over this chunk (same length as ``flux``) — from
    which the local dispersion is derived.

    ``recenter`` : if True (default), shift ``ip_v``/``ip_y`` to their own
    centroid (in velocity space) before building the discrete kernel; set
    False to use the IP exactly as given (e.g. computed with
    ``compute_psf_chunk_rv(..., recenter=False)``).
    """
    flux = np.asarray(flux, float)

    if mps_per_pix is None:
        if wave_chunk is None:
            raise ValueError("Provide either mps_per_pix or wave_chunk.")
        wave_chunk = np.asarray(wave_chunk, float)
        mps_per_pix = _kernel.local_mps_per_pixel(wave_chunk, 0, len(wave_chunk))

    if not np.isfinite(ip_y).any() or np.nanmax(ip_y) <= 0:
        k_pix = np.zeros(2 * pixel_radius + 1, dtype=float)
        k_pix[pixel_radius] = 1.0
    else:
        k_pix = _kernel.rv_ip_to_integer_pixel_kernel(
            ip_v, ip_y, mps_per_pix, pixel_radius=pixel_radius, recenter=recenter,
        )

    pad = len(k_pix) // 2
    flux_pad = _kernel.reflect_pad(flux, pad)
    conv_pad = fftconvolve(flux_pad, k_pix, mode="same")
    return conv_pad[pad:-pad] if pad > 0 else conv_pad
