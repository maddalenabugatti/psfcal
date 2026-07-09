"""
Items 3, 4, 5 — Richardson-Lucy deconvolution of a chunk (pixel- or
RV-space PSF), and of a full order given per-chunk PSFs (overlap-add).

Ported from ``deconvolve_spectra_lucy.py`` and ``deconvolve_spectra_lucy_rv.py``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import _kernel
from . import _continuum
from ._rl import rl_deconvolve_1d, RL_ITERS_DEFAULT, L2_DAMP_DEFAULT

PIXEL_RADIUS_DEFAULT = 7
OVERLAP_PIX_DEFAULT = 12


def _continuum_split(flux: np.ndarray, continuum_normalize: bool, cont_kernel_size: int):
    if continuum_normalize:
        cont = _continuum.smooth_continuum(flux, kernel_size=cont_kernel_size)
        return flux / cont, cont
    return flux, np.ones_like(flux)


def deconvolve_chunk_pixel(
    flux,
    psf_x,
    psf_y,
    *,
    pixel_radius: int = PIXEL_RADIUS_DEFAULT,
    rl_iters: int = RL_ITERS_DEFAULT,
    l2_damp: float = L2_DAMP_DEFAULT,
    overlap_pix: Optional[int] = None,
    continuum_normalize: bool = False,
    cont_kernel_size: int = _continuum.CONT_MEDFILT_KSIZE,
) -> np.ndarray:
    """
    Richardson-Lucy deconvolve a single chunk's flux with a pixel-space PSF.

    Parameters
    ----------
    flux : 1-D flux array for this chunk.
    psf_x, psf_y : PSF sampled on an arbitrary pixel-offset grid (e.g. from
        :func:`psfcal.compute_psf_chunk_pixel`).
    overlap_pix : if given, the flux is internally reflect-padded by this
        many samples before deconvolving (reduces edge artefacts for a
        short, isolated chunk with no neighbouring data available); the
        padding is cropped back off before returning.
    """
    flux = np.asarray(flux, float)

    if not np.isfinite(psf_y).any() or np.nanmax(psf_y) <= 0:
        k_pix = np.array([1.0])
    else:
        k_pix = _kernel.psf_to_integer_pixel_kernel(psf_x, psf_y, pixel_radius=pixel_radius)

    work, cont = _continuum_split(flux, continuum_normalize, cont_kernel_size)

    pad = int(overlap_pix) if overlap_pix else 0
    work_pad = _kernel.reflect_pad(work, pad) if pad > 0 else work

    deconv_pad = rl_deconvolve_1d(work_pad, k_pix, n_iter=rl_iters, l2_damp=l2_damp)
    deconv = deconv_pad[pad: pad + len(flux)] if pad > 0 else deconv_pad

    return deconv * cont


def deconvolve_chunk_rv(
    flux,
    ip_v,
    ip_y,
    *,
    wave_chunk=None,
    mps_per_pix: Optional[float] = None,
    pixel_radius: int = PIXEL_RADIUS_DEFAULT,
    rl_iters: int = RL_ITERS_DEFAULT,
    l2_damp: float = L2_DAMP_DEFAULT,
    overlap_pix: Optional[int] = None,
    continuum_normalize: bool = False,
    cont_kernel_size: int = _continuum.CONT_MEDFILT_KSIZE,
) -> np.ndarray:
    """
    Richardson-Lucy deconvolve a single chunk's flux with an RV-space IP.

    Either pass ``mps_per_pix`` directly, or ``wave_chunk`` — the local
    wavelength solution over this chunk (same length as ``flux``) — from
    which the local dispersion is derived.
    """
    flux = np.asarray(flux, float)

    if mps_per_pix is None:
        if wave_chunk is None:
            raise ValueError("Provide either mps_per_pix or wave_chunk.")
        wave_chunk = np.asarray(wave_chunk, float)
        mps_per_pix = _kernel.local_mps_per_pixel(wave_chunk, 0, len(wave_chunk))

    if not np.isfinite(ip_y).any() or np.nanmax(ip_y) <= 0:
        k_pix = np.array([1.0])
    else:
        k_pix = _kernel.rv_ip_to_integer_pixel_kernel(ip_v, ip_y, mps_per_pix, pixel_radius=pixel_radius)

    work, cont = _continuum_split(flux, continuum_normalize, cont_kernel_size)

    pad = int(overlap_pix) if overlap_pix else 0
    work_pad = _kernel.reflect_pad(work, pad) if pad > 0 else work

    deconv_pad = rl_deconvolve_1d(work_pad, k_pix, n_iter=rl_iters, l2_damp=l2_damp)
    deconv = deconv_pad[pad: pad + len(flux)] if pad > 0 else deconv_pad

    return deconv * cont


def deconvolve_order(
    order_flux,
    psi_cube,
    offsets_cube,
    chunk_edges,
    *,
    scale: str = "pixel",
    wave_order=None,
    ok_chunks=None,
    rl_iters: int = RL_ITERS_DEFAULT,
    l2_damp: float = L2_DAMP_DEFAULT,
    pixel_radius: int = PIXEL_RADIUS_DEFAULT,
    overlap_pix: int = OVERLAP_PIX_DEFAULT,
    continuum_normalize: bool = True,
    cont_kernel_size: int = _continuum.CONT_MEDFILT_KSIZE,
) -> np.ndarray:
    """
    Deconvolve a full order with a different PSF per chunk, using
    overlap-add stitching with a cosine taper across chunk boundaries.

    Parameters
    ----------
    order_flux : 1-D array, length n_pix.
    psi_cube : (n_chunks, n_offsets) PSF sampled per chunk.
    offsets_cube : (n_chunks, n_offsets) offset grid per chunk (pixel
        offsets if ``scale="pixel"``, velocity offsets in m/s if
        ``scale="rv"``).
    chunk_edges : (n_chunks, 2) detector pixel edges (no margin).
    scale : "pixel" or "rv".
    wave_order : required if ``scale="rv"`` — static per-order wavelength
        solution, length n_pix, used to derive the local dispersion for
        each chunk.
    ok_chunks : optional (n_chunks,) validity mask.
    """
    scale = scale.lower()
    if scale not in ("pixel", "rv", "velocity"):
        raise ValueError("scale must be 'pixel' or 'rv'")
    if scale in ("rv", "velocity") and wave_order is None:
        raise ValueError("wave_order is required when scale='rv'")

    y = np.asarray(order_flux, float)
    n_pix = y.size
    n_chunks = int(psi_cube.shape[0])

    overlap_pix = int(max(overlap_pix, pixel_radius))
    work, cont = _continuum_split(y, continuum_normalize, cont_kernel_size)

    out = np.zeros(n_pix, dtype=float)
    wsum = np.zeros(n_pix, dtype=float)

    for ci in range(n_chunks):
        if ok_chunks is not None and int(ok_chunks[ci]) == 0:
            continue

        lo = int(chunk_edges[ci, 0])
        hi = int(chunk_edges[ci, 1])
        lo_exp = max(0, lo - overlap_pix)
        hi_exp = min(n_pix, hi + overlap_pix)

        y_chunk = work[lo_exp:hi_exp]

        offs = offsets_cube[ci]
        psi = psi_cube[ci]

        if not np.isfinite(psi).any() or np.nanmax(psi) <= 0:
            k_pix = np.array([1.0])
        elif scale == "pixel":
            k_pix = _kernel.psf_to_integer_pixel_kernel(offs, psi, pixel_radius=pixel_radius)
        else:
            mps_per_pix = _kernel.local_mps_per_pixel(wave_order, lo, hi)
            k_pix = _kernel.rv_ip_to_integer_pixel_kernel(offs, psi, mps_per_pix, pixel_radius=pixel_radius)

        x_chunk = rl_deconvolve_1d(y_chunk, k_pix, n_iter=rl_iters, l2_damp=l2_damp)

        left_pad = lo - lo_exp
        core = x_chunk[left_pad: left_pad + (hi - lo)]

        n_core = hi - lo
        w = np.ones(n_core, dtype=float)
        t = min(overlap_pix, n_core // 2)
        if t > 0:
            win = _kernel.cosine_window(2 * t)
            w[:t] *= win[:t]
            w[-t:] *= win[-t:]

        out[lo:hi] += core * w
        wsum[lo:hi] += w

    out /= np.maximum(wsum, 1e-12)
    return out * cont
