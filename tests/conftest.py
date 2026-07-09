"""Synthetic fixtures: a known asymmetric kernel, a comb of emission
lines, and helpers to build pixel-space and RV-space test chunks."""

from __future__ import annotations

import numpy as np
import pytest

C_LIGHT_MPS = 299_792_458.0


def _asym_kernel_continuous(dx: np.ndarray) -> np.ndarray:
    """A slightly asymmetric PSF: sum of two Gaussians of different width."""
    g1 = np.exp(-0.5 * (dx / 0.9) ** 2)
    g2 = 0.35 * np.exp(-0.5 * ((dx - 0.4) / 1.6) ** 2)
    return g1 + g2


@pytest.fixture
def true_kernel():
    """Dense, area-normalised, centroid-centred reference kernel."""
    dx = np.linspace(-8, 8, 4000)
    y = _asym_kernel_continuous(dx)
    area = np.trapz(y, dx)
    y = y / area
    centroid = np.trapz(dx * y, dx)
    dx = dx - centroid
    area = np.trapz(y, dx)
    y = y / area
    return dx, y


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def synthetic_pixel_chunk(true_kernel, rng):
    """
    A synthetic chunk: a comb of emission lines on a detector pixel grid,
    each individually convolved with ``true_kernel``, plus Poisson-like
    noise. Returns (x, y, y_err, true_kernel).
    """
    kx, ky = true_kernel
    n_pix = 400
    x = np.arange(n_pix, dtype=float)

    # Start/end the comb close to the array edges: psfcal._lines detects
    # trough-to-trough line boundaries and defaults the outermost boundary
    # to the array edge itself (there's no real neighbouring line to form a
    # trough with) -- if the first/last line sits far from x=0/x=n_pix-1,
    # that one "half-line" stack gets a wide, asymmetric offset range that
    # can bias the fit. Keeping the comb flush with the edges keeps that
    # edge effect negligible, matching how a real chunk (with margin
    # pixels) is used in practice.
    line_centres = np.arange(2, n_pix - 2, 18.3)
    y = np.full(n_pix, 50.0)  # small background

    for cen in line_centres:
        amp = 4000.0
        psi = np.interp(x - cen, kx, ky, left=0.0, right=0.0)
        y = y + amp * psi

    y_err = np.sqrt(np.abs(y) + 1.0)
    y_noisy = y + rng.normal(0.0, y_err)
    y_noisy = np.clip(y_noisy, 0.0, None)

    return x, y_noisy, y_err, (kx, ky), line_centres


@pytest.fixture
def synthetic_rv_chunk(true_kernel, rng):
    """
    A synthetic chunk with a linear wavelength solution and a comb of
    theoretical line positions, each rendered with ``true_kernel``
    (interpreted directly as a velocity-offset kernel for simplicity, with
    mps_per_pix chosen so the kernel width in pixels matches
    synthetic_pixel_chunk).
    """
    kx, ky = true_kernel  # kx in "pixel-equivalent" units
    n_pix = 400
    x = np.arange(n_pix, dtype=float)

    mps_per_pix = 800.0  # m/s per pixel, HARPS-like
    lam0 = 5000e-10  # 5000 Angstrom in metres
    dlam_dpix = lam0 * mps_per_pix / C_LIGHT_MPS
    wave_order = lam0 + x * dlam_dpix

    kv = kx * mps_per_pix  # kernel offset grid in m/s

    line_pix_centres = np.arange(2, n_pix - 2, 18.3)
    theoretical_wavelengths = np.interp(line_pix_centres, x, wave_order)

    y = np.full(n_pix, 50.0)
    for lam_theo in theoretical_wavelengths:
        amp = 4000.0
        dv = C_LIGHT_MPS * (wave_order - lam_theo) / lam_theo
        psi = np.interp(dv, kv, ky, left=0.0, right=0.0)
        y = y + amp * psi

    y_err = np.sqrt(np.abs(y) + 1.0)
    y_noisy = y + rng.normal(0.0, y_err)
    y_noisy = np.clip(y_noisy, 0.0, None)

    return dict(
        x=x, y=y_noisy, y_err=y_err,
        wave_order=wave_order,
        theoretical_wavelengths=theoretical_wavelengths,
        kernel_v=(kv, ky),
        mps_per_pix=mps_per_pix,
    )
