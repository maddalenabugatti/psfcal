"""
Item 8 — RV shift between two spectra via cross-correlation and a Gaussian
fit to the CCF peak.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

from ._continuum import smooth_continuum, CONT_MEDFILT_KSIZE
from ._kernel import C_LIGHT_MPS


def _gauss_peak(x, A, mu, tau, y0=0.0):
    """Plain (non area-normalised) Gaussian, for fitting a CCF peak."""
    return A * np.exp(-0.5 * ((x - mu) / tau) ** 2) + y0


def compute_rv_shift(
    wave1,
    flux1,
    wave2,
    flux2,
    *,
    velocity_range=(-20_000.0, 20_000.0),
    velocity_step: float = 250.0,
    fit_half_width: Optional[float] = None,
    continuum_normalize: bool = True,
    cont_kernel_size: int = CONT_MEDFILT_KSIZE,
) -> dict:
    """
    Compute the RV shift of spectrum 2 relative to spectrum 1 by
    cross-correlation, with a Gaussian fit to the CCF peak for sub-grid
    precision.

    Parameters
    ----------
    wave1, flux1 : reference spectrum (wavelength array + flux array).
    wave2, flux2 : spectrum to measure the shift of, same units as wave1.
    velocity_range : (v_min, v_max) trial velocity range to search, m/s.
    velocity_step : trial velocity grid spacing, m/s.
    fit_half_width : half-width (m/s) of the window around the CCF peak
        used for the Gaussian fit. Defaults to ``20 * velocity_step``.
    continuum_normalize : normalise both spectra with
        :func:`psfcal._continuum.smooth_continuum` before correlating.

    Returns
    -------
    dict with keys: rv (m/s), rv_err (m/s, 1-sigma), ccf_v, ccf, popt
    (Gaussian fit parameters, or None if the fit failed / was skipped).
    """
    wave1 = np.asarray(wave1, float)
    flux1 = np.asarray(flux1, float)
    wave2 = np.asarray(wave2, float)
    flux2 = np.asarray(flux2, float)

    if continuum_normalize:
        f1 = flux1 / smooth_continuum(flux1, kernel_size=cont_kernel_size)
        f2 = flux2 / smooth_continuum(flux2, kernel_size=cont_kernel_size)
    else:
        f1, f2 = flux1, flux2

    v_grid = np.arange(velocity_range[0], velocity_range[1] + velocity_step, velocity_step)
    ccf = np.full(v_grid.size, np.nan)

    a_full = f1 - 1.0
    for i, v in enumerate(v_grid):
        # trial redshift: sample flux2 at the wave1 grid Doppler-shifted by
        # +v, so v > 0 recovers "spectrum 2 is redshifted relative to
        # spectrum 1 by v" (the usual RV convention).
        query = wave1 * (1.0 + v / C_LIGHT_MPS)
        f2_interp = np.interp(query, wave2, f2, left=np.nan, right=np.nan)
        mask = np.isfinite(f2_interp) & np.isfinite(f1)
        if mask.sum() < 5:
            continue
        a = a_full[mask]
        b = f2_interp[mask] - 1.0
        denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
        ccf[i] = float(np.sum(a * b) / denom) if denom > 0 else 0.0

    if not np.any(np.isfinite(ccf)):
        raise ValueError(
            "CCF could not be computed over the requested velocity range "
            "- check that wave1 and wave2 overlap."
        )

    i_peak = int(np.nanargmax(ccf))
    v_peak = float(v_grid[i_peak])

    if fit_half_width is None:
        fit_half_width = 20.0 * velocity_step

    win = (np.abs(v_grid - v_peak) <= fit_half_width) & np.isfinite(ccf)
    if win.sum() < 4:
        return dict(rv=v_peak, rv_err=float(velocity_step), ccf_v=v_grid, ccf=ccf, popt=None)

    p0 = (float(ccf[i_peak]), v_peak, max(fit_half_width / 2.0, velocity_step), 0.0)
    try:
        popt, pcov = curve_fit(_gauss_peak, v_grid[win], ccf[win], p0=p0, maxfev=10_000)
        rv = float(popt[1])
        rv_err = float(np.sqrt(pcov[1, 1])) if np.isfinite(pcov[1, 1]) and pcov[1, 1] >= 0 else np.nan
    except Exception:
        popt = None
        rv = v_peak
        rv_err = float(velocity_step)

    return dict(rv=rv, rv_err=rv_err, ccf_v=v_grid, ccf=ccf, popt=popt)
