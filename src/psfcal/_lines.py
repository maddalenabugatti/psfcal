"""
Line/peak detection, background estimation, stacking and per-line refit
helpers, shared by :mod:`psfcal.psf_pixel` (pixel-space, Gaussian-centred
stacking) and :mod:`psfcal.psf_rv` (RV-space, theory-anchored stacking).

Ported from ``psf_tools_4.py`` (pixel-space) and
``compute_psf_all_orders_mj2024.py`` (theory-guided RV-space).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.signal as signal
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import curve_fit, least_squares

C_LIGHT_MPS = 299_792_458.0


def _gauss_np(x, A, mu, tau, y0=0.0):
    return A * np.exp(-0.5 * ((x - mu) / tau) ** 2) / (np.sqrt(2 * np.pi) * tau) + y0


# ─────────────────────────────────────────────────────────────
#  §4.3  Pixel-space peak / minimum detection
# ─────────────────────────────────────────────────────────────

def detect_lines_pixel(
    x: np.ndarray,
    y: np.ndarray,
    min_peak_distance: int = 3,
    min_peak_height_fraction: float = 0.05,
    max_half_window: int | None = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns ``(peak_indices, windows)``.

    ``windows`` is an ``(n_peaks, 2)`` array of inclusive ``[left, right]``
    pixel-index bounds, one independently-found window per peak (see
    :func:`line_windows`). Slice line ``k``'s data as ``x[l:r+1]`` with
    ``l, r = windows[k]``.
    """
    threshold = min_peak_height_fraction * np.nanmax(y)
    peak_idx, _ = signal.find_peaks(y, distance=min_peak_distance, height=threshold)
    if len(peak_idx) == 0:
        raise ValueError(
            "No peaks found. Try lowering min_peak_height_fraction or min_peak_distance."
        )

    windows = line_windows(y, peak_idx, max_half_window=max_half_window)
    return peak_idx, windows


# ─────────────────────────────────────────────────────────────
#  Theory-guided peak detection (RV-space)
# ─────────────────────────────────────────────────────────────

def detect_lines_theory(
    x: np.ndarray,
    y: np.ndarray,
    wave_order: np.ndarray,
    theoretical_wavelengths: np.ndarray,
    search_window_pix: int = 3,
    min_peak_height_fraction: float = 0.05,
    match_tol_pix: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Locate the observed peak nearest to each theoretical line, within
    ``search_window_pix`` pixels of its theory-predicted pixel position
    (derived from ``wave_order``, the full-order static wavelength
    solution).

    Returns ``(peak_idx, theory_x, theory_wav)`` — local peak index (into
    ``x``/``y``), the theory-predicted pixel position and the theoretical
    wavelength for each accepted line, in ascending pixel order.
    """
    threshold = min_peak_height_fraction * np.nanmax(y)
    fromm = int(x[0])

    pix_full = np.arange(len(wave_order), dtype=float)
    pix_of_theory = np.interp(theoretical_wavelengths, wave_order, pix_full)

    chunk_mask = (pix_of_theory >= x[0]) & (pix_of_theory <= x[-1])
    pix_theory_chunk = pix_of_theory[chunk_mask]
    wav_theory_chunk = theoretical_wavelengths[chunk_mask]

    peak_idx_list, theory_x_list, theory_wav_list = [], [], []

    for p_theo, lam_theo in zip(pix_theory_chunk, wav_theory_chunk):
        p_local = p_theo - fromm
        lo = int(max(0, np.floor(p_local - search_window_pix)))
        hi = int(min(len(y) - 1, np.ceil(p_local + search_window_pix)))
        if hi <= lo:
            continue
        local_peak = int(np.argmax(y[lo:hi + 1])) + lo
        if y[local_peak] < threshold:
            continue
        if peak_idx_list and abs(local_peak - peak_idx_list[-1]) < 2:
            continue
        if abs((local_peak + fromm) - p_theo) > match_tol_pix:
            continue
        peak_idx_list.append(local_peak)
        theory_x_list.append(p_theo)
        theory_wav_list.append(lam_theo)

    return (
        np.array(peak_idx_list, dtype=int),
        np.array(theory_x_list, dtype=float),
        np.array(theory_wav_list, dtype=float),
    )


def line_windows(
    y: np.ndarray,
    peak_idx: np.ndarray,
    max_half_window: int | None = 15,
) -> np.ndarray:
    """
    For each peak, independently find its own two bounding minima -- the
    trough of the raw array between it and its left neighbour, and between
    it and its right neighbour -- rather than relying on a single set of
    boundaries shared between consecutive *detected* peaks.

    This matters because if an intermediate peak fails detection (too
    faint, too close to a stronger neighbour, etc.), a shared-boundary
    scheme bridges straight over it: the "trough" between the two
    surviving detections then sits partway up the missed peak's flank,
    pulling that entire missed peak's flux into whichever line's window it
    falls in and biasing/broadening the stacked PSF (visible as extra
    bumps in the wings of the reconstructed profile). Bounding each
    window to ``max_half_window`` pixels on either side of its own peak
    (in addition to never crossing the immediate neighbouring peak
    position) caps how far such contamination can reach even when it
    occurs.

    Returns an ``(n_peaks, 2)`` array of inclusive ``[left, right]``
    pixel-index bounds.
    """
    n = len(y)
    peak_idx = np.asarray(peak_idx, dtype=int)
    windows = np.zeros((len(peak_idx), 2), dtype=int)

    for k, p in enumerate(peak_idx):
        left_bound = int(peak_idx[k - 1]) if k > 0 else 0
        right_bound = int(peak_idx[k + 1]) if k < len(peak_idx) - 1 else n - 1
        if max_half_window is not None:
            left_bound = max(left_bound, p - max_half_window)
            right_bound = min(right_bound, p + max_half_window)
        left_bound = int(np.clip(left_bound, 0, p))
        right_bound = int(np.clip(right_bound, p, n - 1))

        left = left_bound + int(np.argmin(y[left_bound:p + 1]))
        right = p + int(np.argmin(y[p:right_bound + 1]))
        windows[k] = (left, right)

    return windows


def estimate_background(
    x: np.ndarray,
    y: np.ndarray,
    windows: np.ndarray,
    half: int = 3,
    q: float = 0.01,
) -> np.ndarray:
    """
    Background estimate via PCHIP interpolation through low-quantile
    points at the midpoint between each pair of consecutive line windows.
    """
    if len(windows) < 3:
        return np.zeros_like(y)
    interior = ((windows[:-1, 1] + windows[1:, 0]) // 2).astype(int)
    min_x = x[interior].astype(float)
    min_y = np.array([
        float(np.quantile(y[max(0, i - half):min(len(y), i + half + 1)], q))
        for i in interior
    ])
    bg = PchipInterpolator(min_x, min_y)(x)
    return np.minimum(bg, y)


# ─────────────────────────────────────────────────────────────
#  §3.1  Single Gaussian fit helper
# ─────────────────────────────────────────────────────────────

def fit_gauss_single(x: np.ndarray, y: np.ndarray, y_err: np.ndarray | None = None) -> np.ndarray:
    """Quick Gaussian fit; returns (A, mu, tau, y0)."""
    i_max = int(np.argmax(y))
    p0 = (float(y[i_max]), float(x[i_max]), 1.0, 0.0)
    sigma = y_err if y_err is not None else np.ones_like(y)
    try:
        popt, _ = curve_fit(
            _gauss_np, x, y, p0=p0, sigma=sigma,
            absolute_sigma=True, maxfev=10_000,
        )
        return popt
    except Exception:
        return np.array(p0)


# ─────────────────────────────────────────────────────────────
#  §3.1  Pixel-space stacking (Eq. 10-11) — Gaussian-centred
# ─────────────────────────────────────────────────────────────

def stack_lines_pixel(
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    peak_idx: np.ndarray,
    windows: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack all lines on their (Gaussian) centres, normalised by integrated
    flux f* = sum(F_i - B_i)  (Eq. 10).  sigma_psi_hat^2 = sigma_F^2 / f*^2
    (Eq. 11).

    ``windows`` is the ``(n_peaks, 2)`` inclusive ``[left, right]`` bound
    array from :func:`line_windows` / :func:`detect_lines_pixel`.

    Lines with too few points or a non-finite/non-positive integrated flux
    are dropped. Returns ``(stacked_dx, stacked_psi_hat,
    stacked_sigma_psi_hat, centres, amplitudes, valid_windows)`` --
    ``valid_windows`` is ``windows`` filtered down to the same lines as
    ``centres``/``amplitudes``, in the same order, so callers (e.g.
    :func:`restack_lines`, :func:`fit_all_lines`) index them consistently.
    Passing the *original*, unfiltered ``windows`` back in for a later
    restack/refit call silently pairs each surviving line with the wrong
    window as soon as any earlier line was dropped.
    """
    X_all, Y_all, E_all = [], [], []
    centres, amplitudes, valid_windows = [], [], []

    for k in range(len(peak_idx)):
        l, r = windows[k]
        xk, yk, ek = x[l:r + 1], y[l:r + 1], y_err[l:r + 1]
        if len(xk) < 4:
            continue

        _, cen, *_ = fit_gauss_single(xk, yk, ek)

        f_star = float(np.sum(yk))
        if not np.isfinite(f_star) or f_star <= 0:
            f_star = max(float(np.trapz(yk, xk)), 1e-30)
        if f_star <= 0:
            continue

        dx = xk - cen
        psi_hat = yk / f_star
        sigma_psi = ek / f_star

        X_all.append(dx)
        Y_all.append(psi_hat)
        E_all.append(sigma_psi)
        centres.append(float(cen))
        amplitudes.append(float(f_star))
        valid_windows.append((l, r))

    if not X_all:
        raise ValueError("No valid lines could be stacked.")

    X_st = np.concatenate(X_all)
    Y_st = np.concatenate(Y_all)
    E_st = np.concatenate(E_all)
    order = np.argsort(X_st)

    return (X_st[order], Y_st[order], E_st[order],
            np.array(centres), np.array(amplitudes),
            np.array(valid_windows, dtype=int))


def restack_lines(
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    windows: np.ndarray,
    centres: np.ndarray,
    amplitudes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-stack (pixel-space) using externally provided (centres, amplitudes)."""
    X_all, Y_all, E_all = [], [], []

    for k in range(len(centres)):
        l, r = windows[k]
        xk, yk, ek = x[l:r + 1], y[l:r + 1], y_err[l:r + 1]
        if len(xk) < 4:
            continue
        cen = centres[k]
        f_star = amplitudes[k]
        if not np.isfinite(f_star) or f_star <= 0:
            continue

        dx = xk - cen
        X_all.append(dx)
        Y_all.append(yk / f_star)
        E_all.append(ek / f_star)

    if not X_all:
        raise ValueError("No valid lines to re-stack.")

    X_st = np.concatenate(X_all)
    Y_st = np.concatenate(Y_all)
    E_st = np.concatenate(E_all)
    order = np.argsort(X_st)
    return X_st[order], Y_st[order], E_st[order]


# ─────────────────────────────────────────────────────────────
#  RV-space stacking — theory-anchored (dx = c (lambda - lambda_theory) / lambda_theory)
# ─────────────────────────────────────────────────────────────

def stack_lines_rv(
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    wave_order: np.ndarray,
    peak_idx: np.ndarray,
    theory_x: np.ndarray,
    theory_wav: np.ndarray,
    max_half_window: int | None = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack all lines relative to their theoretical wavelength, using the
    true wavelength solution to convert pixel offsets to velocity offsets:

        dv = c (lambda(pixel) - lambda_theory) / lambda_theory

    Each line's window is its own independently-found pair of bounding
    minima (see :func:`line_windows`), capped to ``max_half_window``
    pixels either side of its peak so a missed neighbouring line can't
    leak into it. Returns (stacked_dv, stacked_psi_hat, stacked_sigma_psi_hat).
    """
    if len(peak_idx) < 2:
        raise ValueError("Need at least 2 matched lines to stack.")

    windows = line_windows(y, peak_idx, max_half_window=max_half_window)
    pix_full = np.arange(len(wave_order), dtype=float)

    X_all, Y_all, E_all = [], [], []

    for k in range(len(peak_idx)):
        l, r = windows[k]
        xk, yk, ek = x[l:r + 1], y[l:r + 1], y_err[l:r + 1]
        if len(xk) < 4 or xk[0] <= x[0] or xk[-1] >= x[-1]:
            continue

        lam_theo = theory_wav[k]
        wav_k = np.interp(xk, pix_full, wave_order)
        dv_k = C_LIGHT_MPS * (wav_k - lam_theo) / lam_theo

        f_star = float(np.sum(yk))
        if not np.isfinite(f_star) or f_star <= 0:
            f_star = max(float(np.trapz(yk, xk)), 1e-30)
        if f_star <= 0:
            continue

        X_all.append(dv_k)
        Y_all.append(yk / f_star)
        E_all.append(ek / f_star)

    if not X_all:
        raise ValueError("No valid lines could be stacked.")

    X_st = np.concatenate(X_all)
    Y_st = np.concatenate(Y_all)
    E_st = np.concatenate(E_all)
    order = np.argsort(X_st)
    return X_st[order], Y_st[order], E_st[order]


# ─────────────────────────────────────────────────────────────
#  §3.1  Weight scheme  (Eq. 16)
# ─────────────────────────────────────────────────────────────

def weights(dx: np.ndarray, x1: float = 2.5, x2: float = 5.0) -> np.ndarray:
    """
    Tapered weights (Eq. 16): w=1 for |dx|<=x1, linearly tapered to 0
    between x1 and x2, w=0 beyond x2.
    """
    adx = np.abs(dx)
    w = np.where(adx <= x1, 1.0,
                 np.where(adx <= x2, (x2 - adx) / (x2 - x1), 0.0))
    return w


# ─────────────────────────────────────────────────────────────
#  §3.1  Per-line refit with empirical IP (Eq. 13-15) — pixel-space only
# ─────────────────────────────────────────────────────────────

def fit_single_line(
    xk: np.ndarray,
    yk: np.ndarray,
    ek: np.ndarray,
    psi_interp,
    cen_init: float,
    amp_init: float,
    centre_bounds: float = 2.0,
    x1: float = 2.5,
    x2: float = 5.0,
) -> Tuple[float, float, float, float]:
    """
    Fit (A*, x*) for one line using the current IP psi as template.
    Model (Eq. 13): I(x; A*, x*) = A* . psi(x - x*).
    Returns (new_amp, new_cen, amp_err, cen_err).
    """
    def _residuals(params):
        amp, cen = params
        dx = xk - cen
        psi_val = np.clip(psi_interp(dx), 0.0, None)
        model = amp * psi_val
        w = weights(dx, x1, x2)
        sigma = np.clip(ek, 1e-30, None)
        return w * (yk - model) / sigma

    p0 = np.array([amp_init, cen_init])
    lo = np.array([0.0, cen_init - centre_bounds])
    hi = np.array([np.inf, cen_init + centre_bounds])

    amp_err = np.nan
    cen_err = np.nan

    try:
        sol = least_squares(
            _residuals, p0, bounds=(lo, hi), method="trf",
            ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=5000,
        )
        new_amp, new_cen = sol.x

        n_pts = int(np.sum(weights(xk - new_cen, x1, x2) > 0))
        n_par = 2
        dof = max(n_pts - n_par, 1)
        J = sol.jac
        JTJ = J.T @ J
        try:
            cov = np.linalg.inv(JTJ) * (2.0 * sol.cost / dof)
            errs = np.sqrt(np.maximum(np.diag(cov), 0.0))
            amp_err, cen_err = errs[0], errs[1]
        except np.linalg.LinAlgError:
            pass

    except Exception:
        new_amp, new_cen = amp_init, cen_init

    if not (np.isfinite(new_amp) and new_amp > 0 and np.isfinite(new_cen)):
        new_amp, new_cen = amp_init, cen_init
        amp_err = np.nan
        cen_err = np.nan

    return float(new_amp), float(new_cen), float(amp_err), float(cen_err)


def fit_all_lines(
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    psf_x: np.ndarray,
    psf_mean: np.ndarray,
    windows: np.ndarray,
    centres_init: np.ndarray,
    amplitudes_init: np.ndarray,
    centre_bounds: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Independently refit each line and return the summed spectrum model.
    PSF mean is interpolated with cubic splines; outside the training
    domain -> zero.
    """
    psi_interp = interp1d(
        psf_x, psf_mean, kind="cubic", bounds_error=False, fill_value=0.0,
    )

    n_lines = len(centres_init)
    new_amps = np.array(amplitudes_init, dtype=float)
    new_cens = np.array(centres_init, dtype=float)
    amp_errs = np.full(n_lines, np.nan)
    cen_errs = np.full(n_lines, np.nan)

    for k in range(n_lines):
        l, r = windows[k]
        xk, yk, ek = x[l:r + 1], y[l:r + 1], y_err[l:r + 1]
        if len(xk) < 3:
            continue
        new_amps[k], new_cens[k], amp_errs[k], cen_errs[k] = fit_single_line(
            xk, yk, ek, psi_interp,
            cen_init=centres_init[k],
            amp_init=amplitudes_init[k],
            centre_bounds=centre_bounds,
        )

    model = np.zeros_like(x, dtype=float)
    for k in range(n_lines):
        dx = x - new_cens[k]
        model += new_amps[k] * np.clip(psi_interp(dx), 0.0, None)

    return new_amps, new_cens, model, amp_errs, cen_errs


# ─────────────────────────────────────────────────────────────
#  RV-space per-line amplitude fit + summed model (theory-anchored)
# ─────────────────────────────────────────────────────────────

def fit_all_lines_rv(
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    wave_order: np.ndarray,
    windows: np.ndarray,
    theory_wav: np.ndarray,
    psf_v: np.ndarray,
    psf_mean: np.ndarray,
    mps_per_pix: float,
    x1: float = 2.5,
    x2: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the summed-per-line reconstruction for an RV-space PSF.

    Unlike :func:`fit_all_lines` (pixel space), each line's centre is
    *fixed* at its theoretical wavelength (that's the whole point of the
    RV-anchored method), so only the per-line amplitude is fit -- a
    closed-form weighted least-squares problem, no iterative optimizer
    needed. Uses the same tapered weight scheme as Eq. 16, with the
    +-2.5/+-5 pixel window scaled to velocity via ``mps_per_pix``.

    Returns ``(model, amplitudes, amplitude_errors)``.
    """
    psi_interp = interp1d(
        psf_v, psf_mean, kind="cubic", bounds_error=False, fill_value=0.0,
    )
    pix_full = np.arange(len(wave_order), dtype=float)
    x1v, x2v = x1 * abs(mps_per_pix), x2 * abs(mps_per_pix)

    n_lines = len(theory_wav)
    amps = np.zeros(n_lines)
    amp_errs = np.full(n_lines, np.nan)
    model = np.zeros_like(x, dtype=float)

    for k in range(n_lines):
        l, r = windows[k]
        xk, yk, ek = x[l:r + 1], y[l:r + 1], y_err[l:r + 1]
        if len(xk) < 2:
            continue

        lam_theo = theory_wav[k]
        wav_k = np.interp(xk, pix_full, wave_order)
        dv_k = C_LIGHT_MPS * (wav_k - lam_theo) / lam_theo
        psi_k = np.clip(psi_interp(dv_k), 0.0, None)
        w_k = weights(dv_k, x1v, x2v)
        sigma_k = np.clip(ek, 1e-30, None)

        num = np.sum(w_k * psi_k * yk / sigma_k ** 2)
        den = np.sum(w_k * psi_k ** 2 / sigma_k ** 2)
        if den > 0:
            amps[k] = num / den
            amp_errs[k] = 1.0 / np.sqrt(den)

        model[l:r + 1] += amps[k] * psi_k

    return model, amps, amp_errs
