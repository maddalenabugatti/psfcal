"""
Item 2 — Reconstruct the instrumental profile (IP/PSF) for a single chunk,
in RV/velocity space, given a wavelength solution and the theoretical
positions of the peaks (e.g. LFC comb modes or a ThAr line list).

Ported from ``compute_psf_all_orders_mj2024.compute_chunk_psf``, extended
to reuse the shared GP engine (:mod:`psfcal._gp`) for optional outer
iteration (secondary-GP variance correction + re-centring, Eq. 20 of
Milakovic & Jethwa 2024) the same way :mod:`psfcal.psf_pixel` does.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import _gp
from . import _kernel
from . import _lines


def compute_psf_chunk_rv(
    x,
    y,
    y_err,
    wave_order,
    theoretical_wavelengths,
    *,
    n_iterations: int = 1,
    n_psf_points: int = 500,
    search_window_pix: int = 3,
    min_peak_height_fraction: float = 0.05,
    min_window_half: int = 3,
    min_quantile: float = 0.01,
    use_secondary_gp: bool = True,
    min_per_bin: int = 15,
    n_recentre_max: int = 1,
    recentre_tol: float = 1e-3,
    gp_length_scale_bounds_mps: tuple = (50.0, 5000.0),
    max_half_window: int | None = 15,
    min_window_width: int | None = 6,
    recenter: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Compute the RV-space IP/PSF for a chunk, anchoring each line to its
    theoretical wavelength rather than a fitted Gaussian centre.

    Parameters
    ----------
    x, y, y_err : 1-D arrays — pixel coordinates, flux and 1-sigma flux
        uncertainty for this chunk.
    wave_order : 1-D array, length = full order n_pix — static wavelength
        solution for the order this chunk belongs to (any consistent unit,
        e.g. metres); used both to match theoretical lines to pixels and to
        convert pixel offsets to velocity offsets.
    theoretical_wavelengths : 1-D array — theoretical line positions (same
        unit as ``wave_order``), e.g. LFC comb modes or a ThAr line list,
        not necessarily restricted to this chunk.
    n_iterations : number of outer GP-refinement iterations. 1 (default)
        reproduces a single primary-GP fit; >1 additionally applies the
        secondary-GP empirical variance correction (§4.6) and the
        re-centring loop (§3.3, Eq. 20) each iteration.
    n_psf_points : dense grid points for the returned PSF.
    search_window_pix : +/- pixels around the theory-predicted pixel to
        search for the observed peak.
    min_window_half, min_quantile : background estimation controls
        (§ Background estimation).
    max_half_window : cap, in pixels either side of each matched peak, on
        how far its stacking window can extend (see :func:`psfcal._lines.
        line_windows`) — prevents a missed/unmatched neighbouring line
        from leaking into this line's window. ``None`` disables the cap.
    min_window_width : drop lines whose window is narrower than this many
        pixels, plus always the first/last line if its window touches the
        chunk's own edge -- see :func:`psfcal._lines.complete_line_mask`.
        ``None`` disables the width check (edge-touching lines are still
        dropped).
    recenter : if True (default), shift the returned ``psf_x`` (and
        ``stacked_x``, ``theory_wavelengths``) as a final step so the
        PSF's own area (trapz) centroid sits at exactly 0 m/s -- corrects
        for a real, systematic offset between the assumed theoretical
        wavelengths and the true line positions (e.g. slightly-off LFC
        comb constants, or a genuine RV difference), which is common and
        otherwise leaves the PSF centred away from 0. ``psf_centroid`` is
        then exactly 0 by construction; ``model``/``line_amps`` are
        unaffected (translation-invariant).

    Returns
    -------
    dict with keys: theta, psf_x (m/s), psf_mean, psf_std, psf_cov,
    psf_centroid, psf_centroid_err, n_lines_used, line_windows, background,
    model, residuals, rms, line_centres (theoretical pixel positions),
    line_amps, line_amp_errors, theory_wavelengths, stacked_x, stacked_y,
    stacked_yerr, secondary_gp_bins, secondary_gp_log_S2,
    secondary_gp_g_at_stacked, history. Has the same ``model``/
    ``line_centres``/``line_windows``/``background`` keys as
    :func:`compute_psf_chunk_pixel`, so :func:`psfcal.plotting.plot_psf`
    and :func:`psfcal.plotting.plot_line_fit` both work on either result.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_err = np.asarray(y_err, dtype=float)
    y_err = np.where(y_err > 0, y_err, np.abs(y) * 0.01 + 1.0)
    wave_order = np.asarray(wave_order, dtype=float)
    theoretical_wavelengths = np.asarray(theoretical_wavelengths, dtype=float)

    peak_idx, theory_x, theory_wav = _lines.detect_lines_theory(
        x, y, wave_order, theoretical_wavelengths,
        search_window_pix=search_window_pix,
        min_peak_height_fraction=min_peak_height_fraction,
    )
    if len(peak_idx) < 2:
        raise ValueError(
            "Fewer than 2 matched theoretical lines in this chunk — cannot build a PSF."
        )
    if verbose:
        print(f"Matched {len(peak_idx)} theoretical lines.")

    windows = _lines.line_windows(y, peak_idx, max_half_window=max_half_window)
    keep = _lines.complete_line_mask(y, windows, min_window_width=min_window_width)
    peak_idx, theory_x, theory_wav, windows = (
        peak_idx[keep], theory_x[keep], theory_wav[keep], windows[keep]
    )
    if len(peak_idx) < 2:
        raise ValueError(
            "Fewer than 2 complete matched theoretical lines in this chunk after "
            "dropping incomplete/truncated windows — cannot build a PSF."
        )
    background = _lines.estimate_background(
        x, y, windows, half=min_window_half, q=min_quantile,
    )
    y_sub = y - background

    X_st, Y_st, E_st = _lines.stack_lines_rv(
        x, y_sub, y_err, wave_order, peak_idx, theory_x, theory_wav,
        max_half_window=max_half_window,
    )

    # local dispersion for this chunk, used to convert the Eq.20 half-pixel
    # re-centring anchor into velocity units
    mps_per_pix = _kernel.local_mps_per_pixel(wave_order, int(x[0]), int(x[-1]) + 1)
    half_width = 0.5 * abs(mps_per_pix) if np.isfinite(mps_per_pix) and mps_per_pix != 0 else 0.5

    # GP kernel length-scale bounds MUST be in the units of X (m/s here),
    # unlike psf_pixel's pixel-scale defaults -- see _gp.train_primary_gp.
    gp_log_scale_bounds = (
        float(np.log(gp_length_scale_bounds_mps[0])),
        float(np.log(gp_length_scale_bounds_mps[1])),
    )

    theta = None
    phi_g = {}
    X_bins_out = np.array([])
    log_S2_out = np.array([])
    g_at_X_out = np.ones_like(X_st)
    sigma_i = E_st.copy()
    psf_x_grid = X_st.copy()
    psf_mean = np.zeros_like(X_st)
    psf_std = np.zeros_like(X_st)
    psf_cov = None
    history = []

    for it in range(max(int(n_iterations), 1)):
        if verbose:
            print(f"\n  -- Iteration {it + 1} / {n_iterations} --")

        sigma_i = E_st.copy()
        # RV-anchored stacks are NOT guaranteed to be centred near 0 (unlike
        # pixel-space, where each line is centred on its own fitted
        # Gaussian) -- a real systematic offset between the assumed
        # theoretical wavelengths and the true line positions shows up as
        # a shifted stack. Seed the mean-function fit at the data's actual
        # peak so curve_fit doesn't fail to converge (see
        # _gp.train_primary_gp's mf_loc_init).
        mf_loc_init = float(X_st[np.argmax(Y_st)])
        theta = _gp.train_primary_gp(
            X_st, Y_st, sigma_i,
            gp_log_scale_bounds=gp_log_scale_bounds, mf_loc_init=mf_loc_init,
        )

        if use_secondary_gp:
            try:
                sigma_i, phi_g, X_bins_out, log_S2_out, g_at_X_out = _gp.empirical_sigma(
                    X_st, Y_st, E_st, theta, min_per_bin=min_per_bin,
                    log_lg_bounds=gp_log_scale_bounds,
                )
            except Exception as exc:
                warnings.warn(f"Secondary GP failed ({exc}); using unmodified errors.")

        theta = _gp.train_primary_gp(
            X_st, Y_st, sigma_i,
            gp_log_scale_bounds=gp_log_scale_bounds, mf_loc_init=mf_loc_init,
        )
        psf_x_grid = np.linspace(float(X_st.min()), float(X_st.max()), n_psf_points)
        psf_mean, psf_std, psf_cov = _gp.predict_primary_gp(
            theta, X_st, Y_st, sigma_i, psf_x_grid,
        )
        psf_mean = np.clip(psf_mean, 0.0, None)

        if n_iterations > 1:
            total_shift = 0.0
            converged = False
            first_step_state = None
            tol_scaled = recentre_tol * max(half_width / 0.5, 1.0)
            for rc in range(n_recentre_max):
                mf_loc_init = float(X_st[np.argmax(Y_st)])
                theta = _gp.train_primary_gp(
                    X_st, Y_st, sigma_i,
                    gp_log_scale_bounds=gp_log_scale_bounds, mf_loc_init=mf_loc_init,
                )
                psf_x_grid = np.linspace(float(X_st.min()), float(X_st.max()), n_psf_points)
                psf_mean, psf_std, psf_cov = _gp.predict_primary_gp(
                    theta, X_st, Y_st, sigma_i, psf_x_grid,
                )
                psf_mean = np.clip(psf_mean, 0.0, None)

                delta = _gp.recentre_shift(psf_x_grid, psf_mean, half_width=half_width)
                total_shift += delta
                X_st = X_st - delta

                if verbose:
                    print(f"   Re-centring step {rc + 1}: delta = {delta:.3f} m/s "
                          f"(cumulative {total_shift:.3f} m/s)")

                if rc == 0:
                    # Eq.20's fixed-point iteration is not guaranteed to
                    # converge for every PSF shape; keep the state after a
                    # single correction as a conservative fallback.
                    first_step_state = (X_st.copy(), psf_x_grid, psf_mean, psf_std, psf_cov, theta)

                if abs(delta) < tol_scaled:
                    converged = True
                    break

            if not converged and n_recentre_max > 1 and first_step_state is not None:
                warnings.warn(
                    f"Re-centring did not converge after {n_recentre_max} steps "
                    f"(last delta = {delta:.1f} m/s, cumulative shift = {total_shift:.1f} m/s); "
                    "falling back to the result after a single re-centring step.",
                    stacklevel=2,
                )
                X_st, psf_x_grid, psf_mean, psf_std, psf_cov, theta = first_step_state

        rms_tmp = float(np.sqrt(np.nanmean(
            ((Y_st - np.interp(X_st, psf_x_grid, psf_mean, left=0.0, right=0.0)) / sigma_i) ** 2
        )))
        history.append({"iter": it, "rms": rms_tmp})
        if verbose:
            print(f"   RMS = {rms_tmp:.3f} sigma")

    psf_mean, psf_std, psf_cov, psf_area_norm = _gp.area_normalise(psf_x_grid, psf_mean, psf_std, psf_cov=psf_cov)

    psf_centroid = float(
        np.trapz(psf_x_grid * psf_mean, psf_x_grid) / np.trapz(psf_mean, psf_x_grid)
    )
    dx = np.diff(psf_x_grid)
    w = np.zeros_like(psf_x_grid)
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    area = np.sum(w * psf_mean)
    grad_mu = (w / area) * (psf_x_grid - psf_centroid)
    psf_centroid_var = float(grad_mu @ psf_cov @ grad_mu)
    psf_centroid_err = float(np.sqrt(max(psf_centroid_var, 0.0)))

    # Per-line amplitude fit (centre fixed at the theoretical wavelength)
    # + summed reconstruction, so RV-space results carry a "model" and
    # "line_centres" just like compute_psf_chunk_pixel does -- lets the
    # same plotting.plot_line_fit(...) diagnostic work on either.
    model, line_amps, line_amp_errors = _lines.fit_all_lines_rv(
        x, y_sub, y_err, wave_order, windows, theory_wav,
        psf_x_grid, psf_mean, mps_per_pix,
    )
    residuals = (y_sub - model) / np.clip(y_err, 1e-30, None)
    rms = float(np.sqrt(np.nanmean(residuals ** 2)))

    recenter_shift = 0.0
    if recenter and psf_centroid != 0.0:
        # Plain post-hoc translation (not the Eq.20 loop above) so the
        # returned PSF's own centroid sits at exactly 0 m/s. theory_wav is
        # shifted by the equivalent wavelength so a re-run using the
        # returned theory_wavelengths reproduces the same, now-centred,
        # anchor -- model/line_amps (already built above) are unaffected
        # (translation-invariant).
        recenter_shift = psf_centroid
        psf_x_grid = psf_x_grid - recenter_shift
        X_st = X_st - recenter_shift
        theory_wav = theory_wav * (1.0 + recenter_shift / _kernel.C_LIGHT_MPS)
        psf_centroid = 0.0

    return dict(
        theta=theta,
        psf_x=psf_x_grid,
        psf_mean=psf_mean,
        psf_std=psf_std,
        psf_cov=psf_cov,
        psf_centroid=psf_centroid,
        psf_area_norm=psf_area_norm,
        recenter_shift=recenter_shift,
        psf_centroid_err=psf_centroid_err,
        n_lines_used=len(peak_idx),
        line_windows=windows,
        background=background,
        model=model,
        residuals=residuals,
        rms=rms,
        line_centres=theory_x,
        line_amps=line_amps,
        line_amp_errors=line_amp_errors,
        theory_wavelengths=theory_wav,
        stacked_x=X_st,
        stacked_y=Y_st,
        stacked_yerr=sigma_i,
        secondary_gp_bins=X_bins_out,
        secondary_gp_log_S2=log_S2_out,
        secondary_gp_g_at_stacked=g_at_X_out,
        history=history,
    )
