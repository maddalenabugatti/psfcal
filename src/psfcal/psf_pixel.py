"""
Item 1 — Reconstruct the instrumental profile (IP/PSF) for a single chunk,
in pixel space, following

    Milakovic & Jethwa, A&A 684, A38 (2024)   [MJ24]

Ported from ``psf_tools_4.compute_psf``.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import _gp
from . import _lines

PIX2MS = 820.0  # 1 HARPS pixel ~ 820 m/s, used only for line RV-error reporting


def compute_psf_chunk_pixel(
    x,
    y,
    y_err=None,
    *,
    n_iterations: int = 1,
    min_peak_distance: int = 3,
    min_peak_height_fraction: float = 0.05,
    n_psf_points: int = 800,
    n_recentre_max: int = 1,
    recentre_tol: float = 1e-3,
    use_secondary_gp: bool = True,
    min_per_bin: int = 15,
    centre_bounds: float = 2.0,
    subtract_background: bool = False,
    min_window_half: int = 3,
    min_quantile: float = 0.01,
    max_half_window: int | None = 15,
    min_window_width: int | None = 6,
    recenter: bool = True,
    sigma_clip_nsigma: float | None = None,
    sigma_clip_max_passes: int = 3,
    student_t_dof: float | None = None,
    student_t_em_iters: int = 5,
    verbose: bool = False,
) -> dict:
    """
    Compute the pixel-space IP/PSF for a chunk from a background-subtracted
    1-D spectrum, following MJ24 section by section.

    The chunk is auto-scanned for individual emission/absorption lines
    (§4.3), which are Gaussian-centred and stacked (§3.1, Eq. 10-11) to
    build sub-pixel sampling of psi_hat(dx). A primary GP (§3.2, Eq. 17-19)
    is fit to the stack, with an optional secondary GP for empirical
    variance correction (§4.6) and iterative re-centring (§3.3, Eq. 20).

    Parameters
    ----------
    x, y   : 1-D arrays — pixel coordinates and flux for this chunk.
    y_err  : 1-D array — 1-sigma flux uncertainties (Poisson estimate if omitted).
    n_iterations : number of outer (line-refit <-> re-stack) iterations (§4.7).
        Default 1 (single pass: stack once, fit the GP, refit line
        centres/amplitudes once against the resulting PSF).
    n_psf_points  : dense grid points for the returned PSF (paper uses 800).
    n_recentre_max, recentre_tol : Eq.20 fixed-point re-centring loop
        controls (§3.3) -- not guaranteed to converge for every PSF shape;
        see ``recenter`` for a simpler, always-exact alternative.
    use_secondary_gp, min_per_bin : §4.6 empirical variance correction controls.
    centre_bounds : max allowed centre shift per line per iteration (pix).
    recenter : if True (default), shift the returned ``psf_x`` (and
        ``stacked_x``, ``line_centres``) as a final step so the PSF's own
        area (trapz) centroid sits at exactly 0 -- a plain post-hoc
        translation, independent of and more robust than the Eq.20 loop
        above. ``psf_centroid`` is then exactly 0 by construction; the
        reconstructed ``model`` is unaffected (translation-invariant).
    subtract_background : if True, estimate and subtract a smooth background
        (low-quantile interpolation through the trough minima, same method
        used internally by :func:`psfcal.compute_psf_chunk_rv`) before
        stacking. The method assumes background-subtracted input (per the
        paper); leave ``y`` pre-subtracted and this ``False`` (default, for
        exact fidelity to the original reference implementation), OR pass
        raw flux with ``subtract_background=True``. Passing raw,
        non-background-subtracted flux with this left ``False`` will bias
        the reconstructed PSF.
    min_window_half, min_quantile : background estimation controls, only
        used when ``subtract_background=True``.
    max_half_window : cap, in pixels either side of each detected peak, on
        how far its stacking window can extend (see :func:`psfcal._lines.
        line_windows`). Prevents a missed neighbouring line (or a
        genuinely wide gap) from leaking into this line's window and
        showing up as extra bumps in the wings of the reconstructed PSF.
        ``None`` disables the cap (window extends to the true trough
        against the neighbouring *detected* peak, whatever the distance).
    min_window_width : drop lines whose window is narrower than this many
        pixels (usually an unusually close neighbour truncating the
        window well short of a real trough, which biases that line's
        integrated-flux normalisation, Eq. 10) -- also always drops the
        first/last line if its window touches the chunk's own edge (an
        unobserved outer flank), regardless of this setting. ``None``
        disables the width check (edge-touching lines are still dropped).
    sigma_clip_nsigma, sigma_clip_max_passes : the primary GP fit (and its
        secondary-GP empirical-variance correction) has no explicit outlier
        rejection anywhere -- a single contaminating stacked point (a faint
        blend that slipped past ``max_half_window``, a cosmic ray, imperfect
        background subtraction) is only downweighted if OTHER nearby points
        also scatter, which doesn't reliably happen in the low-S/N wings,
        where a lone bad point can visibly pull the fit. After each
        recentring pass, this does an iterative MAD-based sigma-clip on the
        stacked residuals against the current GP fit (same robust-clip
        convention used elsewhere in this project, e.g. RV sigma-clipping),
        removing points beyond ``sigma_clip_nsigma`` robust-sigma and
        refitting the GP on the survivors, up to ``sigma_clip_max_passes``
        times (stops early once nothing more gets clipped). Pass
        ``sigma_clip_nsigma=None`` to disable and recover the exact previous
        behaviour.

        Default is None (off) -- opt in deliberately, same as student_t_dof
        below, rather than running unconditionally on every chunk. Effect is
        genuinely chunk-dependent, not uniformly beneficial: on one real
        HARPS LFC chunk (order 65, a genuinely noisy one), 8.0 removed 32
        points (presumably real outliers) while leaving the core-region RMS
        almost unchanged (1.50 sigma, +2.7% vs unclipped) -- 5.0 was worse,
        removing 67 points and *increasing* core RMS by 68% (1.46 -> 2.45
        sigma), clearly removing real signal, not just contamination. But on
        a different, unusually clean/high-S/N chunk (order 70, chunk 15 of
        a separate exposure), 8.0 nearly TRIPLED the core RMS (1.27 -> 3.57
        sigma) relative to no clipping at all: the chunk's own intrinsic
        scatter was tight enough that the iterative clip-then-refit loop
        chased a small systematic (not random) shape mismatch and clipped
        points that weren't really outliers. A tighter clip is not
        automatically better, and neither is clipping at all -- always check
        core-region RMS before vs after on the actual chunk you're using
        before enabling this.
    student_t_dof, student_t_em_iters : smoother alternative/complement to
        the hard sigma-clip above -- an EM/IRLS approximation to Student-t
        process regression (the primary GP's own Gaussian kernel is kept
        as-is; only the per-point effective noise fed into it changes).
        Each round: fit the GP, compute standardised residuals r_i against
        the ORIGINAL sigma, then reweight w_i = (dof+1)/(dof+r_i^2) and
        inflate that point's effective sigma by 1/sqrt(w_i) before the next
        fit. Smaller ``student_t_dof`` (e.g. 3-4) means heavier tails --
        more aggressive downweighting -- while every point stays smoothly
        downweighted rather than hard-included/excluded, so a genuinely
        informative point that's merely somewhat noisy doesn't get thrown
        away the way a hard clip would; ``student_t_dof=None`` (default)
        disables this and only sigma-clipping (if enabled) applies. Runs
        AFTER sigma-clipping, on whatever points survived it.

    Returns
    -------
    dict with keys: theta, psf_x, psf_mean, psf_std, psf_cov, psf_centroid,
    psf_centroid_err, model, residuals, rms, background, line_centres,
    line_amps, line_centre_errors, line_amp_errors, line_rv_errors,
    stacked_x, stacked_y, stacked_yerr, secondary_gp_bins,
    secondary_gp_log_S2, secondary_gp_g_at_stacked, history.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if y_err is None:
        y_err = np.sqrt(np.abs(y) + 1.0)
        warnings.warn("y_err not provided; using sqrt(|y|+1) as Poisson estimate.")
    y_err = np.asarray(y_err, dtype=float)
    y_err = np.where(y_err > 0, y_err, np.abs(y) * 0.01 + 1.0)

    peak_idx, windows = _lines.detect_lines_pixel(
        x, y,
        min_peak_distance=min_peak_distance,
        min_peak_height_fraction=min_peak_height_fraction,
        max_half_window=max_half_window,
        min_window_width=min_window_width,
    )
    if verbose:
        print(f"Detected {len(peak_idx)} lines.")

    if subtract_background:
        background = _lines.estimate_background(
            x, y, windows, half=min_window_half, q=min_quantile,
        )
        y = y - background
    else:
        background = np.zeros_like(y)

    X_st, Y_st, E_st, centres, amplitudes, windows = _lines.stack_lines_pixel(
        x, y, y_err, peak_idx, windows,
    )
    # stack_lines_pixel drops lines with too few points or an invalid
    # integrated flux; `windows` is reassigned to its filtered, aligned
    # return value so every later fit_all_lines/restack_lines call indexes
    # windows[k] against the SAME line as centres[k]/amplitudes[k] -- using
    # the original, unfiltered windows here would silently mis-pair lines
    # (and desync array lengths) as soon as any line was dropped.

    theta = None
    model = np.zeros_like(y)
    amp_errors = np.full(len(centres), np.nan)
    cen_errors = np.full(len(centres), np.nan)
    history = []

    phi_g = {}
    X_bins_out = np.array([])
    log_S2_out = np.array([])
    g_at_X_out = np.ones_like(X_st)
    sigma_i = E_st.copy()
    psf_x_grid = X_st.copy()
    psf_mean = np.zeros_like(X_st)
    psf_std = np.zeros_like(X_st)
    psf_cov = None

    for it in range(max(int(n_iterations), 1)):
        if verbose:
            print(f"\n  -- Iteration {it + 1} / {n_iterations} --")

        sigma_i = E_st.copy()
        theta = _gp.train_primary_gp(X_st, Y_st, sigma_i)

        if use_secondary_gp:
            sigma_i, phi_g, X_bins_out, log_S2_out, g_at_X_out = _gp.empirical_sigma(
                X_st, Y_st, E_st, theta, min_per_bin=min_per_bin,
            )
            if verbose:
                print(f"   Secondary GP done.  g range: "
                      f"[{g_at_X_out.min():.2f}, {g_at_X_out.max():.2f}]")

        total_shift = 0.0
        converged = False
        first_step_state = None
        for rc in range(n_recentre_max):
            theta = _gp.train_primary_gp(X_st, Y_st, sigma_i)

            psf_x_grid = np.linspace(float(X_st.min()), float(X_st.max()), n_psf_points)
            psf_mean, psf_std, psf_cov = _gp.predict_primary_gp(
                theta, X_st, Y_st, sigma_i, psf_x_grid,
            )
            psf_mean = np.clip(psf_mean, 0.0, None)

            delta = _gp.recentre_shift(psf_x_grid, psf_mean)
            total_shift += delta
            X_st = X_st - delta

            if verbose:
                print(f"   Re-centring step {rc + 1}: delta = {delta:.6f} pix "
                      f"(cumulative {total_shift:.6f} pix)")

            if rc == 0:
                # Eq.20's fixed-point iteration is not guaranteed to converge
                # for every PSF shape (observed to drift indefinitely for
                # sufficiently asymmetric profiles); keep the state after a
                # single correction as a conservative fallback.
                first_step_state = (X_st.copy(), psf_x_grid, psf_mean, psf_std, psf_cov, theta)

            if abs(delta) < recentre_tol:
                if verbose:
                    print(f"   Re-centring converged after {rc + 1} step(s).")
                converged = True
                break

        if not converged and n_recentre_max > 1 and first_step_state is not None:
            warnings.warn(
                f"Re-centring did not converge after {n_recentre_max} steps "
                f"(last delta = {delta:.4f}, cumulative shift = {total_shift:.4f}); "
                "falling back to the result after a single re-centring step. "
                "Pass n_recentre_max=1 explicitly to silence this and always "
                "use that behaviour, or inspect delta with verbose=True.",
                stacklevel=2,
            )
            X_st, psf_x_grid, psf_mean, psf_std, psf_cov, theta = first_step_state

        # ── Robustify against outliers -- see sigma_clip_nsigma's docstring
        #    entry for why the secondary-GP variance correction alone doesn't
        #    reliably catch a lone contaminating point, especially in the
        #    low-S/N wings. Iterative MAD-based sigma-clip on the stacked
        #    residuals against the current GP fit, refitting on survivors.
        n_clipped_this_iter = 0
        if sigma_clip_nsigma is not None:
            for _ in range(max(int(sigma_clip_max_passes), 1)):
                psf_at_Xst = np.interp(X_st, psf_x_grid, psf_mean)
                resid = (Y_st - psf_at_Xst) / np.clip(sigma_i, 1e-30, None)
                center = np.median(resid)
                mad = np.median(np.abs(resid - center))
                sigma_robust = 1.4826 * mad
                if not np.isfinite(sigma_robust) or sigma_robust <= 0:
                    break
                keep = np.abs(resid - center) <= sigma_clip_nsigma * sigma_robust
                n_bad = int((~keep).sum())
                if n_bad == 0:
                    break
                X_st, Y_st, E_st, sigma_i = X_st[keep], Y_st[keep], E_st[keep], sigma_i[keep]
                n_clipped_this_iter += n_bad
                theta = _gp.train_primary_gp(X_st, Y_st, sigma_i)
                psf_mean, psf_std, psf_cov = _gp.predict_primary_gp(
                    theta, X_st, Y_st, sigma_i, psf_x_grid,
                )
                psf_mean = np.clip(psf_mean, 0.0, None)
            if n_clipped_this_iter:
                # X_st shrank -- re-tighten the evaluation grid to the surviving
                # points' own (densely-supported) range and re-predict on it, so
                # the returned/plotted PSF doesn't silently extrapolate past
                # where any clipped point used to be. Uses the 99th percentile
                # of |X_st|, not the bare max: the max is just whichever single
                # surviving point happens to sit farthest out, which can be an
                # isolated straggler with a real data gap between it and the
                # rest of the stack -- extending the grid all the way to it
                # produces a wide, technically-honest but visually-misleading
                # uncertainty flare over that unsupported gap. Kept SYMMETRIC
                # around 0 (not plain [X_st.min(), X_st.max()]) -- clipping
                # rarely removes the same number of points from both wings, so
                # an asymmetric range would make the curve look shifted even
                # though the fit itself hasn't actually moved.
                half_range = float(np.percentile(np.abs(X_st), 99))
                psf_x_grid = np.linspace(-half_range, half_range, n_psf_points)
                psf_mean, psf_std, psf_cov = _gp.predict_primary_gp(
                    theta, X_st, Y_st, sigma_i, psf_x_grid,
                )
                psf_mean = np.clip(psf_mean, 0.0, None)
            if verbose and n_clipped_this_iter:
                print(f"   Sigma-clip: removed {n_clipped_this_iter} outlier stacked "
                      f"point(s) (> {sigma_clip_nsigma} robust-sigma).")

        # ── Student-t EM reweighting (smooth alternative/complement to the
        #    hard clip above -- see student_t_dof's docstring entry) ────────
        if student_t_dof is not None:
            sigma_orig = sigma_i.copy()
            for _ in range(max(int(student_t_em_iters), 1)):
                theta = _gp.train_primary_gp(X_st, Y_st, sigma_i)
                psf_mean, psf_std, psf_cov = _gp.predict_primary_gp(
                    theta, X_st, Y_st, sigma_i, psf_x_grid,
                )
                psf_mean = np.clip(psf_mean, 0.0, None)
                psf_at_Xst = np.interp(X_st, psf_x_grid, psf_mean)
                resid = (Y_st - psf_at_Xst) / np.clip(sigma_orig, 1e-30, None)
                w = (student_t_dof + 1.0) / (student_t_dof + resid ** 2)
                w = np.clip(w, 1e-6, 1.0)
                sigma_i = sigma_orig / np.sqrt(w)
            if verbose:
                n_downweighted = int(np.sum(w < 0.5))
                print(f"   Student-t reweight (dof={student_t_dof}): "
                      f"{n_downweighted} point(s) downweighted by >2x.")

        psf_mean, psf_std, psf_cov, psf_area_norm = _gp.area_normalise(psf_x_grid, psf_mean, psf_std, psf_cov=psf_cov)

        amplitudes, centres, model, amp_errors, cen_errors = _lines.fit_all_lines(
            x, y, y_err,
            psf_x_grid, psf_mean,
            windows=windows,
            centres_init=centres,
            amplitudes_init=amplitudes,
            centre_bounds=centre_bounds,
        )

        X_st, Y_st, E_st = _lines.restack_lines(
            x, y, y_err, windows, centres, amplitudes,
        )
        # restack_lines rebuilds X_st/Y_st/E_st from scratch (from the original
        # per-line windows), independently of whatever sigma-clip/Student-t did
        # above -- sigma_i must be recomputed to match this fresh length, or the
        # stacked_yerr returned below can silently mismatch stacked_x/stacked_y
        # in length (when sigma-clip actually removed points) or, worse, match
        # in length but hold stale, no-longer-corresponding values. When
        # use_secondary_gp is on, falling back to plain E_st here (rather than
        # recomputing the empirical/secondary-GP sigma) would silently drop the
        # §4.6 variance correction from the RETURNED stacked_yerr -- understating
        # sigma wherever the true scatter exceeds raw photon noise, and inflating
        # residual_sigma = (Y_st - psf_at_Xst)/stacked_yerr into an artificial,
        # structured-looking pattern with no corresponding problem in the fit
        # itself (psf_mean/theta are unaffected either way -- they're already
        # fixed above, before this restack runs).
        if use_secondary_gp:
            sigma_i, phi_g, X_bins_out, log_S2_out, g_at_X_out = _gp.empirical_sigma(
                X_st, Y_st, E_st, theta, min_per_bin=min_per_bin,
            )
        else:
            sigma_i = E_st.copy()

        res_tmp = (y - model) / np.clip(y_err, 1e-30, None)
        rms_tmp = float(np.sqrt(np.nanmean(res_tmp ** 2)))
        history.append({"iter": it, "rms": rms_tmp, "n_clipped": n_clipped_this_iter})
        if verbose:
            print(f"   RMS = {rms_tmp:.3f} sigma   |   n_lines = {len(centres)}")

    residuals = (y - model) / np.clip(y_err, 1e-30, None)
    rms = float(np.sqrt(np.nanmean(residuals ** 2)))

    rv_errors = np.where(np.isfinite(cen_errors), cen_errors * PIX2MS, np.nan)

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

    recenter_shift = 0.0
    if recenter and psf_centroid != 0.0:
        # Shift the offset grid so the PSF's own (trapz) centroid sits at
        # 0. This is a plain post-hoc translation, not the Eq.20 fixed-point
        # correction above (which does not reliably converge, see
        # n_recentre_max) -- it just guarantees psf_centroid == 0 exactly.
        # line_centres shift oppositely by construction (dx = x - centre),
        # so the reconstructed `model` is mathematically unchanged.
        recenter_shift = psf_centroid
        psf_x_grid = psf_x_grid - recenter_shift
        X_st = X_st - recenter_shift
        centres = centres + recenter_shift
        psf_centroid = 0.0

    return dict(
        theta=theta,
        psf_x=psf_x_grid,
        psf_mean=psf_mean,
        psf_std=psf_std,
        psf_cov=psf_cov,
        psf_centroid=psf_centroid,
        psf_centroid_err=psf_centroid_err,
        psf_area_norm=psf_area_norm,
        recenter_shift=recenter_shift,
        model=model,
        residuals=residuals,
        rms=rms,
        background=background,
        line_windows=windows,
        line_centres=centres,
        line_amps=amplitudes,
        line_centre_errors=cen_errors,
        line_amp_errors=amp_errors,
        line_rv_errors=rv_errors,
        stacked_x=X_st,
        stacked_y=Y_st,
        stacked_yerr=sigma_i if use_secondary_gp else E_st,
        secondary_gp_bins=X_bins_out,
        secondary_gp_log_S2=log_S2_out,
        secondary_gp_g_at_stacked=g_at_X_out,
        history=history,
    )
