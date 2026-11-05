"""
Optional diagnostic plots for the results of
:func:`psfcal.compute_psf_chunk_pixel` / :func:`psfcal.compute_psf_chunk_rv`.

Requires ``matplotlib`` (``pip install psfcal[plot]``).
"""

from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "psfcal.plotting requires matplotlib. Install it with `pip install psfcal[plot]`."
    ) from exc

from scipy.interpolate import CubicSpline

from . import _gp


def plot_psf(
    psf_result: dict,
    title: str = "IP reconstruction (area-normalised)",
    show_mean: bool = False,
    robust_ylim: bool = True,
    figsize=(7, 6),
    truth=None,
    xlabel: str = "Offset",
):
    """
    Top panel: stacked data + GP model. Bottom panel: normalised residuals.

    ``stacked_y`` (psi_hat, Eq. 10) is on the scale the GP was actually
    trained on -- *before* ``psf_mean``'s area normalisation (Eq. 17,
    dividing by ``psf_result["psf_area_norm"]``). Comparing them directly
    either makes the data look artificially huge relative to psf_mean, or
    (worse) makes the bottom-panel residuals look enormous even for a good
    fit, since ``stacked_y`` hasn't been put on the same footing as the
    normalised curve it's being differenced against. Both stacked_y and
    stacked_yerr are divided by ``psf_area_norm`` here so the comparison
    (and the residuals) are correct.

    ``show_mean`` (default False) additionally plots the Gaussian mean
    function m(x) and the GP correction psi-m, normalised by the same
    ``psf_area_norm`` and evaluated at the pre-recentring location (via
    ``psf_result["recenter_shift"]``) so they land on the same scale and
    offset as ``psf_mean`` rather than their own, generally much larger,
    natural scale.

    ``truth`` (optional ``(x, y)`` tuple) overlays a known PSF as a dashed
    black line, for synthetic tests. It must be on the same x axis and
    area-normalised in the same units as ``psf_mean``.
    ``xlabel`` labels the offset axis (e.g. ``"Offset [pixel]"`` or
    ``"Offset [m/s]"``).

    ``robust_ylim`` (default True) sets the y-axis range from the fitted
    curve and the data's own central values (a 0.5-99.5 percentile), not
    from how far any single error bar whisker extends -- a handful of
    pixels with genuinely large propagated flux uncertainty (bad/noisy
    pixels, cosmic ray hits, etc.) otherwise blow out the axis and make
    everything else hard to read; those points are still plotted, just
    potentially with their error whisker extending past the frame.
    """
    sx = psf_result["stacked_x"]
    sy = psf_result["stacked_y"]
    syerr = psf_result["stacked_yerr"]
    px = psf_result["psf_x"]
    py = psf_result["psf_mean"]
    pe = psf_result["psf_std"]
    area_norm = psf_result.get("psf_area_norm", 1.0) or 1.0

    sy_n = sy / area_norm
    syerr_n = syerr / area_norm

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    ax1.errorbar(sx, sy_n, yerr=syerr_n, fmt=".", ms=3, color="0.45",
                 elinewidth=0.4, alpha=0.6, label="y data", zorder=2)
    ax1.set_ylabel(r"Empirical IP $\psi$ (area-normalised)")
    ax1.plot(px, py, color="firebrick", lw=1.8, label=r"Empirical IP $\psi$", zorder=4)

    if show_mean:
        theta = psf_result.get("theta")
        if theta:
            import jax.numpy as jnp
            shift = psf_result.get("recenter_shift", 0.0)
            m_vals = np.array(_gp.gaussian_mean(theta, jnp.array(px) + shift)) / area_norm
            ax1.plot(px, m_vals, color="steelblue", lw=1.2, ls="--", label="Mean fn $m(x)$", zorder=3)
            ax1.plot(px, py - m_vals, color="mediumpurple", lw=1.0, ls=":",
                     label=r"$\psi - m$", zorder=3)

    if np.any(pe > 0):
        ax1.fill_between(px, py - pe, py + pe, alpha=0.20, color="firebrick")
    if truth is not None:
        ax1.plot(truth[0], truth[1], color="k", lw=1.2, ls="--", label="true PSF", zorder=5)
    ax1.axhline(0, color="k", lw=0.6, ls=":")
    ax1.legend(fontsize=8)
    ax1.set_title(title)

    if robust_ylim:
        y_vals = np.concatenate([py, sy_n]) if len(sy_n) else py
        y_vals = y_vals[np.isfinite(y_vals)]
        if len(y_vals):
            ylo, yhi = np.percentile(y_vals, [0.5, 99.5])
            margin = 0.15 * max(yhi - ylo, 1e-30)
            ax1.set_ylim(min(0.0, ylo - margin), yhi + margin)

    spl = CubicSpline(px, py)
    res = (sy_n - spl(sx)) / np.clip(syerr_n, 1e-30, None)
    ax2.scatter(sx, res, s=6, color="0.25", alpha=0.7, zorder=2)
    ax2.axhline(0, color="firebrick", lw=1.5, zorder=3)
    ax2.axhspan(-1, 1, color="steelblue", alpha=0.12, zorder=0)
    ax2.axhspan(-3, 3, color="salmon", alpha=0.12, zorder=0)
    ax2.set_ylabel(r"$(y - \psi)\,/\,\sigma$")
    ax2.set_xlabel(xlabel)
    ax2.set_ylim(-6, 6)

    fig.subplots_adjust(left=0.14, right=0.97, top=0.94, bottom=0.10)
    return fig, (ax1, ax2)


def plot_line_fit(
    x_data,
    y_data,
    y_err,
    psf_result: dict,
    sigma_band: float = 3.0,
    theoretical_x=None,
    trim_to_windows: bool = True,
    figsize=(14, 6),
):
    """
    Plot the raw chunk data against the summed per-line model, with the
    fitted line centres (``psf_result["line_centres"]``) marked as vertical
    lines — the diagnostic to check whether individual LFC/ThAr peaks were
    detected and centred correctly (pixel-space result only, i.e. from
    :func:`psfcal.compute_psf_chunk_pixel`).

    Pass ``theoretical_x`` (e.g. theoretical LFC line positions converted
    to pixel coordinates via ``np.interp(theoretical_wavelengths,
    wave_order, pixel_array)``) to additionally overlay the expected
    positions for comparison against the fitted ones.

    ``y_data`` should be the same raw flux you passed into
    ``compute_psf_chunk_pixel`` — if that call used
    ``subtract_background=True``, the fitted ``model`` was fit against the
    background-subtracted flux, so this function automatically subtracts
    ``psf_result["background"]`` from ``y_data`` before plotting/computing
    residuals (a no-op if background subtraction wasn't used).

    ``trim_to_windows`` (default True) drops pixels before the first
    *complete* line's window and after the last *complete* one: pixels
    outside any line's window were never part of the stacking (``model``
    is 0 there regardless of the nonzero data, producing huge spurious
    residuals at the plot's edges), and a line whose own window touches
    the raw chunk's edge (index 0 or the last index) is dropped too --
    its outer flank was never actually observed (the chunk starts/ends
    mid-line), so it renders as a visibly truncated, incomplete peak.
    """
    if "model" not in psf_result:
        raise KeyError("psf_result has no 'model' key (only compute_psf_chunk_pixel results have one).")

    centres = psf_result["line_centres"]
    model = psf_result["model"]

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data, dtype=float)
    y_err = np.asarray(y_err, dtype=float)
    background = psf_result.get("background")
    if background is not None and np.any(np.asarray(background) != 0):
        y_data = y_data - np.asarray(background)

    windows = psf_result.get("line_windows")
    if trim_to_windows and windows is not None and len(windows) > 0:
        first_k, last_k = 0, len(windows) - 1
        n_total = len(x_data)
        if windows[first_k, 0] <= 0 and first_k < last_k:
            first_k += 1
        if windows[last_k, 1] >= n_total - 1 and last_k > first_k:
            last_k -= 1
        lo_idx, hi_idx = int(windows[first_k, 0]), int(windows[last_k, 1])
        keep = slice(lo_idx, hi_idx + 1)
        x_data, y_data, y_err, model = x_data[keep], y_data[keep], y_err[keep], model[keep]

    residuals = (y_data - model) / np.clip(y_err, 1e-30, None)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    ax1.errorbar(x_data, y_data, yerr=y_err, fmt="o", ms=3, color="0.25",
                 elinewidth=0.5, alpha=0.7, label="data (background-subtracted)", zorder=2)
    ax1.plot(x_data, model, color="firebrick", lw=1.5,
             label=r"model ($\Sigma$ single-line PSF fits)", zorder=3)
    for i, cen in enumerate(centres):
        ax1.axvline(cen, color="sandybrown", lw=0.9, alpha=0.8,
                     label="fitted centre" if i == 0 else None, zorder=4)
    if theoretical_x is not None:
        for i, tx in enumerate(np.asarray(theoretical_x, dtype=float)):
            ax1.axvline(tx, color="seagreen", lw=0.9, ls="--", alpha=0.8,
                        label="theoretical centre" if i == 0 else None, zorder=4)
    ax1.set_ylabel("Flux (e-)")
    ax1.legend(fontsize=9)

    ax2.scatter(x_data, residuals, s=8, color="0.25", alpha=0.8, zorder=2)
    ax2.axhline(0, color="firebrick", lw=1.5, zorder=3)
    ax2.axhspan(-sigma_band, sigma_band, color="salmon", alpha=0.20, zorder=0)
    ax2.axhspan(-1, 1, color="steelblue", alpha=0.12, zorder=0)
    for cen in centres:
        ax2.axvline(cen, color="sandybrown", lw=0.7, alpha=0.6)
    ax2.set_ylabel(r"$(y - \mathrm{model})\,/\,\sigma$")
    ax2.set_xlabel("Pixel")

    return fig, (ax1, ax2)


def plot_secondary_gp(psf_result: dict, figsize=(7, 4)):
    """Secondary GP for empirical variance (Fig. 5 of Milakovic & Jethwa 2024)."""
    if psf_result.get("secondary_gp_bins") is None or len(psf_result["secondary_gp_bins"]) == 0:
        print("No secondary GP data in result.")
        return None, None

    X_bins = psf_result["secondary_gp_bins"]
    log_S2 = psf_result["secondary_gp_log_S2"]
    g_x = psf_result["secondary_gp_g_at_stacked"]
    sx = psf_result["stacked_x"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(X_bins, log_S2, s=30, color="k", zorder=4, label=r"$\log S^2$")
    order = np.argsort(sx)
    ax.plot(sx[order], np.log(g_x[order]), color="k", lw=1.5, label=r"$g(\Delta x)$ posterior mean")
    ax.set_xlabel("Distance from centre")
    ax.set_ylabel(r"$\log(S^2)$")
    ax.set_title("Secondary GP for empirical variance")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig, ax


def plot_ccf(rv_result: dict, figsize=(7, 4)):
    """Plot the cross-correlation function and its Gaussian peak fit (item 8)."""
    v = rv_result["ccf_v"]
    ccf = rv_result["ccf"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(v, ccf, ".-", color="0.3", lw=1, ms=3, label="CCF")
    ax.axvline(rv_result["rv"], color="firebrick", lw=1.5,
               label=f"RV = {rv_result['rv']:.1f} m/s")
    if rv_result.get("popt") is not None:
        from .rv_shift import _gauss_peak
        vv = np.linspace(v.min(), v.max(), 500)
        ax.plot(vv, _gauss_peak(vv, *rv_result["popt"]), color="steelblue", lw=1.2, ls="--", label="Gaussian fit")
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Normalised CCF")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig, ax
