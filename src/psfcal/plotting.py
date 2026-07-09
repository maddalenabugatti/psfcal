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


def plot_psf(psf_result: dict, title: str = "IP reconstruction (area-normalised)", figsize=(7, 6)):
    """
    Top panel: stacked data + GP model (+ mean function, if ``theta`` is
    present). Bottom panel: normalised residuals.
    """
    sx = psf_result["stacked_x"]
    sy = psf_result["stacked_y"]
    syerr = psf_result["stacked_yerr"]
    px = psf_result["psf_x"]
    py = psf_result["psf_mean"]
    pe = psf_result["psf_std"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    ax1.errorbar(sx, sy, yerr=syerr, fmt=".", ms=3, color="0.45",
                 elinewidth=0.4, alpha=0.6, label=r"$\hat\psi$ data", zorder=2)
    ax1.plot(px, py, color="firebrick", lw=1.8, label=r"Empirical IP $\psi$", zorder=4)

    theta = psf_result.get("theta")
    if theta:
        import jax.numpy as jnp
        m_vals = np.array(_gp.gaussian_mean(theta, jnp.array(px)))
        ax1.plot(px, m_vals, color="steelblue", lw=1.2, ls="--", label="Mean fn $m(x)$", zorder=3)
        ax1.plot(px, py - m_vals, color="mediumpurple", lw=1.0, ls=":",
                 label=r"$\psi - m$", zorder=3)

    if np.any(pe > 0):
        ax1.fill_between(px, py - pe, py + pe, alpha=0.20, color="firebrick")
    ax1.axhline(0, color="k", lw=0.6, ls=":")
    ax1.set_ylabel("Intensity (normalised)")
    ax1.set_title(title)
    ax1.legend(fontsize=8)

    spl = CubicSpline(px, py)
    res = (sy - spl(sx)) / np.clip(syerr, 1e-30, None)
    ax2.scatter(sx, res, s=6, color="0.25", alpha=0.7, zorder=2)
    ax2.axhline(0, color="firebrick", lw=1.5, zorder=3)
    ax2.axhspan(-1, 1, color="steelblue", alpha=0.12, zorder=0)
    ax2.axhspan(-3, 3, color="salmon", alpha=0.12, zorder=0)
    ax2.set_ylabel(r"$(y - \psi)\,/\,\sigma$")
    ax2.set_xlabel("Offset")
    ax2.set_ylim(-6, 6)

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_line_fit(x_data, y_data, y_err, psf_result: dict, sigma_band: float = 3.0, figsize=(14, 6)):
    """Plot data vs summed per-line model and normalised residuals (pixel-space result only)."""
    if "model" not in psf_result:
        raise KeyError("psf_result has no 'model' key (only compute_psf_chunk_pixel results have one).")

    centres = psf_result["line_centres"]
    model = psf_result["model"]
    residuals = (np.asarray(y_data) - model) / np.clip(y_err, 1e-30, None)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    ax1.errorbar(x_data, y_data, yerr=y_err, fmt="o", ms=3, color="0.25",
                 elinewidth=0.5, alpha=0.7, label="data", zorder=2)
    ax1.plot(x_data, model, color="firebrick", lw=1.5,
             label=r"model ($\Sigma$ single-line PSF fits)", zorder=3)
    for cen in centres:
        ax1.axvline(cen, color="sandybrown", lw=0.7, alpha=0.6)
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
