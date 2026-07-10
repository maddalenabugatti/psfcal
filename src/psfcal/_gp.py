"""
Shared Gaussian-Process engine for IP/PSF reconstruction, following

    Milakovic & Jethwa, A&A 684, A38 (2024)   [MJ24]

Scale-agnostic: ``X`` is just an offset array in whatever unit the caller
uses (pixels for :mod:`psfcal.psf_pixel`, m/s for :mod:`psfcal.psf_rv`).

Ported from ``psf_tools_4.py``, section references point to MJ24.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

import jax
import jax.numpy as jnp
import jaxopt
from tinygp import kernels, GaussianProcess

jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────────────────────
#  §3.2  Gaussian mean function  (Eq. 17)
# ─────────────────────────────────────────────────────────────

def gaussian_mean(theta: dict, X: jnp.ndarray) -> jnp.ndarray:
    """Area-normalised Gaussian mean function m(x; theta) = A . N(x; mu, tau) + y0."""
    A = theta["mf_amp"]
    mu = theta["mf_loc"]
    tau = jnp.exp(theta["mf_log_sig"])
    y0 = theta["mf_y0"]
    gauss = (jnp.exp(-0.5 * jnp.square((X - mu) / tau))
             / (jnp.sqrt(2.0 * jnp.pi) * tau))
    return A * gauss + y0


def _gauss_np(x, A, mu, tau, y0=0.0):
    """Numpy Gaussian (Eq. 17) for initialisation / CCF peak fitting."""
    return A * np.exp(-0.5 * ((x - mu) / tau) ** 2) / (np.sqrt(2 * np.pi) * tau) + y0


# ─────────────────────────────────────────────────────────────
#  §3.2 / §4.5  Primary GP builder  (Eqs. 18-19)
# ─────────────────────────────────────────────────────────────

def build_primary_gp(theta: dict, X: jnp.ndarray, sigma_i: jnp.ndarray) -> GaussianProcess:
    """
    GP for the IP.  Kernel: a^2 exp(-|xi-xj|^2 / 2l^2) + sigma_i^2 delta_ij
    (Eq. 18-19).  Mean function: Gaussian m(x; theta) (Eq. 17).
    """
    a = jnp.exp(theta["gp_log_amp"])
    l = jnp.exp(theta["gp_log_scale"])
    s0 = jnp.exp(theta["log_sigma0"])

    kernel = a ** 2 * kernels.ExpSquared(l)
    var_tot = jnp.square(sigma_i) + jnp.square(s0)  # Eq. 19

    for kw in ("diag", "noise"):
        try:
            return GaussianProcess(kernel, X, **{kw: var_tot}, mean=partial(gaussian_mean, theta))
        except TypeError:
            pass
    return GaussianProcess(kernel, X, noise=jnp.diag(var_tot), mean=partial(gaussian_mean, theta))


# ─────────────────────────────────────────────────────────────
#  §4.6  Secondary GP builder  (Eq. 23)
# ─────────────────────────────────────────────────────────────

def build_secondary_gp(phi_g: dict, X_bins: jnp.ndarray, log_var_S2: jnp.ndarray) -> GaussianProcess:
    """GP for log(S^2), the secondary GP for empirical variance estimation."""
    Cg = jnp.exp(phi_g["log_Cg"])
    ag = jnp.exp(phi_g["log_ag"])
    lg = jnp.exp(phi_g["log_lg"])

    kernel = ag ** 2 * kernels.ExpSquared(lg)
    var_diag = jnp.exp(log_var_S2)

    for kw in ("diag", "noise"):
        try:
            return GaussianProcess(kernel, X_bins, **{kw: var_diag},
                                    mean=lambda x: Cg * jnp.ones_like(x))
        except TypeError:
            pass
    return GaussianProcess(kernel, X_bins, noise=jnp.diag(var_diag),
                            mean=lambda x: Cg * jnp.ones_like(x))


# ─────────────────────────────────────────────────────────────
#  Loss functions  (JIT-compiled)
# ─────────────────────────────────────────────────────────────

@jax.jit
def _primary_loss(theta: dict, X: jnp.ndarray, Y: jnp.ndarray, sigma_i: jnp.ndarray) -> jnp.ndarray:
    """Negative log-likelihood for the primary GP (Eq. 9)."""
    gp = build_primary_gp(theta, X, sigma_i)
    Y0 = Y - gaussian_mean(theta, X)
    return -gp.log_probability(Y0)


@jax.jit
def _secondary_loss(phi_g: dict, X_bins: jnp.ndarray, log_S2: jnp.ndarray,
                     log_var_S2: jnp.ndarray) -> jnp.ndarray:
    """Negative log-likelihood for the secondary GP (Sect. 4.6)."""
    gp = build_secondary_gp(phi_g, X_bins, log_var_S2)
    Cg = jnp.exp(phi_g["log_Cg"])
    Y0 = log_S2 - Cg
    return -gp.log_probability(Y0)


# ─────────────────────────────────────────────────────────────
#  §4.5  Train primary GP hyper-parameters
# ─────────────────────────────────────────────────────────────

def train_primary_gp(
    X: np.ndarray,
    Y: np.ndarray,
    sigma_i: np.ndarray,
    *,
    gp_log_scale_bounds: Tuple[float, float] = (-1.0, 1.0),
    mf_loc_init: float | None = None,
) -> dict:
    """
    Optimise primary GP hyper-parameters by maximising Eq. (9).

    Initialisation follows §4.5: (A, mu, tau, y0) from a nonlinear
    least-squares Gaussian fit to psi_hat, constrained within +-5 standard
    deviations (kappa). log a = 1, log sigma0 = -5 defaults.
    Bounds: -4 <= log a <= 4, -15 <= log sigma0 <= 1.5.

    ``gp_log_scale_bounds`` bounds the GP kernel length-scale in log space
    and MUST be scaled to the units of ``X``: the paper's default
    (-1, 1) -- i.e. a length scale of ~0.4-2.7 -- assumes pixel-space
    offsets. RV-space offsets (m/s) need correspondingly larger bounds,
    e.g. ``(log(50), log(5000))``, or the GP kernel will collapse to an
    inappropriate length scale and fail to reconstruct the PSF (see
    :mod:`psfcal.psf_rv`).

    ``mf_loc_init`` seeds the Gaussian mean-function's centre for the
    initial curve_fit. Defaults to 0.0, which is correct for pixel-space
    stacking (each line is centred on its own fitted Gaussian, so the
    stack is centred near 0 by construction) but WRONG for RV-space
    stacking, where lines are anchored to a theoretical wavelength that
    can carry a real, non-zero systematic offset from 0 -- with a bad
    initial guess far from the data's actual peak, curve_fit can fail to
    converge (raising ``OptimizeWarning``) and the GP collapses to a
    near-flat fit. Pass e.g. ``X[np.argmax(Y)]`` for RV-space calls.
    """
    X_j = jnp.array(X)
    Y_j = jnp.array(Y)
    sigma_j = jnp.array(sigma_i)

    loc0 = float(mf_loc_init) if mf_loc_init is not None else 0.0
    X_lo, X_hi = float(np.min(X)), float(np.max(X))
    X_range = max(X_hi - X_lo, 1e-6)
    # Cap the mean-function width to the same physically-sensible line-width
    # range as the GP kernel's length scale (gp_log_scale_bounds), not the
    # full data span: a width up to X_range let the bootstrap fit (and
    # later the mean-function's own theta bounds, below) converge on an
    # absurdly wide "line" spanning the whole chunk whenever the GP kernel
    # amplitude got starved to ~0, since nothing else constrained it.
    tau_hi = min(X_range, float(np.exp(gp_log_scale_bounds[1])))
    p0 = (
        max(float(np.max(Y)), 1e-12),
        float(np.clip(loc0, X_lo, X_hi)),
        float(np.clip(max(float(np.std(X)) / 3.0, 0.1), 1e-6, tau_hi)),
        0.0,
    )
    # Bounded (amplitude >= 0 in particular -- see mf_amp note below) so
    # curve_fit uses 'trf', not the unconstrained default: for RV-space
    # data (Y values ~1e-4, X spanning thousands of units) unconstrained
    # least-squares has been observed to converge to wildly wrong-signed,
    # wrong-scale parameters.
    bounds = ((0.0, X_lo, 1e-6, -np.inf), (np.inf, X_hi, tau_hi, np.inf))
    try:
        popt, pcov = curve_fit(
            _gauss_np, X, Y, sigma=sigma_i, p0=p0, bounds=bounds,
            absolute_sigma=True, maxfev=20_000,
        )
        perr = np.sqrt(np.diag(np.abs(pcov)))
        perr = np.clip(perr, 1e-6, np.abs(popt) + 1.0)
    except Exception:
        popt = np.array(p0)
        perr = np.abs(popt) * 0.3 + 1e-4

    kappa = 5.0  # +-5 sigma constraint (§4.5)

    gp_lo, gp_hi = gp_log_scale_bounds
    gp_log_scale_init = float(np.clip(np.log(max(abs(popt[2]), 1e-8)), gp_lo, gp_hi))

    # gp_log_amp (the GP kernel's own amplitude) must be scaled to Y's
    # actual variance, exactly like gp_log_scale is scaled to X's units --
    # a hardcoded absolute range (e.g. the paper's pixel-space [-4, 4])
    # can be wildly mismatched to Y for RV-space stacks, whose scale
    # varies a lot with domain width/SNR. Observed in practice: fixed at
    # [-4, 4], the kernel amplitude got pinned at the lower bound for one
    # RV chunk (real data was smaller still), starving the GP's flexible
    # correction term entirely and forcing the parametric mean function
    # to explain the whole stack alone -- it responded by blowing up into
    # an absurdly wide, offset Gaussian instead of tracking the line.
    y_scale = max(float(np.std(Y)), 1e-12)
    gp_log_amp_init = float(np.log(y_scale))
    gp_log_amp_lo = gp_log_amp_init - 4.0
    gp_log_amp_hi = gp_log_amp_init + 4.0

    # mf_log_sig (the mean function's own width) is likewise capped to the
    # same physically-sensible line-width range as the kernel length scale
    # -- the kappa*perr window alone doesn't prevent it drifting far
    # outside that range (see tau_hi note above); take the intersection,
    # falling back to the full kernel range if kappa*perr excludes it
    # entirely (a poorly-constrained bootstrap fit).
    mf_log_sig_est = float(np.log(max(abs(popt[2]), 1e-4)))
    mf_log_sig_lo = max(float(np.log(max(abs(popt[2]) - kappa * perr[2], 1e-4))), gp_lo)
    mf_log_sig_hi = min(float(np.log(abs(popt[2]) + kappa * perr[2] + 1e-4)), gp_hi)
    if mf_log_sig_hi <= mf_log_sig_lo:
        mf_log_sig_lo, mf_log_sig_hi = gp_lo, gp_hi
    mf_log_sig_init = float(np.clip(mf_log_sig_est, mf_log_sig_lo, mf_log_sig_hi))

    theta0 = dict(
        mf_amp=max(0.0, float(popt[0])),
        mf_loc=float(popt[1]),
        mf_log_sig=mf_log_sig_init,
        mf_y0=float(popt[3]),
        gp_log_amp=gp_log_amp_init,
        gp_log_scale=gp_log_scale_init,
        log_sigma0=-5.0,
    )
    lower = dict(
        # mf_amp represents a real emission line and must stay
        # non-negative: the GP kernel's own flexible correction term can
        # explain the stacked data just as well without the parametric
        # mean function contributing anything, so nothing anchors its
        # sign otherwise -- observed in practice to let the joint
        # optimizer drift to a large *negative* amplitude that the
        # correction term then cancels out, leaving the returned PSF
        # (mean + correction) collapsed to ~0 everywhere.
        mf_amp=max(0.0, float(popt[0] - kappa * perr[0])),
        mf_loc=float(popt[1] - kappa * perr[1]),
        mf_log_sig=mf_log_sig_lo,
        mf_y0=float(popt[3] - kappa * perr[3]),
        gp_log_amp=gp_log_amp_lo,
        gp_log_scale=gp_lo,
        log_sigma0=-15.0,
    )
    upper = dict(
        mf_amp=float(popt[0] + kappa * perr[0]),
        mf_loc=float(popt[1] + kappa * perr[1]),
        mf_log_sig=mf_log_sig_hi,
        mf_y0=float(popt[3] + kappa * perr[3]),
        gp_log_amp=gp_log_amp_hi,
        gp_log_scale=gp_hi,
        log_sigma0=1.5,
    )

    theta0_j = jax.tree_util.tree_map(jnp.asarray, theta0)
    solver = jaxopt.ScipyBoundedMinimize(
        fun=partial(_primary_loss, X=X_j, Y=Y_j, sigma_i=sigma_j),
        method="l-bfgs-b",
    )
    sol = solver.run(theta0_j, bounds=(lower, upper))
    return sol.params


# ─────────────────────────────────────────────────────────────
#  §4.6  Train secondary GP for empirical variance
# ─────────────────────────────────────────────────────────────

def train_secondary_gp(
    X_bins: np.ndarray,
    log_S2: np.ndarray,
    log_var_S2: np.ndarray,
    *,
    log_lg_bounds: Tuple[float, float] = (-1.0, 3.0),
) -> dict:
    """
    Train secondary GP on (x_bin, log S^2) pairs.

    Bounds follow §4.6: -10 <= log Cg <= 5, -3 <= log ag <= 3.
    ``log_lg_bounds`` (default -1..3, i.e. length scale ~0.4-20) must be
    scaled to the units of ``X_bins`` the same way as
    :func:`train_primary_gp`'s ``gp_log_scale_bounds``.
    """
    X_j = jnp.array(X_bins)
    Y_j = jnp.array(log_S2)
    LV_j = jnp.array(log_var_S2)

    lg_lo, lg_hi = log_lg_bounds
    log_lg_init = float(np.clip(0.0, lg_lo, lg_hi))

    phi0 = dict(log_Cg=float(np.median(log_S2)), log_ag=0.0, log_lg=log_lg_init)
    lower = dict(log_Cg=-10.0, log_ag=-3.0, log_lg=lg_lo)
    upper = dict(log_Cg=5.0, log_ag=3.0, log_lg=lg_hi)

    phi0_j = jax.tree_util.tree_map(jnp.asarray, phi0)
    solver = jaxopt.ScipyBoundedMinimize(
        fun=partial(_secondary_loss, X_bins=X_j, log_S2=Y_j, log_var_S2=LV_j),
        method="l-bfgs-b",
    )
    sol = solver.run(phi0_j, bounds=(lower, upper))
    return sol.params


# ─────────────────────────────────────────────────────────────
#  §4.6  Build g(dx) and return empirically modified sigma_i
# ─────────────────────────────────────────────────────────────

def empirical_sigma(
    X_st: np.ndarray,
    Y_st: np.ndarray,
    E_st: np.ndarray,
    theta: dict,
    min_per_bin: int = 15,
    *,
    log_lg_bounds: Tuple[float, float] = (-1.0, 3.0),
) -> Tuple[np.ndarray, dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement §4.6 in full.

    Returns
    -------
    sigma_mod : modified noise array sigma_i = sqrt(g(dx_i) sigma_psi_hat_i^2 + sigma0^2)
    phi_g     : trained secondary-GP hyper-parameters
    X_bins    : bin centres used for secondary GP
    log_S2    : log S^2 values
    g_at_X    : g(dx_i) evaluated at every stacked point
    """
    import warnings

    X_j = jnp.array(X_st)
    Y_j = jnp.array(Y_st)
    E_j = jnp.array(E_st)

    gp = build_primary_gp(theta, X_j, E_j)
    m = np.array(gaussian_mean(theta, X_j))
    Y0 = Y_j - jnp.array(m)
    _, cnd = gp.condition(Y0, X_j)
    psi_at_data = m + np.array(cnd.mean)

    residuals = (Y_st - psi_at_data) / np.clip(E_st, 1e-30, None)

    order = np.argsort(X_st)
    X_sorted = X_st[order]
    r_sorted = residuals[order]
    N_total = len(X_sorted)

    bin_centres, log_S2_list, log_var_S2_list = [], [], []
    i = 0
    while i + min_per_bin <= N_total:
        chunk_x = X_sorted[i: i + min_per_bin]
        chunk_r = r_sorted[i: i + min_per_bin]
        n = len(chunk_r)
        S2 = float(np.var(chunk_r, ddof=1))
        if S2 <= 0:
            i += min_per_bin
            continue
        mu4 = float(np.mean((chunk_r - chunk_r.mean()) ** 4))
        var_S2 = max((mu4 - (n - 3) / (n - 1) * S2 ** 2) / n, 1e-30)  # Eq. 25
        bin_centres.append(float(np.mean(chunk_x)))
        log_S2_list.append(np.log(S2))
        log_var_S2_list.append(np.log(var_S2))
        i += min_per_bin

    if len(bin_centres) < 3:
        warnings.warn(
            "Too few bins for secondary GP variance estimation "
            f"(got {len(bin_centres)}). Returning unmodified errors.",
            stacklevel=3,
        )
        sigma_mod = E_st.copy()
        return sigma_mod, {}, np.array([]), np.array([]), np.ones_like(X_st)

    X_bins = np.array(bin_centres)
    log_S2 = np.array(log_S2_list)
    log_var_S2 = np.array(log_var_S2_list)

    phi_g = train_secondary_gp(X_bins, log_S2, log_var_S2, log_lg_bounds=log_lg_bounds)

    X_bins_j = jnp.array(X_bins)
    log_S2_j = jnp.array(log_S2)
    log_varS2_j = jnp.array(log_var_S2)
    X_all_j = jnp.array(X_st)

    gp_sec = build_secondary_gp(phi_g, X_bins_j, log_varS2_j)
    Cg = float(jnp.exp(phi_g["log_Cg"]))
    Y0_sec = log_S2_j - Cg
    _, cnd_sec = gp_sec.condition(Y0_sec, X_all_j)
    log_g = Cg + np.array(cnd_sec.mean)
    g_at_X = np.exp(np.clip(log_g, -10.0, 10.0))

    sigma0 = float(jnp.exp(theta["log_sigma0"]))
    sigma_mod = np.sqrt(np.clip(g_at_X * E_st ** 2 + sigma0 ** 2, 1e-30, None))

    return sigma_mod, phi_g, X_bins, log_S2, g_at_X


# ─────────────────────────────────────────────────────────────
#  GP prediction
# ─────────────────────────────────────────────────────────────

def predict_primary_gp(
    theta: dict,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    sigma_i: np.ndarray,
    X_new: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior mean, std and covariance of the primary GP at X_new."""
    X_j = jnp.array(X_train)
    Y_j = jnp.array(Y_train)
    E_j = jnp.array(sigma_i)
    Xn_j = jnp.array(X_new)

    gp = build_primary_gp(theta, X_j, E_j)
    m_tr = gaussian_mean(theta, X_j)
    Y0 = Y_j - m_tr
    _, cnd = gp.condition(Y0, Xn_j)

    m_new = np.array(gaussian_mean(theta, Xn_j))
    mean = m_new + np.array(cnd.mean)
    cov = np.array(cnd.covariance)
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    return mean, std, cov


# ─────────────────────────────────────────────────────────────
#  §3.3  Re-centring  (Eq. 20)
# ─────────────────────────────────────────────────────────────

def recentre_shift(psf_x: np.ndarray, psf_mean: np.ndarray, half_width: float = 0.5) -> float:
    """
    Compute the shift needed so that psi(-h) = psi(+h) (Eq. 20), where h is
    half a pixel in the units of ``psf_x``.

    For pixel-space PSFs h = 0.5 (the default). For RV-space IPs, pass
    ``half_width = 0.5 * local_mps_per_pixel`` so the anchor points remain
    half a detector pixel wide in velocity units.

    Returns the shift delta in the units of psf_x.
    """
    spl = CubicSpline(psf_x, psf_mean)
    psi_m = float(spl(-half_width))
    psi_p = float(spl(+half_width))
    dpsi_m = float(spl(-half_width, 1))
    dpsi_p = float(spl(+half_width, 1))
    denom = dpsi_m + dpsi_p
    if abs(denom) < 1e-20:
        return 0.0
    return (psi_p - psi_m) / denom


# ─────────────────────────────────────────────────────────────
#  Area normalisation with Jacobian error propagation
# ─────────────────────────────────────────────────────────────

def area_normalise(
    psf_x: np.ndarray,
    psf_mean: np.ndarray,
    psf_std: np.ndarray,
    psf_cov: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
    """
    Normalise the PSF to unit area and propagate the uncertainty.

    For g = f / A with A = w^T f (trapezoid weights), the Jacobian is
    J = I/A - (f w^T)/A^2, and cov_norm = J . cov . J^T.

    Returns ``(mean_norm, std_norm, cov_norm, area)`` -- ``area`` is the
    divisor applied (1.0 if normalisation was skipped because area <= 0),
    needed by callers that want to compare pre-normalisation quantities
    (e.g. stacked_y, or the raw mean function) on the same footing as the
    normalised PSF: ``quantity / area`` puts it on the post-normalisation
    scale.
    """
    area = np.trapz(psf_mean, psf_x)
    if area <= 0:
        return psf_mean, psf_std, psf_cov, 1.0

    mean_norm = psf_mean / area

    dx = np.diff(psf_x)
    w = np.zeros_like(psf_x)
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    w[1:-1] = 0.5 * (dx[:-1] + dx[1:])

    if psf_cov is not None:
        n = len(psf_mean)
        J = np.eye(n) / area - np.outer(psf_mean, w) / area ** 2
        cov_norm = J @ psf_cov @ J.T
        std_norm = np.sqrt(np.clip(np.diag(cov_norm), 0.0, None))
    else:
        sigma_A_sq = np.sum((w * psf_std) ** 2)
        std_norm = np.sqrt(
            (psf_std / area) ** 2 + (mean_norm / area) ** 2 * sigma_A_sq
        )
        cov_norm = np.diag(std_norm ** 2)

    return mean_norm, std_norm, cov_norm, float(area)
