"""
Multi-order, multi-chunk PSF/IP cube orchestration.

Thin wrappers around :func:`psfcal.compute_psf_chunk_pixel` /
:func:`psfcal.compute_psf_chunk_rv`, looping over chunks and orders the
same way the original per-order/per-chunk driver scripts did, and
assembling the results into a cube. No new PSF-fitting logic lives here —
each chunk is computed by calling the existing single-chunk functions.

The returned ``psf_cube``/``dx_cube``/``chunk_edges``/``ok`` feed directly
into :func:`psfcal.deconvolve_order` (index ``[order_row]`` to get the
per-order 2-D arrays it expects).
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple

import numpy as np

from .psf_pixel import compute_psf_chunk_pixel
from .psf_rv import compute_psf_chunk_rv

C_LIGHT_MPS = 299_792_458.0


def make_chunk_edges(
    n_pix: int, n_chunks: int, margin: int = 0
) -> Tuple[list, np.ndarray]:
    """
    Split one order (length ``n_pix``) into ``n_chunks`` equal pieces.

    Returns ``(chunks, raw_edges)``:
        chunks    -- list of ``(start, end)`` index pairs *including*
            ``margin`` pixels of extra context on each side (end
            exclusive) -- use these to slice flux/err/wave for fitting.
        raw_edges -- ``(n_chunks, 2)`` array of ``(start, end)`` *without*
            margin -- the true, non-overlapping chunk boundaries, suitable
            directly as :func:`psfcal.deconvolve_order`'s ``chunk_edges``.
    """
    edges = np.linspace(0, int(n_pix), int(n_chunks) + 1).astype(int)
    raw_edges = np.stack([edges[:-1], edges[1:]], axis=1)
    chunks = [
        (max(0, int(edges[i]) - margin), min(int(n_pix), int(edges[i + 1]) + margin))
        for i in range(int(n_chunks))
    ]
    return chunks, raw_edges


def generate_lfc_wavelengths(
    wave_min: float,
    wave_max: float,
    nu0: float,
    delta_nu: float,
) -> np.ndarray:
    """
    Theoretical laser-frequency-comb wavelengths within ``[wave_min,
    wave_max]`` (same unit as ``nu0``/``delta_nu`` are wavelength-paired
    against, e.g. metres), from the comb equation
    ``nu_n = nu0 + n * delta_nu`` (n integer).
    """
    nu_max = C_LIGHT_MPS / wave_min
    nu_min = C_LIGHT_MPS / wave_max
    n_min = int(np.ceil((nu_min - nu0) / delta_nu))
    n_max = int(np.floor((nu_max - nu0) / delta_nu))
    ns = np.arange(n_min, n_max + 1)
    wav = np.sort(C_LIGHT_MPS / (nu0 + ns * delta_nu))
    return wav[(wav >= wave_min) & (wav <= wave_max)]


def compute_psf_cube_pixel(
    flux,
    err,
    orders_1based: Sequence[int],
    *,
    n_chunks: int = 32,
    chunk_margin_pix: int = 10,
    n_psf_points: int = 800,
    subtract_background: bool = True,
    verbose: bool = False,
    **compute_psf_kwargs,
) -> dict:
    """
    Compute the pixel-space PSF (:func:`compute_psf_chunk_pixel`) for
    every chunk of every requested order, assembled into a cube.

    Parameters
    ----------
    flux, err : ``(n_orders, n_pix)`` full-detector arrays (e.g. an S2D
        file's SCIDATA/ERRDATA extensions).
    orders_1based : which orders (1-based row index into ``flux``/``err``)
        to process.
    n_chunks, chunk_margin_pix : how many chunks per order, and how much
        extra context (pixels) to include on each side of a chunk for
        line detection/stacking -- the boundary reported in
        ``chunk_edges`` is still the raw, margin-free one.
    n_psf_points : forwarded to ``compute_psf_chunk_pixel``; fixed across
        every chunk so the results stack into one cube.
    subtract_background : forwarded to ``compute_psf_chunk_pixel``
        (default True here, unlike that function's own default of False,
        since raw S2D flux chunks are not background-subtracted).
    **compute_psf_kwargs : any other ``compute_psf_chunk_pixel`` keyword
        (``n_iterations``, ``recenter``, ``use_secondary_gp``,
        ``min_peak_height_fraction``, ``max_half_window``, ...).

    A chunk where fitting fails (e.g. too few detected lines) is left as
    NaN in the cube and 0 in ``ok``; a warning is issued naming the order
    and chunk.

    Returns
    -------
    dict with keys:
        psf_cube      -- (n_sel, n_chunks, n_psf_points) PSF mean
        dx_cube       -- (n_sel, n_chunks, n_psf_points) pixel-offset grid
        var_cube      -- (n_sel, n_chunks, n_psf_points) PSF variance (std^2)
        ok            -- (n_sel, n_chunks) int, 1 where that chunk's fit succeeded
        chunk_edges   -- (n_chunks, 2) raw (margin-free) pixel boundaries,
                          shared across orders
        orders_1based -- as given, as an array

    ``psf_cube[i]``/``dx_cube[i]``/``chunk_edges``/``ok[i]`` feed directly
    into ``deconvolve_order(order_flux, psf_cube[i], dx_cube[i],
    chunk_edges, scale="pixel", ok_chunks=ok[i])``.
    """
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float)
    n_pix = flux.shape[1]
    n_sel = len(orders_1based)

    chunks, raw_edges = make_chunk_edges(n_pix, n_chunks, chunk_margin_pix)

    psf_cube = np.full((n_sel, n_chunks, n_psf_points), np.nan)
    dx_cube = np.full_like(psf_cube, np.nan)
    var_cube = np.full_like(psf_cube, np.nan)
    ok = np.zeros((n_sel, n_chunks), dtype=np.int16)

    for oi, order_1b in enumerate(orders_1based):
        row = int(order_1b) - 1
        spectrum = flux[row]
        spec_err = err[row]
        if verbose:
            print(f"Order {order_1b} ({oi + 1}/{n_sel})")

        for ci, (a, b) in enumerate(chunks):
            x = np.arange(a, b, dtype=float)
            y = spectrum[a:b]
            yerr = spec_err[a:b]

            try:
                result = compute_psf_chunk_pixel(
                    x, y, yerr,
                    n_psf_points=n_psf_points,
                    subtract_background=subtract_background,
                    verbose=False,
                    **compute_psf_kwargs,
                )
            except Exception as exc:
                warnings.warn(f"order {order_1b} chunk {ci}: PSF fit failed ({exc}); skipped.")
                continue

            psf_cube[oi, ci, :] = result["psf_mean"]
            dx_cube[oi, ci, :] = result["psf_x"]
            var_cube[oi, ci, :] = np.square(result["psf_std"])
            ok[oi, ci] = 1

    return dict(
        psf_cube=psf_cube,
        dx_cube=dx_cube,
        var_cube=var_cube,
        ok=ok,
        chunk_edges=raw_edges,
        orders_1based=np.asarray(orders_1based),
    )


def compute_psf_cube_rv(
    flux,
    err,
    wave,
    orders_1based: Sequence[int],
    *,
    theoretical_wavelengths: Optional[np.ndarray] = None,
    lfc_nu0: Optional[float] = None,
    lfc_delta_nu: Optional[float] = None,
    n_chunks: int = 32,
    chunk_margin_pix: int = 10,
    n_psf_points: int = 500,
    verbose: bool = False,
    **compute_psf_kwargs,
) -> dict:
    """
    Compute the RV-space IP (:func:`compute_psf_chunk_rv`) for every chunk
    of every requested order, assembled into a cube.

    Parameters
    ----------
    flux, err, wave : ``(n_orders, n_pix)`` full-detector arrays. ``wave``
        must already be in whatever unit you want ``theoretical_wavelengths``
        expressed in (e.g. metres) -- convert before calling if needed.
    orders_1based : which orders (1-based row index) to process.
    theoretical_wavelengths : full line list (same unit as ``wave``), used
        as-is for every order -- ``compute_psf_chunk_rv`` internally
        restricts it to whatever falls within each chunk, so passing one
        list spanning every order is fine. Provide this, OR both
        ``lfc_nu0``/``lfc_delta_nu`` to generate an LFC comb automatically
        (via :func:`generate_lfc_wavelengths`) spanning the full range of
        ``wave``.
    lfc_nu0, lfc_delta_nu : LFC comb central frequency and mode spacing
        [Hz], used only if ``theoretical_wavelengths`` is not given.
    n_chunks, chunk_margin_pix : how many chunks per order, and how much
        extra context (pixels) to include on each side of a chunk.
    n_psf_points : forwarded to ``compute_psf_chunk_rv``; fixed across
        every chunk so the results stack into one cube.
    **compute_psf_kwargs : any other ``compute_psf_chunk_rv`` keyword
        (``n_iterations``, ``recenter``, ``use_secondary_gp``,
        ``search_window_pix``, ``max_half_window``, ...).

    A chunk where fitting fails (e.g. fewer than 2 matched theoretical
    lines) is left as NaN in the cube and 0 in ``ok``; a warning is issued
    naming the order and chunk.

    Returns
    -------
    dict with keys:
        psf_cube      -- (n_sel, n_chunks, n_psf_points) IP mean
        dx_cube       -- (n_sel, n_chunks, n_psf_points) velocity-offset grid (m/s)
        var_cube      -- (n_sel, n_chunks, n_psf_points) IP variance (std^2)
        ok            -- (n_sel, n_chunks) int, 1 where that chunk's fit succeeded
        n_lines_used  -- (n_sel, n_chunks) int, matched theoretical lines per chunk
        chunk_edges   -- (n_chunks, 2) raw (margin-free) pixel boundaries,
                          shared across orders
        orders_1based -- as given, as an array

    ``psf_cube[i]``/``dx_cube[i]``/``chunk_edges``/``ok[i]`` feed directly
    into ``deconvolve_order(order_flux, psf_cube[i], dx_cube[i],
    chunk_edges, scale="rv", wave_order=wave[orders_1based[i]-1],
    ok_chunks=ok[i])``.
    """
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float)
    wave = np.asarray(wave, dtype=float)
    n_pix = flux.shape[1]
    n_sel = len(orders_1based)

    if theoretical_wavelengths is None:
        if lfc_nu0 is None or lfc_delta_nu is None:
            raise ValueError(
                "Provide theoretical_wavelengths, or both lfc_nu0 and lfc_delta_nu."
            )
        theoretical_wavelengths = generate_lfc_wavelengths(
            float(np.nanmin(wave)), float(np.nanmax(wave)), lfc_nu0, lfc_delta_nu,
        )
    theoretical_wavelengths = np.asarray(theoretical_wavelengths, dtype=float)

    chunks, raw_edges = make_chunk_edges(n_pix, n_chunks, chunk_margin_pix)

    psf_cube = np.full((n_sel, n_chunks, n_psf_points), np.nan)
    dx_cube = np.full_like(psf_cube, np.nan)
    var_cube = np.full_like(psf_cube, np.nan)
    ok = np.zeros((n_sel, n_chunks), dtype=np.int16)
    n_lines_used = np.zeros((n_sel, n_chunks), dtype=np.int32)

    for oi, order_1b in enumerate(orders_1based):
        row = int(order_1b) - 1
        spectrum = flux[row]
        spec_err = err[row]
        wave_order = wave[row]
        if verbose:
            print(f"Order {order_1b} ({oi + 1}/{n_sel})")

        for ci, (a, b) in enumerate(chunks):
            x = np.arange(a, b, dtype=float)
            y = spectrum[a:b]
            yerr = np.clip(spec_err[a:b], 1e-6, None)

            try:
                result = compute_psf_chunk_rv(
                    x, y, yerr, wave_order, theoretical_wavelengths,
                    n_psf_points=n_psf_points,
                    verbose=False,
                    **compute_psf_kwargs,
                )
            except Exception as exc:
                warnings.warn(f"order {order_1b} chunk {ci}: PSF fit failed ({exc}); skipped.")
                continue

            psf_cube[oi, ci, :] = result["psf_mean"]
            dx_cube[oi, ci, :] = result["psf_x"]
            var_cube[oi, ci, :] = np.square(result["psf_std"])
            n_lines_used[oi, ci] = result["n_lines_used"]
            ok[oi, ci] = 1

    return dict(
        psf_cube=psf_cube,
        dx_cube=dx_cube,
        var_cube=var_cube,
        ok=ok,
        n_lines_used=n_lines_used,
        chunk_edges=raw_edges,
        orders_1based=np.asarray(orders_1based),
    )
