import numpy as np
import pytest

from psfcal import (
    compute_psf_cube_pixel,
    compute_psf_cube_rv,
    deconvolve_order,
    make_chunk_edges,
    generate_lfc_wavelengths,
)

C_LIGHT_MPS = 299_792_458.0


def test_make_chunk_edges_no_margin():
    chunks, raw_edges = make_chunk_edges(100, 4, margin=0)
    assert raw_edges.tolist() == [[0, 25], [25, 50], [50, 75], [75, 100]]
    assert chunks == [(0, 25), (25, 50), (50, 75), (75, 100)]


def test_make_chunk_edges_with_margin_clips_at_boundaries():
    chunks, raw_edges = make_chunk_edges(100, 4, margin=10)
    # margin extends each chunk but clips at the array edges
    assert chunks[0] == (0, 35)  # left edge clipped to 0
    assert chunks[-1] == (65, 100)  # right edge clipped to 100
    assert chunks[1] == (15, 60)
    # raw_edges (no margin) unaffected
    assert raw_edges.tolist() == [[0, 25], [25, 50], [50, 75], [75, 100]]


def test_generate_lfc_wavelengths_matches_comb_equation():
    nu0, delta_nu = 288_060_050_003_000.0, 18.0e9
    wave_min, wave_max = 500e-9, 500.5e-9
    wav = generate_lfc_wavelengths(wave_min, wave_max, nu0, delta_nu)
    assert len(wav) > 0
    assert np.all(wav >= wave_min) and np.all(wav <= wave_max)
    # each wavelength should correspond to an (approximately) integer mode number
    ns = (C_LIGHT_MPS / wav - nu0) / delta_nu
    assert np.allclose(ns, np.round(ns), atol=1e-6)


def _make_kernel():
    dx = np.linspace(-8, 8, 4000)
    g1 = np.exp(-0.5 * (dx / 0.9) ** 2)
    g2 = 0.35 * np.exp(-0.5 * ((dx - 0.4) / 1.6) ** 2)
    y = g1 + g2
    y /= np.trapz(y, dx)
    c = np.trapz(dx * y, dx)
    dx = dx - c
    y /= np.trapz(y, dx)
    return dx, y


def _make_synthetic_orders(n_orders=2, n_pix=400, seed=0):
    kx, ky = _make_kernel()
    rng = np.random.default_rng(seed)
    line_centres = np.arange(2, n_pix - 2, 18.3)

    flux = np.zeros((n_orders, n_pix))
    err = np.zeros((n_orders, n_pix))
    for row in range(n_orders):
        x = np.arange(n_pix, dtype=float)
        y = np.full(n_pix, 50.0)
        for c in line_centres:
            y += 4000.0 * np.interp(x - c, kx, ky, left=0.0, right=0.0)
        y_err = np.sqrt(np.abs(y) + 1.0)
        flux[row] = np.clip(y + rng.normal(0.0, y_err), 0.0, None)
        err[row] = y_err
    return flux, err, kx, ky, line_centres


def test_compute_psf_cube_pixel_basic():
    flux, err, *_ = _make_synthetic_orders(n_orders=2, n_pix=400)

    cube = compute_psf_cube_pixel(
        flux, err, orders_1based=[1, 2],
        n_chunks=2, chunk_margin_pix=5, n_psf_points=200,
        n_recentre_max=1, use_secondary_gp=False,
    )

    assert cube["psi_cube"].shape == (2, 2, 200)
    assert cube["dx_cube"].shape == (2, 2, 200)
    assert cube["var_cube"].shape == (2, 2, 200)
    assert cube["ok"].shape == (2, 2)
    assert np.all(cube["ok"] == 1)
    assert np.all(np.isfinite(cube["psi_cube"]))
    np.testing.assert_array_equal(cube["orders_1based"], [1, 2])

    # feeds directly into deconvolve_order
    deconv = deconvolve_order(
        flux[0], cube["psi_cube"][0], cube["dx_cube"][0], cube["chunk_edges"],
        scale="pixel", ok_chunks=cube["ok"][0], rl_iters=5,
    )
    assert deconv.shape == flux[0].shape
    assert np.all(np.isfinite(deconv))


def test_compute_psf_cube_pixel_skips_failed_chunk_gracefully():
    flux, err, *_ = _make_synthetic_orders(n_orders=1, n_pix=400)
    # blank out the whole second half (well past any real line's tail) so
    # its chunk has no peaks to detect at all
    flux[0, 150:400] = 50.0
    err[0, 150:400] = np.sqrt(50.0)

    with pytest.warns(UserWarning, match="PSF fit failed"):
        cube = compute_psf_cube_pixel(
            flux, err, orders_1based=[1],
            n_chunks=2, chunk_margin_pix=0, n_psf_points=150,
            n_recentre_max=1, use_secondary_gp=False,
        )

    assert cube["ok"][0, 0] == 1
    assert cube["ok"][0, 1] == 0
    assert np.all(np.isnan(cube["psi_cube"][0, 1]))


def _make_synthetic_rv_orders(n_orders=2, n_pix=400, seed=1):
    kx, ky = _make_kernel()
    rng = np.random.default_rng(seed)
    mps_per_pix = 800.0
    lam0 = 5000e-10

    flux = np.zeros((n_orders, n_pix))
    err = np.zeros((n_orders, n_pix))
    wave = np.zeros((n_orders, n_pix))
    for row in range(n_orders):
        x = np.arange(n_pix, dtype=float)
        wave_order = lam0 + x * (lam0 * mps_per_pix / C_LIGHT_MPS)
        kv = kx * mps_per_pix
        line_pix_centres = np.arange(2, n_pix - 2, 9.15)
        theo_wav = np.interp(line_pix_centres, x, wave_order)
        y = np.full(n_pix, 50.0)
        for lam_theo in theo_wav:
            dv = C_LIGHT_MPS * (wave_order - lam_theo) / lam_theo
            y += 4000.0 * np.interp(dv, kv, ky, left=0.0, right=0.0)
        y_err = np.sqrt(np.abs(y) + 1.0)
        flux[row] = np.clip(y + rng.normal(0.0, y_err), 0.0, None)
        err[row] = y_err
        wave[row] = wave_order
    return flux, err, wave


def test_compute_psf_cube_rv_basic():
    flux, err, wave = _make_synthetic_rv_orders(n_orders=2, n_pix=400)
    # pass the exact theoretical grid used to build the synthetic data
    x = np.arange(400, dtype=float)
    lam0 = 5000e-10
    mps_per_pix = 800.0
    wave_order0 = lam0 + x * (lam0 * mps_per_pix / C_LIGHT_MPS)
    line_pix_centres = np.arange(2, 400 - 2, 9.15)
    theoretical_wavelengths = np.interp(line_pix_centres, x, wave_order0)

    cube = compute_psf_cube_rv(
        flux, err, wave, orders_1based=[1, 2],
        theoretical_wavelengths=theoretical_wavelengths,
        n_chunks=2, chunk_margin_pix=5, n_psf_points=150,
    )

    assert cube["psi_cube"].shape == (2, 2, 150)
    assert cube["ok"].shape == (2, 2)
    assert np.all(cube["ok"] == 1)
    assert np.all(cube["n_lines_used"] >= 2)
    assert np.all(np.isfinite(cube["psi_cube"]))

    deconv = deconvolve_order(
        flux[0], cube["psi_cube"][0], cube["dx_cube"][0], cube["chunk_edges"],
        scale="rv", wave_order=wave[0], ok_chunks=cube["ok"][0], rl_iters=5,
    )
    assert deconv.shape == flux[0].shape
    assert np.all(np.isfinite(deconv))


def test_compute_psf_cube_rv_requires_theoretical_source():
    flux, err, wave = _make_synthetic_rv_orders(n_orders=1, n_pix=200)
    with pytest.raises(ValueError):
        compute_psf_cube_rv(flux, err, wave, orders_1based=[1], n_chunks=2)
