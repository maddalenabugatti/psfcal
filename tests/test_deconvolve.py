import numpy as np

from psfcal import (
    convolve_chunk_pixel,
    convolve_chunk_rv,
    deconvolve_chunk_pixel,
    deconvolve_chunk_rv,
    deconvolve_order,
)


def _true_spectrum(n_pix=200, rng=None):
    rng = rng or np.random.default_rng(7)
    flux = np.full(n_pix, 100.0)
    centres = rng.choice(np.arange(15, n_pix - 15), size=10, replace=False)
    for c in centres:
        flux[c] += rng.uniform(500, 3000)
    return flux


def test_deconvolve_pixel_round_trip(true_kernel):
    kx, ky = true_kernel
    mask = np.abs(kx) <= 7
    psf_x, psf_y = kx[mask], ky[mask]

    true_flux = _true_spectrum()
    observed = convolve_chunk_pixel(true_flux, psf_x, psf_y, pixel_radius=7)
    recovered = deconvolve_chunk_pixel(
        observed, psf_x, psf_y, pixel_radius=7, rl_iters=150, overlap_pix=10,
    )

    core = slice(20, -20)
    # normalise by the dynamic range rather than per-pixel value: RL with a
    # finite iteration count leaves mild ringing in the near-baseline
    # pixels next to a sharp spike, which blows up a per-pixel relative
    # error but is not indicative of a broken deconvolution.
    scale = true_flux[core].max() - true_flux[core].min()
    nrmse = np.sqrt(np.mean((recovered[core] - true_flux[core]) ** 2)) / scale
    assert nrmse < 0.05


def test_deconvolve_rv_round_trip(true_kernel):
    kx, ky = true_kernel
    mask = np.abs(kx) <= 7
    psf_x, psf_y = kx[mask], ky[mask]
    mps_per_pix = 800.0
    ip_v = psf_x * mps_per_pix

    true_flux = _true_spectrum()
    observed = convolve_chunk_rv(true_flux, ip_v, psf_y, mps_per_pix=mps_per_pix, pixel_radius=7)
    recovered = deconvolve_chunk_rv(
        observed, ip_v, psf_y, mps_per_pix=mps_per_pix, pixel_radius=7, rl_iters=150, overlap_pix=10,
    )

    core = slice(20, -20)
    scale = true_flux[core].max() - true_flux[core].min()
    nrmse = np.sqrt(np.mean((recovered[core] - true_flux[core]) ** 2)) / scale
    assert nrmse < 0.05


def test_deconvolve_order_overlap_add_no_seams(true_kernel):
    kx, ky = true_kernel
    mask = np.abs(kx) <= 7
    psf_x, psf_y = kx[mask], ky[mask]

    n_pix = 400
    n_chunks = 4
    edges = np.linspace(0, n_pix, n_chunks + 1).astype(int)
    chunk_edges = np.stack([edges[:-1], edges[1:]], axis=1)

    true_flux = _true_spectrum(n_pix=n_pix)
    observed = convolve_chunk_pixel(true_flux, psf_x, psf_y, pixel_radius=7)

    psi_cube = np.tile(psf_y, (n_chunks, 1))
    offsets_cube = np.tile(psf_x, (n_chunks, 1))

    recovered = deconvolve_order(
        observed, psi_cube, offsets_cube, chunk_edges,
        scale="pixel", rl_iters=40, pixel_radius=7, overlap_pix=12,
        continuum_normalize=False,
    )

    # single-chunk deconvolution of the whole thing should match closely
    # away from the array edges (sanity check that stitching doesn't
    # introduce large seam artefacts at the internal chunk boundaries)
    single = deconvolve_chunk_pixel(observed, psf_x, psf_y, pixel_radius=7, rl_iters=40)

    core = slice(20, -20)
    diff = np.abs(recovered[core] - single[core])
    assert np.median(diff) < 5.0
