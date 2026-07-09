import numpy as np

from psfcal import convolve_chunk_pixel, convolve_chunk_rv


def test_convolve_pixel_matches_manual_placement(true_kernel):
    kx, ky = true_kernel
    mask = np.abs(kx) <= 7
    psf_x, psf_y = kx[mask], ky[mask]

    n_pix = 100
    flux = np.zeros(n_pix)
    flux[50] = 1000.0  # single delta line

    conv = convolve_chunk_pixel(flux, psf_x, psf_y, pixel_radius=7)

    # convolving a delta at pixel 50 should reproduce the (renormalised,
    # centroid-centred) kernel shape around pixel 50
    offs = np.arange(-7, 8)
    expected = np.interp(offs, psf_x, psf_y, left=0.0, right=0.0)
    expected = np.clip(expected, 0.0, None)
    expected = expected / expected.sum() * 1000.0

    got = conv[50 - 7: 50 + 8]
    assert np.allclose(got, expected, atol=1.0)
    # energy should be (approximately) conserved
    assert abs(conv.sum() - flux.sum()) < 5.0


def test_convolve_rv_matches_pixel_when_dispersion_constant(true_kernel):
    kx, ky = true_kernel
    mask = np.abs(kx) <= 7
    psf_x, psf_y = kx[mask], ky[mask]

    mps_per_pix = 800.0
    ip_v = psf_x * mps_per_pix

    n_pix = 100
    flux = np.zeros(n_pix)
    flux[50] = 1000.0

    conv_pixel = convolve_chunk_pixel(flux, psf_x, psf_y, pixel_radius=7)
    conv_rv = convolve_chunk_rv(flux, ip_v, psf_y, mps_per_pix=mps_per_pix, pixel_radius=7)

    assert np.allclose(conv_pixel, conv_rv, atol=1e-6)
