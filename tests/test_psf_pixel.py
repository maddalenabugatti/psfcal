import numpy as np

from psfcal import compute_psf_chunk_pixel


def test_recovers_kernel_shape_and_centroid(synthetic_pixel_chunk):
    x, y, y_err, (kx, ky), _ = synthetic_pixel_chunk

    # n_recentre_max=1: with only ~22 widely-spaced lines (a deliberately
    # sparse synthetic chunk, for test speed) the iterative Eq.20
    # re-centring step -- identical behaviour confirmed against the
    # original psf_tools_4.compute_psf on this same data -- does not
    # monotonically converge; a single re-centring step already recovers
    # a well-centred, well-shaped PSF for this data, so we cap it here to
    # get a stable regression check. Real chunks have far denser line
    # combs (e.g. LFC modes ~1-2 pixels apart) where this is a non-issue.
    result = compute_psf_chunk_pixel(
        x, y, y_err,
        n_iterations=1,
        n_psf_points=400,
        use_secondary_gp=True,
        n_recentre_max=1,
        verbose=False,
    )

    assert np.all(np.isfinite(result["psf_mean"]))
    assert abs(result["psf_centroid"]) < 0.2

    # compare recovered PSF against the true kernel on the overlapping domain
    px, py = result["psf_x"], result["psf_mean"]
    ky_true_on_grid = np.interp(px, kx, ky, left=0.0, right=0.0)
    ky_true_on_grid /= np.trapz(ky_true_on_grid, px)

    mask = np.abs(px) < 3.0
    resid = py[mask] - ky_true_on_grid[mask]
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    assert rmse < 0.08

    assert result["rms"] < 10.0


def test_n_iterations_default_is_one(synthetic_pixel_chunk):
    x, y, y_err, *_ = synthetic_pixel_chunk
    result = compute_psf_chunk_pixel(x, y, y_err, n_psf_points=200)
    assert len(result["history"]) == 1
