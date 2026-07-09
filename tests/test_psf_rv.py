import numpy as np

from psfcal import compute_psf_chunk_rv


def test_recovers_kernel_shape_rv(synthetic_rv_chunk):
    d = synthetic_rv_chunk
    kv, ky = d["kernel_v"]

    result = compute_psf_chunk_rv(
        d["x"], d["y"], d["y_err"],
        d["wave_order"], d["theoretical_wavelengths"],
        n_iterations=1,
        n_psf_points=300,
        use_secondary_gp=True,
        verbose=False,
    )

    assert result["n_lines_used"] >= 15
    assert np.all(np.isfinite(result["psf_mean"]))

    px, py = result["psf_x"], result["psf_mean"]
    ky_true_on_grid = np.interp(px, kv, ky, left=0.0, right=0.0)
    ky_true_on_grid /= np.trapz(ky_true_on_grid, px)

    mask = np.abs(px) < 3.0 * d["mps_per_pix"]
    resid = py[mask] - ky_true_on_grid[mask]
    # normalise residual by the peak amplitude of the true (area-normalised) kernel
    rmse_norm = float(np.sqrt(np.mean(resid ** 2))) / np.max(ky_true_on_grid)
    assert rmse_norm < 0.15


def test_raises_with_too_few_lines(synthetic_rv_chunk):
    d = synthetic_rv_chunk
    import pytest
    with pytest.raises(ValueError):
        compute_psf_chunk_rv(
            d["x"], d["y"], d["y_err"],
            d["wave_order"], d["theoretical_wavelengths"][:1],
        )
