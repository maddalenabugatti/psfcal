import numpy as np

from psfcal import compute_rv_shift

C_LIGHT_MPS = 299_792_458.0


def _absorption_spectrum(wave, line_centres, depth=0.4, width=0.03e-10):
    flux = np.ones_like(wave)
    for c in line_centres:
        flux -= depth * np.exp(-0.5 * ((wave - c) / width) ** 2)
    return flux


def test_recovers_known_velocity_shift():
    wave = np.linspace(5000e-10, 5010e-10, 4000)
    line_centres = np.linspace(5001e-10, 5009e-10, 12)

    flux1 = _absorption_spectrum(wave, line_centres)

    v_true = 3500.0  # m/s
    shifted_centres = line_centres * (1.0 + v_true / C_LIGHT_MPS)
    flux2 = _absorption_spectrum(wave, shifted_centres)

    result = compute_rv_shift(
        wave, flux1, wave, flux2,
        velocity_range=(-10_000.0, 10_000.0),
        velocity_step=100.0,
        continuum_normalize=False,
    )

    assert abs(result["rv"] - v_true) < 150.0
    assert result["rv_err"] is None or result["rv_err"] >= 0 or np.isnan(result["rv_err"])


def test_zero_shift_recovers_zero():
    wave = np.linspace(5000e-10, 5010e-10, 4000)
    line_centres = np.linspace(5001e-10, 5009e-10, 12)
    flux1 = _absorption_spectrum(wave, line_centres)

    result = compute_rv_shift(
        wave, flux1, wave, flux1.copy(),
        velocity_range=(-5_000.0, 5_000.0),
        velocity_step=100.0,
        continuum_normalize=False,
    )
    assert abs(result["rv"]) < 100.0
