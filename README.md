# psfcal

Instrumental-profile (PSF/IP) reconstruction, forward convolution,
Richardson–Lucy deconvolution, and RV-shift (cross-correlation) toolkit for
echelle spectra, built around the GP-based IP reconstruction method of
Milaković & Jethwa (2024), A&A 684, A38.

Operates chunk-by-chunk on plain numpy arrays (pixel or RV/velocity space) —
no FITS I/O baked in.

## Install

```bash
pip install -e .[plot,test]
```

Core dependencies: `numpy`, `scipy`, `jax`, `jaxopt`, `tinygp`.
`matplotlib` is optional (`psfcal[plot]`), needed only for `psfcal.plotting`.

## API

| # | Function | Purpose |
|---|----------|---------|
| 1 | `compute_psf_chunk_pixel(x, y, y_err=None, n_iterations=1, ...)` | Reconstruct the PSF/IP for a chunk in pixel space |
| 2 | `compute_psf_chunk_rv(x, y, y_err, wave_order, theoretical_wavelengths, n_iterations=1, ...)` | Reconstruct the PSF/IP for a chunk in RV space, given a wavelength solution and theoretical line positions |
| 3 | `deconvolve_chunk_pixel(flux, psf_x, psf_y, ...)` | Richardson–Lucy deconvolve a chunk with a pixel-space PSF |
| 4 | `deconvolve_chunk_rv(flux, ip_v, ip_y, wave_chunk=..., ...)` | Richardson–Lucy deconvolve a chunk with an RV-space IP |
| 5 | `deconvolve_order(order_flux, psi_cube, offsets_cube, chunk_edges, scale="pixel"\|"rv", ...)` | Deconvolve a full order from per-chunk PSFs (overlap-add) |
| 6 | `convolve_chunk_pixel(flux, psf_x, psf_y, ...)` | Forward-convolve a chunk with a pixel-space PSF |
| 7 | `convolve_chunk_rv(flux, ip_v, ip_y, wave_chunk=..., ...)` | Forward-convolve a chunk with an RV-space PSF |
| 8 | `compute_rv_shift(wave1, flux1, wave2, flux2, ...)` | RV shift between two spectra via cross-correlation + Gaussian peak fit |

### Multi-order/multi-chunk cube helpers

Thin orchestration layer over items 1/2 — loops over chunks and orders and
assembles the results into a cube compatible with `deconvolve_order`.

| Function | Purpose |
|----------|---------|
| `compute_psf_cube_pixel(flux, err, orders_1based, n_chunks=32, ...)` | Run `compute_psf_chunk_pixel` over every chunk of every order |
| `compute_psf_cube_rv(flux, err, wave, orders_1based, theoretical_wavelengths=..., n_chunks=32, ...)` | Run `compute_psf_chunk_rv` over every chunk of every order |
| `make_chunk_edges(n_pix, n_chunks, margin=0)` | Split one order into chunk index ranges (with/without margin) |
| `generate_lfc_wavelengths(wave_min, wave_max, nu0, delta_nu)` | Theoretical LFC comb wavelengths from the comb equation |

```python
cube = psfcal.compute_psf_cube_pixel(flux, err, orders_1based=[1, 2, 3], n_chunks=32)
# cube["psi_cube"], cube["dx_cube"], cube["chunk_edges"], cube["ok"] feed
# straight into deconvolve_order, one order (index i) at a time:
deconv = psfcal.deconvolve_order(
    flux[orders_1based[i] - 1], cube["psi_cube"][i], cube["dx_cube"][i],
    cube["chunk_edges"], scale="pixel", ok_chunks=cube["ok"][i],
)
```

```python
import psfcal

psf = psfcal.compute_psf_chunk_pixel(x, y, y_err)
deconv = psfcal.deconvolve_chunk_pixel(flux, psf["psf_x"], psf["psf_mean"])
recon = psfcal.convolve_chunk_pixel(deconv, psf["psf_x"], psf["psf_mean"])

shift = psfcal.compute_rv_shift(wave1, flux1, wave2, flux2)
print(shift["rv"], shift["rv_err"])
```

## Testing

```bash
pytest -q
```
