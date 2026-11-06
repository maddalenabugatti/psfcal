# psfcal

Instrumental-profile (PSF/IP) reconstruction, forward convolution,
Richardson–Lucy deconvolution, and RV-shift (cross-correlation) toolkit for
echelle spectra, built around the GP-based IP reconstruction method of
Milaković & Jethwa (2024), A&A 684, A38.

Operates chunk-by-chunk on plain numpy arrays (pixel or RV/velocity space) —
no FITS I/O baked in.

## Install

Directly from GitHub (needs Python ≥ 3.10):

```bash
pip install "psfcal[plot] @ git+https://github.com/maddalenabugatti/psfcal.git"
```

Or from a local clone, e.g. to run the tests or edit the code:

```bash
git clone https://github.com/maddalenabugatti/psfcal.git
cd psfcal
pip install -e ".[plot,test]"
```

Core dependencies: `numpy`, `scipy`, `jax`, `jaxopt`, `tinygp`.
`matplotlib` is optional (`psfcal[plot]`), needed only for `psfcal.plotting`.

## Tutorials

The notebooks are in [`examples/`](examples/). The first two use synthetic data; the last two run on small real HARPS and ESPRESSO LFC extracts bundled in `examples/data/` (public ESO archive data).

| Notebook | What it shows |
|---|---|
| [`quickstart.ipynb`](examples/quickstart.ipynb) | How to call every function, in pixel and RV space, on synthetic data (the reconstruction is overlaid on the known true PSF) |
| [`parameters.ipynb`](examples/parameters.ipynb) | What the main options **change**, with plots: chunk length and S/N, background subtraction, outlier handling, iterations, secondary GP, deconvolution iterations and damping, chunk geometry. Each option is scored against a known true PSF. |
| [`harps_lfc_real_data.ipynb`](examples/harps_lfc_real_data.ipynb) | A **real HARPS laser-frequency-comb chunk**: reconstruct the PSF, see why sub-pixel stacking works, why background subtraction matters, and deconvolve the comb with its own PSF |
| [`espresso_lfc_real_data.ipynb`](examples/espresso_lfc_real_data.ipynb) | The same on a **real ESPRESSO LFC exposure**: PSFs of three chunks of one order (they differ), comparison with HARPS, background and deconvolution |

The last three notebooks load cached fit results (`examples/parameters_cache.pkl`, `examples/data/*_fit_cache.pkl`), so they open instantly.
Set `RECOMPUTE = True` in a notebook's first cell to refit everything (about 1 min per fit; ~15 min for `parameters.ipynb`).

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

## Choosing options

The defaults work well for HARPS LFC data. The options below are the ones worth knowing about
(measured on a synthetic chunk in [`parameters.ipynb`](examples/parameters.ipynb)).

| Option | Default | Change it when… |
|---|---|---|
| `subtract_background` | `False` in `compute_psf_chunk_pixel`, `True` in `compute_psf_cube_pixel` | your data still contain a background. **The most important option:** leaving it off on a sloped background raised the PSF error from ~4% to ~45%. |
| `n_chunks` / chunk length | 32 chunks per order | a chunk holds too few lines (error roughly halves each time the number of lines doubles) or the PSF varies along the order (use shorter chunks) |
| `n_iterations` | 1 | rarely; 3 iterations gave the same PSF at ~2–3× the run time |
| `use_secondary_gp` | `True` | set `False` for a faster fit with a nearly identical PSF |
| `sigma_clip_nsigma`, `student_t_dof` | off | real outliers (cosmic rays, blends) show in the residuals. Check before enabling: in the notebook the soft `student_t_dof` helped while the hard sigma clip made the PSF worse. |
| `max_half_window`, `min_window_width` | 15, 6 | lines are blended or very unevenly spaced |
| `rl_iters` | 10 | choose from an error-vs-iterations plot on your data; the optimum was ~100–300 in the notebook example |
| `l2_damp` | 3e-6 | keep it small. It acts on the flux **in its own units**: with `continuum_normalize=False` and raw counts of ~1e5 the default stalls the deconvolution, so scale the flux to ~1 first (the default `continuum_normalize=True` does this for you) |
| `n_psf_points` | 800 (pixel) / 500 (RV) | only the sampling of the returned PSF |

Run time: one GP fit takes tens of seconds per chunk (first call also compiles JAX code), so a full cube takes a while.

## Acknowledgements

This package, its tutorials and its documentation were developed with the help of Claude (Anthropic), an AI assistant.
The scientific method, the choices and the validation against HARPS data are the author's.

## Citation

If you use psfcal, please cite the paper that describes its application to HARPS:

> Bugatti, M., Hirano, T., Pepe, F., Lovis, C., Dumusque, X., Tavella, S., et al.
> *PSF correction of HARPS-S data* (in preparation).

and the GP-based IP reconstruction method it implements:

> Milaković, D. & Jethwa, P. 2024, A&A, 684, A38.

GitHub's "Cite this repository" button uses [`CITATION.cff`](CITATION.cff).

## Testing

```bash
pytest -q
```
