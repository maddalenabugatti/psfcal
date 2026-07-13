"""
psfcal — instrumental-profile reconstruction, forward convolution,
Richardson-Lucy deconvolution and RV-shift toolkit for echelle spectra.
"""

from .psf_pixel import compute_psf_chunk_pixel
from .psf_rv import compute_psf_chunk_rv
from .deconvolve import deconvolve_chunk_pixel, deconvolve_chunk_rv, deconvolve_order
from .convolve import convolve_chunk_pixel, convolve_chunk_rv
from .rv_shift import compute_rv_shift
from .psf_cube import (
    compute_psf_cube_pixel,
    compute_psf_cube_rv,
    make_chunk_edges,
    generate_lfc_wavelengths,
)

__version__ = "0.1.0"

__all__ = [
    "compute_psf_chunk_pixel",
    "compute_psf_chunk_rv",
    "deconvolve_chunk_pixel",
    "deconvolve_chunk_rv",
    "deconvolve_order",
    "convolve_chunk_pixel",
    "convolve_chunk_rv",
    "compute_rv_shift",
    "compute_psf_cube_pixel",
    "compute_psf_cube_rv",
    "make_chunk_edges",
    "generate_lfc_wavelengths",
]
