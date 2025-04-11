"""reference_spectra.py — Load and store Vega and Solar reference spectra.

Spectra are returned as JAX arrays with:
    wavelength [Å], flux [erg / s / cm² / Å]
"""

import jax.numpy as jnp
import numpy as np
from importlib.resources import files
from astropy.io import fits as pyfits

__all__ = ["vega", "solar"]

# Conversion: from 1 AU to 10 pc solid angle
AU_TO_10PC_SOLID_ANGLE_RATIO = (1.0 / (3600 * 180 / np.pi * 10))**2

def _load_fits_spectrum(path, flux_scale=1.0):
    with pyfits.open(path) as hdul:
        wave = hdul[1].data["WAVELENGTH"]
        flux = hdul[1].data["FLUX"] * flux_scale
    return jnp.column_stack((wave, flux))


# ----------
# Vega Spectrum
# ----------
try:
    vega_path = files("sedpy_jax").joinpath("data/alpha_lyr_stis_005.fits")
    vega = _load_fits_spectrum(vega_path)
except Exception as e:
    raise FileNotFoundError("Could not load Vega reference spectrum.") from e

# ----------
# Solar Spectrum (converted to 10 pc)
# ----------
try:
    solar_path = files("sedpy_jax").joinpath("data/sun_kurucz93.fits")
    solar = _load_fits_spectrum(solar_path, flux_scale=AU_TO_10PC_SOLID_ANGLE_RATIO)
except Exception as e:
    raise FileNotFoundError("Could not load Solar reference spectrum.") from e