"""reference_spectra.py — Load and store Vega and Solar reference spectra.

Spectra are returned as JAX arrays with:
    wavelength [Å], flux [erg / s / cm² / Å]
"""

import jax.numpy as jnp
from astropy.io import fits
from pathlib import Path
import numpy as np
from importlib.resources import files


__all__ = ["vega", "solar"]

# Conversion: from 1 AU to 10 pc solid angle
AU_TO_10PC_SOLID_ANGLE_RATIO = (1.0 / (3600 * 180 / np.pi * 10))**2


def _load_fits_spectrum(path, flux_scale=1.0):
    """Load a 2-column (wavelength, flux) spectrum from a FITS file,
    and return it as a JAX array (converted from native-endian float32).
    """
    with fits.open(path) as hdul:
        wave = np.array(hdul[1].data["WAVELENGTH"]).astype("<f8")  # Force native-endian
        flux = np.array(hdul[1].data["FLUX"]).astype("<f8") * flux_scale
    return jnp.column_stack((wave, flux))


# ----------
# Vega Spectrum
# ----------

vega_path = Path(__file__).resolve().parent / "data" / "alpha_lyr_stis_005.fits"

if not vega_path.exists():
    raise FileNotFoundError(f"Vega reference spectrum not found at: {vega_path}")

vega = _load_fits_spectrum(vega_path)

# ----------
# Solar Spectrum (converted to 10 pc)
# ----------
try:
    solar_path = files("sedpy_jax").joinpath("data/sun_kurucz93.fits")
    solar = _load_fits_spectrum(solar_path, flux_scale=AU_TO_10PC_SOLID_ANGLE_RATIO)
except Exception as e:
    raise FileNotFoundError("Could not load Solar reference spectrum.") from e