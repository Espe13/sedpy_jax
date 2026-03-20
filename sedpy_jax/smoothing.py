"""smoothing.py — JAX spectral smoothing for sedpy_jax.

Three FFT-based broadening functions, all JAX-traceable through ``spec`` and
the smoothing width parameter when the wavelength grids are fixed:

    smooth_vel_fft   — constant velocity width σ_v  [km/s],  log-λ FFT
    smooth_wave_fft  — constant wavelength width σ_λ [Å],    linear-λ FFT
    smooth_lsf_fft   — wavelength-dependent LSF σ(λ) [Å],   CDF-transform FFT

A low-level kernel used by all three methods:

    smooth_fft       — Gaussian multiplication in Fourier space

A top-level dispatcher matching the sedpy API:

    smoothspec       — routes to the appropriate method

Factory functions that precompute the wavelength grid once and return a
fully JIT-compilable closure (recommended for SED fitting loops):

    make_vel_smoother(wave, outwave)  →  smoother(spec, sigma_v)
    make_wave_smoother(wave, outwave) →  smoother(spec, sigma_l)
    make_lsf_smoother(wave, sigma_lsf, outwave) →  smoother(spec)

Physical conventions
--------------------
- Wavelengths in Angstroms, monotonically increasing.
- All σ are Gaussian *dispersions* (not FWHM); FWHM = σ × 2√(2 ln 2) ≈ 2.355 σ.
- Velocity dispersions in km/s.
- Input resolution ``inres`` is subtracted in quadrature:
  σ_eff = √(σ_target² − σ_inres²).

JAX traceability
----------------
``smooth_fft``, and the closures returned by the factory functions, are fully
JAX-differentiable with respect to ``spec`` and the width parameter.  The
``wave`` and ``outwave`` arrays must be *concrete* (not traced) because the
FFT grid size is determined from ``len(wave)`` at call time.  In a fitting
loop, these are fixed instrument/model properties, so this is never an issue.
"""

import numpy as np
import jax.numpy as jnp
from .observate import jax_interp

__all__ = [
    "smooth_fft",
    "smooth_vel_fft",
    "smooth_wave_fft",
    "smooth_lsf_fft",
    "smoothspec",
    "make_vel_smoother",
    "make_wave_smoother",
    "make_lsf_smoother",
]

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

CKMS = 2.998e5          # speed of light [km/s]
SIGMA_TO_FWHM = 2.3548  # 2√(2 ln 2)


# ---------------------------------------------------------------------------
# Low-level FFT kernel
# ---------------------------------------------------------------------------

def smooth_fft(dx, spec, sigma):
    """Gaussian convolution in Fourier space.

    Multiplies the one-sided Fourier transform of ``spec`` by the analytical
    Fourier transform of a Gaussian:

        taper(ν) = exp(−2π² σ² ν²)

    where ν = rfftfreq(N) / dx has units [1/dx].

    This implements an exact circular (periodic) convolution.  Edge effects
    are present if the spectrum is not smooth at both ends; pad the input
    spectrum before calling if this is a concern.

    Parameters
    ----------
    dx : float
        Pixel spacing, *same units as* ``sigma``.  For velocity smoothing
        this is [km/s]; for wavelength smoothing this is [Å].
    spec : (N,) JAX array
        Flux on a uniform grid.  Must be on a *regularly spaced* grid with
        spacing ``dx``.
    sigma : float or scalar JAX array
        Gaussian dispersion, *same units as* ``dx``.

    Returns
    -------
    (N,) JAX array
        Smoothed spectrum on the same grid as ``spec``.
    """
    N = spec.shape[0]
    # Fourier frequencies in cycles per unit [dx], shape (N//2+1,)
    nu = jnp.fft.rfftfreq(N, d=dx)
    # Analytical FT of a Gaussian; taper → 1 at ν=0, → 0 at high ν
    taper = jnp.exp(-2.0 * jnp.pi**2 * sigma**2 * nu**2)
    # Forward FFT → multiply by taper → inverse FFT
    # n=N ensures irfft returns exactly N points regardless of parity
    return jnp.fft.irfft(jnp.fft.rfft(spec) * taper, n=N)


# ---------------------------------------------------------------------------
# Grid helpers (NumPy — called once at initialisation, not inside JIT)
# ---------------------------------------------------------------------------

def _log_grid(wave):
    """Resample ``wave`` to a log-uniform grid of length 2^k [NumPy].

    Returns
    -------
    w_log : (nnew,) ndarray
        Log-uniform wavelength grid covering the same range as ``wave``.
    dv : float
        Pixel spacing in velocity units [km/s]:  dv = c × Δ(ln λ).
    """
    wave = np.asarray(wave, dtype=np.float64)
    nw   = len(wave)
    nnew = int(2 ** np.ceil(np.log2(nw)))
    ln_min = np.log(wave[0])
    ln_max = np.log(wave[-1])
    w_log  = np.exp(np.linspace(ln_min, ln_max, nnew))
    dv     = CKMS * (ln_max - ln_min) / (nnew - 1)  # [km/s per pixel]
    return w_log, dv


def _lin_grid(wave):
    """Resample ``wave`` to a linear-uniform grid of length 2^k [NumPy].

    Returns
    -------
    w_lin : (nnew,) ndarray
        Linear-uniform wavelength grid covering the same range as ``wave``.
    dw : float
        Pixel spacing [Å].
    """
    wave  = np.asarray(wave, dtype=np.float64)
    nw    = len(wave)
    nnew  = int(2 ** np.ceil(np.log2(nw)))
    w_lin = np.linspace(wave[0], wave[-1], nnew)
    dw    = (wave[-1] - wave[0]) / (nnew - 1)  # [Å per pixel]
    return w_lin, dw


def _lsf_grid(wave, sigma_lsf, pix_per_sigma=2):
    """Compute the CDF-transform grid for wavelength-dependent LSF [NumPy].

    The coordinate transform x(λ) = ∫_λ0^λ dλ′/σ(λ′) maps the input grid
    to one where the LSF dispersion is (approximately) constant in x-units,
    enabling a single FFT convolution.

    Parameters
    ----------
    wave : (N,) array
        Input wavelength grid [Å].
    sigma_lsf : (N,) array
        Gaussian dispersion of the LSF at each wavelength [Å].
    pix_per_sigma : float, optional
        Minimum number of pixels per σ in the resampled x-grid.
        Higher values improve accuracy at the cost of more FFT points.

    Returns
    -------
    lam : (nx,) ndarray
        Wavelength values corresponding to the uniform x-grid.
    dx : float
        Pixel spacing of the uniform x-grid (dimensionless).
    x_per_sigma : float
        LSF width in units of the x-grid spacing.
    """
    wave      = np.asarray(wave,     dtype=np.float64)
    sigma_lsf = np.asarray(sigma_lsf, dtype=np.float64)

    # CDF of dλ/σ(λ) — the coordinate x is proportional to "resolution units"
    dw  = np.gradient(wave)
    cdf = np.cumsum(dw / sigma_lsf)
    cdf /= cdf[-1]

    # Resolution of the x-coordinate: x per original pixel, x per sigma
    x_per_pixel = np.gradient(cdf)
    sigma_per_pixel = dw / sigma_lsf                      # Δλ / σ(λ) [dimensionless]
    x_per_sigma = float(np.nanmedian(x_per_pixel / sigma_per_pixel))

    # Number of points on the x-grid: smallest power of 2 that is critically sampled
    nx = int(2 ** np.ceil(np.log2(pix_per_sigma / x_per_sigma)))

    # Uniform x-grid and corresponding λ-grid via inverse CDF
    x   = np.linspace(0.0, 1.0, nx)
    dx  = 1.0 / nx                     # spacing of x-grid [dimensionless]
    lam = np.interp(x, cdf, wave)

    return lam, dx, x_per_sigma


# ---------------------------------------------------------------------------
# Functional API  (wave/outwave must be concrete, spec and sigma are traced)
# ---------------------------------------------------------------------------

def smooth_vel_fft(wave, spec, sigma_v, outwave, inres=0.0):
    """Broaden a spectrum by a Gaussian of constant velocity width (FFT).

    Works in log-λ space where a constant velocity Gaussian is a constant
    *additive* shift in ln λ.  The spectrum is interpolated onto a
    log-uniform grid of length 2^k, convolved in Fourier space, then
    interpolated to ``outwave``.

    Parameters
    ----------
    wave : (N,) array-like
        Input wavelength grid [Å], monotonically increasing.  Must be
        *concrete* (not a traced JAX value).
    spec : (N,) JAX array
        Input flux.  JAX-differentiable.
    sigma_v : float or scalar JAX array
        Target Gaussian dispersion [km/s].  JAX-differentiable.
    outwave : (M,) array-like
        Output wavelength grid [Å].  Must be concrete.
    inres : float, optional
        Intrinsic velocity resolution of the input spectrum [km/s].
        Subtracted in quadrature: σ_eff = √(σ_v² − inres²).

    Returns
    -------
    (M,) JAX array
        Smoothed flux on ``outwave``.
    """
    sigma_eff = jnp.sqrt(jnp.maximum(sigma_v**2 - inres**2, 0.0))

    w_log, dv = _log_grid(np.asarray(wave))
    w_log_j   = jnp.asarray(w_log)
    out_j     = jnp.asarray(outwave)

    spec_log  = jax_interp(w_log_j, jnp.asarray(wave), spec)
    spec_conv = smooth_fft(dv, spec_log, sigma_eff)
    return jax_interp(out_j, w_log_j, spec_conv)


def smooth_wave_fft(wave, spec, sigma_l, outwave, inres=0.0):
    """Broaden a spectrum by a Gaussian of constant wavelength width (FFT).

    Works in linear-λ space where a constant-λ Gaussian is a constant
    additive shift in λ.  The spectrum is interpolated onto a linear-uniform
    grid of length 2^k, convolved in Fourier space, then interpolated to
    ``outwave``.

    Parameters
    ----------
    wave : (N,) array-like
        Input wavelength grid [Å], monotonically increasing.  Concrete.
    spec : (N,) JAX array
        Input flux.  JAX-differentiable.
    sigma_l : float or scalar JAX array
        Target Gaussian dispersion [Å].  JAX-differentiable.
    outwave : (M,) array-like
        Output wavelength grid [Å].  Concrete.
    inres : float, optional
        Intrinsic wavelength resolution of input spectrum [Å].
        Subtracted in quadrature.

    Returns
    -------
    (M,) JAX array
        Smoothed flux on ``outwave``.
    """
    sigma_eff = jnp.sqrt(jnp.maximum(sigma_l**2 - inres**2, 0.0))

    w_lin, dw = _lin_grid(np.asarray(wave))
    w_lin_j   = jnp.asarray(w_lin)
    out_j     = jnp.asarray(outwave)

    spec_lin  = jax_interp(w_lin_j, jnp.asarray(wave), spec)
    spec_conv = smooth_fft(dw, spec_lin, sigma_eff)
    return jax_interp(out_j, w_lin_j, spec_conv)


def smooth_lsf_fft(wave, spec, sigma_lsf, outwave, pix_per_sigma=2):
    """Broaden a spectrum by a wavelength-dependent Gaussian LSF (FFT).

    The coordinate transform x(λ) = ∫ dλ/σ(λ) maps the problem to one
    where the LSF dispersion is constant in x-units, enabling a single
    FFT convolution.  The spectrum is interpolated to a uniform x-grid,
    convolved, then interpolated back to ``outwave``.

    Reference: Appendix of Cappellari (2017), MNRAS 466, 798.

    Parameters
    ----------
    wave : (N,) array-like
        Input wavelength grid [Å], monotonically increasing.  Concrete.
    spec : (N,) JAX array
        Input flux.  JAX-differentiable.
    sigma_lsf : (N,) array-like
        Gaussian dispersion of the LSF at each wavelength [Å].  Must be
        *concrete* (usually a fixed instrumental property).
    outwave : (M,) array-like
        Output wavelength grid [Å].  Concrete.
    pix_per_sigma : float, optional
        Minimum pixels per σ on the x-grid.  Increase for higher accuracy.

    Returns
    -------
    (M,) JAX array
        Smoothed flux on ``outwave``.
    """
    lam, dx, x_per_sigma = _lsf_grid(
        np.asarray(wave), np.asarray(sigma_lsf), pix_per_sigma
    )
    lam_j  = jnp.asarray(lam)
    wave_j = jnp.asarray(wave)
    out_j  = jnp.asarray(outwave)

    # Interpolate spectrum to uniform x-grid (via lam(x) ↔ λ)
    spec_x    = jax_interp(lam_j, wave_j, spec)
    # Convolve with Gaussian of width x_per_sigma on a dx-spaced grid
    spec_conv = smooth_fft(dx, spec_x, x_per_sigma)
    # Interpolate to output wavelength grid
    return jax_interp(out_j, lam_j, spec_conv)


# ---------------------------------------------------------------------------
# Factory functions  (recommended for repeated calls inside a fitting loop)
# ---------------------------------------------------------------------------

def make_vel_smoother(wave, outwave, inres=0.0):
    """Return a JIT-compilable velocity smoother with precomputed grid.

    Call this *once* at initialisation.  The returned function is fully
    JAX-differentiable with respect to both ``spec`` and ``sigma_v``.

    Parameters
    ----------
    wave : (N,) array-like
        Input wavelength grid [Å].  Fixed for lifetime of the smoother.
    outwave : (M,) array-like
        Output wavelength grid [Å].  Fixed.
    inres : float, optional
        Intrinsic velocity resolution of input [km/s], subtracted in quadrature.

    Returns
    -------
    smoother : callable
        ``smoother(spec, sigma_v)`` → (M,) JAX array.
    """
    w_log, dv = _log_grid(np.asarray(wave))
    w_log_j   = jnp.asarray(w_log)
    wave_j    = jnp.asarray(wave)
    out_j     = jnp.asarray(outwave)

    def smoother(spec, sigma_v):
        sigma_eff = jnp.sqrt(jnp.maximum(sigma_v**2 - inres**2, 0.0))
        spec_log  = jax_interp(w_log_j, wave_j, spec)
        spec_conv = smooth_fft(dv, spec_log, sigma_eff)
        return jax_interp(out_j, w_log_j, spec_conv)

    return smoother


def make_wave_smoother(wave, outwave, inres=0.0):
    """Return a JIT-compilable wavelength smoother with precomputed grid.

    Parameters
    ----------
    wave : (N,) array-like
        Input wavelength grid [Å].  Fixed.
    outwave : (M,) array-like
        Output wavelength grid [Å].  Fixed.
    inres : float, optional
        Intrinsic wavelength resolution of input [Å], subtracted in quadrature.

    Returns
    -------
    smoother : callable
        ``smoother(spec, sigma_l)`` → (M,) JAX array.
    """
    w_lin, dw = _lin_grid(np.asarray(wave))
    w_lin_j   = jnp.asarray(w_lin)
    wave_j    = jnp.asarray(wave)
    out_j     = jnp.asarray(outwave)

    def smoother(spec, sigma_l):
        sigma_eff = jnp.sqrt(jnp.maximum(sigma_l**2 - inres**2, 0.0))
        spec_lin  = jax_interp(w_lin_j, wave_j, spec)
        spec_conv = smooth_fft(dw, spec_lin, sigma_eff)
        return jax_interp(out_j, w_lin_j, spec_conv)

    return smoother


def make_lsf_smoother(wave, sigma_lsf, outwave, pix_per_sigma=2):
    """Return a JIT-compilable LSF smoother with precomputed CDF grid.

    ``sigma_lsf`` must be fixed (typically an instrumental property).
    The returned function is JAX-differentiable with respect to ``spec``.

    Parameters
    ----------
    wave : (N,) array-like
        Input wavelength grid [Å].  Fixed.
    sigma_lsf : (N,) array-like
        LSF dispersion at each wavelength [Å].  Fixed.
    outwave : (M,) array-like
        Output wavelength grid [Å].  Fixed.
    pix_per_sigma : float, optional
        Minimum pixels per σ on the x-grid.

    Returns
    -------
    smoother : callable
        ``smoother(spec)`` → (M,) JAX array.
    """
    lam, dx, x_per_sigma = _lsf_grid(
        np.asarray(wave), np.asarray(sigma_lsf), pix_per_sigma
    )
    lam_j  = jnp.asarray(lam)
    wave_j = jnp.asarray(wave)
    out_j  = jnp.asarray(outwave)

    def smoother(spec):
        spec_x    = jax_interp(lam_j, wave_j, spec)
        spec_conv = smooth_fft(dx, spec_x, x_per_sigma)
        return jax_interp(out_j, lam_j, spec_conv)

    return smoother


# ---------------------------------------------------------------------------
# Top-level dispatcher  (sedpy-compatible API)
# ---------------------------------------------------------------------------

def smoothspec(wave, spec, resolution, outwave=None, smoothtype="vel",
               inres=0.0, **kwargs):
    """Smooth a spectrum to a target resolution.

    Drop-in replacement for ``sedpy.smoothing.smoothspec`` using JAX FFTs.

    Parameters
    ----------
    wave : (N,) array-like
        Input wavelength grid [Å].  Must be concrete (not a traced value).
    spec : (N,) JAX array
        Input flux.
    resolution : float or scalar JAX array
        Smoothing parameter.  Units depend on ``smoothtype``:
        - ``"vel"``    — velocity dispersion σ_v [km/s]
        - ``"R"``      — spectral resolution R = λ/σ_λ (= c/σ_v); converted
                         internally to km/s
        - ``"lambda"`` — wavelength dispersion σ_λ [Å]
        - ``"lsf"``    — wavelength-dependent LSF; ``resolution`` must be
                         an (N,) array of σ(λ) [Å]
    outwave : (M,) array-like or None, optional
        Output wavelength grid [Å].  If ``None``, ``wave`` is used.
    smoothtype : {"vel", "R", "lambda", "lsf"}, optional
        Type of smoothing kernel.
    inres : float, optional
        Resolution of the input spectrum in the same units as ``resolution``
        (i.e. km/s for ``"vel"`` and ``"R"``, Å for ``"lambda"``).
        Subtracted in quadrature from ``resolution`` before smoothing.

    Returns
    -------
    (M,) JAX array
        Smoothed flux on ``outwave``.
    """
    if outwave is None:
        outwave = wave

    if smoothtype == "vel":
        return smooth_vel_fft(wave, spec, resolution, outwave, inres=inres)

    elif smoothtype == "R":
        # R = λ/σ_λ = c/σ_v  ⟹  σ_v = c/R [km/s]
        sigma_v  = CKMS / resolution
        inres_v  = CKMS / inres if inres > 0 else 0.0
        return smooth_vel_fft(wave, spec, sigma_v, outwave, inres=inres_v)

    elif smoothtype == "lambda":
        return smooth_wave_fft(wave, spec, resolution, outwave, inres=inres)

    elif smoothtype == "lsf":
        return smooth_lsf_fft(wave, spec, resolution, outwave,
                               pix_per_sigma=kwargs.get("pix_per_sigma", 2))

    else:
        raise ValueError(
            f"smoothtype '{smoothtype}' is not recognised.  "
            "Choose from: 'vel', 'R', 'lambda', 'lsf'."
        )
