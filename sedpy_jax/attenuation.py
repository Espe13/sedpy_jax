# -*- coding: utf-8 -*-

"""attenuation.py  - A collection of simple functions implementing various
proposed attenuation curves.
"""

import jax.numpy as jnp
import warnings, sys

# --------------------
# ATTENUATION CURVES
# --------------------

__all__ = ["calzetti", "chevallard", "conroy", "noll",
           "powerlaw", "drude", "broken_powerlaw",
           "cardelli", "smc", "lmc"]

def powerlaw(wave, tau_v=1.0, alpha=1.0, **kwargs):
    """Simple power-law attenuation, normalized to 5500 Å.

    :param wave:
        The wavelengths at which optical depth estimates are desired.
    :param tau_v: (default: 1)
        The optical depth at 5500 Å, used to normalize the
        attenuation curve.
    :returns tau:
        The optical depth at each wavelength.
    """
    return tau_v * (wave / 5500.0) ** (-alpha)



def calzetti(wave, tau_v=1.0, R_v=4.05, **kwargs):
    """Calzetti et al. 2000 starburst attenuation curve with FUV and NIR extrapolations.

    :param wave: array of wavelengths in Angstrom
    :param tau_v: optical depth at 5500 Å
    :param R_v: ratio of total to selective extinction
    :returns: optical depth at each wavelength
    """
    # Define k(lambda) functions
    def k1(x): return 2.659 * (-1.857 + 1.040 * x)
    def k2(x): return 2.659 * (-2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3)

    # Precompute reference slopes and normalization
    uv = jnp.array([0.12, 0.13]) * 1e4
    kuv = k2(1e4 / uv) + R_v
    uv_slope = (kuv[1] - kuv[0]) / (uv[1] - uv[0])

    ir = jnp.array([2.19, 2.20]) * 1e4
    kir = k1(1e4 / ir) + R_v
    ir_slope = (kir[1] - kir[0]) / (ir[1] - ir[0])

    k_v = k2(1e4 / 5500.0) + R_v

    x = 1e4 / wave
    ktot = jnp.zeros_like(wave)

    # Define masks
    uinds = (wave >= 1200.0) & (wave < 6300.0)
    oinds = (wave >= 6300.0) & (wave <= 22000.0)
    xinds = wave < 1200.0
    iinds = wave > 22000.0

    # Construct ktot piecewise
    ktot = jnp.where(oinds, k1(x) + R_v, ktot)
    ktot = jnp.where(uinds, k2(x) + R_v, ktot)
    ktot = jnp.where(xinds, kuv[0] + (wave - uv[0]) * uv_slope, ktot)
    ktot = jnp.where(iinds, kir[1] + (wave - ir[1]) * ir_slope, ktot)

    # Ensure no negative attenuation
    ktot = jnp.maximum(ktot, 0.0)

    return tau_v * (ktot / k_v)


def drude(x, x0=4.59, gamma=0.90, **extras):
    """Drude profile for the 2175 Å bump.

    :param x:
        Inverse wavelength (inverse microns) at which values for the drude
        profile are requested.

    :param gamma:
        Width of the Drude profile (inverse microns).

    :param x0:
        Center of the Drude profile (inverse microns).

    :returns k_lambda:
        The value of the Drude profile at x, normalized such that the peak is 1.
    """
    return (x * gamma) ** 2 / ((x ** 2 - x0 ** 2) ** 2 + (x * gamma) ** 2)


def noll(wave, tau_v=1.0, delta=0.0, c_r=0.0, Ebump=0.0, **kwargs):
    """Noll 2009 attenuation curve (Calzetti + Drude bump + power-law tilt).

    :param wave: array of wavelengths in Ångström
    :param tau_v: V-band optical depth
    :param Ebump: bump strength (normalized Drude at 2175 Å)
    :param delta: slope of power-law modifying Calzetti
    :param c_r: constant modifying effective R_v (e.g., c_r = -delta)
    :returns: optical depth τ(λ)
    """
    # Calzetti baseline (normalized to τ_V = 1)
    kcalz = calzetti(wave, tau_v=1.0, R_v=4.05) - 1.0
    
    # Drude bump at 2175 Å, input in inverse μm
    x = 1e4 / wave  # inverse microns
    bump = Ebump / 4.05 * drude(x, **kwargs)
    
    # Modified attenuation curve
    k = kcalz + bump
    a = (k * (1.0 - 1.12 * c_r) + 1.0) * (wave / 5500.0) ** delta
    return a * tau_v



def chevallard(wave, tau_v=1.0, **kwargs):
    """Chevallard et al. (2013) attenuation curve (disk RT-inspired, no UV bump).

    :param wave: Wavelengths in Ångström at which to evaluate optical depth.
    :param tau_v: Optical depth at 5500 Å.
    :returns: Optical depth at each wavelength.
    """
    alpha_v = 2.8 / (1.0 + jnp.sqrt(tau_v))  # Slope term
    bb = 0.3 - 0.05 * tau_v  # Slope modifier
    alpha = alpha_v + bb * (wave * 1e-4 - 0.55)  # Total slope
    tau_lambda = tau_v * (wave / 5500.0) ** (-alpha)
    return tau_lambda



def conroy(wave, tau_v=1.0, R_v=3.1, f_bump=0.6, **kwargs):
    """Conroy & Schiminovich (2010) attenuation curve, including a reduced UV bump.

    :param wave: Wavelengths in Ångström
    :param tau_v: V-band optical depth
    :param R_v: Total-to-selective extinction ratio
    :param f_bump: UV bump strength relative to Cardelli
    :returns: Optical depth τ(λ)
    """
    x = 1e4 / wave  # inverse micron
    a = jnp.zeros_like(x)
    b = jnp.zeros_like(x)

    # IR (0.3 <= x < 1.1)
    ir = (x >= 0.3) & (x < 1.1)
    a_ir = 0.574 * x**1.61
    b_ir = -0.527 * x**1.61
    a = jnp.where(ir, a_ir, a)
    b = jnp.where(ir, b_ir, b)

    # Optical (1.1 <= x < 3.3)
    opt = (x >= 1.1) & (x < 3.3)
    y = x - 1.82
    y_opt = y
    a_opt = (1. + 0.177 * y_opt - 0.504 * y_opt**2 - 0.0243 * y_opt**3 +
             0.721 * y_opt**4 + 0.0198 * y_opt**5 - 0.7750 * y_opt**6 +
             0.330 * y_opt**7)
    b_opt = (1.413 * y_opt + 2.283 * y_opt**2 + 1.072 * y_opt**3 -
             5.384 * y_opt**4 - 0.622 * y_opt**5 + 5.303 * y_opt**6 -
             2.090 * y_opt**7)
    a = jnp.where(opt, a_opt, a)
    b = jnp.where(opt, b_opt, b)

    # NUV (3.3 <= x < 5.9)
    nuv = (x >= 3.3) & (x < 5.9)
    tmp = (-0.0370 + 0.0469 * f_bump - 0.601 * f_bump / R_v + 0.542 / R_v)
    fa_nuv = (3.3 / x)**6 * tmp
    tmp1 = 0.104 * f_bump / ((x - 4.67)**2 + 0.341)
    a_nuv = 1.752 - 0.316 * x - tmp1 + fa_nuv
    tmp2 = 1.206 * f_bump / ((x - 4.62)**2 + 0.263)
    b_nuv = -3.09 + 1.825 * x + tmp2
    a = jnp.where(nuv, a_nuv, a)
    b = jnp.where(nuv, b_nuv, b)

    # FUV (5.9 <= x < 8.0)
    fuv = (x >= 5.9) & (x < 8.0)
    dx = x - 5.9
    fa_fuv = -0.0447 * dx**2 - 0.00978 * dx**3
    fb_fuv = 0.213 * dx**2 + 0.121 * dx**3
    tmp1 = 0.104 * f_bump / ((x - 4.67)**2 + 0.341)
    tmp2 = 1.206 * f_bump / ((x - 4.62)**2 + 0.263)
    a_fuv = 1.752 - 0.316 * x - tmp1 + fa_fuv
    b_fuv = -3.09 + 1.825 * x + tmp2 + fb_fuv
    a = jnp.where(fuv, a_fuv, a)
    b = jnp.where(fuv, b_fuv, b)

    # Main attenuation law
    alam = a + b / R_v

    # XUV (x >= 8.0)
    xuv = x >= 8.0
    x8 = 8.0
    dx8 = x8 - 5.9
    fa = -0.0447 * dx8**2 - 0.00978 * dx8**3
    fb = 0.213 * dx8**2 + 0.121 * dx8**3
    tmp1 = 0.104 * f_bump / ((x8 - 4.67)**2 + 0.341)
    tmp2 = 1.206 * f_bump / ((x8 - 4.62)**2 + 0.263)
    af = 1.752 - 0.316 * x8 - tmp1 + fa
    bf = -3.09 + 1.825 * x8 + tmp2 + fb
    a8 = af + bf / R_v
    alam = jnp.where(xuv, (x8 / x)**(-1.3) * a8, alam)

    return tau_v * alam




def cardelli(wave, tau_v=1.0, R_v=3.1, **kwargs):
    """Cardelli, Clayton, & Mathis (1989) extinction curve with O'Donnell (1994) UV update.

    :param wave: Wavelengths in Ångström
    :param tau_v: Optical depth at 5500 Å
    :param R_v: Total-to-selective extinction ratio
    :returns: Optical depth τ(λ)
    """
    mic = wave * 1e-4  # Convert to microns
    x = 1.0 / mic
    a = jnp.zeros_like(x)
    b = jnp.zeros_like(x)

    # Define wavelength ranges
    w1 = (x >= 1.1) & (x <= 3.3)   # Optical
    w2 = (x >= 0.3) & (x < 1.1)    # NIR
    w3 = (x > 3.3) & (x <= 8.0)    # UV
    w4 = (x > 8.0) & (x <= 10.0)   # FUV
    wsh = x > 10.0                 # Too short
    wlg = x < 0.3                  # Too long

    # Optical: 1.1 <= x <= 3.3
    y = x - 1.82
    a_w1 = (1. + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3 +
            0.72085 * y**4 + 0.01979 * y**5 - 0.77530 * y**6 +
            0.32999 * y**7)
    b_w1 = (1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3 -
            5.38434 * y**4 - 0.62251 * y**5 + 5.30260 * y**6 -
            2.09002 * y**7)
    a = jnp.where(w1, a_w1, a)
    b = jnp.where(w1, b_w1, b)

    # NIR: x < 1.1
    y = x**1.61
    a_w2 = 0.574 * y
    b_w2 = -0.527 * y
    a = jnp.where(w2, a_w2, a)
    b = jnp.where(w2, b_w2, b)

    # UV: 3.3 < x <= 8.0
    fa = jnp.zeros_like(x)
    fb = jnp.zeros_like(x)
    ou = (x > 5.9) & w3
    y_ou = x - 5.9
    fa_ou = -0.04473 * y_ou**2 - 0.009779 * y_ou**3
    fb_ou = 0.2130 * y_ou**2 + 0.1207 * y_ou**3
    fa = jnp.where(ou, fa_ou, 0.0)
    fb = jnp.where(ou, fb_ou, 0.0)
    a_w3 = 1.752 - 0.316 * x - 0.104 / ((x - 4.67)**2 + 0.341) + fa
    b_w3 = -3.090 + 1.825 * x + 1.206 / ((x - 4.62)**2 + 0.263) + fb
    a = jnp.where(w3, a_w3, a)
    b = jnp.where(w3, b_w3, b)

    # FUV: 8.0 < x <= 10.0
    y = x - 8.0
    a_w4 = -1.073 - 0.628 * y + 0.137 * y**2 - 0.070 * y**3
    b_w4 = 13.670 + 4.257 * y - 0.420 * y**2 + 0.374 * y**3
    a = jnp.where(w4, a_w4, a)
    b = jnp.where(w4, b_w4, b)

    # Final optical depth curve
    tau = a + b / R_v
    tau = jnp.where(wsh | wlg, 0.0, tau)  # Zero out outside valid range

    return tau_v * tau


def smc(wave, tau_v=1.0, **kwargs):
    """Pei (1992) SMC extinction curve.

    :param wave: Wavelengths in Ångström
    :param tau_v: Optical depth at 5500 Å
    :returns: Optical depth τ(λ)
    """
    mic = wave * 1e-4  # Convert to microns
    aa = jnp.array([185., 27., 0.005, 0.010, 0.012, 0.030])
    ll = jnp.array([0.042, 0.08, 0.22, 9.7, 18., 25.])
    bb = jnp.array([90., 5.50, -1.95, -1.95, -1.80, 0.00])
    nn = jnp.array([2.0, 4.0, 2.0, 2.0, 2.0, 2.0])

    abs_ab = jnp.zeros_like(mic)
    norm_v = 0.0
    mic_5500 = 5500.0 * 1e-4  # Microns at 5500Å

    for i in range(len(aa)):
        norm_term = aa[i] / ((mic_5500 / ll[i])**nn[i] + (ll[i] / mic_5500)**nn[i] + bb[i])
        abs_term = aa[i] / ((mic / ll[i])**nn[i] + (ll[i] / mic)**nn[i] + bb[i])
        norm_v += norm_term
        abs_ab += abs_term

    return tau_v * (abs_ab / norm_v)


def lmc(wave, tau_v=1.0, **kwargs):
    """Pei (1992) LMC extinction curve.

    :param wave: Wavelengths in Ångström
    :param tau_v: Optical depth at 5500 Å
    :returns: Optical depth τ(λ)
    """
    mic = wave * 1e-4  # Convert to microns
    aa = jnp.array([175., 19., 0.023, 0.005, 0.006, 0.020])
    ll = jnp.array([0.046, 0.08, 0.22, 9.7, 18., 25.])
    bb = jnp.array([90., 5.50, -1.95, -1.95, -1.80, 0.00])
    nn = jnp.array([2.0, 4.5, 2.0, 2.0, 2.0, 2.0])

    abs_ab = jnp.zeros_like(mic)
    norm_v = 0.0
    mic_5500 = 5500.0 * 1e-4  # 5500 Å in microns

    for i in range(len(aa)):
        norm_term = aa[i] / ((mic_5500 / ll[i])**nn[i] + (ll[i] / mic_5500)**nn[i] + bb[i])
        abs_term = aa[i] / ((mic / ll[i])**nn[i] + (ll[i] / mic)**nn[i] + bb[i])
        norm_v += norm_term
        abs_ab += abs_term

    return tau_v * (abs_ab / norm_v)
