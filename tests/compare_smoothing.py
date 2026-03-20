"""compare_smoothing.py — sedpy vs sedpy_jax side-by-side comparison figure.

Produces a single figure with four panels:
  (a) velocity smoothing:  sedpy vs sedpy_jax, residuals
  (b) wavelength smoothing: sedpy vs sedpy_jax, residuals
  (c) wavelength-dependent LSF: sedpy vs sedpy_jax, residuals

Run from the phmc environment:
    python compare_smoothing.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Imports — both packages must be on the active environment's path
# ---------------------------------------------------------------------------

import sedpy.smoothing as sedpy_ref
from sedpy_jax.smoothing import (
    smooth_vel_fft, smooth_wave_fft, smooth_lsf_fft,
    make_vel_smoother, make_wave_smoother, make_lsf_smoother,
)

# ---------------------------------------------------------------------------
# Mock spectrum: power-law continuum + six narrow emission lines
# ---------------------------------------------------------------------------

CKMS = 2.998e5  # km/s

def make_spectrum(wave, sigma_v_lines=12.0):
    """F_λ ∝ λ^{−1.5} continuum plus six optical/NIR emission lines."""
    spec = (wave / 5500.0) ** (-1.5)
    lines = [
        (3727.0, 0.8),   # [O II]
        (4861.0, 1.2),   # Hβ
        (5007.0, 2.0),   # [O III]
        (6563.0, 3.0),   # Hα
        (6717.0, 0.5),   # [S II]
        (8498.0, 0.3),   # Ca II
    ]
    for lc, lf in lines:
        sigma_aa = lc * sigma_v_lines / CKMS
        spec += lf * np.exp(-0.5 * ((wave - lc) / sigma_aa) ** 2)
    line_centres = [l[0] for l in lines]
    return spec, line_centres

# ---------------------------------------------------------------------------
# Wavelength grids
# ---------------------------------------------------------------------------

wave    = np.arange(3500.0, 10001.0, 0.5)    # high-res input [Å]
outwave = np.arange(3700.0,  9801.0, 1.5)    # output grid [Å]

spec_np, line_centres = make_spectrum(wave)
spec_j = jnp.array(spec_np)

# Smoothing parameters
sigma_v   = 150.0                                          # km/s  (velocity)
sigma_l   = 2.5                                            # Å     (wavelength)
sigma_lsf = 1.0 + 3.5 * (wave - wave[0]) / (wave[-1] - wave[0])  # 1.0→4.5 Å

# ---------------------------------------------------------------------------
# Run both implementations
# ---------------------------------------------------------------------------

# sedpy (NumPy reference)
s_vel_ref  = sedpy_ref.smooth_vel_fft( wave, spec_np, outwave, sigma_v)
s_wave_ref = sedpy_ref.smooth_wave_fft(wave, spec_np, outwave, sigma_l)
s_lsf_ref  = sedpy_ref.smooth_lsf_fft(wave, spec_np, outwave, sigma=sigma_lsf)

# sedpy_jax eager (no JIT)
s_vel_jax  = np.asarray(smooth_vel_fft( wave, spec_j, sigma_v,   outwave))
s_wave_jax = np.asarray(smooth_wave_fft(wave, spec_j, sigma_l,   outwave))
s_lsf_jax  = np.asarray(smooth_lsf_fft( wave, spec_j, sigma_lsf, outwave))

# ---------------------------------------------------------------------------
# JIT compilation check
# ---------------------------------------------------------------------------
# The factory functions capture wave/outwave as closed-over constants, so the
# returned closures are unconditionally JIT-compilable.
# The functional API requires static_argnums for wave and outwave because JAX
# must know their values at trace time (array shape depends on len(wave)).

print("\nJIT compilation check:")

import time as _time

def _jit_test(label, fn, *args):
    """Attempt jax.jit, report success/failure and speedup."""
    try:
        # If fn is already a JAX-compiled function (has .lower), don't re-wrap;
        # otherwise compile it now.
        fn_jit = fn if hasattr(fn, "lower") else jax.jit(fn)
        # First call compiles
        t0 = _time.perf_counter()
        out_jit = fn_jit(*args)
        out_jit.block_until_ready()
        t_compile = _time.perf_counter() - t0
        # Second call uses the cache
        t0 = _time.perf_counter()
        out_jit = fn_jit(*args)
        out_jit.block_until_ready()
        t_run = _time.perf_counter() - t0
        # Eager timing for comparison
        t0 = _time.perf_counter()
        out_eager = fn(*args)
        out_eager.block_until_ready()
        t_eager = _time.perf_counter() - t0
        speedup = t_eager / t_run if t_run > 0 else float("inf")
        print(f"  {label:35s}  OK   compile={t_compile*1e3:6.1f} ms  "
              f"cached={t_run*1e3:5.2f} ms  eager={t_eager*1e3:5.2f} ms  "
              f"speedup={speedup:.1f}x")
        return np.asarray(out_jit)
    except Exception as exc:
        print(f"  {label:35s}  FAIL  {type(exc).__name__}: {exc}")
        return None

# --- Factory closures (recommended path, always JIT-able) ---
vel_smoother  = make_vel_smoother( wave, outwave)
wave_smoother = make_wave_smoother(wave, outwave)
lsf_smoother  = make_lsf_smoother( wave, sigma_lsf, outwave)

s_vel_jit  = _jit_test("make_vel_smoother  (factory)",
                        vel_smoother,  spec_j, jnp.array(sigma_v))
s_wave_jit = _jit_test("make_wave_smoother (factory)",
                        wave_smoother, spec_j, jnp.array(sigma_l))
s_lsf_jit  = _jit_test("make_lsf_smoother  (factory)",
                        lsf_smoother,  spec_j)

# --- Functional API with static_argnums ---
# wave is arg 0, outwave is arg 3; mark them static so JAX treats them as
# compile-time constants rather than traced values.
vel_jit_fn  = jax.jit(smooth_vel_fft,  static_argnums=(0, 3))
wave_jit_fn = jax.jit(smooth_wave_fft, static_argnums=(0, 3))
lsf_jit_fn  = jax.jit(smooth_lsf_fft,  static_argnums=(0, 2, 3))

_jit_test("smooth_vel_fft  (static_argnums)",
          vel_jit_fn,  tuple(wave), spec_j, jnp.array(sigma_v),   tuple(outwave))
_jit_test("smooth_wave_fft (static_argnums)",
          wave_jit_fn, tuple(wave), spec_j, jnp.array(sigma_l),   tuple(outwave))
_jit_test("smooth_lsf_fft  (static_argnums)",
          lsf_jit_fn,  tuple(wave), spec_j, tuple(sigma_lsf),     tuple(outwave))

# Verify JIT outputs match eager outputs (factory path)
print("\nJIT vs eager consistency (factory closures):")
def _check(label, jit_out, eager_out):
    if jit_out is None:
        print(f"  {label:30s}  skipped (JIT failed)")
        return
    diff = np.max(np.abs(jit_out - eager_out))
    print(f"  {label:30s}  max|JIT − eager| = {diff:.2e}")

_check("velocity",   s_vel_jit,  s_vel_jax)
_check("wavelength", s_wave_jit, s_wave_jax)
_check("LSF",        s_lsf_jit,  s_lsf_jax)

# ---------------------------------------------------------------------------
# Residuals: (jax − ref) / ref × 100 %
# ---------------------------------------------------------------------------

def pct_residual(jax_arr, ref_arr, floor_frac=1e-3):
    """Signed percentage residual, masked where |ref| < floor_frac * max|ref|."""
    ref_safe = np.where(
        np.abs(ref_arr) > floor_frac * np.max(np.abs(ref_arr)),
        ref_arr, np.nan
    )
    return 100.0 * (jax_arr - ref_arr) / ref_safe

r_vel  = pct_residual(s_vel_jax,  s_vel_ref)
r_wave = pct_residual(s_wave_jax, s_wave_ref)
r_lsf  = pct_residual(s_lsf_jax,  s_lsf_ref)

# Quantify
def stats(r, label):
    finite = r[np.isfinite(r)]
    print(f"  {label:30s}  max|Δ| = {np.max(np.abs(finite)):.4f}%  "
          f"median|Δ| = {np.median(np.abs(finite)):.4f}%")

print("\nResidual statistics (sedpy_jax − sedpy) / sedpy:")
stats(r_vel,  "velocity  (σ_v = 150 km/s)")
stats(r_wave, "wavelength (σ_λ = 2.5 Å)")
stats(r_lsf,  "LSF        (σ = 1.0–4.5 Å)")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family"   : "serif",
    "font.size"     : 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top"     : True,
    "ytick.right"   : True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi"    : 180,
})

C_REF  = "#1a3a5c"   # deep navy   — sedpy reference
C_JAX  = "#e07b39"   # burnt orange — sedpy_jax
C_RES  = "#c0392b"   # deep red    — residual
C_LINE = "#bdc3c7"   # light grey  — line markers
C_LSF  = "#27ae60"   # green       — LSF curve

fig = plt.figure(figsize=(14, 9))

# 3 columns, each with a 3:1 spectrum:residual split
outer = gridspec.GridSpec(1, 3, wspace=0.32, figure=fig)
inner = [
    gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i],
                                     height_ratios=[3, 1], hspace=0.06)
    for i in range(3)
]

axes = [[fig.add_subplot(inner[col][row]) for row in range(2)]
        for col in range(3)]

# ---- helper to annotate lines ------------------------------------------
def mark_lines(ax):
    for lc in line_centres:
        if outwave[0] < lc < outwave[-1]:
            ax.axvline(lc, color=C_LINE, lw=0.6, ls=":", zorder=0)

# ---- panel (a): velocity -----------------------------------------------
ax_s, ax_r = axes[0]

ax_s.plot(outwave, s_vel_ref, color=C_REF, lw=1.4, label="sedpy (NumPy)")
ax_s.plot(outwave, s_vel_jax, color=C_JAX, lw=0.9, ls="--", label="sedpy_jax (JAX)")
mark_lines(ax_s)
ax_s.set_ylabel(r"Flux [arb.]")
ax_s.set_title(rf"(a) Velocity:  $\sigma_v = {sigma_v:.0f}$ km/s", loc="left")
ax_s.legend()
ax_s.set_xticklabels([])

ax_r.plot(outwave, r_vel, color=C_RES, lw=0.8)
ax_r.axhline(0, color="k", lw=0.6, ls="--")
ax_r.set_xlabel(r"Wavelength [$\AA$]")
ax_r.set_ylabel(r"$\Delta$ [%]")
ax_r.set_ylim(-0.5, 0.5)
mark_lines(ax_r)

# ---- panel (b): wavelength ---------------------------------------------
ax_s, ax_r = axes[1]

ax_s.plot(outwave, s_wave_ref, color=C_REF, lw=1.4, label="sedpy (NumPy)")
ax_s.plot(outwave, s_wave_jax, color=C_JAX, lw=0.9, ls="--", label="sedpy_jax (JAX)")
mark_lines(ax_s)
ax_s.set_ylabel(r"Flux [arb.]")
ax_s.set_title(rf"(b) Wavelength:  $\sigma_\lambda = {sigma_l:.1f}$ Å", loc="left")
ax_s.legend()
ax_s.set_xticklabels([])

ax_r.plot(outwave, r_wave, color=C_RES, lw=0.8)
ax_r.axhline(0, color="k", lw=0.6, ls="--")
ax_r.set_xlabel(r"Wavelength [$\AA$]")
ax_r.set_ylabel(r"$\Delta$ [%]")
ax_r.set_ylim(-0.5, 0.5)
mark_lines(ax_r)

# ---- panel (c): LSF ----------------------------------------------------
ax_s, ax_r = axes[2]

ax_s.plot(outwave, s_lsf_ref, color=C_REF, lw=1.4, label="sedpy (NumPy)")
ax_s.plot(outwave, s_lsf_jax, color=C_JAX, lw=0.9, ls="--", label="sedpy_jax (JAX)")
mark_lines(ax_s)

# Overlay the LSF profile on a twin axis
ax_lsf = ax_s.twinx()
ax_lsf.plot(outwave, np.interp(outwave, wave, sigma_lsf),
            color=C_LSF, lw=0.8, ls=":", alpha=0.7)
ax_lsf.set_ylabel(r"$\sigma_\mathrm{LSF}$ [$\AA$]", color=C_LSF, fontsize=8)
ax_lsf.tick_params(axis="y", colors=C_LSF, labelsize=7)
ax_lsf.set_ylim(0, 12)

ax_s.set_ylabel(r"Flux [arb.]")
ax_s.set_title(r"(c) LSF:  $\sigma(\lambda)=1.0$–$4.5$ Å", loc="left")
ax_s.legend()
ax_s.set_xticklabels([])

ax_r.plot(outwave, r_lsf, color=C_RES, lw=0.8)
ax_r.axhline(0, color="k", lw=0.6, ls="--")
ax_r.set_xlabel(r"Wavelength [$\AA$]")
ax_r.set_ylabel(r"$\Delta$ [%]")
ax_r.set_ylim(-2.0, 2.0)
mark_lines(ax_r)

# ---- shared formatting -------------------------------------------------
for col in range(3):
    for row in range(2):
        axes[col][row].set_xlim(outwave[0], outwave[-1])

fig.suptitle("sedpy vs sedpy_jax — spectral smoothing comparison",
             fontsize=11, y=1.01)

outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "smoothing_comparison.pdf")
fig.savefig(outfile, bbox_inches="tight")
print(f"\nFigure saved: {outfile}")
