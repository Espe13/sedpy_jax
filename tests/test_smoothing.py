"""test_smoothing.py — Validation of sedpy_jax.smoothing against sedpy.

Primary validation metric: sedpy_jax must reproduce sedpy (NumPy reference)
to within floating-point agreement.  Tests 2–4 REQUIRE sedpy to be present
in the active environment and will be skipped otherwise.

Direct Gaussian quadrature is available as slow secondary context; set
COMPARE_DIRECT = True at the top of this file to enable it.

Tests
-----
Test 1 — smooth_fft kernel width
    Convolve a unit-spacing Gaussian; verify σ_out = √(σ_in² + σ_kernel²).

Test 2 — smooth_vel_fft vs sedpy
    Velocity smoothing on a mock optical spectrum.
    Pass criterion: max|sedpy_jax − sedpy| / sedpy < 0.02 %.

Test 3 — smooth_wave_fft vs sedpy
    Wavelength smoothing.  Same pass criterion.

Test 4 — smooth_lsf_fft vs sedpy
    Wavelength-dependent LSF (linearly increasing with λ).
    Pass criterion: max|Δ| < 0.5 % (CDF-transform FFT — same algorithm in
    both implementations; any deviation is due to float64 ordering only).

Test 5 — gradient through σ_v
    jax.grad w.r.t. sigma_v agrees with central finite difference to < 0.5 %.

Test 6 — input resolution subtraction
    σ_total = √(σ1² + σ2²) consistency: two-step vs one-step < 0.5 %.

Usage
-----
    python test_smoothing.py
"""

import sys
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Set True to also run slow O(N_out × N_in) Gaussian quadrature as extra
# context.  Not needed for validation — kept only for diagnostic purposes.
COMPARE_DIRECT = False

# Number of repeated calls used for timing benchmarks in Tests 2–4.
N_BENCH = 1000

# ---------------------------------------------------------------------------
# sedpy_jax import
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))

from sedpy_jax.smoothing import (
    smooth_fft, smooth_vel_fft, smooth_wave_fft, smooth_lsf_fft,
    make_vel_smoother, make_wave_smoother, make_lsf_smoother,
    smoothspec, CKMS, SIGMA_TO_FWHM,
)

# ---------------------------------------------------------------------------
# sedpy import (required for Tests 2–4)
# ---------------------------------------------------------------------------
_HAS_SEDPY = False
try:
    import sedpy.smoothing as sedpy_ref
    _HAS_SEDPY = True
    print("sedpy found — primary comparison target available.")
except ImportError:
    print("sedpy not found — Tests 2–4 will be SKIPPED "
          "(sedpy is required for the primary validation).")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIG_DIR = os.path.join(_here, "test_smoothing_figs")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family"           : "serif",
    "font.size"             : 11,
    "axes.labelsize"        : 12,
    "axes.titlesize"        : 12,
    "legend.fontsize"       : 10,
    "xtick.direction"       : "in",
    "ytick.direction"       : "in",
    "xtick.top"             : True,
    "ytick.right"           : True,
    "xtick.minor.visible"   : True,
    "ytick.minor.visible"   : True,
    "figure.dpi"            : 150,
})

C_SEDPY = "#1a3a5c"   # deep navy    — sedpy (reference)
C_JAX   = "#e07b39"   # burnt orange — sedpy_jax
C_RESID = "#c0392b"   # deep red     — residual
C_LSF   = "#27ae60"   # green        — LSF profile
C_LINE  = "#bdc3c7"   # light grey   — emission line markers

# ---------------------------------------------------------------------------
# Mock spectrum
# ---------------------------------------------------------------------------

def make_mock_spectrum(wave, sigma_v_lines=30.0):
    """Power-law continuum plus six Gaussian emission lines."""
    spec = (wave / 5500.0) ** (-1.5)
    line_centres = [3727.0, 4861.0, 5007.0, 6563.0, 6717.0, 8498.0]
    line_fluxes  = [0.8,    1.2,    2.0,    3.0,    0.5,    0.3  ]
    sigma_lam    = sigma_v_lines / CKMS
    for lc, lf in zip(line_centres, line_fluxes):
        spec += lf * np.exp(-0.5 * ((wave - lc) / (lc * sigma_lam)) ** 2)
    return spec, line_centres

# ---------------------------------------------------------------------------
# Direct Gaussian quadrature (slow O(N²) reference)
# Only called when COMPARE_DIRECT = True.
# ---------------------------------------------------------------------------

def smooth_vel_direct(wave, spec, sigma_v, outwave, inres=0.0, nsigma=12):
    sigma_eff_sq = sigma_v**2 - inres**2
    if sigma_eff_sq <= 0.0:
        return np.interp(outwave, wave, spec)
    sigma_eff = np.sqrt(sigma_eff_sq) / CKMS
    ln_wave   = np.log(wave)
    flux      = np.zeros(len(outwave))
    for i, w in enumerate(outwave):
        x = (ln_wave - np.log(w)) / sigma_eff
        mask = np.abs(x) < nsigma
        xi, si = x[mask], spec[mask]
        k = np.exp(-0.5 * xi**2)
        n = np.trapezoid(k, xi)
        flux[i] = np.trapezoid(k * si, xi) / n if n > 0 else 0.0
    return flux


def smooth_wave_direct(wave, spec, sigma_l, outwave, inres=0.0, nsigma=12):
    sigma_eff_sq = sigma_l**2 - inres**2
    if sigma_eff_sq <= 0.0:
        return np.interp(outwave, wave, spec)
    sigma_eff = np.sqrt(sigma_eff_sq)
    flux      = np.zeros(len(outwave))
    for i, w in enumerate(outwave):
        x = (wave - w) / sigma_eff
        mask = np.abs(x) < nsigma
        xi, si = x[mask], spec[mask]
        k = np.exp(-0.5 * xi**2)
        n = np.trapezoid(k, xi)
        flux[i] = np.trapezoid(k * si, xi) / n if n > 0 else 0.0
    return flux


def smooth_lsf_direct(wave, spec, sigma_lsf_fn, outwave, nsigma=10):
    flux = np.zeros(len(outwave))
    for i, w in enumerate(outwave):
        sigma_w = sigma_lsf_fn(w)
        x       = (wave - w) / sigma_w
        mask    = np.abs(x) < nsigma
        xi, si  = x[mask], spec[mask]
        k  = np.exp(-0.5 * xi**2)
        n  = np.trapezoid(k, xi)
        flux[i] = np.trapezoid(k * si, xi) / n if n > 0 else 0.0
    return flux

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def residual_pct(arr, ref, floor_frac=1e-3):
    """Signed percentage residual; NaN where |ref| < floor_frac × max|ref|."""
    ref_safe = np.where(
        np.abs(ref) > floor_frac * np.max(np.abs(ref)), ref, np.nan
    )
    return 100.0 * (np.asarray(arr) - ref) / ref_safe


def _time_calls(fn, args, N=N_BENCH, jax_fn=False):
    """
    Return mean wall-clock time per call in milliseconds over N iterations.

    For JAX functions (jax_fn=True) one warm-up call is made first to ensure
    compilation is complete, then block_until_ready() is called after each
    iteration to include device synchronisation.
    """
    if jax_fn:
        out = fn(*args)
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        t0 = time.perf_counter()
        for _ in range(N):
            out = fn(*args)
            out.block_until_ready()
    else:
        t0 = time.perf_counter()
        for _ in range(N):
            fn(*args)
    return (time.perf_counter() - t0) / N * 1e3   # ms per call


def mark_lines(ax, line_centres, outwave):
    for lc in line_centres:
        if outwave[0] < lc < outwave[-1]:
            ax.axvline(lc, color=C_LINE, lw=0.5, ls=":", zorder=0, alpha=0.8)


def _make_sedpy_wave_panel(fig, gs_row, outwave, s_ref, s_jax, resid, lines,
                           title, ylabel_resid, ylim_resid):
    """Two-panel layout: spectrum (top) + residual (bottom)."""
    ax0 = fig.add_subplot(gs_row[0])
    ax1 = fig.add_subplot(gs_row[1], sharex=ax0)

    ax0.plot(outwave, s_ref, color=C_SEDPY, lw=1.5,
             label="sedpy (NumPy, reference)")
    ax0.plot(outwave, s_jax, color=C_JAX,   lw=0.9, ls="--", alpha=0.9,
             label="sedpy_jax (JAX)")
    mark_lines(ax0, lines, outwave)
    ax0.set_ylabel(r"Flux [arb.]")
    ax0.set_title(title, loc="left")
    ax0.legend()
    ax0.tick_params(labelbottom=False)
    ax0.set_xlim(outwave[0], outwave[-1])

    ax1.plot(outwave, resid, color=C_RESID, lw=0.9)
    ax1.axhline(0, color="k", lw=0.6, ls="--")
    mark_lines(ax1, lines, outwave)
    ax1.set_xlabel(r"Wavelength [$\AA$]")
    ax1.set_ylabel(ylabel_resid)
    ax1.set_ylim(*ylim_resid)
    ax1.set_xlim(outwave[0], outwave[-1])

    return ax0, ax1


# ===========================================================================
# Test 1 — smooth_fft kernel correctness
# ===========================================================================

def test_fft_kernel():
    """
    Convolve a known Gaussian with smooth_fft; verify σ_out = √(σ_in² + σ_k²).
    No comparison to sedpy needed — tests the raw FFT convolution primitive.
    """
    print("\n=== Test 1: smooth_fft kernel correctness ===")

    N            = 2048
    dx           = 1.0
    # np.arange guarantees spacing exactly dx=1.0; np.linspace(-500,500,2048)
    # would give step 1000/2047 ≈ 0.4885, causing a ~33% width error.
    x            = np.arange(-N // 2, N // 2, dtype=np.float64)
    sigma_in     = 50.0
    sigma_kernel = 80.0
    sigma_theory = np.sqrt(sigma_in**2 + sigma_kernel**2)

    spec     = np.exp(-0.5 * (x / sigma_in) ** 2)
    spec_out = np.asarray(smooth_fft(dx, jnp.array(spec), sigma_kernel))

    # Measure σ by interpolating to the e^{-1/2} level
    peak   = spec_out.max()
    half_e = peak * np.exp(-0.5)
    icen   = np.argmax(spec_out)
    right  = np.interp(half_e, spec_out[icen:][::-1], x[icen:][::-1])
    left   = np.interp(half_e, spec_out[:icen+1], x[:icen+1])
    sigma_meas = (right - left) / 2.0

    err    = abs(sigma_meas - sigma_theory) / sigma_theory * 100.0
    status = "PASS" if err < 0.5 else "FAIL"
    print(f"  σ_theory  = {sigma_theory:.4f}")
    print(f"  σ_measured = {sigma_meas:.4f}")
    print(f"  Error = {err:.4f}%  [{status}]")

    # --- Figure ---
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(8, 5),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    theory_out = peak * np.exp(-0.5 * (x / sigma_theory) ** 2)

    ax0.plot(x, spec,      color=C_SEDPY, lw=1.5,
             label=rf"Input ($\sigma_\mathrm{{in}}={sigma_in}$)")
    ax0.plot(x, spec_out,  color=C_JAX,   lw=1.5,
             label="Smoothed (JAX FFT)")
    ax0.plot(x, theory_out, color=C_RESID, lw=1.2, ls="--",
             label=rf"Analytical $\sigma={sigma_theory:.2f}$")
    ax0.set_ylabel("Flux")
    ax0.legend()
    ax0.set_title(
        rf"Test 1 — FFT kernel: $\sigma_\mathrm{{in}}={sigma_in}$, "
        rf"$\sigma_k={sigma_kernel}$,  error = {err:.4f}%  [{status}]"
    )

    resid = residual_pct(spec_out, theory_out)
    ax1.plot(x, resid, color=C_RESID, lw=1.0)
    ax1.axhline(0, color="k", lw=0.7, ls="--")
    ax1.set_xlabel("Grid coordinate")
    ax1.set_ylabel("Residual (%)")
    ax1.set_xlim(-400, 400)
    ax1.set_ylim(-2, 2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "test1_fft_kernel.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: test1_fft_kernel.pdf")
    return status == "PASS"


# ===========================================================================
# Test 2 — smooth_vel_fft: sedpy_jax vs sedpy
# ===========================================================================

def test_vel_fft():
    """
    Velocity smoothing: compare sedpy_jax to the sedpy NumPy reference.

    Both implementations use the identical FFT algorithm on a log-λ grid;
    differences are purely floating-point rounding (expected < 0.02 %).
    """
    print("\n=== Test 2: smooth_vel_fft  —  sedpy_jax vs sedpy ===")
    if not _HAS_SEDPY:
        print("  SKIP — sedpy not available.")
        return None

    wave    = np.arange(3500.0, 10001.0, 0.5)
    spec_np, lines = make_mock_spectrum(wave, sigma_v_lines=15.0)
    spec_j  = jnp.array(spec_np)
    outwave = np.arange(3600.0, 9801.0, 1.0)
    sigma_v = 150.0   # km/s

    s_ref = sedpy_ref.smooth_vel_fft(wave, spec_np, outwave, sigma_v)
    s_jax = np.asarray(smooth_vel_fft(wave, spec_j, sigma_v, outwave))

    resid  = residual_pct(s_jax, s_ref)
    rmax   = np.nanmax(np.abs(resid))
    rmed   = np.nanmedian(np.abs(resid))
    status = "PASS" if rmax < 0.02 else "FAIL"

    print(f"  σ_v = {sigma_v} km/s")
    print(f"  sedpy_jax vs sedpy:  max|Δ| = {rmax:.5f}%   "
          f"median|Δ| = {rmed:.5f}%   [{status}]")

    # --- Timing benchmark (N_BENCH calls each) ---
    vel_smoother_jit = jax.jit(make_vel_smoother(wave, outwave))
    vel_smoother_jit(spec_j, jnp.array(sigma_v)).block_until_ready()   # compile

    t_sedpy = _time_calls(sedpy_ref.smooth_vel_fft,
                          (wave, spec_np, outwave, sigma_v))
    t_eager = _time_calls(smooth_vel_fft,
                          (wave, spec_j, sigma_v, outwave), jax_fn=True)
    t_jit   = _time_calls(vel_smoother_jit,
                          (spec_j, jnp.array(sigma_v)),     jax_fn=True)
    speedup = t_sedpy / t_jit

    print(f"  Timing ({N_BENCH} calls each):")
    print(f"    sedpy (NumPy):           {t_sedpy:8.3f} ms/call")
    print(f"    sedpy_jax (eager):       {t_eager:8.3f} ms/call")
    print(f"    sedpy_jax (JIT, cached): {t_jit:8.3f} ms/call")
    print(f"    Speedup JIT vs sedpy:    {speedup:8.1f}×")

    if COMPARE_DIRECT:
        print("  Direct quadrature ... ", end="", flush=True)
        s_quad = smooth_vel_direct(wave, spec_np, sigma_v, outwave)
        rq = np.nanmax(np.abs(residual_pct(s_jax, s_quad)))
        print(f"max|Δ| vs direct = {rq:.4f}%")

    # --- Figure ---
    fig = plt.figure(figsize=(10, 5))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    _make_sedpy_wave_panel(
        fig, gs, outwave, s_ref, s_jax, resid, lines,
        title=(rf"Test 2 — velocity smoothing ($\sigma_v = {sigma_v}$ km/s):  "
               rf"max$|\Delta|$ = {rmax:.5f}%  [{status}]"),
        ylabel_resid=r"$(f_\mathrm{JAX} - f_\mathrm{sedpy})/f_\mathrm{sedpy}$  [%]",
        ylim_resid=(-0.05, 0.05),
    )
    fig.savefig(os.path.join(FIG_DIR, "test2_vel_fft.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: test2_vel_fft.pdf")
    return status == "PASS"


# ===========================================================================
# Test 3 — smooth_wave_fft: sedpy_jax vs sedpy
# ===========================================================================

def test_wave_fft():
    """
    Wavelength smoothing: compare sedpy_jax to sedpy NumPy reference.
    Same algorithm → differences expected < 0.02 %.
    """
    print("\n=== Test 3: smooth_wave_fft  —  sedpy_jax vs sedpy ===")
    if not _HAS_SEDPY:
        print("  SKIP — sedpy not available.")
        return None

    wave    = np.arange(3500.0, 10001.0, 0.5)
    spec_np, lines = make_mock_spectrum(wave, sigma_v_lines=15.0)
    spec_j  = jnp.array(spec_np)
    outwave = np.arange(3600.0, 9801.0, 1.0)
    sigma_l = 2.5   # Å

    s_ref = sedpy_ref.smooth_wave_fft(wave, spec_np, outwave, sigma_l)
    s_jax = np.asarray(smooth_wave_fft(wave, spec_j, sigma_l, outwave))

    resid  = residual_pct(s_jax, s_ref)
    rmax   = np.nanmax(np.abs(resid))
    rmed   = np.nanmedian(np.abs(resid))
    status = "PASS" if rmax < 0.02 else "FAIL"

    print(f"  σ_λ = {sigma_l} Å")
    print(f"  sedpy_jax vs sedpy:  max|Δ| = {rmax:.5f}%   "
          f"median|Δ| = {rmed:.5f}%   [{status}]")

    # --- Timing benchmark ---
    wave_smoother_jit = jax.jit(make_wave_smoother(wave, outwave))
    wave_smoother_jit(spec_j, jnp.array(sigma_l)).block_until_ready()   # compile

    t_sedpy = _time_calls(sedpy_ref.smooth_wave_fft,
                          (wave, spec_np, outwave, sigma_l))
    t_eager = _time_calls(smooth_wave_fft,
                          (wave, spec_j, sigma_l, outwave), jax_fn=True)
    t_jit   = _time_calls(wave_smoother_jit,
                          (spec_j, jnp.array(sigma_l)),     jax_fn=True)
    speedup = t_sedpy / t_jit

    print(f"  Timing ({N_BENCH} calls each):")
    print(f"    sedpy (NumPy):           {t_sedpy:8.3f} ms/call")
    print(f"    sedpy_jax (eager):       {t_eager:8.3f} ms/call")
    print(f"    sedpy_jax (JIT, cached): {t_jit:8.3f} ms/call")
    print(f"    Speedup JIT vs sedpy:    {speedup:8.1f}×")

    if COMPARE_DIRECT:
        print("  Direct quadrature ... ", end="", flush=True)
        s_quad = smooth_wave_direct(wave, spec_np, sigma_l, outwave)
        rq = np.nanmax(np.abs(residual_pct(s_jax, s_quad)))
        print(f"max|Δ| vs direct = {rq:.4f}%")

    # --- Figure ---
    fig = plt.figure(figsize=(10, 5))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    _make_sedpy_wave_panel(
        fig, gs, outwave, s_ref, s_jax, resid, lines,
        title=(rf"Test 3 — wavelength smoothing ($\sigma_\lambda = {sigma_l}$ Å):  "
               rf"max$|\Delta|$ = {rmax:.5f}%  [{status}]"),
        ylabel_resid=r"$(f_\mathrm{JAX} - f_\mathrm{sedpy})/f_\mathrm{sedpy}$  [%]",
        ylim_resid=(-0.05, 0.05),
    )
    fig.savefig(os.path.join(FIG_DIR, "test3_wave_fft.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: test3_wave_fft.pdf")
    return status == "PASS"


# ===========================================================================
# Test 4 — smooth_lsf_fft: sedpy_jax vs sedpy
# ===========================================================================

def test_lsf_fft():
    """
    Wavelength-dependent LSF smoothing: sedpy_jax vs sedpy NumPy reference.

    Both use the CDF-transform FFT approximation (x = ∫dλ/σ, constant-kernel
    FFT in x-space).  The CDF transform is inherently approximate w.r.t.
    exact quadrature, but sedpy_jax and sedpy should agree with each other
    to < 0.5 % (differences are only float64 arithmetic ordering).
    """
    print("\n=== Test 4: smooth_lsf_fft  —  sedpy_jax vs sedpy ===")
    if not _HAS_SEDPY:
        print("  SKIP — sedpy not available.")
        return None

    wave    = np.arange(3500.0, 10001.0, 0.5)
    spec_np, lines = make_mock_spectrum(wave, sigma_v_lines=15.0)
    spec_j  = jnp.array(spec_np)
    outwave = np.arange(3700.0, 9700.0, 2.0)

    sigma0, sigma1 = 1.0, 4.0
    sigma_lsf_fn = (lambda lam:
        sigma0 + (sigma1 - sigma0) * (lam - wave[0]) / (wave[-1] - wave[0]))
    sigma_lsf    = sigma_lsf_fn(wave)

    s_ref = sedpy_ref.smooth_lsf_fft(wave, spec_np, outwave, sigma=sigma_lsf)
    s_jax = np.asarray(smooth_lsf_fft(wave, spec_j, sigma_lsf, outwave))

    resid  = residual_pct(s_jax, s_ref)
    rmax   = np.nanmax(np.abs(resid))
    rmed   = np.nanmedian(np.abs(resid))
    # Both use the identical CDF-transform FFT approximation; agreement should
    # be tight (differences are only from float64 arithmetic ordering).
    status = "PASS" if rmax < 0.5 else "FAIL"

    print(f"  σ_LSF: {sigma0:.1f} → {sigma1:.1f} Å  (linear in λ)")
    print(f"  sedpy_jax vs sedpy:  max|Δ| = {rmax:.5f}%   "
          f"median|Δ| = {rmed:.5f}%   [{status}]")

    # --- Timing benchmark ---
    lsf_smoother_jit = jax.jit(make_lsf_smoother(wave, sigma_lsf, outwave))
    lsf_smoother_jit(spec_j).block_until_ready()   # compile

    t_sedpy = _time_calls(
        lambda: sedpy_ref.smooth_lsf_fft(wave, spec_np, outwave, sigma=sigma_lsf),
        ())
    t_eager = _time_calls(smooth_lsf_fft,
                          (wave, spec_j, sigma_lsf, outwave), jax_fn=True)
    t_jit   = _time_calls(lsf_smoother_jit, (spec_j,),         jax_fn=True)
    speedup = t_sedpy / t_jit

    print(f"  Timing ({N_BENCH} calls each):")
    print(f"    sedpy (NumPy):           {t_sedpy:8.3f} ms/call")
    print(f"    sedpy_jax (eager):       {t_eager:8.3f} ms/call")
    print(f"    sedpy_jax (JIT, cached): {t_jit:8.3f} ms/call")
    print(f"    Speedup JIT vs sedpy:    {speedup:8.1f}×")

    if COMPARE_DIRECT:
        print("  Direct quadrature ... ", end="", flush=True)
        s_quad = smooth_lsf_direct(wave, spec_np, sigma_lsf_fn, outwave)
        rq = np.nanmax(np.abs(residual_pct(s_jax, s_quad)))
        print(f"max|Δ| vs direct = {rq:.4f}%  "
              f"(CDF-FFT is an approximation to exact quadrature)")

    # --- Figure: spectrum + residual + LSF profile ---
    fig = plt.figure(figsize=(10, 7))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 0.7], hspace=0.06)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)

    ax0.plot(outwave, s_ref, color=C_SEDPY, lw=1.5,
             label="sedpy (NumPy, reference)")
    ax0.plot(outwave, s_jax, color=C_JAX,   lw=0.9, ls="--", alpha=0.9,
             label="sedpy_jax (JAX)")
    mark_lines(ax0, lines, outwave)
    ax0.set_ylabel(r"Flux [arb.]")
    ax0.set_title(
        rf"Test 4 — LSF smoothing ($\sigma={sigma0:.1f}$–${sigma1:.1f}$ Å):  "
        rf"max$|\Delta|$ = {rmax:.5f}%  [{status}]",
        loc="left"
    )
    ax0.legend()
    ax0.tick_params(labelbottom=False)

    ax1.plot(outwave, resid, color=C_RESID, lw=0.9)
    ax1.axhline(0, color="k", lw=0.6, ls="--")
    mark_lines(ax1, lines, outwave)
    ax1.set_ylabel(
        r"$(f_\mathrm{JAX} - f_\mathrm{sedpy})/f_\mathrm{sedpy}$  [%]"
    )
    ax1.set_ylim(-0.8, 0.8)
    ax1.tick_params(labelbottom=False)

    ax2.plot(wave, sigma_lsf, color=C_LSF, lw=1.2)
    ax2.set_xlabel(r"Wavelength [$\AA$]")
    ax2.set_ylabel(r"$\sigma_\mathrm{LSF}$ [$\AA$]")

    for ax in (ax0, ax1, ax2):
        ax.set_xlim(outwave[0], outwave[-1])

    fig.savefig(os.path.join(FIG_DIR, "test4_lsf_fft.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: test4_lsf_fft.pdf")
    return status == "PASS"


# ===========================================================================
# Test 5 — gradient through sigma_v
# ===========================================================================

def test_gradient():
    """
    Verify jax.grad differentiates through smooth_vel_fft w.r.t. sigma_v.
    Central finite difference must agree to < 0.5 %.
    No sedpy comparison needed — tests JAX autodiff plumbing.
    """
    print("\n=== Test 5: gradient of smooth_vel_fft w.r.t. sigma_v ===")

    wave    = np.arange(4000.0, 8001.0, 1.0)
    spec_np, _ = make_mock_spectrum(wave, sigma_v_lines=20.0)
    spec_j  = jnp.array(spec_np)
    outwave = wave.copy()
    sigma_v = 120.0

    def loss(sv):
        return jnp.sum(smooth_vel_fft(wave, spec_j, sv, outwave))

    grad_auto = float(jax.grad(loss)(jnp.array(sigma_v)))

    h = 0.5   # km/s
    grad_fd = (float(loss(jnp.array(sigma_v + h))) -
               float(loss(jnp.array(sigma_v - h)))) / (2 * h)

    rel_err = abs(grad_auto - grad_fd) / abs(grad_fd) * 100.0
    status  = "PASS" if rel_err < 0.5 else "FAIL"

    print(f"  σ_v = {sigma_v} km/s,  h = {h} km/s")
    print(f"  dI/dσ_v  (auto-diff)   = {grad_auto:.6e}")
    print(f"  dI/dσ_v  (finite-diff) = {grad_fd:.6e}")
    print(f"  Relative error = {rel_err:.4f}%  [{status}]")

    sigmas    = np.linspace(30.0, 400.0, 60)
    integrals = np.array([float(loss(jnp.array(s))) for s in sigmas])
    grads_fd  = np.gradient(integrals, sigmas)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(sigmas, integrals, color=C_SEDPY, lw=1.5)
    axes[0].axvline(sigma_v, color=C_RESID, lw=1.2, ls="--",
                    label=rf"$\sigma_v = {sigma_v}$ km/s")
    axes[0].set_xlabel(r"$\sigma_v$ [km/s]")
    axes[0].set_ylabel(r"$I(\sigma_v) = \sum F_\mathrm{smooth}$")
    axes[0].set_title("Integrated flux vs. smoothing width")
    axes[0].legend()

    axes[1].plot(sigmas, grads_fd, color=C_SEDPY, lw=1.5, label="Finite difference")
    axes[1].axvline(sigma_v, color=C_RESID, lw=1.2, ls="--")
    axes[1].axhline(grad_auto, color=C_JAX, lw=1.2, ls=":",
                    label=rf"Auto-diff at $\sigma_v={sigma_v}$")
    axes[1].set_xlabel(r"$\sigma_v$ [km/s]")
    axes[1].set_ylabel(r"$dI/d\sigma_v$")
    axes[1].set_title(rf"Gradient  [error = {rel_err:.4f}\%  {status}]")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "test5_gradient.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: test5_gradient.pdf")
    return status == "PASS"


# ===========================================================================
# Test 6 — input resolution subtraction
# ===========================================================================

def test_inres_subtraction():
    """
    Quadrature consistency: smooth(σ_total) == smooth(σ1) then smooth(σ2)
    where σ_total² = σ1² + σ2².  No sedpy comparison needed.
    """
    print("\n=== Test 6: input resolution subtraction ===")

    wave    = np.arange(3500.0, 10001.0, 0.5)
    spec_np, _ = make_mock_spectrum(wave, sigma_v_lines=15.0)
    spec_j  = jnp.array(spec_np)
    outwave = np.arange(3700.0, 9801.0, 1.0)

    sigma1      = 100.0
    sigma2      =  80.0
    sigma_total = np.sqrt(sigma1**2 + sigma2**2)

    spec_step1   = smooth_vel_fft(wave, spec_j,  sigma1, wave)
    spec_step2   = np.asarray(smooth_vel_fft(wave, spec_step1, sigma2, outwave))
    spec_onestep = np.asarray(smooth_vel_fft(wave, spec_j, sigma_total, outwave))
    spec_inres   = np.asarray(smooth_vel_fft(wave, spec_step1, sigma_total, outwave, inres=sigma1))

    r_twostep = residual_pct(spec_step2, spec_onestep)
    r_inres   = residual_pct(spec_inres,  spec_onestep)
    rmax = max(np.nanmax(np.abs(r_twostep)), np.nanmax(np.abs(r_inres)))
    status = "PASS" if rmax < 0.5 else "FAIL"

    print(f"  σ1={sigma1}, σ2={sigma2}, σ_total={sigma_total:.2f} km/s")
    print(f"  Two-step vs one-step  max|Δ| = {np.nanmax(np.abs(r_twostep)):.4f}%")
    print(f"  inres subtract        max|Δ| = {np.nanmax(np.abs(r_inres)):.4f}%")
    print(f"  [{status}]")

    fig, axes = plt.subplots(3, 1, figsize=(10, 7),
                             gridspec_kw={"height_ratios": [3, 1, 1]},
                             sharex=True)
    axes[0].plot(outwave, spec_onestep, color=C_SEDPY, lw=1.5,
                 label=rf"One-step $\sigma_\mathrm{{total}}={sigma_total:.1f}$ km/s")
    axes[0].plot(outwave, spec_step2,   color=C_JAX,   lw=1.0, ls="--",
                 label=rf"Two-step ($\sigma_1={sigma1}$, $\sigma_2={sigma2}$ km/s)")
    axes[0].plot(outwave, spec_inres,   color=C_LSF,   lw=0.8, ls=":",
                 label="inres subtraction")
    axes[0].set_ylabel(r"Flux [arb.]")
    axes[0].legend(fontsize=9)
    axes[0].set_title(rf"Test 6 — quadrature resolution subtraction  [{status}]")

    axes[1].plot(outwave, r_twostep, color=C_JAX,   lw=0.9, label="Two-step residual")
    axes[1].axhline(0, color="k", lw=0.6, ls="--")
    axes[1].set_ylabel("Residual (%)")
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].legend(fontsize=9)

    axes[2].plot(outwave, r_inres,   color=C_LSF,   lw=0.9, label="inres residual")
    axes[2].axhline(0, color="k", lw=0.6, ls="--")
    axes[2].set_xlabel(r"Wavelength [$\AA$]")
    axes[2].set_ylabel("Residual (%)")
    axes[2].set_ylim(-0.5, 0.5)
    axes[2].legend(fontsize=9)

    for ax in axes:
        ax.set_xlim(outwave[0], outwave[-1])

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "test6_inres.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: test6_inres.pdf")
    return status == "PASS"


# ===========================================================================
# Timing figure — sedpy vs sedpy_jax (eager) vs sedpy_jax (JIT × N_BENCH)
# ===========================================================================

def make_timing_figure():
    """
    Bar chart comparing wall-clock time per call (ms) for sedpy (NumPy),
    sedpy_jax eager, and sedpy_jax JIT (cached after one compile call) over
    all three smoothing modes.  N_BENCH = 1000 calls each.

    Grid: one shared input/output at moderate resolution (13 002 input pixels,
    4 068 output pixels) typical of a Prospector/Ceridwen likelihood evaluation.
    """
    print(f"\n=== Timing figure ({N_BENCH} calls per method per mode) ===")
    if not _HAS_SEDPY:
        print("  SKIP — sedpy not available.")
        return

    wave    = np.arange(3500.0, 10001.0, 0.5)
    spec_np, _ = make_mock_spectrum(wave, sigma_v_lines=15.0)
    spec_j  = jnp.array(spec_np)
    outwave = np.arange(3600.0, 9801.0, 1.5)

    sigma_v   = 150.0
    sigma_l   = 2.5
    sigma_lsf = 1.0 + 3.5 * (wave - wave[0]) / (wave[-1] - wave[0])

    # Build JIT closures and compile
    vj = jax.jit(make_vel_smoother( wave, outwave))
    wj = jax.jit(make_wave_smoother(wave, outwave))
    lj = jax.jit(make_lsf_smoother( wave, sigma_lsf, outwave))
    vj(spec_j, jnp.array(sigma_v)).block_until_ready()
    wj(spec_j, jnp.array(sigma_l)).block_until_ready()
    lj(spec_j).block_until_ready()

    modes   = ["Velocity", "Wavelength", "LSF"]
    t_sedpy = [
        _time_calls(sedpy_ref.smooth_vel_fft,
                    (wave, spec_np, outwave, sigma_v)),
        _time_calls(sedpy_ref.smooth_wave_fft,
                    (wave, spec_np, outwave, sigma_l)),
        _time_calls(
            lambda: sedpy_ref.smooth_lsf_fft(wave, spec_np, outwave, sigma=sigma_lsf),
            ()),
    ]
    t_eager = [
        _time_calls(smooth_vel_fft,
                    (wave, spec_j, sigma_v, outwave), jax_fn=True),
        _time_calls(smooth_wave_fft,
                    (wave, spec_j, sigma_l, outwave), jax_fn=True),
        _time_calls(smooth_lsf_fft,
                    (wave, spec_j, sigma_lsf, outwave), jax_fn=True),
    ]
    t_jit = [
        _time_calls(vj, (spec_j, jnp.array(sigma_v)), jax_fn=True),
        _time_calls(wj, (spec_j, jnp.array(sigma_l)), jax_fn=True),
        _time_calls(lj, (spec_j,),                    jax_fn=True),
    ]

    for m, ts, te, tj in zip(modes, t_sedpy, t_eager, t_jit):
        print(f"  {m:12s}  sedpy={ts:.3f} ms   eager={te:.3f} ms   "
              f"JIT={tj:.3f} ms   speedup={ts/tj:.1f}×")

    # Bar chart
    x      = np.arange(len(modes))
    width  = 0.25

    fig, (ax, ax_su) = plt.subplots(1, 2, figsize=(12, 4))

    bars_s = ax.bar(x - width, t_sedpy, width, color=C_SEDPY, label="sedpy (NumPy)")
    bars_e = ax.bar(x,         t_eager, width, color="#888888", label="sedpy_jax (eager)")
    bars_j = ax.bar(x + width, t_jit,   width, color=C_JAX,    label="sedpy_jax (JIT)")

    # Annotate bars with ms values
    for bars in (bars_s, bars_e, bars_j):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.03,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Time per call [ms]  (log scale)")
    ax.set_title(f"Wall-clock time  ({N_BENCH} calls each)")
    ax.legend(fontsize=9)

    # Speedup panel
    speedups = [ts / tj for ts, tj in zip(t_sedpy, t_jit)]
    bars_su  = ax_su.bar(x, speedups, 0.5, color=C_JAX, alpha=0.85)
    for bar, su in zip(bars_su, speedups):
        ax_su.text(bar.get_x() + bar.get_width() / 2, su + 0.3,
                   f"{su:.1f}×", ha="center", va="bottom", fontsize=11,
                   fontweight="bold")
    ax_su.axhline(1, color="k", lw=0.8, ls="--")
    ax_su.set_xticks(x)
    ax_su.set_xticklabels(modes)
    ax_su.set_ylabel("Speedup  (sedpy / sedpy_jax JIT)")
    ax_su.set_title("JIT speedup over sedpy")
    ax_su.set_ylim(0, max(speedups) * 1.25)

    fig.suptitle(
        f"sedpy vs sedpy_jax timing  —  {N_BENCH} calls, "
        f"{len(wave):,} input px, {len(outwave):,} output px",
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "timing_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: timing_comparison.pdf")


# ===========================================================================
# Summary figure — all three smoothing modes on one spectrum
# ===========================================================================

def make_summary_figure():
    """Single comparison figure: sedpy vs sedpy_jax for all three modes."""
    print("\n=== Summary figure ===")
    if not _HAS_SEDPY:
        print("  SKIP — sedpy not available.")
        return

    wave    = np.arange(4000.0, 9001.0, 0.5)
    spec_np, lines = make_mock_spectrum(wave, sigma_v_lines=10.0)
    spec_j  = jnp.array(spec_np)
    outwave = np.arange(4200.0, 8801.0, 2.0)

    sigma_v   = 200.0
    sigma_l   = 3.0
    sigma_lsf = 1.5 + 3.0 * (wave - 4000.0) / 5000.0   # 1.5 → 4.5 Å

    # sedpy reference
    sv_ref  = sedpy_ref.smooth_vel_fft( wave, spec_np, outwave, sigma_v)
    sw_ref  = sedpy_ref.smooth_wave_fft(wave, spec_np, outwave, sigma_l)
    sl_ref  = sedpy_ref.smooth_lsf_fft(wave, spec_np, outwave, sigma=sigma_lsf)

    # sedpy_jax
    sv_jax  = np.asarray(smooth_vel_fft( wave, spec_j, sigma_v,   outwave))
    sw_jax  = np.asarray(smooth_wave_fft(wave, spec_j, sigma_l,   outwave))
    sl_jax  = np.asarray(smooth_lsf_fft( wave, spec_j, sigma_lsf, outwave))

    r_vel  = residual_pct(sv_jax, sv_ref)
    r_wave = residual_pct(sw_jax, sw_ref)
    r_lsf  = residual_pct(sl_jax, sl_ref)

    print(f"  vel:  max|Δ| = {np.nanmax(np.abs(r_vel)):.5f}%")
    print(f"  wave: max|Δ| = {np.nanmax(np.abs(r_wave)):.5f}%")
    print(f"  lsf:  max|Δ| = {np.nanmax(np.abs(r_lsf)):.5f}%")

    # 3 columns × (spectrum + residual)
    fig = plt.figure(figsize=(15, 6))
    outer = gridspec.GridSpec(1, 3, wspace=0.30, figure=fig)
    inner = [
        gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i],
                                         height_ratios=[3, 1], hspace=0.06)
        for i in range(3)
    ]
    axs = [[fig.add_subplot(inner[c][r]) for r in range(2)] for c in range(3)]

    datasets = [
        (sv_ref, sv_jax, r_vel,  rf"(a) Velocity  $\sigma_v={sigma_v:.0f}$ km/s",
         (-0.05, 0.05)),
        (sw_ref, sw_jax, r_wave, rf"(b) Wavelength  $\sigma_\lambda={sigma_l}$ Å",
         (-0.05, 0.05)),
        (sl_ref, sl_jax, r_lsf,  r"(c) LSF  $\sigma=1.5$–$4.5$ Å",
         (-0.8,   0.8)),
    ]

    for col, (ref, jax_arr, res, title, ylim) in enumerate(datasets):
        ax0, ax1 = axs[col]
        rmax_col = np.nanmax(np.abs(res))

        ax0.plot(outwave, ref,     color=C_SEDPY, lw=1.5, label="sedpy")
        ax0.plot(outwave, jax_arr, color=C_JAX,   lw=0.9, ls="--", alpha=0.9,
                 label="sedpy_jax")
        mark_lines(ax0, lines, outwave)
        ax0.set_title(f"{title}\nmax$|\\Delta|$={rmax_col:.5f}%", loc="left", fontsize=10)
        ax0.set_ylabel(r"Flux [arb.]")
        ax0.legend(fontsize=9)
        ax0.tick_params(labelbottom=False)

        ax1.plot(outwave, res, color=C_RESID, lw=0.8)
        ax1.axhline(0, color="k", lw=0.6, ls="--")
        mark_lines(ax1, lines, outwave)
        ax1.set_xlabel(r"Wavelength [$\AA$]")
        ax1.set_ylabel(r"$\Delta$ [%]")
        ax1.set_ylim(*ylim)

        for ax in (ax0, ax1):
            ax.set_xlim(outwave[0], outwave[-1])

    fig.suptitle("sedpy_jax vs sedpy — all smoothing modes", fontsize=12)
    fig.savefig(os.path.join(FIG_DIR, "summary_smoothing_modes.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: summary_smoothing_modes.pdf")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  sedpy_jax.smoothing — validation suite")
    print("  Primary metric: agreement with sedpy (NumPy reference)")
    print("=" * 60)

    jax.config.update("jax_enable_x64", True)

    results = {}
    results["Test 1: FFT kernel"]        = test_fft_kernel()
    results["Test 2: vel  vs sedpy"]     = test_vel_fft()
    results["Test 3: wave vs sedpy"]     = test_wave_fft()
    results["Test 4: LSF  vs sedpy"]     = test_lsf_fft()
    results["Test 5: gradient"]          = test_gradient()
    results["Test 6: inres subtraction"] = test_inres_subtraction()

    make_timing_figure()
    make_summary_figure()

    print("\n" + "=" * 60)
    print("  RESULTS  (sedpy_jax vs sedpy as primary criterion)")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        if passed is None:
            mark = "SKIP"
        else:
            mark = "PASS" if passed else "FAIL"
            all_pass = all_pass and passed
        print(f"  {mark}  {name}")

    print("=" * 60)
    skipped = sum(v is None for v in results.values())
    if skipped:
        print(f"  {skipped} test(s) skipped — install sedpy in this environment.")
    print(f"  {'All executed tests passed.' if all_pass else 'Some tests FAILED.'}")
    print(f"  Figures saved to: {FIG_DIR}/")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)
