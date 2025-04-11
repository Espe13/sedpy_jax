# sedpy_jax

**JAX-accelerated SED fitting tools, filter handling, and broadband photometry projection for astronomy.**

Built as a modern, GPU-ready reimplementation of `sedpy` to support differentiable, scalable spectral energy distribution (SED) fitting — ideal for simulation-based inference, gradient-based samplers (e.g. HMC, NUTS), and high-throughput photometric pipelines.

---

## 🚀 Features

- 🧠 **Fully JAX-compatible** filter operations and broadband projection
- 📦 Clean, modular implementation of the `Filter` and `FilterSet` classes
- 🌈 Easy visualization of filter transmission curves with colormap scaling
- 🧮 Accurate AB and Vega magnitude computation
- 📚 Utility functions for wavelength conversion, rebinning, and interpolation
- 🧰 Built-in effective wavelength, pivot wavelength, width, and zero-point computation

---

## 📦 Installation

### From GitHub (recommended for users)

```bash
pip install git+https://github.com/Espe13/sedpy_jax.git

For development (recommended for contributors)

git clone https://github.com/Espe13/sedpy_jax.git
cd sedpy_jax
pip install -e .



⸻

🛠 Example: Using FilterSet to Project SEDs

from sedpy_jax.observate import FilterSet
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Load a set of filters
fset = FilterSet(['sdss_u0', 'sdss_g0', 'sdss_r0', 'acs_wfc_f435w'])

# Plot transmission curves
fset.display(normalize=True)
plt.show()

# Project an SED onto filters
wave = jnp.linspace(3000, 10000, 500)
flux = jnp.exp(-0.5 * ((wave - 6000) / 800)**2)  # toy SED
maggies = fset.get_sed_maggies(flux, sourcewave=wave)



⸻

📊 Table: Filter Properties

Name	λ_eff [Å]	Width [Å]	AB→Vega	N pts
sdss_u0	3551.3	569.0	-0.0679	180
sdss_g0	4686.5	1282.1	-0.0206	220
sdss_r0	6165.2	1113.2	0.0217	190
acs_wfc_f435w	4317.8	947.3	0.0132	202



⸻

🌈 Example: Transmission Plot

Filters plotted with color based on their effective wavelength. Optionally normalized.

⸻

📂 Included Tools
	•	Filter: load and manipulate individual transmission curves
	•	FilterSet: project SEDs across many filters efficiently
	•	air2vac / vac2air: wavelength conversion (Allen/Morton standards)
	•	rebin: flux-conserving rebinning of filters
	•	getSED(): compute AB mags or maggies from source spectra

⸻

🧪 Tests

To run tests:

pytest tests/



⸻

📮 Feedback & Contributions

Feel free to open Issues or pull requests.

📧 Created and maintained by Amanda Stoffers (@Espe13)

⸻

📜 License

MIT License. See LICENSE.

---

### ✨ Suggestions

- Save an example filter plot as `docs/images/filter_transmission_example.png` to show off in the README.
- Use `rich` or `tabulate` in code if you want to generate the property table programmatically.
- Add badges (`PyPI`, `Tests`, `Docs`) once you're ready to publish or CI test.

