# gfckit

**General-Fractional-Calculus toolkit** for data-driven identification of
degradation memory kernels in **batteries and corrosion**.

Aging chemistry is a *memory* (nonlocal-in-time) process and is usually
*non-Fickian*. `gfckit` models a degradation observable `u(t)` by a general
fractional evolution and identifies, from data, the **anomalous order α**, the
**memory horizon λ**, and an **internal clock g** (Arrhenius / stress-weighted
time). Built on Tarasov's parametric general fractional calculus
(arXiv:2509.12218). The identifiability theory behind this toolkit is
developed in a companion paper, currently in preparation for *Communications
in Nonlinear Science and Numerical Simulation*; a citation will be added here
once it is posted.

Everything numerical is **pure JAX** (no numpy/scipy in the compute path);
matplotlib is used only for plotting.

## Install

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra plot --extra test
```

This creates `.venv` and installs `gfckit` in editable mode with its optional
plotting and testing dependencies. Run any command below with `uv run`, e.g.
`uv run pytest tests -q`, or activate the environment first:

```bash
source .venv/bin/activate
```

Without uv, a plain venv works the same way:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[plot,test]"
```

## Quick start

```python
import jax; jax.config.update("jax_enable_x64", True)
from gfckit import build_conditions, recover_params_weak

# synthetic: one shared tempered-power-law kernel, several Arrhenius clocks
rates = [0.6, 0.85, 1.1, 1.4]
t, u_list, ds_list = build_conditions(alpha=0.5, lam=0.5, c_sink=1.0,
                                      rates=rates, N=600, dt=0.02, noise=0.05)

# noise-robust recovery of (alpha, lambda) from all conditions at once
alpha_hat, lam_hat = recover_params_weak(u_list, rates, c_sink=1.0, N=600, dt=0.02)
print(alpha_hat, lam_hat)   # ~ (0.5, 0.5)
```

## Examples (write figures to `figures/`)

See [`examples/README.md`](examples/README.md) for the **equations each one solves**.

```bash
python examples/order_identification.py    # strong vs weak order recovery vs noise
python examples/multicondition.py          # 4-panel multi-condition summary
python examples/physical_data.py           # mechanistic data (Newton-Krylov) -> gfckit
python examples/fick_diffusion.py          # Fick's law systems; parabolic uptake
```

## Modules

| Module | Contents |
|---|---|
| `operators` | L1 Caputo derivative, kernel L1 weights, `apply_gfd`, `substitution_gfd` |
| `kernels`   | tempered power law, exponential-sum (completely-monotone) kernels |
| `generate`  | analytic `t^α` ground truth, GFODE forward solve, multi-condition data |
| `identify`  | order recovery (strong/weak), `(α,λ)` recovery (strong/weak), free kernel |
| `physics`   | mechanistic models (SEI reaction-diffusion, corrosion PDM), solved via Jacobian-free Newton-Krylov |
| `fick`      | analytic Fick's-law solutions (erfc, Gaussian, slab), flux, uptake |
| `plotting`  | summary figures |

## Cross-model validation

`physics` generates aging data from **independent mechanistic models** (moving-boundary
reaction-diffusion for SEI; high-field Point Defect Model for corrosion), solved with
**Jacobian-free Newton–Krylov**. Feeding those curves to `identify` is a genuine test
(not self-fulfilling): SEI recovers an effective exponent ≈0.59 (diffusion-limited,
trending to Fickian ½); the passivating PDM film needs a *tempered* (finite-memory)
kernel. `fick` supplies the Fickian baseline where diffusion-limited uptake ~ √t gives
exactly α = ½.

## Methods, honestly

- **Order α (single curve):** well-posed; weak/integral form is noise-robust.
- **Tempered `(α,λ)` (multi-condition):** shared kernel across Arrhenius clocks;
  error falls as conditions are added. Weak form handles noise.
- **Free nonparametric kernel:** ill-posed from few conditions, improves with
  more, but needs strong regularization (future work).

## Tests

```bash
pytest tests -q
```
