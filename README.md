# gfckit

**G**eneral-**F**ractional-**C**alculus toolkit for degradation modelling: it
recovers the memory structure of an aging process from a measured curve, and
maps how much of that structure the measurement can actually support.

Installed and imported as `gfckit`.

> **gfckit fits a general fractional evolution equation to a degradation
> curve and tells you which part of the fitted memory kernel the data could
> have supported in the first place.**

Aging chemistry, battery capacity fade, corrosion film growth, is a memory
(nonlocal-in-time) process and usually non-Fickian. It is tempting to invert
a single aging curve for the underlying kernel and read off the physics.
gfckit tests that directly: it recovers an effective anomalous order, a
tempered pair `(α, λ)`, or a full order spectrum `w(α)`, from data, and
ships the diagnostics that say which of those is actually identifiable at a
given noise level, condition count, and observation window rather than just
returning a number. A separate module answers a companion question for
batteries specifically: given only what an experimenter could measure
(capacity fade, a differential-voltage curve, impedance), which degradation
mechanism was actually responsible.

## About

Recovering a memory kernel's order from data is established practice;
knowing whether it was recoverable at all from that data is not. gfckit is
built around that gap. Its computational core (`operators`, `kernels`,
`generate`, `identify`) is pure JAX: the L1 Caputo derivative and a general
fractional evolution equation whose kernel is Sonine rather than a fixed
power law, differentiable end to end, so gradient-based recovery of `(α,
λ)` or of a full order spectrum works directly. Recovery is checked against
independent, non-fractional ground truth rather than against itself:
`physics` generates aging curves from a moving-boundary
solid-electrolyte-interphase (SEI) model and a Point Defect Model (PDM) of
passive-film growth, both solved by Jacobian-free Newton-Krylov, and
`delevie` supplies a closed-form analytic benchmark (a cylindrical porous
electrode) whose high-frequency order is known exactly to be one half.
`identifiability` makes the package's central negative result constructive:
given a reference order spectrum, it searches for the farthest spectrum
whose response the data still cannot rule out, and the size of that gap is
the identifiability boundary, not an assumption about it.

The battery-specific half of the package (`mechanism_id`, `pybamm_gen`,
`pybamm_data`, `eis_gen`, `aged_eis`, `real_data`) answers a different but
related question on a labelled synthetic benchmark built from PyBaMM's
single-particle model with the O'Kane 2022 degradation submodels: given
capacity fade, a differential-voltage fingerprint, or impedance, and never
the underlying per-mechanism decomposition, can a simple k-nearest-neighbour
classifier recover which mechanism actually dominated a cell's aging.
Scoring is done by leave-one-configuration-out cross-validation, because the
three operating conditions sharing a parameter configuration are
near-replicates and a naive leave-one-cell-out split inflates the reported
accuracy.

## Who this is for

Researchers analysing aging, corrosion, or impedance data from batteries,
corrosion films, or other memory-bearing degradation processes, and method
developers who want a tested differentiable general-fractional operator
stack together with the identifiability diagnostics that a plain curve fit
does not provide.

## What it does

| Module | Contents |
|---|---|
| `operators` | L1 Caputo derivative, kernel L1 weights, `apply_gfd`, `substitution_gfd` (Tarasov parametric general fractional derivative: warp, apply, unwarp) |
| `kernels` | tempered power-law and exponential-sum (completely-monotone) memory kernels |
| `generate` | analytic `t^α` ground truth, general fractional ODE forward solve, multi-condition synthetic data |
| `identify` | order recovery (strong/weak form), `(α, λ)` recovery (strong/weak form, single or joint multi-condition), free-kernel and order-spectrum recovery via JAX autodiff |
| `identifiability` | the constructive non-identifiability sweep: the farthest order spectrum the data cannot exclude, and the size of that gap |
| `physics` | independent mechanistic ground truth: SEI reaction-diffusion and corrosion PDM, via Jacobian-free Newton-Krylov |
| `delevie` | the closed-form de Levie cylindrical-pore impedance, its CPE exponent fit, and the high-frequency crossover where that exponent is exactly one half |
| `fick` | analytic Fick's-law solutions (erfc, Gaussian, finite slab), flux, and diffusion-limited uptake |
| `laplace` | fixed-Talbot numerical inverse Laplace transform, used to validate the distributed-order response independently of the time-domain solver |
| `complex` | multi-term fractional relaxation, variable-order response, and a knee-shaped capacity-fade generator used as a non-power-law stress test |
| `mechanism_id` | feature construction, k-NN mechanism classification and regression, leave-one-configuration-out and leave-one-cell-out cross-validation, Wilson intervals, McNemar's test |
| `pybamm_gen`, `pybamm_data` | the labelled PyBaMM SPMe + O'Kane-2022 aging ensemble: generation, capacity/dV-dQ observables, and the hidden per-mechanism ground truth |
| `eis_gen`, `aged_eis` | mechanism-parameter EIS sensitivity spectra, and per-cell aged impedance reconstructed from each cell's own cycled state |
| `real_data` | loaders for open real-cell datasets (Zhang et al. 2020, Nature Communications, 12 commercial cells; the Oxford Battery Degradation Dataset) used to test whether the synthetic-benchmark associations transfer |
| `realistic`, `benchmark` | Arrhenius-scaled synthetic aging cohorts and CSV export, for scenarios outside the PyBaMM ensemble |
| `plotting`, `figstyle` | the figure style shared by every plot in the package and in the accompanying manuscripts |

### Cross-model validation

`physics` and `delevie` are the two checks that keep `identify` honest.
Feeding `physics`-generated curves to `identify` is a genuine test, not a
self-fulfilling one, because the two share no code: SEI reaction-diffusion
recovers an effective order near 0.59 (diffusion-limited, trending toward
the Fickian one-half as the film thins); the passivating PDM film needs a
tempered kernel, not a pure power law, because it saturates rather than
growing indefinitely. `delevie`'s cylindrical pore has a high-frequency
order known analytically to be exactly one half; its own CPE fit
(`fit_cpe_exponent`) returns 0.500.

### Methods, honestly

- **Effective order α, single curve:** well posed; the weak (integral) form
  is noise-robust up to several percent noise, where the strong (derivative)
  form collapses.
- **Tempered pair `(α, λ)`, single curve:** the two parameters trade off
  along a ridge and are not separately identifiable from one condition.
- **Tempered pair `(α, λ)`, multiple Arrhenius conditions:** identifiable
  jointly; the error falls as conditions are added.
- **Free order spectrum `w(α)`:** not identifiable from a realistic
  observation window even in the noise-free limit. A two-order memory
  collapses onto a single intermediate order that fits the generating data
  *better* than the true spectrum does, because the distinct power-law
  regimes separate only across time decades a short window does not
  contain. `identifiability` quantifies exactly how far that boundary is
  from a given experiment.
- **Mechanism attribution (battery):** identifiability is set by how
  strongly, and how smoothly, a mechanism perturbs a given observable.
  Active-material loss is recoverable as a continuous fraction; lithium
  plating perturbs capacity strongly but almost binarily and is not
  recoverable; particle cracking can be the largest single loss channel in
  a cell and remain invisible to both capacity and impedance, depending on
  how the mechanism is bookkept in the underlying degradation model.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[plot,test]"
```

Requires Python 3.9+. The compute path depends only on `jax`; `matplotlib`
is required for the plotting and figure-reproduction modules, `pytest` for
the test suite.

## Quickstart

Recover a tempered kernel `(α, λ)` jointly across several Arrhenius
conditions, from noisy synthetic data with a known answer:

```python
import jax; jax.config.update("jax_enable_x64", True)
from gfckit import build_conditions, recover_params_weak

rates = [0.6, 0.85, 1.1, 1.4]   # relative Arrhenius clock rates, one per condition
t, u_list, ds_list = build_conditions(
    alpha=0.5, lam=0.5, c_sink=1.0, rates=rates, N=600, dt=0.02, noise=0.05,
)

alpha_hat, lam_hat = recover_params_weak(u_list, rates, c_sink=1.0, N=600, dt=0.02)
print(alpha_hat, lam_hat)   # close to (0.5, 0.5); a single condition would not pin these down
```

Validate the estimator against a closed-form benchmark whose answer is known
exactly, not just plausible:

```python
import jax.numpy as jnp
from gfckit import fit_cpe_exponent
from gfckit.delevie import pore_impedance

omega = jnp.logspace(2, 6, 200)          # well above the pore's crossover frequency
Z = pore_impedance(omega, R=1.0, C=1.0, L=1.0)
alpha_hat, amplitude, rms = fit_cpe_exponent(omega, jnp.abs(Z))
print(alpha_hat)                          # 0.500, the exact high-frequency order
```

Attribute the dominant degradation mechanism behind a cell's capacity-fade
curve, scored the way the ground truth demands (grouped by configuration,
not by cell):

```python
from gfckit.pybamm_data import load_cells
from gfckit.mechanism_id import build_matrix, grouped_knn, groups_from_ids

cells = load_cells("data/pybamm_targeted")   # the labelled PyBaMM aging ensemble
X, y = build_matrix(cells, use_capacity=True, use_dvdq=False)
groups = groups_from_ids(cells)

y_hat = grouped_knn(X, y, groups, k=3)       # leave-one-configuration-out predictions
accuracy = (y_hat == y).mean()               # 0.375, matching the paper exactly
```

## Examples

The `examples/` directory walks through the full pipeline. See
[`examples/README.md`](examples/README.md) for the equation each script
solves and what its numbers mean. Every example writes its figure to
`figures/` and is run from the package root:

```bash
python examples/order_identification.py    # strong vs. weak order recovery vs. noise
python examples/multicondition.py          # tempered (alpha, lambda) recovery across conditions
python examples/physical_data.py           # independent mechanistic ground truth -> gfckit
python examples/fick_diffusion.py          # Fick's-law reference systems; parabolic uptake
python examples/delevie_validation.py      # closed-form CPE benchmark, alpha = 1/2 exactly
python examples/pybamm_ensemble.py         # build the labelled battery aging ensemble
```

## Data

`data/` holds the loaders and manifests for the open real-cell datasets used
to test transfer beyond the synthetic benchmark; see `data/DATA.md`.

## Testing

```bash
pytest tests -q
```

The suite checks the computational core against analytic and independent
results: the Caputo L1 derivative against known closed forms, order and
tempered-kernel recovery against the parameters that generated the data,
and the fixed-Talbot inverse Laplace transform against the time-domain
solver.

## Related work

- Tarasov's parametric general fractional calculus, the framework the
  `operators` module implements (arXiv:2509.12218).
- de Levie, *Electrochim. Acta* 8, 751 (1963). The closed-form porous
  electrode used as the analytic benchmark in `delevie`.

## Citation

If you use this package, please cite it; see [`CITATION.cff`](CITATION.cff).

## License

MIT
