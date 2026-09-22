# Contributing to gfckit

Contributions are welcome: bug reports, failing test cases, new kernels, new
identification estimators, new dataset loaders, and documentation fixes.

## Reporting a bug

Open an issue with the smallest script that reproduces it, the output you
got, and the output you expected. Include the versions of `gfckit`, `jax`,
and Python.

A recovered parameter that looks physically wrong is a bug report even if
you are not certain it is wrong. The identifiability boundaries this package
maps were found the same way, by noticing a number that could not be right.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[plot,test]"
pytest tests -q
```

## House rules

**Every test must be able to fail.** Verify against an independent
computation (an analytic solution, a closed-form limit, an unrelated
estimator) or a known ground truth, never against the module's own output.

**Keep the compute path pure JAX.** `numpy`/`scipy` are for I/O, plotting,
and glue code only. If a function needs to be `jax.grad`-able or
`jax.jit`-able, it stays inside the JAX path; mixing backends silently
breaks both.

**State an estimator's failure mode, not just its success case.** This
package exists to map identifiability boundaries, so a new estimator is
incomplete without a statement of the regime where it is expected to fail
(noise level, condition count, observation window). See
`identifiability.py` and `mechanism_id.py` for the pattern.

**Physics-generated data must come from an independent model.** Ground
truth for validating an estimator (`physics.py`, `pybamm_gen.py`) must not
share code with the estimator itself. Cross-validation against your own
forward model tests self-consistency, not correctness.

**Quantitative claims in the docs come from a script.** If you add a number
to a docstring or to the README, add the example that produces it, or point
to the one that does.

## Pull requests

- One logical change per pull request.
- Tests must pass (`pytest tests -q`).
- Explain in the description what was wrong and how you know the fix works.
  If you found a defect, say what the failing check was.

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
