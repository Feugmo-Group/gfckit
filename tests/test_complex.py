"""Tests for the non-power-law generators and the benchmark suite."""
import jax
import jax.numpy as jnp

from gfckit.complex import (multiterm_fractional_relaxation,
                            variable_order_solution, knee_capacity_fade,
                            local_loglog_slope)
from gfckit.benchmark import make_benchmark

jax.config.update("jax_enable_x64", True)


def test_multiterm_reduces_to_single_power_law():
    """One term with alpha, weight 1: recovers ^C D^a u = S => u ~ t^a (const slope)."""
    t, u = multiterm_fractional_relaxation([0.5], [1.0], lam=0.0, S=1.0,
                                           u0=0.0, dt=0.02, nsteps=400)
    _, sl = local_loglog_slope(t, u)
    assert float(jnp.std(sl[50:])) < 0.02        # single power law => flat slope


def test_multiterm_two_orders_not_power_law_but_monotone():
    t, u = multiterm_fractional_relaxation([0.3, 0.8], [0.6, 0.4], lam=0.02,
                                           S=0.03, u0=0.0, dt=0.05, nsteps=400)
    assert bool((u[1:] >= u[:-1] - 1e-9).all())


def test_variable_order_slope_varies():
    def path(x):
        return 0.3 + 0.6 / (1.0 + jnp.exp(-(x - 10.0) / 2.0))
    t, u = variable_order_solution(path, lam=0.02, S=0.03, u0=0.0,
                                   dt=0.05, nsteps=400)
    _, sl = local_loglog_slope(t, u)
    assert float(jnp.std(sl[20:])) > 0.1         # clearly not a single power law


def test_knee_is_superlinear_late():
    t = jnp.linspace(1.0, 540.0, 400)
    y = knee_capacity_fade(t, A_sei=0.6, B_plate=0.18, t_knee=350.0, w_knee=30.0)
    _, sl = local_loglog_slope(t, y)
    assert float(sl[-1]) > 0.7                    # knee pushes exponent above sqrt


def test_benchmark_families_separate():
    """Power-law curves have tiny slope-variation; non-power-law curves large."""
    ds = make_benchmark(seed=0)
    for d in ds:
        if d["family"] == "power-law":
            assert d["slope_var"] < 0.05
        else:
            assert d["slope_var"] > 0.02 or d["pl_R2"] < 0.999
