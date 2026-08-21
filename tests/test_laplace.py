"""Tests for the Talbot inverse Laplace transform and dynamic-range analysis."""
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from gfckit.laplace import talbot_inverse, distributed_order_response

jax.config.update("jax_enable_x64", True)


def test_talbot_matches_powerlaw():
    """Single order: L^{-1}[s^{-(1+a)}] = t^a / Gamma(1+a)."""
    for a in [0.3, 0.7]:
        F = lambda s, a=a: 1.0 / (s * s ** a)
        for t in [0.1, 1.0, 10.0]:
            approx = float(talbot_inverse(F, t))
            exact = float(t ** a / jnp.exp(gammaln(1 + a)))
            assert abs(approx - exact) / exact < 1e-4


def test_two_order_slope_spans_both_orders():
    """Over 8 decades the local slope of a two-order memory spans [a_lo, a_hi]."""
    t = jnp.logspace(-4.0, 4.0, 120)
    u = distributed_order_response((0.2, 0.9), (0.5, 0.5), t)
    slope = jnp.gradient(jnp.log(u), jnp.log(t))
    assert float(slope.max()) > 0.85
    assert float(slope.min()) < 0.25


def test_two_order_slope_narrow_over_two_decades():
    """Over only 2 decades about the crossover, far less of the range is seen."""
    t = jnp.logspace(-1.0, 1.0, 80)          # ~2 decades centered near t=1
    u = distributed_order_response((0.2, 0.9), (0.5, 0.5), t)
    slope = jnp.gradient(jnp.log(u), jnp.log(t))
    assert (float(slope.max()) - float(slope.min())) < 0.6   # << 0.7 full range
