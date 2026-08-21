"""Correctness tests for the forward operator against analytic identities."""
import jax
import jax.numpy as jnp
import pytest

from gfckit.operators import caputo_L1, gamma

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7, 1.0])
def test_caputo_of_powerlaw_is_constant(alpha):
    """^C D^a [t^a] should equal Gamma(a+1) on the interior."""
    N = 4000
    t = jnp.linspace(0.0, 5.0, N)
    ds = float(t[1] - t[0])
    d = caputo_L1(t ** alpha, ds, alpha)
    est = float(jnp.median(d[N // 4:]))
    assert abs(est - float(gamma(alpha + 1.0))) < 1e-3


def test_fickian_limit():
    """alpha=1 reduces to the ordinary derivative: ^C D^1 [t] = 1."""
    N = 2000
    t = jnp.linspace(0.0, 5.0, N)
    ds = float(t[1] - t[0])
    d = caputo_L1(t, ds, 1.0)
    assert abs(float(jnp.median(d[N // 4:])) - 1.0) < 1e-6
