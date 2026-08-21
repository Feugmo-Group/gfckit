"""Tests for the identification methods."""
import jax
import jax.numpy as jnp

from gfckit.generate import analytic_powerlaw, build_conditions
from gfckit.identify import (recover_order_weak, recover_params_strong,
                             recover_params_weak)

jax.config.update("jax_enable_x64", True)


def test_order_recovery_clean():
    """Weak-form order recovery is exact on clean analytic data."""
    N = 2000
    t = jnp.linspace(0.0, 5.0, N)
    for alpha in [0.5, 0.7]:
        u, _ = analytic_powerlaw(t, alpha)
        a = float(recover_order_weak(u, t, jnp.linspace(0.05, 1.5, 291)))
        assert abs(a - alpha) < 0.01


def test_multicondition_beats_single():
    """4 conditions recover (alpha,lam) better than 1 (identifiability)."""
    alpha, lam, c = 0.5, 0.5, 1.0
    N, dt, n_lag = 600, 0.02, 250
    rates = [0.6, 0.85, 1.1, 1.4]
    _, u1, ds1 = build_conditions(alpha, lam, c, rates[:1], N, dt)
    _, u4, ds4 = build_conditions(alpha, lam, c, rates, N, dt)
    a1, l1 = recover_params_strong(u1, ds1, c, n_lag)
    a4, l4 = recover_params_strong(u4, ds4, c, n_lag)
    err1 = abs(a1 - alpha) + abs(l1 - lam)
    err4 = abs(a4 - alpha) + abs(l4 - lam)
    assert err4 < err1


def test_weak_form_noise_robust():
    """Weak form recovers alpha within 0.1 at 5% noise where strong form fails."""
    alpha, lam, c = 0.5, 0.5, 1.0
    N, dt = 600, 0.02
    rates = [0.6, 0.85, 1.1, 1.4]
    _, u, _ = build_conditions(alpha, lam, c, rates, N, dt, noise=0.05, seed=7)
    a_w, _ = recover_params_weak(u, rates, c, N, dt)
    assert abs(a_w - alpha) < 0.1
