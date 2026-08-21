"""Tests for the mechanistic (Newton-Krylov) forward models and Fick solutions."""
import jax
import jax.numpy as jnp

from gfckit.physics import simulate_sei_reaction_diffusion, simulate_pdm_film_growth
from gfckit.fick import diffusion_uptake, semi_infinite_constant_surface
from gfckit.identify import recover_order_weak

jax.config.update("jax_enable_x64", True)


def test_sei_monotone_and_subfickian():
    """SEI film grows monotonically; effective exponent between 1/2 and 1."""
    t, L = simulate_sei_reaction_diffusion(nsteps=300)
    assert bool((L[1:] >= L[:-1] - 1e-9).all())
    a = float(recover_order_weak(L[t > 1.0], t[t > 1.0], jnp.linspace(0.3, 1.2, 91)))
    assert 0.45 < a < 1.0


def test_pdm_passivates_to_steady_state():
    """PDM film approaches the analytic steady-state thickness."""
    t, L, L_ss = simulate_pdm_film_growth(nsteps=500)
    assert abs(float(L[-1]) - L_ss) < 0.05
    assert bool((L[1:] >= L[:-1] - 1e-9).all())


def test_fick_uptake_is_parabolic():
    """Diffusion-limited uptake ~ sqrt(t): identified alpha ~ 1/2."""
    t = jnp.linspace(0.02, 8.0, 400)
    u = diffusion_uptake(t, D=1.0)
    a = float(recover_order_weak(u, t, jnp.linspace(0.2, 1.0, 161)))
    assert abs(a - 0.5) < 0.03


def test_fick_erfc_boundary_values():
    """erfc profile: c(0,t)=c_s and c(inf,t)->0."""
    x = jnp.array([0.0, 50.0])
    c = semi_infinite_constant_surface(x, t=1.0, D=1.0, cs=2.0)
    assert abs(float(c[0]) - 2.0) < 1e-9
    assert float(c[1]) < 1e-6
