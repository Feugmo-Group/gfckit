"""Tests for de Levie's porous electrode -- the one model in the package whose
answer is fixed by the literature rather than by us, so the tolerances here are
deliberately tight."""
import jax
import jax.numpy as jnp
import numpy as np

from gfckit.delevie import (crossover_frequency, fit_cpe_exponent,
                            local_exponent, pore_impedance,
                            semi_infinite_pore_impedance)

jax.config.update("jax_enable_x64", True)

R, C, L = 1.0e6, 1.0e-3, 1.0e-4
WC = float(crossover_frequency(R, C, L))


def _hf_grid(lo=2, hi=5, n=400):
    return jnp.logspace(np.log10(WC) + lo, np.log10(WC) + hi, n)


def test_high_frequency_exponent_is_exactly_one_half():
    """de Levie's known result: the HF branch is a CPE with alpha = 1/2."""
    w = _hf_grid()
    alpha, _, rms = fit_cpe_exponent(w, jnp.abs(pore_impedance(w, R, C, L)))
    assert abs(float(alpha) - 0.5) < 1e-5
    assert float(rms) < 1e-6


def test_high_frequency_phase_is_minus_45_degrees():
    w = _hf_grid()
    phase = jnp.angle(pore_impedance(w, R, C, L), deg=True)
    assert abs(float(jnp.mean(phase)) + 45.0) < 1e-3


def test_high_frequency_amplitude_is_sqrt_R_over_C():
    w = _hf_grid()
    _, A, _ = fit_cpe_exponent(w, jnp.abs(pore_impedance(w, R, C, L)))
    assert abs(float(A) / np.sqrt(R / C) - 1.0) < 1e-4


def test_tends_to_semi_infinite_limit_above_crossover():
    w = _hf_grid()
    Z, Zinf = pore_impedance(w, R, C, L), semi_infinite_pore_impedance(w, R, C)
    assert float(jnp.max(jnp.abs(Z - Zinf) / jnp.abs(Zinf))) < 1e-5


def test_low_frequency_branch_is_a_single_capacitor():
    """Below w_c the whole pore charges together: Z -> 1/(jwCL), exponent 1."""
    w = jnp.logspace(np.log10(WC) - 6, np.log10(WC) - 3, 300)
    alpha, _, _ = fit_cpe_exponent(w, jnp.abs(pore_impedance(w, R, C, L)))
    assert abs(float(alpha) - 1.0) < 1e-4
    rel = jnp.abs(pore_impedance(w, R, C, L) - 1.0 / (1j * w * C * L))
    assert float(jnp.max(rel * jnp.abs(1j * w * C * L))) < 1e-3


def test_geometry_is_invisible_above_the_crossover():
    """The point of Sec 3.4: equal R/C and any length give identical responses
    while the window stays well above w_c.  Uses the same 2-5 decade window the
    reported alpha = 0.500 is measured on."""
    w = _hf_grid()
    ref = pore_impedance(w, R, C, L)
    for fac, Lf in [(10.0, 3.0), (100.0, 0.5), (1000.0, 7.0)]:
        other = pore_impedance(w, R * fac, C * fac, L * Lf)
        assert float(jnp.max(jnp.abs(other - ref) / jnp.abs(ref))) < 1e-5


def test_geometry_degeneracy_reaches_machine_precision():
    """Three decades above w_c the pores are not merely close, they agree to
    roughly double precision -- which is what makes the claim analytic rather
    than numerical."""
    w = _hf_grid(lo=3, hi=6)
    ref = pore_impedance(w, R, C, L)
    other = pore_impedance(w, R * 10.0, C * 10.0, L * 3.0)
    assert float(jnp.max(jnp.abs(other - ref) / jnp.abs(ref))) < 1e-12


def test_geometry_becomes_visible_at_the_crossover():
    """...and is recoverable once the window reaches down to w_c, otherwise the
    claim above would be vacuous."""
    w = jnp.logspace(np.log10(WC) - 2, np.log10(WC) + 1, 400)
    ref = jnp.abs(pore_impedance(w, R, C, L))
    other = jnp.abs(pore_impedance(w, R * 10.0, C * 10.0, L * 3.0))
    scale = jnp.sum(ref * other) / jnp.sum(other ** 2)
    assert float(jnp.max(jnp.abs(scale * other - ref) / ref)) > 0.01


def test_local_exponent_undershoots_below_one_half():
    """The transition is not monotone -- it dips to ~0.36 near 5 w_c.  A narrow
    window placed there reports neither analytic limit."""
    w = jnp.logspace(np.log10(WC) - 1, np.log10(WC) + 3, 4000)
    ex = local_exponent(w, jnp.abs(pore_impedance(w, R, C, L)))
    assert float(jnp.min(ex)) < 0.45
    assert abs(float(jnp.min(ex)) - 0.360) < 0.02
    assert 1.0 < float(w[int(jnp.argmin(ex))]) / WC < 20.0


def test_exponent_endpoints_bracket_the_two_analytic_limits():
    w = jnp.logspace(np.log10(WC) - 4, np.log10(WC) + 5, 900)
    ex = local_exponent(w, jnp.abs(pore_impedance(w, R, C, L)))
    assert abs(float(ex[0]) - 1.0) < 1e-3
    assert abs(float(ex[-1]) - 0.5) < 1e-3
