"""Constructive non-identifiability of the fractional-order spectrum.

Section 3.5 of the paper shows *empirically* that a two-order memory collapses to
a single effective order.  This module makes the statement constructive: given a
reference spectrum, it finds the spectrum FARTHEST from it (in 1-Wasserstein
distance on the order axis) whose response still matches the reference to within
a stated relative-RMS noise floor, after the free amplitude and offset that a
real fit would absorb.

A non-zero maximal gap is a certificate of non-identifiability: two spectra that
far apart generate data an experimenter at that noise floor cannot tell apart.

The sweep needs ~10^5 forward solves, so the fixed-Talbot inversion of
``laplace.py`` is re-expressed with the nodes d_k/t as an (n_t x M) matrix; the
whole response is then one array expression and the sweep vmaps over spectra.
Agrees with ``laplace.distributed_order_response`` to ~4e-10 relative.
"""
import itertools

import jax
import jax.numpy as jnp
import numpy as np

A_MIN, A_MAX = 0.05, 0.95      # admissible order axis
_M = 48                        # Talbot nodes


def talbot_nodes(t, M=_M):
    """Fixed-Talbot nodes s (n_t x M) and weights g for the times t."""
    k = jnp.arange(1, M)
    th = k * jnp.pi / M
    ctn = 1.0 / jnp.tan(th)
    d0 = 2.0 * M / 5.0
    dk = (2.0 * k * jnp.pi / 5.0) * ctn + 1j * (2.0 * k * jnp.pi / 5.0)
    g0 = 0.5 * jnp.exp(d0)
    gk = (1.0 + 1j * th * (1.0 + ctn ** 2) - 1j * ctn) * jnp.exp(dk)
    d = jnp.concatenate([jnp.array([d0 + 0j]), dk])
    g = jnp.concatenate([jnp.array([g0 + 0j]), gk])
    return d[None, :] / t[:, None], g


def make_response(t, M=_M):
    """Return response(a_lo, a_hi, w) -> u(t) for the two-order spectrum
    w*delta(a_lo) + (1-w)*delta(a_hi), vmap- and jit-friendly."""
    s, g = talbot_nodes(t, M)
    pref = 2.0 / (5.0 * t)

    def response(a_lo, a_hi, w):
        denom = w * s ** a_lo + (1.0 - w) * s ** a_hi
        return pref * jnp.real(g[None, :] / (s * denom)).sum(axis=1)

    return response


def match_residual(u_ref, u):
    """Relative RMS residual after the best free amplitude and offset.

    The source strength S and any additive baseline are nuisance parameters an
    experimenter fits, so they must be projected out before asking whether two
    curves are distinguishable.
    """
    Phi = jnp.stack([u, jnp.ones_like(u)], axis=1)
    coef, *_ = jnp.linalg.lstsq(Phi, u_ref, rcond=None)
    resid = u_ref - Phi @ coef
    return jnp.sqrt(jnp.mean(resid ** 2)) / jnp.sqrt(jnp.mean(u_ref ** 2))


def w1_gap(orders_a, w_a, orders_b, w_b, a_min=A_MIN, a_max=A_MAX):
    """1-Wasserstein distance between two discrete spectra, normalised by the
    width of the order axis: the fraction of the axis the mass has moved."""
    knots = np.unique(np.concatenate([np.asarray(orders_a, float),
                                      np.asarray(orders_b, float),
                                      [a_min, a_max]]))

    def cdf(orders, w):
        o, w = np.asarray(orders, float), np.asarray(w, float)
        return np.array([w[o <= k].sum() for k in knots])

    d = np.abs(cdf(orders_a, w_a) - cdf(orders_b, w_b))
    return float(np.sum(d[:-1] * np.diff(knots)) / (a_max - a_min))


def sweep_residuals(orders_ref, w_ref, D, n_t=300, n_a=55, n_w=61, M=_M):
    """Residual of every two-order spectrum on an (n_a x n_a x n_w) grid against
    the reference, over a window of D decades centred on the crossover.

    Returns (a_lo, a_hi, w, residual) as flat numpy arrays.
    """
    t = jnp.logspace(-D / 2, D / 2, n_t)
    response = make_response(t, M)
    u_ref = response(orders_ref[0], orders_ref[1], w_ref[0])

    a_grid = np.linspace(A_MIN, A_MAX, n_a)
    lo, hi = np.array(np.meshgrid(a_grid, a_grid, indexing="ij")).reshape(2, -1)
    keep = lo < hi
    lo, hi = lo[keep], hi[keep]
    w_grid = np.linspace(0.0, 1.0, n_w)
    LO = np.repeat(lo, n_w)
    HI = np.repeat(hi, n_w)
    WW = np.tile(w_grid, len(lo))

    fn = jax.jit(jax.vmap(lambda a, b, w: match_residual(u_ref, response(a, b, w))))
    resid = np.asarray(fn(jnp.array(LO), jnp.array(HI), jnp.array(WW)))
    return LO, HI, WW, np.where(np.isfinite(resid), resid, np.inf)


def max_gap(orders_ref, w_ref, sweep, floor):
    """Largest spectral gap among spectra admissible at the given noise floor.

    Returns (gap, best) with best = (a_lo, a_hi, w, residual), or (0.0, None).
    """
    LO, HI, WW, resid = sweep
    idx = np.flatnonzero(resid <= floor)
    if idx.size == 0:
        return 0.0, None
    gaps = np.array([w1_gap(orders_ref, w_ref, (LO[i], HI[i]), (WW[i], 1 - WW[i]))
                     for i in idx])
    j = int(np.argmax(gaps))
    k = idx[j]
    return float(gaps[j]), (float(LO[k]), float(HI[k]), float(WW[k]), float(resid[k]))
