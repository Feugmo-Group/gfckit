"""Numerical inverse Laplace transform (fixed Talbot) and distributed-order
responses over arbitrarily many time decades (pure JAX).

The L1 solvers use a uniform grid and so cannot span many decades cheaply.  For
the identifiability study we need the response of a distributed-order relaxation
on a logarithmic time axis covering 6-8 decades; the fixed-Talbot method
(Abate & Valko) inverts the closed-form Laplace transform pointwise at any t.

Distributed-order relaxation  (sum_i w_i ^C D^{a_i}) u + lambda u = S,  u(0)=0
has Laplace transform  u_hat(s) = S / ( s ( sum_i w_i s^{a_i} + lambda ) ).
"""
import jax.numpy as jnp


def talbot_inverse(F, t, M=48):
    """Fixed-Talbot inverse Laplace of F at scalar time t (Abate-Valko)."""
    k = jnp.arange(1, M)
    th = k * jnp.pi / M
    ctn = 1.0 / jnp.tan(th)
    d0 = 2.0 * M / 5.0
    dk = (2.0 * k * jnp.pi / 5.0) * ctn + 1j * (2.0 * k * jnp.pi / 5.0)
    g0 = 0.5 * jnp.exp(d0)
    gk = (1.0 + 1j * th * (1.0 + ctn ** 2) - 1j * ctn) * jnp.exp(dk)
    return (2.0 / (5.0 * t)) * (jnp.real(g0 * F(d0 / t))
                                + jnp.sum(jnp.real(gk * F(dk / t))))


def distributed_order_response(alphas, weights, t, lam=0.0, S=1.0, M=48):
    """u(t) for  (sum_i w_i ^C D^{a_i}) u + lam u = S  on the given times t
    (any log-spaced grid), via Talbot inversion of the closed-form transform."""
    aw = list(zip(alphas, weights))

    def F(s):
        denom = sum(w * s ** a for a, w in aw) + lam
        return S / (s * denom)

    return jnp.array([talbot_inverse(F, float(ti), M) for ti in t])
