"""Analytic solutions of Fick's laws for canonical concentration / flow systems
(pure JAX).  These are the *normal* (Fickian) diffusion baselines: mean-squared
displacement grows linearly in time and diffusion-limited uptake grows as
sqrt(t) -- exactly the parabolic growth law that anomalous / memory kinetics
deviate from.

Fick's first law (flux):   J = -D dc/dx
Fick's second law:         dc/dt = D d^2c/dx^2

Systems provided:
  * semi_infinite_constant_surface  -- fixed surface concentration (erfc)
  * semi_infinite_surface_flux      -- surface flux ~ t^{-1/2} (Fick 1st law)
  * diffusion_uptake                -- cumulative uptake ~ t^{1/2} (parabolic)
  * instantaneous_point_source      -- thin-film / Gaussian spreading
  * finite_slab                     -- slab held at c_s on both faces (series)
  * msd_gaussian                    -- <x^2> = 2 D t  (Fickian signature)
"""
import jax.numpy as jnp
from jax.scipy.special import erfc


def semi_infinite_constant_surface(x, t, D, cs=1.0):
    """Semi-infinite solid, surface held at c_s, initially 0.
    c(x,t) = c_s * erfc( x / (2 sqrt(D t)) ).  (Diffusion couple, carburizing.)"""
    return cs * erfc(x / (2.0 * jnp.sqrt(D * t)))


def semi_infinite_surface_flux(t, D, cs=1.0):
    """Fick's-first-law flux into the surface: J(0,t) = c_s sqrt(D/(pi t))."""
    return cs * jnp.sqrt(D / (jnp.pi * t))


def diffusion_uptake(t, D, cs=1.0):
    """Cumulative amount absorbed = integral_0^t J(0,tau) dtau
       = 2 c_s sqrt(D t / pi)  ~  t^{1/2}  (the parabolic law)."""
    return 2.0 * cs * jnp.sqrt(D * t / jnp.pi)


def instantaneous_point_source(x, t, D, M=1.0):
    """Instantaneous planar source of strength M at x=0 (thin-film couple):
    c(x,t) = M / sqrt(4 pi D t) * exp(-x^2 / (4 D t))  (Gaussian)."""
    return M / jnp.sqrt(4.0 * jnp.pi * D * t) * jnp.exp(-x ** 2 / (4.0 * D * t))


def finite_slab(x, t, D, L, cs=1.0, nterms=80):
    """Slab 0<=x<=L, both faces held at c_s, initially 0 (Fourier series):
    c = c_s [ 1 - (4/pi) sum_{m odd} (1/m) sin(m pi x/L) exp(-D (m pi/L)^2 t) ]."""
    m = 2.0 * jnp.arange(nterms) + 1.0
    series = ((1.0 / m)[:, None]
              * jnp.sin(m[:, None] * jnp.pi * x[None, :] / L)
              * jnp.exp(-D * (m[:, None] * jnp.pi / L) ** 2 * t))
    return cs * (1.0 - (4.0 / jnp.pi) * series.sum(0))


def msd_gaussian(t, D):
    """Mean-squared displacement for Fickian diffusion: <x^2> = 2 D t (alpha=1).
    Anomalous diffusion replaces this by <x^2> ~ t^alpha with alpha != 1."""
    return 2.0 * D * t
