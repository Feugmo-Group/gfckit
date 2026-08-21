"""Memory-kernel functions (pure JAX)."""
import jax.numpy as jnp
from .operators import gamma


def tempered_powerlaw_kernel(tau, alpha, lam):
    """K(tau) = tau^{-alpha} e^{-lam tau} / Gamma(1-alpha).

    Singular at tau=0, completely monotone, finite memory horizon 1/lam.
    The physically sensible memory for SEI growth / corrosion.
    """
    return tau ** (-alpha) * jnp.exp(-lam * tau) / gamma(1.0 - alpha)


def expsum_kernel(tau, a, r):
    """K(tau) = sum_m a_m e^{-r_m tau}, a_m>=0 (completely monotone basis)."""
    return (a[None, :] * jnp.exp(-r[None, :] * tau[:, None])).sum(1)
