"""Core general-fractional operators (pure JAX, no numpy/scipy).

All operators here are the discrete building blocks described in
`docs/gfc_degradation_identification.pdf`:
  * caputo_L1                 -- ordinary Caputo derivative, L1 scheme
  * tempered_powerlaw_L1_weights, expsum_L1_weights -- kernel L1 weights
  * apply_gfd                 -- general fractional derivative given weights
  * substitution_gfd          -- Tarasov parametric GFD: warp -> apply -> unwarp
"""
import jax.numpy as jnp
from jax.scipy.special import gammaln, gammainc


def gamma(x):
    """Gamma function for positive x via exp(gammaln)."""
    return jnp.exp(gammaln(x))


def caputo_L1(u, ds, alpha):
    """Caputo derivative ^C D^alpha u on a uniform grid (spacing ds), L1 scheme.

    ^C D^a u(t_n) ~= 1/(Gamma(2-a) ds^a) sum_k b_{n-k}(u_k-u_{k-1}),
    b_j = (j+1)^{1-a} - j^{1-a}.  The j=0 term of j^{1-a} is forced to 0 so the
    scheme is also correct at a=1 (jax evaluates 0**0=1, which would zero b_0).
    """
    N = u.shape[0]
    j = jnp.arange(N)
    jpow = jnp.where(j == 0, 0.0, j ** (1.0 - alpha))
    b = (j + 1.0) ** (1.0 - alpha) - jpow
    du = jnp.diff(u, prepend=u[0])
    d = jnp.convolve(du, b)[:N]                     # O(N) memory causal convolution
    return d / (gamma(2.0 - alpha) * ds ** alpha)


def expsum_L1_weights(a, r, n_lag, ds):
    """L1 weights of the completely-monotone kernel K(tau)=sum_m a_m e^{-r_m tau}.

    w_j = (1/ds) integral_{j ds}^{(j+1) ds} K(tau) dtau  (exact for exp sums).
    a : (M,) non-negative weights ; r : (M,) rates.
    """
    j = jnp.arange(n_lag)
    lo = j[:, None] * ds
    hi = (j[:, None] + 1.0) * ds
    integ = (jnp.exp(-r[None, :] * lo) - jnp.exp(-r[None, :] * hi)) / r[None, :]
    return (integ @ a) / ds


def tempered_powerlaw_L1_weights(alpha, lam, n_lag, ds):
    """L1 weights of the tempered power law K(tau)=tau^{-a} e^{-lam tau}/Gamma(1-a).

    Uses  integral_0^y tau^{-a} e^{-lam tau} dtau = lam^{a-1} Gamma(1-a) P(1-a, lam y)
    (the Gamma(1-a) cancels the kernel prefactor), P = regularized lower gamma.
    """
    j = jnp.arange(n_lag + 1)
    y = j * ds
    P = gammainc(1.0 - alpha, lam * y)
    cum = lam ** (alpha - 1.0) * P
    return jnp.diff(cum) / ds


def apply_gfd(v, w):
    """Discrete general fractional derivative (K*v')(s_n) given L1 weights w.

    (K*v')(s_n) ~= sum_{j>=0} w_j (v_{n-j} - v_{n-j-1}) -- causal, pure jax.
    """
    N = v.shape[0]
    dv = jnp.diff(v, prepend=v[0])
    return jnp.convolve(dv, w)[:N]                  # O(N) memory causal convolution


def substitution_gfd(u, t, g_of_t, w):
    """Tarasov parametric GFD of u w.r.t. clock g: warp -> apply GFD -> unwarp.

    Implements  D^{(K),*}_{+,g} u = Q_g ( D^{(K),*} Q_g^{-1} u ).
    g_of_t : samples of the internal clock g(t) on the physical grid t.
    """
    N = u.shape[0]
    s_uniform = jnp.linspace(g_of_t[0], g_of_t[-1], N)
    v = jnp.interp(s_uniform, g_of_t, u)          # u in (uniform) warped time
    d_warp = apply_gfd(v, w)                       # ordinary GFD in warped time
    return jnp.interp(g_of_t, s_uniform, d_warp)   # warp back to t
