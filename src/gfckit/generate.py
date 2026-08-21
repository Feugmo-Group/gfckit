"""Synthetic ground-truth generation (pure JAX).

Two analytic/semi-analytic families:
  * analytic_powerlaw   -- u(t)=t^alpha, exact soln of ^C D^alpha u = Gamma(a+1)
  * solve_gfode_constant -- forward solve of (K*v')(s)=c for a tempered-PL kernel
  * build_conditions     -- multi-condition data: same kernel, Arrhenius clocks
"""
import jax
import jax.numpy as jnp
from .operators import gamma, tempered_powerlaw_L1_weights


def analytic_powerlaw(t, alpha):
    """u = t^alpha ; exact solution of ^C D^alpha u = Gamma(alpha+1)."""
    return t ** alpha, gamma(alpha + 1.0)


def solve_gfode_constant(alpha, lam, c_sink, s_grid):
    """Solve the general-fractional relaxation (K*v')(s) = c_sink with the
    tempered-power-law kernel, v(0)=0, by one lower-triangular solve.

    W dv = rhs, with W the lower-triangular Toeplitz L1-weight matrix; v=cumsum(dv).
    """
    N = s_grid.shape[0]
    ds = s_grid[1] - s_grid[0]
    w = tempered_powerlaw_L1_weights(alpha, lam, N, ds)
    n = jnp.arange(N)
    lag = n[:, None] - n[None, :]
    W = jnp.where(lag >= 0, w[jnp.clip(lag, 0, N - 1)], 0.0)
    rhs = jnp.full((N,), c_sink).at[0].set(0.0)             # dv_0 = 0
    dv = jax.scipy.linalg.solve_triangular(W, rhs, lower=True)
    return jnp.cumsum(dv)


def build_conditions(alpha, lam, c_sink, rates, N, dt, noise=0.0, seed=0):
    """Multi-condition data: one shared kernel, per-condition Arrhenius clocks.

    Each condition c has a linear clock g_c(t)=k_c t, so u_c(t)=v(k_c t) where v
    solves the GFODE once in warped time.  Returns (t, u_list, ds_list).
    """
    t = jnp.arange(N) * dt
    kmax = float(max(rates))
    s_grid = jnp.arange(N) * (dt * kmax)
    v = solve_gfode_constant(alpha, lam, c_sink, s_grid)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(rates))
    u_list, ds_list = [], []
    for kc, kk in zip(rates, keys):
        u = jnp.interp(kc * t, s_grid, v)
        u = u + noise * float(jnp.std(u)) * jax.random.normal(kk, (N,))
        u_list.append(u)
        ds_list.append(kc * dt)
    return t, u_list, ds_list
