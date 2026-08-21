"""Complex, NON-power-law synthetic dynamics (pure JAX).

A single fractional order alpha (a pure power law) is the trivial case.  The
novelty of *general* fractional calculus is memory that is NOT a single power
law.  This module generates three such families, each with known ground truth:

  * multiterm_fractional_relaxation -- distributed / multi-order memory:
        sum_i w_i ^C D^{alpha_i} u = -lambda u + S.
    A superposition of power laws => multi-scaling, no single exponent fits.

  * variable_order_solution -- TIME-VARYING order alpha(t): the anomalous
    exponent itself drifts (mechanism crossover, e.g. reaction -> diffusion ->
    plating).  Locally power-law, globally not.

  * knee_capacity_fade -- realistic multi-mechanism battery fade with a "knee":
    gradual sqrt-t SEI growth plus an accelerating plating term that switches on.

All are stepped with a linear/implicit L1 scheme (the relaxation is linear, so
the per-step solve is exact); ground truth is the order spectrum / order-path /
mechanism parameters.
"""
import jax
import jax.numpy as jnp
from .operators import gamma


def _b_weights(alpha, N):
    """L1 history weights b_j=(j+1)^{1-a}-j^{1-a} (j=0 term of j^{1-a} forced 0)."""
    j = jnp.arange(N)
    return (j + 1.0) ** (1.0 - alpha) - jnp.where(j == 0, 0.0, j ** (1.0 - alpha))


# --------------------------------------------------------------------------
# Multi-term (distributed-order) fractional relaxation.
#   sum_i w_i ^C D^{alpha_i} u(t) = -lambda u(t) + S,   u(0)=u0.
# Ground truth: the spectrum {(alpha_i, w_i)}, plus (lambda, S).
# --------------------------------------------------------------------------
def multiterm_fractional_relaxation(alphas, weights, lam, S, u0, dt, nsteps):
    # Linear => solve the whole history at once (lower-triangular Toeplitz solve).
    N = nsteps + 1
    W = sum((jnp.asarray(w) / (gamma(2.0 - a) * dt ** a)) * _b_weights(a, N)
            for a, w in zip(alphas, weights))          # combined L1 weight vector
    n = jnp.arange(N)
    lag = n[:, None] - n[None, :]
    tri = lag >= 0
    M = jnp.where(tri, W[jnp.clip(lag, 0, N - 1)], 0.0) + lam * tri
    rhs = jnp.full((N,), S - lam * u0).at[0].set(0.0)   # du_0 = 0
    du = jax.scipy.linalg.solve_triangular(M, rhs, lower=True)
    return n * dt, u0 + jnp.cumsum(du)


# --------------------------------------------------------------------------
# Variable-order fractional relaxation: alpha = alpha_of_t(t) drifts in time.
#   ^C D^{alpha(t)} u(t) = -lambda u + S,  u(0)=u0.
# Ground truth: the order path alpha(t).
# --------------------------------------------------------------------------
def variable_order_solution(alpha_of_t, lam, S, u0, dt, nsteps):
    # Variable order => each row n uses alpha(t_n); build M row-wise, one solve.
    N = nsteps + 1
    n = jnp.arange(N)
    alpha = alpha_of_t(n * dt)                          # (N,) must accept an array
    lag = n[:, None] - n[None, :]
    tri = lag >= 0
    lagc = jnp.clip(lag, 0, N - 1) * 1.0
    ex = 1.0 - alpha[:, None]                           # per-row exponent
    bw = (lagc + 1.0) ** ex - jnp.where(lagc == 0.0, 0.0, lagc ** ex)
    rho = 1.0 / (gamma(2.0 - alpha) * dt ** alpha)      # (N,)
    M = jnp.where(tri, rho[:, None] * bw, 0.0) + lam * tri
    rhs = jnp.full((N,), S - lam * u0).at[0].set(0.0)
    du = jax.scipy.linalg.solve_triangular(M, rhs, lower=True)
    return n * dt, u0 + jnp.cumsum(du)


# --------------------------------------------------------------------------
# Realistic multi-mechanism capacity fade with a "knee".
#   loss(t) = A_sei sqrt(t)              (gradual SEI, diffusion-limited)
#           + B_plate softplus((t-t_knee)/w)^2   (accelerating plating onset)
# The knee (sudden acceleration) is a central, hard phenomenon in battery aging
# and is emphatically not a single power law.
# --------------------------------------------------------------------------
def knee_capacity_fade(t, A_sei=0.6, B_plate=0.004, t_knee=300.0, w_knee=40.0):
    sei = A_sei * jnp.sqrt(t)
    z = (t - t_knee) / w_knee
    plate = B_plate * jnp.logaddexp(z, 0.0) ** 2          # softplus(z)^2
    return sei + plate


def forward_multiterm_spectrum(w, G, tri, lag_idx, lam, S, u0):
    """Differentiable forward solve of  (sum_j w_j ^C D^{alpha_j}) u + lam u = S.
    G[j] = rho_j * b^{(alpha_j)} (precomputed); returns u on the grid."""
    N = tri.shape[0]
    W = w @ G                                            # combined L1 weights
    M = jnp.where(tri, W[lag_idx], 0.0) + lam * tri
    rhs = jnp.full((N,), S - lam * u0).at[0].set(0.0)
    du = jax.scipy.linalg.solve_triangular(M, rhs, lower=True)
    return u0 + jnp.cumsum(du)


def spectrum_design(order_grid, N, dt):
    """Precompute G (J,N), the lower-triangular mask, and the lag gather index."""
    G = jnp.stack([(1.0 / (gamma(2.0 - a) * dt ** a)) * _b_weights(a, N)
                   for a in order_grid])
    n = jnp.arange(N)
    lag = n[:, None] - n[None, :]
    return G, (lag >= 0), jnp.clip(lag, 0, N - 1)


def build_multiterm_conditions(alphas, weights, lam, S, rates, N, dt,
                               noise=0.0, seed=0):
    """Multi-condition data from ONE shared multi-term spectrum seen through
    Arrhenius clocks g_c(t)=k_c t: u_c(t)=v(k_c t), v solved once in warped time.
    Returns (t, [u_c], s_grid, v)."""
    import jax
    kmax = float(max(rates))
    dt_warp = dt * kmax
    s_grid, v = multiterm_fractional_relaxation(alphas, weights, lam, S, 0.0,
                                                dt_warp, N - 1)
    t = jnp.arange(N) * dt
    keys = jax.random.split(jax.random.PRNGKey(seed), len(rates))
    u_list = []
    for kc, kk in zip(rates, keys):
        u = jnp.interp(kc * t, s_grid, v)
        if noise > 0:
            u = u + noise * float(jnp.std(u)) * jax.random.normal(kk, (N,))
        u_list.append(u)
    return t, u_list, s_grid, v


def local_loglog_slope(t, y):
    """Effective local exponent d(ln y)/d(ln t).  Constant <=> pure power law."""
    lt, ly = jnp.log(t[1:]), jnp.log(jnp.clip(y[1:], 1e-12))
    slope = jnp.gradient(ly, lt)
    return t[1:], slope
