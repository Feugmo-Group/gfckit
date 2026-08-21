"""Mechanistic forward models for degradation, solved with Jacobian-free
Newton-Krylov (pure JAX).

These generate synthetic aging data from INDEPENDENT physics (not from the
fractional kernel), so feeding them to `gfckit.identify` is a genuine
cross-model test: does the memory-kernel identification recover a physically
meaningful exponent / memory horizon?

  * newton_krylov               -- JFNK solver (GMRES + autodiff JVPs)
  * simulate_sei_reaction_diffusion -- moving-boundary reaction-diffusion SEI
                                       growth (Landau transform); implicit.
  * simulate_pdm_film_growth    -- Point Defect Model high-field oxide growth
                                       vs. dissolution; implicit, stiff.
"""
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------
# Jacobian-Free Newton-Krylov: solve F(x)=0.
# Each Newton step solves J dx = -F(x) with GMRES, where the Jacobian-vector
# product J v = d/de F(x+e v) is supplied matrix-free by forward-mode autodiff
# (jax.jvp).  No Jacobian is ever assembled.
# --------------------------------------------------------------------------
def newton_krylov(F, x0, tol=1e-9, maxiter=60):
    def jvp_op(x):
        return lambda v: jax.jvp(F, (x,), (v,))[1]

    def cond(state):
        x, res, it = state
        return (res > tol) & (it < maxiter)

    def body(state):
        x, res, it = state
        Fx = F(x)
        dx, _ = gmres(jvp_op(x), -Fx, tol=1e-8, atol=1e-12, maxiter=200)
        x = x + dx
        return x, jnp.linalg.norm(F(x)), it + 1

    x0 = jnp.asarray(x0, dtype=jnp.float64)
    state = (x0, jnp.linalg.norm(F(x0)), 0)
    x, _, _ = jax.lax.while_loop(cond, body, state)
    return x


# --------------------------------------------------------------------------
# SEI growth: moving-boundary reaction-diffusion (capacity fade ~ SEI thickness)
#
# Solvent diffuses through the SEI film [0, L(t)] and reacts at the electrode
# (x=0), forming SEI and consuming Li.  Landau transform xi=x/L(t) maps to a
# fixed grid; unknowns per step are the concentration profile u_0..u_n and L.
# Reaction-limited early (L~t), diffusion-limited late (L~sqrt(t)).
# --------------------------------------------------------------------------
def simulate_sei_reaction_diffusion(D=1.0, k=1.0, Om=1.0, cb=1.0,
                                    L0=0.05, n=20, dt=0.02, nsteps=400):
    h = 1.0 / n
    xi = jnp.arange(n + 1) * h

    def residual(X, Xold):
        u, L = X[:n + 1], X[n + 1]
        uold, Lold = Xold[:n + 1], Xold[n + 1]
        Lp = (L - Lold) / dt
        lap = (u[2:] - 2 * u[1:-1] + u[:-2]) / h ** 2
        adv = xi[1:-1] * Lp / L * (u[2:] - u[:-2]) / (2 * h)
        rint = (u[1:-1] - uold[1:-1]) / dt - (D / L ** 2) * lap - adv
        r0 = D / L * (u[1] - u[0]) / h - k * u[0]          # electrode flux = reaction
        rn = u[n] - cb                                     # electrolyte Dirichlet
        rL = Lp - Om * k * u[0]                            # SEI growth = reaction rate
        return jnp.concatenate([r0[None], rint, rn[None], rL[None]])

    @jax.jit
    def step(Xold):
        return newton_krylov(lambda X: residual(X, Xold), Xold)

    X = jnp.concatenate([jnp.full(n + 1, cb), jnp.array([L0])])
    Ls = [float(L0)]
    for _ in range(nsteps):
        X = step(X)
        Ls.append(float(X[n + 1]))
    t = jnp.arange(nsteps + 1) * dt
    return t, jnp.array(Ls)                                # L(t) ~ capacity loss


# --------------------------------------------------------------------------
# Point Defect Model (corrosion): high-field oxide growth vs. dissolution.
#
# dL/dt = k_g exp(b*dV / L)  -  k_d
# Field E ~ dV/L drives Cabrera-Mott / PDM cation ejection (film formation);
# k_d is chemical dissolution at the film/solution interface.  With k_g < k_d
# the film passivates to a steady thickness L_ss where the two balance.
# Stiff at small L -> solved implicitly (backward Euler + Newton-Krylov).
# --------------------------------------------------------------------------
def simulate_pdm_film_growth(kg=0.5, kd=1.0, bdV=1.0, L0=0.5,
                             dt=0.02, nsteps=400):
    def rate(L):
        return kg * jnp.exp(jnp.minimum(bdV / L, 40.0)) - kd

    def residual(X, Xold):
        L, Lold = X[0], Xold[0]
        return jnp.array([(L - Lold) / dt - rate(L)])

    @jax.jit
    def step(Xold):
        return newton_krylov(lambda X: residual(X, Xold), Xold)

    X = jnp.array([L0])
    Ls = [float(L0)]
    for _ in range(nsteps):
        X = step(X)
        Ls.append(float(X[0]))
    t = jnp.arange(nsteps + 1) * dt
    L_ss = bdV / jnp.log(kd / kg)                          # analytic steady state
    return t, jnp.array(Ls), float(L_ss)
