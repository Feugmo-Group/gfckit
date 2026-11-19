"""Is the published SEI moving-boundary solution converged?

simulate_sei_reaction_diffusion uses Jacobian-free Newton-Krylov with
unpreconditioned GMRES.  The Jacobian of this system has condition number
1e6-1e7, and GMRES stalls: at the published grid (n = 20) Newton stops at a
residual of ~2e-4 instead of 1e-9, and at n = 40 GMRES returns a zero update so
the film never grows.  The loop exits at its iteration cap without signalling.

This script solves the SAME discrete equations with Newton and a dense direct
linear solve (the system has only n + 2 unknowns), checks that each step
converges to 1e-10, refines the grid and the step, and compares

  * L(t) with the published (JFNK, n = 20) curve,
  * the weak-form effective order on t > 1 (published value 0.59),
  * L(t) with the Deal-Grove linear-parabolic law, the quasi-steady limit.
"""
import jax
import jax.numpy as jnp
import numpy as np

from gfckit.identify import recover_order_weak
from gfckit.physics import simulate_sei_reaction_diffusion

jax.config.update("jax_enable_x64", True)
GRID = jnp.linspace(0.3, 1.2, 181)


def sei_direct(D=1.0, k=1.0, Om=1.0, cb=1.0, L0=0.05, n=20, dt=0.02, nsteps=400):
    h = 1.0 / n
    xi = jnp.arange(n + 1) * h

    def residual(X, Xold):
        u, L = X[:n + 1], X[n + 1]
        uold, Lold = Xold[:n + 1], Xold[n + 1]
        Lp = (L - Lold) / dt
        lap = (u[2:] - 2 * u[1:-1] + u[:-2]) / h ** 2
        adv = xi[1:-1] * Lp / L * (u[2:] - u[:-2]) / (2 * h)
        rint = (u[1:-1] - uold[1:-1]) / dt - (D / L ** 2) * lap - adv
        r0 = D / L * (u[1] - u[0]) / h - k * u[0]
        rn = u[n] - cb
        rL = Lp - Om * k * u[0]
        return jnp.concatenate([r0[None], rint, rn[None], rL[None]])

    @jax.jit
    def newton_step(x, xold):
        F = lambda X: residual(X, xold)
        return x - jnp.linalg.solve(jax.jacfwd(F)(x), F(x)), jnp.linalg.norm(F(x))

    @jax.jit
    def res(x, xold):
        return jnp.linalg.norm(residual(x, xold))

    X = jnp.concatenate([jnp.full(n + 1, cb), jnp.array([L0])])
    Ls, worst = [L0], 0.0
    for _ in range(nsteps):
        xold, x = X, X
        for _it in range(50):
            x, _ = newton_step(x, xold)
            r = float(res(x, xold))
            if r < 1e-10:
                break
        worst = max(worst, r)
        X = x
        Ls.append(float(X[n + 1]))
    return np.arange(nsteps + 1) * dt, np.array(Ls), worst


def deal_grove(t, D=1.0, k=1.0, Om=1.0, cb=1.0, L0=0.05):
    a = k / (2 * D)
    c = L0 + a * L0 ** 2 + Om * k * cb * t
    return (-1 + np.sqrt(1 + 4 * a * c)) / (2 * a)


def alpha(t, L):
    m = t > 1.0
    return float(recover_order_weak(jnp.asarray(L[m]), jnp.asarray(t[m]), GRID))


def main():
    t0, L0 = simulate_sei_reaction_diffusion(nsteps=400, dt=0.02)
    t0, L0 = np.asarray(t0), np.asarray(L0)
    print(f"published JFNK n=20 dt=0.02: alpha={alpha(t0, L0):.3f}  L(8)={L0[-1]:.4f}")
    ref = None
    for n, dt, ns in ((20, 0.02, 400), (40, 0.02, 400), (80, 0.02, 400),
                      (80, 0.01, 800), (160, 0.01, 800)):
        t, L, worst = sei_direct(n=n, dt=dt, nsteps=ns)
        dg = np.max(np.abs(L - deal_grove(t))[t >= 1] / deal_grove(t)[t >= 1])
        on0 = np.interp(t0, t, L)
        dpub = np.max(np.abs(L0 - on0) / on0)
        print(f"direct n={n:<4}dt={dt}: worst step residual {worst:.1e}, "
              f"alpha={alpha(t, L):.3f}, L(8)={L[-1]:.4f}, "
              f"max rel diff vs published {dpub:.2e}, vs Deal-Grove (t>=1) {dg:.2e}",
              flush=True)


if __name__ == "__main__" and __import__("sys").argv[-1] != "--slow":
    main()


def slow_growth_limit():
    """Deal-Grove is the quasi-steady limit, reached when the film grows slowly
    compared with diffusion across it (small Om).  The gap should close."""
    print("\nslow-growth limit (direct solver, n=40, same dimensionless time Om*t):")
    for Om in (1.0, 0.3, 0.1, 0.03):
        dt = 0.02 / Om
        t, L, worst = sei_direct(Om=Om, n=40, dt=dt, nsteps=400)
        ref = deal_grove(t, Om=Om)
        m = t * Om >= 1
        print(f"  Om={Om:<5} max rel diff vs Deal-Grove (Om t>=1) "
              f"{np.max(np.abs(L - ref)[m] / ref[m]):.2e}", flush=True)


if __name__ == "__main__" and __import__("sys").argv[-1] == "--slow":
    slow_growth_limit()
