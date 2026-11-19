"""Forward-consistency and optimiser checks for the two-order spectrum benchmark (Sec. 4.5).

Benchmark: orders {0.3, 0.8}, weights {0.6, 0.4}, mu = 0.02, S0 = 0.03, dt = 0.03,
800 steps (2.9 decades).  The fitting grid of spectrum_convergence.py has 25 orders
between 0.1 and 0.95 and contains neither 0.3 nor 0.8.

  1. forward   evaluate the generating spectrum on that grid (nearest node, split
               between neighbouring nodes, with and without refitting mu and S0)
               and on a grid that contains 0.3 and 0.8.  Nearest node gives the
               R^2 = 0.9872 quoted in the paper; on-grid the forward map is exact.
  2. loss      data + ridge loss at the generating spectrum and at the saved
               collapsed fits (needs data/synthetic/spectrum_convergence.npz).
  3. lbfgs     the same objective minimised with L-BFGS-B from the uniform start
               and three random starts, on both grids.
  4. adam      Adam (as spectrum_convergence.fit) on the on-grid grid, 4e4 steps.

usage: python spectrum_forward_check.py [forward] [loss] [lbfgs] [adam]   (default: forward loss)
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize

from gfckit.complex import (forward_multiterm_spectrum, multiterm_fractional_relaxation,
                            spectrum_design)

jax.config.update("jax_enable_x64", True)
DT, NSTEPS, RIDGE = 0.03, 800, 1e-5
GRID25 = np.linspace(0.1, 0.95, 25)
GRIDON = np.arange(0.10, 0.9501, 0.05)          # contains 0.3 and 0.8


def data():
    t, u = multiterm_fractional_relaxation([0.3, 0.8], [0.6, 0.4], lam=0.02, S=0.03,
                                           u0=0.0, dt=DT, nsteps=NSTEPS)
    return t, u


def r2(u, um):
    return float(1 - jnp.sum((u - um) ** 2) / jnp.sum((u - jnp.mean(u)) ** 2))


def model(grid, u):
    G, tri, lag = spectrum_design(jnp.asarray(grid), u.shape[0], DT)
    return lambda w, mu, S: forward_multiterm_spectrum(w, G, tri, lag, mu, S, 0.0)


def truth_weights(grid, how):
    w = np.zeros(len(grid))
    for a, wt in ((0.3, 0.6), (0.8, 0.4)):
        if how == "nearest":
            w[np.argmin(abs(grid - a))] += wt
        elif how == "split":
            k = np.searchsorted(grid, a) - 1
            f = (a - grid[k]) / (grid[k + 1] - grid[k])
            w[k] += wt * (1 - f)
            w[k + 1] += wt * f
        else:                                   # exact: the order is a node
            w[np.argmin(abs(grid - a))] += wt
    return w


def refit(fwd, u, w):
    f = lambda p: np.asarray(fwd(jnp.asarray(w), p[0], p[1]) - u)
    return least_squares(f, [0.02, 0.03]).x


def loss_terms(fwd, u, w, mu, S):
    um = fwd(jnp.asarray(w), mu, S)
    dat = float(jnp.mean((um - u) ** 2) / (jnp.mean(u ** 2) + 1e-12))
    return dat, RIDGE * float(jnp.mean(jnp.diff(jnp.asarray(w)) ** 2)), r2(u, um)


def forward():
    t, u = data()
    print(f"window: {np.log10(t[-1] / t[1]):.2f} decades")
    f25 = model(GRID25, u)
    for how in ("nearest", "split"):
        w = truth_weights(GRID25, how)
        print(f"25-node grid, {how:<8} true mu,S: R2 = {r2(u, f25(jnp.asarray(w), 0.02, 0.03)):.7f}")
        mu, S = refit(f25, u, w)
        print(f"25-node grid, {how:<8} refit mu,S ({mu:.4f},{S:.4f}): "
              f"R2 = {r2(u, f25(jnp.asarray(w), mu, S)):.9f}")
    fon = model(GRIDON, u)
    w = truth_weights(GRIDON, "exact")
    d = float(jnp.max(jnp.abs(u - fon(jnp.asarray(w), 0.02, 0.03))))
    print(f"grid with 0.3 and 0.8: max |u - u_model| = {d:.1e}")


def loss():
    t, u = data()
    f25 = model(GRID25, u)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "synthetic", "spectrum_convergence.npz")
    w = truth_weights(GRID25, "split")
    mu, S = refit(f25, u, w)
    dat, pen, rr = loss_terms(f25, u, w, mu, S)
    print(f"generating spectrum (split): data {dat:.2e} + ridge {pen:.2e} = {dat + pen:.2e}, R2 {rr:.9f}")
    if os.path.exists(path):
        d = np.load(path)
        for k in ("uniform", "seed1"):
            wc = d[f"multiterm__{k}__w"]
            mu, S = refit(f25, u, wc)
            dat, pen, rr = loss_terms(f25, u, wc, mu, S)
            print(f"collapsed fit ({k}): mean order {float((wc * GRID25).sum()):.3f}, "
                  f"data {dat:.2e} + ridge {pen:.2e} = {dat + pen:.2e}, R2 {rr:.6f}")


def lbfgs():
    t, u = data()
    for gname, grid in (("25-node", GRID25), ("on-grid", GRIDON)):
        fwd = model(grid, u)
        J = len(grid)

        def obj(x):
            w = jax.nn.softmax(x[:J])
            um = fwd(w, jax.nn.softplus(x[J]), x[J + 1])
            return (jnp.mean((um - u) ** 2) / (jnp.mean(u ** 2) + 1e-12)
                    + RIDGE * jnp.mean(jnp.diff(w) ** 2))

        vg = jax.jit(jax.value_and_grad(obj))
        starts = {"uniform": np.zeros(J)}
        for s in (1, 2, 3):
            starts[f"seed{s}"] = np.asarray(jax.random.normal(jax.random.PRNGKey(s), (J,)))
        for name, th in starts.items():
            x0 = np.concatenate([th, [-2.0, 1.0]])
            res = minimize(lambda x: tuple(np.asarray(v, dtype=float) for v in vg(jnp.asarray(x))),
                           x0, jac=True, method="L-BFGS-B", options={"maxiter": 20000, "maxfun": 40000})
            w = np.asarray(jax.nn.softmax(res.x[:J]))
            um = fwd(jnp.asarray(w), float(jax.nn.softplus(res.x[J])), res.x[J + 1])
            top = [(round(float(a), 2), round(float(b), 3)) for a, b in zip(grid, w) if b > 0.05]
            print(f"L-BFGS {gname} {name}: iters {res.nit}, loss {res.fun:.2e}, R2 {r2(u, um):.8f}, "
                  f"mean {float((w * grid).sum()):.3f}, modes {top}", flush=True)


def adam():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import spectrum_convergence as sc
    sc.STEPS = 40000
    t, u = data()
    sc.GRID = jnp.asarray(GRIDON)
    for name, th in [("uniform", jnp.zeros(len(GRIDON)))] + \
            [(f"seed{s}", jax.random.normal(jax.random.PRNGKey(s), (len(GRIDON),))) for s in (1, 2, 3)]:
        w, h = sc.fit(t, u, th)
        top = [(round(float(a), 2), round(float(b), 3)) for a, b in zip(GRIDON, w) if b > 0.05]
        print(f"Adam on-grid {name}: R2@40k {h[-1]:.8f}, mean {float((w * GRIDON).sum()):.3f}, modes {top}",
              flush=True)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["forward", "loss"]
    for name in todo:
        print(f"\n== {name} ==")
        globals()[name]()
