"""Convergence and initialisation check for the order-spectrum fits (Sec. 4.5).

recover_order_spectrum starts from a uniform spectrum and runs a fixed number of
Adam steps with no stopping rule.  This script reruns the same objective on the
paper's two benchmarks (the {0.3,0.8} two-order relaxation and the PDM
passivation curve) and records

  * the reconstruction R^2 every 2,000 steps up to 80,000, and
  * the result from four starting points: the uniform default and three random
    spectra (seeds 1-3), each with the same optimiser settings.

The data are noise-free, as in the paper, so there is no noise seed.  Writes
data/synthetic/spectrum_convergence.npz and prints a summary.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from gfckit.complex import (forward_multiterm_spectrum, multiterm_fractional_relaxation,
                            spectrum_design)
from gfckit.physics import simulate_pdm_film_growth

jax.config.update("jax_enable_x64", True)   # as in identify_nonpowerlaw.py

STEPS, EVERY, LR, RIDGE = 80000, 2000, 0.06, 1e-5
GRID = jnp.linspace(0.1, 0.95, 25)


def r2(u, um):
    return float(1 - jnp.sum((u - um) ** 2) / jnp.sum((u - jnp.mean(u)) ** 2))


def fit(t, u, theta0):
    N, dt, u0 = u.shape[0], float(t[1] - t[0]), u[0]
    G, tri, lag = spectrum_design(GRID, N, dt)

    def model(p):
        w = jax.nn.softmax(p["theta"])
        return w, forward_multiterm_spectrum(w, G, tri, lag, jax.nn.softplus(p["lam"]),
                                             p["S"], u0)

    def loss(p):
        w, um = model(p)
        return jnp.mean((um - u) ** 2) / (jnp.mean(u ** 2) + 1e-12) \
            + RIDGE * jnp.mean(jnp.diff(w) ** 2)

    b1, b2, eps = 0.9, 0.999, 1e-8

    @jax.jit
    def step(p, m, v, i):
        g = jax.grad(loss)(p)
        lr_i = LR * (1.0 - 0.9 * i / STEPS)
        m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
        p = jax.tree_util.tree_map(
            lambda pp, mm, vv: pp - lr_i * (mm / (1 - b1 ** i))
            / (jnp.sqrt(vv / (1 - b2 ** i)) + eps), p, m, v)
        return p, m, v

    p = {"theta": theta0, "lam": jnp.array(-2.0), "S": jnp.array(1.0)}
    m = jax.tree_util.tree_map(jnp.zeros_like, p)
    v = jax.tree_util.tree_map(jnp.zeros_like, p)
    hist = []
    for i in range(1, STEPS + 1):
        p, m, v = step(p, m, v, float(i))
        if i % EVERY == 0:
            hist.append(r2(u, model(p)[1]))
    w, um = model(p)
    return np.asarray(w), np.array(hist)


def summary(w):
    w = np.asarray(w)
    mean = float(np.sum(w * np.asarray(GRID)))
    top = np.argsort(w)[::-1][:3]
    return mean, float(GRID[np.argmax(w)]), float(w[top].sum())


def main():
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cases = {}
    t, u = multiterm_fractional_relaxation([0.3, 0.8], [0.6, 0.4], lam=0.02, S=0.03,
                                           u0=0.0, dt=0.03, nsteps=800)
    cases["multiterm"] = (t, u)
    t, u, _ = simulate_pdm_film_growth(nsteps=700, dt=0.02)
    cases["pdm_passivation"] = (t, u)

    inits = {"uniform": jnp.zeros(GRID.shape[0])}
    for s in (1, 2, 3):
        inits[f"seed{s}"] = jax.random.normal(jax.random.PRNGKey(s), (GRID.shape[0],))

    store = {"steps": np.arange(EVERY, STEPS + 1, EVERY), "grid": np.asarray(GRID)}
    for cname, (t, u) in cases.items():
        print(f"\n{cname}")
        print(f"  {'init':<8}{'R2@5k':>10}{'R2@20k':>10}{'R2@50k':>10}{'R2@80k':>10}"
              f"{'mean a':>8}{'peak a':>8}{'top3':>8}")
        for iname, th in inits.items():
            w, h = fit(t, u, th)
            store[f"{cname}__{iname}__w"] = w
            store[f"{cname}__{iname}__r2"] = h
            mean, peak, top3 = summary(w)
            at = lambda k: h[k // EVERY - 1]
            print(f"  {iname:<8}{at(6000):>10.5f}{at(20000):>10.5f}{at(50000):>10.5f}"
                  f"{at(80000):>10.5f}{mean:>8.3f}{peak:>8.3f}{top3:>8.4f}", flush=True)
    out = os.path.join(HERE, "data", "synthetic", "spectrum_convergence.npz")
    np.savez(out, **store)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
