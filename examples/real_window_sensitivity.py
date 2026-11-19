"""Sensitivity of the real-cell results (Sec. 4.7) to the choices made in
real_window.py.

The cut-offs used there (R^2 > 0.95, exclusion of fits pinned to the search
bound of the order grid, window start at the first strictly positive loss) were
set after the fits were seen.  This script asks whether the conclusions depend
on them:

  1. decades spanned, with the window start defined by a fade threshold
     (first point where the loss exceeds 0, 1, 2 or 5 % of the final loss);
  2. number of surviving order fits, and their alpha range, for R^2 cut-offs
     0.90, 0.95, 0.98, 0.99 and an upper order bound of 1.6 or 2.5;
  3. a residual-bootstrap 95 % interval on alpha for each surviving fit;
  4. the relative RMS residual of each surviving power-law fit, as an
     estimate of the noise floor the real data show.

Writes data/real/real_window_sensitivity.txt.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from gfckit.identify import recover_order_weak
from gfckit.real_data import load_oxford, load_zhang_capacity

jax.config.update("jax_enable_x64", True)

D_REQUIRED = 3.1
MIN_POINTS = 20
N_BOOT = 200


def _series(cycle, cap):
    cycle, cap = np.asarray(cycle, float), np.asarray(cap, float)
    keep = cycle > 0
    cycle, cap = cycle[keep], cap[keep]
    loss = cap[0] - cap
    keep = loss > 0
    return cycle[keep], loss[keep]


def _fit(c, l, amax):
    grid = jnp.linspace(0.05, amax, int(round((amax - 0.05) / 0.00775)) + 1)
    a = float(recover_order_weak(jnp.asarray(l), jnp.asarray(c), grid))
    Phi = np.stack([c ** a, np.ones_like(c)], axis=1)
    coef, *_ = np.linalg.lstsq(Phi, l, rcond=None)
    fit = Phi @ coef
    r2 = 1 - np.sum((l - fit) ** 2) / np.sum((l - l.mean()) ** 2)
    pinned = abs(a - 0.05) < 1e-6 or abs(a - amax) < 1e-6
    return a, float(r2), pinned, fit, grid


def main():
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.join(HERE, "data", "real")
    cells = []
    ox = load_oxford(os.path.join(root, "oxford", "oxford.mat"), n_dva=2)
    for name, v in sorted(ox.items()):
        cells.append(("Oxford", name, *_series(v["cycle"], v["capacity_mAh"])))
    zh = load_zhang_capacity(os.path.join(root, "zhang2020"))
    for (temp, cell), (cyc, cap) in sorted(zh.items()):
        cells.append(("Zhang", f"{temp}C-{cell}", *_series(cyc, cap)))
    cells = [c for c in cells if len(c[2]) >= MIN_POINTS]

    out = []
    p = out.append
    p(f"{len(cells)} cells\n")

    # 1. window start definition
    p("1. Decades spanned vs window-start threshold (fraction of final loss)")
    for thr in (0.0, 0.01, 0.02, 0.05):
        dec = []
        for _, _, c, l in cells:
            k = np.argmax(l > thr * l[-1]) if thr > 0 else 0
            dec.append(np.log10(c[-1] / c[k]))
        dec = np.array(dec)
        p(f"   thr {thr:4.2f}: {dec.min():.2f}-{dec.max():.2f} decades, "
          f"median {np.median(dec):.2f}, reaching {D_REQUIRED}: "
          f"{int((dec >= D_REQUIRED).sum())}/{len(dec)}")

    # 2. cut-off sensitivity
    p("\n2. Surviving order fits vs R^2 cut-off and upper order bound")
    fits = {}
    for amax in (1.6, 2.5):
        fits[amax] = [_fit(c, l, amax) for _, _, c, l in cells]
        for cut in (0.90, 0.95, 0.98, 0.99):
            g = [(cells[i][0], f[0]) for i, f in enumerate(fits[amax])
                 if f[1] > cut and not f[2]]
            if g:
                al = np.array([a for _, a in g])
                nox = sum(d == "Oxford" for d, _ in g)
                p(f"   bound {amax}, R2>{cut}: n={len(g)} (Oxford {nox}, "
                  f"Zhang {len(g) - nox}), alpha {al.min():.2f}-{al.max():.2f}, "
                  f"median {np.median(al):.2f}")
            else:
                p(f"   bound {amax}, R2>{cut}: n=0")

    # 3 + 4. bootstrap intervals and noise floor, at the published settings
    p("\n3. Surviving fits at the published settings (bound 1.6, R2>0.95):")
    p(f"   {'cell':<16}{'decades':>8}{'alpha':>8}{'95% CI':>16}{'R2':>9}"
      f"{'rel RMS':>9}")
    rng = np.random.default_rng(0)
    floors = []
    for (ds, name, c, l), (a, r2, pinned, fit, grid) in zip(cells, fits[1.6]):
        if r2 <= 0.95 or pinned:
            continue
        res = l - fit
        boot = []
        for _ in range(N_BOOT):
            lb = fit + rng.choice(res, size=res.size, replace=True)
            boot.append(float(recover_order_weak(jnp.asarray(lb),
                                                 jnp.asarray(c), grid)))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rel = float(np.sqrt(np.mean(res ** 2)) / np.ptp(l))
        floors.append(rel)
        p(f"   {ds + ' ' + name:<16}{np.log10(c[-1] / c[0]):>8.2f}{a:>8.3f}"
          f"{f'[{lo:.3f}, {hi:.3f}]':>16}{r2:>9.4f}{100 * rel:>8.2f}%")
    p(f"\n4. Relative RMS residual (noise-floor estimate) of surviving fits: "
      f"{100 * min(floors):.2f}-{100 * max(floors):.2f} %, "
      f"median {100 * np.median(floors):.2f} %")

    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(root, "real_window_sensitivity.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
