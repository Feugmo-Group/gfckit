"""Does the identifiability boundary bite on real cells?

Section 3.6 derives, on synthetic data, that ~3.1 decades of observation are
needed to resolve 90% of a well-separated order range.  That is a statement
about experiments, so it can be checked against real ones without knowing any
cell's ground truth.

Two independent public datasets are used:
  * Oxford Battery Degradation Dataset 1 -- 8 cells, periodic pseudo-OCV checkups
  * Zhang et al. 2020 (Zenodo 3633835)   -- 12 cells, per-cycle capacity, 3 temps

For each cell we measure two things that need no inverse problem at all:
  1. the DYNAMIC RANGE actually spanned, in decades of cycle number, from the
     first usable measurement to the last -- the quantity Sec 3.6 puts a
     threshold on;
  2. the EFFECTIVE ORDER alpha from a single power-law fit to the capacity loss,
     which Sec 3.2 says is the one thing robustly recoverable.

Deliberately NOT done here: fitting a free order spectrum to real curves.  On
these data the spectrum objective lands below the nested single-power-law
special case on 6 of 8 Oxford cells, and Sec 3.5's own rule is that a richer
model fitting worse than a model it contains diagnoses optimizer failure rather
than ill-posedness, so no inference may be drawn from it.  The dynamic-range
statement above is stronger anyway, and needs no optimizer.

Writes figures/real_window.png.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from gfckit.identify import powerlaw_reconstruction
from gfckit.plotting import plot_real_window, mirror_figure
from gfckit.real_data import load_oxford, load_zhang_capacity

jax.config.update("jax_enable_x64", True)

D_REQUIRED = 3.1                 # decades, from Sec 3.6 (90% of the order range)
MIN_POINTS = 20                  # a cell needs enough checkups to fit at all


def _fit(cycle, cap):
    """Decades spanned and the effective order of the capacity-loss curve."""
    cycle, cap = np.asarray(cycle, float), np.asarray(cap, float)
    keep = cycle > 0
    cycle, cap = cycle[keep], cap[keep]
    loss = cap[0] - cap                       # fade measured from the first point
    keep = loss > 0                           # log-log needs strictly positive
    cycle, loss = cycle[keep], loss[keep]
    if len(cycle) < MIN_POINTS:
        return None
    decades = float(np.log10(cycle[-1] / cycle[0]))
    a, r2 = powerlaw_reconstruction(jnp.asarray(cycle), jnp.asarray(loss))
    return decades, float(a), float(r2)


def main():
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(HERE, "figures", "real_window.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    root = os.path.join(HERE, "data", "real")

    rows = []
    ox = load_oxford(os.path.join(root, "oxford", "oxford.mat"), n_dva=2)
    for name, v in sorted(ox.items()):
        r = _fit(v["cycle"], v["capacity_mAh"])
        if r:
            rows.append(("Oxford", name, *r))

    zh = load_zhang_capacity(os.path.join(root, "zhang2020"))
    for (temp, cell), (cyc, cap) in sorted(zh.items()):
        r = _fit(cyc, cap)
        if r:
            rows.append(("Zhang", f"{temp}C-{cell}", *r))

    print(f"{'dataset':<9}{'cell':<10}{'decades':>9}{'alpha':>8}{'R2':>9}")
    for _ds, _name, _dec, _a, _r2 in rows:
        print(f"{_ds:<9}{_name:<10}{_dec:>9.2f}{_a:>8.3f}{_r2:>9.4f}")

    ds = [r[0] for r in rows]
    dec = np.array([r[2] for r in rows])
    al = np.array([r[3] for r in rows])
    r2 = np.array([r[4] for r in rows])

    # A fit that lands on the edge of the search grid has not identified an
    # order, it has run out of room -- exclude it rather than quote the bound.
    pinned = (np.abs(al - al.min()) < 1e-6) | (np.abs(al - 1.6) < 1e-6)
    good = (r2 > 0.95) & ~pinned

    print(f"\n{len(rows)} cells across 2 independent datasets")
    print(f"  dynamic range: {dec.min():.2f} - {dec.max():.2f} decades "
          f"(median {np.median(dec):.2f})")
    print(f"  cells reaching the {D_REQUIRED} decades Sec 3.6 requires: "
          f"{int((dec >= D_REQUIRED).sum())} of {len(dec)}")
    print(f"  shortfall of the best cell: "
          f"{D_REQUIRED - dec.max():.2f} decades")
    print(f"  discarded fits: {int(pinned.sum())} pinned to a grid bound, "
          f"{int(((r2 <= 0.95) & ~pinned).sum())} with R^2 <= 0.95")
    print(f"  effective order (usable fits, n={int(good.sum())}): "
          f"{al[good].min():.2f} - {al[good].max():.2f}, "
          f"median {np.median(al[good]):.2f}")
    print(f"  their fit quality: median R^2 = {np.median(r2[good]):.4f}")
    for lbl in ("Oxford", "Zhang"):
        m = np.array([d == lbl for d in ds]) & good
        if m.any():
            print(f"    {lbl}: n={int(m.sum())}, alpha "
                  f"{al[m].min():.2f}-{al[m].max():.2f}, "
                  f"median R^2 {np.median(r2[m]):.4f}")

    rows = [r + (bool(g),) for r, g in zip(rows, good)]

    plot_real_window(rows, D_REQUIRED, out)
    # keep the manuscript figure directory in step with its generator; the two
    # used to be reconciled by hand
    mirror_figure(out, os.path.join(HERE, os.pardir, "manuscripts",
                                    "gfc_identifiability", "figures"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
