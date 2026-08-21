"""Constructive proof that the fractional-order spectrum is not identifiable.

Section 3.5 shows empirically that a two-order memory collapses to one effective
order.  This script makes it constructive: for the paper's reference spectrum
{0.3,0.8} with weights {0.6,0.4} it finds the FARTHEST spectrum (1-Wasserstein
on the order axis) whose response is still indistinguishable at a given noise
floor, after the free amplitude and offset a real fit absorbs.

The maximal gap is a certificate: two spectra that far apart generate data no
experimenter at that precision can separate.  Sweeping the window width and the
noise floor shows identifiability needs BOTH dynamic range and precision --
neither alone suffices.

Writes figures/keystone_construction.png.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from gfckit.identifiability import (make_response, max_gap, sweep_residuals,
                                    w1_gap)
from gfckit.plotting import plot_keystone, mirror_figure

jax.config.update("jax_enable_x64", True)

ORDERS_REF, W_REF = (0.3, 0.8), (0.6, 0.4)     # the Sec 3.5 example
D_MAIN = 2.0                                    # decades: typical experiment
FLOOR_MAIN = 0.01                               # 1% relative RMS
FLOORS = (0.001, 0.005, 0.01, 0.02)
D_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


def main():
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(HERE, "figures", "keystone_construction.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"reference spectrum {{{ORDERS_REF[0]}, {ORDERS_REF[1]}}} "
          f"weights {{{W_REF[0]}, {W_REF[1]}}}\n")

    # ---- gap surviving at each (window, noise floor) ----------------------
    curves = {f: [] for f in FLOORS}
    best_main = None
    print(f"  {'D (dec)':>9}" + "".join(f"{f'floor {f:.1%}':>14}" for f in FLOORS))
    for D in D_GRID:
        sweep = sweep_residuals(ORDERS_REF, W_REF, D)
        row = []
        for f in FLOORS:
            g, best = max_gap(ORDERS_REF, W_REF, sweep, f)
            curves[f].append(g)
            row.append(g)
            if D == D_MAIN and f == FLOOR_MAIN:
                best_main = best
        print(f"  {D:>9.1f}" + "".join(f"{g:>14.1%}" for g in row))

    a_lo, a_hi, w_lo, resid = best_main
    gap_main = w1_gap(ORDERS_REF, W_REF, (a_lo, a_hi), (w_lo, 1 - w_lo))
    print(f"\nkeystone pair at D={D_MAIN:.0f} decades, {FLOOR_MAIN:.0%} noise floor")
    print(f"  reference : {{{ORDERS_REF[0]}, {ORDERS_REF[1]}}} "
          f"weights {{{W_REF[0]:.2f}, {W_REF[1]:.2f}}}")
    print(f"  twin      : {{{a_lo:.3f}, {a_hi:.3f}}} "
          f"weights {{{w_lo:.2f}, {1 - w_lo:.2f}}}")
    print(f"  residual  : {resid:.2e}  (below the {FLOOR_MAIN:.0%} floor)")
    print(f"  W1 gap    : {gap_main:.4f}  ({gap_main:.1%} of the order axis)")

    mean_ref = sum(a * w for a, w in zip(ORDERS_REF, W_REF))
    mean_twin = a_lo * w_lo + a_hi * (1 - w_lo)
    print(f"  mean order: reference {mean_ref:.4f} vs twin {mean_twin:.4f} "
          f"-- the observable both share")

    # ---- the keystone pair's curves --------------------------------------
    t = jnp.logspace(-D_MAIN / 2, D_MAIN / 2, 300)
    response = make_response(t)
    u_ref = response(ORDERS_REF[0], ORDERS_REF[1], W_REF[0])
    u_twin = response(a_lo, a_hi, w_lo)

    path = plot_keystone(ORDERS_REF, W_REF, (a_lo, a_hi, w_lo), t, u_ref, u_twin,
                         gap_main, FLOOR_MAIN, D_MAIN, np.array(D_GRID),
                         [(f, curves[f]) for f in FLOORS], out)
    # keep the manuscript figure directory in step with its generator; the two
    # used to be reconciled by hand
    mirror_figure(path, os.path.join(HERE, os.pardir, "manuscripts",
                                    "gfc_identifiability", "figures"))
    print(f"\nwrote figure -> {path}")


if __name__ == "__main__":
    main()
