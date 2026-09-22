"""What high-temperature MD screening gets wrong when it extrapolates to 300 K.

The standard computational-screening protocol for solid electrolytes is: run MD
at several elevated temperatures where the ions actually move, fit D = MSD/6t at
each, fit ln D against 1/T, and extrapolate to room temperature.

That protocol assumes one Arrhenius law holds over the whole range. It does not,
if the transport is subdiffusive below some crossover T*, because D itself stops
being well defined: for MSD ~ t^beta with beta < 1 the quantity MSD/6t depends on
how long you ran. Extrapolating from above T* to below it therefore crosses a
change in the transport mechanism, not just a change in rate.

This script measures the size of that error directly. It fits Arrhenius to the
high-temperature runs only -- the ones a screening study would have -- and
compares the extrapolation against the D actually measured at 300 K.

Usage:
    python examples/fracmd_arrhenius_bias.py [composition]
"""
import os
import sys

import numpy as np

from fracmd_msd import load_tiers
from fracmd_alpha_T import collect, KB_EV, ROOT

FS_TO_S = 1e-15
A2_TO_CM2 = 1e-16


def apparent_D(run):
    """D = MSD/6t at the end of the window, in cm^2/s.

    This is what a screening study reports. Below T* it is not a diffusion
    coefficient at all, only the number that formula returns for this run
    length -- which is the point.
    """
    t, m = load_tiers(run)
    return m[-1] * A2_TO_CM2 / (6.0 * t[-1] * FS_TO_S)


def collect_D(comp):
    base = os.path.join(ROOT, comp)
    out = {}
    for tdir in sorted(os.listdir(base)):
        if not tdir.startswith("T"):
            continue
        T = int(tdir[1:])
        for rdir in sorted(os.listdir(os.path.join(base, tdir))):
            try:
                out.setdefault(T, []).append(
                    apparent_D(os.path.join(base, tdir, rdir)))
            except (FileNotFoundError, ValueError):
                pass
    return out


def main(comp="67Li2S-33P2S5"):
    D = collect_D(comp)
    B = collect(comp)
    if len(D) < 3:
        print(f"{comp}: need at least three temperatures")
        return
    print(f"{comp}\n")
    print("   T(K)   D = MSD/6t (cm^2/s)   beta")
    for T in sorted(D):
        b = np.mean(B[T]) if T in B else np.nan
        print(f"  {T:5d}   {np.mean(D[T]):.3e}            {b:.3f}")

    # E_0 and T* from the coldest run, as in fracmd_alpha_T
    T_lo = min(B)
    E0 = KB_EV * T_lo / float(np.mean(B[T_lo]))
    Tstar = E0 / KB_EV
    hot = sorted(T for T in D if T > Tstar)
    if len(hot) < 2 or 300 not in D:
        print("\n  need >=2 temperatures above T* and a 300 K reference")
        return

    print(f"\n  T* = {Tstar:.0f} K, so a screening study using only T > T* has "
          f"{hot}")
    x = np.array([1.0 / T for T in hot])
    y = np.array([np.log(np.mean(D[T])) for T in hot])
    slope, intercept = np.polyfit(x, y, 1)
    Ea = -slope * KB_EV
    D300_pred = np.exp(slope / 300.0 + intercept)
    D300_true = float(np.mean(D[300]))
    ratio = D300_pred / D300_true

    print(f"  Arrhenius fit to those alone: Ea = {Ea*1000:.0f} meV")
    print()
    print(f"  extrapolated D(300 K) : {D300_pred:.3e} cm^2/s")
    print(f"  measured     D(300 K) : {D300_true:.3e} cm^2/s")
    print(f"  overestimate          : {ratio:.2f}x")
    print()
    print("  The extrapolation is optimistic because it assumes the transport")
    print("  that produced the high-temperature points continues below T*, where")
    print("  in fact beta falls from ~1 to "
          f"{float(np.mean(B[300])):.2f} and MSD/6t is no longer a rate constant.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "67Li2S-33P2S5")
