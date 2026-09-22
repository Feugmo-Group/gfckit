"""Sec 3.5: can a distributed order spectrum be resolved from a 7.3-decade MSD?

The companion identifiability paper shows that recovering a distributed order
spectrum from a 1-3 decade window is hopeless -- the objective actively prefers a
single collapsed order. An MD trajectory spans 7.3 decades, so this is the test
from the resolvable side: does the preference reverse once the window is wide
enough?

Two deliberate departures from how the companion paper does it.

First, the observable. gfckit's recover_order_spectrum fits a relaxation
equation, (sum_j w_j D^{a_j}) u + lambda u = S. An MSD is not that. The natural
distributed-order model for an MSD is a nonnegative superposition of power laws,

    MSD(t) = sum_j w_j t^{a_j},   w_j >= 0,

so that is what is fitted here. Using the relaxation machinery on an MSD would
be a category error dressed up as an identifiability result.

Second, the solver. That superposition is linear in w, so with a nonnegativity
constraint it is a convex NNLS problem with a unique optimum. This matters: on
real battery cells the companion paper's gradient-based spectrum fit landed
BELOW its own nested single-power-law special case on 6 of 8 cells, which by its
own diagnostic means optimizer failure and licenses no inference either way. NNLS
cannot fail that way, so a collapse here is a statement about the data rather
than about the optimizer.

The MSD carries physical crossovers -- ballistic, caged, subdiffusive, Fickian --
that are not a distributed memory. Fitting the whole window would "resolve"
structure that is really those crossovers. The test is therefore run on the
transport regime alone, above the cage, and the full window is reported only to
show the difference.

Usage:
    python examples/fracmd_spectrum.py [composition] [T]
"""
import os
import sys

import numpy as np
from scipy.optimize import nnls

from fracmd_msd import load_tiers
from fracmd_alpha_T import ROOT

ALPHA_GRID = np.linspace(0.05, 1.40, 136)      # 0.01 spacing
CAGE_END_FS = 1.0e3                            # 1 ps: past ballistic and cage


def rel_rms(m, pred):
    """Root-mean-square RELATIVE residual. One objective, used for fitting and
    for scoring, so that the nested comparison below is meaningful."""
    return float(np.sqrt(np.mean((pred / m - 1.0) ** 2)))


def fit_single(t, m, grid=ALPHA_GRID):
    """Best single power law under the SAME relative objective as the spectrum.

    Fitted by scanning alpha and solving for the amplitude in closed form, so
    this is exactly the one-bin special case of fit_spectrum. Fitting it in
    log space instead -- the obvious thing -- would optimise a different
    objective and make the nested check meaningless.
    """
    best = (np.inf, None, None)
    for a in grid:
        col = t ** a
        k = np.sum(col / m) / np.sum((col / m) ** 2)   # min rel. residual
        e = rel_rms(m, k * col)
        if e < best[0]:
            best = (e, a, k)
    return best[1], best[0]


def fit_spectrum(t, m, grid=ALPHA_GRID):
    """Nonnegative superposition of power laws: convex, unique optimum.

    Rows are weighted by 1/MSD so the residual is relative. Without that the
    objective is dominated by the last decade -- over seven decades the final
    points are ~10^6 times larger than the first -- and the fit simply ignores
    everything else. Dividing both sides by m keeps the problem linear in w, so
    it is still NNLS and still has a unique optimum.
    """
    A = np.power.outer(t, grid)
    Aw = A / m[:, None]
    scale = Aw.max(axis=0)
    w, _ = nnls(Aw / scale, np.ones_like(m))
    w = w / scale
    return w, rel_rms(m, A @ w)


def occupancy(t, w, grid=ALPHA_GRID, frac=0.01):
    """Bins carrying real weight, measured by CONTRIBUTION rather than by w.

    A component contributes w_j * t^{a_j}, and t^{a_j} varies by orders of
    magnitude across the grid, so thresholding on w alone hides high-alpha
    components that need only a tiny coefficient to matter.
    """
    contrib = w * (t[-1] ** grid)          # contribution at the end of the window
    tot = contrib.sum()
    if tot <= 0:
        return np.zeros_like(w, dtype=bool), np.nan
    keep = contrib > frac * tot
    centre = float((grid[keep] * contrib[keep]).sum() / contrib[keep].sum())
    return keep, centre


def report(label, t, m):
    a1, e1 = fit_single(t, m)
    w, es = fit_spectrum(t, m)
    keep, centre = occupancy(t, w)
    span = np.log10(t[-1] / t[0])
    print(f"  {label}")
    print(f"    window            : {span:.2f} decades, {len(t)} points")
    print(f"    single power law  : alpha = {a1:.3f}, rel-RMS = {e1:.4%}")
    print(f"    free spectrum     : rel-RMS = {es:.4%}, "
          f"{int(keep.sum())} of {len(ALPHA_GRID)} bins carry >1% of the signal")
    if keep.any():
        print(f"    spectrum support  : {ALPHA_GRID[keep].min():.2f} - "
              f"{ALPHA_GRID[keep].max():.2f}, centroid {centre:.3f}")
    ok = es <= e1 + 1e-12
    print(f"    nested check      : {'PASS' if ok else 'FAIL'} "
          f"(spectrum {'<=' if ok else '>'} its own one-bin special case)")
    return e1, es, int(keep.sum())


def main(comp="67Li2S-33P2S5", T=300):
    base = os.path.join(ROOT, comp, f"T{T}")
    runs = sorted(os.listdir(base))
    t, m = load_tiers(os.path.join(base, runs[0]))
    print(f"{comp}  T = {T} K  ({runs[0]})\n")

    report("full window (includes ballistic and cage)", t, m)
    print()
    sel = t >= CAGE_END_FS
    r1_t, rs_t, n_t = report("transport regime only (t > 1 ps)", t[sel], m[sel])

    print()
    lo = t[sel][-1] / 10 ** 2.0
    sub = (t >= lo)
    r1_s, rs_s, n_s = report("a 2-decade sub-window, for comparison",
                             t[sub], m[sub])

    print("\n  ---")
    print(f"  error reduction from the extra freedom, transport window : "
          f"{r1_t:.3%} -> {rs_t:.3%}")
    print(f"  error reduction from the extra freedom, 2-decade window  : "
          f"{r1_s:.3%} -> {rs_s:.3%}")
    if n_t <= 2:
        print("  The spectrum collapses to a spike even with 4+ decades: the"
              "\n  width is not recoverable here either.")
    else:
        print(f"  The spectrum retains {n_t} bins over the transport window.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "67Li2S-33P2S5",
         int(sys.argv[2]) if len(sys.argv) > 2 else 300)
