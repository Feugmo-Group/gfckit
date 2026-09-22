"""Positive control for the distributed-order spectrum estimator.

Section 3.5 of the GFC paper reports that a two-order memory collapses onto a
single intermediate order.  That is a claim about identifiability, but on its own
it does not exclude the alternative reading that the softmax spectrum estimator
simply cannot represent a bimodal spectrum.  This script separates the two.

The estimator is given progressively more of the one resource the paper argues is
decisive -- observation-window width in decades -- with everything else held
generous (well-separated orders, equal weights, no noise, converged optimizer).
If the recovery transitions from collapsed to bimodal as the window widens, then
the estimator works and the collapse is a property of the data.  If it stays
collapsed even at the widest window, the negative result is partly a statement
about this estimator and the paper must say so.

Window width is set by N at fixed dt: the sampled range runs from dt to N*dt, so
the span is log10(N) decades.  Cost scales steeply with N (the forward solve is a
dense triangular solve), which caps the sweep near 3.5 decades.

Writes figures/spectrum_positive_control.{pdf,png} and prints the table.
"""
import os
import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gfckit.complex import multiterm_fractional_relaxation          # noqa: E402
from gfckit.identifiability import w1_gap                            # noqa: E402
from gfckit.identify import recover_order_spectrum                   # noqa: E402

DT = 0.03
LAM, S = 0.02, 0.03
GRID = jnp.linspace(0.05, 0.95, 37)

# 50k steps converges the paper's own {0.3,0.8} benchmark (R^2 = 0.99997) but
# not this one: the {0.2,0.9} control at N=1600 reaches only R^2 = 0.9957 at 50k
# and 0.9992 at 200k, and its recovered spectrum changes from one mode to two
# between the two.  By the paper's own diagnostic that 50k result is optimizer
# failure rather than ill-posedness, so the control is run to 200k throughout.
STEPS = 200000

# Well-separated pair with equal weights: the most favourable bimodal target.
TRUE_ORDERS, TRUE_W = [0.2, 0.9], [0.5, 0.5]

# The paper's own Sec. 3.5 benchmark, run at its published settings as the
# reference "collapse" case.
PAPER_ORDERS, PAPER_W, PAPER_N = [0.3, 0.8], [0.6, 0.4], 800


def peaks(w, grid, frac=0.10):
    """Return (n_modes, list of (order, mass)) for contiguous bins holding
    more than `frac` of the total mass, merging adjacent occupied bins."""
    w = jnp.asarray(w)
    occupied = w > frac * float(jnp.max(w))
    modes, run = [], []
    for i, occ in enumerate(occupied):
        if bool(occ):
            run.append(i)
        elif run:
            modes.append(run)
            run = []
    if run:
        modes.append(run)
    out = []
    for run in modes:
        mass = float(sum(w[i] for i in run))
        centre = float(sum(float(grid[i]) * float(w[i]) for i in run) / (mass + 1e-30))
        out.append((centre, mass))
    return len(out), out


def run_case(label, orders, weights, N):
    t, u = multiterm_fractional_relaxation(orders, weights, lam=LAM, S=S,
                                           u0=0.0, dt=DT, nsteps=N - 1)
    w, r2, _ = recover_order_spectrum(t, u, GRID, steps=STEPS)
    gap = float(w1_gap(jnp.asarray(orders), jnp.asarray(weights), GRID, w))
    n_modes, modes = peaks(w, GRID)
    span = float(jnp.log10(jnp.asarray(float(N))))
    print(f"{label:22s} N={N:5d} span={span:.2f} dec  R2={r2:.6f}  "
          f"W1 gap={gap*100:5.1f}%  modes={n_modes}  "
          + "  ".join(f"(a={c:.2f}, m={m:.2f})" for c, m in modes))
    return dict(label=label, N=N, span=span, r2=float(r2), gap=gap,
                n_modes=n_modes, modes=modes, w=w, orders=orders, weights=weights)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "figures", "spectrum_positive_control.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"order grid: {GRID.shape[0]} bins on [{float(GRID[0]):.2f}, "
          f"{float(GRID[-1]):.2f}]; {STEPS} Adam steps; noise-free\n")
    print("reference case (the paper's Sec. 3.5 benchmark):")
    results = [run_case("paper {0.3,0.8}", PAPER_ORDERS, PAPER_W, PAPER_N)]

    # N is capped at 1600 by cost: the forward solve is a dense triangular solve,
    # so a converged 200k-step fit at N=3200 takes hours rather than minutes.
    print("\npositive control, widening the window at fixed dt:")
    for N in (800, 1600):
        results.append(run_case("control {0.2,0.9}", TRUE_ORDERS, TRUE_W, N))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ctrl = [r for r in results if r["label"].startswith("control")]
        fig, ax = plt.subplots(1, len(ctrl), figsize=(3.1 * len(ctrl), 2.6),
                               sharey=True)
        for a, r in zip(ax, ctrl):
            a.bar([float(x) for x in GRID], [float(x) for x in r["w"]],
                  width=0.024, color="0.35")
            for o in r["orders"]:
                a.axvline(o, ls="--", color="crimson", lw=1.0)
            a.set_title(f"{r['span']:.2f} decades\n$R^2$={r['r2']:.4f}, "
                        f"gap={r['gap']*100:.0f}%", fontsize=8)
            a.set_xlabel(r"order $\alpha$")
        ax[0].set_ylabel(r"recovered weight $w(\alpha)$")
        fig.suptitle("Spectrum estimator, noise-free, true orders dashed",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
        print(f"\nwrote figure -> {out}")
    except Exception as exc:                                   # pragma: no cover
        print(f"\n(plotting skipped: {exc!r})")


if __name__ == "__main__":
    main()
