"""Sensitivity of the transport exponent, and of the barrier-scale drift, to the
fitting window.

The refutation in Sec. IV rests on the implied barrier scale k_B T / beta
drifting with temperature: 18.7% for x=67 and 14.8% for x=70 over 250-350 K,
measured as (max-min)/mean.  beta itself is fitted as the log-log slope over the
last three decades of the stitched MSD.  That window is a choice, and the local
slope is still rising at the end of the trajectory, so the natural objection is
that the drift is an artifact of where the window was placed.

This script recomputes beta and the drift over a range of window widths, using
the same cached MSD tiers and the same estimator as the paper.  If the drift
survives across windows, the conclusion is robust to the choice; if it does not,
the paper's central number depends on it and must say so.

Prints a table and writes figures/fracmd_beta_window.{pdf,png}.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fracmd_msd import load_tiers, runs_root                                    # noqa: E402

KB_EV = 8.617333262e-5
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = runs_root()

WINDOWS = (2.0, 2.5, 3.0, 3.5, 4.0)      # decades, measured back from the end
COMPS = ("67Li2S-33P2S5", "70Li2S-30P2S5", "75Li2S-25P2S5")
DRIFT_RANGE = (250, 350)                  # the unsaturated range used in the paper


def beta_tail(t, m, decades):
    lt, lm = np.log10(t), np.log10(m)
    sel = lt >= lt[-1] - decades
    if sel.sum() < 10:
        return np.nan
    return float(np.polyfit(lt[sel], lm[sel], 1)[0])


def collect(comp, decades):
    """{T: [beta per replicate]} for one composition at one window width."""
    base = os.path.join(ROOT, comp)
    out = {}
    if not os.path.isdir(base):
        return out
    for tdir in sorted(os.listdir(base)):
        if not tdir.startswith("T"):
            continue
        try:
            T = int(tdir[1:])
        except ValueError:
            continue
        tpath = os.path.join(base, tdir)
        for rdir in sorted(os.listdir(tpath)):
            run = os.path.join(tpath, rdir)
            if not os.path.isdir(run):
                continue
            try:
                t, m = load_tiers(run)
            except FileNotFoundError:
                continue
            b = beta_tail(t, m, decades)
            if np.isfinite(b):
                out.setdefault(T, []).append(b)
    return out


def drift(data):
    """(max-min)/mean of k_B T / beta over the unsaturated temperature range."""
    lo, hi = DRIFT_RANGE
    e0 = [KB_EV * T / float(np.mean(b)) * 1000.0
          for T, b in sorted(data.items()) if lo <= T <= hi]
    if len(e0) < 2:
        return np.nan, e0
    return (max(e0) - min(e0)) / (sum(e0) / len(e0)), e0


def main():
    print(f"window sweep: beta = log-log slope over the last W decades\n"
          f"drift = (max-min)/mean of k_B T / beta over "
          f"{DRIFT_RANGE[0]}-{DRIFT_RANGE[1]} K\n")
    table = {}
    for comp in COMPS:
        print(f"=== {comp} ===")
        print("   W(dec)   " + "".join(f"{T:>9d}" for T in (250, 275, 300, 350))
              + "     drift")
        table[comp] = {}
        for W in WINDOWS:
            data = collect(comp, W)
            if not data:
                print(f"   {W:4.1f}     (no runs)")
                continue
            d, e0 = drift(data)
            betas = "".join(
                f"{np.mean(data[T]):9.4f}" if T in data else "        -"
                for T in (250, 275, 300, 350))
            dtxt = "     n/a" if not np.isfinite(d) else f"{d*100:7.1f}%"
            print(f"   {W:4.1f}  {betas}  {dtxt}")
            table[comp][W] = dict(drift=d, e0=e0,
                                  betas={T: float(np.mean(b))
                                         for T, b in data.items()})
        print()

    print("Paper reports 18.7% (x=67) and 14.8% (x=70) at W = 3.0 decades.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.4, 3.0))
        for comp in COMPS:
            ws = [W for W in WINDOWS if W in table[comp]
                  and np.isfinite(table[comp][W]["drift"])]
            if not ws:
                continue
            ax.plot(ws, [table[comp][W]["drift"] * 100 for W in ws],
                    marker="o", label=comp.split("Li")[0] + " (x)")
        ax.axvline(3.0, ls="--", color="0.5", lw=1.0)
        ax.set(xlabel="fitting window (decades back from end of run)",
               ylabel=r"drift in $k_BT/\beta$  (%)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        out = os.path.join(HERE, "figures", "fracmd_beta_window.pdf")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
        print(f"\nwrote figure -> {out}")
    except Exception as exc:                                   # pragma: no cover
        print(f"\n(plotting skipped: {exc!r})")


if __name__ == "__main__":
    main()
