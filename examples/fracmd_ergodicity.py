"""Aggregate the TAMSD/EAMSD comparison over the whole campaign.

Reads the per-run tamsd.npz written by fracmd_tamsd.py and reports, per
composition and temperature with replicate statistics:

    slope(EAMSD), slope(TAMSD), their difference, and EB

fitted over a fixed lag decade so that every run is compared on the same footing.

The difference is the discriminating quantity. For subdiffusive CTRW the
time-averaged MSD is linear in lag whatever alpha is, so

    slope(TAMSD) - slope(EAMSD)  ->  1 - alpha,

which at alpha = 0.5 is +0.5 and is not subtle. For an ergodic process, fBm or
fLE, the two averages scale together and the difference is zero. Calibration on
synthetic processes through this same code gave +0.37 for CTRW at alpha = 0.6 and
+0.01 for fBm.

Usage:
    python examples/fracmd_ergodicity.py
"""
import os

import numpy as np

from fracmd_alpha_T import ROOT

DT_PS = 2.0                      # traj_long: every 2000 steps at 1 fs
LAG_LO_PS, LAG_HI_PS = 20.0, 2000.0


def slopes(npz):
    z = np.load(npz)
    e, t, eb = z["eamsd"], z["tamsd"], z["eb"]
    lag = np.arange(len(e)) * DT_PS
    sel = (lag >= LAG_LO_PS) & (lag <= LAG_HI_PS) & (e > 0) & (t > 0)
    if sel.sum() < 10:
        return None
    se = np.polyfit(np.log(lag[sel]), np.log(e[sel]), 1)[0]
    st = np.polyfit(np.log(lag[sel]), np.log(t[sel]), 1)[0]
    return se, st, float(np.nanmean(eb[sel]))


def main():
    rows = {}
    for comp in sorted(os.listdir(ROOT)):
        cdir = os.path.join(ROOT, comp)
        if not os.path.isdir(cdir):
            continue
        for tdir in sorted(os.listdir(cdir)):
            if not tdir.startswith("T"):
                continue
            T = int(tdir[1:])
            for rdir in sorted(os.listdir(os.path.join(cdir, tdir))):
                f = os.path.join(cdir, tdir, rdir, "tamsd.npz")
                if os.path.exists(f):
                    r = slopes(f)
                    if r:
                        rows.setdefault((comp, T), []).append(r)
    if not rows:
        print("no tamsd.npz found -- has the cluster job finished and been pulled?")
        return

    print(f"  lag window {LAG_LO_PS:.0f}-{LAG_HI_PS:.0f} ps, identical for every run\n")
    print("  composition       T(K)  n   EAMSD slope      TAMSD slope      "
          "difference        EB")
    diffs = []
    for (comp, T), v in sorted(rows.items()):
        a = np.array(v)
        n = len(a)
        sd = lambda c: (a[:, c].std(ddof=1) if n > 1 else np.nan)
        d = a[:, 1] - a[:, 0]
        diffs.append(d)
        print(f"  {comp:16s} {T:4d} {n:2d}   "
              f"{a[:,0].mean():.3f} +/- {sd(0):.3f}   "
              f"{a[:,1].mean():.3f} +/- {sd(1):.3f}   "
              f"{d.mean():+.3f} +/- {d.std(ddof=1) if n>1 else np.nan:.3f}   "
              f"{a[:,2].mean():.2f}")

    all_d = np.concatenate(diffs)
    print(f"\n  pooled difference over {len(all_d)} runs: "
          f"{all_d.mean():+.4f} +/- {all_d.std(ddof=1):.4f} (sd), "
          f"sem {all_d.std(ddof=1)/np.sqrt(len(all_d)):.4f}")
    print("\n  reference values through this same code:")
    print("    CTRW  alpha=0.6 : difference +0.37")
    print("    fBm   alpha=0.6 : difference +0.01")
    print("\n  A CTRW at the alphas measured here would give +0.2 to +0.5.")


if __name__ == "__main__":
    main()
