"""First look at a fracMD run: Li mean-squared displacement across all tiers.

The LAMMPS campaign writes the Li MSD in three overlapping tiers so that a
single run covers femtoseconds to tens of nanoseconds without an enormous dump:

    tier A   1 fs sampling, to    10 ps
    tier B 100 fs sampling, to     1 ns
    tier C  10 ps sampling, to    20 ns

Stitched together that is ~7.3 decades, which is the span the identifiability
argument in the FCAA paper turns on.  This script combines the tiers and
measures the local log-log slope beta(t) in MSD ~ t^beta: beta = 1 is Fickian,
beta < 1 is subdiffusive and is what a memory kernel encodes.

Trajectories stay on the cluster; only these reduced tiers are in the repo.
Usage:  python examples/fracmd_msd.py [run_dir]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT_FS = 1.0                       # LAMMPS timestep, from log.lammps


def runs_root():
    """Directory holding <composition>/T*/r* run directories.

    Two layouts are supported so that one copy of these scripts works both in
    the source tree and in the deposited archive, whose runs sit under runs/
    and whose tables are gzipped:

        <repo>/data/fracmd/<composition>/T*/r*        (source tree)
        <deposit>/runs/<composition>/T*/r*            (Zenodo archive)

    Set FRACMD_ROOT to override.
    """
    env = os.environ.get("FRACMD_ROOT")
    if env:
        return os.path.expanduser(env)
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(HERE, "data", "fracmd"),          # source tree
                 os.path.join(os.path.dirname(here), "runs"),   # archive: ../runs
                 os.path.join(here, "runs")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(HERE, "data", "fracmd")


ROOT = runs_root()
DEFAULT = os.path.join(ROOT, "67Li2S-33P2S5", "T300", "r1")


def load_tiers(run_dir, tiers="ABC"):
    """Concatenate the MSD tiers into one (t [fs], msd_tot [A^2]) curve.

    Accepts either plain or gzipped tables: the archive stores them as .gz.
    """
    t_all, m_all = [], []
    for tier in tiers:
        base = os.path.join(run_dir, f"msd_tier{tier}.dat")
        path = next((p for p in (base, base + ".gz") if os.path.exists(p)), None)
        if path is None:
            continue
        a = np.loadtxt(path, comments="#")      # numpy reads .gz transparently
        t_all.append(a[:, 0] * DT_FS)
        m_all.append(a[:, 4])
    if not t_all:
        raise FileNotFoundError(f"no msd_tier*.dat[.gz] under {run_dir}")
    t = np.concatenate(t_all)
    m = np.concatenate(m_all)
    t, idx = np.unique(t, return_index=True)      # drop duplicated tier edges
    m = m[idx]
    keep = t > 0                                   # t = 0 has no log
    return t[keep], m[keep]


def decade_slopes(t, m):
    """Local log-log slope per decade, so the ballistic / caged / diffusive
    regimes stay separate instead of being averaged into one number."""
    lt, lm = np.log10(t), np.log10(m)
    edges = np.arange(np.floor(lt[0]), np.ceil(lt[-1]) + 1e-9, 1.0)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (lt >= lo) & (lt < hi)
        if sel.sum() >= 5:
            out.append((lo, hi, np.polyfit(lt[sel], lm[sel], 1)[0], m[sel][-1]))
    return out


def main(run_dir=DEFAULT):
    t, m = load_tiers(run_dir)
    span = np.log10(t[-1]) - np.log10(t[0])
    print(f"{os.path.relpath(run_dir, HERE)}")
    print(f"{len(t)} points, {t[0]:.4g} fs -> {t[-1]/1e6:.1f} ns "
          f"= {span:.2f} decades;  MSD {m[0]:.4g} -> {m[-1]:.4g} A^2\n")

    print("  decade window        beta      MSD_end [A^2]")
    for lo, hi, beta, mend in decade_slopes(t, m):
        print(f"  1e{lo:.0f} - 1e{hi:.0f} fs   {beta:8.3f}   {mend:12.4g}")

    lt, lm = np.log10(t), np.log10(m)
    tail = lt >= lt[-1] - 3.0
    print(f"\nlast 3 decades: beta = {np.polyfit(lt[tail], lm[tail], 1)[0]:.4f}")

    # a single power law over the whole span -- the "collapsed" description the
    # identifiability argument asks about
    beta, c = np.polyfit(lt, lm, 1)
    resid = lm - (beta * lt + c)
    r2 = 1 - np.sum(resid ** 2) / np.sum((lm - lm.mean()) ** 2)
    print(f"single power law over all {span:.2f} decades: beta = {beta:.4f}, "
          f"log-RMS = {np.sqrt(np.mean(resid ** 2)):.4f}, R^2 = {r2:.5f}")

    D = m[-1] / (6.0 * t[-1] * 1e-15) * 1e-16      # A^2/fs -> cm^2/s
    print(f"apparent D from 6Dt at the last point = {D:.3e} cm^2/s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
