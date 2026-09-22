"""Site-energy landscape: E_0 and E_max from per-atom potential energies.

The CTRW derivation assumes Li site depths are exponentially distributed,
p(E) = E_0^{-1} exp(-E/E_0), and predicts alpha = k_B T / E_0. The production
campaign dumped no energies, so a dedicated 2 ns NVE run wrote per-atom potential
energy on the same 2 ps grid as the position dump.

A site energy is not an instantaneous quantity: an ion's potential energy
fluctuates by k_B T on every vibration. What the theory means by the depth of a
site is the energy of the basin, so here each ion's energy is averaged over the
interval it stays within d_c of where it settled, using the same residence
definition as the waiting-time analysis. Vibrational noise averages down over the
tens to hundreds of frames a typical residence lasts; what survives is the
basin depth.

Energies are referenced to the median site energy of the composition, since only
the width of the distribution enters Eq. (alpha) and the absolute per-atom energy
in a many-body force field carries an arbitrary offset.

Two numbers come out:
  E_0    the exponential decay width of the deep tail, by maximum likelihood
  E_max  the deepest basin sampled, which sets the tempering rate in Eq. (lambda)

Run on the cluster:
    python3 fracmd_landscape.py <landscape_dir> [d_c]
"""
import gzip
import os
import sys

import numpy as np

DT_PS = 2.0
KCAL_TO_MEV = 43.3641           # 1 kcal/mol in meV


def read_pe_traj(path):
    """-> positions (n_frames, n_li, 3) and per-atom PE (n_frames, n_li)."""
    pos, pe = [], []
    with gzip.open(path, "rt") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            fh.readline(); fh.readline()
            n = int(fh.readline())
            for _ in range(5):
                fh.readline()
            b = np.fromstring("".join(fh.readline() for _ in range(n)),
                              sep=" ").reshape(n, 5)      # id xu yu zu c_peat
            pos.append(b[:, 1:4].astype(np.float32))
            pe.append(b[:, 4].astype(np.float32))
    return np.asarray(pos), np.asarray(pe)


def site_energies(pos, pe, d_c):
    """Mean PE over each completed residence, pooled over ions."""
    nf, na, _ = pos.shape
    out = []
    for a in range(na):
        x, e = pos[:, a, :], pe[:, a]
        anchor, last = x[0].copy(), 0
        for t in range(1, nf):
            if np.sum((x[t] - anchor) ** 2) > d_c * d_c:
                if last > 0 and t - last >= 3:      # need a few frames to average
                    out.append(e[last:t].mean())
                last, anchor = t, x[t].copy()
    return np.asarray(out, dtype=float)


def exp_tail_mle(depth, qmin=0.5):
    """E_0 from the exponential tail of the depth distribution.

    For p(E) ~ exp(-E/E_0) above a cutoff, the MLE of E_0 is the mean excess
    over that cutoff. Reported across several cutoffs, because a genuine
    exponential gives the same answer at each and a curved distribution does not.
    """
    out = []
    for q in (qmin, 0.7, 0.8, 0.9):
        c = np.quantile(depth, q)
        x = depth[depth >= c]
        if len(x) > 100:
            out.append((q, float(np.mean(x - c)), len(x)))
    return out


def main(run_dir, d_c=2.0):
    pos, pe = read_pe_traj(os.path.join(run_dir, "pe_2ns.lammpstrj.gz"))
    print(f"{run_dir}: {pos.shape[0]} frames x {pos.shape[1]} Li")

    se = site_energies(pos, pe, d_c) * KCAL_TO_MEV      # LAMMPS real units
    if len(se) < 500:
        print(f"  only {len(se)} residences at d_c={d_c} -- widen the threshold")
        return
    # depth below the median site: deeper site = larger positive depth
    depth = np.median(se) - se
    depth = depth[depth > 0]

    print(f"  {len(se)} residences, {len(depth)} below the median")
    print(f"  site-energy spread (10-90%): "
          f"{np.percentile(se,10)-np.median(se):+.0f} to "
          f"{np.percentile(se,90)-np.median(se):+.0f} meV about the median")
    print("  E_0 from the exponential tail:")
    for q, e0, n in exp_tail_mle(depth):
        print(f"    above the {q:.0%} quantile: E_0 = {e0:6.1f} meV  (n={n})")
    print(f"  E_max (deepest basin sampled)      : {depth.max():.0f} meV")
    print(f"  E_max (99.9th percentile, robust)  : {np.percentile(depth,99.9):.0f} meV")

    np.savez_compressed(os.path.join(run_dir, "landscape.npz"),
                        site_energies_meV=se, depth_meV=depth, d_c=d_c)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 2.0)
