"""Compare the Li site-energy landscape under the class2 force field and MACE.

Section (landscape) of the LiPS paper measures E_0 -- the exponential width of
the deep tail of the Li site-energy distribution -- and feeds it to the
parameter-free test alpha = k_B T / E_0. That measurement inherits whatever the
force field says about site depths, and the same force field is known to
underestimate the Li migration barrier by ~119 meV.

This recomputes E_0 and E_max from per-atom energies produced by a MACE model
fitted to DFT (the LiPS-25 benchmark models) on the *same* configurations, so
the classical and MACE landscapes differ only in the energy model. The
residence definition, the 2 ps grid and the trajectory are identical.

What this does and does not test: sampling is still classical, so this asks
whether the classical force field misreports the depths of the basins it
visits, not whether it visits the right basins. A large change in E_0 would
undercut the parameter-free test as published; agreement would strengthen it.

Units differ between the two runs and this is the easy thing to get wrong. The
classical run used LAMMPS `real` units and wrote kcal/mol; the MACE run used
`metal` units and wrote eV. Both are converted to meV here.

Usage:
    python fracmd_landscape_mace.py <mace_dump> [classical_npz] [d_c]
"""
import os
import sys

import numpy as np

KCAL_TO_MEV = 43.3641          # 1 kcal/mol in meV   (classical, real units)
EV_TO_MEV = 1000.0             # 1 eV in meV         (MACE, metal units)


def read_traj(path):
    """-> positions (n_frames, n_li, 3), per-atom energy (n_frames, n_li).

    Reads the plain-text dump written by the rerun: `id xu yu zu c_peat`,
    sorted by id, Li only.
    """
    pos, pe = [], []
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            fh.readline()                      # timestep
            fh.readline()                      # ITEM: NUMBER OF ATOMS
            n = int(fh.readline())
            for _ in range(5):                 # box (1 header + 3) + ATOMS hdr
                fh.readline()
            b = np.array([fh.readline().split() for _ in range(n)], dtype=float)
            pos.append(b[:, 1:4].astype(np.float32))
            pe.append(b[:, 4].astype(np.float32))
    return np.asarray(pos), np.asarray(pe)


def site_energies(pos, pe, d_c):
    """Mean energy over each completed residence, pooled over ions.

    Identical to the definition in fracmd_landscape.py: an ion is in a site
    while it stays within d_c of where it settled, and the site energy is its
    energy averaged over that interval, so vibrational noise averages down.
    """
    nf, na, _ = pos.shape
    out = []
    for a in range(na):
        x, e = pos[:, a, :], pe[:, a]
        anchor, last = x[0].copy(), 0
        for t in range(1, nf):
            if np.sum((x[t] - anchor) ** 2) > d_c * d_c:
                if last > 0 and t - last >= 3:
                    out.append(e[last:t].mean())
                last, anchor = t, x[t].copy()
    return np.asarray(out, dtype=float)


def exp_tail_mle(depth):
    """E_0 as the mean excess over a cutoff, reported at several cutoffs.

    A genuine exponential tail gives the same E_0 at every cutoff. The published
    classical result does not: E_0 falls by 19-31% across these cutoffs, which
    is why the paper reports the spread rather than a single number.
    """
    out = []
    for q in (0.5, 0.7, 0.8, 0.9):
        c = np.quantile(depth, q)
        x = depth[depth >= c]
        if len(x) > 100:
            out.append((q, float(np.mean(x - c)), len(x)))
    return out


def landscape(pos, pe_mev, d_c, label):
    se = site_energies(pos, pe_mev, d_c)
    if len(se) < 500:
        print(f"  {label}: only {len(se)} residences at d_c={d_c}")
        return None
    depth = np.median(se) - se
    depth = depth[depth > 0]
    print(f"  {label}: {len(se)} residences, {len(depth)} below the median")
    rows = exp_tail_mle(depth)
    for q, e0, n in rows:
        print(f"    E_0 above the {q:.0%} quantile: {e0:7.1f} meV  (n={n})")
    e0s = [r[1] for r in rows]
    print(f"    spread across cutoffs: {100*(max(e0s)-min(e0s))/max(e0s):.0f}%")
    print(f"    E_max (99.9th pct)   : {np.percentile(depth, 99.9):.0f} meV")
    return {"site_energies_meV": se, "depth_meV": depth,
            "E0_by_cutoff": np.array(rows), "d_c": d_c}


def main(mace_dump, classical_npz=None, d_c=2.0):
    print(f"{mace_dump}")
    pos, pe = read_traj(mace_dump)
    print(f"  {pos.shape[0]} frames x {pos.shape[1]} Li")

    mace = landscape(pos, pe * EV_TO_MEV, d_c, "MACE   ")

    if classical_npz and os.path.exists(classical_npz):
        z = np.load(classical_npz)
        cd = z["depth_meV"]
        rows = exp_tail_mle(cd)
        print("  class2 (from the published run):")
        for q, e0, n in rows:
            print(f"    E_0 above the {q:.0%} quantile: {e0:7.1f} meV  (n={n})")
        if mace is not None:
            print("  ratio MACE/class2 at each cutoff:")
            for (q, em, _), (_, ec, _) in zip(mace["E0_by_cutoff"], rows):
                print(f"    {q:.0%}: {em/ec:5.2f}x   ({em:.1f} vs {ec:.1f} meV)")

    if mace is not None:
        out = os.path.join(os.path.dirname(mace_dump) or ".",
                           "landscape_mace.npz")
        np.savez_compressed(out, **mace)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1],
         sys.argv[2] if len(sys.argv) > 2 else None,
         float(sys.argv[3]) if len(sys.argv) > 3 else 2.0)
