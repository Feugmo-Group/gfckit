"""Per-Li mobility, keyed by LAMMPS atom id, for correlation against structure.

The composition-level comparison of F_IS against beta has three points and is
therefore weak. The informative version is within a single structure: do the
mobile Li sit at systematically different local environments from the immobile
ones? That is thousands of atoms rather than three, and it is a direct
structure-dynamics correlation rather than a trend across samples.

This runs on the cluster, where the trajectories are, and writes a small file
holding per-atom TAMSD at a reference lag together with the atom ids, so it can
be joined to the F_IS computed from the corresponding data file.

Ids matter: traj_long dumps Li only, while the .data file holds all species, so
the join has to be on id and not on row order.

Usage (on nibi):
    python3 fracmd_per_atom_mobility.py <run_dir> [lag_ps]
"""
import gzip
import os
import sys

import numpy as np


def read_traj_with_ids(path):
    """-> (n_frames, n_atoms, 3) positions and the atom ids of the columns."""
    frames, ids = [], None
    with gzip.open(path, "rt") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            fh.readline()
            fh.readline()
            n = int(fh.readline())
            for _ in range(5):
                fh.readline()
            block = np.fromstring("".join(fh.readline() for _ in range(n)),
                                  sep=" ").reshape(n, 4)
            if ids is None:
                ids = block[:, 0].astype(np.int64)
            frames.append(block[:, 1:4].astype(np.float32))
    return np.asarray(frames), ids


def tamsd_at_lag(R, k):
    """Time-averaged MSD of every particle at a single lag of k frames.

    Only one lag is needed, so this is the direct average over time origins
    rather than the FFT over all lags.
    """
    d = R[k:] - R[:-k]                       # (n_frames-k, n_atoms, 3)
    return np.square(d).sum(axis=2).mean(axis=0)


def main(run_dir, lag_ps=200.0, dt_ps=2.0):
    R, ids = read_traj_with_ids(os.path.join(run_dir, "traj_long.lammpstrj.gz"))
    k = max(1, int(round(lag_ps / dt_ps)))
    tam = tamsd_at_lag(R, k)
    tot = np.square(R[-1] - R[0]).sum(axis=1)      # net displacement squared
    out = os.path.join(run_dir, "per_atom_mobility.npz")
    np.savez_compressed(out, ids=ids, tamsd=tam, total_sq_disp=tot,
                        lag_ps=lag_ps, n_frames=R.shape[0])
    print(f"{run_dir}: {R.shape[0]} frames x {len(ids)} Li, lag {lag_ps} ps")
    print(f"  TAMSD spread over atoms: median {np.median(tam):.3f}, "
          f"10-90% {np.percentile(tam,10):.3f}-{np.percentile(tam,90):.3f} A^2")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 200.0)
