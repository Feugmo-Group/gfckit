"""Sec 4.5: is the subdiffusion trap-limited (CTRW) or viscoelastic (fLE)?

This is the test Sec 4.3 depends on. alpha = k_B T / E_0 is a continuous-time
random-walk result: it presupposes trap-limited transport with a diverging mean
waiting time, which breaks ergodicity. If the dynamics are instead fractional
Langevin or fractional Brownian -- both ergodic -- then that relation has no
basis, and any agreement with it is coincidence.

The two cases are distinguished by comparing the ensemble-averaged MSD with the
TIME-averaged MSD of single particles:

    EAMSD(t)   = < |r_i(t) - r_i(0)|^2 >_i
    TAMSD_i(D) = 1/(T-D) * int_0^{T-D} |r_i(t+D) - r_i(t)|^2 dt

  * subdiffusive CTRW: EAMSD ~ t^alpha, but TAMSD ~ D / T^{1-alpha}. The time
    average is LINEAR in lag whatever alpha is, and shrinks as the trace gets
    longer -- the walk ages. Individual TAMSDs also scatter irreducibly.
  * fBm / fLE: TAMSD = EAMSD ~ D^alpha, and the scatter vanishes as T grows.

So the discriminating signature is the pair of log-log slopes, plus the
ergodicity-breaking parameter

    EB(D) = ( <d2^2> - <d2>^2 ) / <d2>^2      over particles, d2 = TAMSD_i(D)

which tends to zero for an ergodic process and to a nonzero constant for CTRW.

Run on the cluster, where the trajectories live:
    python3 fracmd_tamsd.py <run_dir> [out.npz]
"""
import gzip
import os
import sys

import numpy as np


def read_traj(path, max_frames=None):
    """Unwrapped Li coordinates from a LAMMPS custom/gz dump -> (n_frames, n_atoms, 3).

    Parsed block-wise rather than line-by-line: the file holds ~25 million
    coordinate lines, and np.fromstring on each one is minutes of pure Python
    per trajectory. The dump was written with `dump_modify sort id`, so the rows
    are already in id order and no sort is needed -- verified on the first frame.
    """
    frames = []
    with gzip.open(path, "rt") as fh:
        first = True
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            fh.readline()                                  # timestep
            fh.readline()                                  # ITEM: NUMBER OF ATOMS
            n = int(fh.readline())
            for _ in range(5):                             # box bounds + ITEM: ATOMS
                fh.readline()
            block = np.fromstring("".join(fh.readline() for _ in range(n)),
                                  sep=" ").reshape(n, 4)
            if first:
                ids = block[:, 0].astype(np.int64)
                if not np.all(np.diff(ids) > 0):
                    raise ValueError("dump is not sorted by id; add an argsort")
                first = False
            frames.append(block[:, 1:4].astype(np.float32))
            if max_frames and len(frames) >= max_frames:
                break
    return np.asarray(frames), None


def tamsd_fft(x):
    """Time-averaged MSD of one coordinate array (n_frames, n_atoms), all lags.

    Uses the Wiener-Khinchin route: computing this directly is O(N^2) per
    particle, which at 9901 frames x 2521 particles is ~10^11 operations.
    """
    n = x.shape[0]
    D = np.square(x).sum(axis=1) if x.ndim > 1 else np.square(x)
    D = np.append(D, 0.0)
    S2 = np.zeros(n)
    fk = np.fft.rfft(x, n=2 * n, axis=0)
    S2 = np.fft.irfft(fk * np.conjugate(fk), axis=0)[:n]
    if S2.ndim > 1:
        S2 = S2.sum(axis=1)
    Q = 2.0 * D.sum()
    S1 = np.empty(n)
    for m in range(n):
        Q -= D[m - 1] + D[n - m]
        S1[m] = Q / (n - m)
    return S1 - 2.0 * S2 / (n - np.arange(n))


def main(run_dir, out=None, max_frames=None):
    traj_path = os.path.join(run_dir, "traj_long.lammpstrj.gz")
    print(f"reading {traj_path}", flush=True)
    R, ids = read_traj(traj_path, max_frames)
    nf, na, _ = R.shape
    print(f"  {nf} frames x {na} Li atoms", flush=True)

    # ensemble-averaged MSD, measured from the first frame
    disp = R - R[0]
    eamsd = np.square(disp).sum(axis=2).mean(axis=1)

    # time-averaged MSD, per particle
    tamsd_i = np.empty((na, nf), dtype=np.float64)
    for a in range(na):
        tamsd_i[a] = tamsd_fft(R[:, a, :].astype(np.float64))
        if a % 500 == 0:
            print(f"  TAMSD {a}/{na}", flush=True)
    tamsd = tamsd_i.mean(axis=0)

    # ergodicity-breaking parameter, over particles at each lag
    with np.errstate(invalid="ignore", divide="ignore"):
        eb = tamsd_i.var(axis=0) / np.square(tamsd_i.mean(axis=0))

    out = out or os.path.join(run_dir, "tamsd.npz")
    np.savez_compressed(out, eamsd=eamsd, tamsd=tamsd, eb=eb,
                        n_frames=nf, n_atoms=na,
                        tamsd_quartiles=np.percentile(tamsd_i, [25, 50, 75], axis=0))
    print(f"wrote {out}", flush=True)

    # a quick read-out; the real comparison happens locally
    lag = np.arange(1, nf)
    for lo, hi, label in [(0.01, 0.1, "early"), (0.1, 0.5, "mid")]:
        s = slice(int(lo * nf), int(hi * nf))
        pe = np.polyfit(np.log(lag[s]), np.log(eamsd[1:][s]), 1)[0]
        pt = np.polyfit(np.log(lag[s]), np.log(tamsd[1:][s]), 1)[0]
        print(f"  {label:6s} lags: EAMSD slope {pe:.3f}   TAMSD slope {pt:.3f}   "
              f"EB {np.nanmean(eb[s]):.3f}")


if __name__ == "__main__":
    main(sys.argv[1],
         sys.argv[2] if len(sys.argv) > 2 else None,
         int(sys.argv[3]) if len(sys.argv) > 3 else None)
