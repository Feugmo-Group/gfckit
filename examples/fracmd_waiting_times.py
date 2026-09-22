"""The waiting-time distribution, tested directly.

CTRW is not defined by subdiffusion; it is defined by a waiting-time density with
a heavy tail,

    psi(tau) ~ tau^{-1-alpha},   alpha < 1,

whose mean diverges. Everything the model predicts -- the fractional operator,
alpha = k_B T / E_0, the broken ergodicity -- follows from that one assumption.
So measuring psi(tau) tests the premise itself, independently of the TAMSD
comparison, which tests a consequence.

A residence is defined by a displacement threshold: a Li is taken to be in the
same site while it stays within d_c of the position where it last settled, and a
jump is recorded when it leaves. d_c is swept, because the answer must not depend
on it -- a genuine power-law tail is scale free, whereas a threshold artefact is
not.

Two things are reported:

  * the tail exponent of psi(tau) from a maximum-likelihood Pareto fit above a
    cutoff, since binning a heavy tail and fitting the histogram is biased;
  * whether the mean waiting time converges, by comparing the sample mean over
    the first and second halves of the trace. A diverging mean does not settle.

Run on the cluster:
    python3 fracmd_waiting_times.py <run_dir>
"""
import gzip
import os
import sys

import numpy as np

DT_PS = 2.0                                   # traj_long sampling


def read_traj(path):
    frames = []
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
                              sep=" ").reshape(n, 4)
            frames.append(b[:, 1:4].astype(np.float32))
    return np.asarray(frames)


def waiting_times(R, d_c):
    """Inter-jump intervals, in frames, pooled over atoms.

    Only completed residences are kept: the first and last of each trace are
    censored by the finite window and including them biases the tail downwards.
    """
    nf, na, _ = R.shape
    out = []
    for a in range(na):
        x = R[:, a, :]
        anchor = x[0].copy()
        last = 0
        for t in range(1, nf):
            if np.sum((x[t] - anchor) ** 2) > d_c * d_c:
                if last > 0:                   # drop the first, it is censored
                    out.append(t - last)
                last = t
                anchor = x[t].copy()
    return np.asarray(out, dtype=float)


def pareto_mle(tau, tmin):
    """Hill estimator: alpha such that psi ~ tau^{-1-alpha} above tmin."""
    x = tau[tau >= tmin]
    if len(x) < 50:
        return np.nan, 0
    return len(x) / np.sum(np.log(x / tmin)), len(x)


def main(run_dir):
    R = read_traj(os.path.join(run_dir, "traj_long.lammpstrj.gz"))
    print(f"{run_dir}: {R.shape[0]} frames x {R.shape[1]} Li\n")
    print("   d_c(A)   n jumps   <tau> (ps)   first half / second half   "
          "tail alpha")
    for d_c in (1.5, 2.0, 2.5, 3.0):
        tau = waiting_times(R, d_c) * DT_PS
        if len(tau) < 100:
            print(f"   {d_c:4.1f}    too few jumps ({len(tau)})")
            continue
        h1 = tau[: len(tau) // 2].mean()
        h2 = tau[len(tau) // 2:].mean()
        a, n_tail = pareto_mle(tau, np.percentile(tau, 90))
        print(f"   {d_c:4.1f}   {len(tau):8d}   {tau.mean():9.2f}   "
              f"{h1:8.2f} / {h2:8.2f}      {a:.2f}  (n={n_tail})")

    print("\n  CTRW needs tail alpha < 1, and then the mean does not converge:")
    print("  the two halves would keep drifting apart as more of the tail is")
    print("  sampled. A stable mean and alpha > 1 mean the premise fails.")


if __name__ == "__main__":
    main(sys.argv[1])
