"""Attempt frequency nu_0 from the Li-projected vibrational density of states.

Sec 2 predicts lambda ~ nu_0 exp(-E_max / k_B T), and nu_0 is the one factor in
that expression measurable without a potential-energy landscape: it is the
characteristic rattling frequency of Li in its cage, read off the velocity
autocorrelation.

    VACF(t) = < v_i(0) . v_i(t) >_i        (Li only)
    VDOS(f) = |FFT[VACF]|                  (one-sided)

nu_0 is reported three ways because they do not agree in a disordered solid and
the difference is not noise: the peak of the VDOS, its first moment, and the
inverse of the first zero-crossing of the VACF. Quoting one alone would hide a
factor of two.

The velocity dump is 4 fs sampling over 20 ps, giving a Nyquist limit of
125 THz and a resolution of 0.05 THz -- comfortable for a mode expected in the
5-20 THz range.

Run on the cluster:
    python3 fracmd_vdos.py <run_dir>
"""
import gzip
import os
import sys

import numpy as np

DT_FS = 4.0                       # vel dump every 4 steps at 1 fs
LI_MASS_MAX = 10.0


def read_vel(path):
    """Li velocities -> (n_frames, n_li, 3). Types are in the dump; Li is the
    type whose mass is smallest, but the dump carries type not mass, so the Li
    type id is taken from the majority-species argument in the caller."""
    frames, types = [], None
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
            b = np.fromstring("".join(fh.readline() for _ in range(n)),
                              sep=" ").reshape(n, 5)          # id type vx vy vz
            if types is None:
                types = b[:, 1].astype(int)
            frames.append(b[:, 2:5].astype(np.float32))
    return np.asarray(frames), types


def vacf(V):
    """Normalised velocity autocorrelation, averaged over atoms and origins."""
    nf = V.shape[0]
    fk = np.fft.rfft(V, n=2 * nf, axis=0)
    ac = np.fft.irfft(fk * np.conjugate(fk), axis=0)[:nf]
    ac = ac.sum(axis=2).mean(axis=1)          # sum xyz, mean over atoms
    ac /= (nf - np.arange(nf))                # unbiased over origins
    return ac / ac[0]


def main(run_dir, li_type=8):
    path = os.path.join(run_dir, "vel_20ps.lammpstrj.gz")
    V, types = read_vel(path)
    li = types == li_type
    if li.sum() == 0:
        raise ValueError(f"no atoms of type {li_type}")
    V = V[:, li, :]
    nf = V.shape[0]
    print(f"{run_dir}: {nf} frames x {li.sum()} Li, dt = {DT_FS} fs")

    c = vacf(V)
    freq = np.fft.rfftfreq(nf, d=DT_FS * 1e-15) / 1e12        # THz
    dos = np.abs(np.fft.rfft(c * np.hanning(nf)))
    dos[0] = 0.0

    peak = freq[np.argmax(dos)]
    first_moment = float((freq * dos).sum() / dos.sum())
    zc = np.where(np.diff(np.sign(c)))[0]
    tau_zero = (zc[0] + 1) * DT_FS if len(zc) else np.nan     # fs
    nu_zero = 1.0 / (4.0 * tau_zero * 1e-15) / 1e12 if np.isfinite(tau_zero) else np.nan

    print(f"  VDOS peak            : {peak:6.2f} THz")
    print(f"  VDOS first moment    : {first_moment:6.2f} THz")
    print(f"  1/(4 x VACF zero)    : {nu_zero:6.2f} THz  "
          f"(first zero at {tau_zero:.0f} fs)")
    np.savez_compressed(os.path.join(run_dir, "vdos.npz"),
                        freq=freq, dos=dos, vacf=c, dt_fs=DT_FS,
                        peak_thz=peak, moment_thz=first_moment,
                        zero_thz=nu_zero)
    print(f"  wrote {os.path.join(run_dir, 'vdos.npz')}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
