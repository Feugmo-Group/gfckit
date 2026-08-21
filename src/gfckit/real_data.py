"""Loaders for open real-cell datasets (numpy only).

Currently: Zhang et al. 2020 (Nature Communications) --- 12 commercial cells with
EIS measured throughout aging plus full cycling capacity. Downloaded to
`gfckit/data/real/zhang2020/` (Zenodo record 3633835). This provides REAL
capacity-fade and impedance observables to complement the PyBaMM benchmark.

EIS file columns : time, cycle, freq/Hz, Re(Z)/Ohm, -Im(Z)/Ohm, |Z|, phase
Capacity columns : time, cycle, ox/red, Capacity/mA.h
File naming      : EIS_state_<roman>_<T>C<cell>.txt , Data_Capacity_<T>C<cell>.txt
where <state> is an aging checkpoint (I = fresh, later = more aged).
"""
import os
import glob
import re
import numpy as np

_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
          "VIII": 8, "IX": 9, "X": 10}


def load_zhang_eis(base):
    """Return {(temp, cell): [(state, freq, Zre, negZim), ...] sorted by state}."""
    out = {}
    for f in glob.glob(os.path.join(base, "EIS data", "*.txt")):
        m = re.search(r"EIS_state_([IVX]+)_(\d+)C(\d+)\.txt", os.path.basename(f))
        if not m:
            continue
        st = _ROMAN.get(m.group(1), 0)
        temp, cell = int(m.group(2)), int(m.group(3))
        d = np.loadtxt(f, skiprows=1)
        out.setdefault((temp, cell), []).append((st, d[:, 2], d[:, 3], d[:, 4]))
    for k in out:
        out[k].sort(key=lambda x: x[0])
    return out


def load_oxford(path, n_dva=6, downsample=400):
    """Load the Oxford Battery Degradation Dataset 1 (.mat, MATLAB v5).
    Returns {cell: {cycle, capacity_mAh, dva:[(cycle,Q,V,dvdq), ...]}}.
    Uses the C/18 low-rate pseudo-OCV discharge (OCVdc) for clean dV/dQ."""
    import scipy.io as sio
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    out = {}
    for k in [x for x in m if not x.startswith("__")]:
        c = m[k]
        chk = sorted(a for a in dir(c) if a.startswith("cyc"))
        cyc, cap = [], []
        for name in chk:
            L = getattr(c, name).OCVdc
            q = np.abs(np.asarray(L.q, float))
            cyc.append(int(name[3:]))
            cap.append(float(q.max()))
        cyc, cap = np.array(cyc, float), np.array(cap, float)
        pick = np.unique(np.linspace(0, len(chk) - 1, n_dva).astype(int))
        dva = []
        for i in pick:
            L = getattr(c, chk[i]).OCVdc
            q = np.abs(np.asarray(L.q, float))
            v = np.asarray(L.v, float)
            o = np.argsort(q)
            q, v = q[o], v[o]
            step = max(1, len(q) // downsample)
            q, v = q[::step], v[::step]
            dvdq = np.gradient(v, q)
            dva.append((int(chk[i][3:]), q, v, dvdq))
        out[k] = dict(cycle=cyc, capacity_mAh=cap, dva=dva)
    return out


def eis_features(freq, Zre, negZim):
    """Summarize one impedance spectrum into interpretable features:
    R_s (ohmic, high-frequency real-axis intercept), R_ct (charge-transfer arc
    size), and the arc peak height.  Returns a dict."""
    f = np.asarray(freq, float)
    zr = np.asarray(Zre, float)
    zi = np.asarray(negZim, float)
    uf, idx = np.unique(f, return_index=True)          # one clean sweep
    order = idx[np.argsort(f[idx])[::-1]]
    f, zr, zi = f[order], zr[order], zi[order]
    hf = f >= 500.0
    R_s = float(zr[hf].min()) if hf.any() else float(zr.min())
    j = int(np.argmin(np.abs(f - 1.0)))                # ~end of charge-transfer arc
    R_ct = float(zr[j] - R_s)
    peak = float(np.nanmax(zi))
    return {"R_s": R_s, "R_ct": R_ct, "peak": peak}


def load_zhang_capacity(base):
    """Return {(temp, cell): (cycles, capacity_per_cycle_mAh)} (discharge fade).

    The released files come in two layouts:
        4 columns: time, cycle number, ox/red, Capacity/mA.h
        6 columns: time, cycle number, ox/red, Ewe/V, I/mA, Capacity/mA.h
    Capacity is the LAST column in both, so it must be located by position
    from the end -- indexing a fixed column 3 silently reads Ewe/V (a voltage
    of ~3.0-4.2 V) for the 6-column files.
    """
    out = {}
    for f in glob.glob(os.path.join(base, "Capacity data", "*.txt")):
        m = re.search(r"Data_Capacity_(\d+)C(\d+)\.txt", os.path.basename(f))
        if not m:
            continue
        temp, cell = int(m.group(1)), int(m.group(2))
        d = np.loadtxt(f, skiprows=1)
        cyc, cap = d[:, 1], d[:, -1]                  # capacity = last column
        cint = cyc.astype(int)
        bnd = np.concatenate([[0], np.where(np.diff(cint) != 0)[0] + 1])
        cycles = cint[bnd]
        percap = np.maximum.reduceat(cap, bnd)        # per-cycle capacity
        if percap.size and (percap.max() < 5.0 or np.ptp(percap) < 1e-6):
            raise ValueError(
                f"{os.path.basename(f)}: capacity column looks degenerate "
                f"(max={percap.max():.3f}, range={np.ptp(percap):.3g}); "
                "check the column layout of this file."
            )
        out[(temp, cell)] = (cycles.astype(float), percap)
    return out
