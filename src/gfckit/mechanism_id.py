"""Mechanism recovery from OBSERVABLES only + identifiability scoring (numpy).

Given the PyBaMM ensemble, extract features from what an experimenter could
measure (the capacity-fade curve; dV/dQ hooks are provided for phase 2) and try
to recover the ground-truth dominant mechanism.  We compare feature sets by
leave-one-out cross-validation to see what is and is not identifiable.
"""
import numpy as np

MECHS = ["SEI", "plating", "cracks", "LAM"]


MIN_FADE_FOR_SHAPE = 1e-3   # below this the fade curve has no resolvable shape


def observable_features(cell):
    """Features from the capacity-fade OBSERVABLE only.
    Returns (vector, names). Shape features capture SEI(~sqrt) vs plating(accel.).

    The three SHAPE features (area, late/early ratio, log-log exponent) are
    undefined for a cell that has not measurably faded, and are returned as NaN
    rather than a default.  A hard-coded default (previously 1.0) is
    indistinguishable from a genuine fit and will be silently averaged or
    plotted as if it were a measurement.  Callers must drop or impute NaNs
    explicitly; see `is_degenerate`.
    """
    n = np.asarray(cell["cycle"], float)
    loss = 1.0 - np.asarray(cell["retention"], float)
    ft = float(max(loss[-1], 0.0))                      # total fade (magnitude)
    if ft > MIN_FADE_FOR_SHAPE and n[-1] > n[0]:
        x = (n - n[0]) / (n[-1] - n[0])
        y = loss / ft
        area = float(np.mean(y - x))                    # >0 accelerating, <0 sqrt-like
        mid = len(loss) // 2
        early = loss[mid] - loss[0]
        late = loss[-1] - loss[mid]
        ratio = float(late / (early + 1e-9))            # late/early fade-rate ratio
        mask = (loss > 1e-6) & (n > 0)
        p = (float(np.polyfit(np.log(n[mask]), np.log(loss[mask]), 1)[0])
             if mask.sum() > 5 else np.nan)             # log-log exponent (loss~n^p)
    else:
        area, ratio, p = np.nan, np.nan, np.nan
    return (np.array([ft, area, ratio, p]),
            ["fade_total", "shape_area", "late_early_ratio", "loglog_exponent"])


def is_degenerate(cell, min_fade=MIN_FADE_FOR_SHAPE):
    """True if the cell has not faded enough for its shape features to be defined."""
    loss = 1.0 - np.asarray(cell["retention"], float)
    return not (float(max(loss[-1], 0.0)) > min_fade)


def dvdq_features(cell, n=8):
    """Resampled dV/dQ (differential voltage) on normalized discharge capacity.
    This is the clean low-rate DVA fingerprint (needs run_cell rpt=True)."""
    Q = np.asarray(cell.get("dvdq_Q", []), float)
    y = np.asarray(cell.get("dvdq", []), float)
    if Q.size < 5:
        return np.zeros(n)
    x = (Q - Q.min()) / (Q.max() - Q.min() + 1e-9)
    o = np.argsort(x)
    x, y = x[o], np.clip(y[o], -10.0, 2.0)
    return np.interp(np.linspace(0.05, 0.95, n), x, y)


def features_for(cell, use_capacity=True, use_dvdq=False, n_dvdq=8):
    parts, names = [], []
    if use_capacity:
        v, nm = observable_features(cell)
        parts.append(v); names += nm
    if use_dvdq:
        parts.append(dvdq_features(cell, n_dvdq))
        names += [f"dvdq_{i}" for i in range(n_dvdq)]
    return np.concatenate(parts), names


def build_matrix(cells, feature_idx=None, use_capacity=True, use_dvdq=False):
    X = np.array([features_for(c, use_capacity, use_dvdq)[0] for c in cells])
    if feature_idx is not None:
        X = X[:, feature_idx]
    if X.ndim == 1:
        X = X[:, None]
    y = np.array([c["dominant"] for c in cells])
    return X, y


# ---------------- mechanism-FRACTION regression ----------------
def mechanism_fractions(cell, labels=MECHS):
    """Ground-truth fraction of end-of-life capacity loss per mechanism."""
    c = cell["contrib"]
    tot = sum(max(c[m], 0.0) for m in labels)
    return np.array([max(c[m], 0.0) / tot if tot > 1e-12 else 0.0 for m in labels])


def loo_knn_regress(X, Y, k=3):
    """Leave-one-out k-NN regression (predict fraction vectors)."""
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    P = np.zeros_like(Y)
    for i in range(len(Y)):
        d = np.sqrt(((Xs - Xs[i]) ** 2).sum(1))
        d[i] = np.inf
        idx = np.argsort(d)[:k]
        P[i] = Y[idx].mean(0)
    return P


def r2_per_column(Y, P):
    ss_res = ((Y - P) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss_res / ss_tot


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k/n.

    Preferred to the normal approximation here because the counts are small
    (n = 24) and the accuracies are far from 0.5, where the Wald interval both
    misses its nominal coverage and can run outside [0, 1)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return centre - half, centre + half


def mcnemar_exact(correct_a, correct_b):
    """Exact (binomial) McNemar test for two classifiers scored on the same items.

    Returns (b, c, p) where b counts items only A got right, c only B. The
    accuracies alone cannot say whether a difference is real, because the two
    feature sets are evaluated on the same cells and their errors are paired;
    only the discordant pairs (b, c) carry information.
    """
    a = np.asarray(correct_a, dtype=bool)
    b_arr = np.asarray(correct_b, dtype=bool)
    b = int((a & ~b_arr).sum())
    c = int((~a & b_arr).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    # two-sided exact binomial test against p = 1/2
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, 2.0 * tail)


def groups_from_ids(cells, sep="__"):
    """Configuration label for each cell: the part of its id before `sep`.
    Replicates of one parameter set share a configuration, and that is the unit
    which must be held out together."""
    out = []
    for c in cells:
        cid = str(c.get("cell_id", c.get("id", "")))
        out.append(cid.split(sep)[0])
    return np.array(out)


def grouped_knn(X, y, groups, k=3):
    """Leave-one-GROUP-out k-NN; deterministic tie-break by summed distance.

    Prefer this to `loo_knn` whenever cells come in replicate sets. Holding out
    a single cell lets its siblings---same parameters, near-identical curves---
    vote for it, which inflates accuracy substantially: on the targeted PyBaMM
    ensemble 37.5% becomes 46%. The inflated number measures whether replicates
    resemble each other, not whether the mechanism is identifiable.
    """
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    pred = np.empty(len(y), dtype=object)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        for i in np.where(te)[0]:
            d = np.sqrt(((Xs[tr] - Xs[i]) ** 2).sum(1))
            idx = np.argsort(d)[:k]
            lab, dist = y[tr][idx], d[idx]
            best, bestn, bestd = None, -1, np.inf
            for L in set(lab):
                n = int((lab == L).sum())
                s = float(dist[lab == L].sum())
                if n > bestn or (n == bestn and s < bestd):
                    best, bestn, bestd = L, n, s
            pred[i] = best
    return pred


def grouped_knn_regress(X, Y, groups, k=3):
    """Leave-one-group-out k-NN regression, the counterpart of `grouped_knn`."""
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    P = np.zeros_like(Y)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        for i in np.where(te)[0]:
            d = np.sqrt(((Xs[tr] - Xs[i]) ** 2).sum(1))
            P[i] = Y[tr][np.argsort(d)[:k]].mean(0)
    return P


def loo_knn(X, y, k=3):
    """Leave-one-out k-NN prediction with z-scored features (numpy only).

    With replicate cells this is optimistically biased; see `grouped_knn`,
    which is what the published numbers use."""
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    preds = []
    for i in range(len(y)):
        d = np.sqrt(((Xs - Xs[i]) ** 2).sum(1))
        d[i] = np.inf
        idx = np.argsort(d)[:k]
        vals, counts = np.unique(y[idx], return_counts=True)
        preds.append(vals[np.argmax(counts)])
    return np.array(preds)


def confusion(y_true, y_pred, labels=MECHS):
    C = np.zeros((len(labels), len(labels)), int)
    idx = {m: i for i, m in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            C[idx[t], idx[p]] += 1
    acc = float(np.trace(C) / max(C.sum(), 1))
    return C, acc
