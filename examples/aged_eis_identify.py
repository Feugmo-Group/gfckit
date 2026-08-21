"""Does per-cell AGED impedance rescue mechanism attribution?

Leave-one-CONFIGURATION-out (not leave-one-cell-out: the ensemble has 3 sibling
cells per configuration, and letting siblings vote inflates accuracy from 37.5%
to 45.8% on capacity alone).

Compares capacity / dV/dQ / aged-EIS feature sets on the same 24 cells, against
a permutation null computed with the same grouping.
"""
import os, sys, csv, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gfckit.pybamm_data import load_cells
from gfckit.mechanism_id import (features_for, mechanism_fractions,
                                 r2_per_column, MECHS)

HERE = os.path.dirname(__file__)
EIS_SUM = os.path.join(HERE, "..", "data", "aged_eis", "summary.csv")
EIS_FEATS = ["R_ct", "peak_negZim", "f_peak", "Zre_lo", "negZim_lo"]  # R_s is constant
RNG = np.random.default_rng(0)


def grouped_knn(X, y, groups, k=3):
    """Leave-one-group-out k-NN; deterministic tie-break by summed distance."""
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
                n = int((lab == L).sum()); s = float(dist[lab == L].sum())
                if n > bestn or (n == bestn and s < bestd):
                    best, bestn, bestd = L, n, s
            pred[i] = best
    return pred


def grouped_knn_regress(X, Y, groups, k=3):
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    P = np.zeros_like(Y)
    for g in np.unique(groups):
        te = groups == g; tr = ~te
        for i in np.where(te)[0]:
            d = np.sqrt(((Xs[tr] - Xs[i]) ** 2).sum(1))
            P[i] = Y[tr][np.argsort(d)[:k]].mean(0)
    return P


def main():
    cells = load_cells(os.path.join(HERE, "..", "data", "pybamm_targeted"))
    eis = {r["cell_id"]: r for r in csv.DictReader(open(EIS_SUM))
           if r["status"] == "ok"}

    keep, groups, y = [], [], []
    for c in cells:
        cid = c.get("cell_id") or c.get("id")
        if cid in eis:
            keep.append(c); groups.append(cid.split("__")[0]); y.append(c["dominant"])
    groups = np.array(groups); y = np.array(y)
    print("cells=%d  configs=%d" % (len(keep), len(set(groups))))
    counts = {L: int((y == L).sum()) for L in sorted(set(y))}
    maj = max(counts.values()) / len(y)
    print("class counts:", counts, " majority baseline=%.1f%%" % (100 * maj))

    def eismat(cs):
        return np.array([[float(eis[c.get("cell_id") or c.get("id")][f])
                          for f in EIS_FEATS] for c in cs])

    # features_for returns (vector, names) -- take the vector
    Xcap = np.array([features_for(c, use_capacity=True, use_dvdq=False)[0] for c in keep])
    Xdv = np.array([features_for(c, use_capacity=True, use_dvdq=True)[0] for c in keep])
    Xeis = eismat(keep)
    SETS = {
        "capacity":                 Xcap,
        "capacity + dV/dQ":         Xdv,
        "aged EIS only":            Xeis,
        "capacity + aged EIS":      np.hstack([Xcap, Xeis]),
        "capacity + dV/dQ + EIS":   np.hstack([Xdv, Xeis]),
    }

    print("\n=== dominant-mechanism attribution (leave-one-configuration-out) ===")
    print("%-26s %8s   %s" % ("features", "accuracy", "permutation p"))
    for name, X in SETS.items():
        p = grouped_knn(X, y, groups)
        acc = float((p == y).mean())
        # Permute at the GROUP level. Permuting WITHIN groups is nearly a no-op,
        # because sibling cells of one configuration almost always share a label;
        # that inflates the null to ~32% and makes the p-value meaningless.
        ug = np.unique(groups)
        blocks = [np.where(groups == g)[0] for g in ug]
        null = []
        for _ in range(1000):
            order = RNG.permutation(len(ug))
            yp = y.copy()
            for b, src in zip(blocks, order):
                sb = blocks[src]
                yp[b] = y[sb][:len(b)] if len(sb) >= len(b) else np.resize(y[sb], len(b))
            null.append(float((grouped_knn(X, yp, groups) == yp).mean()))
        pv = (1 + sum(n >= acc for n in null)) / (1 + len(null))
        print("%-26s %7.1f%%   p=%.3f  (null mean %.1f%%)"
              % (name, 100 * acc, pv, 100 * np.mean(null)))

    print("\n=== per-mechanism recall (capacity vs capacity+EIS) ===")
    pc = grouped_knn(Xcap, y, groups)
    pe = grouped_knn(SETS["capacity + aged EIS"], y, groups)
    for L in sorted(set(y)):
        m = y == L
        print("  %-8s n=%d   capacity %d/%d   capacity+EIS %d/%d"
              % (L, m.sum(), (pc[m] == L).sum(), m.sum(), (pe[m] == L).sum(), m.sum()))

    print("\n=== mechanism-fraction R^2 (leave-one-configuration-out) ===")
    Y = np.array([mechanism_fractions(c) for c in keep])
    print("%-26s %s" % ("features", "  ".join("%9s" % m for m in MECHS)))
    for name in ("capacity", "capacity + aged EIS"):
        P = grouped_knn_regress(SETS[name], Y, groups)
        r2 = r2_per_column(Y, P)
        print("%-26s %s" % (name, "  ".join("%9.3f" % v for v in r2)))


if __name__ == "__main__":
    main()
