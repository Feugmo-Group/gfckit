"""Statistical rigor report for the revision (addresses reviewer concerns):
sample sizes, majority baseline, Wilson CIs on accuracy, permutation-null p-value,
McNemar test for capacity vs capacity+dV/dQ, real-cell correlation p/CI/Spearman,
and clustering adjusted Rand index + permutation purity null.
"""
import os
import numpy as np
from scipy import stats

from gfckit.pybamm_data import load_cells
from gfckit.mechanism_id import (build_matrix, loo_knn, grouped_knn, confusion,
                                 mechanism_fractions, loo_knn_regress,
                                 grouped_knn_regress,
                                 r2_per_column, observable_features, MECHS)
from gfckit.real_data import load_zhang_capacity, load_zhang_eis, eis_features
from scipy.cluster.vq import kmeans2


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cells = load_cells(os.path.join(HERE, "data", "pybamm_targeted"))
    y = np.array([c["dominant"] for c in cells])
    N = len(y)
    vals, counts = np.unique(y, return_counts=True)
    maj = counts.max() / N
    print(f"=== SYNTHETIC CLASSIFIER (n={N}) ===")
    print("per-class:", dict(zip(vals, counts)), f" majority baseline={maj:.0%}")

    Xa, _ = build_matrix(cells, use_capacity=True, use_dvdq=False)
    Xb, _ = build_matrix(cells, use_capacity=True, use_dvdq=True)

    # PRIMARY protocol: leave-one-CONFIGURATION-out. Replicate cells of one
    # parameter set share a label, so a cell-level split lets a held-out cell be
    # classified by its own siblings. The manuscript reports the grouped numbers.
    groups = np.array([(c.get("cell_id") or c.get("id")).split("__")[0] for c in cells])
    pa, pb = grouped_knn(Xa, y, groups, 3), grouped_knn(Xb, y, groups, 3)
    acc_a, acc_b = (pa == y).mean(), (pb == y).mean()
    ka, kb = int((pa == y).sum()), int((pb == y).sum())
    print(f"[leave-one-configuration-out, {len(set(groups))} configs]  <-- manuscript")
    print(f"capacity: acc={acc_a:.3f} ({ka}/{N}) 95%CI={tuple(round(x,2) for x in wilson(ka,N))}")
    print(f"cap+dVdQ: acc={acc_b:.3f} ({kb}/{N}) 95%CI={tuple(round(x,2) for x in wilson(kb,N))}")

    # Biased comparison, reported in the manuscript only to quantify the inflation.
    la, lb = loo_knn(Xa, y, 3), loo_knn(Xb, y, 3)
    print(f"[leave-one-CELL-out, biased by replicate structure]")
    print(f"capacity: acc={(la==y).mean():.3f}   cap+dVdQ: acc={(lb==y).mean():.3f}")

    # Configuration-level permutation null: permute whole configuration blocks,
    # not individual cells, or the null is inflated to ~32% and meaningless.
    ug = np.unique(groups)
    blocks = [np.where(groups == g)[0] for g in ug]
    for nm, X, acc in (("capacity", Xa, acc_a), ("cap+dVdQ", Xb, acc_b)):
        rng = np.random.default_rng(0)
        null = []
        for _ in range(10000):
            order = rng.permutation(len(ug))
            ys = y.copy()
            for b, src in zip(blocks, order):
                sb = blocks[src]
                ys[b] = y[sb][:len(b)] if len(sb) >= len(b) else np.resize(y[sb], len(b))
            null.append((grouped_knn(X, ys, groups, 3) == ys).mean())
        null = np.array(null)
        p_perm = (1 + (null >= acc).sum()) / (1 + len(null))
        print(f"permutation null ({nm}, 1e4 perms): mean={null.mean():.3f}, p={p_perm:.3f}")

    # McNemar: cap vs cap+dvdq
    b = int(((pa != y) & (pb == y)).sum())  # cap wrong, dvdq right
    c = int(((pa == y) & (pb != y)).sum())  # cap right, dvdq wrong
    p_mcn = stats.binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    print(f"McNemar cap vs cap+dvdq: b={b} c={c} exact p={p_mcn:.3f}")

    # per-mechanism fraction R^2
    Y = np.array([mechanism_fractions(c) for c in cells])
    # Grouped, to match the classification protocol and the manuscript.
    r2a = r2_per_column(Y, grouped_knn_regress(Xa, Y, groups, 3))
    r2b = r2_per_column(Y, grouped_knn_regress(Xb, Y, groups, 3))
    print("fraction R2 (cap / cap+dvdq):")
    for i, m in enumerate(MECHS):
        print(f"   {m:8s} {r2a[i]:+.2f} / {r2b[i]:+.2f}")

    # === REAL: Zhang R_ct vs fade ===
    base = os.path.join(HERE, "data", "real", "zhang2020")
    cap = load_zhang_capacity(base)
    eis = load_zhang_eis(base)
    common = sorted(set(cap) & set(eis))
    fade, rct = [], []
    for k in common:
        cyc, cp = cap[k]
        fade.append(1 - cp[-1] / cp[0])
        rct.append(eis_features(*eis[k][-1][1:])["R_ct"])
    fade, rct = np.array(fade), np.array(rct)
    r, p = stats.pearsonr(fade, rct)
    rho, prho = stats.spearmanr(fade, rct)
    zc = np.arctanh(r); se = 1 / np.sqrt(len(fade) - 3)
    ci = (np.tanh(zc - 1.96 * se), np.tanh(zc + 1.96 * se))
    print(f"\n=== REAL Zhang (n={len(fade)}) ===")
    print(f"R_ct vs fade: Pearson r={r:.2f} p={p:.3f} 95%CI=({ci[0]:.2f},{ci[1]:.2f}); "
          f"Spearman rho={rho:.2f} p={prho:.3f}")
    print("  (note: 6 EIS-change/absolute features were scanned; R_ct chosen by |r|)")

    # clustering adjusted Rand + permutation purity
    temps = np.array([k[0] for k in common])
    X = []
    for k in common:
        cyc, cp = cap[k]
        cf, _ = observable_features({"cycle": cyc, "retention": cp / cp[0]})
        ag = eis_features(*eis[k][-1][1:])
        fr = eis_features(*eis[k][0][1:])
        X.append([*cf, ag["R_s"], ag["R_ct"], ag["peak"],
                  ag["R_ct"] - fr["R_ct"], ag["R_s"] - fr["R_s"]])
    X = np.array(X); Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    _, km = kmeans2(Xs, 3, seed=0, minit="++")

    def purity(cl, lab):
        return sum(max(np.bincount(lab[cl == c]).max() if (cl == c).any() else 0
                       for c in set(cl)) for _ in [0]) / len(lab)
    tint = np.unique(temps, return_inverse=True)[1]
    def pur(cl):
        s = 0
        for c in set(cl):
            s += np.bincount(tint[cl == c]).max()
        return s / len(cl)
    obs_pur = pur(km)
    ari = _adjusted_rand(km, tint)
    rng2 = np.random.default_rng(1)
    nullp = np.array([pur(rng2.permutation(km)) for _ in range(2000)])
    print(f"\n=== REAL clustering (n={len(temps)}) ===")
    print(f"k-means purity={obs_pur:.2f} (majority baseline={np.bincount(tint).max()/len(tint):.2f}); "
          f"adjusted Rand={ari:.2f}; permutation p={(nullp>=obs_pur).mean():.3f}")


def _adjusted_rand(a, b):
    from itertools import combinations
    n = len(a)
    idx = list(combinations(range(n), 2))
    same_a = np.array([a[i] == a[j] for i, j in idx])
    same_b = np.array([b[i] == b[j] for i, j in idx])
    tp = np.sum(same_a & same_b); tn = np.sum(~same_a & ~same_b)
    fp = np.sum(same_a & ~same_b); fn = np.sum(~same_a & same_b)
    # Adjusted Rand
    from math import comb
    return 2 * (tp * tn - fp * fn) / ((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn) + 1e-12)


if __name__ == "__main__":
    main()
