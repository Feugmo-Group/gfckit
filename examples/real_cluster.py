"""Unsupervised structure discovery on the real cells (Zhang et al. 2020).

We build a combined capacity + EIS feature vector per cell, standardize, and ask
(without using any labels) whether the cells organize into groups --- and whether
those groups line up with temperature / knee behaviour. PCA embedding + k-means +
Ward hierarchical clustering.

Writes figures/real_clustering.png.
"""
import os
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.cluster.hierarchy import linkage, fcluster

from gfckit.real_data import load_zhang_capacity, load_zhang_eis, eis_features
from gfckit.mechanism_id import observable_features
from gfckit.plotting import plot_real_clustering


def main():
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = os.path.join(HERE, "data", "real", "zhang2020")
    out = os.path.join(HERE, "figures", "real_clustering.png")

    cap = load_zhang_capacity(base)
    eis = load_zhang_eis(base)
    common = sorted(set(cap) & set(eis))

    X, temps, labels = [], [], []
    for key in common:
        cyc, cp = cap[key]
        ret = cp / cp[0]
        cf, names = observable_features({"cycle": cyc, "retention": ret})
        st = eis[key]
        fr, ag = eis_features(*st[0][1:]), eis_features(*st[-1][1:])
        X.append([*cf, ag["R_s"], ag["R_ct"], ag["peak"],
                  ag["R_ct"] - fr["R_ct"], ag["R_s"] - fr["R_s"]])
        temps.append(key[0])
        labels.append(f"{key[0]}C-{key[1]:02d}")
    X = np.array(X)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)

    # PCA (SVD)
    Uc = Xs - Xs.mean(0)
    U, S, Vt = np.linalg.svd(Uc, full_matrices=False)
    pc = U[:, :2] * S[:2]
    var = (S ** 2) / (S ** 2).sum()
    print(f"{len(common)} cells; PCA variance explained PC1,PC2 = "
          f"{var[0]:.0%}, {var[1]:.0%}")

    # k-means (k = 3, matching the three temperatures) and Ward linkage
    _, km = kmeans2(Xs, 3, seed=0, minit="++")
    Z = linkage(Xs, method="ward")
    hc = fcluster(Z, t=3, criterion="maxclust")

    # how well does unsupervised clustering line up with temperature?
    def purity(clusters):
        tot = 0
        for c in set(clusters):
            ts = [temps[i] for i in range(len(temps)) if clusters[i] == c]
            tot += max([ts.count(t) for t in set(ts)])
        return tot / len(temps)
    print(f"k-means vs temperature purity = {purity(km):.0%}")
    print(f"Ward     vs temperature purity = {purity(hc):.0%}")

    path = plot_real_clustering(pc, temps, km, Z, labels, out)
    print("\nInterpretation: if clusters align with temperature, the combined")
    print("capacity+EIS signature encodes the aging condition without supervision.")
    print(f"wrote figure -> {path}")


if __name__ == "__main__":
    main()
