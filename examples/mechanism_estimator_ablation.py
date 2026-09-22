"""Estimator ablation and standardisation check for the mechanism-recovery result.

The paper defends k-NN (k=3) as deliberately weak, so that a negative result
bounds the observable rather than the classifier.  The defence is only complete
if a stronger estimator does not overturn it, and that was never tested.  This
script re-scores the same labelled ensemble under a range of classifiers, all
under the same leave-one-configuration-out protocol, and reports a permutation
null for the best of them.

It also checks one implementation detail.  `mechanism_id.grouped_knn`
standardises features on the whole ensemble before the fold loop, so the held-out
configuration contributes to the mean and variance used to scale it.  That is a
(mild) leak.  Every estimator here is run both ways: `global` reproduces the
published pipeline, `fold` refits the scaler on the training split alone.

Usage:  python examples/mechanism_estimator_ablation.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gfckit.mechanism_id import build_matrix, groups_from_ids           # noqa: E402
from gfckit.pybamm_data import load_cells                               # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "pybamm_targeted")
N_PERM = 10000
SEED = 20260909


def estimators():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
    from sklearn.tree import DecisionTreeClassifier
    return [
        ("k-NN k=1", lambda: KNeighborsClassifier(n_neighbors=1)),
        ("k-NN k=3 (sklearn)", lambda: KNeighborsClassifier(n_neighbors=3)),
        ("k-NN k=5", lambda: KNeighborsClassifier(n_neighbors=5)),
        ("nearest centroid", lambda: NearestCentroid()),
        ("LDA", lambda: LinearDiscriminantAnalysis()),
        ("logistic regression", lambda: LogisticRegression(max_iter=5000)),
        ("gaussian naive Bayes", lambda: GaussianNB()),
        ("decision tree d=3", lambda: DecisionTreeClassifier(max_depth=3,
                                                             random_state=0)),
        ("random forest", lambda: RandomForestClassifier(n_estimators=500,
                                                         random_state=0)),
    ]


def loco_score(make, X, y, groups, scaling="global"):
    """Leave-one-configuration-out accuracy for one estimator."""
    pred = np.empty(len(y), dtype=object)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        Xtr, Xte = X[tr], X[te]
        if scaling == "global":
            mu, sd = X.mean(0), X.std(0) + 1e-9          # leaks the held-out fold
        else:
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9      # training split only
        clf = make()
        clf.fit((Xtr - mu) / sd, y[tr])
        pred[te] = clf.predict((Xte - mu) / sd)
    return float((pred == y).mean()), pred


def permutation_null(make, X, y, groups, scaling, n=N_PERM, seed=SEED):
    """Permute labels between configurations as whole blocks."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    obs, _ = loco_score(make, X, y, groups, scaling)
    block = {g: y[groups == g] for g in ug}
    null = np.empty(n)
    for i in range(n):
        order = rng.permutation(len(ug))
        yp = np.empty(len(y), dtype=object)
        for g, src in zip(ug, ug[order]):
            tgt = np.where(groups == g)[0]
            s = block[src]
            yp[tgt] = s[:len(tgt)] if len(s) >= len(tgt) else np.resize(s, len(tgt))
        null[i], _ = loco_score(make, X, yp, groups, scaling)
    p = float((null >= obs).sum() + 1) / (n + 1)
    return obs, float(null.mean()), p


def main():
    cells = load_cells(DATA)
    groups = np.asarray(groups_from_ids(cells))
    sets = {
        "capacity": build_matrix(cells, use_capacity=True, use_dvdq=False),
        "capacity + dV/dQ": build_matrix(cells, use_capacity=True, use_dvdq=True),
    }
    y = sets["capacity"][1]
    print(f"{len(y)} cells, {len(np.unique(groups))} configurations, "
          f"{len(set(y))} classes")
    for name, (X, _) in sets.items():
        print(f"  {name}: d = {X.shape[1]}")
    counts = {c: int((y == c).sum()) for c in sorted(set(y))}
    print(f"  class counts: {counts}  majority baseline "
          f"= {max(counts.values())/len(y)*100:.1f}%\n")

    from gfckit.mechanism_id import grouped_knn

    rows = []
    for fname, (X, _) in sets.items():
        print(f"=== {fname} ===")
        print(f"{'estimator':24s} {'global scaling':>15s} {'fold scaling':>14s}")
        # The paper's own estimator: k-NN k=3 with a deterministic tie-break by
        # summed distance. sklearn's k-NN breaks ties by class order instead, so
        # the two differ on this small, tie-prone problem; both are shown.
        a_paper = float((grouped_knn(X, y, groups, k=3) == y).mean())
        print(f"{'k-NN k=3 (paper, gfckit)':24s} {a_paper*100:14.1f}% "
              f"{'(global only)':>14s}")
        rows.append((fname, "k-NN k=3 (paper, gfckit)", a_paper, np.nan))
        for label, make in estimators():
            a_g, _ = loco_score(make, X, y, groups, "global")
            a_f, _ = loco_score(make, X, y, groups, "fold")
            print(f"{label:24s} {a_g*100:14.1f}% {a_f*100:13.1f}%")
            rows.append((fname, label, a_g, a_f))
        print()

    # Selecting the best of several estimators and then testing it inflates
    # significance. Report both the uncorrected p for the winner and a
    # max-statistic p, in which each permutation contributes the best accuracy
    # over the whole family -- the correct null for "we tried them all".
    X = sets["capacity"][0]
    fam = estimators()
    cand = [(a, lbl) for (fn, lbl, a, _) in rows
            if fn == "capacity" and lbl != "k-NN k=3 (paper, gfckit)"]
    best_acc, best_lbl = max(cand)
    make = dict(fam)[best_lbl]

    print(f"best estimator on capacity features: {best_lbl} at {best_acc*100:.1f}%")
    obs, nullmean, p = permutation_null(make, X, y, groups, "global", n=N_PERM)
    print(f"  uncorrected: null mean {nullmean*100:.1f}%   p = {p:.3f}   "
          f"({N_PERM} permutations)")

    n_max = 2000
    rng = np.random.default_rng(SEED + 1)
    ug = np.unique(groups)
    block = {g: y[groups == g] for g in ug}
    null_max = np.empty(n_max)
    for i in range(n_max):
        order = rng.permutation(len(ug))
        yp = np.empty(len(y), dtype=object)
        for g, src in zip(ug, ug[order]):
            tgt = np.where(groups == g)[0]
            s = block[src]
            yp[tgt] = s[:len(tgt)] if len(s) >= len(tgt) else np.resize(s, len(tgt))
        null_max[i] = max(loco_score(mk, X, yp, groups, "global")[0]
                          for _, mk in fam)
    p_max = float((null_max >= best_acc).sum() + 1) / (n_max + 1)
    print(f"  max-statistic over the {len(fam)} estimators: "
          f"null mean of the family maximum {null_max.mean()*100:.1f}%   "
          f"p = {p_max:.3f}   ({n_max} permutations)")


if __name__ == "__main__":
    main()
