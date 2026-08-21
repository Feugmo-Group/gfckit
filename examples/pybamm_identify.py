"""Mechanism-recovery / identifiability experiment on the PyBaMM ensemble.

Three parts:
  1. classification of the dominant mechanism: capacity only vs capacity + dV/dQ
  2. regression of the mechanism FRACTIONS: capacity only vs capacity + dV/dQ
Adding the clean low-rate dV/dQ (DVA) fingerprint is expected to rescue the
mechanisms (cracks/LAM) that capacity alone cannot resolve.

Scoring is leave-one-CONFIGURATION-out: replicate cells of one parameter set are
held out together, so siblings cannot classify each other. Plain leave-one-out is
reported alongside it only to show the size of the bias it carries (on the
targeted ensemble it reads 46% where the honest number is 37.5%).

Reports the Wilson 95% CI on each accuracy and an exact McNemar test between the
two feature sets, both of which the manuscript quotes; they live here rather than
in a one-off script so the numbers in the paper can be regenerated.

Writes figures/pybamm_confusion{tag}.png (grouped -- this is the published
figure), figures/pybamm_confusion{tag}_ungrouped.png, and
figures/pybamm_fraction_r2{tag}.png.
"""
import os
import sys
import numpy as np

from gfckit.pybamm_data import load_cells
from gfckit.mechanism_id import (build_matrix, loo_knn, confusion,
                                 mechanism_fractions, loo_knn_regress,
                                 grouped_knn, grouped_knn_regress,
                                 groups_from_ids, r2_per_column,
                                 wilson_interval, mcnemar_exact, MECHS)
from gfckit.plotting import plot_confusion_pair, plot_fraction_r2, mirror_figure

FIG_DIRS = {"pybamm_targeted": ["manuscripts/battery_mechanisms/figures"]}


def main():
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Default to the ensemble the manuscript reports, so running this script with
    # no arguments reproduces the published numbers.
    subdir = sys.argv[1] if len(sys.argv) > 1 else "pybamm_targeted"
    tag = "" if subdir == "pybamm" else "_" + subdir.replace("pybamm_", "")
    data_dir = os.path.join(HERE, "data", subdir)
    cells = load_cells(data_dir)
    have_dvdq = all("dvdq" in c and c["dvdq"].size > 5 for c in cells)
    print(f"{len(cells)} cells; clean dV/dQ available: {have_dvdq}")

    # ---- 1. classification: capacity vs capacity + dV/dQ ----
    Xa, y = build_matrix(cells, use_capacity=True, use_dvdq=False)
    Xb, _ = build_matrix(cells, use_capacity=True, use_dvdq=True)
    groups = groups_from_ids(cells)
    print(f"{len(set(groups))} configurations over {len(cells)} cells")

    # grouped -- the honest protocol, and the one the paper reports
    pred_a = grouped_knn(Xa, y, groups, k=3)
    pred_b = grouped_knn(Xb, y, groups, k=3)
    Ca, acc_a = confusion(y, pred_a)
    Cb, acc_b = confusion(y, pred_b)
    print(f"\n[classification, leave-one-configuration-out]"
          f"  capacity: {acc_a:.1%}   capacity+dV/dQ: {acc_b:.1%}")

    # The accuracies alone cannot say whether the drop is real: both feature
    # sets are scored on the same cells, so the comparison is paired.
    n = len(y)
    for lbl, acc in (("capacity", acc_a), ("capacity+dV/dQ", acc_b)):
        lo, hi = wilson_interval(int(round(acc * n)), n)
        print(f"    {lbl:<16} {acc:.1%}   Wilson 95% CI [{lo:.1%}, {hi:.1%}]")
    b, c, p_mc = mcnemar_exact(pred_a == y, pred_b == y)
    print(f"    McNemar: {b} cells only capacity got right, {c} only "
          f"capacity+dV/dQ; exact two-sided p = {p_mc:.3f}")
    print(f"    majority baseline: {max(np.bincount(np.unique(y, return_inverse=True)[1]))}"
          f"/{n} = {max(np.bincount(np.unique(y, return_inverse=True)[1]))/n:.1%}")
    print("  per-class recall (capacity+dV/dQ):")
    for i, m in enumerate(MECHS):
        n = Cb[i].sum()
        print(f"    {m:8s}: {Cb[i,i]}/{n}" if n else f"    {m:8s}: (none)")
    f1 = plot_confusion_pair(Ca, acc_a, "capacity only",
                             Cb, acc_b, r"capacity $+$ d$V$/d$Q$", MECHS,
                             os.path.join(HERE, "figures", f"pybamm_confusion{tag}.pdf"),
                             protocol="leave-one-configuration-out")
    mirror_figure(f1, *[os.path.join(os.path.dirname(HERE), d)
                        for d in FIG_DIRS.get(subdir, [])])

    # plain leave-one-out, kept only to quantify the bias it introduces
    Ua, u_a = confusion(y, loo_knn(Xa, y, k=3))
    Ub, u_b = confusion(y, loo_knn(Xb, y, k=3))
    print(f"[classification, plain leave-one-out (BIASED)]"
          f"  capacity: {u_a:.1%}   capacity+dV/dQ: {u_b:.1%}")
    f1b = plot_confusion_pair(Ua, u_a, "capacity only",
                              Ub, u_b, r"capacity $+$ d$V$/d$Q$", MECHS,
                              os.path.join(HERE, "figures",
                                           f"pybamm_confusion{tag}_ungrouped.pdf"),
                              protocol="plain leave-one-out, biased by replicates")

    # ---- 2. fraction regression: capacity vs capacity + dV/dQ ----
    Y = np.array([mechanism_fractions(c) for c in cells])
    r2_cap = r2_per_column(Y, grouped_knn_regress(Xa, Y, groups, k=3))
    r2_dvdq = r2_per_column(Y, grouped_knn_regress(Xb, Y, groups, k=3))
    print("\n[fraction R^2]     ", "  ".join(f"{m}" for m in MECHS))
    print("  capacity     :  " + "  ".join(f"{v:+.2f}" for v in r2_cap))
    print("  capacity+dvdq:  " + "  ".join(f"{v:+.2f}" for v in r2_dvdq))
    f2 = plot_fraction_r2(MECHS, r2_cap, r2_dvdq,
                          os.path.join(HERE, "figures", f"pybamm_fraction_r2{tag}.png"))

    print(f"\nwrote {f1}\n      {f1b}\n      {f2}")


if __name__ == "__main__":
    main()
