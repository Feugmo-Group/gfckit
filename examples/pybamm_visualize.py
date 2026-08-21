"""Visualize the PyBaMM aging ensemble (observables + ground truth).

Writes figures/pybamm_overview.png:
  (a) capacity fade curves, colored by true dominant mechanism
  (b) ground-truth mechanism split per config
  (c) dV/dQ signatures, one representative cell per mechanism
  (d) dominant-mechanism map over config x condition (shows the flips)
"""
import os, glob, sys
import numpy as np

from gfckit.pybamm_data import load_cells
from gfckit.plotting import (plot_pybamm_overview, plot_pybamm_curves,
                             mirror_figure)

# The manuscript figure directory is written from here. It used to be refreshed
# by hand, and two figures ended up out of step with their generators.
FIG_DIRS = {"pybamm_targeted": ["manuscripts/battery_mechanisms/figures"]}


def main():
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Ensemble selected by argv[1], same convention as pybamm_identify.py, so the
    # figures and the reported numbers cannot silently come from different runs.
    # Default to the ensemble the manuscript reports, matching
    # pybamm_identify.py. The two defaults used to differ, so running both with
    # no arguments produced an overview figure and a confusion figure from two
    # different ensembles, which is exactly what the convention above forbids.
    subdir = sys.argv[1] if len(sys.argv) > 1 else "pybamm_targeted"
    tag = "" if subdir == "pybamm" else "_" + subdir.replace("pybamm_", "")
    data_dir = os.path.join(HERE, "data", subdir)
    fig_dir = os.path.join(HERE, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cells = load_cells(data_dir)
    print(f"loaded {len(cells)} cells from data/{subdir}")
    counts = {}
    for c in cells:
        counts[c["dominant"]] = counts.get(c["dominant"], 0) + 1
    print("dominant-mechanism counts:", counts)

    p1 = plot_pybamm_overview(cells, os.path.join(fig_dir,
                                                  f"pybamm_overview{tag}.pdf"))
    p2 = plot_pybamm_curves(cells, os.path.join(fig_dir, f"pybamm_curves{tag}.png"))
    # the manuscript calls this figure pybamm_overview, without the tag
    root = os.path.dirname(HERE)
    for d in FIG_DIRS.get(subdir, []):
        dest = os.path.join(root, d)
        os.makedirs(dest, exist_ok=True)
        import shutil
        for ext in (".pdf", ".png"):
            src = os.path.splitext(p1)[0] + ext
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest, "pybamm_overview" + ext))
    print(f"wrote {p1}\n      {p2}")


if __name__ == "__main__":
    main()
