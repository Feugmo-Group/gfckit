"""Draw the study pipeline diagram -> figures/pipeline.{pdf,png} (matplotlib).

Authored at the printed width (6.48 in) so the type in the figure is the size
the reader sees; see gfckit.figstyle. Axis units are inches, one-to-one with the
page, which is why the coordinates below look like a page layout.
"""
import os
import sys

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.pyplot as plt

from gfckit import figstyle as fs

W, H = fs.FULL, 2.78                     # figure size, inches
BOX_W = 2.02
ROW_A_Y, ROW_A_H = 1.70, 0.94            # model chain
ROW_B_Y, ROW_B_H = 0.44, 0.80            # identification chain + ground truth
COL_X = (0.04, 2.22, 4.40)               # left edges of the three columns


def box(ax, xy, w, h, title, body, fc):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.015,rounding_size=0.05",
                                linewidth=0.9, edgecolor="0.35", facecolor=fc))
    ax.text(x + w / 2, y + h - 0.07, title, ha="center", va="top",
            fontsize=8, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.25, body, ha="center", va="top",
            fontsize=7, linespacing=1.30)


def arrow(ax, pts, ls="-", color="0.25", lw=1.0):
    """Arrow through a polyline; a single elbow is drawn as two segments so the
    head always lands square on the target edge."""
    for a, b in zip(pts[:-1], pts[1:-1]):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-", lw=lw, color=color,
                                     linestyle=ls, shrinkA=0, shrinkB=0))
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                                 mutation_scale=9, lw=lw, color=color,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def main():
    fs.use()
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(HERE, "figures", "pipeline.pdf")

    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")

    ya, ha = ROW_A_Y, ROW_A_H
    yb, hb = ROW_B_Y, ROW_B_H

    box(ax, (COL_X[0], ya), BOX_W, ha, "1. Electrochemical model",
        "SPMe (PyBaMM)\nEqs (1)–(4): particle diffusion,\n"
        "Butler–Volmer\n$\\rightarrow$ voltage $V(t)$", "#dbe9f6")
    box(ax, (COL_X[1], ya), BOX_W, ha, "2. Degradation submodels",
        "SEI, plating, LAM, cracking\ncoupled to the model\n"
        "(full equations in the SI)", "#dbe9f6")
    box(ax, (COL_X[2], ya), BOX_W, ha, "3. Observables",
        "capacity $Q(n)$,\nlow-rate d$V$/d$Q$,\nEIS $Z(\\omega)$", "#e6f2e0")

    box(ax, (COL_X[1], yb), BOX_W, hb, "4. Identification",
        "features $\\rightarrow$ $k$-NN\nclassify mechanism /\n"
        "regress fractions", "#f6e9d8")
    box(ax, (COL_X[2], yb), BOX_W, hb, "5. Results",
        "confusion matrix, $R^2$,\nidentifiability map", "#f6dede")
    box(ax, (COL_X[0], yb), BOX_W, hb, "Ground truth (hidden)",
        "per-mechanism loss\n$Q_m(n)$, Eq. (5)\n(used only to score)", "#ededed")

    # --- the model chain, left to right ------------------------------------
    for i in (0, 1):
        x0 = COL_X[i] + BOX_W
        arrow(ax, [(x0 + 0.02, ya + ha / 2), (COL_X[i + 1] - 0.02, ya + ha / 2)])

    # --- 3 -> 4: down the right, then back left (serpentine) ---------------
    xa, xb = COL_X[2] + BOX_W / 2, COL_X[1] + BOX_W / 2
    ymid = yb + hb + 0.20
    arrow(ax, [(xa, ya - 0.02), (xa, ymid), (xb, ymid), (xb, yb + hb + 0.02)])

    # --- 4 -> 5 -------------------------------------------------------------
    arrow(ax, [(COL_X[1] + BOX_W + 0.02, yb + hb / 2),
               (COL_X[2] - 0.02, yb + hb / 2)])

    # --- 2 -> ground truth --------------------------------------------------
    xg = COL_X[0] + BOX_W / 2
    arrow(ax, [(COL_X[1] + 0.36, ya - 0.02), (COL_X[1] + 0.36, yb + hb + 0.20),
               (xg, yb + hb + 0.20), (xg, yb + hb + 0.02)])

    # --- ground truth -> results: scoring only, never an input --------------
    # The arrow elbows up into the bottom edge of the Results box; it used to
    # stop in blank space to the left of it and point at nothing.
    ys = 0.08
    xr = COL_X[2] + BOX_W / 2
    arrow(ax, [(xg, yb - 0.02), (xg, ys), (xr, ys), (xr, yb - 0.02)],
          ls=(0, (3.5, 2.5)), color="#cc2b2b", lw=1.0)
    ax.text((xg + xr) / 2, ys + 0.04, "score (never an input)", color="#cc2b2b",
            fontsize=7, ha="center", va="bottom")

    paths = fs.save(fig, out)
    fs.mirror(paths, os.path.join(os.path.dirname(HERE), "manuscripts",
                                  "battery_mechanisms", "figures"))
    print("wrote", ", ".join(paths))


if __name__ == "__main__":
    sys.exit(main())
