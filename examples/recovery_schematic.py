"""Draw the paper's Fig. 1 schematic -> figures/recovery_schematic.{pdf,png}.

What the inverse problem actually recovers: the true physics is never inverted;
from one noisy sparse curve a fixed-form fractional equation is fitted, and only
a low-dimensional memory descriptor survives.

This is a diagram, not a computation -- no data is fitted here. It exists so that
every figure in the paper has a generator in the released package; the middle
panel's curve is the analytic sqrt(t) uptake law rather than a hand-drawn spline,
so even the illustration is the real growth law, and the check-up markers carry
the scatter and the sparsity the panel claims.

GEOMETRY.  The drawing coordinates ARE inches, and the figure is the width of
the manuscript text block, so a font size set here is the point size the reader
sees.  The previous version was 14.6 in wide and shrank by 0.44 in the column,
which put its smallest type near 4.7 pt.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from gfckit.plotting import TEXTWIDTH_IN, save_paper_figure, use_paper_style, mirror_figure

GREY = "#f0f0f0"
W, H = TEXTWIDTH_IN, 3.15          # inches; the axes span exactly this


def panel(ax, x, y, w, h, fc=GREY, ec="0.45"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.05",
                                linewidth=0.8, edgecolor=ec, facecolor=fc))


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                                 lw=1.4, color="0.1", shrinkA=0, shrinkB=0))


def verdict(ax, x, y, mark, color, title, sub):
    """One row of the 'what survives' panel: a coloured pill and two lines."""
    ax.add_patch(FancyBboxPatch((x, y - 0.075), 0.24, 0.15,
                                boxstyle="round,pad=0.012,rounding_size=0.07",
                                linewidth=0, facecolor=color))
    ax.text(x + 0.12, y + 0.005, mark, ha="center", va="center", fontsize=8,
            color="white", fontweight="bold")
    ax.text(x + 0.33, y + 0.045, title, ha="left", va="center", fontsize=8)
    ax.text(x + 0.33, y - 0.075, sub, ha="left", va="center", fontsize=7,
            color="0.35", style="italic")


def main():
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(HERE, "figures", "recovery_schematic.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    use_paper_style()
    # the axes fill the figure exactly, so a tight bbox plus padding would make
    # the saved page wider than the text block and LaTeX would then scale the
    # type DOWN; drop the padding so one drawn point stays one printed point
    plt.rcParams["savefig.pad_inches"] = 0.0
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    yb, hb = 0.78, 1.78                     # panel baseline and height
    # wide gaps between the boxes: the arrow captions live there, and at 0.4 in
    # they overprinted the boxes on both sides
    boxes = [(0.045, 1.90, "TRUE PHYSICS"),
             (2.565, 1.35, "WHAT YOU MEASURE"),
             (4.535, 1.90, "WHAT SURVIVES")]
    for x, w, head in boxes:
        panel(ax, x, yb, w, hb)
        ax.text(x + w / 2, yb + hb + 0.15, head, ha="center", va="center",
                fontsize=8.5, fontweight="bold")
    ax.text(0.995, yb + hb + 0.055, "(unknown to the solver)", ha="center",
            va="center", fontsize=7.0, color="0.4", style="italic")

    # --- left: the physics that is never inverted -------------------------
    ax.text(0.995, yb + 1.36, r"$\dfrac{\partial c}{\partial t}"
            r"=\nabla\!\cdot(D\nabla c)-R(c)$", ha="center", va="center",
            fontsize=9.5)
    ax.text(0.995, yb + 0.80, r"moving boundary  $\delta(t)$", ha="center",
            va="center", fontsize=7.5)
    ax.text(0.995, yb + 0.52, r"rate constants  $D,\,k,\,\dots$", ha="center",
            va="center", fontsize=7.5)
    ax.text(0.995, yb + 0.17,
            "solid-electrolyte interphase (SEI)\nor passive-film growth",
            ha="center", va="center", fontsize=7.0, color="0.4")

    # --- middle: one sparse noisy curve, the real sqrt(t) uptake law ------
    # the box asserts "noisy, sparse"; the curve drawn under six pristine
    # markers said neither, so the markers now carry check-up scatter
    inset = fig.add_axes([2.755 / W, (yb + 0.50) / H, 1.03 / W, 1.05 / H])
    t = np.linspace(0.0, 1.0, 300)
    inset.plot(t, np.sqrt(t), color="C0", lw=1.4, zorder=2)
    rng = np.random.default_rng(3)
    tk = np.array([0.06, 0.19, 0.36, 0.56, 0.78, 0.98])
    yk = np.sqrt(tk) * (1.0 + 0.055 * rng.standard_normal(tk.size))
    inset.plot(tk, yk, "o", mfc="white", mec="0.2", mew=0.8, ms=3.6, zorder=3)
    inset.set_xticks([]); inset.set_yticks([])
    inset.set_xlabel("time", fontsize=7, labelpad=1.5)
    inset.set_ylabel("$u(t)$", fontsize=7, labelpad=1.5)
    for s in inset.spines.values():
        s.set_edgecolor("0.35")
        s.set_linewidth(0.6)
    ax.text(3.24, yb + 0.23, "one scalar curve:\nsparse check-ups, noisy",
            ha="center", va="center", fontsize=7, color="0.3")

    # --- right: the three verdicts ----------------------------------------
    verdict(ax, 4.655, yb + 1.34, "✓", "#2e9e4f",
            r"effective order $\alpha$", "robust to 5% noise")
    verdict(ax, 4.655, yb + 0.86, "∼", "#e8a317",
            r"tempered $(\alpha,\lambda)$", "needs several conditions")
    verdict(ax, 4.655, yb + 0.38, "✗", "#cc2b2b",
            r"spectrum $w(\alpha)$", "not identifiable")

    # --- arrows between panels --------------------------------------------
    ym = yb + hb / 2
    arrow(ax, (2.055, ym), (2.475, ym))
    ax.text(2.265, ym + 0.09, "observe", ha="center", va="bottom",
            fontsize=7, style="italic", color="0.25")
    arrow(ax, (4.025, ym), (4.445, ym))
    # the fitted equation belongs to this arrow; set loose below and to the side
    # of it, it read as an unattached fragment of the figure
    ax.text(4.235, ym + 0.09, "fit", ha="center", va="bottom",
            fontsize=7, style="italic", color="0.25")
    ax.text(4.235, ym - 0.11, r"$\mathbb{D}^{(M)}u=f$", ha="center",
            va="top", fontsize=7.5)

    # --- the banner: what is not recovered --------------------------------
    panel(ax, 0.03, 0.06, W - 0.06, 0.52, fc="#fdeeee", ec="#cc2b2b")
    ax.text(W / 2, 0.42, "NOT recovered:  the governing PDE   •   the "
            "kernel shape   •   the microscopic rate constants",
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="#cc2b2b")
    ax.text(W / 2, 0.20, "Inversion collapses to a low-dimensional memory "
            "descriptor (at most an effective order).", ha="center",
            va="center", fontsize=7)

    out = save_paper_figure(fig, out)
    plt.close(fig)
    # keep the manuscript figure directory in step with its generator; the two
    # used to be reconciled by hand
    mirror_figure(out, os.path.join(HERE, os.pardir, "manuscripts",
                                    "gfc_identifiability", "figures"))
    print("wrote", out)


if __name__ == "__main__":
    main()
