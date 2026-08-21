"""Draw the battery-paper Fig. 2 schematic -> figures/battery_recovery.{pdf,png}.

What the pipeline recovers and what it does not: the hidden SPMe physics on the
left is never inverted; from three observables in the centre only a
dominant-mechanism label and the mixing fractions survive on the right.

Distinct from examples/recovery_schematic.py, which draws the *identifiability*
paper's version of the same idea; the two manuscripts previously kept two
different figures under one file name, and only one of them had a generator.

The three centre panels are schematic sketches of the observables, not data, and
are drawn in neutral grey so that colour means "mechanism" everywhere in the
paper and nothing else. The caption says so as well.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from gfckit import figstyle as fs

W, H = fs.FULL, 3.10
PANEL_Y, PANEL_H = 0.72, 1.86            # the three grey panels
PANELS = [(0.03, 2.10, "TRUE PHYSICS"),          # (x, width, heading)
          (2.60, 1.28, "WHAT YOU MEASURE"),
          (4.35, 2.10, "WHAT SURVIVES")]
GREY = "#f2f2f2"
INK = "0.12"


def shrink_to_fit(fig, txt, max_w_in, floor=6.4):
    """Reduce a label's size until it fits ``max_w_in``, and say so if it cannot.

    Several labels in the previous version of this figure ran off the edge of
    the panel they belonged to. Measuring them here means the layout cannot
    silently regress when a string is edited.
    """
    r = fig.canvas.get_renderer()
    while txt.get_fontsize() > floor:
        w = txt.get_window_extent(renderer=r).width / fig.dpi
        if w <= max_w_in:
            return txt
        txt.set_fontsize(txt.get_fontsize() - 0.2)
    w = txt.get_window_extent(renderer=r).width / fig.dpi
    if w > max_w_in:
        print(f"  WARNING: {txt.get_text()[:40]!r} is {w:.2f} in wide, "
              f"panel allows {max_w_in:.2f} in")
    return txt


def panel(ax, x, w, fc=GREY, ec="0.45"):
    ax.add_patch(FancyBboxPatch((x, PANEL_Y), w, PANEL_H,
                                boxstyle="round,pad=0.015,rounding_size=0.05",
                                linewidth=0.9, edgecolor=ec, facecolor=fc))


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                 lw=1.6, color=INK, shrinkA=0, shrinkB=0))


def verdict(fig, ax, x, y, mark, face, title, sub, maxw):
    """One row of the 'what survives' column: a neutral pill, then two lines.

    The pill is grey, not red/orange/green: colour in this paper means
    mechanism, and a coloured verdict icon made it mean two things at once.
    The glyph and the text carry the verdict on their own.
    """
    ax.add_patch(FancyBboxPatch((x, y - 0.055), 0.20, 0.15,
                                boxstyle="round,pad=0.012,rounding_size=0.05",
                                linewidth=0, facecolor=face))
    ax.text(x + 0.10, y + 0.018, mark, ha="center", va="center", fontsize=7.5,
            color="white", fontweight="bold")
    shrink_to_fit(fig, ax.text(x + 0.27, y + 0.055, title, ha="left",
                               va="center", fontsize=7.5), maxw)
    shrink_to_fit(fig, ax.text(x + 0.27, y - 0.075, sub, ha="left",
                               va="center", fontsize=7.0, color="0.35",
                               style="italic"), maxw)


def mini(fig, rect, draw, title, xlab, ylab):
    """A schematic observable. Deliberately unnumbered: these are sketches, and
    drawing ticks on them would imply values that do not exist."""
    a = fig.add_axes(rect)
    draw(a)
    a.set_xticks([]); a.set_yticks([])
    a.set_title(title, fontsize=7.0, pad=1.6, fontweight="normal", color="0.15")
    a.set_xlabel(xlab, fontsize=6.6, labelpad=1.0, color="0.35")
    a.set_ylabel(ylab, fontsize=6.6, labelpad=1.0, color="0.35")
    for s in a.spines.values():
        s.set_edgecolor("0.45"); s.set_linewidth(0.6)
    return a


def main():
    fs.use()
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(HERE, "figures", "battery_recovery.pdf")

    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")

    top = PANEL_Y + PANEL_H
    for x, w, head in PANELS:
        panel(ax, x, w)
        ax.text(x + w / 2, top + 0.20, head, ha="center", va="center",
                fontsize=8.5, fontweight="bold")
    # the subtitle sits clear above the panel border; it used to be struck
    # through by it
    ax.text(PANELS[0][0] + PANELS[0][1] / 2, top + 0.07,
            "(unknown to the estimator)", ha="center", va="center",
            fontsize=7.0, color="0.4", style="italic")

    # ---- left: the physics that is never inverted -------------------------
    x0, w0 = PANELS[0][0], PANELS[0][1]
    cx = x0 + w0 / 2
    shrink_to_fit(fig, ax.text(
        cx, top - 0.11,
        "SPMe, Eqs (1)–(4): solid diffusion,\n"
        "electrolyte transport, Butler–Volmer\n"
        r"$\rightarrow$ voltage $V(t)$",
        ha="center", va="top", fontsize=7, linespacing=1.30), w0 - 0.14)

    ly = top - 0.62
    for i, (m, lbl) in enumerate([("SEI", "SEI growth"),
                                  ("plating", "lithium plating"),
                                  ("cracks", "particle cracking"),
                                  ("LAM", "loss of active material")]):
        yy = ly - i * 0.165
        ax.add_patch(FancyBboxPatch((x0 + 0.30, yy - 0.045), 0.13, 0.09,
                                    boxstyle="square,pad=0.004", linewidth=0,
                                    facecolor=fs.MECH_COLORS[m]))
        ax.text(x0 + 0.50, yy, lbl, ha="left", va="center", fontsize=7.2)

    shrink_to_fit(fig, ax.text(
        cx, PANEL_Y + 0.36,
        r"rate constants $k_{\mathrm{SEI}},\ j_{0,\mathrm{plate}},\ "
        r"\beta_{\mathrm{LAM}},\ k_{\mathrm{cr}}$",
        ha="center", va="center", fontsize=7, color="0.3"), w0 - 0.14)
    shrink_to_fit(fig, ax.text(
        cx, PANEL_Y + 0.17,
        "per-mechanism loss $Q_m(n)$, Eq. (5),\nused only to score",
        ha="center", va="center", fontsize=7.0, color="0.3",
        linespacing=1.30), w0 - 0.14)

    # ---- centre: the three observables, drawn neutral ---------------------
    x1, w1 = PANELS[1][0], PANELS[1][1]

    def to_rect(ix, iy, iw, ih):
        return [ix / W, iy / H, iw / W, ih / H]

    mw, mh = 0.92, 0.32
    mx = x1 + (w1 - mw) / 2 + 0.10
    n = np.linspace(0, 1, 200)
    mini(fig, to_rect(mx, PANEL_Y + 1.38, mw, mh),
         lambda a: a.plot(n, 1 - 0.28 * np.sqrt(n), color=INK, lw=1.1),
         r"capacity $Q(n)$", "cycle $n$", "$Q$")
    q = np.linspace(0, 1, 400)
    dv = -(np.exp(-((q - 0.32) / 0.055) ** 2) + 0.8 *
           np.exp(-((q - 0.68) / 0.05) ** 2))
    mini(fig, to_rect(mx, PANEL_Y + 0.76, mw, mh),
         lambda a: a.plot(q, dv, color=INK, lw=1.1),
         r"low-rate d$V$/d$Q$", "$Q$", "d$V$/d$Q$")
    th = np.linspace(np.pi, 0, 300)
    zr = 0.5 + 0.5 * np.cos(th)
    zi = 0.42 * np.sin(th)
    tail = np.linspace(0, 0.34, 60)
    mini(fig, to_rect(mx, PANEL_Y + 0.15, mw, mh),
         lambda a: (a.plot(np.r_[zr, 1 + tail], np.r_[zi, tail], color=INK,
                           lw=1.1), a.set_ylim(-0.02, 0.5)),
         r"EIS $Z(\omega)$", r"Re $Z$", r"$-$Im $Z$")

    # ---- right: what survives ---------------------------------------------
    x2, w2 = PANELS[2][0], PANELS[2][1]
    vy = top - 0.26
    vx = x2 + 0.14
    vw = x2 + w2 - (vx + 0.27) - 0.07
    verdict(fig, ax, vx, vy, "~", "0.42", "LAM: fraction partial",
            r"$3/6$ label; fraction $R^2 = +0.53$", vw)
    verdict(fig, ax, vx, vy - 0.44, "✗", "0.18", "SEI: not recovered",
            r"$2/7$ label; fraction $R^2 = -0.77$", vw)
    verdict(fig, ax, vx, vy - 0.88, "✗", "0.18",
            "plating: never identified", r"$0/5$ label; fraction $R^2 = -0.77$",
            vw)
    verdict(fig, ax, vx, vy - 1.32, "✗", "0.18", "cracking: label only",
            r"$4/6$, an artefact of near-zero fade", vw)

    # ---- arrows between the panels ----------------------------------------
    ymid = PANEL_Y + PANEL_H / 2
    a0x0, a0x1 = x0 + w0 + 0.03, x1 - 0.03
    arrow(ax, (a0x0, ymid), (a0x1, ymid))
    ax.text((a0x0 + a0x1) / 2, ymid + 0.08, "observe", ha="center", va="bottom",
            fontsize=7, style="italic", color="0.25")
    a1x0, a1x1 = x1 + w1 + 0.03, x2 - 0.03
    arrow(ax, (a1x0, ymid), (a1x1, ymid))
    ax.text((a1x0 + a1x1) / 2, ymid + 0.08, "attribute", ha="center",
            va="bottom", fontsize=7, style="italic", color="0.25")
    ax.text((a1x0 + a1x1) / 2, ymid - 0.09, r"$\hat m,\ \hat f_m$", ha="center",
            va="top", fontsize=7.5, color="0.25")

    # ---- the banner: what is not recovered --------------------------------
    ax.add_patch(FancyBboxPatch((0.03, 0.06), W - 0.06, 0.54,
                                boxstyle="round,pad=0.012,rounding_size=0.05",
                                linewidth=0.9, edgecolor="#cc2b2b",
                                facecolor="#fdeeee"))
    ax.text(W / 2, 0.42, "NOT recovered:   the governing equations   ·   "
            "the rate constants   ·   the internal state",
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="#cc2b2b")
    ax.text(W / 2, 0.21, "Only a dominant-mechanism label and the mixing "
            "fractions are recovered, scored against the hidden decomposition.",
            ha="center", va="center", fontsize=7, color="0.2")

    paths = fs.save(fig, out)
    fs.mirror(paths, os.path.join(os.path.dirname(HERE), "manuscripts",
                                  "battery_mechanisms", "figures"))
    print("wrote", ", ".join(paths))


if __name__ == "__main__":
    sys.exit(main())
