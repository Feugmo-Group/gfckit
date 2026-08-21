"""Figure style for the hand-drawn schematics of the battery manuscript.

The rcParams, the printed text width and the PDF+PNG writer all come from
``gfckit.plotting``, so there is exactly one style definition in the package.
What lives here is what only the schematics need: the mechanism palette in a
form the diagram code can index, a degree sign that types correctly, and a
helper that copies the generated files into a manuscript's figure directory.

Schematic scripts are authored at the printed width (``FULL``) so that a point
in the figure is a point on the page; see the note in ``gfckit.plotting``.
"""
import os

import matplotlib.pyplot as plt                    # noqa: F401  (re-exported use)

from .plotting import (PAPER_RC, TEXTWIDTH_IN, use_paper_style,
                       save_paper_figure, panel_label, _MECH_COLORS,
                       DEG, mirror_figure)

__all__ = ["FULL", "TEXT_WIDTH_IN", "RC", "use", "save", "mirror",
           "MECH_COLORS", "DEG", "NEUTRAL", "NEUTRAL_LIGHT", "panel_label",
           "BASE", "TITLE", "TICK", "LEGEND", "SMALL"]

TEXT_WIDTH_IN = TEXTWIDTH_IN
FULL = TEXTWIDTH_IN
RC = PAPER_RC

BASE = PAPER_RC["font.size"]
TITLE = PAPER_RC["axes.titlesize"]
TICK = PAPER_RC["xtick.labelsize"]
LEGEND = PAPER_RC["legend.fontsize"]
SMALL = 7.0

MECH_COLORS = _MECH_COLORS
NEUTRAL = "0.25"
NEUTRAL_LIGHT = "0.55"


def use():
    """Apply the shared rcParams. Call once at the top of a figure script."""
    use_paper_style()


def save(fig, out_path, also_png=True):
    """Write ``out_path`` as a vector PDF, plus a raster preview beside it."""
    pdf = save_paper_figure(fig, out_path)
    plt.close(fig)
    stem = os.path.splitext(pdf)[0]
    return [pdf] + ([stem + ".png"] if also_png else [])


def mirror(paths, *dirs):
    """Copy the generated files into every manuscript figure directory."""
    return mirror_figure(list(paths), *dirs)
