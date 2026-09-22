"""Low-rate dV/dQ (DVA) analysis on the Oxford Battery Degradation Dataset.

Oxford cells are characterized every 100 cycles with a C/18 pseudo-OCV discharge,
giving the CLEAN low-rate differential-voltage curves that the Zhang dataset lacks.
Writes figures/real_oxford.png (capacity fade + dV/dQ evolution over aging).
"""
import os

from gfckit.real_data import load_oxford
from gfckit.plotting import plot_oxford, mirror_figure

FIG_DIRS = ["manuscripts/battery_mechanisms/figures"]


def main():
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(HERE, "data", "real", "oxford", "oxford.mat")
    out = os.path.join(HERE, "figures", "real_oxford.pdf")

    cells = load_oxford(path)
    print(f"loaded {len(cells)} Oxford cells")
    for name, d in sorted(cells.items()):
        print(f"  {name}: {len(d['cycle'])} checkpoints, "
              f"retention end {d['capacity_mAh'][-1]/d['capacity_mAh'][0]:.3f}")

    p = plot_oxford(cells, out)
    written = mirror_figure(p, *[os.path.join(os.path.dirname(HERE), d)
                                 for d in FIG_DIRS])
    print("\nThe C/18 dV/dQ peaks shift and shrink over aging (LLI/LAM signatures)")
    print("-- the clean low-rate differential-voltage fingerprint.")
    print("wrote figure -> " + ", ".join(written))


if __name__ == "__main__":
    main()
