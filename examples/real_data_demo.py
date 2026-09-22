"""Demonstrate the observables on REAL cells (Zhang et al. 2020).

Loads the downloaded dataset (capacity fade + EIS over aging) and writes
figures/real_zhang_overview.png -- the real-data counterpart of the PyBaMM study,
to pre-empt the "does it hold on real data?" question.
"""
import os

from gfckit.real_data import load_zhang_capacity, load_zhang_eis
from gfckit.plotting import plot_real_overview


def main():
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = os.path.join(HERE, "data", "real", "zhang2020")
    out = os.path.join(HERE, "figures", "real_zhang_overview.png")

    cap = load_zhang_capacity(base)
    eis = load_zhang_eis(base)
    print(f"loaded {len(cap)} capacity-fade cells, {len(eis)} cells with EIS")

    # one cell's EIS evolution over aging states
    key = sorted(eis, key=lambda k: -len(eis[k]))[0]
    allst = eis[key]
    pick = sorted({0, len(allst) // 3, 2 * len(allst) // 3, len(allst) - 1})
    states = [allst[i] for i in pick]                 # ~4 states for clarity
    print(f"EIS evolution shown for cell {key} across {len(states)} aging states")

    path = plot_real_overview(cap, states, out)
    print(f"wrote figure -> {path}")


if __name__ == "__main__":
    main()
