"""EIS mechanism-sensitivity demonstration (phase-2 observable).

Computes impedance (pybammeis) for mechanism-representative parameter
perturbations and plots the Nyquist signatures. The point: EIS adds a frequency
dimension that separates mechanisms (LAM vs LLI vs added resistance) which are
degenerate in a capacity-fade curve.

Writes figures/pybamm_eis_nyquist.png.
"""
import os
import numpy as np

from gfckit.eis_gen import compute_eis, mechanism_scenarios
from gfckit.plotting import plot_nyquist, mirror_figure

FIG_DIRS = ["manuscripts/battery_mechanisms/figures"]


def main():
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ROOT = os.path.dirname(HERE)
    out = os.path.join(HERE, "figures", "pybamm_eis_nyquist.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    freqs = np.logspace(-2, 4, 40)
    scenarios = mechanism_scenarios()
    results = []
    for label, ov in scenarios.items():
        try:
            Z = compute_eis(freqs, overrides=ov)
            results.append((label, Z))
            print(f"  {label:28s} Z(low f)={Z[0].real*1e3:.1f}{Z[0].imag*1e3:+.1f}j mOhm")
        except Exception as e:
            print(f"  {label:28s} EIS failed: {type(e).__name__}: {e}")

    path = plot_nyquist(results, out, freqs=freqs)
    written = mirror_figure(path, *[os.path.join(ROOT, d) for d in FIG_DIRS])
    print(f"\nInterpretation: distinct Nyquist signatures => EIS separates")
    print("mechanisms (LAM/LLI/resistance) that capacity fade alone cannot.")
    print("wrote figure -> " + ", ".join(written))


if __name__ == "__main__":
    main()
