"""Draw the nonlocal-operator schematic -> figures/nonlocal_operator.{pdf,png}.

(a) why a fractional operator is nonlocal: a local d/dt reads one instant, the
    Caputo operator weights the whole history with (t-tau)^{-alpha}/Gamma(1-alpha);
(b) where that same half-order operator shows up in the observables: the
    Warburg/CPE tail of the impedance, alongside the capacity-fade power law,
    which is a fractional relaxation of general order.

Explanatory rather than quantitative, but every curve is a real function --
the memory kernel is the actual Caputo weight and panel (b) is an actual
Randles-with-Warburg impedance evaluated over a frequency sweep -- so the
picture cannot drift away from the algebra it illustrates. Panel (b)'s axes are
normalised by R_ct, which is the only honest scale here: the circuit values are
dimensionless, so labelling the axes in ohms would invent units.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as gamma_fn

from gfckit import figstyle as fs

ALPHA = 0.5           # half-order: the Warburg / CPE case
R_S = 0.15            # series (electrolyte) resistance, in units of R_ct
R_CT = 1.0            # charge-transfer resistance -- the normalising scale
C_DL = 1.0            # double-layer capacitance
SIGMA = 0.045         # Warburg coefficient -- small enough that the R_ct arc
                      # closes before the diffusion tail takes over

BLUE = "#1f77b4"
RED = "#d62728"
PURPLE = "#6a3d9a"
GREEN = "#2ca02c"

XLIM_B, YLIM_B = (0.0, 2.05), (0.0, 0.95)
BOX_ASPECT = (YLIM_B[1] - YLIM_B[0]) / (XLIM_B[1] - XLIM_B[0])


def main():
    fs.use()
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(HERE, "figures", "nonlocal_operator.pdf")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(fs.FULL, 2.05))

    # ---- (a) locality vs memory -------------------------------------------
    t = np.linspace(0.0, 1.0, 600)
    signal = 0.62 + 0.16 * np.sin(2.0 * np.pi * t + 0.7) \
        + 0.06 * np.sin(4.0 * np.pi * t + 1.9)
    axa.plot(t, signal, color="0.15", lw=1.6)

    # the Caputo weight itself, normalised only for display
    tau = np.linspace(0.0, 0.9995, 600)
    kern = (1.0 - tau) ** (-ALPHA) / gamma_fn(1.0 - ALPHA)
    kern = 0.02 + 0.30 * kern / kern[np.searchsorted(tau, 0.995)]
    axa.fill_between(tau, 0.0, kern, color="#c8dcf0", alpha=0.85)
    axa.plot(tau, kern, color=BLUE, lw=1.4)

    axa.axvline(1.0, color=RED, ls="--", lw=1.4)
    axa.plot([1.0], [signal[-1]], "o", color=RED, ms=5)

    # the black curve is the most prominent object in the panel and used to go
    # unnamed; label it where it is unambiguous
    axa.annotate(r"signal history $f(\tau)$",
                 xy=(0.42, float(np.interp(0.42, t, signal))),
                 xytext=(0.34, 1.03), fontsize=7, color="0.15", ha="center",
                 va="top",
                 arrowprops=dict(arrowstyle="->", color="0.35", lw=0.9,
                                 shrinkB=2))
    axa.annotate("local $\\mathrm{d}/\\mathrm{d}t$:\npresent instant only",
                 xy=(1.0, signal[-1]), xytext=(0.80, 0.33), fontsize=7,
                 color=RED, ha="center", va="top",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.9, shrinkB=3))
    # anchor the kernel arrow ON the shaded band, not above it: it used to end
    # in white space a few hundred pixels clear of the curve
    tau_a = 0.62
    k_a = float(np.interp(tau_a, tau, kern))
    axa.annotate(r"memory kernel" "\n" r"$(t-\tau)^{-\alpha}/\Gamma(1-\alpha)$",
                 xy=(tau_a, 0.5 * k_a), xytext=(0.26, 0.34), fontsize=7,
                 color=BLUE, ha="center", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9,
                                 shrinkB=1))

    axa.set_title("(a) the operator weights the whole history",
                  fontsize=8, loc="left", pad=3)
    axa.set_xlabel(r"past $\tau$  $\longrightarrow$  now $t$")
    axa.set_ylabel("signal and memory weight\n(arbitrary scale)", fontsize=6.6)
    axa.set_xlim(0, 1.04)
    axa.set_ylim(0, 1.05)
    axa.set_xticks([])
    axa.set_yticks([])
    axa.set_box_aspect(BOX_ASPECT)

    # ---- (b) the same operator in the observables -------------------------
    # Randles cell with a Warburg tail:
    #   Z = Rs + Rct/(1 + i w Rct Cdl) + sigma (i w)^{-1/2}
    w = np.logspace(-4.0, 3.5, 6000)
    Z = R_S + R_CT / (1.0 + 1j * w * R_CT * C_DL) + SIGMA * (1j * w) ** (-0.5)
    axb.plot(Z.real / R_CT, -Z.imag / R_CT, color=PURPLE, lw=1.6)

    # the pure (i w)^{-1/2} asymptote is a 45-degree line through the foot of
    # the arc -- drawn from the algebra, not sketched
    foot = (R_S + R_CT) / R_CT
    x45 = np.linspace(foot + 0.02, foot + 0.75, 50)
    axb.plot(x45, x45 - foot, ls=":", color=RED, lw=1.3)

    axb.text(0.035, 0.885, r"EIS:  $i(t) = Q_{\mathrm{CPE}}\,{}^{C}\!D_t^{1/2}\,v(t)$"
             "   (half-order)", transform=axb.transAxes, fontsize=7)
    axb.text(0.035, 0.735, r"fade:  $Q_{\mathrm{loss}}\sim n^{p}$"
             r"  $\Leftrightarrow$  ${}^{C}\!D^{\alpha}Q =$ const  (order $\alpha$)",
             transform=axb.transAxes, fontsize=7, color=GREEN)
    axb.text(0.30, 0.24, r"$R_{\mathrm{ct}}$ arc", transform=axb.transAxes,
             fontsize=7, color="0.35")
    # the label names the red dotted asymptote, so it is drawn in that colour;
    # in purple it read as an annotation of the Nyquist curve
    axb.text(1.78, 0.32, r"$(\mathrm{i}\omega)^{-1/2}$", fontsize=7.5,
             color=RED, ha="center")

    axb.set_title("(b) the same operators in the observables",
                  fontsize=8, loc="left", pad=3)
    axb.set_xlabel(r"$\mathrm{Re}\,Z\,/\,R_{\mathrm{ct}}$")
    axb.set_ylabel(r"$-\,\mathrm{Im}\,Z\,/\,R_{\mathrm{ct}}$")
    axb.set_xlim(*XLIM_B)
    axb.set_ylim(*YLIM_B)
    axb.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    axb.set_yticks([0.0, 0.25, 0.5, 0.75])
    axb.set_box_aspect(BOX_ASPECT)     # equal data scaling: 1 unit x = 1 unit y

    fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.16,
                        wspace=0.30)
    paths = fs.save(fig, out)
    fs.mirror(paths, os.path.join(os.path.dirname(HERE), "manuscripts",
                                  "battery_mechanisms", "figures"))
    print("wrote", ", ".join(paths))


if __name__ == "__main__":
    sys.exit(main())
