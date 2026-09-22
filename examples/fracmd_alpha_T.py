"""alpha(T) across the fracMD campaign: the parameter-free test, and its limits.

The LiPS paper's central claim is that the subdiffusive exponent obeys

    alpha = k_B T / E_0                                   (trap-limited CTRW)

with no adjustable constants: alpha comes from the MSD, E_0 from the site-energy
landscape. This script does the MSD half -- it fits alpha for every completed run
and compares against that law.

The comparison has a ceiling that matters more than it might appear. The relation
only has content while alpha < 1: above T* = E_0/k_B the mean waiting time is
finite, transport is ordinary diffusion, and alpha saturates at 1 for every
temperature. Points above T* therefore test *where the crossover is*, which is a
real prediction, but they carry no information about the slope. Fitting a line
through them would manufacture agreement.

So the model plotted here is

    alpha(T) = min(k_B T / E_0, 1)

and the report separates temperatures that constrain the slope from those that
only corroborate the crossover.

Usage:
    python examples/fracmd_alpha_T.py [composition]      # default 67Li2S-33P2S5
"""
import os
import sys

import numpy as np

from fracmd_msd import load_tiers, runs_root

KB_EV = 8.617333262e-5          # eV/K
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = runs_root()
TAIL_DECADES = 3.0              # the diffusive end of the window


def beta_tail(t, m, decades=TAIL_DECADES):
    """Log-log slope over the last `decades` of the window.

    Measured at the long-time end deliberately: the early window is ballistic
    (beta -> 2) and then caged (beta well below the asymptote), neither of which
    is the transport exponent the CTRW law refers to.
    """
    lt, lm = np.log10(t), np.log10(m)
    sel = lt >= lt[-1] - decades
    return float(np.polyfit(lt[sel], lm[sel], 1)[0])


def collect(comp):
    """{T: [beta per replicate]} for every completed run of one composition."""
    base = os.path.join(ROOT, comp)
    out = {}
    if not os.path.isdir(base):
        return out
    for tdir in sorted(os.listdir(base)):
        if not tdir.startswith("T"):
            continue
        T = int(tdir[1:])
        for rdir in sorted(os.listdir(os.path.join(base, tdir))):
            run = os.path.join(base, tdir, rdir)
            try:
                t, m = load_tiers(run)
            except (FileNotFoundError, ValueError):
                continue
            out.setdefault(T, []).append(beta_tail(t, m))
    return out


def main(comp="67Li2S-33P2S5"):
    data = collect(comp)
    if not data:
        print(f"no completed runs for {comp}")
        return
    print(f"{comp}\n")
    print("   T(K)   n   beta            E0 = kB*T/beta")
    for T in sorted(data):
        b = np.array(data[T])
        sd = b.std(ddof=1) if len(b) > 1 else np.nan
        sdtxt = f"+/- {sd:.4f}" if len(b) > 1 else "          "
        print(f"  {T:5d}  {len(b):2d}   {b.mean():.4f} {sdtxt}   "
              f"{KB_EV * T / b.mean() * 1000:6.1f} meV")

    # Calibrate E_0 on the lowest temperature available -- the only regime where
    # alpha is unsaturated and so the only one that carries scale information.
    T_lo = min(data)
    b_lo = float(np.mean(data[T_lo]))
    if b_lo >= 0.98:
        print(f"\n  WARNING: even the lowest temperature ({T_lo} K) sits at the "
              f"Fickian ceiling.\n  No run in this set constrains E_0.")
        return
    E0 = KB_EV * T_lo / b_lo
    Tstar = E0 / KB_EV
    print(f"\n  E_0 calibrated on {T_lo} K : {E0*1000:.1f} meV")
    print(f"  implied crossover T*      : {Tstar:.0f} K")

    print("\n   T(K)   predicted   observed   regime")
    for T in sorted(data):
        pred = min(KB_EV * T / E0, 1.0)
        obs = float(np.mean(data[T]))
        regime = "constrains slope" if T < Tstar else "crossover only (capped)"
        flag = "" if abs(pred - obs) < 0.05 else "   <-- off by "\
            f"{abs(pred-obs):.3f}"
        print(f"  {T:5d}   {pred:9.3f}   {obs:8.4f}   {regime}{flag}")

    below = [T for T in data if T < Tstar]
    print(f"\n  {len(below)} of {len(data)} temperatures lie below T* and can test "
          f"the linear relation.")
    if len(below) < 3:
        print("  That is too few to fit a slope. Testing alpha = kB*T/E0 properly")
        print(f"  needs runs below {T_lo} K, where alpha sits well under 1.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "67Li2S-33P2S5")
