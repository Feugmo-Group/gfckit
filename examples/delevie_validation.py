"""External analytical validation of the identifier on de Levie's porous electrode.

Sections 3.4's other models (SEI reaction-diffusion, PDM film growth) are our own
generators, so agreement there proves only that the method is self-consistent.
de Levie (1963) is different: the pore is solved in closed form and the answer is
known in advance -- the high-frequency exponent is exactly 1/2.  Recovering
alpha = 0.500 is therefore a check against textbook truth.

The same closed form then supplies a second, sharper result.  In the
high-frequency branch

    Z -> sqrt(R/C) (j w)^(-1/2),

so the response depends on R and C only through their ratio and on the pore
length L not at all.  Pores of genuinely different geometry are therefore not
merely hard to tell apart -- they are analytically identical, and stay identical
until the measurement window reaches down to the crossover w_c = 1/(RCL^2).
That is Section 3.5's degeneracy again, but with a closed-form certificate and
from textbook electrochemistry rather than from our own construction.

Writes figures/delevie_validation.png.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from gfckit.delevie import (crossover_frequency, fit_cpe_exponent,
                            local_exponent, pore_impedance,
                            semi_infinite_pore_impedance)
from gfckit.plotting import plot_delevie, mirror_figure

jax.config.update("jax_enable_x64", True)

R, C, L = 1.0e6, 1.0e-3, 1.0e-4        # ohm/m, F/m, m -- one reference pore
FLOOR = 0.01                            # 1% relative noise floor, as in Sec 3.5
WIDTH = 3.0                             # decades: a typical impedance sweep
# Position of the window's LOWER edge, in decades below w_c.  Negative means the
# window sits entirely above the crossover, where the pore looks semi-infinite
# and the geometry is analytically invisible; positive means it reaches past the
# crossover, which is the only place the geometry is written down.
DEC_GRID = np.arange(-3.0, 1.01, 0.25)

# Pores that share R/C exactly but differ in absolute R, C and in length.
# Equal R/C is what makes them degenerate; different L is what we want to see.
GEOMS = [(r"$L\times3$,  $R,C\times10$", 10.0, 3.0),
         (r"$L\times\frac{1}{2}$, $R,C\times100$", 100.0, 0.5),
         (r"$L\times8$,  $R,C\times\frac{1}{20}$", 0.05, 8.0)]


def main():
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(HERE, "figures", "delevie_validation.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    wc = float(crossover_frequency(R, C, L))
    print(f"reference pore: R={R:g} ohm/m, C={C:g} F/m, L={L:g} m")
    print(f"crossover w_c = {wc:.4g} rad/s\n")

    # --- 1. the exponent is exactly 1/2, and we recover it ------------------
    w_hf = jnp.logspace(np.log10(wc) + 2, np.log10(wc) + 5, 400)
    Z_hf = pore_impedance(w_hf, R, C, L)
    alpha_hf, A_hf, rms_hf = fit_cpe_exponent(w_hf, jnp.abs(Z_hf))
    print(f"high-frequency branch (2-5 decades above w_c):")
    print(f"  recovered alpha = {float(alpha_hf):.6f}   (exact value 1/2)")
    print(f"  fit RMS         = {float(rms_hf):.2e}")
    print(f"  mean phase      = {float(jnp.mean(jnp.angle(Z_hf, deg=True))):.4f} deg"
          f"   (exact -45)")
    print(f"  amplitude A     = {float(A_hf):.6g}  vs sqrt(R/C) = "
          f"{np.sqrt(R / C):.6g}")
    err = float(jnp.max(jnp.abs(Z_hf - semi_infinite_pore_impedance(w_hf, R, C))
                        / jnp.abs(semi_infinite_pore_impedance(w_hf, R, C))))
    print(f"  |Z - Z_semi-inf| = {err:.2e} relative\n")

    # the same fit with 1% noise, so the quoted order is not a noiseless artefact
    key = jax.random.PRNGKey(0)
    noisy = jnp.abs(Z_hf) * (1.0 + FLOOR * jax.random.normal(key, w_hf.shape))
    a_n, _, r_n = fit_cpe_exponent(w_hf, noisy)
    print(f"  with {FLOOR:.0%} noise: alpha = {float(a_n):.4f}, "
          f"fit RMS = {float(r_n):.2%}\n")

    # --- 2. the full sweep, -1/2 to -1 --------------------------------------
    w_full = jnp.logspace(np.log10(wc) - 4, np.log10(wc) + 5, 900)
    Z_full = pore_impedance(w_full, R, C, L)
    ex = local_exponent(w_full, jnp.abs(Z_full))
    print(f"local exponent: {float(ex[-1]):.4f} (HF) -> {float(ex[0]):.4f} (LF)\n")

    # --- 3. geometry is invisible until the window reaches w_c --------------
    # For each window we compare each pore against the reference after the free
    # amplitude a real fit would absorb, and record the worst relative deviation.
    print(f"max deviation between pores, {WIDTH:.0f}-decade window, vs where its")
    print("lower edge sits relative to w_c (negative = entirely above it):")
    print("  edge   " + "".join(f"{lbl[:14]:>16}" for lbl, _, _ in GEOMS))
    curves = [[] for _ in GEOMS]
    for d in DEC_GRID:
        lo = np.log10(wc) - d                     # d decades below w_c
        w = jnp.logspace(lo, lo + WIDTH, 400)
        ref = jnp.abs(pore_impedance(w, R, C, L))
        for i, (_, fac, Lf) in enumerate(GEOMS):
            other = jnp.abs(pore_impedance(w, R * fac, C * fac, L * Lf))
            scale = jnp.sum(ref * other) / jnp.sum(other ** 2)   # best amplitude
            curves[i].append(float(jnp.max(jnp.abs(scale * other - ref) / ref)))
        print(f"  {d:5.2f}  " + "".join(f"{c[-1]:16.3e}" for c in curves))

    dev_curve = [(lbl, np.array(c)) for (lbl, _, _), c in zip(GEOMS, curves)]
    print()
    for (lbl, c) in dev_curve:
        hit = np.argmax(c > FLOOR) if (c > FLOOR).any() else None
        print(f"  {lbl}: rises above the {FLOOR:.0%} floor once the window edge "
              + (f"reaches {DEC_GRID[hit]:+.2f} decades of w_c" if hit is not None
                 else "-- never, in any window tested"))

    plot_delevie(w_full, jnp.abs(Z_full), wc, alpha_hf, rms_hf, w_full, ex,
                 GEOMS, FLOOR, DEC_GRID, dev_curve, out)
    # keep the manuscript figure directory in step with its generator; the two
    # used to be reconciled by hand
    mirror_figure(out, os.path.join(HERE, os.pardir, "manuscripts",
                                    "gfc_identifiability", "figures"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
