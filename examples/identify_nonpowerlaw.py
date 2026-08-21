"""THE CORE RESULT: a single fractional order (power law) cannot fit saturating,
passivating degradation, but a GENERAL fractional operator -- an identified
distributed-order spectrum w(alpha) -- can (PDM passivation: R^2 0.862 -> 0.998).

For a genuinely multi-scale (two-order) memory, by contrast, single-curve spectrum
recovery is ILL-POSED: the identified spectrum does NOT reproduce the bimodal
GROUND TRUTH {0.3, 0.8}. The recovered mass instead collapses toward one
effective intermediate order (see the "Honest interpretation" printout in main()).

For each dataset we compare:
  * single_order_residual   -- best single power law (one order alpha)
  * recover_order_spectrum  -- distributed-order spectrum (general fractional)
and plot the recovered spectra.

THE VARIABLE-ORDER CASE IS FITTED BUT NOT PLOTTED.  Its spectrum fit reaches
R^2 = 0.657 against 0.993 for a single power law, and the paper's Sec. 3.5 rule
is that a richer model losing to a simpler one supports no inference.  Here the
cause is neither ill-posedness nor optimizer failure, and we checked both:

  * Convergence.  R^2 = 0.656996 at 50k Adam steps, 0.656996 at 200k, 0.656992
    at 200k with lr 0.02, 0.654678 at 500k, with the recovered mass in the same
    single bin (alpha = 0.419, weight 1.000) every time.  The fit is converged.
  * The comparator.  The 0.993 comes from alpha = 1.600, which is exactly the
    top of powerlaw_reconstruction's search grid.  Widening the grid gives
    alpha = 1.91 at R^2 = 0.997, so the reference number was a bound, not a fit.
  * The real cause is model misspecification.  alpha(t) sweeps 0.3 -> 0.9 here
    and the response grows like t^1.9, while the distributed-order model is a
    fixed mixture over the grid alpha in [0.1, 0.95] and can produce at most
    t^0.95.  The data lie outside the model class, so the R^2 gap measures that
    mismatch and says nothing about whether a spectrum is identifiable.

Plotting the panel anyway would have the figure assert what Sec. 3.5 forbids.
The number is printed below so the result stays on the record.

Writes figures/order_spectra.{pdf,png}.
"""
import os
import jax
import jax.numpy as jnp
import numpy as np

from gfckit.generate import analytic_powerlaw
from gfckit.complex import (multiterm_fractional_relaxation,
                            variable_order_solution)
from gfckit.physics import simulate_pdm_film_growth
from gfckit.identify import powerlaw_reconstruction, recover_order_spectrum
from gfckit.plotting import plot_order_spectra, mirror_figure

jax.config.update("jax_enable_x64", True)


def main():
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(HERE, "figures", "order_spectra.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    order_grid = jnp.linspace(0.1, 0.95, 25)     # orders live in (0,1)
    curves = []

    # --- power-law baselines: expect a single spike ---
    t = jnp.linspace(1e-3, 6.0, 700)
    u, _ = analytic_powerlaw(t, 0.5)
    curves.append(("powerlaw_0.5", "power-law", t, u, [0.5]))
    u, _ = analytic_powerlaw(t, 0.75)
    curves.append(("powerlaw_0.75", "power-law", t, u, [0.75]))

    # --- multi-term: spectrum recovery is ill-posed here; it collapses to a
    # single effective order instead of recovering the bimodal {0.3, 0.8} ---
    t, u = multiterm_fractional_relaxation([0.3, 0.8], [0.6, 0.4], lam=0.02,
                                           S=0.03, u0=0.0, dt=0.03, nsteps=800)
    curves.append(("multiterm", "non-power-law", t, u, [0.3, 0.8]))

    # --- variable order 0.3->0.9: expect a spread spectrum ---
    def path(x):
        return 0.3 + 0.6 / (1.0 + jnp.exp(-(x - 12.0) / 2.5))
    t, u = variable_order_solution(path, lam=0.02, S=0.03, u0=0.0,
                                   dt=0.03, nsteps=800)
    curves.append(("variable_order", "non-power-law", t, u, [0.3, 0.9]))
    PLOT_EXCLUDE = {"variable_order"}          # see the module docstring

    # --- PDM passivation ---
    t, u, _ = simulate_pdm_film_growth(nsteps=700, dt=0.02)
    curves.append(("pdm_passivation", "non-power-law", t, u, []))

    # The spectrum fits are ~15 minutes of Adam; cache them so the figure can be
    # re-drawn without re-fitting, and so the numbers in the figure are the
    # numbers this run printed.
    cache = os.path.join(HERE, "data", "synthetic", "order_spectra_fits.npz")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    cached = {}
    if os.environ.get("GFCKIT_REFIT") != "1" and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        cached = {k: z[k] for k in z.files}
        print(f"(using cached fits from {cache}; set GFCKIT_REFIT=1 to refit)")

    print(f"{'dataset':<16}{'family':<15}{'PL alpha':>9}{'R2 power-law':>14}"
          f"{'R2 spectrum':>13}")
    results = []
    store = {}
    for name, family, t, u, gt in curves:
        key_w, key_r = f"{name}__w", f"{name}__r2"
        if key_w in cached:
            w = jnp.asarray(cached[key_w])
            r2_dist = float(cached[key_r][1])
            a_pl, r2_pl = float(cached[key_r][2]), float(cached[key_r][0])
        else:
            a_pl, r2_pl = powerlaw_reconstruction(t, u)
            w, r2_dist, _ = recover_order_spectrum(t, u, order_grid)
            a_pl, r2_pl, r2_dist = float(a_pl), float(r2_pl), float(r2_dist)
        store[key_w] = np.asarray(w)
        store[key_r] = np.array([r2_pl, r2_dist, a_pl])
        flag = "   [fitted, not plotted]" if name in PLOT_EXCLUDE else ""
        print(f"{name:<16}{family:<15}{a_pl:>9.3f}{r2_pl:>14.4f}{r2_dist:>13.4f}"
              f"{flag}")
        if name in PLOT_EXCLUDE:
            continue
        results.append(dict(name=name, family=family, order_grid=order_grid,
                            w=w, res_pl=1.0 - r2_pl, res_dist=1.0 - r2_dist,
                            ground_truth=gt))
    np.savez(cache, **store)

    path = plot_order_spectra(results, out)
    print("\nHonest interpretation:")
    print("  * power-law data -> sharp single spike at the true order (recovered).")
    print("  * PDM passivation -> general spectrum is a single, sharply resolved")
    print("    intermediate order and BEATS a single power law (R^2 0.862 -> 0.998):")
    print("    saturation cannot be described by a two-parameter power law.")
    print("  * multi-term -> single-curve spectrum recovery is ILL-POSED: mass")
    print("    collapses toward one effective order (a multi-scale memory ~ one")
    print("    intermediate order on a single trajectory).")
    print("  * variable-order -> fitted but NOT plotted: the response grows like")
    print("    t^1.9, outside what a fixed mixture over alpha in [0.1,0.95] can")
    print("    produce, so its low R^2 measures model misspecification and")
    print("    supports no identifiability inference (see the module docstring).")
    print("  => resolution: multi-condition (Arrhenius) identification, as for the")
    print("     tempered (alpha,lambda) case in examples/multicondition.py.")
    # keep the manuscript figure directory in step with its generator; the two
    # used to be reconciled by hand
    mirror_figure(path, os.path.join(HERE, os.pardir, "manuscripts",
                                    "gfc_identifiability", "figures"))
    print(f"wrote figure -> {path}")


if __name__ == "__main__":
    main()
