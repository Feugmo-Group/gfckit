"""Cross-model test: generate aging data from INDEPENDENT mechanistic models
(solved with Jacobian-free Newton-Krylov), then identify the effective memory
exponent with gfckit.

Models (see examples/README.md for the equations):
  * SEI growth  -- moving-boundary reaction-diffusion (capacity fade)
  * Corrosion   -- Point Defect Model high-field oxide growth vs. dissolution

Writes figures/physical_models.png.
"""
import os
import jax
import jax.numpy as jnp

from gfckit.physics import simulate_sei_reaction_diffusion, simulate_pdm_film_growth
from gfckit.identify import recover_order_weak
from gfckit.plotting import plot_physical_models, mirror_figure

jax.config.update("jax_enable_x64", True)


def powerlaw_fit(t, y, alpha, mask=None):
    """Least-squares A t^alpha + B for a given alpha (for plotting/goodness).

    `mask` restricts the least squares to the window the order was identified
    on.  Fitting the coefficients on the whole record while identifying alpha on
    a sub-window mixes two different fits and drags the drawn curve outside the
    data (to L = -0.3 at t -> 0, below a thickness that is positive by
    construction)."""
    Phi = jnp.stack([t ** alpha, jnp.ones_like(t)], axis=1)
    Phi_fit = Phi if mask is None else Phi[mask]
    y_fit = y if mask is None else y[mask]
    coef, *_ = jnp.linalg.lstsq(Phi_fit, y_fit, rcond=None)
    return Phi @ coef, coef


def main():
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(HERE, "figures", "physical_models.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # --- SEI reaction-diffusion (Newton-Krylov) ---
    t_sei, L_sei = simulate_sei_reaction_diffusion(nsteps=400, dt=0.02)
    # identify effective exponent on the diffusion-dominated part (t > 1)
    T_FIT_LO = 1.0
    mask = t_sei > T_FIT_LO
    grid = jnp.linspace(0.3, 1.2, 181)
    a_sei = float(recover_order_weak(L_sei[mask], t_sei[mask], grid))
    sei_fit, _ = powerlaw_fit(t_sei, L_sei, a_sei, mask=mask)
    print(f"SEI reaction-diffusion:  effective alpha = {a_sei:.3f}  "
          f"(diffusion-limited -> 1/2; reaction-limited -> 1)")

    # --- PDM film growth (Newton-Krylov) ---
    t_pdm, L_pdm, L_ss = simulate_pdm_film_growth(nsteps=400, dt=0.02)
    grid2 = jnp.linspace(0.05, 1.2, 231)
    a_pdm = float(recover_order_weak(L_pdm, t_pdm, grid2))
    print(f"PDM film growth:         effective alpha = {a_pdm:.3f}  "
          f"(passivates to L_ss={L_ss:.2f} -> a single power law fits poorly;")
    print("                          the saturation is the finite memory horizon lambda)")

    # the local slope over the fit window is what Fig. 5(c) shows; print it so
    # the trend the body claims is checkable from the log, not only from the plot
    import numpy as _np_
    ts, Ls = _np_.asarray(t_sei), _np_.asarray(L_sei)
    m = ts > 0
    sl = _np_.gradient(_np_.log(Ls[m]), _np_.log(ts[m]))
    inw = ts[m] >= T_FIT_LO
    print(f"SEI local log-log slope: peaks at {sl.max():.3f} near "
          f"t={ts[m][sl.argmax()]:.2f}; over the fit window it falls "
          f"{sl[inw][0]:.3f} -> {sl[-1]:.3f} (toward the Fickian 1/2)")

    path = plot_physical_models(t_sei, L_sei, sei_fit, a_sei,
                                t_pdm, L_pdm, L_ss, out, t_fit_lo=T_FIT_LO)
    # keep the manuscript figure directory in step with its generator; the two
    # used to be reconciled by hand
    mirror_figure(path, os.path.join(HERE, os.pardir, "manuscripts",
                                    "gfc_identifiability", "figures"))
    print(f"\nwrote figure -> {path}")


if __name__ == "__main__":
    main()
