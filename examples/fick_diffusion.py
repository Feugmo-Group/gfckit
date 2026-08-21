"""Fick's-law diffusion in several concentration / flow systems, and the link
to parabolic (sqrt-t) growth.

Shows (see examples/README.md for the equations):
  * concentration profiles for a semi-infinite solid (erfc) and an
    instantaneous planar source (Gaussian) at several times;
  * diffusion-limited cumulative uptake ~ sqrt(t), whose gfckit-identified
    exponent is alpha ~ 1/2 -- i.e. classical Fickian growth is the alpha=1/2
    special case that anomalous (memory) kinetics deviate from.

Writes figures/fick_diffusion.png.
"""
import os
import jax
import jax.numpy as jnp

from gfckit.fick import (semi_infinite_constant_surface, instantaneous_point_source,
                         diffusion_uptake)
from gfckit.identify import recover_order_weak
from gfckit.plotting import plot_fick

jax.config.update("jax_enable_x64", True)


def main():
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(HERE, "figures", "fick_diffusion.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    D = 1.0
    x = jnp.linspace(0.0, 6.0, 300)
    t_snapshots = [0.25, 1.0, 4.0]

    xg = jnp.linspace(-6.0, 6.0, 300)
    profiles = {"semi_infinite": [], "gaussian": []}
    for ts in t_snapshots:
        profiles["semi_infinite"].append(semi_infinite_constant_surface(x, ts, D))
        profiles["gaussian"].append(instantaneous_point_source(xg, ts, D))

    # diffusion-limited uptake ~ sqrt(t)  -> identify alpha (expect ~0.5)
    t = jnp.linspace(0.02, 8.0, 400)
    uptake = diffusion_uptake(t, D)
    a_uptake = float(recover_order_weak(uptake, t, jnp.linspace(0.2, 1.0, 161)))
    Phi = jnp.stack([t ** a_uptake, jnp.ones_like(t)], axis=1)
    coef, *_ = jnp.linalg.lstsq(Phi, uptake, rcond=None)
    uptake_fit = Phi @ coef

    print("Fick's law -- different concentration/flow systems:")
    print("  * semi-infinite fixed surface: c = c_s erfc(x/2 sqrt(Dt))")
    print("  * instantaneous source:        c = M/sqrt(4 pi D t) exp(-x^2/4Dt)")
    print(f"  * diffusion-limited uptake ~ sqrt(t): identified alpha = {a_uptake:.3f}"
          "  (Fickian = 1/2)")

    path = plot_fick(x, xg, profiles, t_snapshots, t, uptake, uptake_fit,
                     a_uptake, out)
    print(f"\nwrote figure -> {path}")


if __name__ == "__main__":
    main()
