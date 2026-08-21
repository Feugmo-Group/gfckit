"""Single-trajectory anomalous-order identification + figure.

Ground truth u(t)=t^alpha (exact soln of ^C D^alpha u = Gamma(a+1)).
Compares the strong (derivative) and weak (integral) forms across noise levels
and writes `figures/order_identification.png`.
"""
import os
import jax
import jax.numpy as jnp

from gfckit.generate import analytic_powerlaw
from gfckit.identify import recover_order_strong, recover_order_weak
from gfckit.plotting import plot_order_identification, mirror_figure

jax.config.update("jax_enable_x64", True)


def main():
    alpha_true = 0.5
    N = 2000
    t = jnp.linspace(0.0, 5.0, N)
    ds = float(t[1] - t[0])
    grid_strong = jnp.linspace(0.05, 1.0, 96)
    grid_weak = jnp.linspace(0.05, 1.5, 291)

    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(HERE, "figures", "order_identification.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    u_clean, _ = analytic_powerlaw(t, alpha_true)
    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05]
    a_strong, a_weak = [], []
    print(f"true alpha = {alpha_true}\n")
    for nz in noise_levels:
        key = jax.random.PRNGKey(int(nz * 1000) + 1)
        u = u_clean + nz * float(jnp.std(u_clean)) * jax.random.normal(key, (N,))
        a_s = float(recover_order_strong(u, ds, grid_strong))
        a_w = float(recover_order_weak(u, t, grid_weak))
        a_strong.append(a_s)
        a_weak.append(a_w)
        print(f"  noise={nz:.1%}:  strong alpha={a_s:.3f}   weak alpha={a_w:.3f}")

    path = plot_order_identification(noise_levels, a_strong, a_weak,
                                     alpha_true, out)
    # keep the manuscript figure directory in step with its generator; the two
    # used to be reconciled by hand
    mirror_figure(path, os.path.join(HERE, os.pardir, "manuscripts",
                                    "gfc_identifiability", "figures"))
    print(f"\nwrote figure -> {path}")


if __name__ == "__main__":
    main()
