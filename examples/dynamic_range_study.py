"""How much DYNAMIC RANGE is needed to resolve a distributed-order spectrum?

A two-order memory {a_lo, a_hi} has two power-law regimes that meet at a
crossover time.  Using a Talbot inverse Laplace transform we compute the exact
response over 8 decades and show:
  (a) the local slope reaches the true orders only far from the crossover;
  (b) the fraction of the order range [a_lo, a_hi] that an observation window of
      width D decades (centered on the crossover) can resolve.

Realistic aging experiments span 1-3 decades, which resolve well under half the
order range -- so the fine spectrum is unidentifiable in practice.

Writes figures/dynamic_range.png.
"""
import os
import jax
import jax.numpy as jnp

from gfckit.laplace import distributed_order_response
from gfckit.plotting import plot_dynamic_range, mirror_figure

jax.config.update("jax_enable_x64", True)


def main():
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(HERE, "figures", "dynamic_range.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    alphas, weights = (0.2, 0.9), (0.5, 0.5)
    a_lo, a_hi = min(alphas), max(alphas)

    # exact response over 8 decades (crossover near t=1 for equal weights)
    t = jnp.logspace(-4.0, 4.0, 200)
    u = distributed_order_response(alphas, weights, t, lam=0.0, S=1.0)
    ln_t, ln_u = jnp.log(t), jnp.log(u)
    slope = jnp.gradient(ln_u, ln_t)

    # fraction of the order range resolved by a window of D decades about t=1
    def slope_at(tt):
        return float(jnp.interp(jnp.log(tt), ln_t, slope))

    D_grid = jnp.arange(1.0, 8.1, 1.0)
    resolved = []
    for D in D_grid:
        hi = slope_at(10.0 ** (-D / 2))       # small-t end -> high order
        lo = slope_at(10.0 ** (D / 2))        # large-t end -> low order
        resolved.append((hi - lo) / (a_hi - a_lo))
    resolved = jnp.clip(jnp.array(resolved), 0.0, 1.0)

    near_hi = t[slope > 0.85 * a_hi]
    near_lo = t[slope < a_lo + 0.05]
    print(f"Two-order memory {{{a_lo}, {a_hi}}} (equal weights):")
    if near_hi.size:
        print(f"  slope near {a_hi} only for t < {float(near_hi.max()):.2g}")
    if near_lo.size:
        print(f"  slope near {a_lo} only for t > {float(near_lo.min()):.2g}")
    print(f"  {'window (decades)':>18}{'resolved fraction':>20}")
    for D, r in zip(D_grid, resolved):
        print(f"  {float(D):>18.0f}{float(r):>20.2f}")
    # threshold
    thr = float(jnp.interp(0.9, resolved, D_grid))
    print(f"\n  ~{thr:.1f} decades needed to resolve 90% of the order range;")
    print(f"  typical experiments (1-3 decades) resolve "
          f"{float(jnp.interp(2.0, D_grid, resolved)):.0%} -> spectrum unidentifiable.")

    path = plot_dynamic_range(t, slope, alphas, D_grid, resolved,
                              exp_window=(1.0, 3.0), out_path=out,
                              t_hi_end=float(near_hi.max()) if near_hi.size else None,
                              t_lo_start=float(near_lo.min()) if near_lo.size else None,
                              d_required=thr)
    # keep the manuscript figure directory in step with its generator; the two
    # used to be reconciled by hand
    mirror_figure(path, os.path.join(HERE, os.pardir, "manuscripts",
                                    "gfc_identifiability", "figures"))
    print(f"\nwrote figure -> {path}")


if __name__ == "__main__":
    main()
