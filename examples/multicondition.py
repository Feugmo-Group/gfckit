"""Multi-condition shared-kernel identification + figure.

Runs the full harness and writes `figures/multicondition_summary.png`:
  (a) aging curves (one kernel, several Arrhenius clocks)
  (b) parameter error vs number of conditions (identifiability)
  (c) strong vs weak form under noise (noise robustness)
  (d) recovered vs true tempered-power-law kernel
"""
import os
import jax
import jax.numpy as jnp

from gfckit.generate import build_conditions
from gfckit.identify import recover_params_strong, recover_params_weak
from gfckit.kernels import tempered_powerlaw_kernel
from gfckit.plotting import plot_multicondition_summary, mirror_figure

jax.config.update("jax_enable_x64", True)


def main():
    alpha, lam, c_sink = 0.5, 0.5, 1.0
    N, dt = 600, 0.02
    all_rates = [0.6, 0.85, 1.1, 1.4]         # normalized Arrhenius clock rates
    n_lag = 250
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(HERE, "figures", "multicondition_summary.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"true kernel: tempered power law  alpha={alpha}, lambda={lam} "
          f"(memory horizon 1/lam={1/lam:.0f}),  sink c={c_sink}\n")

    # (b) parameter error vs number of conditions (strong form, clean data)
    print("[b] (alpha,lambda) error vs number of conditions (clean):")
    n_conditions, err_a, err_l = [], [], []
    for C in [1, 2, 3, 4]:
        _, u_list, ds_list = build_conditions(alpha, lam, c_sink,
                                              all_rates[:C], N, dt, noise=0.0)
        a_hat, l_hat = recover_params_strong(u_list, ds_list, c_sink, n_lag)
        n_conditions.append(C)
        err_a.append(abs(a_hat - alpha))
        err_l.append(abs(l_hat - lam))
        print(f"    C={C}: alpha_hat={a_hat:.3f}, lambda_hat={l_hat:.3f}  "
              f"|err|=({err_a[-1]:.3f},{err_l[-1]:.3f})")

    # (c) strong vs weak under noise (C=4)
    print("\n[c] alpha recovery vs noise, strong vs weak (C=4):")
    noise_levels = [0.0, 0.01, 0.03, 0.05, 0.08]
    a_strong, a_weak = [], []
    for nz in noise_levels:
        _, u_list, ds_list = build_conditions(alpha, lam, c_sink, all_rates,
                                              N, dt, noise=nz, seed=7)
        a_s, _ = recover_params_strong(u_list, ds_list, c_sink, n_lag)
        a_w, _ = recover_params_weak(u_list, all_rates, c_sink, N, dt)
        a_strong.append(a_s)
        a_weak.append(a_w)
        print(f"    noise={nz:.0%}: strong alpha={a_s:.3f}   weak alpha={a_w:.3f}")

    # (a) data curves (clean C=4) + one noisy overlay
    t, u_clean, _ = build_conditions(alpha, lam, c_sink, all_rates, N, dt, noise=0.0)
    _, u_noisy_list, _ = build_conditions(alpha, lam, c_sink, all_rates, N, dt,
                                          noise=0.05, seed=7)
    u_noisy = u_noisy_list[-1]

    # (d) recovered kernel from weak form at 5% noise
    _, u_list, ds_list = build_conditions(alpha, lam, c_sink, all_rates, N, dt,
                                          noise=0.05, seed=7)
    a_hat, l_hat = recover_params_weak(u_list, all_rates, c_sink, N, dt)
    print(f"\n[d] recovered kernel (weak, 5% noise): alpha={a_hat:.3f}, lambda={l_hat:.3f}")
    tau = jnp.linspace(0.05, 6.0, 300)
    K_true = tempered_powerlaw_kernel(tau, alpha, lam)
    K_hat = tempered_powerlaw_kernel(tau, a_hat, l_hat)

    path = plot_multicondition_summary(
        t, u_clean, u_noisy, all_rates,
        n_conditions, err_a, err_l,
        noise_levels, a_strong, a_weak,
        tau, K_true, K_hat, alpha, lam, out)
    # keep the manuscript figure directory in step with its generator; the two
    # used to be reconciled by hand
    mirror_figure(path, os.path.join(HERE, os.pardir, "manuscripts",
                                    "gfc_identifiability", "figures"))
    print(f"\nwrote figure -> {path}")


if __name__ == "__main__":
    main()
