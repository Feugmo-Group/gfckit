"""Check the two Newton-Krylov mechanistic solvers against outside references.

SEI model.  When the concentration profile in the film is quasi-steady, it is
linear, the flux balance D (c_b - u_0) / L = k u_0 gives u_0 = c_b / (1 + k L / D),
and the growth law dL/dt = Om k u_0 integrates to the linear-parabolic law of
Deal and Grove (1965):
    L + k L^2 / (2 D) = L_0 + k L_0^2 / (2 D) + Om k c_b t.
This is a published closed form that the moving-boundary solver does not use.
It holds when the film moves slowly compared with diffusion across it, so the
check reports the error over the whole run and over its late part.

PDM model.  dL/dt = k_g exp(b dV / L) - k_d has the analytic steady state
L_ss = b dV / ln(k_d / k_g).  The transient is checked against an independent
adaptive integrator (scipy LSODA, rtol 1e-10), and the time step of the
backward-Euler solver is halved to show first-order convergence.

Settings are those of physical_data.py (Fig. 4 of the paper).
"""
import jax
import numpy as np
from scipy.integrate import solve_ivp

from gfckit.physics import simulate_pdm_film_growth, simulate_sei_reaction_diffusion

jax.config.update("jax_enable_x64", True)


def deal_grove(t, D=1.0, k=1.0, Om=1.0, cb=1.0, L0=0.05):
    a = k / (2 * D)
    c = L0 + a * L0 ** 2 + Om * k * cb * t
    return (-1 + np.sqrt(1 + 4 * a * c)) / (2 * a)


def main():
    print("SEI moving-boundary solver vs Deal-Grove closed form")
    for dt, nsteps in ((0.02, 400), (0.01, 800)):
        t, L = simulate_sei_reaction_diffusion(nsteps=nsteps, dt=dt)
        t, L = np.asarray(t), np.asarray(L)
        ref = deal_grove(t)
        rel = np.abs(L - ref) / ref
        late = t >= 1.0
        print(f"  dt={dt}: max rel. error {rel.max():.2e} (all t), "
              f"{rel[late].max():.2e} (t>=1); L(8)={L[-1]:.4f} vs {ref[-1]:.4f}")

    print("\nPDM implicit solver vs analytic steady state and LSODA")
    kg, kd, bdV, L0 = 0.5, 1.0, 1.0, 0.5
    for dt, nsteps in ((0.02, 400), (0.01, 800), (0.005, 1600)):
        t, L, L_ss = simulate_pdm_film_growth(nsteps=nsteps, dt=dt)
        t, L = np.asarray(t), np.asarray(L)
        sol = solve_ivp(lambda _t, y: kg * np.exp(min(bdV / y[0], 40.0)) - kd,
                        (0, t[-1]), [L0], t_eval=t, method="LSODA",
                        rtol=1e-10, atol=1e-12)
        err = np.max(np.abs(L - sol.y[0]) / sol.y[0])
        print(f"  dt={dt}: max rel. error vs LSODA {err:.2e}; "
              f"L(end)={L[-1]:.5f}, analytic L_ss={L_ss:.5f}")


if __name__ == "__main__":
    main()
