"""A labeled synthetic BENCHMARK suite of degradation curves (pure JAX).

Assembles two families with known ground truth, each with realistic sparse,
noisy sampling, for validating memory-kernel identification:

  POWER-LAW family (single anomalous order -- the easy/known baseline)
    * powerlaw_alpha    : u=t^alpha analytic
    * fick_uptake       : diffusion-limited sqrt(t) uptake (alpha=1/2)
    * sei_reduced       : mechanistic reaction-diffusion SEI (approx. power law)

  NON-POWER-LAW family (the novel, hard case -- needs GENERAL fractional calculus)
    * multiterm         : distributed / multi-order fractional relaxation
    * variable_order    : time-varying anomalous order alpha(t)
    * knee_battery       : multi-mechanism capacity fade with a plating knee
    * pdm_passivation    : Point Defect Model film growth (saturating)

For each curve we record the clean signal, a realistic measured signal, the
ground truth, the best single-power-law exponent and its R^2, and the variation
of the local log-log slope (≈0 => power law; large => not).
"""
import jax
import jax.numpy as jnp

from .generate import analytic_powerlaw
from .fick import diffusion_uptake
from .realistic import reduced_sei_thickness, apply_measurement
from .complex import (multiterm_fractional_relaxation, variable_order_solution,
                      knee_capacity_fade, local_loglog_slope)
from .physics import simulate_pdm_film_growth
from .identify import recover_order_weak


def _powerlaw_quality(t, y):
    """Best single-power-law exponent, its R^2, and local-slope variation."""
    a = float(recover_order_weak(y, t, jnp.linspace(0.05, 1.6, 201)))
    Phi = jnp.stack([t ** a, jnp.ones_like(t)], axis=1)
    coef, *_ = jnp.linalg.lstsq(Phi, y, rcond=None)
    yhat = Phi @ coef
    ss_res = float(jnp.sum((y - yhat) ** 2))
    ss_tot = float(jnp.sum((y - jnp.mean(y)) ** 2)) + 1e-12
    R2 = 1.0 - ss_res / ss_tot
    _, slope = local_loglog_slope(t, y)
    n0 = slope.shape[0] // 10
    slope_var = float(jnp.std(slope[n0:]))
    return a, R2, slope_var


def _measure(key, t, y, sigma_abs):
    """Sparse (~40 check-ups) + realistic measurement noise."""
    n = t.shape[0]
    step = max(1, n // 40)
    idx = jnp.arange(0, n, step)
    tk, yk = t[idx], y[idx]
    ym = apply_measurement(key, tk, yk, sigma_abs=sigma_abs, sigma_rel=0.001,
                           drift=0.5 * sigma_abs, outlier_p=0.02,
                           outlier_scale=3 * sigma_abs,
                           quant=max(sigma_abs / 5, 1e-4))
    return tk, yk, ym


def make_benchmark(seed=0):
    key = jax.random.PRNGKey(seed)
    out = []

    def add(name, family, t, y, gt, sigma_abs):
        nonlocal key
        key, kk = jax.random.split(key)
        tk, yk, ym = _measure(kk, t, y, sigma_abs)
        a, R2, sv = _powerlaw_quality(tk, jnp.clip(yk, 1e-9))
        out.append(dict(name=name, family=family, t=tk, y_clean=yk, y_meas=ym,
                        ground_truth=gt, pl_alpha=a, pl_R2=R2, slope_var=sv))

    # ---------------- POWER-LAW family ----------------
    t = jnp.linspace(1e-3, 5.0, 600)
    u, _ = analytic_powerlaw(t, 0.5)
    add("powerlaw_alpha", "power-law", t, u,
        {"alpha": 0.5}, sigma_abs=0.01)

    t = jnp.linspace(0.02, 8.0, 600)
    add("fick_uptake", "power-law", t, diffusion_uptake(t, D=1.0),
        {"alpha": 0.5, "law": "2 c_s sqrt(Dt/pi)"}, sigma_abs=0.02)

    t = jnp.linspace(1e-3, 540.0, 600)
    L = reduced_sei_thickness(t, k=0.02, D=0.01)
    add("sei_reduced", "power-law", t, 3.0 * L,
        {"k": 0.02, "D": 0.01, "note": "reaction->diffusion, ->1/2"}, sigma_abs=0.05)

    # ---------------- NON-POWER-LAW family ----------------
    tt, u = multiterm_fractional_relaxation(alphas=[0.3, 0.8], weights=[0.6, 0.4],
                                            lam=0.02, S=0.03, u0=0.0,
                                            dt=0.05, nsteps=500)
    add("multiterm", "non-power-law", tt, u,
        {"alphas": [0.3, 0.8], "weights": [0.6, 0.4], "lam": 0.02}, sigma_abs=2e-3)

    def alpha_path(x):
        return 0.3 + 0.6 / (1.0 + jnp.exp(-(x - 10.0) / 2.0))
    tt, u = variable_order_solution(alpha_path, lam=0.02, S=0.03, u0=0.0,
                                    dt=0.05, nsteps=500)
    add("variable_order", "non-power-law", tt, u,
        {"alpha_path": "0.3 -> 0.9 sigmoid at t=10"}, sigma_abs=3e-3)

    t = jnp.linspace(1.0, 540.0, 600)
    y = knee_capacity_fade(t, A_sei=0.6, B_plate=0.18, t_knee=350.0, w_knee=30.0)
    add("knee_battery", "non-power-law", t, y,
        {"A_sei": 0.6, "B_plate": 0.18, "t_knee": 350.0}, sigma_abs=0.08)

    t, L, L_ss = simulate_pdm_film_growth(nsteps=500, dt=0.02)
    add("pdm_passivation", "non-power-law", t, L,
        {"L_ss": L_ss, "model": "Point Defect Model"}, sigma_abs=0.01)

    return out


def benchmark_to_csv(datasets, path):
    """Long-format CSV of all measured curves; stdlib only."""
    lines = ["dataset,family,t,y_measured,y_clean,pl_alpha,pl_R2,slope_var"]
    for d in datasets:
        t = list(map(float, d["t"]))
        ym = list(map(float, d["y_meas"]))
        yc = list(map(float, d["y_clean"]))
        for ti, mi, ci in zip(t, ym, yc):
            lines.append(f"{d['name']},{d['family']},{ti:.4f},{mi:.5f},{ci:.5f},"
                         f"{d['pl_alpha']:.4f},{d['pl_R2']:.4f},{d['slope_var']:.4f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
