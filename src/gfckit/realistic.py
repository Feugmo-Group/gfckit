"""Realistic synthetic aging-data generation (pure JAX + stdlib I/O).

Turns the mechanistic / fractional forward models into publication-grade
synthetic datasets by adding the realism a reviewer expects:

  * real units and an Arrhenius temperature dependence;
  * an accelerated-aging TEST MATRIX (several temperatures) with REPLICATE cells;
  * cell-to-cell VARIABILITY (log-normal kinetic-parameter spread);
  * sparse, jittered CHECK-UP sampling (not a dense clean grid);
  * a realistic MEASUREMENT model: additive + multiplicative noise, slow sensor
    drift, occasional outliers, and quantization.

Ground truth (kinetic parameters, noise-free curve) is stored with every cell,
so identification accuracy can be scored exactly.
"""
import jax
import jax.numpy as jnp

kB_eV = 8.617333262e-5      # Boltzmann constant [eV/K]


# --------------------------------------------------------------------------
# Arrhenius acceleration relative to a reference temperature.
# --------------------------------------------------------------------------
def arrhenius_factor(T_C, Ea_eV, T_ref_C=25.0):
    """exp[-Ea/kB (1/T - 1/T_ref)] ; >1 for T>T_ref (faster aging)."""
    T = T_C + 273.15
    Tref = T_ref_C + 273.15
    return jnp.exp(-Ea_eV / kB_eV * (1.0 / T - 1.0 / Tref))


# --------------------------------------------------------------------------
# Reduced SEI growth law (mixed reaction-diffusion), EXACT closed form:
#   dL/dt = 1 / (1/k + L/D)   =>   L(t) = D ( sqrt(1/k^2 + 2t/D) - 1/k )
# reaction-limited early (L ~ k t), diffusion-limited late (L ~ sqrt(2 D t)).
# --------------------------------------------------------------------------
def reduced_sei_thickness(t, k, D):
    return D * (jnp.sqrt(1.0 / k ** 2 + 2.0 * t / D) - 1.0 / k)


# --------------------------------------------------------------------------
# Sparse, jittered check-up sampling (capacity checks every ~`every` days).
# --------------------------------------------------------------------------
def checkup_times(key, t_max, every=14.0, jitter=2.0):
    n = int(t_max // every)
    base = jnp.arange(1, n + 1) * every
    jit = jitter * jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0)
    t = jnp.clip(base + jit, 0.0, t_max)
    return jnp.concatenate([jnp.zeros(1), t])


# --------------------------------------------------------------------------
# Realistic measurement model applied to a clean curve y(t).
# --------------------------------------------------------------------------
def apply_measurement(key, t, y, sigma_abs=0.1, sigma_rel=0.002,
                      drift=0.05, outlier_p=0.01, outlier_scale=1.0,
                      quant=0.01):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    n = y.shape[0]
    add = sigma_abs * jax.random.normal(k1, (n,))
    mult = sigma_rel * y * jax.random.normal(k2, (n,))
    slope = drift * jax.random.normal(k3, ())            # per-cell sensor drift
    drift_term = slope * (t / (t[-1] + 1e-9))
    mask = jax.random.uniform(k4, (n,)) < outlier_p
    outliers = mask * outlier_scale * jax.random.normal(k4, (n,))
    ym = y + add + mult + drift_term + outliers
    return jnp.round(ym / quant) * quant                 # quantization


# --------------------------------------------------------------------------
# Battery calendar-aging cohort (mechanistic, realistic).
# Observable: capacity retention (%). Ground truth: per-cell (k, D).
# --------------------------------------------------------------------------
def generate_battery_cohort(temps_C=(25.0, 40.0, 55.0), n_cells=4, t_max=540.0,
                            checkup_every=14.0, k25=0.02, D25=0.01,
                            Ea_k=0.55, Ea_D=0.50, cap_scale=3.0,
                            cell_cv=0.08, seed=0):
    """Return a list of per-cell records with realistic capacity-fade data.

    Each record: dict(temp_C, cell, t_days, capacity_pct (measured),
    capacity_true_pct, k, D).
    """
    key = jax.random.PRNGKey(seed)
    records = []
    for T in temps_C:
        aT_k = float(arrhenius_factor(T, Ea_k))
        aT_D = float(arrhenius_factor(T, Ea_D))
        for cell in range(n_cells):
            key, kc, kt, km = jax.random.split(key, 4)
            zk, zd = jax.random.normal(kc, (2,))
            k = k25 * aT_k * jnp.exp(cell_cv * zk)         # cell-to-cell spread
            D = D25 * aT_D * jnp.exp(cell_cv * zd)
            t = checkup_times(kt, t_max, checkup_every)
            L = reduced_sei_thickness(t, k, D)
            cap_true = 100.0 - cap_scale * L
            cap_meas = apply_measurement(km, t, cap_true,
                                         sigma_abs=0.12, sigma_rel=0.001,
                                         drift=0.15, outlier_p=0.01,
                                         outlier_scale=0.8, quant=0.02)
            records.append(dict(temp_C=float(T), cell=int(cell), t_days=t,
                                capacity_pct=cap_meas, capacity_true_pct=cap_true,
                                k=float(k), D=float(D)))
    return records


# --------------------------------------------------------------------------
# CSV export (long format), stdlib only -- a shareable benchmark dataset.
# --------------------------------------------------------------------------
def cohort_to_csv(records, path):
    lines = ["temp_C,cell,day,capacity_pct,capacity_true_pct,k,D"]
    for r in records:
        t = list(map(float, r["t_days"]))
        cm = list(map(float, r["capacity_pct"]))
        ct = list(map(float, r["capacity_true_pct"]))
        for ti, ci, cti in zip(t, cm, ct):
            lines.append(f"{r['temp_C']:.1f},{r['cell']},{ti:.3f},"
                         f"{ci:.4f},{cti:.4f},{r['k']:.6g},{r['D']:.6g}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
