"""Generate the full synthetic benchmark: power-law AND non-power-law families.

Writes:
  data/benchmark.csv               -- all measured curves (long format)
  data/battery_cohort.csv          -- realistic multi-temperature battery cohort
  figures/benchmark_gallery.png    -- every curve + best single-power-law fit
  figures/benchmark_diagnostic.png -- proof that some curves are NOT power laws
"""
import os
import jax

from gfckit.benchmark import make_benchmark, benchmark_to_csv
from gfckit.realistic import generate_battery_cohort, cohort_to_csv
from gfckit.plotting import plot_benchmark_gallery, plot_benchmark_diagnostic, mirror_figure

jax.config.update("jax_enable_x64", True)


def main():
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(HERE, "data", "synthetic")
    fig_dir = os.path.join(HERE, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # ---- the labeled benchmark suite (both families) ----
    ds = make_benchmark(seed=0)
    print("Benchmark suite (single-power-law test per curve):")
    print(f"  {'dataset':<18}{'family':<16}{'pl_alpha':>9}{'R^2':>9}{'slope_var':>11}")
    for d in ds:
        verdict = "power-law" if (d["pl_R2"] > 0.999 and d["slope_var"] < 0.05) \
            else "NOT power-law"
        print(f"  {d['name']:<18}{d['family']:<16}{d['pl_alpha']:>9.3f}"
              f"{d['pl_R2']:>9.4f}{d['slope_var']:>11.3f}   {verdict}")

    csv1 = benchmark_to_csv(ds, os.path.join(data_dir, "benchmark.csv"))

    # ---- realistic multi-temperature battery cohort ----
    cohort = generate_battery_cohort(temps_C=(25.0, 40.0, 55.0), n_cells=4, seed=1)
    csv2 = cohort_to_csv(cohort, os.path.join(data_dir, "battery_cohort.csv"))
    n_rows = sum(len(r["t_days"]) for r in cohort)
    print(f"\nBattery cohort: {len(cohort)} cells x check-ups = {n_rows} rows "
          f"(3 temperatures, cell-to-cell variability, sparse noisy sampling)")

    # ---- figures ----
    g1 = plot_benchmark_gallery(ds, os.path.join(fig_dir, "benchmark_gallery.png"))
    g2 = plot_benchmark_diagnostic(ds, os.path.join(fig_dir, "benchmark_diagnostic.png"))

    # keep the manuscript figure directory in step with its generator; the two
    # used to be reconciled by hand
    mirror_figure(g2, os.path.join(HERE, os.pardir, "manuscripts",
                                    "gfc_identifiability", "figures"))
    print("\nwrote:")
    for p in (csv1, csv2, g1, g2):
        print("  " + p)


if __name__ == "__main__":
    main()
