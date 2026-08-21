"""gfckit -- General-Fractional-Calculus toolkit for data-driven identification
of degradation memory kernels (batteries, corrosion).

Based on Tarasov's parametric general fractional calculus (arXiv:2509.12218).
See `docs/gfc_degradation_identification.pdf` for the full mathematical roadmap.
"""
from .operators import (
    gamma, caputo_L1, apply_gfd, substitution_gfd,
    expsum_L1_weights, tempered_powerlaw_L1_weights,
)
from .kernels import tempered_powerlaw_kernel, expsum_kernel
from .generate import analytic_powerlaw, solve_gfode_constant, build_conditions
from .identify import (
    recover_order_strong, recover_order_weak,
    recover_params_strong, recover_params_weak,
    recover_free_kernel, kernel_shape_error,
    powerlaw_reconstruction, recover_order_spectrum,
)
from .physics import (
    newton_krylov, simulate_sei_reaction_diffusion, simulate_pdm_film_growth,
)
from .fick import (
    semi_infinite_constant_surface, semi_infinite_surface_flux,
    diffusion_uptake, instantaneous_point_source, finite_slab, msd_gaussian,
)
from .delevie import (
    pore_impedance, semi_infinite_pore_impedance, crossover_frequency,
    cpe_impedance, fit_cpe_exponent, local_exponent,
)
from .complex import (
    multiterm_fractional_relaxation, variable_order_solution,
    knee_capacity_fade, local_loglog_slope,
)
from .realistic import (
    arrhenius_factor, reduced_sei_thickness, generate_battery_cohort,
    cohort_to_csv, apply_measurement, checkup_times,
)
from .benchmark import make_benchmark, benchmark_to_csv
from .laplace import talbot_inverse, distributed_order_response

__version__ = "0.1.0"

__all__ = [
    "gamma", "caputo_L1", "apply_gfd", "substitution_gfd",
    "expsum_L1_weights", "tempered_powerlaw_L1_weights",
    "tempered_powerlaw_kernel", "expsum_kernel",
    "analytic_powerlaw", "solve_gfode_constant", "build_conditions",
    "recover_order_strong", "recover_order_weak",
    "recover_params_strong", "recover_params_weak",
    "recover_free_kernel", "kernel_shape_error",
    "powerlaw_reconstruction", "recover_order_spectrum",
    "newton_krylov", "simulate_sei_reaction_diffusion",
    "simulate_pdm_film_growth",
    "semi_infinite_constant_surface", "semi_infinite_surface_flux",
    "diffusion_uptake", "instantaneous_point_source", "finite_slab",
    "msd_gaussian",
    "pore_impedance", "semi_infinite_pore_impedance", "crossover_frequency",
    "cpe_impedance", "fit_cpe_exponent", "local_exponent",
    "multiterm_fractional_relaxation", "variable_order_solution",
    "knee_capacity_fade", "local_loglog_slope",
    "arrhenius_factor", "reduced_sei_thickness", "generate_battery_cohort",
    "cohort_to_csv", "apply_measurement", "checkup_times",
    "make_benchmark", "benchmark_to_csv",
    "talbot_inverse", "distributed_order_response",
]
