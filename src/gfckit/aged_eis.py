"""Per-cell AGED electrochemical impedance spectroscopy.

The companion analysis (`eis_gen.py`) could only do a mechanism-parameter
SENSITIVITY study, because `pybammeis.EISSimulation` builds its own model and
cannot ingest a cycled solution object.  That left the paper's central
prescription -- "capacity cannot see cracking or LAM, so use impedance" --
untested on the paper's own benchmark.

This module closes that gap.  The approach is:

  1. cycle the cell to end of life and read the AGED INTERNAL STATE out of the
     PyBaMM solution (active-material volume fractions, porosity, SEI and
     crack-SEI thickness, lithium inventory, roughness);
  2. build a parameter set carrying those aged values;
  3. solve EIS on that parameter set at a controlled state of charge.

This is faithful to how impedance is actually measured in the laboratory: an
aged cell is brought to a specified SOC and then perturbed.  It is NOT a claim
that the aged concentration PROFILES are reproduced -- EIS is a small-signal
measurement about a controlled operating point, so the operating point is set by
SOC, and the degradation enters through the parameters above.

WHAT MOVES THE IMPEDANCE (measured, not assumed -- each parameter was swept
individually and the response recorded):

    negative active material volume fraction 0.75 -> 0.40 : R_ct  +79 %
    SEI thickness                            5e-9 -> 5e-7 : R_ct +116 %
    negative porosity                        0.25 -> 0.15 : R_s   +44 %
    lithium inventory loss                      0 -> 20 % : R_ct   +5 %
    SEI-on-cracks thickness                 5e-13 -> 5e-7 : R_ct    0 %

*** CRACKING IS STRUCTURALLY UNOBSERVABLE IN THIS FRAMEWORK. ***
Two independent checks establish this and both are negative:
  (a) the intercalation surface area is EXACTLY unchanged by cycling a cracking
      cell ("Negative electrode surface area to volume ratio" 3.8396e5 -> 3.8396e5,
      +0.00 %, while the roughness ratio sits at 5.3), so crack roughness never
      reaches the charge-transfer resistance;
  (b) sweeping the crack-SEI thickness over six decades leaves R_ct flat.
PyBaMM couples crack roughness only to the SEI-on-cracks REACTION area, never to
the intercalation area.  Together with the electrolyte-reservoir ceiling that
already caps crack capacity loss at 2.7 % of Q0, cracking cannot be resolved by
capacity OR impedance here.  The benchmark therefore cannot test the prescription
"capacity misses cracking, so use impedance"; that must be stated as a bound on
the study rather than reported as a negative physical finding about real cells.

The SEI submodel MUST be active in the EIS model: with only
"SEI film resistance" set and no "SEI" option there is no SEI state for it to act
on, and SEI thickness silently has zero effect on Z.
"""
import numpy as np

try:
    import pybamm
    import pybammeis
except Exception as e:            # pragma: no cover
    pybamm = None
    _ERR = e

# aged-state variable -> how we read it out of the solution
_STATE_VARS = {
    "eps_am_n":   "X-averaged negative electrode active material volume fraction",
    "eps_am_p":   "Positive electrode active material volume fraction",
    "eps_n":      "Negative electrode porosity",
    "L_sei":      "X-averaged negative SEI thickness [m]",
    "roughness":  "X-averaged negative electrode roughness ratio",
    "Li_part":    "Total lithium in particles [mol]",
    "LLI_pct":    "Loss of lithium inventory [%]",
}

# frequencies: 6 decades, the range a real potentiostat sweeps
DEFAULT_FREQS = np.logspace(-2, 4, 40)


def extract_aged_state(sol):
    """Read the end-of-life internal state out of a cycled solution."""
    out = {}
    for key, name in _STATE_VARS.items():
        try:
            v = np.asarray(sol[name].entries, dtype=float)
            out[key] = float(np.mean(v[..., -1])) if v.ndim > 1 else float(v[-1])
        except Exception:
            out[key] = np.nan
    # crack-SEI is optional (only present for cracking configs)
    try:
        v = np.asarray(sol["X-averaged negative SEI on cracks thickness [m]"].entries,
                       dtype=float)
        out["L_sei_cr"] = float(np.mean(v[..., -1])) if v.ndim > 1 else float(v[-1])
    except Exception:
        out["L_sei_cr"] = np.nan
    return out


def aged_parameter_values(aged, parameter_set="OKane2022"):
    """Build a ParameterValues carrying the aged state."""
    p = pybamm.ParameterValues(parameter_set)
    fresh_Li = None
    def setif(key, val):
        if key in p and val is not None and np.isfinite(val):
            p[key] = float(val)

    setif("Negative electrode active material volume fraction", aged.get("eps_am_n"))
    setif("Positive electrode active material volume fraction", aged.get("eps_am_p"))
    setif("Negative electrode porosity", aged.get("eps_n"))
    setif("Initial SEI thickness [m]", aged.get("L_sei"))
    setif("Initial SEI on cracks thickness [m]", aged.get("L_sei_cr"))

    # lithium inventory loss -> scale the initial stoichiometry of both electrodes
    lli = aged.get("LLI_pct")
    if lli is not None and np.isfinite(lli) and 0.0 <= lli < 100.0:
        f = 1.0 - lli / 100.0
        for k in ("Initial concentration in negative electrode [mol.m-3]",
                  "Initial concentration in positive electrode [mol.m-3]"):
            if k in p:
                p[k] = float(p[k]) * f
    return p


def compute_aged_eis(aged, freqs=None, parameter_set="OKane2022",
                     initial_soc=0.5, film_resistance="average"):
    """Solve EIS for an aged parameter state. Returns (freqs, Z complex)."""
    if pybamm is None:                       # pragma: no cover
        raise ImportError(f"pybamm/pybammeis not available: {_ERR}")
    freqs = DEFAULT_FREQS if freqs is None else np.asarray(freqs, float)
    # ONE common model for every cell, so the only thing that differs between
    # cells is the aged parameter state and the comparison stays controlled.
    # The "SEI" submodel is required (not merely "SEI film resistance"): without
    # a SEI state the film-resistance option is a silent no-op and SEI thickness
    # has zero effect on Z.  Verified by sweeping L_sei over two decades.
    opts = {"surface form": "differential",
            "SEI": "solvent-diffusion limited",
            "SEI film resistance": film_resistance}
    model = pybamm.lithium_ion.SPMe(options=opts)
    p = aged_parameter_values(aged, parameter_set)
    sim = pybammeis.EISSimulation(model, parameter_values=p, initial_soc=initial_soc)
    sim.solve(freqs)
    return freqs, np.asarray(sim.solution)


def eis_features(freqs, Z):
    """Scalar impedance descriptors (mirrors real_data.eis_features on Zhang data)."""
    Zre, Zim = np.real(Z), -np.imag(Z)        # Nyquist convention: -Im on y
    o = np.argsort(freqs)
    f, Zre, Zim = freqs[o], Zre[o], Zim[o]
    hi = f >= max(f[0], 5e2)
    R_s = float(np.min(Zre[hi])) if hi.any() else float(np.min(Zre))
    i1 = int(np.argmin(np.abs(f - 1.0)))      # ~1 Hz: after the charge-transfer arc
    R_ct = float(Zre[i1] - R_s)
    ipk = int(np.argmax(Zim))
    return {"R_s": R_s, "R_ct": R_ct,
            "peak_negZim": float(Zim[ipk]), "f_peak": float(f[ipk]),
            "Zre_lo": float(Zre[0]), "negZim_lo": float(Zim[0])}
