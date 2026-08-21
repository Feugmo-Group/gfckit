"""Electrochemical Impedance Spectroscopy (EIS) via pybammeis.

EIS adds the FREQUENCY dimension (6-7 decades) that a single aging curve lacks.
`pybammeis.EISSimulation` computes impedance from a model at a given SOC with
given parameters; it does not ingest an aged state directly, so per-cell aged
EIS would require injecting the degraded state.  Here we instead run a
mechanism-parameter SENSITIVITY study: perturb the parameters each mechanism
changes (SEI/contact resistance up; active material down for LAM; lithium down
for LLI) and show that the Nyquist signatures separate -- i.e. EIS distinguishes
mechanisms that capacity alone cannot.
"""
import numpy as np

try:
    import pybamm
    import pybammeis
except Exception as e:
    pybamm = None
    _ERR = e


def compute_eis(freqs, overrides=None, model_options=None,
                parameter_set="OKane2022", initial_soc=0.5):
    """Return complex impedance Z(freqs) for a model with parameter overrides.

    "contact resistance" is on by default. PyBaMM ignores the "Contact
    resistance [Ohm]" parameter unless the model is built with that option, so
    with it off the contact-resistance scenario returned a spectrum bit-identical
    to the unperturbed one, and the Nyquist figure carried two curves under two
    different labels. With it on the spectrum shifts by exactly the series
    resistance, as a contact resistance must.
    """
    if pybamm is None:
        raise ImportError(f"pybamm/pybammeis not available: {_ERR}")
    opts = {"surface form": "differential", "contact resistance": "true"}
    if model_options:
        opts.update(model_options)
    model = pybamm.lithium_ion.SPMe(options=opts)
    params = pybamm.ParameterValues(parameter_set)
    if overrides:
        for k, v in overrides.items():
            if k in params:
                params[k] = v
    eis = pybammeis.EISSimulation(model, parameter_values=params,
                                  initial_soc=initial_soc)
    eis.solve(np.asarray(freqs, float))
    return np.asarray(eis.solution)


def mechanism_scenarios(parameter_set="OKane2022"):
    """Mechanism-representative parameter perturbations (only existing keys used)."""
    base = pybamm.ParameterValues(parameter_set)

    def sc(key, factor):
        return {key: base[key] * factor} if key in base else {}

    # Labels are what the figure legend prints, so they spell the perturbation
    # out. "SEI/contact R (+)" said neither which quantity moved nor by how
    # much, and the caption defined only two of the four series.
    scenarios = {"fresh (baseline)": {}}
    lam = {}
    lam.update(sc("Negative electrode active material volume fraction", 0.75))
    lam.update(sc("Positive electrode active material volume fraction", 0.75))
    scenarios[r"LAM ($-25\%$ active material)"] = lam
    scenarios[r"LLI ($-20\%$ lithium inventory)"] = sc(
        "Initial concentration in negative electrode [mol.m-3]", 0.8)
    if "Contact resistance [Ohm]" in base:
        dr = 0.02
        r_new = float(base["Contact resistance [Ohm]"]) + dr
        scenarios[fr"SEI/contact resistance ($+{dr * 1e3:.0f}$ m$\Omega$)"] = \
            {"Contact resistance [Ohm]": r_new}
    else:
        scenarios[r"low negative-electrode conductivity ($\times0.4$)"] = sc(
            "Negative electrode conductivity [S.m-1]", 0.4)
    return scenarios
