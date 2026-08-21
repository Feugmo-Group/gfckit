"""Generate realistic battery-aging data from PyBaMM (independent physics-based
ground truth) for the mechanism-recovery / identifiability study.

PyBaMM is the simulator (numpy/scipy/casadi under the hood); it is kept separate
from the pure-JAX gfckit identification core.  A run returns, per cycle:
  * the OBSERVABLE   -- capacity (and, on request, a low-rate voltage curve for dV/dQ);
  * the GROUND TRUTH -- capacity loss attributed to each mechanism (SEI, plating,
    SEI-on-cracks, LAM), which the identification is scored against.

Mechanisms available (see PyBaMM `BatteryModelOptions`):
  SEI: reaction / solvent-diffusion / electron-migration / interstitial-diffusion
       / ec-reaction / tunnelling limited
  lithium plating: reversible / partially reversible / irreversible
  particle mechanics: swelling only / swelling and cracking
  loss of active material: stress- / reaction- / current-driven (+combinations)
  SEI on cracks: true / false
"""
import numpy as np

try:
    import pybamm
except Exception as e:                       # keep gfckit importable without pybamm
    pybamm = None
    _IMPORT_ERROR = e


# ground-truth summary-variable names -> short keys (best-effort; absent ones -> 0)
_GT = {
    "LLI_pct":          "Loss of lithium inventory [%]",
    "loss_SEI_Ah":      "Loss of capacity to negative SEI [A.h]",
    "loss_plating_Ah":  "Loss of capacity to negative lithium plating [A.h]",
    "loss_cracks_Ah":   "Loss of capacity to negative SEI on cracks [A.h]",
    "LAM_neg_pct":      "Loss of active material in negative electrode [%]",
    "LAM_pos_pct":      "Loss of active material in positive electrode [%]",
}


def _safe_sv(sv, name, n):
    try:
        return np.asarray(sv[name], dtype=float)
    except Exception:
        return np.zeros(n)


def run_cell(mechanisms, fidelity="SPMe", parameter_set="OKane2022",
             n_cycles=100, T_celsius=25.0, discharge_c=1.0, charge_c=0.5,
             v_lo=2.5, v_hi=4.2, rpt=True, rpt_c=20, parameter_overrides=None,
             solver=None, experiment_period=None):
    """Simulate a cycling experiment and return (observables+ground-truth dict).

    If rpt=True, a low-rate C/`rpt_c` diagnostic discharge is appended at the end
    (a Reference Performance Test) so a CLEAN dV/dQ (differential-voltage) and
    dQ/dV (incremental-capacity) curve can be extracted -- exactly the standard
    diagnostic used on real Li-ion charge/discharge cycler data.

    mechanisms : dict of PyBaMM options, e.g.
        {"SEI": "solvent-diffusion limited",
         "lithium plating": "partially reversible",
         "particle mechanics": "swelling and cracking",
         "SEI on cracks": "true",
         "loss of active material": "stress-driven"}
    """
    if pybamm is None:
        raise ImportError(f"PyBaMM not available: {_IMPORT_ERROR}")
    Model = (pybamm.lithium_ion.SPMe if fidelity == "SPMe"
             else pybamm.lithium_ion.DFN)
    model = Model(options=dict(mechanisms))
    params = pybamm.ParameterValues(parameter_set)
    params["Ambient temperature [K]"] = 273.15 + T_celsius
    if parameter_overrides:                   # amplify mechanism rates, etc.
        for k, v in parameter_overrides.items():
            if k in params:
                params[k] = v

    cycle_block = (f"Discharge at {discharge_c}C until {v_lo}V",
                   f"Charge at {charge_c}C until {v_hi}V",
                   f"Hold at {v_hi}V until C/50")
    blocks = [cycle_block] * n_cycles
    if rpt:                                   # low-rate diagnostic (cell is full here)
        blocks.append((f"Discharge at C/{rpt_c} until {v_lo}V",))
    exp_kw = {} if experiment_period is None else {"period": experiment_period}
    sim = pybamm.Simulation(model, parameter_values=params,
                            experiment=pybamm.Experiment(blocks, **exp_kw),
                            solver=solver)
    sol = sim.solve()

    n_fade = n_cycles                         # exclude the RPT from the fade curve
    sv = sol.summary_variables
    cap = _safe_sv(sv, "Capacity [A.h]", len(sol.cycles))[:n_fade]
    out = {
        "cycle": np.arange(1, cap.size + 1),
        "capacity_Ah": cap,
        "capacity_retention": cap / cap[0] if cap.size and cap[0] else cap,
        "config": dict(mechanisms),
        "fidelity": fidelity, "T_celsius": T_celsius,
        "discharge_c": discharge_c, "charge_c": charge_c, "has_rpt": rpt,
    }
    for key, name in _GT.items():
        out[key] = _safe_sv(sv, name, len(sol.cycles))[:n_fade]
    return out, sol


def _discharge_curve(cycle):
    """Clean V vs discharge-capacity for a cycle's DISCHARGE step only."""
    step = cycle.steps[0]                     # step 0 = discharge
    Q = np.asarray(step["Discharge capacity [A.h]"].entries, dtype=float)
    try:
        V = np.asarray(step["Terminal voltage [V]"].entries, dtype=float)
    except Exception:
        V = np.asarray(step["Voltage [V]"].entries, dtype=float)
    order = np.argsort(Q)
    Q, V = Q[order], V[order]
    keep = np.concatenate([[True], np.diff(Q) > 1e-6])
    return Q[keep], V[keep]


def reference_dvdq(sol, cycle_index=-1):
    """Clean dV/dQ (DVA) and dQ/dV (ICA) from a cycle's discharge step.
    With rpt=True in run_cell, cycle_index=-1 is the low-rate RPT -> clean peaks."""
    Q, V = _discharge_curve(sol.cycles[cycle_index])
    dvdq = np.gradient(V, Q)
    dqdv = 1.0 / (dvdq + np.sign(dvdq + 1e-12) * 1e-6)   # ICA (incremental capacity)
    return Q, V, dvdq


def charge_discharge_curves(sol, n_cycles, indices=("first", "mid", "last", "rpt")):
    """Raw discharge V(Q) curves at chosen cycles -- the cycler-like curve data.
    Returns dict name -> (Q, V)."""
    out = {}
    picks = {"first": 0, "mid": max(0, n_cycles // 2), "last": n_cycles - 1,
             "rpt": len(sol.cycles) - 1}
    for name in indices:
        i = picks[name]
        if 0 <= i < len(sol.cycles):
            try:
                out[name] = _discharge_curve(sol.cycles[i])
            except Exception:
                pass
    return out


def dominant_mechanism(record):
    """Ground-truth label: which mechanism removed the most capacity by end of life."""
    contribs = {
        "SEI": float(record["loss_SEI_Ah"][-1]) if record["loss_SEI_Ah"].size else 0.0,
        "plating": float(record["loss_plating_Ah"][-1]) if record["loss_plating_Ah"].size else 0.0,
        "cracks": float(record["loss_cracks_Ah"][-1]) if record["loss_cracks_Ah"].size else 0.0,
        # LAM is a %; convert with initial capacity for a rough Ah-equivalent
        "LAM": float(record["LAM_neg_pct"][-1]) / 100.0 * float(record["capacity_Ah"][0])
               if record["LAM_neg_pct"].size else 0.0,
    }
    return max(contribs, key=contribs.get), contribs
