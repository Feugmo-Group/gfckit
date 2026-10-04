"""PyBaMM checks behind the cracking result (paper Section 3.3).

  (0) the electrolyte lithium reservoir that caps crack-driven SEI loss
      (paper: 0.144 A h, 2.8% of the 5.11 A h first-cycle capacity Q0);
  (a) cycling a cracking cell leaves the intercalation surface area unchanged
      while the roughness ratio grows (paper: 3.8396e5 m-1 at the first and last
      cycle, roughness 5.3);
  (b) sweeping the crack-SEI thickness over six decades leaves the
      charge-transfer resistance flat (paper: flat to five figures).

Check (b) solves the impedance with the particle-mechanics and SEI-on-cracks
submodels switched on, so the crack-SEI thickness is a live parameter of the
model being perturbed.

Needs pybamm and pybammeis; runs in under a minute.
"""
import os

import numpy as np
import pybamm
import pybammeis

from gfckit.aged_eis import eis_features
from gfckit.pybamm_gen import run_cell

CRACK_OPTS = {"SEI": "solvent-diffusion limited",
              "particle mechanics": "swelling and cracking",
              "SEI on cracks": "true"}
ENS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "pybamm_targeted")
RHO_KEY = "Negative electrode number of cracks per unit area [m-2]"


def reservoir():
    p = pybamm.ParameterValues("OKane2022")
    F = 96485.33212
    A = p["Electrode width [m]"] * p["Electrode height [m]"]
    vol = sum(p[f"{r} porosity"] * p[f"{r} thickness [m]"]
              for r in ("Negative electrode", "Separator", "Positive electrode")) * A
    mol = p["Initial concentration in electrolyte [mol.m-3]"] * vol
    ah = mol * F / 3600
    # Q0: first-cycle discharge capacity of the ensemble cells
    q0 = float(np.load(os.path.join(ENS, "cracks_strong__mild_25C.npz"))["capacity_Ah"][0])
    print(f"(0) electrolyte lithium at t=0: {mol:.2e} mol = {ah:.3f} A h = {ah / q0:.1%} of "
          f"Q0 = {q0:.2f} A h   paper: 0.144 A h, 2.8% of 5.11 A h")


def surface_area():
    p0 = pybamm.ParameterValues("OKane2022")
    _, sol = run_cell(CRACK_OPTS, fidelity="SPMe", n_cycles=200, T_celsius=25.0,
                      discharge_c=1.0, charge_c=0.5, rpt=False,
                      parameter_overrides={RHO_KEY: 2.25 * p0[RHO_KEY]})
    a = np.asarray(sol["Negative electrode surface area to volume ratio [m-1]"].entries)
    rough = np.asarray(sol["X-averaged negative electrode roughness ratio"].entries)
    first, last = float(np.mean(a[..., 0])), float(np.mean(a[..., -1]))
    print(f"(a) surface area to volume ratio: first {first:.5g}, last {last:.5g} m-1 "
          f"({last / first - 1:+.2e}); roughness ratio {float(rough[-1]):.2f}"
          f"   paper: 3.8396e5 at both, roughness 5.3")


def crack_sei_sweep():
    model = pybamm.lithium_ion.SPMe(options={"surface form": "differential", **CRACK_OPTS})
    freqs = np.logspace(-2, 4, 40)
    print("(b) R_ct vs initial crack-SEI thickness, cracking submodels on:")
    out = []
    for L in (5e-13, 5e-11, 5e-9, 5e-7):
        p = pybamm.ParameterValues("OKane2022")
        p["Initial SEI on cracks thickness [m]"] = L
        sim = pybammeis.EISSimulation(model, parameter_values=p, initial_soc=0.5)
        sim.solve(freqs)
        rct = eis_features(freqs, np.asarray(sim.solution))["R_ct"]
        out.append(rct)
        print(f"      L = {L:.0e} m   R_ct = {rct:.6g} Ohm")
    out = np.array(out)
    print(f"    relative spread {np.ptp(out) / out.mean():.1e}   paper: flat to five figures")


if __name__ == "__main__":
    reservoir()
    crack_sei_sweep()
    surface_area()
