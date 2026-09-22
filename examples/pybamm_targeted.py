"""Balanced, AMPLIFIED PyBaMM ensemble so each mechanism is a clear dominant
contributor -- to test mechanism recovery on balanced classes.

Physical note: LAM and cracking amplify their *contribution fraction* but move
bulk capacity only weakly (they change impedance / electrode balance more than
cyclable capacity). This is exactly why capacity-based recovery struggles for
them and EIS / dV-dQ are needed -- the study makes that quantitative.

Writes data/pybamm_targeted/<cell_id>.npz + summary.csv (resumable).
"""
import os, json, time
import numpy as np

from gfckit.pybamm_gen import (run_cell, reference_dvdq, dominant_mechanism,
                               charge_discharge_curves)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "pybamm_targeted")
os.makedirs(OUT, exist_ok=True)
SUMMARY = os.path.join(OUT, "summary.csv")

SD = "solvent-diffusion limited"
AMP_LAM = {"Negative electrode LAM constant proportional term [s-1]": 2.8e-5,
           "Positive electrode LAM constant proportional term [s-1]": 2.8e-5}
AMP_LAM_MILD = {"Negative electrode LAM constant proportional term [s-1]": 8.3e-6,
                "Positive electrode LAM constant proportional term [s-1]": 8.3e-6}
# --- cracking amplification ---------------------------------------------------
# The SEI-on-cracks reaction draws its lithium from the ELECTROLYTE reservoir
# (PyBaMM: "Total lithium lost from electrolyte [mol]" == "Loss of lithium to
# negative SEI on cracks [mol]").  That reservoir holds only ~5.1e-3 mol
# ~= 0.137 A.h for OKane2022, so the cumulative crack loss is hard-capped there;
# past it the electrolyte concentration collapses (c_e -> ~20 mol.m-3) and IDA
# fails with IDA_ERR_FAIL.  This is a physical/model singularity, NOT a solver
# tolerance problem (rtol 1e-4..1e-8 and CasadiSolver all fail identically).
# Crack surface area scales as (roughness-1) = 2*rho_cr*l_cr0*w_cr, so amplifying
# rho_cr and l_cr0 together (the old 6.4e16 / 1.5e-7 setting) inflated it ~151x
# and drained the electrolyte inside one cycle -> 0-2 cycles completed.
# 2.25x the OKane2022 crack density is the LARGEST amplification that completes
# 200 cycles + RPT at all three conditions (2.5x dies at cycle 199 in hot_45C).
RHO_CR_0 = 3.18e15                      # OKane2022 default [m-2]
AMP_CRACK = {"Negative electrode number of cracks per unit area [m-2]": RHO_CR_0}
AMP_CRACK2 = {"Negative electrode number of cracks per unit area [m-2]": 2.25 * RHO_CR_0}

# (name, mechanisms, parameter_overrides) -- 2 variants per mechanism => balanced
CONFIGS = [
    ("SEI_ec",       {"SEI": "ec reaction limited"}, {}),
    ("SEI_solvent",  {"SEI": SD}, {}),
    ("plating_irr",  {"SEI": "reaction limited", "lithium plating": "irreversible"}, {}),
    ("plating_pr",   {"SEI": SD, "lithium plating": "partially reversible"}, {}),
    ("LAM_strong",   {"SEI": SD, "loss of active material": "stress-driven"}, AMP_LAM),
    ("LAM_mild",     {"SEI": SD, "loss of active material": "stress-driven"}, AMP_LAM_MILD),
    ("cracks_strong",{"SEI": SD, "particle mechanics": "swelling and cracking",
                      "SEI on cracks": "true"}, AMP_CRACK2),
    ("cracks_mild",  {"SEI": SD, "particle mechanics": "swelling and cracking",
                      "SEI on cracks": "true"}, AMP_CRACK),
]
CONDITIONS = {
    "mild_25C": dict(T_celsius=25.0, discharge_c=1.0, charge_c=0.5),
    "hot_45C": dict(T_celsius=45.0, discharge_c=1.0, charge_c=0.5),
    "cold_10C_fast": dict(T_celsius=10.0, discharge_c=1.0, charge_c=1.0),
}
N_CYCLES = 200


def main():
    if not os.path.exists(SUMMARY):
        with open(SUMMARY, "w") as f:
            f.write("cell_id,config,condition,n_cycles,retention_end,dominant,"
                    "loss_SEI,loss_plating,loss_cracks,LAM_neg_pct,status\n")
    total = len(CONFIGS) * len(CONDITIONS)
    done = 0
    for cname, mechs, ov in CONFIGS:
        for condname, cond in CONDITIONS.items():
            done += 1
            cell_id = f"{cname}__{condname}"
            npz = os.path.join(OUT, cell_id + ".npz")
            if os.path.exists(npz):
                print(f"[{done}/{total}] skip {cell_id}", flush=True); continue
            t0 = time.time()
            try:
                rec, sol = run_cell(mechs, fidelity="SPMe", n_cycles=N_CYCLES,
                                    parameter_overrides=ov, **cond)
                Q, V, dvdq = reference_dvdq(sol, -1)
                cd = charge_discharge_curves(sol, N_CYCLES)
                lab, contr = dominant_mechanism(rec)
                arrays = dict(
                    cycle=rec["cycle"], capacity_Ah=rec["capacity_Ah"],
                    capacity_retention=rec["capacity_retention"],
                    loss_SEI_Ah=rec["loss_SEI_Ah"], loss_plating_Ah=rec["loss_plating_Ah"],
                    loss_cracks_Ah=rec["loss_cracks_Ah"], LAM_neg_pct=rec["LAM_neg_pct"],
                    LLI_pct=rec["LLI_pct"], dvdq_Q=Q, dvdq_V=V, dvdq=dvdq,
                    config=json.dumps(mechs), condition=json.dumps(cond), dominant=lab)
                for nm, (cq, cv) in cd.items():
                    arrays[f"vq_{nm}_Q"] = cq; arrays[f"vq_{nm}_V"] = cv
                np.savez_compressed(npz, **arrays)
                line = (f"{cell_id},{cname},{condname},{len(rec['cycle'])},"
                        f"{rec['capacity_retention'][-1]:.4f},{lab},"
                        f"{rec['loss_SEI_Ah'][-1]:.4f},{rec['loss_plating_Ah'][-1]:.4f},"
                        f"{rec['loss_cracks_Ah'][-1]:.4f},{rec['LAM_neg_pct'][-1]:.3f},ok\n")
                status = f"ok ret={rec['capacity_retention'][-1]:.3f} dom={lab}"
            except Exception as e:
                line = f"{cell_id},{cname},{condname},0,,,,,,,FAIL:{type(e).__name__}\n"
                status = f"FAIL {type(e).__name__}"
            with open(SUMMARY, "a") as f:
                f.write(line)
            print(f"[{done}/{total}] {cell_id} {status} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nTARGETED ENSEMBLE DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
