"""Generate the FULL PyBaMM aging ensemble for the mechanism-recovery study.

Design:
  * ~12 mechanism configurations (SEI variants; +plating; +LAM; +cracking; combos)
  * 3 accelerated conditions (mild 25C; hot 45C -> SEI; cold+fast 10C/1C -> plating)
  * long cycling (default 300 cycles) with SPMe

Each cell is saved incrementally to data/pybamm/<id>.npz (RESUMABLE: existing
cells are skipped) and appended to data/pybamm/summary.csv.  Robust to per-cell
solver failures (logged, skipped).  Observables = capacity/retention + dV/dQ;
ground truth = per-mechanism capacity loss.
"""
import os, json, time, traceback
import numpy as np

from gfckit.pybamm_gen import (run_cell, reference_dvdq, dominant_mechanism,
                               charge_discharge_curves)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "pybamm")
os.makedirs(OUT, exist_ok=True)

SEI_SD = "solvent-diffusion limited"

CONFIGS = {
    "SEI_solvent":     {"SEI": SEI_SD},
    "SEI_reaction":    {"SEI": "reaction limited"},
    "SEI_ec":          {"SEI": "ec reaction limited"},
    "SEI_plating_pr":  {"SEI": SEI_SD, "lithium plating": "partially reversible"},
    "SEI_plating_irr": {"SEI": SEI_SD, "lithium plating": "irreversible"},
    "SEI_LAM_stress":  {"SEI": SEI_SD, "loss of active material": "stress-driven"},
    "SEI_LAM_reaction":{"SEI": SEI_SD, "loss of active material": "reaction-driven"},
    "SEI_crack":       {"SEI": SEI_SD, "particle mechanics": "swelling and cracking",
                        "SEI on cracks": "true"},
    "SEI_plating_LAM": {"SEI": SEI_SD, "lithium plating": "partially reversible",
                        "loss of active material": "stress-driven"},
    "full":            {"SEI": SEI_SD, "lithium plating": "partially reversible",
                        "particle mechanics": "swelling and cracking",
                        "SEI on cracks": "true", "loss of active material": "stress-driven"},
    "plating_heavy":   {"SEI": "reaction limited", "lithium plating": "irreversible"},
    "LAM_current":     {"SEI": SEI_SD, "loss of active material": "current-driven"},
}

CONDITIONS = {
    "mild_25C":   dict(T_celsius=25.0, discharge_c=1.0, charge_c=0.5),
    "hot_45C":    dict(T_celsius=45.0, discharge_c=1.0, charge_c=0.5),
    "cold_10C_fast": dict(T_celsius=10.0, discharge_c=1.0, charge_c=1.0),
}

N_CYCLES = 300
SUMMARY = os.path.join(OUT, "summary.csv")


def main():
    if not os.path.exists(SUMMARY):
        with open(SUMMARY, "w") as f:
            f.write("cell_id,config,condition,n_cycles,retention_end,dominant,"
                    "loss_SEI,loss_plating,loss_cracks,LAM_neg_pct,status\n")
    t_all = time.time()
    total = len(CONFIGS) * len(CONDITIONS)
    done = 0
    for cname, mechs in CONFIGS.items():
        for condname, cond in CONDITIONS.items():
            done += 1
            cell_id = f"{cname}__{condname}"
            npz = os.path.join(OUT, cell_id + ".npz")
            if os.path.exists(npz):
                print(f"[{done}/{total}] skip {cell_id} (exists)", flush=True)
                continue
            t0 = time.time()
            try:
                rec, sol = run_cell(mechs, fidelity="SPMe", n_cycles=N_CYCLES, **cond)
                Q, V, dvdq = reference_dvdq(sol, -1)          # clean low-rate RPT
                cd = charge_discharge_curves(sol, N_CYCLES)   # raw cycler-like curves
                lab, contr = dominant_mechanism(rec)
                arrays = dict(
                    cycle=rec["cycle"], capacity_Ah=rec["capacity_Ah"],
                    capacity_retention=rec["capacity_retention"],
                    loss_SEI_Ah=rec["loss_SEI_Ah"], loss_plating_Ah=rec["loss_plating_Ah"],
                    loss_cracks_Ah=rec["loss_cracks_Ah"], LAM_neg_pct=rec["LAM_neg_pct"],
                    LLI_pct=rec["LLI_pct"], dvdq_Q=Q, dvdq_V=V, dvdq=dvdq,
                    config=json.dumps(mechs), condition=json.dumps(cond), dominant=lab)
                for nm, (cq, cv) in cd.items():               # BOL/mid/EOL + RPT curves
                    arrays[f"vq_{nm}_Q"] = cq
                    arrays[f"vq_{nm}_V"] = cv
                np.savez_compressed(npz, **arrays)
                line = (f"{cell_id},{cname},{condname},{len(rec['cycle'])},"
                        f"{rec['capacity_retention'][-1]:.4f},{lab},"
                        f"{rec['loss_SEI_Ah'][-1]:.4f},{rec['loss_plating_Ah'][-1]:.4f},"
                        f"{rec['loss_cracks_Ah'][-1]:.4f},{rec['LAM_neg_pct'][-1]:.3f},ok\n")
                status = f"ok  ret={rec['capacity_retention'][-1]:.3f} dom={lab}"
            except Exception as e:
                line = f"{cell_id},{cname},{condname},0,,,,,,,FAIL:{type(e).__name__}\n"
                status = f"FAIL {type(e).__name__}: {e}"
                traceback.print_exc()
            with open(SUMMARY, "a") as f:
                f.write(line)
            print(f"[{done}/{total}] {cell_id}  {status}  ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"\nENSEMBLE DONE in {(time.time()-t_all)/60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
