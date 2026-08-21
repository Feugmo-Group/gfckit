"""Per-cell AGED EIS for the 24-cell targeted ensemble.

Re-cycles each cell, extracts the end-of-life internal state from the PyBaMM
solution, and solves EIS on a parameter set carrying that aged state.
Writes gfckit/data/aged_eis/<cell_id>.npz and summary.csv.

See gfckit.aged_eis for what does and does not move the impedance -- in
particular, cracking is structurally unobservable in this framework.
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gfckit.pybamm_gen import run_cell, dominant_mechanism
from gfckit.aged_eis import (extract_aged_state, compute_aged_eis,
                             eis_features, DEFAULT_FREQS)

sys.path.insert(0, os.path.dirname(__file__))
from pybamm_targeted import CONFIGS, CONDITIONS, N_CYCLES

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "aged_eis")
os.makedirs(OUT, exist_ok=True)
SUMMARY = os.path.join(OUT, "summary.csv")
FEATS = ["R_s", "R_ct", "peak_negZim", "f_peak", "Zre_lo", "negZim_lo"]


def main():
    if not os.path.exists(SUMMARY):
        with open(SUMMARY, "w") as f:
            f.write("cell_id,config,condition,dominant,retention_end,"
                    + ",".join(FEATS) + ",eps_am_n,eps_am_p,eps_n,L_sei,"
                    "roughness,LLI_pct,status\n")
    total, done = len(CONFIGS) * len(CONDITIONS), 0
    for cname, mechs, ov in CONFIGS:
        for condname, cond in CONDITIONS.items():
            done += 1
            cell_id = f"{cname}__{condname}"
            npz = os.path.join(OUT, cell_id + ".npz")
            if os.path.exists(npz):
                print(f"[{done}/{total}] skip {cell_id}", flush=True)
                continue
            t0 = time.time()
            try:
                rec, sol = run_cell(mechs, fidelity="SPMe", n_cycles=N_CYCLES,
                                    parameter_overrides=ov, rpt=False, **cond)
                aged = extract_aged_state(sol)
                freqs, Z = compute_aged_eis(aged)
                ft = eis_features(freqs, Z)
                lab, _ = dominant_mechanism(rec)
                ret = float(rec["capacity_retention"][-1])
                np.savez_compressed(
                    npz, freqs=freqs, Z_re=np.real(Z), Z_im=np.imag(Z),
                    config=json.dumps(dict(mechs)), condition=json.dumps(cond),
                    cell_id=cell_id, config_name=cname, condition_name=condname,
                    dominant=lab, retention_end=ret,
                    **{k: float(v) for k, v in aged.items()},
                    **{k: float(v) for k, v in ft.items()})
                row = [cell_id, cname, condname, lab, "%.4f" % ret]
                row += ["%.6g" % ft[k] for k in FEATS]
                row += ["%.6g" % aged.get(k, np.nan) for k in
                        ("eps_am_n", "eps_am_p", "eps_n", "L_sei",
                         "roughness", "LLI_pct")]
                row.append("ok")
                status = "ok"
            except Exception as e:
                row = [cell_id, cname, condname, "NA", "NA"] + ["NA"] * (len(FEATS) + 6)
                row.append("FAIL:%s" % type(e).__name__)
                status = "FAIL:%s: %s" % (type(e).__name__, str(e)[:80])
            with open(SUMMARY, "a") as f:
                f.write(",".join(row) + "\n")
            print("[%d/%d] %-30s %6.0fs  %s" % (done, total, cell_id,
                                                time.time() - t0, status),
                  flush=True)


if __name__ == "__main__":
    main()
