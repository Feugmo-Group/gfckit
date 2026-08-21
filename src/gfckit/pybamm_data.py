"""Load the saved PyBaMM aging ensemble into clean records (numpy only).

Each record exposes the OBSERVABLES (capacity, dV/dQ) and the GROUND TRUTH
(per-mechanism loss + dominant label). Shared by the visualization and the
mechanism-recovery experiment.
"""
import os
import glob
import numpy as np

MECHS = ["SEI", "plating", "cracks", "LAM"]


def load_cells(data_dir):
    """Return a list of cell dicts from data_dir/*.npz (skips failed/empty)."""
    cells = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        d = np.load(path, allow_pickle=True)
        if d["capacity_retention"].size < 5:
            continue
        cell_id = os.path.splitext(os.path.basename(path))[0]
        config, condition = cell_id.split("__", 1)
        cap0 = float(d["capacity_Ah"][0]) if d["capacity_Ah"].size else 1.0

        def endval(key):
            a = d[key]
            return float(a[-1]) if a.size else 0.0

        contrib = {
            "SEI": endval("loss_SEI_Ah"),
            "plating": endval("loss_plating_Ah"),
            "cracks": endval("loss_cracks_Ah"),
            "LAM": endval("LAM_neg_pct") / 100.0 * cap0,   # %→Ah-equivalent
        }
        rec = dict(
            cell_id=cell_id, config=config, condition=condition,
            cycle=np.asarray(d["cycle"], float),
            retention=np.asarray(d["capacity_retention"], float),
            dominant=str(d["dominant"]),
            contrib=contrib,
            dvdq_Q=np.asarray(d["dvdq_Q"], float),
            dvdq_V=np.asarray(d["dvdq_V"], float),
            dvdq=np.asarray(d["dvdq"], float),
        )
        # raw charge/discharge (cycler-like) curves, if present
        for nm in ("first", "mid", "last", "rpt"):
            if f"vq_{nm}_Q" in d:
                rec[f"vq_{nm}"] = (np.asarray(d[f"vq_{nm}_Q"], float),
                                   np.asarray(d[f"vq_{nm}_V"], float))
        cells.append(rec)
    return cells
