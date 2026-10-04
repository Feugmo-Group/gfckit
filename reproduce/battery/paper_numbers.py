"""Numbers in the battery paper that no other script prints.

Tetsassi Feugmo, "Data-driven identifiability of battery degradation mechanisms:
what capacity, differential voltage, and impedance can each resolve"
(Journal of Energy Storage, under review).

Reads only shipped or downloaded data, runs in seconds:

  * data/pybamm_targeted/   the 24-cell ensemble (shipped with the repository)
  * data/aged_eis/          per-cell aged impedance (shipped)
  * data/real/zhang2020/    Zhang et al. 2020, Zenodo 3633835 (download first;
                            the real-cell block is skipped if it is missing)

Each line prints the computed value next to the value printed in the paper,
with the section it comes from.
"""
import csv
import os

import numpy as np

from gfckit.mechanism_id import observable_features

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENS = os.path.join(ROOT, "data", "pybamm_targeted")
AGED = os.path.join(ROOT, "data", "aged_eis")
ZHANG = os.path.join(ROOT, "data", "real", "zhang2020")
CONDS = ("mild_25C", "hot_45C", "cold_10C_fast")


def row(label, value, paper, where):
    print(f"  {label:<52s} {value:<26s} paper: {paper:<22s} [{where}]")


def load(cell_id):
    return np.load(os.path.join(ENS, cell_id + ".npz"))


def gamma(d):
    feat, _ = observable_features({"cycle": d["cycle"], "retention": d["capacity_retention"]})
    return float(feat[3])


def ensemble():
    print("\n=== Synthetic ensemble (data/pybamm_targeted) ===")
    configs = {}
    for f in sorted(os.listdir(ENS)):
        if f.endswith(".npz"):
            configs.setdefault(f.split("__")[0], []).append(f[:-4])

    # fade-shape exponent gamma by family (Sec. 3.1)
    fam = {"stress-driven LAM": ["LAM_strong", "LAM_mild"],
           "irreversible plating": ["plating_irr"],
           "SEI": ["SEI_ec", "SEI_solvent"],
           "cracking": ["cracks_strong", "cracks_mild"]}
    paper = {"stress-driven LAM": "1.12 (0.95-1.29)", "irreversible plating": "0.51 (0.39-0.61)",
             "SEI": "0.72 (0.61-0.89)", "cracking": "0.83"}
    for name, cfgs in fam.items():
        g = np.array([gamma(load(c)) for k in cfgs for c in configs[k]])
        g = g[np.isfinite(g)]
        row(f"gamma, {name} (mean, range; n={g.size})",
            f"{g.mean():.2f} ({g.min():.2f}-{g.max():.2f})", paper[name], "3.1")

    ret = {c: float(load(c)["capacity_retention"][-1]) for k in configs for c in configs[k]}
    fade = {c: 1.0 - r for c, r in ret.items()}

    # LAM grades smoothly, plating is near binary (Sec. 3.2)
    lam = [fade[c] for k in ("LAM_strong", "LAM_mild") for c in configs[k]]
    irr = [fade[c] for c in configs["plating_irr"]]
    pr = [fade[c] for c in configs["plating_pr"]]
    row("LAM end-of-life fade span", f"{min(lam):.0%} to {max(lam):.0%}", "2% to 95%", "3.2")
    row("irreversible plating fade", f"{min(irr):.0%}-{max(irr):.0%}", "66-72%", "3.2")
    row("partially reversible plating fade, max", f"{max(pr):.2%}", "under 1%", "3.2")
    by_dom = {}
    for k in configs:
        for c in configs[k]:
            by_dom.setdefault(str(load(c)["dominant"]), []).append(fade[c])
    row("mean fade, plating- vs LAM-dominant cells",
        f"{np.mean(by_dom['plating']):.2f} vs {np.mean(by_dom['LAM']):.2f}", "0.41 vs 0.29", "3.2")
    n99 = sum(r > 0.99 for r in ret.values())
    row("cells retaining more than 99% of capacity", f"{n99} of {len(ret)}", "12 of 24", "3.4")

    # cracking: the matched crack-free control is SEI_solvent under the same condition
    # (same SEI submodel, no particle mechanics), Sec. 3.3
    shifts, shares, fades = [], [], []
    for k in ("cracks_strong", "cracks_mild"):
        for cond in CONDS:
            d = load(f"{k}__{cond}")
            shifts.append(ret[f"{k}__{cond}"] - ret[f"SEI_solvent__{cond}"])
            losses = np.array([d["loss_SEI_Ah"][-1], d["loss_plating_Ah"][-1],
                               d["loss_cracks_Ah"][-1]])
            shares.append(losses[2] / losses.sum())
            fades.append(fade[f"{k}__{cond}"])
    shifts = np.array(shifts)
    row("crack retention shift vs crack-free control",
        f"{shifts.min():+.1e} to {shifts.max():+.1e}", "+5.7e-05 to +1.7e-04", "3.3")
    row("  median shift", f"{np.median(shifts):+.1e}", "+1.0e-04", "3.3")
    row("crack share of attributed loss", f"{min(shares):.0%}-{max(shares):.0%}", "70-87%", "3.3")
    row("crack cells, largest total fade", f"{max(fades):.2%}", "0.39%", "3.3")


def matched_pair():
    print("\n=== Aged impedance (data/aged_eis) ===")
    with open(os.path.join(AGED, "summary.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    by_id = {r["cell_id"]: r for r in rows}
    # pairs of degraded cells (<99% retention) with different dominant mechanisms
    # whose end-of-life retention agrees to within 1%
    pairs = [(a, b) for i, a in enumerate(rows) for b in rows[i + 1:]
             if a["dominant"] != b["dominant"]
             and max(float(a["retention_end"]), float(b["retention_end"])) < 0.99
             and abs(float(a["retention_end"]) - float(b["retention_end"])) < 0.01]
    row("capacity-matched pairs (within 1%)",
        "; ".join(f"{a['cell_id']} / {b['cell_id']}" for a, b in pairs),
        "two, sharing a cell", "3.4")
    a, b = by_id["SEI_ec__mild_25C"], by_id["LAM_strong__mild_25C"]
    ra, rb = float(a["R_ct"]), float(b["R_ct"])
    row("pair reported in the paper: retention",
        f"{float(a['retention_end']):.1%} / {float(b['retention_end']):.1%}",
        "SEI 67.1% / LAM 67.8%", "3.4")
    row("  R_ct", f"{ra:.3f} vs {rb:.3f} Ohm", "0.146 vs 0.044 Ohm", "3.4")
    row("  R_ct ratio", f"{max(ra, rb) / min(ra, rb):.1f}", "3.3", "3.4")


def real_cells():
    print("\n=== Real cells (Zhang et al. 2020) ===")
    if not os.path.isdir(ZHANG):
        print("  data/real/zhang2020 not found; see reproduce/battery/README.md to download it")
        return
    from scipy import stats

    from gfckit.real_data import eis_features, load_zhang_capacity, load_zhang_eis

    cap, eis = load_zhang_capacity(ZHANG), load_zhang_eis(ZHANG)
    T, fade, g, rct_end, drift = [], [], [], [], []
    for key in sorted(set(cap) & set(eis)):
        cyc, cp = cap[key]
        ret = cp / cp[0]
        feat, _ = observable_features({"cycle": cyc, "retention": ret})
        states = eis[key]
        r0 = eis_features(*states[0][1:])["R_ct"]
        r1 = eis_features(*states[-1][1:])["R_ct"]
        T.append(float(key[0])); fade.append(1.0 - ret[-1]); g.append(float(feat[3]))
        rct_end.append(r1); drift.append(r1 / r0 - 1.0)
    T, fade, g, rct_end = map(np.array, (T, fade, g, rct_end))

    row("cells", f"{T.size}", "12", "3.5")
    row("mean R_ct change, first to last state", f"{np.mean(drift):+.1%}", "-0.7%", "3.5")
    row("R_ct vs temperature, Pearson r", f"{stats.pearsonr(T, rct_end)[0]:+.2f}", "-0.83", "3.5")
    row("mean end-of-life R_ct at 25 C / 45 C",
        f"{rct_end[T == 25].mean():.2f} / {rct_end[T == 45].mean():.2f} Ohm",
        "1.37 / 0.48 Ohm", "3.5")
    row("fade vs temperature, Pearson r", f"{stats.pearsonr(T, fade)[0]:+.2f}", "+0.24", "3.5")
    lr = stats.linregress(T, g)
    row("gamma vs temperature: slope, r, p",
        f"{lr.slope:+.3f}/C, {lr.rvalue:+.2f}, {lr.pvalue:.3f}", "+0.016/C, +0.63, 0.027", "3.5")
    for t in sorted(set(T)):
        sel = g[T == t]
        row(f"  gamma at {t:.0f} C (n={sel.size})",
            f"{sel.mean():.2f} ({sel.min():.2f}-{sel.max():.2f})",
            {25: "0.29 (0.06-0.56)", 35: "0.49", 45: "0.59"}[int(t)], "3.5")


if __name__ == "__main__":
    ensemble()
    matched_pair()
    real_cells()
