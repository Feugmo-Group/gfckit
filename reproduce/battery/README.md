# Reproducing the battery paper

C. G. Tetsassi Feugmo, *Data-driven identifiability of battery degradation
mechanisms: what capacity, differential voltage, and impedance can each
resolve*, Journal of Energy Storage (under review).

This folder reproduces every number and figure in that paper. The 24-cell
PyBaMM ensemble and the per-cell aged impedance ship with the repository
(`data/pybamm_targeted/`, `data/aged_eis/`), so the analyses run without any
simulation. The two public real-cell datasets are downloaded by a script.

## Setup

```bash
git clone https://github.com/Feugmo-Group/gfckit && cd gfckit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[battery]"
```

The `battery` extra installs PyBaMM 26.7.0.0, pybammeis 0.1.6 and scikit-learn
1.8.0, the versions the paper's results come from, plus SciPy and matplotlib.
Two of these pins matter: other PyBaMM releases change the dV/dQ results (see
below), and scikit-learn 1.9 changes the linear discriminant analysis score in
the estimator ablation from 12.5% to 20.8%.

## Run

```bash
reproduce/battery/fetch_real_data.sh   # Zhang 2020 + Oxford datasets, ~390 MB, once
reproduce/battery/run_all.sh           # every analysis and figure from the shipped data
```

`run_all.sh` writes one log per step to `reproduce/battery/logs/` and the
figures to `figures/`. Everything except the estimator ablation finishes in about
a minute; the ablation's family-maximum permutation null (2,000 permutations of
nine classifiers) takes about 1.5 hours. `run_all.sh --quick` skips that one
step.

To re-simulate the ensemble and the aged impedance from scratch with PyBaMM
instead of using the shipped files, run

```bash
reproduce/battery/run_all.sh --regenerate
```

Regeneration takes about eight minutes on a laptop and overwrites
`data/pybamm_targeted/` and `data/aged_eis/` (`git checkout data/` restores the
shipped files). With PyBaMM 26.7.0.0 and pybammeis 0.1.6, the versions the
`battery` extra pins, it reproduces the shipped files exactly: the ensemble bit
for bit, the impedance to 1e-15.

Other PyBaMM releases (26.5.0, 26.6.2, 26.7.1 and 26.8.0 were tested) age the
stress-driven LAM cells slightly differently, by up to 1.3 percentage points of
retention; every dominant-mechanism label is unchanged. On an ensemble
regenerated with PyBaMM 26.8.0, the capacity-only results are unchanged (37.5%
attribution, p = 0.064, all four fraction R^2 values), but the dV/dQ results
move: capacity + dV/dQ scores 33.3% (8/24) instead of 20.8%.

## What each step reproduces

Section numbers refer to the paper. Every value below was checked against a
clean run of these scripts.

### Synthetic benchmark

| Paper | Step (log) | Expected |
|---|---|---|
| Sec. 2.2 and Appendix B: the 8 configurations x 3 conditions, 200 cycles, C/20 reference test | `01_ensemble` (`examples/pybamm_targeted.py`) | 24 cells; dominant mechanism SEI 7, cracks 6, LAM 6, plating 5 |
| Sec. 3.1: fade exponent gamma by family | `15_paper_numbers` | LAM 1.12 (0.95-1.29); irreversible plating 0.51 (0.39-0.61); SEI 0.72 (0.61-0.89); cracking 0.83 |
| Sec. 3.2: attribution from capacity, leave-one-configuration-out | `10_identify`, `11_stats` | 37.5% (9/24), Wilson CI [21.2, 57.3]%, majority baseline 29.2% |
| Sec. 3.2: configuration-level permutation null | `11_stats` | null mean 14.1%, p = 0.064 |
| Sec. 3.2: capacity + dV/dQ | `10_identify`, `11_stats` | 20.8% (5/24), CI [9.2, 40.5]%, p = 0.30; McNemar p = 0.125 |
| Sec. 3.2: leave-one-cell-out (biased) | `10_identify` | 45.8% |
| Sec. 3.2: eight further classifiers, family-maximum null, tie-break and scaler checks | `12_estimator_ablation` | span 12.5% to 50.0%; 1-NN 50.0% (p = 0.026 alone; 0.079 as family maximum, null mean 29.6%); first-class-wins tie-break 29.2%; LDA 12.5% to 50.0% and nearest centroid 41.7% to 50.0% with dV/dQ |
| Sec. 3.2: per-class recall from capacity | `13_aged_eis_identify` | cracks 4/6, LAM 3/6, SEI 2/7, plating 0/5 |
| Sec. 3.2: mechanism-fraction R^2 from capacity | `10_identify`, `11_stats` | SEI -0.77, plating -0.77, cracks -0.46, LAM +0.53 |
| Sec. 3.2: LAM grades smoothly, plating is near binary | `15_paper_numbers` | LAM fade 2% to 95%; plating 66-72% or under 1%; mean fade 0.41 (plating) vs 0.29 (LAM) |
| Sec. 3.3: electrolyte reservoir | `14_cracking_checks` | 0.144 A h, 2.8% of Q0 = 5.11 A h |
| Sec. 3.3: crack retention shift against the crack-free control (`SEI_solvent`, same condition) | `15_paper_numbers` | +5.7e-5 to +1.7e-4, median +1.0e-4; cracks carry 70-87% of attributed loss; fade at most 0.39% |
| Sec. 3.3: intercalation area unchanged by cracking | `14_cracking_checks` | 3.8396e5 m-1 at first and last cycle, roughness 5.35 |
| Sec. 3.3: crack-SEI thickness sweep, 5e-13 to 5e-7 m | `14_cracking_checks` | R_ct constant (0.0252334 Ohm) |
| Sec. 3.4: aged impedance | `02_aged_eis` (data), `13_aged_eis_identify` | capacity 37.5% (p = 0.075); + aged EIS 33.3%; EIS alone 16.7%; R^2 SEI -1.06, plating -0.90 |
| Sec. 3.4: half the ensemble barely ages; the matched pair | `15_paper_numbers` | 12 of 24 cells above 99% retention; SEI 67.1% vs LAM 67.8%, R_ct 0.146 vs 0.044 Ohm (x3.3) |

### Real cells (Section 3.5, needs `fetch_real_data.sh`)

| Paper | Step (log) | Expected |
|---|---|---|
| R_ct vs capacity fade, 12 Zhang cells | `11_stats`, `30_real_zhang` | Pearson r = -0.62, p = 0.032, CI [-0.88, -0.07]; Spearman -0.70, p = 0.011 |
| R_ct drift and temperature confound | `15_paper_numbers` | R_ct -0.7%; R_ct vs T r = -0.83 (1.37 vs 0.48 Ohm at 25/45 C); fade vs T r = +0.24 |
| Fade exponent vs temperature | `15_paper_numbers`, `30_real_zhang` | 25 C 0.29 (0.06-0.56), 35 C 0.49, 45 C 0.59; slope +0.016 per C, r = +0.63, p = 0.027 |
| Clustering of capacity + EIS features | `11_stats`, `31_real_cluster` | purity 75% (baseline 67%), adjusted Rand 0.23, permutation p = 0.40 |
| Oxford dV/dQ evolution | `32_real_oxford` | Fig. 9 |

### Figures

| Figure | Step | File in `figures/` |
|---|---|---|
| Fig. 1 workflow | `20_fig_pipeline` | `pipeline.pdf` |
| Fig. 2 what is recovered | `21_fig_recovery` | `battery_recovery.pdf` |
| Fig. 3 nonlocal operators | `22_fig_nonlocal` | `nonlocal_operator.pdf` |
| Fig. 5 aging ensemble | `23_fig_overview` | `pybamm_overview_targeted.pdf` |
| Fig. 6 confusion matrices | `10_identify` | `pybamm_confusion_targeted.pdf` |
| Fig. 7 Nyquist sensitivity | `24_fig_eis` | `pybamm_eis_nyquist.pdf` |
| Fig. 8 real-cell features | `30_real_zhang` | `real_features.pdf` |
| Fig. 9 Oxford dataset | `32_real_oxford` | `real_oxford.pdf` |

Fig. 4 (the identification pipeline) is a LaTeX diagram with no computed content.

## Notes


- The permutation tests use fixed seeds, so the p-values reproduce exactly.
  `11_stats` and `13_aged_eis_identify` compute the capacity null with two
  separate implementations and report p = 0.064 and p = 0.075; the paper
  discusses the difference (Sec. 3.4).
- The crack-free control for the cracking cells is the `SEI_solvent`
  configuration: same SEI submodel, same three conditions, no particle
  mechanics.
