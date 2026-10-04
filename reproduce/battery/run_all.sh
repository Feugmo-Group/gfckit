#!/usr/bin/env bash
# Reproduce every number and figure of the battery paper.
#
#   reproduce/battery/run_all.sh                 every analysis from the shipped data
#   reproduce/battery/run_all.sh --quick         the same, without the estimator ablation
#                                                (its family-maximum null takes hours)
#   reproduce/battery/run_all.sh --regenerate    first re-simulate the ensemble and
#                                                the aged impedance with PyBaMM (~8 min)
#
# Output goes to reproduce/battery/logs/ (one log per step) and figures/.
# The real-cell steps run only if data/real/ exists; fetch it with
# reproduce/battery/fetch_real_data.sh.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOGS="$ROOT/reproduce/battery/logs"
PY="${PYTHON:-python}"
cd "$ROOT"
mkdir -p "$LOGS" figures
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

FAILED=()
step() {   # name script [args]
  local name="$1"; shift
  printf '%-34s' "$name"
  local t0=$SECONDS
  if "$PY" "$@" > "$LOGS/$name.log" 2>&1; then
    echo "ok   ($((SECONDS - t0)) s)  logs/$name.log"
  else
    echo "FAIL ($((SECONDS - t0)) s)  logs/$name.log"; FAILED+=("$name")
  fi
}

QUICK=0; REGEN=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --regenerate) REGEN=1 ;;
    *) echo "unknown option $arg"; exit 2 ;;
  esac
done

if [ "$REGEN" = 1 ]; then
  echo "== regenerating data with PyBaMM (the generators skip cells already on disk) =="
  rm -rf data/pybamm_targeted data/aged_eis
  step 01_ensemble          examples/pybamm_targeted.py
  step 02_aged_eis          examples/aged_eis_run.py
fi

echo "== synthetic benchmark (Sections 3.1-3.4) =="
step 10_identify          examples/pybamm_identify.py
step 11_stats             examples/stats_report.py
if [ "$QUICK" = 1 ]; then
  echo "12_estimator_ablation              skipped (--quick)"
else
  step 12_estimator_ablation examples/mechanism_estimator_ablation.py
fi
step 13_aged_eis_identify examples/aged_eis_identify.py
step 14_cracking_checks   reproduce/battery/cracking_checks.py
step 15_paper_numbers     reproduce/battery/paper_numbers.py

echo "== figures =="
step 20_fig_pipeline      examples/make_pipeline_figure.py
step 21_fig_recovery      examples/battery_recovery_schematic.py
step 22_fig_nonlocal      examples/nonlocal_operator_figure.py
step 23_fig_overview      examples/pybamm_visualize.py
step 24_fig_eis           examples/pybamm_eis.py

if [ -d data/real/zhang2020 ] && [ -f data/real/oxford/oxford.mat ]; then
  echo "== real cells (Section 3.5) =="
  step 30_real_zhang      examples/real_identify.py
  step 31_real_cluster    examples/real_cluster.py
  step 32_real_oxford     examples/real_oxford.py
else
  echo "== real cells skipped: run reproduce/battery/fetch_real_data.sh first =="
fi

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "failed: ${FAILED[*]}"; exit 1
fi
echo "done. Compare the logs with the expected values in reproduce/battery/README.md."
