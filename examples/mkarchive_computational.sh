#!/bin/bash
# Assemble the deposit for the two computational papers.
#
# Unlike the LiPS campaign, these papers have no bulk data: everything is
# computed on demand from synthetic models or public datasets. What they do have
# that is nowhere on disk is the console output. Each script prints the numbers
# the manuscript quotes -- recovered orders, error floors, classification
# accuracies -- and then exits, leaving only the figure behind. A reader
# checking a number in the text against the code has to re-run it to see it.
#
# So this captures stdout per script alongside the figure it produced, giving
# each deposited figure a machine-readable record of the run that made it.
#
# Usage:  bash gfckit/examples/mkarchive_computational.sh [gfc|battery|all]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
EX="$ROOT/gfckit/examples"

GFC_SCRIPTS="recovery_schematic generate_benchmark order_identification \
multicondition physical_data delevie_validation keystone_construction \
identify_nonpowerlaw dynamic_range_study real_window"

# battery_recovery_schematic draws the battery paper's Fig. 2; recovery_schematic
# draws the identifiability paper's. The two used to share one file name in two
# figure directories, and only one of them had a generator.
BAT_SCRIPTS="make_pipeline_figure battery_recovery_schematic nonlocal_operator_figure \
pybamm_visualize pybamm_identify pybamm_eis real_identify real_oxford"

run_set() {
    local paper="$1"; shift
    local out="$ROOT/data/$paper"
    mkdir -p "$out"/{scripts,outputs,figures}
    echo "### $paper"
    for s in $@; do
        [ -f "$EX/$s.py" ] || { echo "  MISSING  $s.py"; continue; }
        cp "$EX/$s.py" "$out/scripts/"
        local log="$out/outputs/$s.log"
        local t0=$SECONDS
        if "$PY" "$EX/$s.py" > "$log" 2>&1; then
            printf '  ok    %-28s %4ds  %s\n' "$s" "$((SECONDS-t0))" \
                "$(wc -l < "$log") lines"
        else
            printf '  FAIL  %-28s %4ds  (see %s)\n' "$s" "$((SECONDS-t0))" \
                "outputs/$s.log"
        fi
    done
    # the figures the manuscript actually includes, taken from the manuscript
    local mfig="$ROOT/manuscripts/$paper/figures"
    if [ -d "$mfig" ]; then
        cp "$mfig"/*.pdf "$mfig"/*.png "$out/figures/" 2>/dev/null
        echo "  figures: $(ls "$out/figures" | wc -l)"
    fi
    # `find -printf` is GNU-only and this half of the archive is built on macOS,
    # where BSD find rejects it and would leave the manifest empty.
    ( cd "$out" && "$PY" -c "
import os, sys
rows = []
for dp, _, fns in os.walk('.'):
    for fn in fns:
        p = os.path.join(dp, fn)
        if os.path.basename(p) != 'MANIFEST.tsv':
            rows.append((os.path.getsize(p), p))
with open('MANIFEST.tsv', 'w') as fh:
    for s, p in sorted(rows, key=lambda r: r[1]):
        fh.write(f'{s}\t{p}\n')
" )
}

case "${1:-all}" in
    gfc)     run_set gfc_identifiability $GFC_SCRIPTS ;;
    battery) run_set battery_mechanisms  $BAT_SCRIPTS ;;
    all)     run_set gfc_identifiability $GFC_SCRIPTS
             run_set battery_mechanisms  $BAT_SCRIPTS ;;
    *) echo "usage: $0 [gfc|battery|all]"; exit 1 ;;
esac
