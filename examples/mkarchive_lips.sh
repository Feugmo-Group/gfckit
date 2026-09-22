#!/bin/bash
# Assemble the LAMMPS side of the LiPS campaign for deposit.
#
# What is kept: everything needed to re-run a state point (input decks, force
# field, starting and final structures, job scripts) and everything the paper's
# numbers were computed from (LAMMPS logs, the three-tier MSD tables, the
# landscape energies).
#
# What is dropped: the 68 GB of raw trajectories, which exceed a Zenodo record
# and are reproducible from the inputs and seeds; restart binaries, which are
# regenerable and architecture-specific; and lammps.out, which duplicates
# log.lammps.
set -e
SRC=~/scratch/fracMD
A=~/scratch/archive_lips
rm -rf "$A"; mkdir -p "$A"/{inputs,runs,landscape,analysis}

cp "$SRC"/common/{in.frac,in.landscape,in.landscape_all} "$A"/inputs/ 2>/dev/null || true
cp "$SRC"/common/*.inc "$SRC"/common/*.data "$SRC"/common/*.sw "$A"/inputs/
cp "$SRC"/generate.py "$SRC"/generate_lowT.py "$SRC"/submit_all.sh "$SRC"/submit_lowT.sh "$A"/inputs/ 2>/dev/null || true
cp "$SRC"/fracmd_*.py "$SRC"/lammps_to_mace_data.py "$A"/analysis/ 2>/dev/null || true

# per-run: job script, log, MSD tables, first and last structure
for d in "$SRC"/runs/*/T*/r*; do
    rel=${d#"$SRC"/runs/}
    o="$A/runs/$rel"; mkdir -p "$o"
    cp "$d"/job.sh "$o"/ 2>/dev/null || true
    for f in log.lammps msd_tierA.dat msd_tierB.dat msd_tierC.dat; do
        [ -f "$d/$f" ] && gzip -c "$d/$f" > "$o/$f.gz"
    done
    for f in "$d"/glass_*.data "$d"/final_*.data; do
        [ -f "$f" ] && gzip -c "$f" > "$o/$(basename "$f").gz"
    done
done

# landscape runs: job, log, extracted site energies
for d in "$SRC"/landscape/* "$SRC"/landscape_all/*; do
    [ -d "$d" ] || continue
    rel=$(basename "$(dirname "$d")")/$(basename "$d")
    o="$A/landscape/$rel"; mkdir -p "$o"
    cp "$d"/job.sh "$d"/landscape.npz "$o"/ 2>/dev/null || true
    [ -f "$d/log.lammps" ] && gzip -c "$d/log.lammps" > "$o/log.lammps.gz"
done

# derived analysis products
find "$SRC" -name '*.npz' -not -path '*/archive_lips/*' -exec cp --parents -t "$A/analysis" {} + 2>/dev/null || \
  for f in $(find "$SRC" -name '*.npz'); do
      r=${f#"$SRC"/}; mkdir -p "$A/analysis/$(dirname "$r")"; cp "$f" "$A/analysis/$r"
  done

cd "$A"
find . -type f -printf '%s\t%p\n' | sort -k2 > MANIFEST.tsv
du -sh "$A"
echo "files: $(find "$A" -type f | wc -l)"
