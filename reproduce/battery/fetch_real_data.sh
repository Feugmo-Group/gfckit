#!/usr/bin/env bash
# Download the two public real-cell datasets used in Section 3.5 of the battery
# paper into data/real/. About 390 MB in total; run once from anywhere.
#
#   Zhang et al. 2020, Nat. Commun. 11, 1706   Zenodo 3633835 (capacity + EIS, 12 cells)
#   Birkl 2017, Oxford Battery Degradation Dataset 1   doi:10.5287/bodleian:KO2kdmYGg
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REAL="$ROOT/data/real"
mkdir -p "$REAL/zhang2020" "$REAL/oxford"

fetch() {   # url dest md5
  if [ -f "$2" ] && [ "$(md5sum_of "$2")" = "$3" ]; then
    echo "have $(basename "$2")"; return
  fi
  echo "downloading $(basename "$2") ..."
  curl -fL --retry 3 -o "$2" "$1"
  [ "$(md5sum_of "$2")" = "$3" ] || { echo "checksum mismatch: $2" >&2; exit 1; }
}
md5sum_of() { if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }

Z="https://zenodo.org/api/records/3633835/files"
fetch "$Z/Capacity%20data.zip/content" "$REAL/zhang2020/Capacity_data.zip" 0dba677f0beeb15b450ef35f5b4faac8
fetch "$Z/EIS%20data.zip/content"      "$REAL/zhang2020/EIS_data.zip"      9f31069d770d7d323d8e2bf21267afe2
(cd "$REAL/zhang2020" && unzip -qo Capacity_data.zip && unzip -qo EIS_data.zip && rm -rf __MACOSX)

ORA="https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/files"
fetch "$ORA/m5ac36a1e2073852e4f1f7dee647909a7" "$REAL/oxford/oxford.mat" 7775c81359ee488a0c905e7654519421

echo "real-cell data ready in $REAL"
