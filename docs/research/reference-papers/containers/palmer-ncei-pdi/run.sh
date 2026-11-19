#!/usr/bin/env bash
# REFIMPL-30: run NCEI's public Fortran Palmer program (pdi.f) on the 344 fixture divisions.
# Nothing from upstream is committed; this fetches it, verifies the pinned checksums, patches
# only the documented parameter statement in a scratch copy, compiles and runs.
#
# usage: run.sh <repo-root> <work-dir>      needs: curl, gfortran, uv (or python3 with numpy)
set -euo pipefail
REPO=$(cd "$1" && pwd)
WORK=$(mkdir -p "$2" && cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=https://www1.ncdc.noaa.gov/pub/data/nidis/software/ncei/palmer
PDI_SHA256=e0c58ccbdc5fe694e9218e317065cf14cb6095de329e1dd43087fab67f20cb9e   # pdi.f, 2013-04-03, 33877 bytes
SOIL_SHA256=40fc3a55a0c91aa1f83ce753b45950b2dbebad7c2fb7d23c9e829416cae588c2  # soilcnst.asc, 1998-01-12, 17286 bytes

mkdir -p "$WORK/upstream" "$WORK/run"
cd "$WORK/upstream"
curl --proto '=https' --tlsv1.2 -sSfL -o pdi.f "$BASE/pdi.f"
curl --proto '=https' --tlsv1.2 -sSfL -o soilcnst.asc "$BASE/soilcnst.asc"
echo "$PDI_SHA256  pdi.f" | shasum -a 256 -c -
echo "$SOIL_SHA256  soilcnst.asc" | shasum -a 256 -c -

# Rows 1-58 of soilcnst.asc are another series whose ids collide with division ids under
# Fortran blank-null reads (division 1010 would read the row for "101 0"); rows 59+ are the
# 344 divisions. prep.py asserts each division's AWC matches tests/fixture/palmer_awc.json.
tail -n +59 soilcnst.asc > soilcnst_div.asc

cd "$WORK/run"
cp ../upstream/soilcnst_div.asc fort.14
uv run --project "$REPO" python -I "$HERE/prep.py" "$REPO" "$WORK/upstream/soilcnst_div.asc" "$WORK/run"

# The program documents that analysis years are set by editing its parameter statement.
sed -e 's/nendyr=2000,/nendyr=2022,/' -e 's/nendmo= 9,/nendmo=12,/' -e 's/iouteyr=2000)/iouteyr=2022)/' \
  ../upstream/pdi.f > pdi_patched.f
gfortran -std=legacy -O0 -w -o pdi pdi_patched.f
./pdi
# Outputs, 344 divisions x 1895-2022: fort.21 WPLM (PMDI), fort.22 PDSI, fort.23 Z, fort.24 PHDI.
for u in 21 22 23 24; do [[ "$(wc -l < fort.$u)" -eq $((344 * 128)) ]]; done
echo "ok: $WORK/run/fort.2[1-4] hold 344 x 128 rows each"
