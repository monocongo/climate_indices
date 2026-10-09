#!/bin/bash
# Runs inside a Fargate task or an EC2 instance: build the Rust extension, then run one stage.
#
# The extension is built here rather than baked into an image, so the task
# definition stays image-agnostic and a stage cannot silently measure a stale
# build of the crate.
#
# Environment:
#   STAGE     routine (default) | spread | grid | grid_synthetic | grid_real
#   GIT_REF   branch or commit to benchmark (default: main)
#   REPEATS   repetitions per backend (default: 15)
#   CPUS      taskset CPU set for the measured process (default: 0-7)
#   NETCDF    prepared precipitation grid, grid stage only
#   TAVG      matching mean-temperature grid, grid stage only
#   GRID_ROWS, GRID_COLS  synthetic grid dimensions, grid_synthetic stage only
#   GRID_ENTRIES, GRID_THREADS  narrow the grid entries or thread counts
#   PERCELL_CELLS, PERCELL_REPEATS  sample size and repetitions for the percell stage
set -euxo pipefail
export HOME=/root

STAGE="${STAGE:-routine}"
GIT_REF="${GIT_REF:-main}"
REPEATS="${REPEATS:-15}"
CPUS="${CPUS:-0-7}"

# Optional narrowing so a first run can verify the path before paying for every
# entry at every thread count. Deliberately unquoted at the call sites: these are
# flag fragments, not one argument.
grid_extra() {
  local extra=""
  [ -n "${GRID_ENTRIES:-}" ] && extra="${extra} --grid-entries ${GRID_ENTRIES}"
  [ -n "${GRID_THREADS:-}" ] && extra="${extra} --threads ${GRID_THREADS}"
  printf '%s' "${extra}"
}

# The minimal AL2023 image ships curl but neither tar nor unzip, and the uv and
# rustup installers both need one of them to unpack their archives. Without
# these the install fails with no useful message at the end of the log.
dnf -y install git gcc gcc-c++ make openssl-devel tar unzip gzip >/dev/null 2>&1

curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
export PATH=/root/.cargo/bin:/root/.local/bin:$PATH

echo "=== host ==="
grep -m1 'model name' /proc/cpuinfo || true
echo "cpus: $(nproc)"
grep -E 'MemTotal|MemAvailable' /proc/meminfo || true

echo "=== build ==="
cd /tmp
git clone --quiet --branch "$GIT_REF" --single-branch https://github.com/monocongo/climate_indices.git ci
cd ci
echo "revision: $(git rev-parse HEAD) $(git log --oneline -1 --format=%s)"
uv sync >/dev/null 2>&1
uv run --no-sync maturin develop --release

echo "=== stage: $STAGE (repeats=$REPEATS, cpus=$CPUS) ==="
case "$STAGE" in
  routine)
    PYTHONWARNINGS=error taskset -c "$CPUS" uv run --no-sync python benchmarks/rust_vs_python.py \
      --repeat "$REPEATS" --output /tmp/routine.txt
    cat /tmp/routine.txt
    ;;
  spread)
    taskset -c "$CPUS" uv run --no-sync python benchmarks/aws/spread.py "$REPEATS"
    cat spread.json
    ;;
  grid)
    if [ -z "${NETCDF:-}" ] || [ -z "${TAVG:-}" ]; then
      echo "grid stage needs NETCDF and TAVG (fixtures are not in the repository)" >&2
      exit 2
    fi
    PYTHONWARNINGS=error taskset -c "$CPUS" uv run --no-sync python benchmarks/rust_vs_python.py \
      --netcdf "$NETCDF" --tavg "$TAVG" --repeat "$REPEATS" $(grid_extra) \
      --output /tmp/grid.txt > /tmp/harness.log 2>&1 || { tail -30 /tmp/harness.log; exit 1; }
    cat /tmp/grid.txt
    ;;
  grid_synthetic)
    rows="${GRID_ROWS:-596}"
    cols="${GRID_COLS:-1385}"
    uv run --no-sync python benchmarks/aws/make_synthetic_grid.py "$rows" "$cols" /tmp/synth
    PYTHONWARNINGS=error taskset -c "$CPUS" uv run --no-sync python benchmarks/rust_vs_python.py \
      --netcdf /tmp/synth/prcp.nc --tavg /tmp/synth/tavg.nc --repeat "$REPEATS" $(grid_extra) \
      --output /tmp/grid.txt > /tmp/harness.log 2>&1 || { tail -30 /tmp/harness.log; exit 1; }
    cat /tmp/grid.txt
    ;;
  percell)
    # scPDSI has no spatial-block path (ADR-0011), so it is measured per cell over a
    # sample. NETCDF and TAVG are the real prepared grid and its raw temperature.
    if [ -z "${NETCDF:-}" ] || [ -z "${TAVG:-}" ]; then
      echo "percell stage needs NETCDF and TAVG" >&2
      exit 2
    fi
    taskset -c "$CPUS" uv run --no-sync python benchmarks/aws/percell.py \
      "$NETCDF" "$TAVG" "${PERCELL_CELLS:-1000}" "${PERCELL_REPEATS:-2}" > /tmp/percell.log 2>&1 \
      || { tail -30 /tmp/percell.log; exit 1; }
    cat /tmp/percell.log
    cat percell.json
    ;;
  grid_real)
    # The real CONUS grid, straight from NOAA's public bucket. These period-of-record
    # objects are appended and reprocessed, so the committed fingerprint from an earlier
    # run does not describe what the bucket serves today: record this retrieval's own
    # headers and digests, as benchmarks/results/nclimgrid_fixture_provenance.txt does.
    base="https://noaa-nclimgrid-monthly-pds.s3.amazonaws.com"
    echo "=== retrieval, $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    for obj in nclimgrid_prcp.nc nclimgrid_tavg.nc; do
      curl -sI "$base/$obj" | tr -d '\r' | grep -iE '^(etag|last-modified|content-length)' | sed "s#^#${obj}: #"
      curl -sSL -o "/tmp/$obj" "$base/$obj"
    done
    echo "=== downloaded bytes ==="
    ls -l /tmp/nclimgrid_prcp.nc /tmp/nclimgrid_tavg.nc
    sha256sum /tmp/nclimgrid_prcp.nc /tmp/nclimgrid_tavg.nc

    # Trim to the calibration-relevant span and apply the documented land-mask and
    # zero-to-0.01mm treatment, so this is byte-identical to what every other
    # benchmark harness prepares for the same object.
    uv run --no-sync python benchmarks/cli_multiprocessing.py prepare \
      /tmp/nclimgrid_prcp.nc /tmp/nclimgrid_prcp_1981_2024.nc --start 1981 --end 2024
    sha256sum /tmp/nclimgrid_prcp_1981_2024.nc

    PYTHONWARNINGS=error taskset -c "$CPUS" uv run --no-sync python benchmarks/rust_vs_python.py \
      --netcdf /tmp/nclimgrid_prcp_1981_2024.nc --tavg /tmp/nclimgrid_tavg.nc \
      --repeat "$REPEATS" $(grid_extra) \
      --output /tmp/grid.txt > /tmp/harness.log 2>&1 || { tail -30 /tmp/harness.log; exit 1; }
    cat /tmp/grid.txt
    ;;
  *)
    echo "unknown STAGE=$STAGE (routine|spread|grid|grid_synthetic|grid_real|percell)" >&2
    exit 2
    ;;
esac

echo "STAGE_DONE $STAGE"
