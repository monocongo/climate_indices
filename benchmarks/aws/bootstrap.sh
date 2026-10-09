#!/bin/bash
# Runs inside a Fargate task: build the Rust extension, then run one benchmark stage.
#
# The extension is built here rather than baked into an image, so the task
# definition stays image-agnostic and a stage cannot silently measure a stale
# build of the crate.
#
# Environment:
#   STAGE     routine (default) | spread | grid
#   GIT_REF   branch or commit to benchmark (default: main)
#   REPEATS   repetitions per backend (default: 15)
#   CPUS      taskset CPU set for the measured process (default: 0-7)
#   NETCDF    prepared precipitation grid, grid stage only
#   TAVG      matching mean-temperature grid, grid stage only
set -euxo pipefail
export HOME=/root

STAGE="${STAGE:-routine}"
GIT_REF="${GIT_REF:-main}"
REPEATS="${REPEATS:-15}"
CPUS="${CPUS:-0-7}"

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
      --netcdf "$NETCDF" --tavg "$TAVG" --repeat "$REPEATS" --output /tmp/grid.txt
    cat /tmp/grid.txt
    ;;
  *)
    echo "unknown STAGE=$STAGE (routine|spread|grid)" >&2
    exit 2
    ;;
esac

echo "STAGE_DONE $STAGE"
