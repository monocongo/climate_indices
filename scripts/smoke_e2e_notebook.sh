#!/usr/bin/env bash
# Smoke-execute notebooks/zarr_dask_spi_spei.ipynb from a fresh kernel.
#
# Prepares the pinned sample inputs (cached under data/e2e/source after the
# first run, checksum-verified on every use), then executes every cell.
# Results go to a scratch directory so a local run never rewrites the
# committed notebook.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

uv run --no-build --with h5py --with zarr scripts/prepare_e2e_inputs.py

scratch="$(mktemp -d)"
uv run --no-build jupyter nbconvert --execute --to notebook \
  --output-dir "$scratch" \
  notebooks/zarr_dask_spi_spei.ipynb
echo "Executed notebook written to ${scratch}/zarr_dask_spi_spei.ipynb"
