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

# Preparation publishes to the repository's data/e2e, and the notebook honours
# CLIMATE_INDICES_E2E_DATA; pin both stages to that same root so an inherited
# override cannot make execution read a store other than the prepared one.
export CLIMATE_INDICES_E2E_DATA="$repo_root/data/e2e"

uv run --no-build --group dev scripts/prepare_e2e_inputs.py

scratch="$(mktemp -d)"
uv run --no-build --group dev jupyter nbconvert --execute --to notebook \
  --output-dir "$scratch" \
  notebooks/zarr_dask_spi_spei.ipynb
echo "Executed notebook written to ${scratch}/zarr_dask_spi_spei.ipynb"
