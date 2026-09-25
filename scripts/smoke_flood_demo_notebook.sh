#!/usr/bin/env bash
# Smoke-execute notebooks/flood_event_brisbane_2011.ipynb from a fresh kernel.
#
# Local use only: the notebook needs the network (NOAA PSL), which is why no CI
# job runs it (#917).
#
# Downloads the yearly NOAA PSL CPC subsets on the first run, then caches them
# under the git-ignored data/flood-demo/. Results go to a scratch directory so a
# local run never rewrites the committed notebook.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

scratch="$(mktemp -d)"
# No --no-build here: local interpreters past the newest cartopy wheel (cp313)
# would otherwise be unable to resolve the dev group.
uv run --group dev jupyter nbconvert --execute --to notebook \
  --output-dir "$scratch" \
  notebooks/flood_event_brisbane_2011.ipynb
echo "Executed notebook written to ${scratch}/flood_event_brisbane_2011.ipynb"
