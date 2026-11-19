#!/usr/bin/env bash
# REFIMPL-30: run ktripa/scpdsi-tools (Python, MIT) on the 344 fixture divisions.
# Upstream is cloned at run time at a pinned commit; nothing from it is committed.
#
# usage: run.sh <repo-root> <work-dir>      needs: git, uv; the repo environment supplies numpy and pandas
set -euo pipefail
REPO=$(cd "$1" && pwd)
WORK=$(mkdir -p "$2" && cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
UPSTREAM_SHA=c73b22c89218236e766fbf2895f2a16ebd8f366e   # ktripa/scpdsi-tools main, 2026-10-21

mkdir -p "$WORK"
[[ -d "$WORK/upstream/.git" ]] || git clone -q https://github.com/ktripa/scpdsi-tools.git "$WORK/upstream"
git -C "$WORK/upstream" checkout -q "$UPSTREAM_SHA"
uv run --project "$REPO" python -I "$HERE/run_all.py" "$WORK/upstream" "$REPO" "$WORK/out"
