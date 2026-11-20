#!/usr/bin/env bash
# REFIMPL-30: run JonKing93/pdsi (MATLAB, MIT) under GNU Octave on the 344 fixture divisions.
# Upstream is cloned at run time at a pinned commit and mounted read-only; the container has no network.
#
# usage: run.sh <repo-root> <work-dir>      needs: git, docker, uv
set -euo pipefail
REPO=$(cd "$1" && pwd)
WORK=$(mkdir -p "$2" && cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
UPSTREAM_SHA=9a636117577f476a93f206897fb91dc94d4eddc2   # JonKing93/pdsi main, 2021-03-11
# gnuoctave/octave 9.2.0 publishes amd64 only, so Apple silicon runs it under emulation.
IMAGE=gnuoctave/octave@sha256:100d394c57b86469748d26ddafcf73a1074338a66dabcababe2e4e05146772b9

mkdir -p "$WORK/data" "$WORK/out"
[[ -d "$WORK/upstream/.git" ]] || git clone -q https://github.com/JonKing93/pdsi.git "$WORK/upstream"
git -C "$WORK/upstream" reset --hard -q "$UPSTREAM_SHA"
git -C "$WORK/upstream" clean -fdx -q
uv run --project "$REPO" python -I "$HERE/prep.py" "$REPO" "$WORK/data"
docker run --rm --platform linux/amd64 --network none \
  -v "$WORK/upstream:/work/upstream:ro" -v "$WORK/data:/work/data:ro" -v "$WORK/out:/work/out" \
  -v "$HERE/jk_run_all.m:/work/jk_run_all.m:ro" \
  "$IMAGE" octave-cli --no-gui /work/jk_run_all.m
