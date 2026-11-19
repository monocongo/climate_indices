"""Summarize the JonKing93/pdsi supplied-PE run against nClimDiv and the library fixtures.

usage: summarize.py <repo> <outdir>      (outdir = the run.sh work-dir's out/)
Comparison is REFIMPL-31's job; this only reproduces the smoke numbers quoted in palmer-comparators.md.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np


def _path(value: str) -> Path:
    """Resolve a command-line path, refusing a relative one that escapes the working directory.

    Same guard as benchmarks/cli_multiprocessing.py (pythonsecurity:S8707); absolute paths are
    the documented usage and pass through after normalisation.
    """
    resolved = os.path.realpath(value)
    base = os.path.realpath(os.getcwd())
    if not os.path.isabs(value) and resolved != base and not resolved.startswith(base + os.sep):
        raise SystemExit(f"{value!r} is outside the working directory")
    return Path(resolved)


repo, out = _path(sys.argv[1]), _path(sys.argv[2])
divs = json.load(open(repo / "tests/fixture/nclimdiv/divisions.json"))
runs = {"pdsi": "supplied_pe_X.txt", "zindex": "supplied_pe_Z.txt", "pmdi": "supplied_pe_Xm.txt"}
for key, fn in runs.items():
    arr = np.loadtxt(out / fn)
    nc = np.load(repo / f"tests/fixture/nclimdiv/{key}.npy").astype(float)
    lib = np.stack([np.load(repo / f"tests/fixture/palmer/{d}/{key}.npy").astype(float) for d in divs])
    for label, ref in (("nClimDiv", nc), ("library", lib)):
        d = np.abs(arr - ref)
        print(
            f"{key:7s} vs {label:8s}: median of per-division medians {np.median(np.median(d, 1)):.4f}, "
            f"within 0.05 {(d <= 0.05).mean():.3f}, months > 1: {int((d > 1).sum())} of {d.size}"
        )
i0 = divs.index("0101")
z = np.abs(
    np.loadtxt(out / "supplied_pe_Z.txt")[i0] - np.load(repo / "tests/fixture/nclimdiv/zindex.npy")[i0].astype(float)
)
print(f"division 0101 Z vs nClimDiv: median {np.median(z):.4f} (pre-registered threshold 0.05)")
pub = np.loadtxt(out / "published_0101_Z.txt")
print(f"as-published 0101: {int(np.isfinite(pub).sum())} of {pub.size} finite")
