"""Run ktripa/scpdsi-tools scpdsi() on the 344 fixture divisions and summarize against the Wells oracle.

usage: run_all.py <upstream-checkout> <repo> <outdir>

Inputs are the committed monthly precipitation and PET (inches), per-division AWC (inches), and the
1931-1990 calibration window, so the run is like-for-like with tests/fixture/palmer/<div>/sc*.npy.
Writes (344, 1536) text matrices; comparison is REFIMPL-32/33's job, the printout only reproduces
the smoke numbers quoted in palmer-comparators.md.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # python -I adds neither the script dir nor cwd
from _safe_path import path_arg as _path  # noqa: E402

sys.path.insert(0, str(_path(sys.argv[1])))  # -I mode does not add the script or cwd to sys.path
import scpdsi_tools  # noqa: E402

repo, out = _path(sys.argv[2]), _path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
divs = json.load(open(repo / "tests/fixture/nclimdiv/divisions.json"))
awc = json.load(open(repo / "tests/fixture/palmer_awc.json"))
fx = repo / "tests/fixture/palmer"
dates = pd.date_range("1895-01-01", periods=1536, freq="MS")
cols = {"pdsi": "scpdsi", "zindex": "sczindex", "phdi": "scphdi", "pmdi": "scpmdi"}
got = {c: [] for c in cols}
for d in divs:
    p = np.load(fx / d / "precips.npy").astype(float)
    e = np.load(fx / d / "pet.npy").astype(float)
    r = scpdsi_tools.scpdsi(
        p, e, dates, awc[d], "monthly", calibration_start="1931-01-01", calibration_end="1990-12-31", units="inch"
    )
    for c in cols:
        got[c].append(r[c].to_numpy())
for c in cols:
    np.savetxt(out / f"{c}.txt", np.stack(got[c]), fmt="%.10g")

for c, ref in cols.items():
    a = np.stack(got[c])
    oracle = np.stack([np.load(fx / d / f"{ref}.npy").astype(float) for d in divs])
    d = np.abs(a - oracle)
    print(
        f"{ref:9s} finite {int(np.isfinite(a).sum())} of {a.size}; median of per-division medians "
        f"{np.nanmedian(np.nanmedian(d, 1)):.4f}, within 5e-5 {np.nanmean(d <= 5e-5):.3f}, "
        f"within 0.05 {np.nanmean(d <= 0.05):.3f}, months > 1: {int(np.nansum(d > 1))}"
    )
