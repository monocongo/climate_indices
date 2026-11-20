"""Parse pdi.f outputs (fort.21 WPLM, 22 PDSI, 23 Z, 24 PHDI) and compare with nClimDiv and library fixtures.

usage: pdi_compare.py <repo> <rundir>
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # python -I adds neither the script dir nor cwd
from _safe_path import path_arg as _path  # noqa: E402

repo, run = _path(sys.argv[1]), _path(sys.argv[2])
divs = json.load(open(repo / "tests/fixture/nclimdiv/divisions.json"))
NAMES = {
    "pdsi": ("fort.22", "pdsi"),
    "zindex": ("fort.23", "zindex"),
    "phdi": ("fort.24", "phdi"),
    "pmdi": ("fort.21", "pmdi"),
}


def parse(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        did = int(line[:4])
        vals = [float(line[9 + 7 * i : 16 + 7 * i]) for i in range(12)]
        out.setdefault(did, []).extend(vals)
    return out


def stat(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[m] - b[m])
    return d


res = {}
for key, (fn, fx) in NAMES.items():
    got = parse(run / fn)
    nc = np.load(repo / f"tests/fixture/nclimdiv/{fx}.npy").astype(float)
    meds, w05, gt1, nmiss = [], [], 0, 0
    lmeds, lw05 = [], []
    for i, d in enumerate(divs):
        a = np.array(got[int(d)], dtype=float)
        a[a <= -99.0] = np.nan
        if a.size != 1536:
            raise SystemExit(f"{d}: {a.size} months, expected 1536")
        dd = stat(a, nc[i])
        meds.append(np.median(dd))
        w05.append((dd <= 0.05).sum())
        gt1 += int((dd > 1).sum())
        nmiss += int(np.isnan(a).sum())
        lib = np.load(repo / f"tests/fixture/palmer/{d}/{fx}.npy").astype(float)
        dl = stat(a, lib)
        lmeds.append(np.median(dl))
        lw05.append((dl <= 0.05).sum())
    n = 344 * 1536
    res[key] = {
        "vs_nclimdiv_median_of_median_absdiff": round(float(np.median(meds)), 4),
        "vs_nclimdiv_frac_within_0p05": round(sum(w05) / n, 4),
        "vs_nclimdiv_months_gt_1": gt1,
        "vs_library_median_of_median_absdiff": round(float(np.median(lmeds)), 4),
        "vs_library_frac_within_0p05": round(sum(lw05) / n, 4),
        "missing": nmiss,
    }
    if key == "pdsi":
        i0 = divs.index("0101")
        a = np.array(got[101])
        d0 = np.abs(a - nc[i0])
        res["pdsi_0101"] = {
            "median": round(float(np.median(d0)), 4),
            "frac_within_0p05": round(float((d0 <= 0.05).mean()), 4),
            "months_gt_1": int((d0 > 1).sum()),
        }
print(json.dumps(res, indent=1))
