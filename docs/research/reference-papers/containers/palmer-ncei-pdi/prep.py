"""Write pdi.f inputs (fort.10 temp, fort.11 precip) for all fixture divisions, and check which
soilcnst.asc row pdi.f will pick for each (first row whose i5 id equals the division id,
Fortran blank-null rule), against tests/fixture/palmer_awc.json.

usage: pdi_prep.py <repo> <soilcnst.asc> <outdir>
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # python -I adds neither the script dir nor cwd
from _safe_path import path_arg as _path  # noqa: E402

repo, soil, out = _path(sys.argv[1]), _path(sys.argv[2]), _path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
divs = json.load(open(repo / "tests/fixture/nclimdiv/divisions.json"))
awc = json.load(open(repo / "tests/fixture/palmer_awc.json"))
YEARS = range(1895, 2023)

with open(out / "fort.10", "w") as ft, open(out / "fort.11", "w") as fp:
    for d in divs:
        base = repo / "tests/fixture/palmer" / d
        # legacy object dtype mixing floats and strings; same parse as tests/test_speibase_reference.py
        raw = np.load(base / "temps.npy", allow_pickle=True)
        t = np.array([float(str(v).split()[0]) for v in raw], dtype=float).reshape(128, 12)
        p = np.load(base / "precips.npy").astype(float).reshape(128, 12)
        for i, y in enumerate(YEARS):
            ft.write(f"{int(d):4d} {y:4d}" + "".join(f"{v:7.2f}" for v in t[i]) + "\n")
            fp.write(f"{int(d):4d} {y:4d}" + "".join(f"{v:7.2f}" for v in p[i]) + "\n")

rows = []
for line in soil.read_text().splitlines():
    idtxt = line[:5].replace(" ", "")  # gfortran default BLANK='NULL'
    rows.append((int(idtxt) if idtxt else 0, float(line[5:13]), line))
bad = []
for d in divs:
    first = next(r for r in rows if r[0] == int(d))
    if abs(first[1] + 1.0 - awc[d]) > 1e-3:
        bad.append((d, first[1] + 1.0, awc[d], first[2]))
print(f"{len(divs)} divisions written; first-match AWC differs from palmer_awc.json for {len(bad)}")
for b in bad:
    print("  ", b)
if bad:
    sys.exit(1)
