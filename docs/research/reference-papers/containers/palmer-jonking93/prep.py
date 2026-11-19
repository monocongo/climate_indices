"""Write JonKing93/pdsi inputs for the 344 fixture divisions as plain-text matrices.

usage: prep.py <repo> <outdir>

precip_in.txt, pet_in.txt: (344, 1536) inches, row order = divisions.json.
awc.txt: (344, 1) total available water capacity, inches (tests/fixture/palmer_awc.json).
temp_c_0101.txt: (1536, 1) degrees C for division 0101, for the as-published run only.
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
out.mkdir(parents=True, exist_ok=True)
divs = json.load(open(repo / "tests/fixture/nclimdiv/divisions.json"))
awc = json.load(open(repo / "tests/fixture/palmer_awc.json"))
fx = repo / "tests/fixture/palmer"
np.savetxt(out / "precip_in.txt", np.stack([np.load(fx / d / "precips.npy").astype(float) for d in divs]), fmt="%.10g")
np.savetxt(out / "pet_in.txt", np.stack([np.load(fx / d / "pet.npy").astype(float) for d in divs]), fmt="%.10g")
np.savetxt(out / "awc.txt", np.array([awc[d] for d in divs]), fmt="%.10g")
# legacy object dtype mixing floats and strings; same parse as tests/test_speibase_reference.py
raw = np.load(fx / "0101" / "temps.npy", allow_pickle=True)
temp_f = np.array([float(str(v).split()[0]) for v in raw], dtype=float)
np.savetxt(out / "temp_c_0101.txt", (temp_f - 32.0) * 5.0 / 9.0, fmt="%.10g")
print(f"{len(divs)} divisions written to {out}")
