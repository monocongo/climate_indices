# PV power/temperature reference

The fixture contains nine synthetic cases from `pvlib.pvsystem.pvwatts_dc`
0.15.0, at commit `247b31f71fc609e20e72ae6c1da186f9b5b45df4`.
`reference.json` records the input arrays, parameters, dependency versions, and
unmodified outputs. `provenance.json` contains its SHA-256 checksum.

This independent implementation checks only the DC power/temperature term.
The test supplies cell temperature through the inverse Jerez thermal equation.
Separate hand-calculated tests check the thermal coefficients. Neither check
validates actual plant generation or the C3S PV configuration.

The absolute tolerance is `1e-12` in dimensionless capacity factor, with
`rtol=0`. It permits float64 arithmetic differences, not model differences.
The raw output includes 1.47. Clipping to [0, 1] is a separate project policy.

Run this comparison from the repository root. The temporary environment does
not add pvlib to the project dependencies:

```bash
uv run --no-project --with pvlib==0.15.0 python - <<'PY'
import json
from pathlib import Path
import numpy as np
import pvlib

path = Path("tests/fixture/energy_solar/reference.json")
reference = json.loads(path.read_text())
assert pvlib.__version__ == reference["pvlib_version"]
result = pvlib.pvsystem.pvwatts_dc(
    np.array(reference["irradiance_w_m2"]),
    np.array(reference["cell_temperature_c"]),
    **reference["parameters"],
)
np.testing.assert_allclose(result, reference["pvwatts_dc"], rtol=0, atol=1e-12)
print(result.tolist())
PY
```

The source software uses BSD-3-Clause. The inputs are synthetic, and the
fixture copies no software implementation or observational data.

References:

- https://github.com/pvlib/pvlib-python/tree/v0.15.0
- https://pvlib-python.readthedocs.io/en/v0.15.0/reference/generated/pvlib.pvsystem.pvwatts_dc.html
- Jerez et al. (2015), Methods equations 1–3: https://doi.org/10.1038/ncomms10014
