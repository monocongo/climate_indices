"""xclim adapter: gamma SPI by the approximate closed-form estimator (REFIMPL-08, issue 1610).

Runs in its own locked environment, never the repository's:
``uv run --locked --project intercomparison/adapters/xclim python intercomparison/adapters/xclim/adapter.py
run <request.json> <out_dir>``. The call follows containers/linux-smoke/smoke_py.py and the
`spi-gamma-thom` configuration of method-matrix.json.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import contract  # noqa: E402


def compute(request: dict[str, Any], inputs: dict[str, tuple[list[str], list[float]]]) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import xarray as xr
    import xclim
    from xclim.indices import standardized_precipitation_index

    times, values = inputs["precip"]
    first, last = request["positions"]["calibration_period"].split("/")
    scale = int(request["positions"]["scale"])
    index = pd.date_range(f"{times[0]}-01", periods=len(times), freq="MS")
    if [t.strftime("%Y-%m") for t in index] != times:
        raise contract.ContractError("times are not consecutive calendar months")
    # Monthly totals under a rate label so xclim does not rescale per day. Gamma SPI is invariant
    # to a constant factor, so the label and the input unit (inches) do not change the index.
    precip = xr.DataArray(np.asarray(values), dims="time", coords={"time": index}, attrs={"units": "mm/d"})
    spi = standardized_precipitation_index(
        precip,
        freq="MS",
        window=scale,
        dist="gamma",
        method="APP",
        fitkwargs={"floc": 0},
        cal_start=f"{first}-01-01",
        cal_end=f"{last}-12-31",
    )
    return {
        "version": xclim.__version__,
        "effective_method": {
            "accumulation": "sum",
            "scale": scale,
            "distribution": "gamma",
            "estimator": "thom_closed_form",  # method='APP' with floc=0, per method-matrix.json
            "zero_handling": "mass_n0_over_n",
            "calendar_grouping": "calendar_month",
            "calibration_period": f"{first}/{last}",
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xarray": xr.__version__,
        },
        "series": {"spi": (times, [float(v) for v in spi.to_numpy()])},
    }


if __name__ == "__main__":
    raise SystemExit(contract.run_adapter("xclim", compute, sys.argv[1:]))
