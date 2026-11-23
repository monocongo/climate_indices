"""xclim adapter: gamma SPI by the approximate closed-form estimator (REFIMPL-08, issue 1610).

Runs in its own locked environment, never the repository's:
``uv run --locked --project intercomparison/adapters/xclim python intercomparison/adapters/xclim/adapter.py
run <request.json> <out_dir>``. The call follows containers/linux-smoke/smoke_py.py and the
`spi-gamma-thom` configuration of method-matrix.json. `distribution` and `estimator` are reported from
the arguments passed; the other positions are fixed inside xclim 1.0.0 and are literals with their source.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import xclim
from xclim.compute import standardized_precipitation_index

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import contract  # noqa: E402

DIST, METHOD, FITKWARGS = "gamma", "APP", {"floc": 0}


def compute(request: dict[str, Any], inputs: dict[str, tuple[list[str], list[float]]]) -> dict[str, Any]:
    times, values = inputs["precip"]
    contract.require_consecutive_months(times)
    first, last = request["positions"]["calibration_period"].split("/")
    scale = int(request["positions"]["scale"])
    index = pd.date_range(f"{times[0]}-01", periods=len(times), freq="MS")
    # Monthly totals under a rate label so xclim does not rescale per day. Gamma SPI is invariant
    # to a constant factor, so the label and the input unit (inches) do not change the index.
    precip = xr.DataArray(np.asarray(values), dims="time", coords={"time": index}, attrs={"units": "mm/d"})
    spi = standardized_precipitation_index(
        precip,
        freq="MS",
        window=scale,
        dist=DIST,
        method=METHOD,
        fitkwargs=FITKWARGS,
        cal_start=f"{first}-01-01",
        cal_end=f"{last}-12-31",
    )
    thom = (METHOD, FITKWARGS.get("floc")) == ("APP", 0)  # APP with the location fixed at 0, per method-matrix.json
    return {
        "version": xclim.__version__,
        "effective_method": {
            # fixed in xclim: the window is a rolling mean (compute/stats.py), equal to the sum divided by `scale`
            "accumulation": "mean",
            "scale": scale,
            "distribution": DIST,
            "estimator": "thom_closed_form" if thom else f"{METHOD} {FITKWARGS}",
            # fixed in xclim: zero_inflated with the zero probability n0/n (compute/_agro.py, stats.py)
            "zero_handling": "mass_n0_over_n",
            # fixed in xclim: the fit groups by `time.month`
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
