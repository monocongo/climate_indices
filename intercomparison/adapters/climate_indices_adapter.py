"""climate_indices adapter: gamma SPI by the library's own default method (REFIMPL-08, issue 1610).

Runs in the repository environment: ``uv run python intercomparison/adapters/climate_indices_adapter.py
run <request.json> <out_dir>``. The effective method is reported from the arguments passed, so a
request that asks for something else surfaces as a warning in the driver.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contract  # noqa: E402


def compute(request: dict[str, Any], inputs: dict[str, tuple[list[str], list[float]]]) -> dict[str, Any]:
    import numpy as np

    import climate_indices
    from climate_indices import spi
    from climate_indices.compute import Periodicity
    from climate_indices.vocabulary import Distribution

    times, values = inputs["precip"]
    start_year = int(times[0][:4])
    first, last = (int(y) for y in request["positions"]["calibration_period"].split("/"))
    scale = int(request["positions"]["scale"])
    zero_handling = "classic"
    result = spi(
        np.asarray(values, dtype=np.float64),
        scale=scale,
        distribution=Distribution.gamma,
        data_start_year=start_year,
        calibration_year_initial=first,
        calibration_year_final=last,
        periodicity=Periodicity.monthly,
        zero_handling=zero_handling,
    )
    return {
        "version": climate_indices.__version__,
        # Names follow method-matrix.json aligned configuration `spi-gamma-thom`.
        "effective_method": {
            "accumulation": "sum",
            "scale": scale,
            "distribution": Distribution.gamma.value,
            "estimator": "thom_closed_form",
            "zero_handling": "mass_n0_over_n" if zero_handling == "classic" else zero_handling,
            "calendar_grouping": "calendar_month",
            "calibration_period": f"{first}/{last}",
        },
        "software": {"python": sys.version.split()[0], "numpy": np.__version__},
        "series": {"spi": (times, result.tolist())},
    }


if __name__ == "__main__":
    raise SystemExit(contract.run_adapter("climate_indices", compute, sys.argv[1:]))
