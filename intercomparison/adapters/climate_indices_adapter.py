"""climate_indices adapter: gamma SPI by the library's own default method (REFIMPL-08, issue 1610).

Runs in the repository environment: ``uv run python intercomparison/adapters/climate_indices_adapter.py
run <request.json> <out_dir>``. Positions that are arguments of the call (`scale`, `distribution`,
`calibration_period`, `zero_handling`, the periodicity) are reported from the values passed. The
others are fixed inside the library, not arguments, so they are literals with the source they come from.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

import climate_indices
from climate_indices import spi
from climate_indices.compute import Periodicity
from climate_indices.vocabulary import Distribution

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contract  # noqa: E402


def compute(request: dict[str, Any], inputs: dict[str, tuple[list[str], list[float]]]) -> dict[str, Any]:
    times, values = inputs["precip"]
    # the library takes the first value as January of `data_start_year`
    contract.require_consecutive_months(times, january_start=True)
    first, last = (int(y) for y in request["positions"]["calibration_period"].split("/"))
    scale = int(request["positions"]["scale"])
    distribution, periodicity, zero_handling = Distribution.gamma, Periodicity.monthly, "classic"
    result = spi(
        np.asarray(values, dtype=np.float64),
        scale=scale,
        distribution=distribution,
        data_start_year=int(times[0][:4]),
        calibration_year_initial=first,
        calibration_year_final=last,
        periodicity=periodicity,
        zero_handling=zero_handling,
    )
    return {
        "version": climate_indices.__version__,
        # Names follow method-matrix.json aligned configuration `spi-gamma-thom`.
        "effective_method": {
            # fixed in the library: the pipeline sums over `scale` (drought/indices.py _standardized_index_pipeline)
            "accumulation": "sum",
            "scale": scale,
            "distribution": distribution.value,
            # fixed in the library: the gamma fit is Thom's closed form (compute.py gamma_parameters)
            "estimator": "thom_closed_form",
            "zero_handling": "mass_n0_over_n" if zero_handling == "classic" else zero_handling,
            "calendar_grouping": "calendar_month" if periodicity is Periodicity.monthly else periodicity.value,
            "calibration_period": f"{first}/{last}",
        },
        "software": {"python": sys.version.split()[0], "numpy": np.__version__},
        "series": {"spi": (times, result.tolist())},
    }


if __name__ == "__main__":
    raise SystemExit(contract.run_adapter("climate_indices", compute, sys.argv[1:]))
