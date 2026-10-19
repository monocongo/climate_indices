"""Measure per-tile peak memory of the compound xarray routes against the ADR-0022 planning budget.

For ``k`` sources, ``T`` times, ``C`` cells, ``B`` realizations, and ``q`` stored metrics the design budgets
``(10*k + 24)*T*C + 8*B*q*C`` bytes, excluding scheduler overhead, and states that it is not a measured limit.
This script measures it. Each stage runs one single-chunk tile on the synchronous scheduler, so one task's
peak is isolated from scheduling, and ``tracemalloc`` records the peak of every NumPy allocation made while
the stage computes. The inputs are built beforehand, so their bytes are added to the traced peak.

``tracemalloc`` sees Python and NumPy allocations, not interpreter, library, or allocator overhead, so the
report is a lower bound on a worker's resident memory. A Dask worker runs several tasks at once and holds
their peaks together; divide a worker's memory limit by its concurrent tasks before choosing a tile.

Run: uv run scripts/measure_compound_xarray_memory.py --output docs/design/compound-xarray-memory.json
The output must be a JSON file below this repository's docs/design directory. ``--quick`` runs a reduced
configuration for a smoke test; its numbers are not the published ones.
"""

from __future__ import annotations

import argparse
import gc
import logging
import platform
import sys
import tracemalloc
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dask
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from characterize_compound_bootstrap import _output_path, _write_report  # noqa: E402

from climate_indices import compound  # noqa: E402
from climate_indices.exceptions import ClimateIndicesWarning  # noqa: E402

SOURCES = 3
DAILY_DAYS = 36_524  # the mean length of 100 Gregorian years, floored; the series ends 2019-12-30
DAILY_CELLS = (4, 16, 64)
MONTHLY_YEARS = 100
BOOTSTRAP_CELLS = (2, 16, 64)
RESAMPLES = 1999
STORED_METRICS = 1
LAG_WINDOW = (0, 30)
TASK_BUDGETS = {"256 MiB": 256 * 2**20, "1 GiB": 2**30}
NAMES = tuple(f"h{i}" for i in range(SOURCES))


def _budget(steps: int, cells: int, *, resamples: int = 0) -> int:
    """Return the design's planning budget in bytes for one tile."""
    return (10 * SOURCES + 24) * steps * cells + 8 * resamples * STORED_METRICS * cells


def _declarations(cadence: str) -> dict[str, Any]:
    return {
        "cadence": dict.fromkeys(NAMES, cadence),
        "input_scale": dict.fromkeys(NAMES, "probability"),
        "thresholds": dict.fromkeys(NAMES, 0.3),
        "directions": dict.fromkeys(NAMES, "below"),
        "target_cadence": cadence,
        "source_metadata": {name: {"calibration_year_initial": 1991, "calibration_year_final": 2020} for name in NAMES},
    }


def _sources(time: np.ndarray, cells: int, seed: int) -> dict[str, xr.DataArray]:
    """Return seeded float64 probability sources as single-chunk Dask arrays on one cell dimension."""
    rng = np.random.default_rng(seed)
    coords = {"time": time.astype("datetime64[ns]"), "cell": np.arange(cells)}
    return {
        name: xr.DataArray(rng.uniform(size=(len(time), cells)), dims=("time", "cell"), coords=coords).chunk(-1)
        for name in NAMES
    }


def _peak(stage: Callable[[], Any], input_bytes: int) -> int:
    """Return the traced peak of one stage, plus the bytes of inputs that existed before tracing began.

    One untraced run first, so lazy imports and caches that a first call fills are not charged to the tile.
    """
    with dask.config.set(scheduler="synchronous"):
        stage()
    gc.collect()
    tracemalloc.start()
    try:
        with dask.config.set(scheduler="synchronous"):
            result = stage()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    del result
    return peak + input_bytes


def _measure(
    name: str,
    cells: tuple[int, ...],
    steps: int,
    time: np.ndarray,
    make_stage: Callable[[xr.Dataset | dict[str, xr.DataArray]], Callable[[], Any]],
    *,
    cadence: str,
    prepared: bool,
    resamples: int = 0,
) -> dict[str, Any]:
    """Measure one stage at several tile widths; report a fixed-plus-per-cell fit and the budget ratio."""
    peaks, inputs, budgets = [], [], []
    for width in cells:
        data = _sources(time, width, seed=width)
        arrays: Any = data
        if prepared:
            arrays = compound.harmonize(data, **_declarations(cadence)).compute()
            arrays = arrays.chunk({"hazard": -1, "time": -1})
        del data  # Prepared stages no longer need the float payloads retained by the source graphs.
        input_bytes = sum(
            array.nbytes for array in (arrays.values() if isinstance(arrays, dict) else arrays.data_vars.values())
        )
        peaks.append(_peak(make_stage(arrays), input_bytes))
        inputs.append(int(input_bytes))
        budgets.append(_budget(steps, width, resamples=resamples))
    # a tile's peak is a fixed per-task cost plus a per-cell cost, so fit both; the bootstrap reuses one
    # cell's realization arrays for every cell, which makes its slope far smaller than its intercept
    slope, intercept = (float(value) for value in np.polyfit(cells, peaks, 1))
    slope = max(slope, 1.0)
    return {
        "stage": name,
        "cadence": cadence,
        "sources": SOURCES,
        "steps": steps,
        "resamples": resamples,
        "cells": list(cells),
        "input_bytes": inputs,
        "peak_bytes": peaks,
        "budget_bytes": budgets,
        "peak_over_budget": [round(peak / budget, 3) for peak, budget in zip(peaks, budgets, strict=True)],
        "fixed_bytes_per_task": round(max(intercept, 0.0)),
        "bytes_per_cell": round(slope),
        "budget_bytes_per_cell": budgets[-1] // cells[-1],
        "bytes_per_cell_over_budget": round(slope / (budgets[-1] / cells[-1]), 3),
        "max_cells_per_task": {
            label: max(int((limit - max(intercept, 0.0)) // slope), 0) for label, limit in TASK_BUDGETS.items()
        },
    }


def measure(*, quick: bool = False) -> dict[str, Any]:
    """Return the full measurement report; ``quick`` shrinks every dimension for a smoke test."""
    with warnings.catch_warnings():
        # the measured tiles are deliberately short or sparse, and every call warns that the route is beta
        warnings.simplefilter("ignore", ClimateIndicesWarning)
        return _measure_all(quick)


def _measure_all(quick: bool) -> dict[str, Any]:
    """Run every stage; see :func:`measure`."""
    days = np.arange("1920-01-01", "2020-01-01", dtype="datetime64[D]")[: 1_000 if quick else DAILY_DAYS]
    months = np.arange("1920-01", f"{1920 + (30 if quick else MONTHLY_YEARS)}-01", dtype="datetime64[M]")
    daily_cells = (2, 4) if quick else DAILY_CELLS
    boot_cells = (1, 2) if quick else BOOTSTRAP_CELLS
    resamples = 19 if quick else RESAMPLES
    daily, monthly = _declarations("daily"), _declarations("monthly")
    for declared in (daily, monthly):
        # the harmonization warns about nothing here; calibration metadata is declared
        declared["join"] = "exact"

    stages = [
        _measure(
            "harmonize",
            daily_cells,
            len(days),
            days,
            lambda arrays: lambda: compound.harmonize(arrays, **daily).compute(),
            cadence="daily",
            prepared=False,
        ),
        _measure(
            "joint_statistics",
            daily_cells,
            len(days),
            days,
            lambda prepared: lambda: compound.joint_statistics(prepared, min_samples=30).compute(),
            cadence="daily",
            prepared=True,
        ),
        _measure(
            "conditional_probability",
            daily_cells,
            len(days),
            days,
            lambda prepared: (
                lambda: compound.conditional_probability(
                    prepared, given=NAMES[0], target=NAMES[1], lag_window=LAG_WINDOW, min_samples=30
                ).compute()
            ),
            cadence="daily",
            prepared=True,
        ),
        _measure(
            "bootstrap_dependence",
            boot_cells,
            len(months),
            months,
            lambda prepared: (
                lambda: compound.bootstrap_dependence(
                    prepared, statistic="joint_probability", seed=1, n_resamples=resamples
                ).compute()
            ),
            cadence="monthly",
            prepared=True,
            resamples=resamples,
        ),
    ]
    return {
        "report": "CMP-08 compound xarray per-tile peak memory",
        "quick": quick,
        "budget_formula": "(10*k + 24)*T*C + 8*B*q*C bytes; k sources, T times, C cells, B realizations, q stored metrics",
        "method": (
            "One single-chunk tile per stage on the synchronous scheduler; tracemalloc peak of all NumPy "
            "allocations while the stage computes, plus the bytes of the inputs built beforehand. Excludes "
            "interpreter and allocator overhead, so it is a lower bound on resident memory."
        ),
        "stored_metrics": STORED_METRICS,
        "conditional_lag_window": list(LAG_WINDOW),
        "stages": stages,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "xarray": xr.__version__,
            "dask": dask.__version__,
        },
    }


def main() -> int:
    """Measure and write the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=_output_path, required=True)
    parser.add_argument("--quick", action="store_true", help="reduced configuration for a smoke test")
    arguments = parser.parse_args()
    logging.disable(logging.CRITICAL)
    report = measure(quick=arguments.quick)
    _write_report(arguments.output, report)
    for stage in report["stages"]:
        print(
            f"{stage['stage']:<26} fixed {stage['fixed_bytes_per_task'] / 2**20:7.2f} MiB/task  "
            f"{stage['bytes_per_cell'] / 2**20:7.3f} MiB/cell ({stage['bytes_per_cell_over_budget']}x budget)  "
            f"cells/task {stage['max_cells_per_task']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
