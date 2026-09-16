"""Unit checks for the parallel-scaling benchmark harness.

Covers the chunking and argument rules; the timing measurements themselves live
in ``benchmarks/parallel_scaling.py`` and are run manually.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import warnings
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import xarray as xr

from climate_indices.exceptions import GoodnessOfFitWarning

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(BENCHMARKS))
try:
    parallel_scaling = _load_module("parallel_scaling", BENCHMARKS / "parallel_scaling.py")
finally:
    sys.path.remove(str(BENCHMARKS))


@pytest.mark.parametrize("workers", [1, 2, 3, 4, 5, 8, 13, 16, 24, 48, 64, 110, 128, 256, 400, 1444, 2000, 3306])
def test_chunking_yields_a_block_per_worker(workers: int) -> None:
    """Every accepted worker count must map to at least that many spatial blocks."""
    grid = xr.DataArray(
        np.zeros((3, parallel_scaling.REFERENCE_LAT, parallel_scaling.REFERENCE_LON)),
        dims=("time", "lat", "lon"),
    )
    chunked = parallel_scaling._chunk_for_workers(grid, workers)
    spatial_chunks = chunked.chunks[1:]
    assert math.prod(len(axis_chunks) for axis_chunks in spatial_chunks) >= workers
    assert sum(spatial_chunks[0]) == parallel_scaling.REFERENCE_LAT
    assert sum(spatial_chunks[1]) == parallel_scaling.REFERENCE_LON


def test_worker_counts_reject_counts_outside_the_grid() -> None:
    """Non-positive and above-cell worker counts fail argument parsing."""
    with pytest.raises(argparse.ArgumentTypeError):
        parallel_scaling._worker_counts("0,2")
    cells = parallel_scaling.REFERENCE_LAT * parallel_scaling.REFERENCE_LON
    with pytest.raises(argparse.ArgumentTypeError):
        parallel_scaling._worker_counts(str(cells + 1))
    assert parallel_scaling._worker_counts("1,4") == (1, 4)


def test_quiet_worker_silences_only_goodness_of_fit() -> None:
    """The worker-side filter must leave unrelated warning categories visible."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parallel_scaling._quiet_worker()
        warnings.warn("fit", GoodnessOfFitWarning, stacklevel=2)
        warnings.warn("other", RuntimeWarning, stacklevel=2)
    assert [type(warning.message) for warning in caught] == [RuntimeWarning]
