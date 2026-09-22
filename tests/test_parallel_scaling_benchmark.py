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
import pandas as pd
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


# left on sys.path for the test session: _measure's processes scheduler pickles worker
# tasks by module name ("parallel_scaling"), and a spawned child re-imports that name
# fresh, so it needs the same path the parent used to load it. Appended, not
# prepended, so a benchmarks/ module can never shadow a site-packages import.
sys.path.append(str(BENCHMARKS))
parallel_scaling = _load_module("parallel_scaling", BENCHMARKS / "parallel_scaling.py")


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
    """Non-positive counts fail parsing; counts above the cells fail validation and exit cleanly."""
    with pytest.raises(argparse.ArgumentTypeError):
        parallel_scaling._worker_counts("0,2")
    cells = parallel_scaling.REFERENCE_LAT * parallel_scaling.REFERENCE_LON
    with pytest.raises(argparse.ArgumentTypeError):
        parallel_scaling._validate_worker_counts((cells + 1,), cells)
    parallel_scaling._validate_worker_counts((1, cells), cells)
    parallel_scaling._require_worker_counts((1, cells), cells)
    with pytest.raises(SystemExit, match="must not exceed"):
        parallel_scaling._require_worker_counts((cells + 1,), cells)
    assert parallel_scaling._worker_counts("1,4") == (1, 4)


def test_netcdf_mode_rejects_arguments_it_cannot_honour(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real-grid mode is SPI-only, its options require --netcdf, and --scale must be at least 1."""
    monkeypatch.setattr(sys, "argv", ["parallel_scaling.py", "--netcdf", "x.nc", "--indices", "spi,spei"])
    with pytest.raises(SystemExit):
        parallel_scaling._parse_args()
    for ignored in (["--scale", "6"], ["--var-name", "precip"], ["--write-output", "out.nc"]):
        monkeypatch.setattr(sys, "argv", ["parallel_scaling.py", *ignored])
        with pytest.raises(SystemExit):
            parallel_scaling._parse_args()
    monkeypatch.setattr(sys, "argv", ["parallel_scaling.py", "--netcdf", "x.nc", "--scale", "0"])
    with pytest.raises(SystemExit):
        parallel_scaling._parse_args()
    monkeypatch.setattr(sys, "argv", ["parallel_scaling.py", "--netcdf", "x.nc", "--var-name", ""])
    with pytest.raises(SystemExit):
        parallel_scaling._parse_args()


def _write_grid_fixture(path: Path) -> None:
    """Write a small lat-lon-time precipitation file shaped like the CHIRPS fixture.

    The first latitude row is all NaN (the ocean); the first longitude cell of the
    second row is NaN too, so the first land cell is at lat index 1, lon index 1 --
    pinning a roll on both spatial axes rather than latitude alone. One land cell
    has a zero month, and the dimension order matches the fixture (lat, lon, time)
    to pin the transpose.
    """
    values = np.full((3, 3, 4), np.nan)
    values[1:, :, :] = 2.0
    values[1, 0, :] = np.nan
    values[1, 1, 1] = 0.0
    xr.Dataset(
        {"precip": (("lat", "lon", "time"), values)},
        coords={
            "lat": [10.0, 20.0, 30.0],
            "lon": [1.0, 2.0, 3.0],
            "time": pd.date_range("2000-01-01", periods=4, freq="MS"),
        },
    ).to_netcdf(path, engine="h5netcdf")


def test_load_netcdf_grid_masks_zeros_and_rolls_the_sampled_cell(tmp_path: Path) -> None:
    """The real-grid loader preserves the ocean mask, replaces zeros, and starts on a land cell."""
    path = tmp_path / "fixture.nc"
    _write_grid_fixture(path)

    grid = parallel_scaling.load_netcdf_grid(str(path), "precip")
    values = grid.precip.values
    valid = grid.valid_cells
    assert valid is not None

    # time first, the zero month replaced, both ocean cells still NaN and rolled last
    assert grid.precip.dims == ("time", "lat", "lon")
    expected = np.full(values.shape, 2.0)
    expected[:, 0, 2] = np.nan
    expected[:, 2, :] = np.nan
    expected[1, 0, 0] = 0.01
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(valid[0], [True, True, False])
    np.testing.assert_array_equal(valid[1], np.ones(3, dtype=bool))
    np.testing.assert_array_equal(valid[2], np.zeros(3, dtype=bool))

    # rolled by one cell on both axes so cell [0, 0] is a land cell, with its label following it
    assert grid.precip.attrs["roll_lat"] == -1
    assert grid.precip.attrs["roll_lon"] == -1
    np.testing.assert_array_equal(grid.precip.lat.values, [20.0, 30.0, 10.0])
    np.testing.assert_array_equal(grid.precip.lon.values, [2.0, 3.0, 1.0])


def test_write_output_restores_the_input_coordinate_order(tmp_path: Path) -> None:
    """``load_netcdf_grid`` rolls the grid; the written NetCDF must not carry that roll."""
    source = tmp_path / "fixture.nc"
    _write_grid_fixture(source)
    grid = parallel_scaling.load_netcdf_grid(str(source), "precip")
    index = parallel_scaling._Index(lambda g: g.precip, 0)

    out_path = tmp_path / "output.nc"
    parallel_scaling._write_output(index, grid, str(out_path))

    with xr.open_dataarray(out_path) as written:
        np.testing.assert_array_equal(written.lat.values, [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(written.lon.values, [1.0, 2.0, 3.0])
        assert bool(np.isnan(written.sel(lat=10.0).values).all())
        assert bool(np.isnan(written.sel(lat=20.0, lon=1.0).isel(time=0).values))
        np.testing.assert_allclose(written.sel(lat=20.0, lon=2.0).values, [2.0, 0.01, 2.0, 2.0])
        assert written.name == "spi"
        # the in-memory roll markers must not survive into the un-rolled file
        assert {"roll_lat", "roll_lon", "land_cells"}.isdisjoint(written.attrs)
    assert parallel_scaling._write_output(index, grid, None) == 0.0


def test_load_netcdf_grid_rejects_an_all_missing_first_time_step(tmp_path: Path) -> None:
    """The land mask needs at least one finite cell in the first time step."""
    path = tmp_path / "empty.nc"
    xr.Dataset(
        {"precip": (("lat", "lon", "time"), np.full((2, 2, 3), np.nan))},
        coords={"lat": [10.0, 20.0], "lon": [1.0, 2.0], "time": pd.date_range("2000-01-01", periods=3, freq="MS")},
    ).to_netcdf(path, engine="h5netcdf")
    with pytest.raises(ValueError, match="no finite cell"):
        parallel_scaling.load_netcdf_grid(str(path), "precip")


def test_require_finite_tail_enforces_the_land_mask() -> None:
    """A masked grid must be finite on land and NaN on the cells the mask excludes."""
    values = np.ones((4, 2, 2))
    valid = np.array([[True, False], [True, True]])
    values[:, ~valid] = np.nan
    parallel_scaling._require_finite_tail(values, 0, valid)

    degenerate = values.copy()
    degenerate[1, 0, 0] = np.nan
    with pytest.raises(RuntimeError, match="land-cell"):
        parallel_scaling._require_finite_tail(degenerate, 0, valid)

    filled_ocean = values.copy()
    filled_ocean[:, 0, 1] = 0.0
    with pytest.raises(RuntimeError, match="marked as missing"):
        parallel_scaling._require_finite_tail(filled_ocean, 0, valid)


def test_measure_threads_the_land_mask_through_dask_workers() -> None:
    """A masked grid survives ``_chunk_for_workers``/``_Grid`` and the ``processes`` round trip:
    the run completes and the NaN ocean cells pass the parent-side finite-tail gate.
    """
    values = np.full((6, 2, 2), 2.0)
    values[:, 1, 1] = np.nan
    precip = xr.DataArray(values, coords={"lat": [10.0, 20.0], "lon": [1.0, 2.0]}, dims=("time", "lat", "lon"))
    valid_cells = np.array([[True, True], [True, False]])
    grid = parallel_scaling._Grid(precip=precip, valid_cells=valid_cells)
    index = parallel_scaling._Index(lambda g: g.precip, 0)

    samples = parallel_scaling._measure(index, grid, workers=2, repeat=1)
    assert len(samples) == 1


def test_measure_rejects_a_run_that_fills_the_land_mask() -> None:
    """``_measure`` applies the land mask to the values the workers returned, not just to the input."""
    values = np.full((6, 2, 2), 2.0)
    values[:, 1, 1] = np.nan
    precip = xr.DataArray(values, coords={"lat": [10.0, 20.0], "lon": [1.0, 2.0]}, dims=("time", "lat", "lon"))
    valid_cells = np.array([[True, True], [True, False]])
    grid = parallel_scaling._Grid(precip=precip, valid_cells=valid_cells)
    index = parallel_scaling._Index(lambda g: g.precip.fillna(1.0), 0)

    with pytest.raises(RuntimeError, match="marked as missing"):
        parallel_scaling._measure(index, grid, workers=2, repeat=1)


def test_quiet_worker_silences_only_goodness_of_fit() -> None:
    """The worker-side filter must leave unrelated warning categories visible."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parallel_scaling._quiet_worker()
        warnings.warn("fit", GoodnessOfFitWarning, stacklevel=2)
        warnings.warn("other", RuntimeWarning, stacklevel=2)
    assert [type(warning.message) for warning in caught] == [RuntimeWarning]


def test_require_finite_tail_honors_the_declared_padding() -> None:
    """Non-finite output is rejected past the index's own leading NaN pad."""
    padded = np.concatenate([np.full(parallel_scaling.SCALE - 1, np.nan), np.ones(4)])
    parallel_scaling._require_finite_tail(padded, parallel_scaling.SCALE - 1)
    unpadded = np.ones(6)
    parallel_scaling._require_finite_tail(unpadded, 0)

    # a NaN beyond the declared pad is a degenerated fit, and PET declares no pad
    beyond_pad = np.concatenate([np.full(parallel_scaling.SCALE - 1, np.nan), [np.nan], np.ones(3)])
    with pytest.raises(RuntimeError, match="padded time steps"):
        parallel_scaling._require_finite_tail(beyond_pad, parallel_scaling.SCALE - 1)
    with pytest.raises(RuntimeError, match="padded time steps"):
        parallel_scaling._require_finite_tail(np.concatenate([[np.nan], np.ones(5)]), 0)


def test_every_benchmarked_index_declares_its_padding() -> None:
    """The README before/after table names these four indices; pads match the kernels."""
    assert set(parallel_scaling._RUNNERS) == {"spi", "spei", "pet", "eddi"}
    pads = {name: index.leading_pad for name, index in parallel_scaling._RUNNERS.items()}
    assert pads == {
        "spi": parallel_scaling.SCALE - 1,
        "spei": parallel_scaling.SCALE - 1,
        "pet": 0,
        "eddi": parallel_scaling.SCALE - 1,
    }
