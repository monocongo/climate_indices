"""Unit and integration checks for the CLI-vs-xarray/Dask benchmark harness (#1121).

Covers the ``prepare`` transform, the equivalence gate, and one small
end-to-end run of the real `climate_indices.__main__.main` CLI entry point
against the xarray/Dask adapter, over a synthetic grid small enough to run in
CI; the full-CONUS timing numbers themselves live in
``benchmarks/cli_multiprocessing.py`` and are run manually.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import spi
from climate_indices.compute import Periodicity
from climate_indices.indices import Distribution

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
    cli_multiprocessing = _load_module("cli_multiprocessing", BENCHMARKS / "cli_multiprocessing.py")
finally:
    sys.path.remove(str(BENCHMARKS))


def _write_source_fixture(path: Path, periods: int, start: str, zero_at: str) -> None:
    """A 3-lat x 4-lon precipitation source shaped like a trimmed nClimGrid slice.

    Row ``lat=10`` is all-NaN, like the ocean mask a real grid carries; one
    land cell is zero at ``zero_at`` to exercise the 0.01 mm replacement.
    """
    rng = np.random.default_rng(1121)
    values = rng.gamma(shape=2.0, scale=20.0, size=(periods, 3, 4)).astype(np.float64)
    values[:, 0, :] = np.nan
    times = pd.date_range(start, periods=periods, freq="MS")
    values[times.get_loc(pd.Timestamp(zero_at)), 1, 2] = 0.0
    xr.Dataset(
        {"precip": (("time", "lat", "lon"), values, {"units": "mm"})},
        coords={"time": times, "lat": [10.0, 20.0, 30.0], "lon": [1.0, 2.0, 3.0, 4.0]},
    ).to_netcdf(path, engine="h5netcdf")


def test_prepare_masks_from_the_first_step_and_replaces_zeros(tmp_path: Path) -> None:
    """prepare trims time, masks every step to the first step's finite cells, and zero-fixes land cells."""
    source = tmp_path / "source.nc"
    _write_source_fixture(source, periods=480, start="1980-01-01", zero_at="1990-06-01")
    target = tmp_path / "prepared.nc"

    cli_multiprocessing._prepare(
        argparse.Namespace(source=str(source), target=str(target), var_name="precip", start=1981, end=2010)
    )

    with xr.open_dataset(target) as ds:
        da = ds["prcp"]
        assert da.dtype == np.float32
        assert da.attrs["units"] == "mm"
        assert da.dims == ("lat", "lon", "time"), "the CLI's shared-array transport accepts only this order"
        assert da.sizes["time"] == 30 * 12
        assert da["time"].dt.year.values[0] == 1981
        assert da["time"].dt.year.values[-1] == 2010

        values = da.transpose("time", "lat", "lon").values
        assert np.isnan(values[:, 0, :]).all(), "the all-NaN source row must stay masked"
        assert np.isfinite(values[:, 1:, :]).all(), "every land cell must be finite in every time step"

        replaced = da.sel(time="1990-06-01", lat=20.0, lon=3.0).item()
        assert replaced == np.float32(0.01)
        assert not np.any(values[:, 1:, :] == 0.0), "no land cell may still read exactly zero"


def test_assert_equivalence_rejects_a_changed_value(tmp_path: Path) -> None:
    """The comparison must fail loudly when a single cell differs between the two outputs."""
    coords = {
        "time": pd.date_range("2000-01-01", periods=2, freq="MS"),
        "lat": [10.0, 20.0, 30.0],
        "lon": [1.0, 2.0, 3.0, 4.0],
    }
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    cli_path = tmp_path / "cli.nc"
    xarray_path = tmp_path / "xarray.nc"
    xr.DataArray(values, coords=coords, dims=["time", "lat", "lon"], name="spi_gamma_06").to_dataset().to_netcdf(
        cli_path
    )
    changed = values.copy()
    changed[0, 0, 0] += 1.0
    xr.DataArray(changed, coords=coords, dims=["time", "lat", "lon"], name="spi_gamma_06").to_dataset().to_netcdf(
        xarray_path
    )

    with pytest.raises(AssertionError):
        cli_multiprocessing._assert_equivalence(str(cli_path), "spi_gamma_06", str(xarray_path))


def test_time_cli_matches_the_xarray_dask_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The real CLI entry point, run single-process, matches the xarray/Dask adapter on the same grid.

    This is the coupling this benchmark script depends on: `_time_cli` wraps
    `climate_indices.__main__._compute_write_index` and `._parallel_process` by
    name, and the equivalence gate must pass before any timing is reported.
    """
    source = tmp_path / "source.nc"
    _write_source_fixture(source, periods=480, start="1980-01-01", zero_at="1990-06-01")
    prepared = tmp_path / "prepared.nc"
    cli_multiprocessing._prepare(
        argparse.Namespace(source=str(source), target=str(prepared), var_name="precip", start=1980, end=2019)
    )

    grid = parallel_scaling.load_netcdf_grid(str(prepared), "prcp")
    xarray_output = tmp_path / "xarray_spi6_gamma.nc"
    spi(
        grid.precip,
        scale=6,
        distribution=Distribution.gamma,
        data_start_year=int(grid.precip["time"].dt.year[0]),
        calibration_year_initial=1990,
        calibration_year_final=2019,
        periodicity=Periodicity.monthly,
    ).astype("float32").to_netcdf(xarray_output)

    args = cli_multiprocessing._parse_args(
        [
            "time",
            str(prepared),
            "--var-name",
            "prcp",
            "--scale",
            "6",
            "--calibration-start",
            "1990",
            "--calibration-end",
            "2019",
            "--repeat",
            "1",
            "--output-dir",
            str(tmp_path / "cli"),
            "--xarray-output",
            str(xarray_output),
            "--multiprocessing",
            "single",
        ]
    )
    cli_multiprocessing._time_cli(args)

    output = capsys.readouterr().out
    assert "equivalence: CLI spi_gamma_06 == xarray" in output
    assert "cli gamma (multiprocessing.Pool, 1 workers): compute samples=[" in output
    assert "cli pearson (multiprocessing.Pool, 1 workers): compute samples=[" in output

    # the warm-up run's own sample must be discarded, not leaked into the
    # reported count -- --repeat 1 above, so exactly one sample per metric
    for distribution in ("gamma", "pearson"):
        match = re.search(rf"cli {distribution} .*?compute samples=\[([^\]]+)\]", output)
        assert match is not None
        assert len(match.group(1).split(",")) == 1, f"{distribution} compute: warm-up sample leaked into the count"
