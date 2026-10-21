"""Icechunk integration tests; existing tests do not require the optional extra."""

from __future__ import annotations

import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import compute, spi
from climate_indices.icechunk_io import open_icechunk, write_icechunk
from climate_indices.indices import Distribution
from tests import conftest


@pytest.fixture
def repository(tmp_path: Path) -> Any:
    icechunk = pytest.importorskip("icechunk")
    return icechunk.Repository.create(icechunk.local_filesystem_storage(str(tmp_path / "repository")))


@pytest.fixture
def dataset() -> xr.Dataset:
    values = np.random.default_rng(7).gamma(2.0, 30.0, (360, 2, 2))
    values[0, 0, 0] = np.nan
    values[12, 1, 1] = 0.0
    return xr.Dataset(
        {"precip": (("time", "lat", "lon"), values, {"units": "mm"})},
        coords={"time": pd.date_range("1990-01-01", periods=360, freq="MS"), "lat": [40.0, 41.0], "lon": [0.0, 1.0]},
        attrs={"source": "synthetic precipitation"},
    )


def test_round_trip(dataset: xr.Dataset, repository: Any) -> None:
    snapshot = write_icechunk(dataset, repository, message="Initial precipitation")
    assert snapshot == repository.lookup_branch("main")
    with xr.open_zarr(repository.readonly_session(branch="main").store, consolidated=False) as direct:
        xr.testing.assert_identical(direct.compute(), dataset)
    with open_icechunk(repository) as reopened:
        xr.testing.assert_identical(reopened.compute(), dataset)


def test_import_without_icechunk() -> None:
    code = """
import sys
sys.modules['icechunk'] = None
import climate_indices
from climate_indices.icechunk_io import open_icechunk, write_icechunk
assert callable(climate_indices.spi)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_spi_storage_and_backend_parity(
    dataset: xr.Dataset, repository: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: bool
) -> None:
    write_icechunk(dataset, repository, message="SPI input")
    dataset.to_netcdf(tmp_path / "precip.nc", engine="h5netcdf")
    dataset.to_zarr(tmp_path / "precip.zarr", zarr_format=2, consolidated=True)

    def calculate(source: xr.Dataset) -> xr.DataArray:
        return spi(
            source.precip.chunk({"time": -1, "lat": 1, "lon": 1}),
            scale=3,
            distribution=Distribution.gamma,
            calibration_year_initial=1990,
            calibration_year_final=2019,
        ).compute(scheduler="synchronous")

    conftest.disable_native(monkeypatch)
    with np.errstate(all="ignore"):
        expected = calculate(dataset)
        recorder = conftest.NativeRecorder(conftest.import_native()) if native else None
        monkeypatch.setattr(compute, "_native", recorder)
        with (
            open_icechunk(repository) as ice,
            xr.open_dataset(tmp_path / "precip.nc") as netcdf,
            xr.open_zarr(tmp_path / "precip.zarr", consolidated=True) as zarr,
        ):
            for source in (ice, netcdf, zarr):
                if recorder is not None:
                    recorder.calls.clear()
                actual = calculate(source)
                conftest.assert_native_parity(actual, expected)
                if recorder is not None:
                    assert recorder.calls == {"gamma_parameters", "gamma_probabilities", "norm_ppf"}
    assert compute._native is recorder


@pytest.mark.parametrize("scheduler", ["threads", "distributed"])
def test_dask_write_uses_to_icechunk(dataset: xr.Dataset, repository: Any, scheduler: str) -> None:
    icechunk_xarray = pytest.importorskip("icechunk.xarray")
    output = spi(
        dataset.precip.chunk({"time": -1, "lat": 1, "lon": 1}),
        scale=3,
        distribution=Distribution.gamma,
        calibration_year_initial=1990,
        calibration_year_final=2019,
    ).to_dataset(name="spi")
    expected = output.compute(scheduler="synchronous")
    with ExitStack() as stack:
        if scheduler == "distributed":
            distributed = pytest.importorskip("distributed")
            stack.enter_context(
                distributed.Client(n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None)
            )
        else:
            stack.enter_context(dask.config.set(scheduler="threads"))
        with (
            patch.object(icechunk_xarray, "to_icechunk", wraps=icechunk_xarray.to_icechunk) as writer,
            patch.object(xr.Dataset, "to_zarr", side_effect=AssertionError("Icechunk must not use Dataset.to_zarr")),
        ):
            write_icechunk(output.chunk({"time": 12}), repository, message="Dask SPI output")
            writer.assert_called_once()
        with open_icechunk(repository) as reopened:
            xr.testing.assert_identical(reopened.compute(), expected)


def test_failed_write_preserves_snapshot(dataset: xr.Dataset, repository: Any) -> None:
    icechunk_xarray = pytest.importorskip("icechunk.xarray")
    snapshot = write_icechunk(dataset, repository, message="Initial input")
    writer = icechunk_xarray.to_icechunk

    def fail_after_write(*args: Any, **kwargs: Any) -> None:
        writer(*args, **kwargs)
        raise RuntimeError("Write failed before commit")

    with patch.object(icechunk_xarray, "to_icechunk", side_effect=fail_after_write):
        with pytest.raises(RuntimeError, match="Write failed before commit"):
            write_icechunk(dataset * 2, repository, message="Failed update", mode="w")
    assert repository.lookup_branch("main") == snapshot
    with open_icechunk(repository) as reopened:
        xr.testing.assert_identical(reopened.compute(), dataset)


def test_replacement_preserves_readers_and_old_snapshot(dataset: xr.Dataset, repository: Any) -> None:
    snapshot = write_icechunk(dataset, repository, message="Initial input")
    with open_icechunk(repository) as original:
        with pytest.raises(FileExistsError):
            write_icechunk(dataset, repository, message="Must not replace")
        assert repository.lookup_branch("main") == snapshot
        replacement = dataset * 2
        write_icechunk(replacement, repository, message="Replace input", mode="w")
        with open_icechunk(repository) as reopened:
            xr.testing.assert_identical(reopened.compute(), replacement)
        xr.testing.assert_identical(original.compute(), dataset)
    with xr.open_zarr(repository.readonly_session(snapshot_id=snapshot).store, consolidated=False) as previous:
        xr.testing.assert_identical(previous.compute(), dataset)
