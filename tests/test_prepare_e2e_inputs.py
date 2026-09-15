"""Regression tests for E2E input generation publication."""

import hashlib
import importlib.util
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import exceptions

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _prepare_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location("prepare_e2e_inputs_test", SCRIPTS_DIR / "prepare_e2e_inputs.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_downloads_use_private_temporary_files(tmp_path, monkeypatch):
    """Concurrent cold-cache downloads must not share an in-progress file."""
    module = _prepare_module(monkeypatch)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    payload = b"source bytes"
    checksum = hashlib.sha256(payload).hexdigest()
    barrier = threading.Barrier(2)

    def fake_urlopen(*args, **kwargs):
        barrier.wait(timeout=5)
        return io.BytesIO(payload)

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    with ThreadPoolExecutor(max_workers=2) as executor:
        downloads = list(executor.map(lambda _: module._cache_source(source_dir, "prcp", checksum), range(2)))

    assert all(path.read_bytes() == payload for path in downloads)
    assert not list(source_dir.glob("*.download"))


def test_failed_regeneration_preserves_current_generation(tmp_path, monkeypatch):
    """Only a fully validated generation may replace the current inputs."""
    pytest.importorskip("h5py")
    pytest.importorskip("zarr")
    module = _prepare_module(monkeypatch)
    times = pd.date_range("1980-01-01", "2016-12-01", freq="MS")
    payloads = {}
    for name, value in (("prcp", 50.0), ("pet", 10.0)):
        source_path = tmp_path / f"nclimgrid_lowres_{name}.nc"
        dataset = xr.Dataset(
            {name: (("time", "lat", "lon"), np.full((len(times), 1, 1), value))},
            coords={"time": times, "lat": [35.0], "lon": [-100.0]},
        )
        dataset[name].attrs["units"] = "mm"
        dataset.to_netcdf(source_path, engine="h5netcdf")
        payloads[source_path.name] = source_path.read_bytes()

    def fake_urlopen(url, **kwargs):
        return io.BytesIO(payloads[Path(url).name])

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        module,
        "SOURCES",
        {
            "prcp": hashlib.sha256(payloads["nclimgrid_lowres_prcp.nc"]).hexdigest(),
            "pet": hashlib.sha256(payloads["nclimgrid_lowres_pet.nc"]).hexdigest(),
        },
    )
    output_dir = tmp_path / "e2e"
    fixture_grid = {"lat": 1, "lon": 1}
    module.prepare_inputs(output_dir, expected_grid=fixture_grid)
    current = output_dir / "current"
    published = current.resolve()
    manifest = json.loads((published / "manifest.json").read_text())
    assert current.is_symlink()

    def fail(*args, **kwargs):
        raise RuntimeError("simulated generation failure")

    monkeypatch.setattr(module, "clean_and_prepare_inputs", fail)
    with pytest.raises(RuntimeError, match="simulated generation failure"):
        module.prepare_inputs(output_dir, expected_grid=fixture_grid)

    assert current.resolve() == published
    assert json.loads((current / "manifest.json").read_text()) == manifest
    assert all(
        (current / name).exists() for name in ("raw_precipitation.nc", "raw_pet.nc", "cache_prepared_input.zarr")
    )


def _write_monthly_inputs(tmp_path, times, units="mm"):
    """Write single-pixel precip/PET NetCDF inputs at the given monthly timestamps."""
    rng = np.random.default_rng(0)
    shape = (len(times), 1, 1)
    precip = rng.gamma(2, 40, shape).astype("float32")
    pet = rng.uniform(20, 100, shape).astype("float32")
    ds = xr.Dataset(
        {"pr": (("time", "lat", "lon"), precip), "pet": (("time", "lat", "lon"), pet)},
        coords={"time": times, "lat": [35.0], "lon": [-100.0]},
    )
    for name in ds:
        ds[name].attrs["units"] = units
    precip_path, pet_path = tmp_path / "precip.nc", tmp_path / "pet.nc"
    ds[["pr"]].to_netcdf(precip_path, engine="scipy")
    ds[["pet"]].to_netcdf(pet_path, engine="scipy")
    return precip_path, pet_path


def test_clean_and_prepare_inputs_rejects_wrong_units(tmp_path, monkeypatch):
    pytest.importorskip("zarr")
    module = _prepare_module(monkeypatch)
    times = pd.date_range("1980-01-01", periods=12, freq="MS")
    precip_path, pet_path = _write_monthly_inputs(tmp_path, times, units="inches")
    with pytest.raises(exceptions.InvalidArgumentError):
        module.clean_and_prepare_inputs(precip_path, pet_path, tmp_path / "prepared.zarr")


def test_clean_and_prepare_inputs_rejects_irregular_timestamps(tmp_path, monkeypatch):
    pytest.importorskip("zarr")
    module = _prepare_module(monkeypatch)
    # 12 monthly timestamps with July 1980 missing (and 1981-01 appended instead).
    times = pd.date_range("1980-01-01", periods=13, freq="MS").delete(6)
    precip_path, pet_path = _write_monthly_inputs(tmp_path, times)
    with pytest.raises(exceptions.CoordinateValidationError):
        module.clean_and_prepare_inputs(precip_path, pet_path, tmp_path / "prepared.zarr")


def test_clean_and_prepare_inputs_accepts_month_end_timestamps(tmp_path, monkeypatch):
    pytest.importorskip("zarr")
    module = _prepare_module(monkeypatch)
    times = pd.date_range("1980-01-31", periods=12, freq=pd.offsets.MonthEnd())
    precip_path, pet_path = _write_monthly_inputs(tmp_path, times)
    prepared = tmp_path / "prepared.zarr"
    module.clean_and_prepare_inputs(precip_path, pet_path, prepared)
    with xr.open_zarr(prepared) as actual:
        xr.testing.assert_equal(actual.time, xr.DataArray(times, dims="time", name="time"))


@pytest.mark.parametrize("times", [np.array([], dtype="datetime64[ns]"), np.array(["not-a-date"])])
def test_monthly_time_validation_uses_coordinate_validation_error(times, monkeypatch):
    module = _prepare_module(monkeypatch)
    with pytest.raises(exceptions.CoordinateValidationError):
        module._validate_monthly_time(times)
