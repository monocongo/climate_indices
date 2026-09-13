"""Offline regression check for the notebook and companion Dask script."""

import ast
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import compute, exceptions, indices

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# The canonical data/calibration contract (matches data/e2e/manifest.json once
# scripts/prepare_e2e_inputs.py has generated it, and both e2e entrypoints).
CANONICAL_YEARS = {"data_start_year": 1980, "cal_start_year": 1981, "cal_end_year": 2010}


@pytest.fixture
def e2e(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    import end_to_end_example

    return end_to_end_example


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


def _extract_pipeline_config_years(source: str) -> dict[str, int]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pipeline_config" for target in node.targets
        ):
            return {
                key.value: value.value
                for key, value in zip(node.value.keys, node.value.values, strict=True)
                if isinstance(key, ast.Constant) and key.value in CANONICAL_YEARS and isinstance(value, ast.Constant)
            }
    raise AssertionError("pipeline_config assignment not found")


def test_pipeline_config_matches_canonical_contract():
    """Guards against reintroducing the stale 1990/1991-2020 configuration."""
    script_source = (SCRIPTS_DIR / "end_to_end_example.py").read_text()
    assert _extract_pipeline_config_years(script_source) == CANONICAL_YEARS

    notebook = json.loads((SCRIPTS_DIR / "e2e_with_dask.ipynb").read_text())
    notebook_source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if "pipeline_config" in "".join(cell.get("source", []))
    )
    assert _extract_pipeline_config_years(notebook_source) == CANONICAL_YEARS


def test_clean_and_prepare_inputs_rejects_wrong_units(tmp_path, e2e):
    pytest.importorskip("zarr")
    times = pd.date_range("1980-01-01", periods=12, freq="MS")
    precip_path, pet_path = _write_monthly_inputs(tmp_path, times, units="inches")
    with pytest.raises(exceptions.InvalidArgumentError):
        e2e.clean_and_prepare_inputs(precip_path, pet_path, tmp_path / "prepared.zarr")


def test_clean_and_prepare_inputs_rejects_irregular_timestamps(tmp_path, e2e):
    pytest.importorskip("zarr")
    # 12 monthly timestamps with July 1980 missing (and 1981-01 appended instead).
    times = pd.date_range("1980-01-01", periods=13, freq="MS").delete(6)
    precip_path, pet_path = _write_monthly_inputs(tmp_path, times)
    with pytest.raises(exceptions.CoordinateValidationError):
        e2e.clean_and_prepare_inputs(precip_path, pet_path, tmp_path / "prepared.zarr")


def test_clean_and_prepare_inputs_accepts_month_end_timestamps(tmp_path, e2e):
    pytest.importorskip("zarr")
    times = pd.date_range("1980-01-31", periods=12, freq=pd.offsets.MonthEnd())
    precip_path, pet_path = _write_monthly_inputs(tmp_path, times)
    prepared = tmp_path / "prepared.zarr"
    e2e.clean_and_prepare_inputs(precip_path, pet_path, prepared)
    with xr.open_zarr(prepared) as actual:
        xr.testing.assert_equal(actual.time, xr.DataArray(times, dims="time", name="time"))


@pytest.mark.parametrize("times", [np.array([], dtype="datetime64[ns]"), np.array(["not-a-date"])])
def test_monthly_time_validation_uses_coordinate_validation_error(times, e2e):
    with pytest.raises(exceptions.CoordinateValidationError):
        e2e._validate_monthly_time(times)


def test_compute_indices_parallel_rejects_stale_data_start_year(tmp_path, e2e):
    pytest.importorskip("zarr")
    times = pd.date_range("1980-01-01", periods=24, freq="MS")
    precip_path, pet_path = _write_monthly_inputs(tmp_path, times)
    prepared = tmp_path / "prepared.zarr"
    e2e.clean_and_prepare_inputs(precip_path, pet_path, prepared)
    config = {
        "scale": 3,
        "distribution_spi": indices.Distribution.gamma,
        "distribution_spei": indices.Distribution.pearson,
        "periodicity": compute.Periodicity.monthly,
        "data_start_year": 1990,
        "cal_start_year": 1990,
        "cal_end_year": 1991,
    }
    with pytest.raises(exceptions.InvalidArgumentError):
        e2e.compute_indices_parallel(prepared, tmp_path / "output.zarr", config)


def test_compute_indices_parallel_rejects_calibration_outside_data_range(tmp_path, e2e):
    pytest.importorskip("zarr")
    times = pd.date_range("1980-01-01", periods=24, freq="MS")
    precip_path, pet_path = _write_monthly_inputs(tmp_path, times)
    prepared = tmp_path / "prepared.zarr"
    e2e.clean_and_prepare_inputs(precip_path, pet_path, prepared)
    config = {
        "scale": 3,
        "distribution_spi": indices.Distribution.gamma,
        "distribution_spei": indices.Distribution.pearson,
        "periodicity": compute.Periodicity.monthly,
        "data_start_year": 1980,
        "cal_start_year": 1981,
        "cal_end_year": 2020,
    }
    with pytest.raises(exceptions.InvalidArgumentError):
        e2e.compute_indices_parallel(prepared, tmp_path / "output.zarr", config)


def test_compute_indices_parallel_rejects_discontinuous_prepared_store(tmp_path, e2e):
    pytest.importorskip("zarr")
    times = pd.date_range("1980-01-01", periods=372, freq="MS").delete(6).append(pd.DatetimeIndex(["2010-12-01"]))
    prepared = tmp_path / "prepared.zarr"
    xr.Dataset(
        {
            "precip": (("time", "lat", "lon"), np.ones((372, 1, 1), dtype="float32")),
            "pet": (("time", "lat", "lon"), np.ones((372, 1, 1), dtype="float32")),
        },
        coords={"time": times, "lat": [35.0], "lon": [-100.0]},
    ).chunk({"time": -1}).to_zarr(prepared, zarr_format=2, consolidated=True)
    config = {
        "scale": 3,
        "distribution_spi": indices.Distribution.gamma,
        "distribution_spei": indices.Distribution.pearson,
        "periodicity": compute.Periodicity.monthly,
        "data_start_year": 1980,
        "cal_start_year": 1981,
        "cal_end_year": 2010,
    }
    with pytest.raises(exceptions.CoordinateValidationError):
        e2e.compute_indices_parallel(prepared, tmp_path / "output.zarr", config)


@pytest.mark.parametrize("entrypoint", ["end_to_end_example.py", "e2e_with_dask.ipynb"])
def test_e2e_pipeline(tmp_path, monkeypatch, entrypoint):
    pytest.importorskip("zarr")
    path = Path(__file__).resolve().parents[1] / "scripts" / entrypoint
    if path.suffix == ".py":
        namespace = runpy.run_path(str(path))
    else:
        namespace = {}
        for cell in json.loads(path.read_text())["cells"]:
            source = "".join(cell.get("source", []))
            if cell["cell_type"] == "code" and ("from end_to_end_example import" in source):
                monkeypatch.chdir(path.parent)
                monkeypatch.syspath_prepend(str(path.parent))
                exec(compile(source, str(path), "exec"), namespace)

    # One land cell and one masked cell, with a complete 30-year Calibration Period.
    rng = np.random.default_rng(42)
    shape = (372, 1, 2)
    precip = rng.gamma(2, 40, shape).astype("float32")
    pet = rng.uniform(20, 100, shape).astype("float32")
    precip[:, 0, 1] = pet[:, 0, 1] = np.nan
    precip[100, 0, 0] = 0.0  # meaningful zero precipitation, distinct from the masked/missing pixel
    ds = xr.Dataset(
        {"pr": (("time", "lat", "lon"), precip), "pet": (("time", "lat", "lon"), pet)},
        coords={"time": pd.date_range("1980-01-01", periods=372, freq="MS"), "lat": [35.0], "lon": [-100.0, -99.0]},
    )
    for name in ds:
        ds[name].attrs["units"] = "mm"
    precip_path, pet_path = tmp_path / "precip.nc", tmp_path / "pet.nc"
    ds[["pr"]].to_netcdf(precip_path, engine="scipy")
    ds[["pet"]].to_netcdf(pet_path, engine="scipy")
    prepared, output = tmp_path / "prepared.zarr", tmp_path / "output.zarr"
    namespace["clean_and_prepare_inputs"](precip_path, pet_path, prepared)
    config = {
        "scale": 3,
        "distribution_spi": indices.Distribution.gamma,
        "distribution_spei": indices.Distribution.pearson,
        "data_start_year": 1980,
        "cal_start_year": 1981,
        "cal_end_year": 2010,
        "periodicity": compute.Periodicity.monthly,
    }
    namespace["compute_indices_parallel"](prepared, output, config)
    with xr.open_zarr(prepared) as actual:
        assert actual.precip.dims == ("time", "lat", "lon")
        assert actual.precip.chunks[0] == (372,)
        np.testing.assert_allclose(actual.wb, ds.pr - ds.pet)
        np.testing.assert_allclose(actual.pet, ds.pet)
    kwargs = {
        "scale": 3,
        "data_start_year": 1980,
        "calibration_year_initial": 1981,
        "calibration_year_final": 2010,
        "periodicity": compute.Periodicity.monthly,
    }
    expected_spi = indices.spi(precip[:, 0, 0].copy(), distribution=indices.Distribution.gamma, **kwargs)
    expected_spei = indices.spei(
        precip[:, 0, 0].copy(), pet[:, 0, 0].copy(), distribution=indices.Distribution.pearson, **kwargs
    )
    with xr.open_zarr(output) as actual:
        np.testing.assert_allclose(actual.spi_3[:, 0, 0], expected_spi, atol=1e-6)
        np.testing.assert_allclose(actual.spei_3[:, 0, 0], expected_spei, atol=1e-6)
        assert actual.spi_3[:, 0, 1].isnull().all()
        assert actual.spei_3[:, 0, 1].isnull().all()
        assert actual.spi_3[:2].isnull().all()
        assert actual.spei_3[:2].isnull().all()
        # Meaningful zero precipitation (index 100, land pixel) stays a real value, not NaN.
        assert np.isfinite(actual.spi_3[100, 0, 0])
        assert np.isfinite(actual.spei_3[100, 0, 0])
        xr.testing.assert_equal(actual.time, ds.time)


def test_failed_run_preserves_existing_output(tmp_path, monkeypatch):
    pytest.importorskip("zarr")
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "end_to_end_example.py"))
    shape = (372, 1, 1)
    prepared, output = tmp_path / "prepared.zarr", tmp_path / "output.zarr"
    rng = np.random.default_rng(1)
    ds = xr.Dataset(
        {
            "precip": (("time", "lat", "lon"), rng.gamma(2, 40, shape).astype("float32")),
            "pet": (("time", "lat", "lon"), rng.uniform(20, 100, shape).astype("float32")),
            "wb": (("time", "lat", "lon"), np.zeros(shape, dtype="float32")),
        },
        coords={"time": pd.date_range("1980-01-01", periods=372, freq="MS"), "lat": [35.0], "lon": [-100.0]},
    )
    ds.chunk({"time": -1}).to_zarr(prepared, zarr_format=2)
    sentinel = xr.Dataset({"old": (("x",), [1.0])})
    sentinel.to_zarr(output, mode="w", zarr_format=2)
    config = {
        "scale": 3,
        "distribution_spi": indices.Distribution.gamma,
        "distribution_spei": indices.Distribution.pearson,
        "data_start_year": 1980,
        "cal_start_year": 1981,
        "cal_end_year": 2010,
        "periodicity": compute.Periodicity.monthly,
    }

    def boom(*args, **kwargs):
        raise RuntimeError("simulated block failure")

    monkeypatch.setattr(xr.Dataset, "to_zarr", boom)
    with pytest.raises(RuntimeError, match="simulated block failure"):
        namespace["compute_indices_parallel"](prepared, output, config)
    monkeypatch.undo()
    with xr.open_zarr(output) as actual:
        xr.testing.assert_equal(actual, sentinel)
