"""Offline regression check for the notebook and companion Dask script."""

import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import compute, indices


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
