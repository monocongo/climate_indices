"""Offline regression checks for the ERA5 energy tutorial and acquisition contract."""

from __future__ import annotations

import importlib.util
import json
import socket
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]


def _preparation_module() -> Any:
    spec = importlib.util.spec_from_file_location("prepare_energy_demo", ROOT / "scripts/prepare_energy_demo_inputs.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_year() -> xr.Dataset:
    times = pd.date_range("2000-01-01", "2000-12-31T18:00", freq="6h")
    shape = (len(times), 3, 4)
    return xr.Dataset(
        {
            "2m_temperature": (("time", "latitude", "longitude"), np.full(shape, 280.0, dtype="float32")),
            "10m_wind_speed": (("time", "latitude", "longitude"), np.full(shape, 5.0, dtype="float32")),
        },
        coords={"time": times, "latitude": [48.0, 51.0, 54.0], "longitude": [3.0, 6.0, 9.0, 12.0]},
    )


def test_acquisition_retains_sample_values_and_validates_cache(tmp_path: Path) -> None:
    """Preparation preserves six-hourly values rather than applying a daily transform."""
    module = _preparation_module()
    source = _source_year()
    source["10m_wind_speed"][0, 0, 0] = 12.0
    # Real ARCO coordinate labels are slightly different from exact degree values.
    source = source.assign_coords(latitude=source.latitude + 1e-12)
    result = module.read_year(source, 2000, tmp_path)
    assert result.sizes["time"] == 366 * 4
    assert result.wind_speed_10m[0, 0, 0].item() == 12.0
    assert result.temperature_k[0, 0, 0].item() == 280.0
    assert result.temperature_k.attrs["units"] == "K"
    assert result.wind_speed_10m.attrs["units"] == "m s-1"
    xr.testing.assert_identical(result, module.read_year(xr.Dataset(), 2000, tmp_path))

    corrupted = result.isel(time=slice(1, None))
    corrupted.to_netcdf(tmp_path / "arco_era5_2000.nc", engine="h5netcdf")
    with pytest.raises(ValueError, match="complete six-hourly"):
        module.read_year(xr.Dataset(), 2000, tmp_path)


@pytest.mark.parametrize("defect", ["time", "coordinates", "missing", "wind", "temperature"])
def test_acquisition_rejects_invalid_inputs(tmp_path: Path, defect: str) -> None:
    module = _preparation_module()
    source = _source_year()
    if defect == "time":
        source = source.isel(time=slice(1, None))
    elif defect == "coordinates":
        source = source.assign_coords(latitude=source.latitude + 0.1)
    elif defect == "missing":
        source["2m_temperature"][0, 0, 0] = np.nan
    elif defect == "wind":
        source["10m_wind_speed"][0, 0, 0] = -1.0
    else:
        source["2m_temperature"][0, 0, 0] = 0.0
    with pytest.raises((ValueError, KeyError)):
        module.read_year(source, 2000, tmp_path)
    assert not (tmp_path / "arco_era5_2000.nc").exists()


def test_energy_notebook_executes_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the actual cells offline; the docs gate separately proves kernel execution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("The energy notebook must not access the network")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.chdir(ROOT / "notebooks")
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    notebook = json.loads((ROOT / "notebooks/energy_demo.ipynb").read_text())
    namespace: dict[str, Any] = {"__name__": "__energy_demo_test__"}
    try:
        with warnings.catch_warnings():
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), "energy_demo.ipynb", "exec"), namespace)
    finally:
        plt.close("all")

    # Pin the nonlinear averaging order independently of the notebook's transform call.
    speed = namespace["weather"].wind_speed_10m.isel(time=slice(0, 4)).values.astype(float) * 10 ** (1 / 7)
    factors = np.where(speed >= 25.0, 0.0, np.clip((speed**3 - 3.5**3) / (13.0**3 - 3.5**3), 0.0, 1.0))
    np.testing.assert_allclose(namespace["wind_cf"].isel(time=0).values, factors.mean(axis=0), atol=1e-15)
    assert set(namespace["scores"].data_vars) == {"srepi", "demand", "srli"}
    assert namespace["wind_events"].sizes["period"] == 30
    assert len(namespace["map_fields"]) == 4
    assert namespace["threshold"] == 1.28
    assert (namespace["runs"].duration >= 3).all()
    assert namespace["provenance"]["size_bytes"] < 5_000_000
