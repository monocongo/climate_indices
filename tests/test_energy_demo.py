"""Offline regression checks for the ERA5 energy tutorial and acquisition contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import climate_indices

ROOT = Path(__file__).resolve().parents[1]
# The tutorial and its docs block locate this checkout from the imported package, which an installed wheel is not in.
requires_checkout_import = pytest.mark.skipif(
    Path(climate_indices.__file__).resolve().parent != ROOT / "src/climate_indices",
    reason="the energy tutorial runs from this checkout, so climate_indices must import from its src/",
)


def _preparation_module() -> Any:
    spec = importlib.util.spec_from_file_location("prepare_energy_demo", ROOT / "scripts/prepare_energy_demo_inputs.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_year() -> xr.Dataset:
    times = pd.date_range("2000-01-01", "2000-12-31T18:00", freq="6h")
    shape = (len(times), 3, 4)
    source = xr.Dataset(
        {
            "2m_temperature": (("time", "latitude", "longitude"), np.full(shape, 280.0, dtype="float32")),
            "10m_wind_speed": (("time", "latitude", "longitude"), np.full(shape, 5.0, dtype="float32")),
        },
        coords={"time": times, "latitude": [48.0, 51.0, 54.0], "longitude": [3.0, 6.0, 9.0, 12.0]},
    )
    source["2m_temperature"].attrs["units"] = "K"
    source["10m_wind_speed"].attrs["units"] = "m s-1"
    return source


def test_acquisition_retains_sample_values_and_validates_cache(tmp_path: Path) -> None:
    """Preparation preserves six-hourly values rather than applying a daily transform."""
    pytest.importorskip("h5py")
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
    corrupted.to_netcdf(module.cache_path(tmp_path, 2000), engine="h5netcdf")
    empty_source = xr.Dataset()
    with pytest.raises(ValueError, match="complete six-hourly"):
        module.read_year(empty_source, 2000, tmp_path)


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
    assert not module.cache_path(tmp_path, 2000).exists()


@requires_checkout_import
def test_energy_notebook_executes_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the actual cells offline; the docs gate separately proves kernel execution."""
    pytest.importorskip("h5py")
    pytest.importorskip("IPython")
    matplotlib = pytest.importorskip("matplotlib")
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


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("field, units", [("2m_temperature", "degC"), ("10m_wind_speed", "km h-1")])
def test_acquisition_rejects_incompatible_units(tmp_path: Path, cached: bool, field: str, units: str) -> None:
    pytest.importorskip("h5py")
    module = _preparation_module()
    source = _source_year()
    source[field].attrs["units"] = units
    if cached:
        source.rename(module.VARIABLES).to_netcdf(module.cache_path(tmp_path, 2000), engine="h5netcdf")
        source = xr.Dataset()
    with pytest.raises(ValueError, match="units|must be in"):
        module.read_year(source, 2000, tmp_path)


@pytest.mark.parametrize("filename", ["inputs.nc", "previous.nc"])
@pytest.mark.parametrize("failure", ["data", "metadata", "publication"])
def test_failed_publication_preserves_existing_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, filename: str
) -> None:
    pytest.importorskip("h5py")
    module = _preparation_module()
    output = tmp_path / filename
    manifest = output.with_suffix(".provenance.json")
    output.write_bytes(b"previous input")
    manifest.write_bytes(b"previous provenance")
    provenance: dict[str, object] = {}
    if failure == "data":

        def fail_write(self: xr.Dataset, path: Path, **kwargs: object) -> None:
            path.write_bytes(b"partial NetCDF")
            raise OSError("simulated data write failure")

        monkeypatch.setattr(xr.Dataset, "to_netcdf", fail_write)
    elif failure == "metadata":
        provenance["unserializable"] = object()
    else:
        original_replace = module.os.replace

        def fail_replace(source: Path, target: Path) -> None:
            if target == manifest:
                raise OSError("simulated manifest replacement failure")
            original_replace(source, target)

        monkeypatch.setattr(module.os, "replace", fail_replace)
    dataset = _source_year().rename(module.VARIABLES)
    with pytest.raises((OSError, TypeError)):
        module.write_outputs(dataset, output, provenance)
    assert output.read_bytes() == b"previous input"
    assert manifest.read_bytes() == b"previous provenance"
    assert set(tmp_path.iterdir()) == {output, manifest}


@pytest.mark.parametrize("filename", ["inputs.nc", "previous.nc"])
def test_successful_publication_records_written_bytes(tmp_path: Path, filename: str) -> None:
    pytest.importorskip("h5py")
    module = _preparation_module()
    output = tmp_path / filename
    output.write_bytes(b"previous input")
    module.write_outputs(_source_year().rename(module.VARIABLES), output, {"source": module.SOURCE_URL})
    provenance = json.loads(output.with_suffix(".provenance.json").read_text())
    assert provenance["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert provenance["size_bytes"] == output.stat().st_size
    assert provenance["source"] == module.SOURCE_URL


@requires_checkout_import
def test_docs_use_current_interpreter_despite_external_kernelspec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for dependency in ("nbformat", "nbclient", "ipykernel"):
        pytest.importorskip(dependency)
    external = tmp_path / "kernels/python3"
    external.mkdir(parents=True)
    (external / "kernel.json").write_text(
        json.dumps(
            {"argv": ["/wrong/python", "-f", "{connection_file}"], "display_name": "Wrong", "language": "python"}
        )
    )
    monkeypatch.setenv("JUPYTER_PATH", str(tmp_path))
    page = (ROOT / "docs/tutorials.md").read_text()
    block = page.split("```{testcode} energy-notebook\n:hide:\n", 1)[1].split("```", 1)[0]
    setup = block.split("executed = client.execute", 1)[0]
    namespace: dict[str, Any] = {}
    exec(compile(setup, "docs/tutorials.md", "exec"), namespace)
    assert Path(namespace["manager"].kernel_spec.argv[0]).resolve() == Path(sys.executable).resolve()
    assert "client.execute(cleanup_kc=True)" in block


def test_annual_cache_reuse_is_bound_to_the_source_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cache made from another source selection is not reused after the source changes."""
    pytest.importorskip("h5py")
    module = _preparation_module()
    source = _source_year()
    module.read_year(source, 2000, tmp_path)
    assert len(list(tmp_path.iterdir())) == 1
    monkeypatch.setattr(module, "SOURCE_URL", "https://example.invalid/other.zarr")
    module.read_year(source, 2000, tmp_path)
    assert len(list(tmp_path.iterdir())) == 2


def test_interrupted_annual_cache_is_not_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _preparation_module()

    def fail_write(self: xr.Dataset, path: Path, **kwargs: object) -> None:
        path.write_bytes(b"partial NetCDF")
        raise OSError("simulated annual write failure")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fail_write)
    source = _source_year()
    with pytest.raises(OSError, match="annual write"):
        module.read_year(source, 2000, tmp_path)
    assert not list(tmp_path.iterdir())
