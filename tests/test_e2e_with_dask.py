"""Offline regression checks for the E2E teaching notebook.

The notebook is the authoritative artifact: these tests execute its code cells
against a tiny synthetic prepared store (located via CLIMATE_INDICES_E2E_DATA)
instead of importing a companion script.
"""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import compute, exceptions, indices

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "zarr_dask_spi_spei.ipynb"

# The canonical data/calibration contract (matches data/e2e/manifest.json once
# scripts/prepare_e2e_inputs.py has generated it, and the notebook).
CANONICAL_YEARS = {"data_start_year": 1980, "cal_start_year": 1981, "cal_end_year": 2010}


def _code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text())
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _executable_cells() -> list[str]:
    # The Dask Client cell is execution mechanics (#829's remit); without it Dask
    # falls back to the local scheduler, which keeps CI memory bounded.
    return [source for source in _code_cells() if "dask.distributed" not in source]


def _exec_cells(namespace: dict, sources: list[str]) -> None:
    for source in sources:
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)


def _split_at_calculation(sources: list[str]) -> tuple[list[str], list[str]]:
    for index, source in enumerate(sources):
        if "open_zarr" in source:
            return sources[:index], sources[index:]
    raise AssertionError("calculation cells not found")


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


def _publish_store(data_root: Path, ds: xr.Dataset) -> None:
    """Write a prepared store and manifest in the generations + current/ layout."""
    generation = data_root / "generations" / "synthetic"
    generation.mkdir(parents=True)
    chunked = ds.chunk({"time": -1, "lat": 1, "lon": 1})
    chunked.to_zarr(generation / "cache_prepared_input.zarr", zarr_format=2, consolidated=True)
    manifest = {
        "source_commit": "synthetic",
        "period": [str(ds.time.dt.strftime("%Y-%m-%d").values[0]), str(ds.time.dt.strftime("%Y-%m-%d").values[-1])],
        "calibration_period": [1981, 2010],
        "sources": {},
        "dimensions": dict(ds.sizes),
        "chunks": [ds.sizes["time"], 1, 1],
    }
    (generation / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (data_root / "current").symlink_to(generation, target_is_directory=True)


@pytest.fixture
def e2e_data(tmp_path, monkeypatch):
    """Synthetic prepared store: one land cell and one masked cell, 1980-2010 monthly."""
    pytest.importorskip("zarr")
    rng = np.random.default_rng(42)
    shape = (372, 2, 2)
    precip = rng.gamma(2, 40, shape).astype("float32")
    pet = rng.uniform(20, 100, shape).astype("float32")
    precip[:, 0, 1] = pet[:, 0, 1] = np.nan  # masked/missing cell
    precip[100, 0, 0] = 0.0  # meaningful zero precipitation, distinct from missing
    ds = xr.Dataset(
        {
            "precip": (("time", "lat", "lon"), precip),
            "pet": (("time", "lat", "lon"), pet),
            "wb": (("time", "lat", "lon"), precip - pet),
        },
        coords={
            "time": pd.date_range("1980-01-01", periods=372, freq="MS"),
            "lat": [35.0, 36.0],
            "lon": [-100.0, -99.0],
        },
    )
    for name in ds:
        ds[name].attrs["units"] = "mm"
    ds["wb"].attrs["long_name"] = "Precipitation minus PET, monthly total"
    _publish_store(tmp_path, ds)
    monkeypatch.setenv("CLIMATE_INDICES_E2E_DATA", str(tmp_path))
    return tmp_path, ds


def test_pipeline_config_matches_canonical_contract():
    """Guards against reintroducing the stale 1990/1991-2020 configuration."""
    assert _extract_pipeline_config_years("\n".join(_code_cells())) == CANONICAL_YEARS


def test_notebook_uses_public_xarray_api():
    """Pins the #825 canonical path: public typed API, no hand-rolled map_blocks."""
    source = "\n".join(_code_cells())
    assert "map_blocks" not in source
    tree = ast.parse(source)
    public_imports = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "climate_indices"
        for alias in node.names
        if alias.name in {"spi", "spei"}
    }
    assert public_imports == {"spi", "spei"}


def test_notebook_pipeline_end_to_end(e2e_data):
    data_root, ds = e2e_data
    namespace: dict = {}
    _exec_cells(namespace, _executable_cells())

    kwargs = {
        "scale": 3,
        "data_start_year": 1980,
        "calibration_year_initial": 1981,
        "calibration_year_final": 2010,
        "periodicity": compute.Periodicity.monthly,
    }
    expected_spi = indices.spi(ds.precip[:, 0, 0].values.copy(), distribution=indices.Distribution.gamma, **kwargs)
    expected_spei = indices.spei(
        ds.precip[:, 0, 0].values.copy(),
        ds.pet[:, 0, 0].values.copy(),
        distribution=indices.Distribution.pearson,
        **kwargs,
    )
    with xr.open_zarr(data_root / "climate_indices_output.zarr", consolidated=True) as actual:
        np.testing.assert_allclose(actual.spi_3[:, 0, 0], expected_spi, atol=1e-6)
        np.testing.assert_allclose(actual.spei_3[:, 0, 0], expected_spei, atol=1e-6)
        assert actual.spi_3[:, 0, 1].isnull().all()
        assert actual.spei_3[:, 0, 1].isnull().all()
        # scale - 1 leading values are unavailable at the 3-month Timescale.
        assert actual.spi_3[:2].isnull().all()
        assert actual.spei_3[:2].isnull().all()
        # Meaningful zero precipitation (index 100, land pixel) stays a real value, not NaN.
        assert np.isfinite(actual.spi_3[100, 0, 0])
        assert np.isfinite(actual.spei_3[100, 0, 0])
        xr.testing.assert_equal(actual.time, ds.time)
        # The typed API computes in float64; the on-disk store must stay float32
        # (matching the float32 mm inputs) rather than silently doubling in size.
        assert actual.spi_3.dtype == np.float32
        assert actual.spei_3.dtype == np.float32


def test_notebook_missing_prepared_inputs_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIMATE_INDICES_E2E_DATA", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="prepare_e2e_inputs"):
        _exec_cells({}, _executable_cells())


def test_notebook_missing_manifest_is_actionable(e2e_data):
    data_root, _ = e2e_data
    (data_root / "current" / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="prepare_e2e_inputs"):
        _exec_cells({}, _executable_cells())


def test_notebook_rejects_stale_data_start_year(e2e_data):
    namespace: dict = {}
    setup, calculation = _split_at_calculation(_executable_cells())
    _exec_cells(namespace, setup)
    namespace["pipeline_config"].update({"data_start_year": 1990, "cal_start_year": 1990, "cal_end_year": 1991})
    with pytest.raises(exceptions.InvalidArgumentError):
        _exec_cells(namespace, calculation)


def test_notebook_rejects_calibration_outside_data_range(e2e_data):
    namespace: dict = {}
    setup, calculation = _split_at_calculation(_executable_cells())
    _exec_cells(namespace, setup)
    namespace["pipeline_config"].update({"cal_end_year": 2020})
    with pytest.raises(exceptions.InvalidArgumentError):
        _exec_cells(namespace, calculation)


def test_notebook_rejects_discontinuous_prepared_store(tmp_path, monkeypatch):
    pytest.importorskip("zarr")
    times = pd.date_range("1980-01-01", periods=372, freq="MS").delete(6).append(pd.DatetimeIndex(["2010-12-01"]))
    ds = xr.Dataset(
        {
            "precip": (("time", "lat", "lon"), np.ones((372, 1, 1), dtype="float32")),
            "pet": (("time", "lat", "lon"), np.ones((372, 1, 1), dtype="float32")),
        },
        coords={"time": times, "lat": [35.0], "lon": [-100.0]},
    )
    for name in ds:
        ds[name].attrs["units"] = "mm"
    _publish_store(tmp_path, ds)
    monkeypatch.setenv("CLIMATE_INDICES_E2E_DATA", str(tmp_path))
    with pytest.raises(exceptions.CoordinateValidationError):
        _exec_cells({}, _executable_cells())


def test_failed_run_preserves_existing_output(e2e_data, monkeypatch):
    data_root, _ = e2e_data
    output = data_root / "climate_indices_output.zarr"
    sentinel = xr.Dataset({"old": (("x",), [1.0])})
    sentinel.to_zarr(output, mode="w", zarr_format=2)

    namespace: dict = {}
    sources = _executable_cells()
    write_index = next(i for i, source in enumerate(sources) if "to_zarr" in source)
    _exec_cells(namespace, sources[:write_index])

    def boom(*args, **kwargs):
        raise RuntimeError("simulated block failure")

    monkeypatch.setattr(xr.Dataset, "to_zarr", boom)
    with pytest.raises(RuntimeError, match="simulated block failure"):
        _exec_cells(namespace, sources[write_index : write_index + 1])
    monkeypatch.undo()
    with xr.open_zarr(output) as actual:
        xr.testing.assert_equal(actual, sentinel)
