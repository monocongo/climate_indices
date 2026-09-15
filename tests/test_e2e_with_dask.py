"""Offline regression checks for the E2E teaching notebook.

The notebook is the authoritative artifact: these tests execute its code cells
against a tiny synthetic prepared store (located via CLIMATE_INDICES_E2E_DATA)
instead of importing a companion script.

Nothing here reaches the network. Optional backends are skipped when absent:
tests that write or reopen Zarr stores importorskip("zarr"), and the plot-cell
test importorskip("matplotlib").
"""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import __version__, compute, exceptions, indices

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "zarr_dask_spi_spei.ipynb"

# The canonical data/calibration contract (matches data/e2e/manifest.json once
# scripts/prepare_e2e_inputs.py has generated it, and the notebook).
CANONICAL_YEARS = {"data_start_year": 1980, "cal_start_year": 1981, "cal_end_year": 2010}


def _code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text())
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _executable_cells() -> list[str]:
    # The Dask Client cell (and its matching close()) is execution mechanics
    # out of scope here (#829) and stays untested; local Dask falls back to
    # the threaded scheduler without it, so calculation-path coverage is
    # unaffected. The plot cells are excluded from every test that reuses
    # this list because matplotlib isn't guaranteed present (e.g. the
    # minimum-dependency CI job); test_notebook_plot_cells_execute covers
    # them separately, gated on matplotlib being importable.
    return [
        source
        for source in _code_cells()
        if "dask.distributed" not in source and ".plot(" not in source and "client.close" not in source
    ]


def _cells_excluding_client() -> list[str]:
    return [source for source in _code_cells() if "dask.distributed" not in source and "client.close" not in source]


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
    ds = _synthetic_dataset()
    _publish_store(tmp_path, ds)
    monkeypatch.setenv("CLIMATE_INDICES_E2E_DATA", str(tmp_path))
    return tmp_path, ds


def _synthetic_dataset() -> xr.Dataset:
    """One land cell, one fully masked cell, and one meaningful zero, 1980-2010 monthly."""
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
    return ds


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


def test_notebook_spi_spei_stay_lazy_until_the_write(e2e_data):
    """The typed API results stay Dask-backed until to_zarr materializes them."""
    namespace: dict = {}
    sources = _executable_cells()
    write_index = next(index for index, source in enumerate(sources) if "to_zarr" in source)
    _exec_cells(namespace, sources[:write_index])
    assert namespace["spi_da"].chunks is not None
    assert namespace["spei_da"].chunks is not None
    assert namespace["spi_da"].dims == ("time", "lat", "lon")


def test_notebook_spei_receives_pet_not_water_balance():
    """SPEI must consume actual PET; a pre-computed water balance is not accepted."""
    tree = ast.parse("\n".join(_code_cells()))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "spei"
    ]
    assert len(calls) == 1
    arguments = {keyword.arg: ast.dump(keyword.value) for keyword in calls[0].keywords}
    assert arguments["precips_mm"] == ast.dump(ast.parse('ds_calc["precip"]', mode="eval").body)
    assert arguments["pet_mm"] == ast.dump(ast.parse('ds_calc["pet"]', mode="eval").body)


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
    with xr.open_zarr(data_root / "climate_indices_output.zarr", consolidated=True) as actual:
        # Every spatial cell, not just one representative series: valid cells
        # must match the direct stable API, all-missing cells must stay NaN.
        for lat in range(ds.sizes["lat"]):
            for lon in range(ds.sizes["lon"]):
                precip_series = ds.precip[:, lat, lon].values.copy()
                if np.isnan(precip_series).all():
                    assert actual.spi_3[:, lat, lon].isnull().all()
                    assert actual.spei_3[:, lat, lon].isnull().all()
                    continue
                expected_spi = indices.spi(precip_series, distribution=indices.Distribution.gamma, **kwargs)
                expected_spei = indices.spei(
                    precip_series,
                    ds.pet[:, lat, lon].values.copy(),
                    distribution=indices.Distribution.pearson,
                    **kwargs,
                )
                np.testing.assert_allclose(actual.spi_3[:, lat, lon], expected_spi, atol=1e-6)
                np.testing.assert_allclose(actual.spei_3[:, lat, lon], expected_spei, atol=1e-6)
        # scale - 1 leading values are unavailable at the 3-month Timescale.
        assert actual.spi_3[:2].isnull().all()
        assert actual.spei_3[:2].isnull().all()
        # Meaningful zero precipitation (index 100, land pixel) stays a real value, not NaN.
        assert np.isfinite(actual.spi_3[100, 0, 0])
        assert np.isfinite(actual.spei_3[100, 0, 0])
        # Coordinates and dimensions survive the write and reopen.
        assert dict(actual.sizes) == dict(ds.sizes)
        xr.testing.assert_equal(actual.time, ds.time)
        np.testing.assert_array_equal(actual.lat.values, ds.lat.values)
        np.testing.assert_array_equal(actual.lon.values, ds.lon.values)
        # The typed API computes in float64; the on-disk store must stay float32
        # (matching the float32 mm inputs) rather than silently doubling in size.
        assert actual.spi_3.dtype == np.float32
        assert actual.spei_3.dtype == np.float32

        # The public API's per-variable metadata must survive the write and reopen;
        # "periodicity" is added by the notebook so the Timescale unit is explicit.
        expected_metadata = {
            "spi_3": {
                "long_name": "Standardized Precipitation Index",
                "units": "dimensionless",
                "scale": 3,
                "distribution": "gamma",
                "calibration_year_initial": 1981,
                "calibration_year_final": 2010,
                "periodicity": "monthly",
                "climate_indices_version": __version__,
            },
            "spei_3": {
                "long_name": "Standardized Precipitation Evapotranspiration Index",
                "units": "dimensionless",
                "scale": 3,
                "distribution": "pearson",
                "calibration_year_initial": 1981,
                "calibration_year_final": 2010,
                "periodicity": "monthly",
                "climate_indices_version": __version__,
            },
        }
        for variable_name, expected_attrs in expected_metadata.items():
            attrs = actual[variable_name].attrs
            for key, value in expected_attrs.items():
                assert attrs.get(key) == value, f"{variable_name}.{key}"
            # CF defines no drought-index standard_name, so precipitation's
            # inherited precipitation_amount must not describe the result.
            assert "standard_name" not in attrs
            assert variable_name.split("_")[0].upper() in attrs["history"]
            assert __version__ in attrs["history"]
        assert "McKee" in actual.spi_3.attrs["references"]
        assert "Vicente-Serrano" in actual.spei_3.attrs["references"]


def test_notebook_reopens_saved_output_with_a_fresh_lazy_handle():
    """Diagnostics must select from the saved store, not the calculated dataset."""
    source = "\n".join(_code_cells())
    tree = ast.parse(source)
    out_ds_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "out_ds" for target in node.targets)
    )
    expected_open = ast.parse("xr.open_zarr(final_output_zarr, consolidated=True)", mode="eval").body
    assert ast.dump(out_ds_assignment.value) == ast.dump(expected_open)
    diagnostic_start = next(
        node.lineno
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "spi_index" for target in node.targets)
    )
    diagnostic_sources = {
        target.id: node.value.value.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.value, ast.Name)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in {"spi_index", "spei_index"}
    }
    assert diagnostic_sources == {"spi_index": "out_ds", "spei_index": "out_ds"}
    assert not any(
        isinstance(node, ast.Name) and node.id == "ds_output" and node.lineno >= diagnostic_start
        for node in ast.walk(tree)
    )
    assert "out_ds.close()" in source


def test_reopened_output_is_lazy_and_independent_of_prepared_inputs(e2e_data):
    """Reopened results stay lazy and self-contained after the inputs are gone."""
    data_root, _ = e2e_data
    _exec_cells({}, _executable_cells())
    (data_root / "current").unlink()
    with xr.open_zarr(data_root / "climate_indices_output.zarr", consolidated=True) as reopened:
        assert reopened["spi_3"].chunks is not None
        assert reopened["spei_3"].chunks is not None
        selected = reopened["spi_3"].isel(time=-1, lat=0, lon=0).compute()
        assert np.isfinite(selected.item())


def test_rerun_replaces_previous_output(e2e_data):
    """A successful rerun replaces the previous completed output store."""
    data_root, _ = e2e_data
    output = data_root / "climate_indices_output.zarr"
    xr.Dataset({"stale": (("x",), [1.0])}).to_zarr(output, mode="w", zarr_format=2)
    _exec_cells({}, _executable_cells())
    with xr.open_zarr(output, consolidated=True) as actual:
        assert set(actual.data_vars) == {"spi_3", "spei_3"}


def test_notebook_rejects_overlapping_output_and_input(e2e_data):
    """Overlapping input/output/staging paths fail before anything is written."""
    data_root, _ = e2e_data
    output = data_root / "climate_indices_output.zarr"
    output.symlink_to(data_root / "current" / "cache_prepared_input.zarr", target_is_directory=True)
    with pytest.raises(exceptions.InvalidArgumentError, match="must not overlap"):
        _exec_cells({}, _executable_cells())
    assert output.is_symlink()


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


def test_notebook_rejects_partially_missing_cells(tmp_path, monkeypatch):
    pytest.importorskip("zarr")
    ds = _synthetic_dataset()
    ds["precip"].values[100, 0, 0] = np.nan
    ds["pet"].values[100, 0, 0] = np.nan
    _publish_store(tmp_path, ds)
    monkeypatch.setenv("CLIMATE_INDICES_E2E_DATA", str(tmp_path))
    with pytest.raises(exceptions.InvalidArgumentError):
        _exec_cells({}, _executable_cells())


def test_notebook_rejects_mismatched_missingness(tmp_path, monkeypatch):
    pytest.importorskip("zarr")
    ds = _synthetic_dataset()
    ds["pet"].values[100, 0, 0] = np.nan  # PET missing where precip is present
    _publish_store(tmp_path, ds)
    monkeypatch.setenv("CLIMATE_INDICES_E2E_DATA", str(tmp_path))
    with pytest.raises(exceptions.InvalidArgumentError):
        _exec_cells({}, _executable_cells())


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
    assert not list(data_root.glob(f".{output.name}.staging-*"))


def test_interrupted_publish_restores_previous_output(e2e_data, monkeypatch):
    """A failed publish rename restores the previous output at its final path."""
    data_root, _ = e2e_data
    output = data_root / "climate_indices_output.zarr"
    sentinel = xr.Dataset({"old": (("x",), [1.0])})
    sentinel.to_zarr(output, mode="w", zarr_format=2)

    namespace: dict = {}
    sources = _executable_cells()
    write_index = next(i for i, source in enumerate(sources) if "to_zarr" in source)
    _exec_cells(namespace, sources[:write_index])

    original_rename = Path.rename

    def flaky_rename(self, target):
        if ".staging-" in self.name:
            raise RuntimeError("simulated crash during publish")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)
    with pytest.raises(RuntimeError, match="simulated crash during publish"):
        _exec_cells(namespace, sources[write_index : write_index + 1])
    monkeypatch.undo()

    backup = data_root / f".{output.name}.previous"
    assert output.exists(), "previous generation was not restored after an interrupted publish"
    assert not backup.exists()
    assert not list(data_root.glob(f".{output.name}.staging-*"))
    with xr.open_zarr(output) as actual:
        xr.testing.assert_equal(actual, sentinel)


def test_notebook_plot_cells_execute(e2e_data):
    """Guards the plot cells against a renamed variable or removed argument."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    namespace: dict = {}
    _exec_cells(namespace, _cells_excluding_client())
