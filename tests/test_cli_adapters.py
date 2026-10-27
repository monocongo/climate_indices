"""Exercise CLI adapters without replacing the dispatch or computation path."""

import argparse
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import xarray as xr

from climate_indices import __main__ as cli
from climate_indices import compute, fire, flood, indices
from climate_indices._cli_transport import Transport
from climate_indices.validation import DatasetLayout


def test_shared_array_adapter_requires_a_complete_implementation() -> None:
    class KernelOnly(cli._SharedArrayAdapter):
        kernel = staticmethod(lambda *args: None)

    assert {
        "compute",
        "write",
        "worker",
        "input_array_keys",
        "build_arguments",
        "run",
    } <= KernelOnly.__abstractmethods__
    with pytest.raises(TypeError, match="abstract"):
        KernelOnly()  # type: ignore[abstract]


@pytest.mark.parametrize("name", tuple(cli._INDEX_REGISTRY))
def test_registry_contains_concrete_adapters(name: str) -> None:
    adapter = cli._registry_for(name)

    assert isinstance(adapter, cli._IndexAdapter)
    assert adapter.index == name
    assert callable(adapter.run)
    assert callable(adapter.validate_arguments)
    assert callable(adapter.validate_inputs)


def test_xarray_adapter_cannot_enter_the_shared_array_path() -> None:
    with pytest.raises(TypeError, match="does not compute from shared arrays"):
        cli._shared_array_adapter("kbdi")


def test_pnp_adapter_runs_directly_and_writes_the_expected_ratios(tmp_path: Path) -> None:
    values = np.arange(1.0, 25.0)
    source = xr.DataArray(
        values,
        dims="time",
        coords={"time": xr.date_range("2000-01-01", periods=24, freq="MS")},
        attrs={"units": "mm"},
        name="precip",
    )
    path = tmp_path / "precip.nc"
    source.to_netcdf(path)
    arguments = argparse.Namespace(
        index="pnp",
        netcdf_precip=str(path),
        var_name_precip="precip",
        output_file_base=str(tmp_path / "out"),
        periodicity=compute.Periodicity.monthly,
        chunksizes="none",
        calibration_start_year=2000,
        calibration_end_year=2001,
        scales=[1],
    )
    adapter = cli._registry_for("pnp")
    adapter.validate_arguments(arguments)
    adapter.run(arguments, DatasetLayout.TIMESERIES, Transport(1))

    # Each calendar month's two observations differ by 12; their mean is the first plus 6.
    expected = values / np.tile(np.arange(1.0, 13.0) + 6.0, 2)
    with xr.open_dataset(tmp_path / "out_pnp_01.nc") as result:
        np.testing.assert_allclose(result["pnp_01"].values, expected, rtol=1e-14, atol=0)
        assert tuple(result.data_vars) == ("pnp_01",)
        assert result["pnp_01"].attrs["units"] == "1"
        np.testing.assert_array_equal(result.time.values, source.time.values)


@pytest.fixture
def adapter_request() -> cli._IndexRequest:
    return cli._IndexRequest(
        index="palmers",
        output_file_base="out",
        input_type=DatasetLayout.TIMESERIES,
        periodicity=compute.Periodicity.monthly,
        chunksizes="none",
        netcdf_precip="precip.nc",
        var_name_precip="precip",
        netcdf_pet="pet.nc",
        var_name_pet="pet",
        netcdf_temp="temp.nc",
        var_name_temp="temp",
        netcdf_awc="awc.nc",
        var_name_awc="awc",
        netcdf_pe="pe.nc",
        var_name_pe="pe",
        scale=1,
        distribution=indices.Distribution.gamma,
    )


@pytest.mark.parametrize(
    ("index", "field", "message"),
    [
        ("spi", "var_name_precip", "precipitation variable name"),
        ("pnp", "var_name_precip", "precipitation variable name"),
        ("spei", "var_name_precip", "precipitation variable name"),
        ("spei", "var_name_pet", "PET variable name"),
        ("pet", "var_name_temp", "temperature variable name"),
        ("palmers", "var_name_precip", "precipitation variable name"),
        ("palmers", "var_name_pet", "PET variable name"),
        ("palmers", "var_name_awc", "AWC variable name"),
    ],
)
def test_shared_input_keys_reject_missing_names(
    adapter_request: cli._IndexRequest, index: str, field: str, message: str
) -> None:
    setattr(adapter_request, field, None)
    adapter = cli._shared_array_adapter(index)
    with pytest.raises(ValueError, match=message):
        adapter.input_array_keys(adapter_request)


@pytest.mark.parametrize(
    ("index", "field"),
    [("spi", "distribution"), ("spi", "scale"), ("spei", "distribution"), ("spei", "scale"), ("pnp", "scale")],
)
def test_output_attributes_reject_missing_parameters(
    adapter_request: cli._IndexRequest, index: str, field: str
) -> None:
    setattr(adapter_request, field, None)
    adapter = cli._shared_array_adapter(index)
    assert isinstance(adapter, cli._SingleOutputAdapter)
    source = xr.DataArray()
    with pytest.raises(ValueError, match=f"the {field} was not validated"):
        adapter.variable_attributes(adapter_request, source)


@pytest.mark.parametrize("method", ["prepare_inputs", "compute"])
@pytest.mark.parametrize(
    ("field", "message"),
    [("var_name_precip", "precipitation variable name"), ("var_name_pet", "PET variable name")],
)
def test_palmer_steps_reject_missing_names(
    adapter_request: cli._IndexRequest, method: str, field: str, message: str
) -> None:
    setattr(adapter_request, field, None)
    if method == "prepare_inputs":
        dataset = xr.Dataset()
        with pytest.raises(ValueError, match=message):
            cli._PalmersAdapter.prepare_inputs(adapter_request, dataset)
    else:
        context = _palmer_context(adapter_request)
        with pytest.raises(ValueError, match=message):
            cli._PalmersAdapter.compute(context)


def _palmer_context(request: cli._IndexRequest) -> cli._ComputeContext:
    return cli._ComputeContext(
        request=request,
        dataset=xr.Dataset(),
        output_dims=("time",),
        output_shape=(1,),
        output_encodings=None,
        output_engine=None,
        arguments={},
        transport=Transport(1),
    )


def test_palmer_compute_requires_awc_name(adapter_request: cli._IndexRequest) -> None:
    adapter_request.var_name_awc = None
    context = _palmer_context(adapter_request)
    with pytest.raises(ValueError, match="AWC variable name"):
        cli._PalmersAdapter.compute(context)


def test_palmer_compute_requires_prepared_awc(adapter_request: cli._IndexRequest) -> None:
    context = _palmer_context(adapter_request)
    with pytest.raises(RuntimeError, match="AWC dataset"):
        cli._PalmersAdapter.compute(context)


@pytest.mark.parametrize(
    ("index", "field", "message"),
    [
        *[
            (index, field, "precipitation input")
            for index in ("kbdi", "pe", "api")
            for field in ("netcdf_precip", "var_name_precip")
        ],
        *[("kbdi", field, "temperature input") for field in ("netcdf_temp", "var_name_temp")],
        *[(index, field, "PE input") for index in ("edi", "flood_index") for field in ("netcdf_pe", "var_name_pe")],
    ],
)
def test_xarray_runners_reject_missing_inputs(
    adapter_request: cli._IndexRequest, monkeypatch: pytest.MonkeyPatch, index: str, field: str, message: str
) -> None:
    arguments = argparse.Namespace(**vars(adapter_request))
    arguments.index = index
    setattr(arguments, field, None)
    if field == "netcdf_pe":
        # Simulate a prerequisite that did not supply its output file.
        monkeypatch.setattr(cli._PEAdapter, "run", staticmethod(lambda *_args: None))
    adapter = cli._registry_for(index)
    transport = Transport(1)
    with pytest.raises(ValueError, match=message):
        adapter.run(arguments, DatasetLayout.TIMESERIES, transport)


def test_pet_runner_requires_a_result(adapter_request: cli._IndexRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    arguments = argparse.Namespace(**vars(adapter_request))
    arguments.index = "pet"
    monkeypatch.setattr(cli, "_compute_write_index", lambda *_args: None)
    transport = Transport(1)
    with pytest.raises(RuntimeError, match="PET computation should return"):
        cli._PETAdapter.run(arguments, DatasetLayout.TIMESERIES, transport)


@pytest.mark.parametrize(
    ("index", "module", "kernel"), [("kbdi", fire, "kbdi"), ("api", flood, "antecedent_precipitation_index")]
)
def test_xarray_runners_require_dataarray_results(
    adapter_request: cli._IndexRequest, monkeypatch: pytest.MonkeyPatch, index: str, module: ModuleType, kernel: str
) -> None:
    arguments = argparse.Namespace(**vars(adapter_request), kbdi_units=None, kbdi_initial=None, api_k=0.9)
    monkeypatch.setattr(
        xr,
        "open_dataset",
        lambda *_args, **_kwargs: xr.Dataset({"precip": ("time", [1.0]), "temp": ("time", [20.0])}),
    )
    monkeypatch.setattr(module, kernel, lambda *_args, **_kwargs: np.zeros(1))
    adapter = cli._registry_for(index)
    transport = Transport(1)
    with pytest.raises(TypeError, match="must return an xarray.DataArray"):
        adapter.run(arguments, DatasetLayout.TIMESERIES, transport)


@pytest.mark.parametrize(
    ("layout", "field"),
    [(DatasetLayout.GRID, "latitudes"), (DatasetLayout.GRID, "longitudes"), (DatasetLayout.DIVISIONS, "divisions")],
)
def test_kbdi_validation_requires_coordinates(
    adapter_request: cli._IndexRequest, monkeypatch: pytest.MonkeyPatch, layout: DatasetLayout, field: str
) -> None:
    coordinates = {"lat": [40.0], "lon": [-100.0]} if layout == DatasetLayout.GRID else {"division": [1]}
    temperature = xr.Dataset(coords={**coordinates, "time": [0]})
    temperature["temp"] = xr.DataArray(np.ones([1] * len(temperature.dims)), dims=tuple(temperature.dims))
    context = cli._InputContext(
        input_type=layout,
        dimensions=temperature["temp"].dims,
        times=temperature.time.values,
        latitudes=None if field == "latitudes" else np.array([40.0]),
        longitudes=None if field == "longitudes" else np.array([-100.0]),
        divisions=None if field == "divisions" else np.array([1]),
    )
    monkeypatch.setattr(xr, "open_dataset", lambda *_args: temperature)
    arguments = argparse.Namespace(**vars(adapter_request))
    with pytest.raises(ValueError, match=field):
        cli._KBDIAdapter.validate_inputs(arguments, context)
