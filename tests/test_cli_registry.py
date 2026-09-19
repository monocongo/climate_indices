"""Tests for the CLI's per-index registrations in ``climate_indices.__main__``.

Each ``--index`` value maps onto a pipeline of registered indices, and each
registered index declares its own validation, kernel, and output. These tests
pin that wiring: the registrations the pipelines name, the order an aggregate
runs them in, and that a runner reaches the registrations directly rather than
through monkeypatched dispatch.
"""

import argparse

import numpy as np
import pytest
import xarray as xr

from climate_indices import __main__ as cli_main
from climate_indices import compute, indices
from climate_indices.__main__ import DatasetLayout

# the --index values the CLI accepts, in the order they are offered
_EXPECTED_INDEX_CHOICES = ("spi", "spei", "pnp", "scaled", "pet", "palmers", "kbdi", "all")

# the indices that compute through the shared-memory arrays, with one kernel each
_SHARED_ARRAY_INDICES = ("spi", "spei", "pnp", "pet", "palmers")


def test_pipelines_cover_the_index_choices():
    assert tuple(cli_main._INDEX_PIPELINES) == _EXPECTED_INDEX_CHOICES


@pytest.mark.parametrize(
    ("layout", "accepted"),
    [
        # a grid variable has to be time-last: the shared-array transport copies
        # storage order and the kernels index a grid's time axis last, so any
        # other order would be standardized along the wrong axis
        (DatasetLayout.GRID, (("lat", "lon", "time"), ("lat", "lon"))),
        # a division variable likewise has to be time-last: the transport
        # copies storage order and the kernels index a division's time axis at
        # position 1, so a time-major store is standardized along the wrong one
        (DatasetLayout.DIVISIONS, (("division", "time"), ("division",))),
        (DatasetLayout.TIMESERIES, (("time",),)),
    ],
)
def test_accepted_dimensions_are_the_orders_the_transport_reads(layout, accepted):
    """The shared-array gate accepts exactly the orders its kernels can index."""
    assert cli_main._accepted_dimensions(layout) == accepted


def test_registry_covers_every_pipeline_member():
    for pipeline, names in cli_main._INDEX_PIPELINES.items():
        for name in names:
            assert name in cli_main._INDEX_REGISTRY, f"{pipeline} runs an unregistered index: {name}"
            assert cli_main._INDEX_REGISTRY[name].index == name


def test_consumers_of_computed_pet_run_it_first():
    # each pipeline that accepts a temperature input in place of a PET file
    # computes PET before the index that consumes it
    assert cli_main._INDEX_PIPELINES["spei"] == ("pet", "spei")
    assert cli_main._INDEX_PIPELINES["scaled"] == ("spi", "pet", "spei", "pnp")
    assert cli_main._INDEX_PIPELINES["palmers"] == ("pet", "palmers")
    assert cli_main._INDEX_PIPELINES["all"] == ("spi", "pet", "spei", "pnp", "palmers")


@pytest.mark.parametrize("index", ("spi", "spei", "pnp", "pet", "palmers", "kbdi"))
def test_each_registration_declares_its_runner(index):
    registration = cli_main._registry_for(index)

    assert registration.run.__name__ == f"_run_{index}"
    assert registration.input_paths


@pytest.mark.parametrize("index", _SHARED_ARRAY_INDICES)
def test_shared_array_indices_declare_their_kernel(index):
    registration = cli_main._registry_for(index)

    assert registration.kernel is not None
    assert registration.worker is not None
    assert registration.input_array_keys is not None
    assert registration.build_arguments is not None
    assert registration.compute is not None
    assert registration.write is not None


def test_kbdi_computes_through_xarray_only():
    registration = cli_main._registry_for("kbdi")

    assert registration.kernel is None
    assert registration.compute is None
    assert registration.write is None
    assert registration.run is cli_main._run_kbdi


def test_handlers_for_index_returns_registrations_in_pipeline_order():
    handlers = cli_main._handlers_for_index("scaled")

    assert [handler.index for handler in handlers] == ["spi", "pet", "spei", "pnp"]
    assert handlers[0] is cli_main._registry_for("spi")


def test_unsupported_index_is_rejected():
    with pytest.raises(ValueError) as error:
        cli_main._handlers_for_index("bogus")

    assert str(error.value) == "Unsupported index: 'bogus'"


def test_process_climate_indices_runs_the_pipeline_in_order(monkeypatch):
    """An --index value runs its registered indices, rather than a chain of ifs."""
    calls: list[tuple[str, DatasetLayout]] = []

    def _recorder(name: str) -> cli_main._IndexRegistration:
        def _run(arguments: argparse.Namespace, input_type: DatasetLayout) -> None:
            calls.append((name, input_type))

        return cli_main._IndexRegistration(index=name, run=_run)

    monkeypatch.setattr(
        cli_main,
        "_INDEX_REGISTRY",
        {name: _recorder(name) for name in cli_main._INDEX_REGISTRY},
    )
    monkeypatch.setattr(cli_main, "_validate_args", lambda _arguments: DatasetLayout.TIMESERIES)

    cli_main.process_climate_indices(argparse.Namespace(index="all", multiprocessing="single"))

    assert calls == [(name, DatasetLayout.TIMESERIES) for name in ("spi", "pet", "spei", "pnp", "palmers")]


def test_requests_carry_only_the_inputs_their_index_declares():
    """An index's request never carries companion inputs it does not read."""
    arguments = argparse.Namespace(
        output_file_base="out",
        periodicity=compute.Periodicity.monthly,
        chunksizes="input",
        calibration_start_year=1980,
        calibration_end_year=2010,
        netcdf_precip="precip.nc",
        var_name_precip="precip",
        netcdf_temp="temp.nc",
        var_name_temp="temp",
        netcdf_pet="pet.nc",
        var_name_pet="pet",
        netcdf_awc="awc.nc",
        var_name_awc="awc",
    )

    request = cli_main._IndexRequest.from_arguments(arguments, index="spi", input_type=DatasetLayout.DIVISIONS)

    assert (request.netcdf_precip, request.var_name_precip) == ("precip.nc", "precip")
    assert (request.netcdf_temp, request.var_name_temp) == (None, None)
    assert (request.netcdf_pet, request.var_name_pet) == (None, None)
    assert (request.netcdf_awc, request.var_name_awc) == (None, None)


def test_temperature_only_spei_computes_and_consumes_pet(monkeypatch):
    """A temperature-only SPEI run computes PET and feeds its output to SPEI."""
    requests: list[cli_main._IndexRequest] = []

    def _record(request: cli_main._IndexRequest) -> tuple[str, str]:
        requests.append(request)
        return ("out_pet.nc", "pet")

    monkeypatch.setattr(cli_main, "_compute_write_index", _record)
    monkeypatch.setattr(cli_main, "_validate_args", lambda _arguments: DatasetLayout.TIMESERIES)
    arguments = argparse.Namespace(
        index="spei",
        multiprocessing="single",
        periodicity=compute.Periodicity.monthly,
        chunksizes="input",
        scales=[1],
        calibration_start_year=None,
        calibration_end_year=None,
        netcdf_precip="precip.nc",
        var_name_precip="precip",
        netcdf_temp="temp.nc",
        var_name_temp="temp",
        netcdf_pet=None,
        var_name_pet=None,
        output_file_base="out",
    )

    cli_main.process_climate_indices(arguments)

    assert [request.index for request in requests] == ["pet"] + ["spei"] * len(indices.Distribution)
    assert requests[0].var_name_precip is None
    assert all(request.netcdf_pet == "out_pet.nc" for request in requests[1:])
    assert arguments.netcdf_pet == "out_pet.nc"


def test_pet_index_computes_pet_when_a_pet_file_is_also_provided(monkeypatch):
    """--index pet computes PET from temperature rather than skipping the run."""
    requests: list[cli_main._IndexRequest] = []

    def _record(request: cli_main._IndexRequest) -> tuple[str, str]:
        requests.append(request)
        return ("out_pet.nc", "pet")

    monkeypatch.setattr(cli_main, "_compute_write_index", _record)
    monkeypatch.setattr(cli_main, "_validate_args", lambda _arguments: DatasetLayout.TIMESERIES)
    arguments = argparse.Namespace(
        index="pet",
        multiprocessing="single",
        periodicity=compute.Periodicity.monthly,
        chunksizes="input",
        calibration_start_year=None,
        calibration_end_year=None,
        netcdf_temp="temp.nc",
        var_name_temp="temp",
        netcdf_pet="provided_pet.nc",
        var_name_pet="pet",
        output_file_base="out",
    )

    cli_main.process_climate_indices(arguments)

    assert [request.index for request in requests] == ["pet"]
    assert arguments.netcdf_pet == "out_pet.nc"


def test_temperature_derived_pet_requires_monthly_periodicity():
    """A daily temperature input is rejected rather than fed to Thornthwaite PET."""
    arguments = argparse.Namespace(
        periodicity=compute.Periodicity.daily,
        netcdf_temp="temp.nc",
        var_name_temp="temp",
        netcdf_pet=None,
        var_name_pet=None,
    )
    context = cli_main._InputContext(
        input_type=DatasetLayout.TIMESERIES,
        dimensions=("time",),
        times=np.array([0]),
    )

    with pytest.raises(ValueError) as error:
        cli_main._validate_pet_or_temperature_input(arguments, context)

    assert str(error.value) == "Invalid periodicity argument for PET: 'daily' -- only 'monthly' is supported"


def test_result_array_is_reallocated_when_the_output_shape_changes(monkeypatch):
    """A reused result buffer is reallocated when an index's output shape differs."""
    monkeypatch.setattr(cli_main, "_global_shared_arrays", {})
    monkeypatch.setattr(cli_main, "_parallel_process", lambda *_args, **_kwargs: None)
    cli_main._allocate_shared_array(cli_main._KEY_RESULT, (1, 12))
    request = cli_main._IndexRequest(
        index="spi",
        output_file_base="out",
        input_type=DatasetLayout.DIVISIONS,
        periodicity=compute.Periodicity.monthly,
        chunksizes="none",
    )
    context = cli_main._ComputeContext(
        request=request,
        dataset=xr.Dataset(),
        output_dims=("time", "division"),
        output_shape=(12, 1),
        output_encodings=None,
        output_engine=None,
        arguments={},
    )

    cli_main._compute_single_array(context)

    assert cli_main._global_shared_arrays[cli_main._KEY_RESULT][cli_main._KEY_SHAPE] == (12, 1)


def test_aggregate_index_requires_the_scales_its_members_need(monkeypatch):
    """`all` includes scaled indices, so it requires --scales like they do."""
    time = xr.date_range("1990-01-01", periods=12, freq="MS")
    coords = {"division": ["0101"], "time": time}
    datasets = {
        "precip.nc": xr.Dataset({"precip": (("division", "time"), np.ones((1, 12)))}, coords=coords),
        "pet.nc": xr.Dataset({"pet": (("division", "time"), np.ones((1, 12)))}, coords=coords),
        "awc.nc": xr.Dataset({"awc": (("division",), [4.0])}, coords={"division": ["0101"]}),
    }
    monkeypatch.setattr(cli_main.xr, "open_dataset", datasets.__getitem__)
    arguments = argparse.Namespace(
        index="all",
        periodicity=compute.Periodicity.monthly,
        scales=None,
        calibration_start_year=1990,
        calibration_end_year=1991,
        netcdf_precip="precip.nc",
        var_name_precip="precip",
        netcdf_temp=None,
        var_name_temp=None,
        netcdf_pet="pet.nc",
        var_name_pet="pet",
        netcdf_awc="awc.nc",
        var_name_awc="awc",
    )

    with pytest.raises(ValueError) as error:
        cli_main._validate_args(arguments)

    assert str(error.value) == (
        "Scaled indices (SPI, SPEI, and/or PNP) specified without "
        "including one or more time scales (missing --scales argument)"
    )
