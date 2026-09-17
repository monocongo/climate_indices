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
from climate_indices import compute
from climate_indices.__main__ import InputType

# the --index values the CLI accepts, in the order they are offered
_EXPECTED_INDEX_CHOICES = ("spi", "spei", "pnp", "scaled", "pet", "palmers", "kbdi", "all")

# the indices that compute through the shared-memory arrays, with one kernel each
_SHARED_ARRAY_INDICES = ("spi", "spei", "pnp", "pet", "palmers")


def test_pipelines_cover_the_index_choices():
    assert tuple(cli_main._INDEX_PIPELINES) == _EXPECTED_INDEX_CHOICES


def test_registry_covers_every_pipeline_member():
    for pipeline, names in cli_main._INDEX_PIPELINES.items():
        for name in names:
            assert name in cli_main._INDEX_REGISTRY, f"{pipeline} runs an unregistered index: {name}"
            assert cli_main._INDEX_REGISTRY[name].index == name


def test_all_runs_pet_before_the_indices_that_consume_it():
    assert cli_main._INDEX_PIPELINES["all"] == ("spi", "pet", "spei", "pnp", "palmers")


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

    assert [handler.index for handler in handlers] == ["spi", "spei", "pnp"]
    assert handlers[0] is cli_main._registry_for("spi")


def test_unsupported_index_is_rejected():
    with pytest.raises(ValueError) as error:
        cli_main._handlers_for_index("bogus")

    assert str(error.value) == "Unsupported index: 'bogus'"


def test_process_climate_indices_runs_the_pipeline_in_order(monkeypatch):
    """An --index value runs its registered indices, rather than a chain of ifs."""
    calls: list[tuple[str, InputType]] = []

    def _recorder(name: str) -> cli_main._IndexRegistration:
        def _run(arguments: argparse.Namespace, input_type: InputType) -> None:
            calls.append((name, input_type))

        return cli_main._IndexRegistration(index=name, run=_run)

    monkeypatch.setattr(
        cli_main,
        "_INDEX_REGISTRY",
        {name: _recorder(name) for name in cli_main._INDEX_REGISTRY},
    )
    monkeypatch.setattr(cli_main, "_validate_args", lambda _arguments: InputType.timeseries)

    cli_main.process_climate_indices(argparse.Namespace(index="all", multiprocessing="single"))

    assert calls == [(name, InputType.timeseries) for name in ("spi", "pet", "spei", "pnp", "palmers")]


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
