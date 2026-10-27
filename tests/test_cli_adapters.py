"""Exercise CLI adapters without replacing the dispatch or computation path."""

import argparse
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from climate_indices import __main__ as cli
from climate_indices import compute
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
