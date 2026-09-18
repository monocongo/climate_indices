"""Tests for CLI output chunk-size handling in climate_indices.__main__."""

import numpy as np
import pytest
import xarray as xr

from climate_indices import __main__ as cli_main
from climate_indices import compute, indices
from climate_indices.__main__ import InputType


def test_spei_chunksizes_follow_output_dimension_order(monkeypatch, tmp_path):
    """
    Chunks copied from the first chunked variable must be reordered by
    dimension name before being applied to the precipitation output dims.
    """
    time = xr.date_range("1990-01-01", periods=24, freq="MS")
    dataset = xr.Dataset(
        {
            # a chunked, time-major variable listed first forces the copied
            # tuple into an order that differs from the precip output dims
            "precip_qc": (("time", "division"), np.ones((24, 1))),
            "precip": (("division", "time"), np.ones((1, 24)), {"units": "mm"}),
            "pet": (("division", "time"), np.ones((1, 24)), {"units": "mm"}),
            "lat": (("division",), [35.0]),
        },
        coords={"division": ["0101"], "time": time},
    )
    dataset["precip_qc"].encoding.update(contiguous=False, chunksizes=(12, 1))
    dataset["precip"].encoding.update(contiguous=False, chunksizes=(1, 12))
    dataset["pet"].encoding.update(contiguous=False, chunksizes=(1, 12))

    monkeypatch.setattr(cli_main, "_global_shared_arrays", {})
    monkeypatch.setattr(cli_main.xr, "open_mfdataset", lambda *_args, **_kwargs: dataset)
    monkeypatch.setattr(cli_main, "_parallel_process", lambda *_args, **_kwargs: None)

    cli_main._compute_write_index(
        cli_main._IndexRequest(
            index="spei",
            netcdf_precip="precip.nc",
            var_name_precip="precip",
            netcdf_pet="pet.nc",
            var_name_pet="pet",
            input_type=InputType.divisions,
            periodicity=compute.Periodicity.monthly,
            chunksizes="input",
            output_file_base=str(tmp_path / "out"),
            scale=3,
            distribution=indices.Distribution.gamma,
            calibration_start_year=1990,
            calibration_end_year=1991,
        )
    )

    with xr.open_dataset(tmp_path / "out_spei_gamma_03.nc") as written:
        variable = written["spei_gamma_03"]
        assert variable.dims == ("division", "time")
        # (12, 1) in the source variable's (time, division) order would be
        # invalid for this (1, 24) output; the copied tuple must be reordered
        assert variable.encoding["chunksizes"] == (1, 12)


@pytest.mark.parametrize(
    ("units", "raw_values"),
    [
        ("fahrenheit", [32.0, 212.0]),
        ("kelvin", [273.15, 373.15]),
    ],
)
def test_normalize_temperature_units_converts_to_celsius(units, raw_values):
    """Fahrenheit and Kelvin inputs must both convert to Celsius in place."""
    dataset = xr.Dataset({"temp": ("time", np.array(raw_values), {"units": units})})

    cli_main._normalize_temperature_units(dataset, "temp")

    np.testing.assert_allclose(dataset["temp"].values, [0.0, 100.0])
