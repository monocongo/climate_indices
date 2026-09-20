"""Tests for CLI output chunk-size handling in climate_indices.__main__."""

import numpy as np
import pytest
import xarray as xr

from climate_indices import __main__ as cli_main
from climate_indices import compute, indices
from climate_indices.__main__ import DatasetLayout


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
            input_type=DatasetLayout.DIVISIONS,
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


def test_pnp_copies_h5netcdf_input_chunksizes(monkeypatch, tmp_path):
    """``--chunksizes input`` preserves chunks reported without ``contiguous``."""
    time = xr.date_range("1990-01-01", periods=24, freq="MS")
    dataset = xr.Dataset(
        {"prcp": (("lat", "lon", "time"), np.ones((2, 3, 24)), {"units": "mm"})},
        coords={"lat": [25.0, 30.0], "lon": [-100.0, -95.0, -90.0], "time": time},
    )
    input_file = tmp_path / "prcp.nc"
    dataset.to_netcdf(input_file, encoding={"prcp": {"chunksizes": (2, 3, 12)}}, engine="h5netcdf")

    # h5netcdf reports on-disk chunks without a `contiguous` key, which is the
    # condition the copied chunk sizes have to survive
    with xr.open_dataset(input_file, engine="h5netcdf") as opened:
        assert "contiguous" not in opened["prcp"].encoding

    monkeypatch.setattr(cli_main, "_global_shared_arrays", {})
    monkeypatch.setattr(cli_main, "_parallel_process", lambda *_args, **_kwargs: None)

    cli_main._compute_write_index(
        cli_main._IndexRequest(
            index="pnp",
            netcdf_precip=str(input_file),
            var_name_precip="prcp",
            input_type=DatasetLayout.GRID,
            periodicity=compute.Periodicity.monthly,
            chunksizes="input",
            output_file_base=str(tmp_path / "out"),
            scale=3,
            calibration_start_year=1990,
            calibration_end_year=1991,
        )
    )

    with xr.open_dataset(tmp_path / "out_pnp_03.nc", engine="h5netcdf") as written:
        assert written["pnp_03"].encoding["chunksizes"] == (2, 3, 12)


def test_input_chunksizes_ignores_contiguous_inputs(tmp_path):
    """Contiguous inputs report no chunks to copy, even without a ``contiguous`` key."""
    time = xr.date_range("1990-01-01", periods=24, freq="MS")
    dataset = xr.Dataset(
        {"prcp": (("lat", "lon", "time"), np.ones((2, 3, 24)), {"units": "mm"})},
        coords={"lat": [25.0, 30.0], "lon": [-100.0, -95.0, -90.0], "time": time},
    )
    input_file = tmp_path / "contiguous.nc"
    dataset.to_netcdf(input_file, engine="h5netcdf")

    with xr.open_mfdataset(input_file) as opened:
        assert "contiguous" not in opened["prcp"].encoding
        assert cli_main._input_chunksizes(opened) == ((), ())

    # a backend that does report the key as True must be honored as well
    dataset["prcp"].encoding.update(contiguous=True, chunksizes=(2, 3, 12))
    assert cli_main._input_chunksizes(dataset) == ((), ())


def test_oversized_input_chunks_are_trimmed_to_the_output_shape(monkeypatch, tmp_path):
    """A chunk larger than the output dimension must not be written verbatim."""
    time = xr.date_range("1990-01-01", periods=24, freq="MS")
    dataset = xr.Dataset(
        {"prcp": (("lat", "lon", "time"), np.ones((2, 3, 24)), {"units": "mm"})},
        coords={"lat": [25.0, 30.0], "lon": [-100.0, -95.0, -90.0], "time": time},
    )
    input_file = tmp_path / "prcp.nc"
    # an unlimited time dimension permits a chunk larger than the data written so far
    dataset.to_netcdf(
        input_file,
        encoding={"prcp": {"chunksizes": (2, 3, 100)}},
        engine="h5netcdf",
        unlimited_dims=["time"],
    )

    monkeypatch.setattr(cli_main, "_global_shared_arrays", {})
    monkeypatch.setattr(cli_main, "_parallel_process", lambda *_args, **_kwargs: None)

    cli_main._compute_write_index(
        cli_main._IndexRequest(
            index="pnp",
            netcdf_precip=str(input_file),
            var_name_precip="prcp",
            input_type=DatasetLayout.GRID,
            periodicity=compute.Periodicity.monthly,
            chunksizes="input",
            output_file_base=str(tmp_path / "out"),
            scale=3,
            calibration_start_year=1990,
            calibration_end_year=1991,
        )
    )

    with xr.open_dataset(tmp_path / "out_pnp_03.nc", engine="h5netcdf") as written:
        assert written["pnp_03"].encoding["chunksizes"] == (2, 3, 24)


def test_daily_oversized_input_chunks_are_trimmed_to_the_written_shape(monkeypatch, tmp_path):
    """The daily write path shrinks 366-day years back to the Gregorian calendar."""
    time = xr.date_range("1990-01-01", periods=365, freq="D")
    dataset = xr.Dataset(
        {"prcp": (("lat", "lon", "time"), np.ones((2, 3, 365)) * 10.0, {"units": "mm"})},
        coords={"lat": [25.0, 30.0], "lon": [-100.0, -95.0, -90.0], "time": time},
    )
    input_file = tmp_path / "prcp.nc"
    dataset.to_netcdf(
        input_file,
        encoding={"prcp": {"chunksizes": (2, 3, 366)}},
        engine="h5netcdf",
        unlimited_dims=["time"],
    )

    monkeypatch.setattr(cli_main, "_global_shared_arrays", {})
    monkeypatch.setattr(cli_main, "_parallel_process", lambda *_args, **_kwargs: None)

    cli_main._compute_write_index(
        cli_main._IndexRequest(
            index="spi",
            netcdf_precip=str(input_file),
            var_name_precip="prcp",
            input_type=DatasetLayout.GRID,
            periodicity=compute.Periodicity.daily,
            chunksizes="input",
            output_file_base=str(tmp_path / "out"),
            scale=30,
            distribution=indices.Distribution.gamma,
            calibration_start_year=1990,
            calibration_end_year=1990,
        )
    )

    with xr.open_dataset(tmp_path / "out_spi_gamma_30.nc", engine="h5netcdf") as written:
        assert written["spi_gamma_30"].shape == (2, 3, 365)
        assert written["spi_gamma_30"].encoding["chunksizes"] == (2, 3, 365)


def test_input_files_are_opened_in_the_requested_order(monkeypatch, tmp_path):
    """De-duplicating the input file list must not reorder the files."""
    time = xr.date_range("1990-01-01", periods=24, freq="MS")
    coords = {"lat": [25.0, 30.0], "lon": [-100.0, -95.0, -90.0], "time": time}
    precip_file = tmp_path / "prcp.nc"
    pet_file = tmp_path / "pet.nc"
    xr.Dataset({"prcp": (("lat", "lon", "time"), np.ones((2, 3, 24)), {"units": "mm"})}, coords=coords).to_netcdf(
        precip_file
    )
    xr.Dataset({"pet": (("lat", "lon", "time"), np.ones((2, 3, 24)), {"units": "mm"})}, coords=coords).to_netcdf(
        pet_file
    )

    opened_file_lists: list[list[str]] = []
    original_open_mfdataset = cli_main.xr.open_mfdataset

    def _recording_open_mfdataset(files, **kwargs):
        opened_file_lists.append(list(files))
        return original_open_mfdataset(files, **kwargs)

    monkeypatch.setattr(cli_main, "_global_shared_arrays", {})
    monkeypatch.setattr(cli_main.xr, "open_mfdataset", _recording_open_mfdataset)
    monkeypatch.setattr(cli_main, "_parallel_process", lambda *_args, **_kwargs: None)

    cli_main._compute_write_index(
        cli_main._IndexRequest(
            index="spei",
            netcdf_precip=str(precip_file),
            var_name_precip="prcp",
            netcdf_pet=str(pet_file),
            var_name_pet="pet",
            input_type=DatasetLayout.GRID,
            periodicity=compute.Periodicity.monthly,
            chunksizes="none",
            output_file_base=str(tmp_path / "out"),
            scale=3,
            distribution=indices.Distribution.gamma,
            calibration_start_year=1990,
            calibration_end_year=1991,
        )
    )

    assert opened_file_lists == [[str(precip_file), str(pet_file)]]


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
