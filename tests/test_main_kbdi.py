"""Tests for the CLI's KBDI dispatch path in climate_indices.__main__.

KBDI must not route through ``_compute_write_index()``: the shared daily path
reshapes inputs into 366-day years and coerces precipitation to mm and
temperature to Celsius, all of which corrupt the KBDI recurrence and its
``units="imperial"`` mode. The CLI therefore dispatches ``--index kbdi``
through the fire module's xarray API in an isolated branch.
"""

import argparse

import numpy as np
import pytest
import xarray as xr

from climate_indices import __main__ as cli_main
from climate_indices import _cli, compute

# KBDI derives mean annual precipitation from at least 30 * 365 days of record
_DAILY_PERIODS = 11000


@pytest.fixture
def kbdi_datasets():
    time = xr.date_range("1990-01-01", periods=_DAILY_PERIODS, freq="D")
    rng = np.random.default_rng(42)
    precip = xr.Dataset(
        {"precip": ("time", rng.gamma(2.0, 2.0, _DAILY_PERIODS), {"units": "mm"})},
        coords={"time": time},
    )
    temperature = xr.Dataset(
        {"tmax": ("time", 25.0 + 5.0 * rng.random(_DAILY_PERIODS), {"units": "degC"})},
        coords={"time": time},
    )
    return {"precip.nc": precip, "temp.nc": temperature}


def _kbdi_arguments(**overrides):
    arguments = {
        "index": "kbdi",
        "periodicity": compute.Periodicity.daily,
        "scales": None,
        "calibration_start_year": None,
        "calibration_end_year": None,
        "netcdf_precip": "precip.nc",
        "var_name_precip": "precip",
        "netcdf_temp": "temp.nc",
        "var_name_temp": "tmax",
        "netcdf_pet": None,
        "var_name_pet": None,
        "netcdf_awc": None,
        "var_name_awc": None,
        "output_file_base": "output",
        "multiprocessing": "single",
        "chunksizes": "none",
        "kbdi_units": "metric",
        "kbdi_initial": 0.0,
    }
    arguments.update(overrides)
    return argparse.Namespace(**arguments)


def _patch_open_dataset(monkeypatch, datasets):
    original_open_dataset = xr.open_dataset

    def _open_dataset(path, **kwargs):
        if path in datasets:
            return datasets[path]
        return original_open_dataset(path, **kwargs)

    monkeypatch.setattr(cli_main.xr, "open_dataset", _open_dataset)
    return original_open_dataset


class TestKBDIValidation:
    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            (
                {"periodicity": compute.Periodicity.monthly},
                "Invalid periodicity argument for KBDI: 'monthly' -- only 'daily' is supported",
            ),
            ({"scales": [1, 3]}, "The --scales argument is not applicable to KBDI"),
            (
                {"calibration_start_year": 1990},
                "The --calibration_start_year and --calibration_end_year arguments are not applicable to KBDI",
            ),
            (
                {"calibration_end_year": 2020},
                "The --calibration_start_year and --calibration_end_year arguments are not applicable to KBDI",
            ),
            ({"netcdf_pet": "pet.nc"}, "The --netcdf_pet and --var_name_pet arguments are not applicable to KBDI"),
            ({"var_name_pet": "pet"}, "The --netcdf_pet and --var_name_pet arguments are not applicable to KBDI"),
            ({"netcdf_awc": "awc.nc"}, "The --netcdf_awc and --var_name_awc arguments are not applicable to KBDI"),
            ({"var_name_awc": "awc"}, "The --netcdf_awc and --var_name_awc arguments are not applicable to KBDI"),
            ({"netcdf_temp": None}, "Missing the required temperature file argument"),
            ({"var_name_temp": None}, "Missing temperature variable name"),
        ],
    )
    def test_rejects_inapplicable_arguments(self, overrides, message):
        with pytest.raises(ValueError) as error:
            cli_main._validate_args(_kbdi_arguments(**overrides))

        assert str(error.value) == message

    def test_rejects_missing_temperature_variable(self, monkeypatch, kbdi_datasets):
        _patch_open_dataset(monkeypatch, kbdi_datasets)

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(_kbdi_arguments(var_name_temp="bogus"))

        assert (
            str(error.value)
            == "Invalid temperature variable name: 'bogus' does not exist in temperature file 'temp.nc'"
        )

    def test_rejects_non_matching_times(self, monkeypatch, kbdi_datasets):
        kbdi_datasets["temp.nc"] = kbdi_datasets["temp.nc"].assign_coords(
            time=kbdi_datasets["temp.nc"]["time"] + np.timedelta64(1, "D")
        )
        _patch_open_dataset(monkeypatch, kbdi_datasets)

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(_kbdi_arguments())

        assert str(error.value) == "Precipitation and temperature variables contain non-matching times"

    def test_rejects_time_dependent_temperature_dimensions(self, monkeypatch, kbdi_datasets):
        kbdi_datasets["temp.nc"] = kbdi_datasets["temp.nc"].expand_dims("division", axis=0)
        _patch_open_dataset(monkeypatch, kbdi_datasets)

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(_kbdi_arguments())

        assert str(error.value) == (
            "Invalid dimensions of the temperature variable: ('division', 'time') "
            "(expected the precipitation variable dimensions: ('time',))"
        )

    def test_accepts_reordered_temperature_dimensions(self, monkeypatch, kbdi_datasets):
        time = xr.date_range("1990-01-01", periods=_DAILY_PERIODS, freq="D")
        rng = np.random.default_rng(0)
        coords = {"lat": [25.0, 26.0], "lon": [-100.0, -99.0, -98.0], "time": time}
        kbdi_datasets["precip.nc"] = xr.Dataset(
            {"precip": (("lat", "lon", "time"), rng.gamma(2.0, 2.0, (2, 3, _DAILY_PERIODS)), {"units": "mm"})},
            coords=coords,
        )
        kbdi_datasets["temp.nc"] = xr.Dataset(
            {"tmax": (("time", "lat", "lon"), 25.0 + 5.0 * rng.random((_DAILY_PERIODS, 2, 3)), {"units": "degC"})},
            coords=coords,
        )
        _patch_open_dataset(monkeypatch, kbdi_datasets)

        assert cli_main._validate_args(_kbdi_arguments()) == cli_main.DatasetLayout.GRID

    def test_accepts_reordered_divisions_dimensions(self, monkeypatch, kbdi_datasets):
        time = xr.date_range("1990-01-01", periods=_DAILY_PERIODS, freq="D")
        rng = np.random.default_rng(1)
        coords = {"division": ["0101"], "time": time}
        kbdi_datasets["precip.nc"] = xr.Dataset(
            {"precip": (("time", "division"), rng.gamma(2.0, 2.0, (_DAILY_PERIODS, 1)), {"units": "mm"})},
            coords=coords,
        )
        kbdi_datasets["temp.nc"] = xr.Dataset(
            {"tmax": (("division", "time"), 25.0 + 5.0 * rng.random((1, _DAILY_PERIODS)), {"units": "degC"})},
            coords=coords,
        )
        _patch_open_dataset(monkeypatch, kbdi_datasets)

        assert cli_main._validate_args(_kbdi_arguments()) == cli_main.DatasetLayout.DIVISIONS


class TestKBDIProcessing:
    @pytest.mark.parametrize(
        ("units", "var_name", "cf_units", "variant"),
        [
            ("metric", "kbdi", "mm", "metric"),
            ("imperial", "kbdi_imperial", "0.01 in", "imperial"),
        ],
    )
    def test_writes_cf_compliant_output(
        self,
        monkeypatch,
        tmp_path,
        kbdi_datasets,
        units,
        var_name,
        cf_units,
        variant,
    ):
        original_open_dataset = _patch_open_dataset(monkeypatch, kbdi_datasets)

        def _fail_compute_write_index(kwargs):
            raise AssertionError("KBDI must not route through _compute_write_index()")

        monkeypatch.setattr(cli_main, "_compute_write_index", _fail_compute_write_index)

        output_file_base = str(tmp_path / "out")
        cli_main.process_climate_indices(
            _kbdi_arguments(output_file_base=output_file_base, kbdi_units=units),
        )

        output_file = tmp_path / f"out_{var_name}.nc"
        assert output_file.exists()
        with original_open_dataset(output_file) as dataset:
            assert list(dataset.data_vars) == [var_name]
            assert dataset[var_name].attrs["long_name"] == "Keetch-Byram Drought Index"
            assert dataset[var_name].attrs["units"] == cf_units
            assert dataset[var_name].attrs["climate_indices_variant"] == variant
            assert dataset[var_name].sizes["time"] == _DAILY_PERIODS
            assert np.isfinite(dataset[var_name].values).all()

    @pytest.mark.parametrize(
        "dims",
        [("lat", "lon", "time"), ("time", "lat", "lon")],
        ids=["lat_lon_time", "time_lat_lon"],
    )
    @pytest.mark.parametrize("shape", [(1, 3), (3, 1)], ids=["1xN", "Nx1"])
    def test_singleton_coordinate_grids(self, monkeypatch, tmp_path, kbdi_datasets, dims, shape):
        # a singleton latitude or longitude has an empty np.diff(): validation
        # must still pass so that matching coordinates reach fire.kbdi()
        time = xr.date_range("1990-01-01", periods=_DAILY_PERIODS, freq="D")
        rng = np.random.default_rng(7)
        n_lat, n_lon = shape
        coords = {"lat": 25.0 + np.arange(n_lat), "lon": -100.0 + np.arange(n_lon), "time": time}
        sizes = {"lat": n_lat, "lon": n_lon, "time": _DAILY_PERIODS}
        values_shape = tuple(sizes[dim] for dim in dims)
        kbdi_datasets["precip.nc"] = xr.DataArray(
            rng.gamma(2.0, 2.0, values_shape),
            dims=dims,
            coords=coords,
            attrs={"units": "mm"},
            name="precip",
        ).to_dataset()
        kbdi_datasets["temp.nc"] = xr.DataArray(
            25.0 + 5.0 * rng.random(values_shape),
            dims=dims,
            coords=coords,
            attrs={"units": "degC"},
            name="tmax",
        ).to_dataset()
        _patch_open_dataset(monkeypatch, kbdi_datasets)

        cli_main.process_climate_indices(_kbdi_arguments(output_file_base=str(tmp_path / "out")))

        with xr.open_dataset(tmp_path / "out_kbdi.nc") as dataset:
            assert dataset["kbdi"].sizes["lat"] == n_lat
            assert dataset["kbdi"].sizes["lon"] == n_lon
            assert np.isfinite(dataset["kbdi"].values).all()

    @pytest.mark.filterwarnings("ignore:The specified chunks separate the stored chunks")
    def test_chunked_inputs_keep_time_whole_and_copy_input_chunksizes(self, monkeypatch, tmp_path):
        time = xr.date_range("1990-01-01", periods=_DAILY_PERIODS, freq="D")
        rng = np.random.default_rng(42)
        coords = {"lat": [25.0, 26.0], "lon": [-100.0, -99.0, -98.0], "time": time}
        precip = xr.Dataset(
            {"precip": (("lat", "lon", "time"), rng.gamma(2.0, 2.0, (2, 3, _DAILY_PERIODS)), {"units": "mm"})},
            coords=coords,
        )
        temperature = xr.Dataset(
            {"tmax": (("time", "lat", "lon"), 25.0 + 5.0 * rng.random((_DAILY_PERIODS, 2, 3)), {"units": "degC"})},
            coords=coords,
        )
        precip_path = tmp_path / "precip.nc"
        temp_path = tmp_path / "temp.nc"
        precip.to_netcdf(precip_path, encoding={"precip": {"chunksizes": (2, 3, 1000)}}, engine="h5netcdf")
        temperature.to_netcdf(temp_path, encoding={"tmax": {"chunksizes": (1000, 2, 3)}}, engine="h5netcdf")

        captured = {}
        original_kbdi = cli_main.fire.kbdi

        # a budget too small to hold one element splits the auto-chunked spatial
        # axes, so the assertions below fail if the CLI stops applying the default
        monkeypatch.setattr(_cli, "DEFAULT_ARRAY_CHUNK_SIZE", "1 B")

        def _capture_kbdi(*args, **kwargs):
            # the xarray adapter calls this same module-level name for each
            # NumPy block; capture only the top-level DataArray invocation
            if isinstance(args[0], xr.DataArray):
                captured["precip"], captured["temp"] = args[0], args[1]
            return original_kbdi(*args, **kwargs)

        monkeypatch.setattr(cli_main.fire, "kbdi", _capture_kbdi)

        cli_main.process_climate_indices(
            _kbdi_arguments(
                netcdf_precip=str(precip_path),
                var_name_precip="precip",
                netcdf_temp=str(temp_path),
                var_name_temp="tmax",
                output_file_base=str(tmp_path / "out"),
                chunksizes="input",
            ),
        )

        # gridded inputs must reach fire.kbdi() as Dask arrays with the full
        # time axis in one chunk (fire.py's recurrence constraint) and the
        # library's default chunk budget driving the auto-chunked spatial axes
        assert captured["precip"].chunks is not None
        assert captured["temp"].chunks is not None
        assert len(captured["precip"].chunks[captured["precip"].dims.index("lat")]) > 1
        assert len(captured["precip"].chunks[captured["precip"].dims.index("time")]) == 1

        with xr.open_dataset(tmp_path / "out_kbdi.nc", engine="h5netcdf") as dataset:
            assert dataset["kbdi"].encoding["chunksizes"] == (2, 3, 1000)
            assert np.isfinite(dataset["kbdi"].values).all()

    def test_oversized_input_chunksizes_are_trimmed(self, tmp_path):
        """An input chunk larger than the written array is trimmed, not dropped."""
        periods = _DAILY_PERIODS
        time = xr.date_range("1990-01-01", periods=periods, freq="D")
        rng = np.random.default_rng(7)
        coords = {"lat": [25.0, 26.0], "lon": [-100.0, -99.0, -98.0], "time": time}
        precip = xr.Dataset(
            {"precip": (("lat", "lon", "time"), rng.gamma(2.0, 2.0, (2, 3, periods)), {"units": "mm"})},
            coords=coords,
        )
        temperature = xr.Dataset(
            {"tmax": (("time", "lat", "lon"), 25.0 + 5.0 * rng.random((periods, 2, 3)), {"units": "degC"})},
            coords=coords,
        )
        precip_path = tmp_path / "precip.nc"
        temp_path = tmp_path / "temp.nc"
        # an unlimited time dimension permits a chunk larger than the data written so far
        precip.to_netcdf(
            precip_path,
            encoding={"precip": {"chunksizes": (2, 3, periods + 1000)}},
            engine="h5netcdf",
            unlimited_dims=["time"],
        )
        temperature.to_netcdf(temp_path, engine="h5netcdf")

        cli_main.process_climate_indices(
            _kbdi_arguments(
                netcdf_precip=str(precip_path),
                var_name_precip="precip",
                netcdf_temp=str(temp_path),
                var_name_temp="tmax",
                output_file_base=str(tmp_path / "out"),
                chunksizes="input",
            ),
        )

        with xr.open_dataset(tmp_path / "out_kbdi.nc", engine="h5netcdf") as dataset:
            assert dataset["kbdi"].encoding["chunksizes"] == (2, 3, periods)
