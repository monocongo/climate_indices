"""Tests for the CLI's flood-index dispatch paths in climate_indices.__main__.

Each flood index reaches the CLI through an isolated xarray-backed branch, as
KBDI does, since the shared daily path reshapes inputs into 366-day years and
would corrupt the daily recurrences and annual maxima. EDI and the Flood Index
consume effective precipitation (PE): they compute it first from a
precipitation input, or take a PE file in its place.
"""

import argparse

import numpy as np
import pytest
import xarray as xr

from climate_indices import __main__ as cli_main
from climate_indices import _cli, compute, flood
from climate_indices.exceptions import InvalidArgumentError

# six complete calendar years of daily record, 2015 through 2020: PE is undefined
# for the first 364 days, and the 2016-2018 calibration years the tests pass
# differ from the ones the library infers, so a CLI that dropped them would
# change the result
_DAILY_PERIODS = 2192
_CALIBRATION = {"calibration_year_initial": 2016, "calibration_year_final": 2018}
_DECAY = 0.9
# variable names other than the ones the CLI writes, so reading a hard-coded name fails
_PRECIP_VARIABLE = "prcp"
_PE_VARIABLE = "eff_pe"

_INDICES = ("pe", "edi", "flood_index", "api")
# the indices computed from PE, which take the calibration years
_FROM_PE = ("edi", "flood_index")
# the flood functions each index calls, in the order it calls them
_FLOOD_CALLS = {
    "pe": {"effective_precipitation"},
    "edi": {"effective_precipitation", "edi"},
    "flood_index": {"effective_precipitation", "flood_index"},
    "api": {"antecedent_precipitation_index"},
}


def _daily_precipitation(*shape: int, dims: tuple[str, ...] = ("time",), coords: dict | None = None) -> xr.DataArray:
    time = xr.date_range("2015-01-01", periods=_DAILY_PERIODS, freq="D")
    rng = np.random.default_rng(42)
    return xr.DataArray(
        rng.gamma(0.4, 8.0, (*shape, _DAILY_PERIODS)),
        dims=dims,
        coords={**(coords or {}), "time": time},
        attrs={"units": "mm"},
        name=_PRECIP_VARIABLE,
    )


def _grid(*dims: str) -> tuple[xr.DataArray, str]:
    coords = {"lat": [25.0, 26.0], "lon": [-100.0, -99.0, -98.0]}
    precipitation = _daily_precipitation(2, 3, dims=("lat", "lon", "time"), coords=coords)
    return precipitation.transpose(*dims), "lat"


def _divisions(*dims: str) -> tuple[xr.DataArray, str]:
    coords = {"division": ["0101", "0102", "0103"]}
    precipitation = _daily_precipitation(3, dims=("division", "time"), coords=coords)
    return precipitation.transpose(*dims), "division"


# each layout's input, and the spatial dimension the CLI is expected to chunk
_LAYOUTS = {
    "lat_lon_time": lambda: _grid("lat", "lon", "time"),
    "time_lat_lon": lambda: _grid("time", "lat", "lon"),
    "division_time": lambda: _divisions("division", "time"),
    "time_division": lambda: _divisions("time", "division"),
}


def _expected(index: str, precipitation: xr.DataArray) -> xr.DataArray:
    """The library's own result for what the CLI writes for an index, on the arguments the tests pass."""
    if index == "api":
        return flood.antecedent_precipitation_index(precipitation, _DECAY)
    pe = flood.effective_precipitation(precipitation)
    if index == "edi":
        return flood.edi(pe, **_CALIBRATION)
    if index == "flood_index":
        return flood.flood_index(pe, **_CALIBRATION, year_start_month=1)
    return pe


def _calibration_arguments(index: str) -> dict[str, int]:
    """The calibration years, for the indices that take them."""
    if index in _FROM_PE:
        return {"calibration_start_year": 2016, "calibration_end_year": 2018}
    return {}


@pytest.fixture
def precipitation():
    return _daily_precipitation()


@pytest.fixture
def precip_file(tmp_path, precipitation):
    path = tmp_path / "precip.nc"
    precipitation.to_netcdf(path)
    return str(path)


@pytest.fixture
def pe_file(tmp_path, precipitation):
    path = tmp_path / "pe.nc"
    flood.effective_precipitation(precipitation).rename(_PE_VARIABLE).to_netcdf(path)
    return str(path)


def _flood_arguments(index, **overrides):
    arguments = {
        "index": index,
        "periodicity": compute.Periodicity.daily,
        "scales": None,
        "calibration_start_year": None,
        "calibration_end_year": None,
        "netcdf_precip": "precip.nc",
        "var_name_precip": _PRECIP_VARIABLE,
        "netcdf_temp": None,
        "var_name_temp": None,
        "netcdf_pet": None,
        "var_name_pet": None,
        "netcdf_pe": None,
        "var_name_pe": None,
        "netcdf_awc": None,
        "var_name_awc": None,
        "year_start_month": None,
        "api_k": None,
        "output_file_base": "output",
        "multiprocessing": "single",
        "chunksizes": "none",
    }
    arguments.update(overrides)
    # a Flood Index and API run needs the argument that defines them
    if index == "flood_index" and arguments["year_start_month"] is None:
        arguments["year_start_month"] = 1
    if index == "api" and arguments["api_k"] is None:
        arguments["api_k"] = _DECAY
    return argparse.Namespace(**arguments)


class TestFloodValidation:
    @pytest.mark.parametrize(
        ("index", "overrides", "message"),
        [
            (
                "pe",
                {"periodicity": compute.Periodicity.monthly},
                "Invalid periodicity argument for pe: 'monthly' -- only 'daily' is supported",
            ),
            ("edi", {"scales": [1, 3]}, "The --scales argument is not applicable to --index edi"),
            (
                "api",
                {"netcdf_temp": "temp.nc"},
                "The --netcdf_temp and --var_name_temp arguments are not applicable to --index api",
            ),
            (
                "pe",
                {"var_name_pet": "pet"},
                "The --netcdf_pet and --var_name_pet arguments are not applicable to --index pe",
            ),
            (
                "flood_index",
                {"netcdf_awc": "awc.nc"},
                "The --netcdf_awc and --var_name_awc arguments are not applicable to --index flood_index",
            ),
            (
                "pe",
                {"calibration_start_year": 2016},
                "The --calibration_start_year and --calibration_end_year arguments are not applicable to --index pe",
            ),
            (
                "api",
                {"calibration_end_year": 2019},
                "The --calibration_start_year and --calibration_end_year arguments are not applicable to --index api",
            ),
            (
                "api",
                {"netcdf_pe": "pe.nc"},
                "The --netcdf_pe and --var_name_pe arguments are not applicable to --index api",
            ),
            ("edi", {"year_start_month": 3}, "The --year_start_month argument is not applicable to --index edi"),
            ("flood_index", {"api_k": 0.9}, "The --api_k argument is not applicable to --index flood_index"),
        ],
    )
    def test_rejects_inapplicable_arguments(self, index, overrides, message):
        arguments = _flood_arguments(index, **overrides)

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(arguments)

        assert str(error.value) == message

    def test_flood_index_requires_a_year_start_month(self):
        arguments = _flood_arguments("flood_index")
        arguments.year_start_month = None

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(arguments)

        assert str(error.value) == "Missing the required --year_start_month argument"

    def test_api_requires_a_decay_constant(self):
        arguments = _flood_arguments("api")
        arguments.api_k = None

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(arguments)

        assert str(error.value) == "Missing the required --api_k argument"

    @pytest.mark.parametrize("index", _FROM_PE)
    def test_rejects_both_precipitation_and_pe_files(self, index):
        arguments = _flood_arguments(index, netcdf_pe="pe.nc", var_name_pe=_PE_VARIABLE)

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(arguments)

        assert (
            str(error.value) == "Both precipitation and PE files were specified, only one of these should be provided"
        )

    def test_rejects_a_pe_variable_name_without_a_pe_file(self):
        arguments = _flood_arguments("edi", var_name_pe=_PE_VARIABLE)

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(arguments)

        assert str(error.value) == "The --var_name_pe argument requires the --netcdf_pe argument"

    @pytest.mark.parametrize("index", _FROM_PE)
    def test_requires_precipitation_or_pe(self, index):
        arguments = _flood_arguments(index, netcdf_precip=None)

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(arguments)

        assert str(error.value) == "Missing the required precipitation file"

    def test_requires_a_pe_variable_name_with_a_pe_file(self, pe_file):
        arguments = _flood_arguments("edi", netcdf_precip=None, netcdf_pe=pe_file)

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(arguments)

        assert str(error.value) == "Missing effective precipitation variable name"

    def test_rejects_missing_pe_variable(self, pe_file):
        arguments = _flood_arguments("edi", netcdf_precip=None, netcdf_pe=pe_file, var_name_pe="bogus")

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(arguments)

        assert str(error.value).startswith("Invalid effective precipitation variable name: 'bogus'")

    def test_rejects_a_pe_file_of_unsupported_dimensions(self, tmp_path):
        path = tmp_path / "pe_bad.nc"
        xr.DataArray(np.ones((2, 3)), dims=("x", "y"), name=_PE_VARIABLE).to_netcdf(path)

        arguments = _flood_arguments("edi", netcdf_precip=None, netcdf_pe=str(path), var_name_pe=_PE_VARIABLE)

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(arguments)

        assert str(error.value).startswith("Invalid dimensions of the effective precipitation variable")

    @pytest.mark.parametrize("index", _FROM_PE)
    def test_a_pe_file_alone_determines_the_input_type(self, index, pe_file):
        arguments = _flood_arguments(index, netcdf_precip=None, netcdf_pe=pe_file, var_name_pe=_PE_VARIABLE)

        assert cli_main._validate_args(arguments) == cli_main.DatasetLayout.TIMESERIES


class TestPrecipitationInputMessages:
    """The shared precipitation validator keeps its messages for the ordinary precipitation input."""

    def test_missing_file(self):
        arguments = argparse.Namespace(netcdf_precip=None, var_name_precip=_PRECIP_VARIABLE)

        with pytest.raises(ValueError) as error:
            cli_main._validate_precipitation_input(arguments)

        assert str(error.value) == "Missing the required precipitation file"

    def test_missing_variable_name(self):
        arguments = argparse.Namespace(netcdf_precip="precip.nc", var_name_precip=None)

        with pytest.raises(ValueError) as error:
            cli_main._validate_precipitation_input(arguments)

        assert str(error.value) == "Missing precipitation variable name"

    def test_unknown_variable(self, precip_file):
        arguments = argparse.Namespace(netcdf_precip=precip_file, var_name_precip="bogus")

        with pytest.raises(ValueError) as error:
            cli_main._validate_precipitation_input(arguments)

        message = str(error.value)
        assert message.startswith("Invalid precipitation variable name: 'bogus'")
        assert message.endswith(f"does not exist in precipitation file '{precip_file}'")

    def test_unsupported_dimensions(self, tmp_path):
        path = tmp_path / "precip_bad.nc"
        xr.DataArray(np.ones((2, 3)), dims=("x", "y"), name=_PRECIP_VARIABLE).to_netcdf(path)
        arguments = argparse.Namespace(netcdf_precip=str(path), var_name_precip=_PRECIP_VARIABLE)

        with pytest.raises(ValueError) as error:
            cli_main._validate_precipitation_input(arguments)

        assert str(error.value).startswith("Invalid dimensions of the precipitation variable")


class TestFloodProcessing:
    def test_pe_writes_cf_annotated_output(self, tmp_path, precipitation, precip_file):
        cli_main.process_climate_indices(
            _flood_arguments("pe", netcdf_precip=precip_file, output_file_base=str(tmp_path / "out"))
        )

        with xr.open_dataset(tmp_path / "out_pe.nc") as dataset:
            assert list(dataset.data_vars) == ["pe"]
            assert dataset["pe"].attrs["long_name"] == "Effective Precipitation"
            assert dataset["pe"].attrs["units"] == "mm"
            np.testing.assert_allclose(dataset["pe"].values, _expected("pe", precipitation).values, equal_nan=True)

    @pytest.mark.parametrize("index", _FROM_PE)
    def test_pe_indices_compute_and_write_pe_first(self, tmp_path, precipitation, precip_file, index):
        cli_main.process_climate_indices(
            _flood_arguments(
                index,
                netcdf_precip=precip_file,
                output_file_base=str(tmp_path / "out"),
                **_calibration_arguments(index),
            )
        )

        expected = _expected(index, precipitation)
        assert (tmp_path / "out_pe.nc").exists()
        with xr.open_dataset(tmp_path / f"out_{index}.nc") as dataset:
            assert list(dataset.data_vars) == [index]
            assert dataset[index].attrs["long_name"] == expected.attrs["long_name"]
            assert dataset[index].attrs["units"] == "dimensionless"
            np.testing.assert_allclose(dataset[index].values, expected.values, equal_nan=True)
            assert np.isfinite(dataset[index].values).any()

    @pytest.mark.parametrize("index", _FROM_PE)
    def test_a_provided_pe_file_replaces_the_pe_step(self, tmp_path, precipitation, pe_file, index):
        cli_main.process_climate_indices(
            _flood_arguments(
                index,
                netcdf_precip=None,
                netcdf_pe=pe_file,
                var_name_pe=_PE_VARIABLE,
                output_file_base=str(tmp_path / "out"),
                **_calibration_arguments(index),
            )
        )

        assert not (tmp_path / "out_pe.nc").exists()
        with xr.open_dataset(tmp_path / f"out_{index}.nc") as dataset:
            np.testing.assert_allclose(dataset[index].values, _expected(index, precipitation).values, equal_nan=True)

    def test_flood_index_honors_the_year_start_month(self, tmp_path, precipitation, precip_file):
        cli_main.process_climate_indices(
            _flood_arguments(
                "flood_index",
                netcdf_precip=precip_file,
                year_start_month=7,
                output_file_base=str(tmp_path / "out"),
                **_calibration_arguments("flood_index"),
            )
        )

        expected = flood.flood_index(
            flood.effective_precipitation(precipitation),
            **_CALIBRATION,
            year_start_month=7,
        )
        with xr.open_dataset(tmp_path / "out_flood_index.nc") as dataset:
            np.testing.assert_allclose(dataset["flood_index"].values, expected.values, equal_nan=True)

    @pytest.mark.parametrize("from_pe_file", [False, True], ids=["from_precipitation", "from_pe_file"])
    def test_flood_index_calibrates_from_the_second_year_by_default(
        self, tmp_path, precipitation, precip_file, pe_file, from_pe_file
    ):
        # PE's first 364 days are undefined, so the record's first year would
        # enter the calibration sample with a maximum from a day or two of PE
        inputs = {"netcdf_precip": precip_file}
        if from_pe_file:
            inputs = {"netcdf_precip": None, "netcdf_pe": pe_file, "var_name_pe": _PE_VARIABLE}
        cli_main.process_climate_indices(
            _flood_arguments("flood_index", output_file_base=str(tmp_path / "out"), **inputs)
        )

        pe = flood.effective_precipitation(precipitation)
        expected = flood.flood_index(pe, calibration_year_initial=2016, year_start_month=1)
        library_default = flood.flood_index(pe, year_start_month=1)
        assert not np.allclose(expected.values, library_default.values, equal_nan=True)
        with xr.open_dataset(tmp_path / "out_flood_index.nc") as dataset:
            np.testing.assert_allclose(dataset["flood_index"].values, expected.values, equal_nan=True)

    def test_flood_index_honors_an_explicit_calibration_start_year(self, tmp_path, precipitation, precip_file):
        cli_main.process_climate_indices(
            _flood_arguments(
                "flood_index",
                netcdf_precip=precip_file,
                calibration_start_year=2017,
                output_file_base=str(tmp_path / "out"),
            )
        )

        expected = flood.flood_index(
            flood.effective_precipitation(precipitation), calibration_year_initial=2017, year_start_month=1
        )
        with xr.open_dataset(tmp_path / "out_flood_index.nc") as dataset:
            np.testing.assert_allclose(dataset["flood_index"].values, expected.values, equal_nan=True)

    def test_api_writes_cf_annotated_output(self, tmp_path, precipitation, precip_file):
        cli_main.process_climate_indices(
            _flood_arguments("api", netcdf_precip=precip_file, api_k=0.8, output_file_base=str(tmp_path / "out"))
        )

        expected = flood.antecedent_precipitation_index(precipitation, 0.8)
        with xr.open_dataset(tmp_path / "out_api.nc") as dataset:
            assert list(dataset.data_vars) == ["api"]
            assert dataset["api"].attrs["long_name"] == "Antecedent Precipitation Index"
            assert dataset["api"].attrs["units"] == "mm"
            np.testing.assert_allclose(dataset["api"].values, expected.values)

    def test_api_rejects_an_out_of_range_decay_constant(self, tmp_path, precip_file):
        arguments = _flood_arguments(
            "api", netcdf_precip=precip_file, api_k=1.5, output_file_base=str(tmp_path / "out")
        )

        with pytest.raises(InvalidArgumentError):
            cli_main.process_climate_indices(arguments)

        assert not (tmp_path / "out_api.nc").exists()

    def test_a_failed_run_keeps_the_previous_output(self, tmp_path, precip_file):
        # the calibration period precedes the record, which the kernel rejects
        # only once the lazy computation runs inside to_netcdf()
        previous = tmp_path / "out_edi.nc"
        xr.Dataset({"edi": ("time", [1.0, 2.0, 3.0])}).to_netcdf(previous)

        arguments = _flood_arguments(
            "edi",
            netcdf_precip=precip_file,
            calibration_start_year=1900,
            calibration_end_year=1990,
            output_file_base=str(tmp_path / "out"),
        )

        with pytest.raises(InvalidArgumentError):
            cli_main.process_climate_indices(arguments)

        with xr.open_dataset(previous) as dataset:
            assert dataset["edi"].values.tolist() == [1.0, 2.0, 3.0]
        assert not (tmp_path / "out_edi.nc.tmp").exists()

    def test_flood_indices_do_not_route_through_the_shared_daily_path(self, monkeypatch, tmp_path, precip_file):
        def _fail_compute_write_index(request):
            raise AssertionError("flood indices must not route through _compute_write_index()")

        monkeypatch.setattr(cli_main, "_compute_write_index", _fail_compute_write_index)

        cli_main.process_climate_indices(
            _flood_arguments("api", netcdf_precip=precip_file, output_file_base=str(tmp_path / "out"))
        )

        assert (tmp_path / "out_api.nc").exists()

    @pytest.mark.parametrize("layout", _LAYOUTS)
    @pytest.mark.parametrize("index", _INDICES)
    def test_spatial_inputs_keep_time_whole_and_chunk_space(self, monkeypatch, tmp_path, index, layout):
        precipitation, spatial_dim = _LAYOUTS[layout]()
        path = tmp_path / "input.nc"
        precipitation.to_netcdf(path)
        expected = _expected(index, precipitation)

        # capture the DataArray each flood call is given
        captured = {}
        for name in _FLOOD_CALLS["edi"] | _FLOOD_CALLS["flood_index"] | _FLOOD_CALLS["api"]:
            original = getattr(flood, name)

            def _capture(data, *args, _name=name, _original=original, **kwargs):
                captured[_name] = data
                return _original(data, *args, **kwargs)

            monkeypatch.setattr(cli_main.flood, name, _capture)
        # a budget too small to hold one element splits the auto-chunked spatial
        # axes, so the assertions below fail if the CLI stops applying the default
        monkeypatch.setattr(_cli, "DEFAULT_ARRAY_CHUNK_SIZE", "1 B")

        cli_main.process_climate_indices(
            _flood_arguments(
                index,
                netcdf_precip=str(path),
                output_file_base=str(tmp_path / "out"),
                **_calibration_arguments(index),
            )
        )

        assert set(captured) == _FLOOD_CALLS[index]
        for data in captured.values():
            assert data.chunks is not None
            assert len(data.chunks[data.dims.index("time")]) == 1
            assert len(data.chunks[data.dims.index(spatial_dim)]) > 1
        with xr.open_dataset(tmp_path / f"out_{index}.nc") as dataset:
            np.testing.assert_allclose(dataset[index].values, expected.values, equal_nan=True)

    @pytest.mark.filterwarnings("ignore:The specified chunks separate the stored chunks")
    @pytest.mark.parametrize("index", _INDICES)
    def test_chunksizes_input_copies_the_input_chunks(self, tmp_path, index):
        precipitation, _ = _LAYOUTS["lat_lon_time"]()
        path = tmp_path / "grid.nc"
        precipitation.to_netcdf(path, encoding={_PRECIP_VARIABLE: {"chunksizes": (2, 3, 500)}}, engine="h5netcdf")

        cli_main.process_climate_indices(
            _flood_arguments(
                index,
                netcdf_precip=str(path),
                chunksizes="input",
                output_file_base=str(tmp_path / "out"),
            )
        )

        with xr.open_dataset(tmp_path / f"out_{index}.nc", engine="h5netcdf") as dataset:
            assert dataset[index].encoding["chunksizes"] == (2, 3, 500)


class TestFloodRegistration:
    def test_aggregate_indices_do_not_run_the_flood_indices(self):
        for aggregate in ("scaled", "all"):
            assert not set(cli_main._INDEX_PIPELINES[aggregate]) & set(_INDICES)

    @pytest.mark.parametrize("index", _INDICES)
    def test_flood_indices_compute_through_xarray_only(self, index):
        registration = cli_main._registry_for(index)

        assert cli_main._INDEX_PIPELINES[index] == (index,)
        assert registration.kernel is None
        assert registration.compute is None
        assert registration.write is None
        assert registration.validate_arguments is cli_main._validate_flood_arguments

    def test_the_cli_parses_the_flood_arguments(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(cli_main, "process_climate_indices", lambda arguments: captured.update(vars(arguments)))

        cli_main.main(
            [
                "--index",
                "flood_index",
                "--periodicity",
                "daily",
                "--netcdf_pe",
                "pe.nc",
                "--var_name_pe",
                _PE_VARIABLE,
                "--year_start_month",
                "10",
                "--api_k",
                "0.85",
                "--output_file_base",
                "out",
            ]
        )

        assert captured["netcdf_pe"] == "pe.nc"
        assert captured["var_name_pe"] == _PE_VARIABLE
        assert captured["year_start_month"] == 10
        assert captured["api_k"] == 0.85

    def test_the_cli_rejects_an_out_of_range_year_start_month(self):
        with pytest.raises(SystemExit):
            cli_main.main(["--index", "flood_index", "--periodicity", "daily", "--year_start_month", "13"])
