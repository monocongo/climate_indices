"""Tests for the CLI's energy-index dispatch paths in climate_indices.__main__.

Each energy index reaches the CLI through an isolated xarray-backed branch, as
KBDI and the flood indices do: their adapters stamp the CF metadata, keep Dask
arrays lazy, and transform each grid cell independently. The degree days take a
temperature input, the wind profile takes a wind speed, and the standardized
indices take the series they standardize.
"""

import argparse
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import xarray as xr

from climate_indices import __main__ as cli_main
from climate_indices import _cli, compute, energy, indices
from climate_indices.exceptions import CoordinateValidationError, InvalidArgumentError

# twelve complete calendar years of daily record, 2015 through 2026: the
# empirical distribution needs at least ten valid calibration values per
# calendar step, so a shorter record standardizes to NaN throughout
_RECORD_YEARS = 12
_DAILY_PERIODS = 365 * _RECORD_YEARS
_HEATING_BASE = 18.0
_COOLING_BASE = 24.0
_HUB_HEIGHT = 100.0
_ROUGHNESS = 0.03
_SCALE = 3
# variable names other than the ones the CLI writes, so reading a hard-coded name fails
_TEMPERATURE_VARIABLE = "t2m"
_WIND_VARIABLE = "sfcWind"
_VALUES_VARIABLE = "cf"

_DEGREE_DAY_INDICES = ("hdd", "cdd", "edd")
_STANDARDIZED_INDICES = ("srepi", "demand_index", "srli")
_INDICES = (*_DEGREE_DAY_INDICES, "wind_profile", *_STANDARDIZED_INDICES)

# the CF registry name of each index's output variable
_OUTPUT_NAMES = {
    "hdd": "heating_degree_days",
    "cdd": "cooling_degree_days",
    "edd": "energy_degree_days",
    "wind_profile": "hub_height_wind_speed",
    "srepi": f"srepi_{_SCALE:02d}",
    "demand_index": f"standardized_demand_index_{_SCALE:02d}",
    "srli": f"srli_{_SCALE:02d}",
}


def _gridded(values_function: Callable[..., np.ndarray], name: str, units: str) -> xr.DataArray:
    coords = {"lat": [25.0, 26.0], "lon": [-100.0, -99.0, -98.0]}
    rng = np.random.default_rng(42)
    return xr.DataArray(
        values_function(rng, (2, 3, _DAILY_PERIODS)),
        dims=("lat", "lon", "time"),
        coords={**coords, "time": xr.date_range("2015-01-01", periods=_DAILY_PERIODS, freq="D")},
        attrs={"units": units},
        name=name,
    )


def _temperature() -> xr.DataArray:
    return _gridded(lambda rng, shape: rng.normal(15.0, 8.0, shape), _TEMPERATURE_VARIABLE, "degC")


def _wind_speed() -> xr.DataArray:
    return _gridded(lambda rng, shape: rng.random(shape) + 2.0, _WIND_VARIABLE, "m s-1")


def _capacity_factor() -> xr.DataArray:
    return _gridded(lambda rng, shape: rng.random(shape), _VALUES_VARIABLE, "1")


def _demand_proxy() -> xr.DataArray:
    return _gridded(lambda rng, shape: rng.random(shape) * 20.0, _VALUES_VARIABLE, "K d")


def _residual_load() -> xr.DataArray:
    return _gridded(lambda rng, shape: rng.normal(0.0, 1.0, shape), "residual", "GW")


_INPUTS = {
    "hdd": _temperature,
    "cdd": _temperature,
    "edd": _temperature,
    "wind_profile": _wind_speed,
    "srepi": _capacity_factor,
    "demand_index": _demand_proxy,
    "srli": _residual_load,
}

# the request field naming each index's input file, and the parser's flags it defines
_INPUT_FIELD = {
    "hdd": "netcdf_temp",
    "cdd": "netcdf_temp",
    "edd": "netcdf_temp",
    "wind_profile": "netcdf_wind",
    "srepi": "netcdf_values",
    "demand_index": "netcdf_values",
    "srli": "netcdf_values",
}
_VARIABLE_FIELD = {
    "netcdf_temp": "var_name_temp",
    "netcdf_wind": "var_name_wind",
    "netcdf_values": "var_name_values",
}
_UNITS = {"hdd": "K d", "cdd": "K d", "edd": "K d", "wind_profile": "m s-1"}

# the energy function each index computes through, for capturing its input
_CHUNKING_FUNCTIONS = {
    "hdd": "heating_degree_days",
    "cdd": "cooling_degree_days",
    "edd": "energy_degree_days",
    "wind_profile": "hub_height_wind_speed",
    "srepi": "srepi",
    "demand_index": "standardized_demand_index",
    "srli": "srli",
}


def _defining_arguments(index: str) -> dict[str, object]:
    """The arguments each index requires beyond its input file."""
    arguments: dict[str, object] = {}
    if index in ("hdd", "edd"):
        arguments["heating_base_temperature"] = _HEATING_BASE
    if index in ("cdd", "edd"):
        arguments["cooling_base_temperature"] = _COOLING_BASE
    if index == "wind_profile":
        arguments["hub_height"] = _HUB_HEIGHT
    if index in _STANDARDIZED_INDICES:
        arguments["scales"] = [_SCALE]
    return arguments


def _energy_arguments(index: str, **overrides: object) -> argparse.Namespace:
    arguments: dict[str, object] = {
        "index": index,
        "periodicity": compute.Periodicity.daily,
        "output_scale": None,
        "zero_handling": "classic",
        "scales": None,
        "calibration_start_year": None,
        "calibration_end_year": None,
        "netcdf_precip": None,
        "var_name_precip": None,
        "netcdf_temp": None,
        "var_name_temp": None,
        "netcdf_pet": None,
        "var_name_pet": None,
        "netcdf_pe": None,
        "var_name_pe": None,
        "netcdf_awc": None,
        "var_name_awc": None,
        "netcdf_wind": None,
        "var_name_wind": None,
        "netcdf_values": None,
        "var_name_values": None,
        "heating_base_temperature": None,
        "cooling_base_temperature": None,
        "reference_height": None,
        "hub_height": None,
        "wind_profile_method": None,
        "wind_shear_alpha": None,
        "roughness_length": None,
        "output_file_base": "output",
        "multiprocessing": "single",
        "chunksizes": "none",
    }
    arguments.update(_defining_arguments(index))
    arguments.update(overrides)
    return argparse.Namespace(**arguments)


def _input_file(tmp_path, index: str) -> tuple[str, xr.DataArray]:
    """Write the index's input series to a NetCDF file, and return its path and values."""
    source = _INPUTS[index]()
    path = tmp_path / "input.nc"
    source.to_netcdf(path)
    return str(path), source


def _arguments_for(index: str, tmp_path, **overrides: object) -> argparse.Namespace:
    """The arguments that compute one index from an input file written for the same test."""
    path, source = _input_file(tmp_path, index)
    field = _INPUT_FIELD[index]
    arguments: dict[str, object] = {
        field: path,
        _VARIABLE_FIELD[field]: source.name,
        "output_file_base": str(tmp_path / "out"),
    }
    arguments.update(overrides)
    return _energy_arguments(index, **arguments)


def _expected(index: str, source: xr.DataArray, scale: int = _SCALE, **kwargs: Any) -> xr.DataArray:
    """The library's own result for what the CLI writes for an index, on the arguments the tests pass."""
    if index == "hdd":
        return energy.heating_degree_days(source, base_temperature=_HEATING_BASE)
    if index == "cdd":
        return energy.cooling_degree_days(source, base_temperature=_COOLING_BASE)
    if index == "edd":
        return energy.energy_degree_days(
            source, heating_base_temperature=_HEATING_BASE, cooling_base_temperature=_COOLING_BASE
        )
    if index == "wind_profile":
        return energy.hub_height_wind_speed(source, 10.0, _HUB_HEIGHT, **kwargs)
    if index == "srepi":
        return energy.srepi(source, scale, **kwargs)
    if index == "demand_index":
        return energy.standardized_demand_index(source, scale, **kwargs)
    return energy.srli(source, scale, **kwargs)


class TestEnergyValidation:
    @pytest.mark.parametrize(
        ("index", "overrides", "message"),
        [
            (
                "hdd",
                {"periodicity": compute.Periodicity.monthly},
                "Invalid periodicity argument for hdd: 'monthly' -- only 'daily' is supported",
            ),
            (
                "hdd",
                {"heating_base_temperature": None},
                "Missing the required --heating_base_temperature argument",
            ),
            (
                "cdd",
                {"cooling_base_temperature": None},
                "Missing the required --cooling_base_temperature argument",
            ),
            (
                "edd",
                {"cooling_base_temperature": None},
                "Missing the required --cooling_base_temperature argument",
            ),
            ("wind_profile", {"hub_height": None}, "Missing the required --hub_height argument"),
            (
                "wind_profile",
                {"wind_profile_method": "log"},
                "Missing the required --roughness_length argument for the log wind profile",
            ),
            (
                "wind_profile",
                {"wind_profile_method": "log", "roughness_length": _ROUGHNESS, "wind_shear_alpha": 0.2},
                "The --wind_shear_alpha argument is not applicable to the log wind profile",
            ),
            (
                "wind_profile",
                {"roughness_length": _ROUGHNESS},
                "The --roughness_length argument requires --wind_profile_method log",
            ),
            (
                "wind_profile",
                {"netcdf_temp": "temperature.nc", "var_name_temp": _TEMPERATURE_VARIABLE},
                "The --netcdf_temp and --var_name_temp arguments are not applicable to --index wind_profile",
            ),
            (
                "srepi",
                {"netcdf_wind": "wind.nc", "var_name_wind": _WIND_VARIABLE},
                "The --netcdf_wind and --var_name_wind arguments are not applicable to --index srepi",
            ),
            (
                "srli",
                {"heating_base_temperature": _HEATING_BASE},
                "The --heating_base_temperature argument is not applicable to --index srli",
            ),
        ],
    )
    def test_rejects_inapplicable_arguments(self, tmp_path, index, overrides, message):
        arguments = _arguments_for(index, tmp_path, **overrides)

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(arguments)

        assert str(error.value) == message

    @pytest.mark.parametrize("index", _STANDARDIZED_INDICES)
    def test_standardized_indices_require_a_scale(self, tmp_path, index):
        arguments = _arguments_for(index, tmp_path, scales=None)

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(arguments)

        assert "missing --scales argument" in str(error.value)

    @pytest.mark.parametrize("index", _STANDARDIZED_INDICES)
    @pytest.mark.parametrize("periodicity", _cli.CLI_PERIODICITIES)
    @pytest.mark.parametrize("invalid_scale", [0, -1, "above_maximum"])
    def test_invalid_scale_lists_write_no_output(self, tmp_path, index, periodicity, invalid_scale):
        maximum = indices.SCALE_MAX_DAILY if periodicity is compute.Periodicity.daily else indices.SCALE_MAX
        if invalid_scale == "above_maximum":
            invalid_scale = maximum + 1
        arguments = _arguments_for(index, tmp_path, periodicity=periodicity, scales=[1, invalid_scale])
        if periodicity is compute.Periodicity.monthly:
            _INPUTS[index]().resample(time="MS").mean().to_netcdf(arguments.netcdf_values)

        with pytest.raises((ValueError, InvalidArgumentError), match="[Ss]cale"):
            cli_main.process_climate_indices(arguments)

        assert not list(tmp_path.glob("out_*.nc"))

    @pytest.mark.parametrize("index", _STANDARDIZED_INDICES)
    @pytest.mark.parametrize("periodicity", _cli.CLI_PERIODICITIES)
    def test_accepts_periodicity_specific_scale_limits(self, index, periodicity):
        maximum = indices.SCALE_MAX_DAILY if periodicity is compute.Periodicity.daily else indices.SCALE_MAX
        arguments = _energy_arguments(index, periodicity=periodicity, scales=[1, maximum])

        cli_main._validate_scales(arguments)

        assert arguments.scales == [1, maximum]

    @pytest.mark.parametrize("index", _STANDARDIZED_INDICES)
    def test_rejects_an_input_whose_cadence_contradicts_the_periodicity(self, tmp_path, index):
        """--periodicity is passed to the adapter, so a monthly input cannot pass as daily."""
        source = (
            _INPUTS[index]()
            .isel(time=slice(0, 12 * _RECORD_YEARS))
            .assign_coords(time=xr.date_range("2015-01-01", periods=12 * _RECORD_YEARS, freq="MS"))
        )
        path = tmp_path / "monthly.nc"
        source.to_netcdf(path)

        arguments = _energy_arguments(
            index,
            netcdf_values=str(path),
            var_name_values=source.name,
            output_file_base=str(tmp_path / "out"),
        )

        with pytest.raises(CoordinateValidationError) as error:
            cli_main.process_climate_indices(arguments)

        assert "does not match the requested daily periodicity" in str(error.value)

    @pytest.mark.parametrize("index", _INDICES)
    def test_requires_the_input_series(self, index):
        arguments = _energy_arguments(index)
        setattr(arguments, _INPUT_FIELD[index], None)
        setattr(arguments, _VARIABLE_FIELD[_INPUT_FIELD[index]], None)

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(arguments)

        assert str(error.value).startswith("Missing the required")

    @pytest.mark.parametrize("index", _INDICES)
    def test_rejects_a_missing_variable(self, tmp_path, index):
        field = _INPUT_FIELD[index]
        arguments = _arguments_for(index, tmp_path, **{_VARIABLE_FIELD[field]: "absent"})

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(arguments)

        assert "'absent' does not exist" in str(error.value)

    def test_pet_alone_keeps_its_monthly_only_rule(self, tmp_path):
        """A daily temperature input is valid for the degree days, which does not relax PET's own rule."""
        path, source = _input_file(tmp_path, "hdd")

        arguments = _energy_arguments("pet", netcdf_temp=path, var_name_temp=source.name)

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(arguments)

        assert "only 'monthly' is supported" in str(error.value)

    def test_a_temperature_driven_spei_keeps_its_monthly_only_rule(self, tmp_path):
        """The degree days accept a daily input without relaxing the PET that SPEI computes."""
        path, source = _input_file(tmp_path, "hdd")

        arguments = _energy_arguments(
            "spei",
            netcdf_precip=path,
            var_name_precip=source.name,
            netcdf_temp=path,
            var_name_temp=source.name,
            scales=[1],
        )

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(arguments)

        assert "only 'monthly' is supported" in str(error.value)

    @pytest.mark.parametrize("index", _STANDARDIZED_INDICES)
    def test_the_output_scale_applies_to_the_standardized_energy_indices(self, tmp_path, index):
        """SREPI, the demand index, and SRLI are standardized outputs, like SPI and SPEI."""
        arguments = _arguments_for(index, tmp_path, output_scale="probability")

        # reaching the input type means --output_scale was accepted
        assert cli_main._validate_args(arguments) is cli_main.DatasetLayout.GRID


class TestEnergyPreconditions:
    @pytest.mark.parametrize("index", ("hdd", "wind_profile", "srepi"))
    def test_input_requires_a_variable_name(self, index):
        arguments = _energy_arguments(index, **{_INPUT_FIELD[index]: "input.nc"})
        request = cli_main._IndexRequest.from_arguments(arguments, index=index, input_type=cli_main.DatasetLayout.GRID)

        with pytest.raises(ValueError, match="the input variable name was not validated"):
            cli_main._energy_input(request)

    @pytest.mark.parametrize(
        ("index", "field"),
        [
            ("hdd", "heating_base_temperature"),
            ("cdd", "cooling_base_temperature"),
            ("edd", "heating_base_temperature"),
            ("edd", "cooling_base_temperature"),
            ("wind_profile", "hub_height"),
        ],
    )
    def test_runner_rejects_an_unvalidated_argument(self, index, field):
        arguments = _energy_arguments(index, **{field: None})
        runner = cli_main._registry_for(index)
        transport = cli_main.Transport(1)

        with pytest.raises(ValueError, match="not validated"):
            runner.run(arguments, cli_main.DatasetLayout.GRID, transport)


class TestEnergyProcessing:
    @pytest.mark.parametrize("index", _INDICES)
    def test_writes_cf_annotated_output(self, tmp_path, index):
        _path, source = _input_file(tmp_path, index)

        cli_main.process_climate_indices(_arguments_for(index, tmp_path))

        name = _OUTPUT_NAMES[index]
        with xr.open_dataset(tmp_path / f"out_{name}.nc") as dataset:
            assert list(dataset.data_vars) == [name]
            assert dataset[name].attrs["long_name"]
            assert dataset[name].attrs["climate_indices_version"]
            if index in _UNITS:
                assert dataset[name].attrs["units"] == _UNITS[index]
            np.testing.assert_allclose(dataset[name].values, _expected(index, source).values, equal_nan=True)

    @pytest.mark.parametrize("index", _STANDARDIZED_INDICES)
    @pytest.mark.parametrize("scales", ([1, _SCALE], [_SCALE, 1, _SCALE, 1]))
    def test_each_scale_writes_its_own_file(self, monkeypatch, tmp_path, index, scales):
        _path, source = _input_file(tmp_path, index)
        written = []
        original = cli_main._write_xarray_index

        def _capture(request, *args, **kwargs):
            written.append(request.scale)
            return original(request, *args, **kwargs)

        monkeypatch.setattr(cli_main, "_write_xarray_index", _capture)

        cli_main.process_climate_indices(_arguments_for(index, tmp_path, scales=scales))

        assert written == list(dict.fromkeys(scales))
        base = _OUTPUT_NAMES[index].rsplit("_", 1)[0]
        for scale in written:
            with xr.open_dataset(tmp_path / f"out_{base}_{scale:02d}.nc") as dataset:
                np.testing.assert_allclose(
                    dataset[f"{base}_{scale:02d}"].values,
                    _expected(index, source, scale=scale).values,
                    equal_nan=True,
                )

    @pytest.mark.parametrize("index", _STANDARDIZED_INDICES)
    def test_a_non_normal_output_scale_names_the_output(self, tmp_path, index):
        _path, source = _input_file(tmp_path, index)

        cli_main.process_climate_indices(_arguments_for(index, tmp_path, output_scale="probability"))

        name = f"{_OUTPUT_NAMES[index]}_probability"
        with xr.open_dataset(tmp_path / f"out_{name}.nc") as dataset:
            np.testing.assert_allclose(
                dataset[name].values,
                _expected(index, source, output_scale="probability").values,
                equal_nan=True,
            )

    def test_the_log_profile_names_its_output_apart_from_the_power_profile(self, tmp_path):
        _path, source = _input_file(tmp_path, "wind_profile")

        cli_main.process_climate_indices(_arguments_for("wind_profile", tmp_path))
        cli_main.process_climate_indices(
            _arguments_for("wind_profile", tmp_path, wind_profile_method="log", roughness_length=_ROUGHNESS)
        )

        expected = {
            "hub_height_wind_speed": _expected("wind_profile", source),
            "hub_height_wind_speed_log": _expected("wind_profile", source, method="log", roughness_length=_ROUGHNESS),
        }
        for name, values in expected.items():
            with xr.open_dataset(tmp_path / f"out_{name}.nc") as dataset:
                np.testing.assert_allclose(dataset[name].values, values.values, equal_nan=True)

    def test_energy_indices_do_not_route_through_the_shared_daily_path(self, monkeypatch, tmp_path):
        def _fail_compute_write_index(request):
            raise AssertionError("energy indices must not route through _compute_write_index()")

        monkeypatch.setattr(cli_main, "_compute_write_index", _fail_compute_write_index)

        cli_main.process_climate_indices(_arguments_for("hdd", tmp_path))

        assert (tmp_path / "out_heating_degree_days.nc").exists()

    @pytest.mark.parametrize("index", _INDICES)
    def test_spatial_inputs_keep_time_whole_and_chunk_space(self, monkeypatch, tmp_path, index):
        captured: dict[str, xr.DataArray] = {}
        name = _CHUNKING_FUNCTIONS[index]
        original = getattr(energy, name)

        def _capture(data, *args, _original=original, **kwargs):
            captured["input"] = data
            return _original(data, *args, **kwargs)

        monkeypatch.setattr(energy, name, _capture)
        # a budget too small to hold one element splits the auto-chunked spatial
        # axes, so the assertions below fail if the CLI stops applying the default
        monkeypatch.setattr(_cli, "DEFAULT_ARRAY_CHUNK_SIZE", "1 B")

        cli_main.process_climate_indices(_arguments_for(index, tmp_path))

        data = captured["input"]
        assert data.chunks is not None
        # the standardized adapters require one time chunk, and the elementwise
        # adapters are given the same layout
        assert len(data.chunks[data.dims.index("time")]) == 1
        assert len(data.chunks[data.dims.index("lat")]) > 1

    @pytest.mark.parametrize("index", _INDICES)
    def test_chunksizes_input_copies_the_input_chunks(self, tmp_path, index):
        source = _INPUTS[index]()
        path = tmp_path / "grid.nc"
        source.to_netcdf(path, encoding={source.name: {"chunksizes": (2, 3, 500)}}, engine="h5netcdf")
        field = _INPUT_FIELD[index]

        cli_main.process_climate_indices(
            _energy_arguments(
                index,
                **{
                    field: str(path),
                    _VARIABLE_FIELD[field]: source.name,
                    "output_file_base": str(tmp_path / "out"),
                    "chunksizes": "input",
                },
            )
        )

        name = _OUTPUT_NAMES[index]
        with xr.open_dataset(tmp_path / f"out_{name}.nc", engine="h5netcdf") as dataset:
            assert dataset[name].encoding["chunksizes"] == (2, 3, 500)


class TestEnergyRegistration:
    def test_aggregate_indices_do_not_run_the_energy_indices(self):
        for aggregate in ("scaled", "all"):
            assert not set(cli_main._INDEX_PIPELINES[aggregate]) & set(_INDICES)

    @pytest.mark.parametrize("index", _INDICES)
    def test_energy_indices_compute_through_xarray_only(self, index):
        registration = cli_main._registry_for(index)

        assert cli_main._INDEX_PIPELINES[index] == (index,)
        assert registration.kernel is None
        assert registration.compute is None
        assert registration.write is None

    def test_the_cli_parses_the_energy_arguments(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(cli_main, "process_climate_indices", lambda arguments: captured.update(vars(arguments)))

        cli_main.main(
            [
                "--index",
                "wind_profile",
                "--periodicity",
                "daily",
                "--netcdf_wind",
                "wind.nc",
                "--var_name_wind",
                _WIND_VARIABLE,
                "--hub_height",
                "100",
                "--wind_profile_method",
                "log",
                "--roughness_length",
                "0.03",
                "--output_file_base",
                "out",
            ]
        )

        assert captured["netcdf_wind"] == "wind.nc"
        assert captured["var_name_wind"] == _WIND_VARIABLE
        assert captured["hub_height"] == _HUB_HEIGHT
        assert captured["wind_profile_method"] == "log"
        assert captured["roughness_length"] == _ROUGHNESS

    def test_the_cli_rejects_an_unknown_wind_profile_method(self):
        with pytest.raises(SystemExit):
            cli_main.main(
                [
                    "--index",
                    "wind_profile",
                    "--periodicity",
                    "daily",
                    "--wind_profile_method",
                    "cubic",
                    "--output_file_base",
                    "out",
                ]
            )
