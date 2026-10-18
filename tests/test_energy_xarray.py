"""Energy DataArray contracts: NumPy parity, units, labels, metadata, and lazy blocks."""

from __future__ import annotations

import warnings
from typing import Any
from unittest import mock

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback

from climate_indices import energy
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.energy import _wind
from climate_indices.exceptions import (
    BetaFeatureWarning,
    CoordinateValidationError,
    InputTypeError,
    InvalidArgumentError,
)

pytestmark = pytest.mark.filterwarnings("ignore::climate_indices.exceptions.BetaFeatureWarning")

CASES: list[tuple[str, dict[str, Any]]] = [
    ("heating_degree_days", {"base_temperature": 18.0}),
    ("cooling_degree_days", {"base_temperature": 18.0}),
    ("energy_degree_days", {"heating_base_temperature": 18.0, "cooling_base_temperature": 24.0}),
    ("hub_height_wind_speed", {"reference_height": 10.0, "hub_height": 100.0}),
    (
        "hub_height_wind_speed",
        {"reference_height": 10.0, "hub_height": 100.0, "method": "log", "roughness_length": 0.03},
    ),
    ("wind_shear_exponent", {}),
]


def _data(name: str) -> xr.DataArray:
    values = np.array([[0.0, 10.0, 20.0], [30.0, np.nan, 10.0], [20.0, 0.0, 30.0], [10.0, 20.0, 0.0]])
    return xr.DataArray(
        values,
        dims=("time", "lat"),
        coords={"time": pd.date_range("2001-02-27", periods=4), "lat": [10.0, 20.0, 30.0]},
        attrs={
            "units": "degC" if "degree_days" in name else "m s-1",
            "standard_name": "air_temperature" if "degree_days" in name else "wind_speed",
            "units_metadata": "temperature: on_scale",
            "cell_methods": "time: mean",
            "history": "prior processing",
            "source": "synthetic",
        },
        name="weather",
    )


def _call(name: str, data: Any, kwargs: dict[str, Any]) -> Any:
    function = getattr(energy, name)
    if name == "wind_shear_exponent":
        return function(data, data * 1.5, **kwargs)
    return function(data, **kwargs)


@pytest.mark.parametrize("name,kwargs", CASES)
@pytest.mark.parametrize("transpose", [False, True])
def test_exact_numpy_parity_and_coordinate_preservation(name: str, kwargs: dict[str, Any], transpose: bool) -> None:
    data = _data(name)
    if transpose:
        data = data.transpose("lat", "time")
    original = data.copy(deep=True)
    result = _call(name, data, kwargs)
    expected = _call(name, data.values, kwargs)
    np.testing.assert_array_equal(result.values, expected)
    assert result.dims == data.dims
    for coordinate in data.coords:
        xr.testing.assert_identical(result.coords[coordinate].variable, data.coords[coordinate].variable)
    xr.testing.assert_identical(data, original)
    assert result.attrs["history"].startswith("prior processing\n")
    assert "climate_indices_version" in result.attrs
    assert result.attrs["source"] == "synthetic"
    assert "cell_methods" not in result.attrs
    if name in CF_METADATA:
        for key, value in CF_METADATA[name].items():
            assert result.attrs[key] == value
    else:
        assert result.attrs["units"] == "1"
    if name in {"energy_degree_days", "wind_shear_exponent"}:
        assert "standard_name" not in result.attrs
    if "degree_days" not in name:
        assert "units_metadata" not in result.attrs


@pytest.mark.parametrize("name,kwargs", CASES)
def test_dask_stays_lazy_with_multiple_time_and_spatial_chunks(name: str, kwargs: dict[str, Any]) -> None:
    data = _data(name)
    lazy = data.chunk({"time": 2, "lat": 1})

    def reject_task(*args: Any) -> None:
        pytest.fail("The adapter evaluated a Dask task before compute().")

    with Callback(pretask=reject_task), pytest.warns(BetaFeatureWarning):  # type: ignore[no-untyped-call]
        result = _call(name, lazy, kwargs)
    assert result.chunks == lazy.chunks
    expected = _call(name, data.values, kwargs)
    np.testing.assert_array_equal(result.compute(scheduler="synchronous").values, expected)


@pytest.mark.parametrize("name,kwargs", CASES[:3])
@pytest.mark.parametrize("units", ["K", "degree_Celsius", "degF", None])
def test_temperature_unit_inference_preserves_attrs(name: str, kwargs: dict[str, Any], units: str | None) -> None:
    celsius = _data(name)
    data = celsius.copy(deep=True)
    if units == "K":
        data.data = data.data + 273.15
    elif units == "degF":
        data.data = data.data * 9 / 5 + 32
    if units is None:
        data.attrs.pop("units")
    else:
        data.attrs["units"] = units
    result = _call(name, data, kwargs)
    np.testing.assert_array_equal(result.values, _call(name, celsius.values, kwargs))
    assert result.attrs["history"].startswith("prior processing\n")
    assert result.attrs["source"] == "synthetic"
    assert result.attrs["units"] == "K d"
    assert result.attrs["units_metadata"] == "temperature: difference"


def test_physical_coordinates_and_explicit_celsius_bases() -> None:
    data = _data("heating_degree_days")
    heating = energy.heating_degree_days(data, base_temperature=18)
    assert heating.air_temperature_threshold.item() == 18
    assert heating.air_temperature_threshold.attrs == {"standard_name": "air_temperature_threshold", "units": "degC"}
    combined = energy.energy_degree_days(data, heating_base_temperature=18, cooling_base_temperature=24)
    assert combined.heating_base_temperature.item() == 18
    assert combined.cooling_base_temperature.item() == 24
    wind = energy.hub_height_wind_speed(_data("hub_height_wind_speed").assign_coords(height=10.0), 10, 100)
    assert wind.height.item() == 100
    assert wind.height.attrs == {"standard_name": "height", "units": "m", "positive": "up"}


def test_labeled_alpha_broadcasts_without_positional_pairing() -> None:
    wind = _data("hub_height_wind_speed").chunk({"time": 2, "lat": 1})
    alpha = xr.DataArray([0.1, 0.2], dims="scenario", coords={"scenario": ["low", "high"]}).chunk({"scenario": 1})
    result = energy.hub_height_wind_speed(wind, 10, 100, alpha=alpha)
    expected = energy.hub_height_wind_speed(wind.compute().values[..., None], 10, 100, alpha=alpha.values)
    assert result.dims == ("time", "lat", "scenario")
    np.testing.assert_array_equal(result.compute().values, expected)


@pytest.mark.parametrize("companion", ["alpha", "upper_wind"])
@pytest.mark.parametrize("dimension", ["time", "lat"])
def test_mismatched_shared_coordinates_are_rejected(companion: str, dimension: str) -> None:
    wind = _data("hub_height_wind_speed")
    other = wind.isel({dimension: slice(None, None, -1)})
    if companion == "alpha":
        other = other.assign_attrs(units="1")
    function = getattr(energy, "hub_height_wind_speed" if companion == "alpha" else "wind_shear_exponent")
    kwargs = (
        {"reference_height": 10, "hub_height": 100, "alpha": other}
        if companion == "alpha"
        else {"wind_speed_upper": other}
    )
    with pytest.raises(CoordinateValidationError, match="must match exactly"):
        function(wind, **kwargs)


@pytest.mark.parametrize("units", ["km/h", "knots", 42, ""])
def test_unsupported_wind_units_are_rejected(units: Any) -> None:
    wind = _data("hub_height_wind_speed")
    wind.attrs["units"] = units
    with pytest.raises(InvalidArgumentError, match="units"):
        energy.hub_height_wind_speed(wind, 10, 100)


@pytest.mark.parametrize("units", ["m s-1", "m/s", "m s**-1", "m s^-1", "m s⁻¹", None])
def test_native_wind_unit_aliases(units: str | None) -> None:
    wind = xr.DataArray([2.0, 4.0], dims="site")
    if units is not None:
        wind.attrs["units"] = units
    actual = energy.hub_height_wind_speed(wind, 10, 100)
    np.testing.assert_array_equal(actual.values, energy.hub_height_wind_speed(wind.values, 10, 100))


@pytest.mark.parametrize("companion", ["alpha", "upper_wind"])
@pytest.mark.parametrize("unlabeled", ["primary", "secondary", "both"])
@pytest.mark.parametrize("lazy", [False, True])
def test_shared_dimensions_require_labels(companion: str, unlabeled: str, lazy: bool) -> None:
    wind = xr.DataArray([2.0, 4.0], dims="site", coords={"site": ["a", "b"]})
    other = wind.copy()
    if unlabeled in {"primary", "both"}:
        wind = wind.drop_vars("site")
    if unlabeled in {"secondary", "both"}:
        other = other.drop_vars("site")
    if lazy:
        wind, other = wind.chunk(site=1), other.chunk(site=1)
    function = getattr(energy, "hub_height_wind_speed" if companion == "alpha" else "wind_shear_exponent")
    kwargs = (
        {"reference_height": 10, "hub_height": 100, "alpha": other}
        if companion == "alpha"
        else {"wind_speed_upper": other}
    )
    with pytest.raises(CoordinateValidationError, match="site"):
        function(wind, **kwargs)


@pytest.mark.parametrize("name,kwargs", CASES[:3])
def test_lazy_auxiliary_time_is_rejected_without_computation(name: str, kwargs: dict[str, Any]) -> None:
    data = _data(name).rename(time="sample").drop_vars("sample")
    time = xr.DataArray(pd.date_range("2001-01-01", periods=4).values, dims="sample").chunk(sample=2).data
    data = data.assign_coords(time=("sample", time)).chunk(sample=2)
    assert data.time.chunks is not None

    def reject_task(*args: Any) -> None:
        pytest.fail("Daily time validation evaluated a Dask task.")

    with Callback(pretask=reject_task), pytest.raises(CoordinateValidationError, match="lazy"):  # type: ignore[no-untyped-call]
        _call(name, data, kwargs)


@pytest.mark.parametrize("name,kwargs", CASES[3:])
def test_wind_blocks_do_not_repeat_operational_side_effects(name: str, kwargs: dict[str, Any]) -> None:
    data = _data(name)
    expected = _call(name, data.values, kwargs)
    with mock.patch.object(_wind, "_logger") as logger, mock.patch.object(_wind, "check_large_array_memory") as memory:
        result = _call(name, data.chunk(time=1, lat=1), kwargs)
        np.testing.assert_array_equal(result.compute(scheduler="synchronous").values, expected)
    logger.bind.assert_not_called()
    memory.assert_not_called()


@pytest.mark.parametrize("roughness", [None, 0.0, -1.0, np.nan, np.inf, 10.0, 100.0])
@pytest.mark.parametrize("reference,hub", [(10.0, 100.0), (100.0, 10.0)])
def test_lazy_log_profile_rejects_invalid_roughness_before_dispatch(
    roughness: float | None, reference: float, hub: float
) -> None:
    wind = _data("hub_height_wind_speed").chunk(time=1, lat=1)

    def reject_task(*args: Any) -> None:
        pytest.fail("Roughness validation evaluated a Dask task.")

    with Callback(pretask=reject_task), pytest.raises(InvalidArgumentError, match="roughness_length"):  # type: ignore[no-untyped-call]
        energy.hub_height_wind_speed(wind, reference, hub, method="log", roughness_length=roughness)


@pytest.mark.parametrize("name,kwargs", CASES[:3])
def test_degree_day_output_replaces_scalar_base_coordinate_metadata(name: str, kwargs: dict[str, Any]) -> None:
    coordinates = (
        {"air_temperature_threshold": 18.0}
        if name != "energy_degree_days"
        else {
            "heating_base_temperature": 18.0,
            "cooling_base_temperature": 24.0,
        }
    )
    data = _data(name).assign_coords({key: xr.DataArray(-99.0, attrs={"source": "input"}) for key in coordinates})
    original = data.copy(deep=True)
    result = _call(name, data, kwargs)
    for key, value in coordinates.items():
        assert result.coords[key].item() == value
        assert result.coords[key].attrs["units"] == "degC"
        assert "source" not in result.coords[key].attrs
    xr.testing.assert_identical(data, original)


def test_unlabeled_non_scalar_companions_and_mixed_winds_are_rejected() -> None:
    wind = _data("hub_height_wind_speed")
    with pytest.raises(InvalidArgumentError, match="labeled"):
        energy.hub_height_wind_speed(wind, 10, 100, alpha=[0.1, 0.2, 0.3])
    with pytest.raises(InputTypeError, match="wind_speed_lower"):
        energy.wind_shear_exponent(wind.values, wind)
    with pytest.raises(InputTypeError, match="wind_speed_upper"):
        energy.wind_shear_exponent(wind, wind.values)
    with pytest.raises(InputTypeError, match="wind_speed"):
        energy.hub_height_wind_speed(wind.values, 10, 100, alpha=wind)


@pytest.mark.parametrize("name,kwargs", CASES)
def test_numpy_calls_do_not_emit_beta_warnings(name: str, kwargs: dict[str, Any]) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _call(name, _data(name).values, kwargs)
    assert not any(issubclass(w.category, BetaFeatureWarning) for w in caught)


@pytest.mark.parametrize("name,kwargs", CASES[:3])
def test_degree_days_require_daily_time_when_attached(name: str, kwargs: dict[str, Any]) -> None:
    data = _data(name).assign_coords(time=pd.date_range("2001-01-01", periods=4, freq="h"))
    with pytest.raises(CoordinateValidationError):
        _call(name, data, kwargs)


def test_unknown_temperature_units_and_non_numeric_inputs_are_rejected() -> None:
    temperature = xr.DataArray([10.0], dims="site", attrs={"units": "unknown"})
    with pytest.raises(InvalidArgumentError):
        energy.heating_degree_days(temperature, base_temperature=18)
    for invalid in [xr.DataArray(["10"]), xr.DataArray([True])]:
        with pytest.raises(InputTypeError):
            energy.heating_degree_days(invalid, base_temperature=18)


def test_scalar_coordinate_cannot_replace_a_dimension() -> None:
    wind = xr.DataArray([2.0, 4.0], dims="height", coords={"height": [10, 20]})
    with pytest.raises(InvalidArgumentError, match="conflicts"):
        energy.hub_height_wind_speed(wind, 10, 100)


@pytest.mark.parametrize("name,kwargs", CASES[:3])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("units,value", [("K", 280.0), ("degF", 50.1)])
@pytest.mark.parametrize("lazy", [False, True])
def test_temperature_conversion_uses_float64_before_arithmetic(
    name: str, kwargs: dict[str, Any], dtype: Any, units: str, value: float, lazy: bool
) -> None:
    raw = np.array([value], dtype=dtype)
    celsius = raw.astype(np.float64)
    celsius = celsius - 273.15 if units == "K" else (celsius - 32.0) * 5.0 / 9.0
    data = xr.DataArray(raw, dims="site", attrs={"units": units})
    if lazy:
        data = data.chunk({"site": 1})
    np.testing.assert_array_equal(_call(name, data, kwargs).compute().values, _call(name, celsius, kwargs))


@pytest.mark.parametrize("bad_time", [["not-a-date"], [np.datetime64("NaT", "ns")], [1]])
def test_single_sample_time_coordinate_still_requires_datetime64(bad_time: Any) -> None:
    data = xr.DataArray([10.0], dims="time", coords={"time": bad_time})
    with pytest.raises(CoordinateValidationError, match="datetime64"):
        energy.heating_degree_days(data, base_temperature=18)


@pytest.mark.parametrize("companion", ["alpha", "upper_wind"])
@pytest.mark.parametrize("scalar", [False, True])
def test_auxiliary_geographic_coordinates_must_match(companion: str, scalar: bool) -> None:
    wind = _data("hub_height_wind_speed").rename(lat="x")
    latitude: Any = 10.0 if scalar else ("x", [10.0, 20.0, 30.0])
    wind = wind.assign_coords(latitude=latitude)
    other = wind.assign_coords(latitude=wind.latitude + 1)
    if companion == "alpha":
        other = other.assign_attrs(units="1")
    function = getattr(energy, "hub_height_wind_speed" if companion == "alpha" else "wind_shear_exponent")
    kwargs = (
        {"reference_height": 10, "hub_height": 100, "alpha": other}
        if companion == "alpha"
        else {"wind_speed_upper": other}
    )
    with pytest.raises(CoordinateValidationError, match="latitude"):
        function(wind, **kwargs)


def test_curvilinear_coordinates_are_checked_without_dask_computation() -> None:
    latitude = np.arange(6).reshape(2, 3)
    wind = xr.DataArray(
        np.full((2, 2, 3), 2.0),
        dims=("time", "y", "x"),
        coords={"time": [0, 1], "y": [0, 1], "x": [0, 1, 2], "latitude": (("y", "x"), latitude)},
    ).chunk({"time": 1, "y": 1, "x": 2})

    def reject_task(*args: Any) -> None:
        pytest.fail("Coordinate validation evaluated a Dask task.")

    with Callback(pretask=reject_task):  # type: ignore[no-untyped-call]
        alpha = wind.assign_attrs(units="1")
        actual = energy.hub_height_wind_speed(wind, 10, 100, alpha=alpha)
        mismatched = wind.assign_coords(latitude=(("y", "x"), latitude + 1))
        with pytest.raises(CoordinateValidationError, match="latitude"):
            energy.wind_shear_exponent(wind, mismatched)
    assert actual.chunks == wind.chunks
    np.testing.assert_array_equal(
        actual.compute().values, energy.hub_height_wind_speed(wind.values, 10, 100, alpha=2.0)
    )


@pytest.mark.parametrize("name,kwargs", CASES)
def test_coordinate_attributes_do_not_depend_on_global_keep_attrs(name: str, kwargs: dict[str, Any]) -> None:
    data = _data(name)
    data.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}
    data.time.attrs = {"bounds": "time_bounds", "axis": "T"}
    if "degree_days" in name:
        data.data = data.data + 273.15
        data.attrs["units"] = "K"
    original = data.copy(deep=True)
    with xr.set_options(keep_attrs=False):  # type: ignore[no-untyped-call]
        result = _call(name, data, kwargs)
    assert result.lat.attrs == original.lat.attrs
    assert result.time.attrs == original.time.attrs
    xr.testing.assert_identical(data, original)


def test_target_height_replaces_scalar_cf_height_alias() -> None:
    wind = _data("hub_height_wind_speed").assign_coords(
        z=xr.DataArray(10.0, attrs={"standard_name": "height", "units": "m"})
    )
    result = energy.hub_height_wind_speed(wind, 10, 100)
    assert "z" not in result.coords
    assert result.height.item() == 100
    assert wind.z.item() == 10
    varying = wind.drop_vars("z").assign_coords(
        z=xr.DataArray([10, 20, 30], dims="lat", attrs={"standard_name": "height"})
    )
    with pytest.raises(InvalidArgumentError, match="varying height"):
        energy.hub_height_wind_speed(varying, 10, 100)


def test_scalar_dask_alpha_remains_lazy() -> None:
    wind = _data("hub_height_wind_speed").chunk({"time": 2, "lat": 1})
    alpha = da.arange(100, chunks="auto").mean() / 100

    def reject_task(*args: Any) -> None:
        pytest.fail("The scalar Dask exponent was evaluated eagerly.")

    with Callback(pretask=reject_task):  # type: ignore[no-untyped-call]
        actual = energy.hub_height_wind_speed(wind, 10, 100, alpha=alpha)
    expected = energy.hub_height_wind_speed(wind.values, 10, 100, alpha=0.495)
    np.testing.assert_array_equal(actual.compute().values, expected)


@pytest.mark.parametrize("units", ["percent", "m", 42])
def test_labeled_alpha_requires_dimensionless_units(units: Any) -> None:
    wind = _data("hub_height_wind_speed")
    alpha = xr.DataArray(20.0, attrs={"units": units})
    with pytest.raises(InvalidArgumentError, match="dimensionless"):
        energy.hub_height_wind_speed(wind, 10, 100, alpha=alpha)


def test_canonical_varying_height_is_rejected() -> None:
    wind = xr.DataArray([2.0, 4.0], dims="site", coords={"height": ("site", [10, 20])})
    with pytest.raises(InvalidArgumentError, match="varying height"):
        energy.hub_height_wind_speed(wind, 10, 100)


def test_lazy_scalar_height_does_not_trigger_coordinate_computation() -> None:
    lower = xr.DataArray(da.ones(2, chunks=1), dims="site", coords={"site": ["a", "b"]}).assign_coords(
        height=xr.DataArray(da.zeros(()), attrs={"standard_name": "height"})
    )
    upper = lower.assign_coords(height=xr.DataArray(da.ones(()) * 100, attrs={"standard_name": "height"}))

    def reject_task(*args: Any) -> None:
        pytest.fail("A lazy scalar height was evaluated during coordinate comparison.")

    with Callback(pretask=reject_task):  # type: ignore[no-untyped-call]
        with pytest.raises(CoordinateValidationError, match="height"):
            energy.wind_shear_exponent(lower, upper)
        result = energy.wind_shear_exponent(lower, lower)
    assert result.chunks == lower.chunks


def test_lazy_shared_coordinate_requires_matching_dimensions() -> None:
    latitude = xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("y", "x")).chunk({"y": 1, "x": 1}).data
    lower = xr.DataArray(np.ones((2, 2)), dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]}).assign_coords(
        latitude=(("y", "x"), latitude)
    )
    upper = lower.assign_coords(latitude=(("x", "y"), latitude))
    assert lower.latitude.data is upper.latitude.data
    with pytest.raises(CoordinateValidationError, match="latitude"):
        energy.wind_shear_exponent(lower, upper)


def test_equal_lazy_coordinate_graphs_are_accepted_without_computation() -> None:
    wind = xr.DataArray(da.ones((2, 2), chunks=1), dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]}).assign_coords(
        latitude=(("y", "x"), xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("y", "x")).chunk({"y": 1, "x": 1}).data)
    )

    def reject_task(*args: Any) -> None:
        pytest.fail("Equal lazy coordinate graphs must not be computed.")

    with Callback(pretask=reject_task):  # type: ignore[no-untyped-call]
        result = energy.wind_shear_exponent(wind, wind.copy())
    assert result.chunks == wind.chunks


def test_varying_hub_height_coordinate_is_rejected() -> None:
    wind = xr.DataArray([2.0, 4.0], dims="site", coords={"hub_height": ("site", [10, 20])})
    with pytest.raises(InvalidArgumentError, match="varying height"):
        energy.hub_height_wind_speed(wind, 10, 100)


@pytest.mark.parametrize("tagged_first", [True, False])
def test_scalar_height_exemption_does_not_depend_on_argument_order(tagged_first: bool) -> None:
    tagged = xr.DataArray(
        [2.0, 4.0],
        dims="site",
        coords={"site": ["a", "b"], "height": xr.DataArray(10.0, attrs={"standard_name": "height"})},
    )
    plain = xr.DataArray([3.0, 6.0], dims="site", coords={"site": ["a", "b"], "height": xr.DataArray(100.0)})
    first, second = (tagged, plain) if tagged_first else (plain, tagged)
    result = energy.wind_shear_exponent(first, second)
    np.testing.assert_array_equal(result.values, energy.wind_shear_exponent(first.values, second.values))
