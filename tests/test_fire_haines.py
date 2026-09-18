"""Tests for the Haines Index (#810): NumPy core, profile entry point, and xarray adapter."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import xarray as xr

from climate_indices import fire
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.exceptions import DataShapeError, InputTypeError, InvalidArgumentError

# The published Haines (1988) tables, restated here so a test failure points at
# the implementation rather than at a shared helper. Stability and moisture
# score 1, 2, or 3 against (< first cut point, < second cut point, else), and
# the moisture term pairs its dewpoint with the lower stability temperature for
# the mid and high variants, the upper one for the low variant.
_PUBLISHED: dict[str, dict[str, object]] = {
    "low": {"stability": (4.0, 8.0), "moisture": (6.0, 10.0), "moisture_from_lower": False},
    "mid": {"stability": (6.0, 11.0), "moisture": (6.0, 13.0), "moisture_from_lower": True},
    "high": {"stability": (18.0, 22.0), "moisture": (15.0, 21.0), "moisture_from_lower": True},
}

_EPSILON = 0.01

_PROFILE_LEVELS = (950.0, 850.0, 700.0, 500.0)
_PROFILE_TEMPERATURE = (32.0, 24.0, 12.0, -8.0)
_PROFILE_DEWPOINT = (26.0, 13.0, 2.0, -20.0)


@pytest.fixture(scope="module", autouse=True)
def disable_logging():
    """Silence the calculation lifecycle events during these tests."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _published_index(
    variant: str,
    temperature_lower: float,
    temperature_upper: float,
    dewpoint: float,
) -> float:
    """Plain-Python transcription of the published tables, sharing no code with the library."""
    spec = _PUBLISHED[variant]
    stability_cuts = spec["stability"]
    moisture_cuts = spec["moisture"]
    assert isinstance(stability_cuts, tuple)
    assert isinstance(moisture_cuts, tuple)
    moisture_temperature = temperature_lower if spec["moisture_from_lower"] else temperature_upper

    def score(delta: float, cut_points: tuple[float, float]) -> int:
        if delta < cut_points[0]:
            return 1
        if delta < cut_points[1]:
            return 2
        return 3

    return float(
        score(temperature_lower - temperature_upper, stability_cuts)
        + score(moisture_temperature - dewpoint, moisture_cuts)
    )


def _deltas_around(cut_points: tuple[float, float]) -> list[float]:
    """Deltas just below, at, and just above each cut point."""
    return [
        cut_points[0] - _EPSILON,
        cut_points[0],
        cut_points[1] - _EPSILON,
        cut_points[1],
    ]


def _levels_for(variant: str) -> tuple[float, float, float]:
    """(lower, upper, moisture) pressure levels of a variant."""
    spec = fire._haines._HAINES_VARIANTS[variant]
    return spec.lower_hpa, spec.upper_hpa, spec.moisture_hpa


@pytest.mark.parametrize("variant", sorted(_PUBLISHED))
def test_haines_index_matches_published_cut_points(variant: str) -> None:
    """Every published bin boundary scores as the table says, on both terms."""
    spec = _PUBLISHED[variant]
    stability_cuts = spec["stability"]
    moisture_cuts = spec["moisture"]
    assert isinstance(stability_cuts, tuple)
    assert isinstance(moisture_cuts, tuple)
    moisture_from_lower = bool(spec["moisture_from_lower"])

    for stability_delta in _deltas_around(stability_cuts):
        for moisture_delta in _deltas_around(moisture_cuts):
            temperature_upper = 0.0
            temperature_lower = temperature_upper + stability_delta
            moisture_temperature = temperature_lower if moisture_from_lower else temperature_upper
            dewpoint = moisture_temperature - moisture_delta

            result = fire.haines_index(
                temperature_lower,
                temperature_upper,
                dewpoint,
                variant=variant,
            )
            assert float(result) == _published_index(variant, temperature_lower, temperature_upper, dewpoint)
            assert 2.0 <= float(result) <= 6.0


def test_haines_index_extremes_are_the_index_range() -> None:
    """A dry, unstable layer scores 6 and a moist, stable one scores 2."""
    assert float(fire.haines_index(30.0, 10.0, -10.0, variant="low")) == 6.0
    assert float(fire.haines_index(10.0, 9.0, 9.5, variant="low")) == 2.0
    assert float(fire.haines_index(24.0, 0.0, -8.0, variant="high")) == 6.0


def test_haines_index_returns_integer_valued_float64() -> None:
    result = fire.haines_index([30.0, 20.0], [24.0, 5.0], [13.0, 0.0], variant="low")
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, np.array([5.0, 4.0]))
    assert np.all(result == np.floor(result))


def test_haines_index_broadcasts_its_inputs() -> None:
    """A column of levels and a row of times combine into a grid."""
    temperature_lower = np.array([[32.0], [20.0]])
    temperature_upper = np.array([[24.0, 20.0]])
    dewpoint = np.array([[13.0, 5.0]])
    result = fire.haines_index(temperature_lower, temperature_upper, dewpoint, variant="low")
    assert result.shape == (2, 2)

    expected = np.array(
        [
            [
                float(fire.haines_index(tl, tu, td, variant="low"))
                for tl, tu, td in zip(
                    np.repeat(temperature_lower[i], 2), temperature_upper[0], dewpoint[0], strict=True
                )
            ]
            for i in range(2)
        ]
    )
    np.testing.assert_allclose(result, expected)


def test_haines_index_incompatible_shapes_raise() -> None:
    with pytest.raises(InvalidArgumentError, match="must broadcast together") as exc_info:
        fire.haines_index(np.zeros(3), np.zeros(4), 5.0, variant="low")
    assert "(3,)" in exc_info.value.argument_value
    assert "(4,)" in exc_info.value.argument_value


def test_haines_index_unknown_variant_raises() -> None:
    with pytest.raises(InvalidArgumentError, match="variant") as exc_info:
        fire.haines_index(30.0, 20.0, 10.0, variant="medium")
    assert exc_info.value.argument_name == "variant"
    assert "medium" in exc_info.value.argument_value


def test_haines_index_variant_is_required() -> None:
    with pytest.raises(TypeError):
        fire.haines_index(30.0, 20.0, 10.0)  # type: ignore[call-arg]


def test_haines_index_nan_input_propagates() -> None:
    assert np.isnan(fire.haines_index(np.nan, 20.0, 10.0, variant="low"))
    assert np.isnan(fire.haines_index(30.0, np.nan, 10.0, variant="low"))
    assert np.isnan(fire.haines_index(30.0, 20.0, np.nan, variant="low"))


def test_haines_index_non_finite_input_is_nan() -> None:
    """An infinite observation is not a value to score: withholding the cell
    is the only outcome that does not fabricate an extreme term."""
    assert np.isnan(float(fire.haines_index(np.inf, 20.0, 10.0, variant="low")))
    assert np.isnan(float(fire.haines_index(30.0, np.inf, 10.0, variant="low")))
    assert np.isnan(float(fire.haines_index(30.0, 20.0, -np.inf, variant="low")))


def test_haines_index_masked_inputs_are_nan() -> None:
    dewpoint = np.ma.masked_array([10.0, 5.0], mask=[True, False])
    result = fire.haines_index([30.0, 30.0], [20.0, 20.0], dewpoint, variant="low")
    assert not np.ma.isMaskedArray(result)
    assert np.isnan(result[0])
    assert np.isfinite(result[1])


def test_haines_index_surface_pressure_below_the_variant_level_is_nan() -> None:
    """The low variant needs 950 hPa; a surface pressure below it means the
    level is below ground, which must mask rather than score."""
    for surface_pressure, expected in ((900.0, np.nan), (950.0, 6.0), (1013.25, 6.0)):
        result = float(fire.haines_index(32.0, 24.0, 13.0, variant="low", surface_pressure_hpa=surface_pressure))
        if np.isnan(expected):
            assert np.isnan(result), surface_pressure
        else:
            assert result == expected, surface_pressure


@pytest.mark.parametrize(
    ("variant", "masking_pressure", "valid_pressure"),
    [("low", 900.0, 1000.0), ("mid", 800.0, 900.0), ("high", 650.0, 750.0)],
)
def test_haines_index_each_variant_masks_at_its_own_level(
    variant: str, masking_pressure: float, valid_pressure: float
) -> None:
    arguments = (32.0, 24.0, 13.0)
    assert np.isnan(
        float(fire.haines_index(*arguments, variant=variant, surface_pressure_hpa=masking_pressure)),
    )
    assert np.isfinite(
        float(fire.haines_index(*arguments, variant=variant, surface_pressure_hpa=valid_pressure)),
    )


def test_haines_index_nan_surface_pressure_is_masked() -> None:
    """Pressure supplied but unknown must not let a below-ground level through."""
    assert np.isnan(float(fire.haines_index(32.0, 24.0, 13.0, variant="low", surface_pressure_hpa=np.nan)))


def test_haines_index_surface_pressure_broadcasts_with_the_inputs() -> None:
    result = fire.haines_index(
        [32.0, 32.0],
        [24.0, 24.0],
        [13.0, 13.0],
        variant="low",
        surface_pressure_hpa=[900.0, 1000.0],
    )
    assert np.isnan(result[0])
    assert result[1] == 6.0


def test_haines_index_does_not_extrapolate_without_surface_pressure() -> None:
    """Without the optional pressure the level scores as given, which is the
    caller's contract: the values are already resolved levels."""
    assert float(fire.haines_index(32.0, 24.0, 13.0, variant="low")) == 6.0


def _direct_profile_index(variant: str, elevation: float) -> float:
    """The level-pair call the profile entry point must reduce to for an elevation."""
    lower_index, upper_index, moisture_index = _levels_for(variant)
    return float(
        fire.haines_index(
            _PROFILE_TEMPERATURE[_PROFILE_LEVELS.index(lower_index)],
            _PROFILE_TEMPERATURE[_PROFILE_LEVELS.index(upper_index)],
            _PROFILE_DEWPOINT[_PROFILE_LEVELS.index(moisture_index)],
            variant=variant,
        )
    )


def test_haines_index_from_profile_matches_the_level_call_at_exact_levels() -> None:
    """A profile already on the variant's levels is scored exactly like the level call."""
    for variant, elevation in (("low", 100.0), ("mid", 600.0), ("high", 1500.0)):
        for level in _levels_for(variant):
            assert level in _PROFILE_LEVELS
        result = fire.haines_index_from_profile(_PROFILE_TEMPERATURE, _PROFILE_DEWPOINT, _PROFILE_LEVELS, elevation)
        assert float(result) == _direct_profile_index(variant, elevation), variant


def test_haines_index_from_profile_selects_the_variant_by_elevation() -> None:
    """The 1000 ft and 3000 ft boundaries: below 1000 ft is low, up to and
    including 3000 ft is mid, above that is high."""
    bands = ((304.9, "low"), (305.0, "mid"), (305.1, "mid"), (914.0, "mid"), (914.1, "high"))
    for elevation, variant in bands:
        result = fire.haines_index_from_profile(_PROFILE_TEMPERATURE, _PROFILE_DEWPOINT, _PROFILE_LEVELS, elevation)
        assert float(result) == _direct_profile_index(variant, elevation), (elevation, variant)


def test_haines_index_from_profile_interpolates_in_log_pressure() -> None:
    """A profile denser than the variant's levels interpolates rather than
    snapping to the nearest level."""
    levels = np.array([1000.0, 900.0, 800.0, 750.0, 600.0, 400.0])
    # linear in temperature, so log-pressure interpolation has a closed form
    temperature = 20.0 + 10.0 * (np.log(900.0) - np.log(levels))
    dewpoint = temperature - 20.0
    result = fire.haines_index_from_profile(temperature, dewpoint, levels, 100.0)

    expected = float(
        fire.haines_index(
            20.0 + 10.0 * (np.log(900.0) - np.log(950.0)),
            20.0 + 10.0 * (np.log(900.0) - np.log(850.0)),
            20.0 + 10.0 * (np.log(900.0) - np.log(850.0)) - 20.0,
            variant="low",
        )
    )
    assert float(result) == expected


def test_haines_index_from_profile_withholds_levels_the_profile_does_not_reach() -> None:
    """A low-elevation cell whose profile stops at 900 hPa is NaN, not clamped."""
    levels = np.array([900.0, 850.0, 700.0, 500.0])
    result = fire.haines_index_from_profile(_PROFILE_TEMPERATURE, _PROFILE_DEWPOINT, levels, np.array([100.0, 600.0]))
    assert np.isnan(result[0])
    assert np.isfinite(result[1])


def test_haines_index_from_profile_withholds_levels_above_the_profile_top() -> None:
    """A high-elevation cell whose profile tops out at 600 hPa is NaN, not
    extrapolated past the profile's own top."""
    levels = np.array([1000.0, 850.0, 700.0, 600.0])
    temperature = np.array([32.0, 24.0, 12.0, 6.0])
    dewpoint = np.array([26.0, 13.0, 2.0, -4.0])
    assert np.isnan(float(fire.haines_index_from_profile(temperature, dewpoint, levels, 2000.0)))


def test_haines_index_from_profile_unknown_elevation_is_nan() -> None:
    """An unknown terrain elevation must withhold the cell rather than default
    to a variant: NaN compares false against every band."""
    assert np.isnan(
        float(fire.haines_index_from_profile(_PROFILE_TEMPERATURE, _PROFILE_DEWPOINT, _PROFILE_LEVELS, np.nan))
    )
    masked = np.ma.masked_array([100.0, 2000.0], mask=[True, False])
    result = fire.haines_index_from_profile(
        np.array([_PROFILE_TEMPERATURE, _PROFILE_TEMPERATURE]),
        np.array([_PROFILE_DEWPOINT, _PROFILE_DEWPOINT]),
        np.array(_PROFILE_LEVELS),
        masked,
    )
    assert np.isnan(result[0])
    assert np.isfinite(result[1])


def test_haines_index_from_profile_non_finite_input_is_nan() -> None:
    """Infinity interpolates to infinity and must withhold the score, not land
    in the most severe bin."""
    temperature = np.array([np.inf, 24.0, 12.0, -8.0])
    assert np.isnan(float(fire.haines_index_from_profile(temperature, _PROFILE_DEWPOINT, _PROFILE_LEVELS, 100.0)))
    dewpoint = np.array([26.0, -np.inf, 2.0, -20.0])
    assert np.isnan(float(fire.haines_index_from_profile(_PROFILE_TEMPERATURE, dewpoint, _PROFILE_LEVELS, 100.0)))


def test_haines_index_from_profile_dewpoint_must_span_the_levels() -> None:
    """Broadcasting must not turn one dewpoint into a whole profile: the API
    promises a value per pressure level."""
    temperature = np.array([_PROFILE_TEMPERATURE, _PROFILE_TEMPERATURE])
    dewpoint = np.array([[26.0], [26.0]])
    with pytest.raises(DataShapeError, match="dewpoint_celsius") as exc_info:
        fire.haines_index_from_profile(temperature, dewpoint, np.array(_PROFILE_LEVELS), 100.0)
    assert exc_info.value.actual_shape == (2, 1)


def test_haines_index_from_profile_broadcasts_a_temperature_profile() -> None:
    """A (levels,) temperature against a (cell, levels) dewpoint keeps the
    profile axis last: the cell axis is not pressure."""
    dewpoint = np.array([_PROFILE_DEWPOINT, _PROFILE_DEWPOINT])
    result = fire.haines_index_from_profile(
        _PROFILE_TEMPERATURE, dewpoint, np.array(_PROFILE_LEVELS), np.array([100.0, 100.0])
    )
    assert result.shape == (2,)
    expected = fire.haines_index_from_profile(_PROFILE_TEMPERATURE, _PROFILE_DEWPOINT, _PROFILE_LEVELS, 100.0)
    np.testing.assert_allclose(result, expected)


def test_haines_index_from_profile_single_level_profile_raises() -> None:
    with pytest.raises(DataShapeError, match="at least two") as exc_info:
        fire.haines_index_from_profile([30.0], [10.0], [950.0], 100.0)
    assert exc_info.value.expected_shape is not None


def test_haines_index_from_profile_non_positive_pressure_raises() -> None:
    with pytest.raises(InvalidArgumentError, match="strictly positive") as exc_info:
        fire.haines_index_from_profile(_PROFILE_TEMPERATURE, _PROFILE_DEWPOINT, [950.0, 850.0, 700.0, 0.0], 100.0)
    assert exc_info.value.argument_name == "pressure_hpa"


def test_haines_index_from_profile_elevation_broadcast_failure_raises() -> None:
    from unittest import mock

    temperature = np.zeros((2, 4))
    dewpoint = np.zeros((2, 4))
    mock_logger = mock.MagicMock()
    mock_logger.bind.return_value = mock_logger
    with mock.patch.object(fire._haines, "_logger", mock_logger):
        with pytest.raises(InvalidArgumentError, match="elevation") as exc_info:
            fire.haines_index_from_profile(temperature, dewpoint, np.array(_PROFILE_LEVELS), np.zeros(3))
    assert exc_info.value.argument_name == "elevation_meters"
    assert [call.args[0] for call in mock_logger.error.call_args_list] == ["calculation_failed"]


def test_haines_index_from_profile_emits_lifecycle_events() -> None:
    """The profile entry point reports the same lifecycle events as its siblings."""
    from unittest import mock

    mock_logger = mock.MagicMock()
    mock_logger.bind.return_value = mock_logger

    with mock.patch.object(fire._haines, "_logger", mock_logger):
        fire.haines_index_from_profile(_PROFILE_TEMPERATURE, _PROFILE_DEWPOINT, _PROFILE_LEVELS, 100.0)

    assert [call.args[0] for call in mock_logger.info.call_args_list] == [
        "calculation_started",
        "calculation_completed",
    ]
    mock_logger.bind.assert_called_once()


def test_haines_index_from_profile_automatically_selects_per_cell() -> None:
    """A grid of elevations picks a variant per cell in one call."""
    temperature = np.array([_PROFILE_TEMPERATURE, _PROFILE_TEMPERATURE])
    dewpoint = np.array([_PROFILE_DEWPOINT, _PROFILE_DEWPOINT])
    elevations = np.array([100.0, 2000.0])
    result = fire.haines_index_from_profile(temperature, dewpoint, np.array(_PROFILE_LEVELS), elevations)
    assert result.shape == (2,)
    for cell, variant in zip(result, ("low", "high"), strict=True):
        assert float(cell) == _direct_profile_index(variant, 0.0)


def test_haines_index_from_profile_moves_the_pressure_axis() -> None:
    """pressure_axis picks the profile dimension and drops it from the output."""
    temperature = np.array([_PROFILE_TEMPERATURE, _PROFILE_TEMPERATURE]).T  # (level, cell)
    dewpoint = np.array([_PROFILE_DEWPOINT, _PROFILE_DEWPOINT]).T
    result = fire.haines_index_from_profile(
        temperature, dewpoint, np.array(_PROFILE_LEVELS), np.array([100.0, 2000.0]), pressure_axis=0
    )
    assert result.shape == (2,)
    np.testing.assert_allclose(
        result,
        fire.haines_index_from_profile(temperature.T, dewpoint.T, np.array(_PROFILE_LEVELS), np.array([100.0, 2000.0])),
    )


def test_haines_index_from_profile_decreasing_pressure_is_required() -> None:
    with pytest.raises(InvalidArgumentError, match="strictly decreasing") as exc_info:
        fire.haines_index_from_profile(
            _PROFILE_TEMPERATURE, _PROFILE_DEWPOINT, np.array([500.0, 700.0, 850.0, 950.0]), 100.0
        )
    assert exc_info.value.argument_name == "pressure_hpa"


def test_haines_index_from_profile_length_mismatch_raises() -> None:
    with pytest.raises(DataShapeError, match="levels") as exc_info:
        fire.haines_index_from_profile(_PROFILE_TEMPERATURE, _PROFILE_DEWPOINT, np.array([950.0, 850.0]), 100.0)
    assert exc_info.value.expected_shape is not None


def test_haines_index_from_profile_multidimensional_pressure_raises() -> None:
    with pytest.raises(DataShapeError, match="one-dimensional"):
        fire.haines_index_from_profile(
            _PROFILE_TEMPERATURE, _PROFILE_DEWPOINT, np.array([[950.0], [850.0], [700.0], [500.0]]), 100.0
        )


def test_haines_index_from_profile_pressure_axis_out_of_range_raises() -> None:
    with pytest.raises(DataShapeError, match="pressure_axis"):
        fire.haines_index_from_profile(
            _PROFILE_TEMPERATURE, _PROFILE_DEWPOINT, np.array(_PROFILE_LEVELS), 100.0, pressure_axis=3
        )


def test_haines_index_from_profile_scalar_temperature_raises() -> None:
    with pytest.raises(DataShapeError, match="profile axis"):
        fire.haines_index_from_profile(30.0, 20.0, np.array(_PROFILE_LEVELS), 100.0)


def test_haines_index_from_profile_rejects_xarray_input() -> None:
    temperature = xr.DataArray(list(_PROFILE_TEMPERATURE), dims=["level"])
    dewpoint = xr.DataArray(list(_PROFILE_DEWPOINT), dims=["level"])
    with pytest.raises(InvalidArgumentError, match="does not accept xr.DataArray") as exc_info:
        fire.haines_index_from_profile(temperature, dewpoint, np.array(_PROFILE_LEVELS), 100.0)
    assert exc_info.value.argument_name == "temperature_celsius/dewpoint_celsius"


def _haines_dataarrays(variant_pairs: tuple[float, float, float] = (32.0, 24.0, 13.0)) -> tuple[xr.DataArray, ...]:
    """Level-pair DataArrays on (time, y, x), the shape a gridded call would have."""
    time = np.arange(3)
    y = np.array([40.0, 41.0])
    x = np.array([-105.0, -104.0])
    lower, upper, dewpoint = variant_pairs
    shape = (time.size, y.size, x.size)
    return tuple(  # type: ignore[return-value]
        xr.DataArray(np.full(shape, value), dims=("time", "y", "x"), coords={"time": time, "y": y, "x": x})
        for value in (lower, upper, dewpoint)
    )


def test_haines_xarray_matches_numpy() -> None:
    temperature_lower, temperature_upper, dewpoint = _haines_dataarrays()
    result = fire.haines_index(temperature_lower, temperature_upper, dewpoint, variant="low")

    expected = fire.haines_index(temperature_lower.values, temperature_upper.values, dewpoint.values, variant="low")
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("time", "y", "x")
    np.testing.assert_allclose(result.values, expected)
    np.testing.assert_array_equal(result.coords["time"].values, temperature_lower.coords["time"].values)


def test_haines_xarray_non_finite_input_is_nan() -> None:
    temperature_lower, temperature_upper, dewpoint = _haines_dataarrays()
    temperature_lower.values[0, 0, 0] = -np.inf
    result = fire.haines_index(temperature_lower, temperature_upper, dewpoint, variant="low")
    assert np.isnan(result.values[0, 0, 0])
    assert np.isfinite(result.values[1:, :, :]).all()


def test_haines_index_numpy_input_returns_ndarray_not_dataarray() -> None:
    array_result = fire.haines_index([32.0], [24.0], [13.0], variant="low")
    assert isinstance(array_result, np.ndarray)

    scalar_result = fire.haines_index(32.0, 24.0, 13.0, variant="low")
    assert isinstance(scalar_result, np.ndarray)
    assert scalar_result.shape == ()
    assert not isinstance(scalar_result, xr.DataArray)


@pytest.mark.parametrize("variant", sorted(_PUBLISHED))
def test_haines_xarray_metadata_matches_registry(variant: str) -> None:
    temperature_lower, temperature_upper, dewpoint = _haines_dataarrays()
    result = fire.haines_index(temperature_lower, temperature_upper, dewpoint, variant=variant)

    for key in ("long_name", "units", "description", "references", "climate_indices_variant"):
        assert result.attrs[key] == CF_METADATA[f"haines_{variant}"][key]
    assert result.attrs["climate_indices_variant"] == variant


def test_haines_xarray_surface_pressure_masks_per_cell() -> None:
    temperature_lower, temperature_upper, dewpoint = _haines_dataarrays()
    surface_pressure = xr.DataArray(
        np.full((3, 2, 2), 1000.0), dims=("time", "y", "x"), coords=temperature_lower.coords
    )
    masked = (surface_pressure.coords["x"] == -104.0).values
    surface_pressure = surface_pressure.where(masked, 900.0)

    result = fire.haines_index(
        temperature_lower, temperature_upper, dewpoint, variant="low", surface_pressure_hpa=surface_pressure
    )
    assert np.isnan(result.values[:, :, 0]).all()
    assert np.isfinite(result.values[:, :, 1]).all()


def test_haines_xarray_scalar_surface_pressure_is_accepted() -> None:
    temperature_lower, temperature_upper, dewpoint = _haines_dataarrays()
    result = fire.haines_index(
        temperature_lower, temperature_upper, dewpoint, variant="low", surface_pressure_hpa=900.0
    )
    assert np.isnan(result.values).all()


def test_haines_xarray_plain_array_surface_pressure_raises() -> None:
    """A plain array would not carry dimension labels, so the adapter rejects it."""
    temperature_lower, temperature_upper, dewpoint = _haines_dataarrays()
    with pytest.raises(InvalidArgumentError, match="surface_pressure_hpa") as exc_info:
        fire.haines_index(
            temperature_lower,
            temperature_upper,
            dewpoint,
            variant="low",
            surface_pressure_hpa=np.full((3, 2, 2), 1000.0),
        )
    assert exc_info.value.argument_name == "surface_pressure_hpa"


def test_haines_xarray_chunked_matches_eager() -> None:
    pytest.importorskip("dask")
    temperature_lower, temperature_upper, dewpoint = _haines_dataarrays()
    eager = fire.haines_index(temperature_lower, temperature_upper, dewpoint, variant="low")

    chunked = fire.haines_index(
        *(array.chunk({"time": 2, "y": 1}) for array in (temperature_lower, temperature_upper, dewpoint)),
        variant="low",
    )
    assert chunked.chunks is not None
    np.testing.assert_allclose(chunked.compute().values, eager.values)


def test_haines_xarray_dask_blocks_do_not_log_per_block() -> None:
    """One xarray operation must not emit lifecycle events (or the below-ground
    warning) once per Dask block, while block exceptions still surface from
    compute()."""
    pytest.importorskip("dask")
    from unittest import mock

    temperature_lower, temperature_upper, dewpoint = _haines_dataarrays()
    surface_pressure = xr.DataArray(
        np.full((3, 2, 2), 1000.0), dims=("time", "y", "x"), coords=temperature_lower.coords
    )
    surface_pressure = surface_pressure.where(surface_pressure.coords["x"] == -104.0, 900.0)
    mock_logger = mock.MagicMock()
    mock_logger.bind.return_value = mock_logger

    with mock.patch.object(fire._haines, "_logger", mock_logger):
        chunked = fire.haines_index(
            *(array.chunk({"time": 2, "y": 1}) for array in (temperature_lower, temperature_upper, dewpoint)),
            variant="low",
            surface_pressure_hpa=surface_pressure.chunk({"time": 2, "y": 1}),
        )
        values = chunked.compute().values

    mock_logger.warning.assert_not_called()
    assert np.isnan(values[:, :, 0]).all()
    assert np.isfinite(values[:, :, 1]).all()


def test_haines_xarray_eager_input_still_warns_below_ground() -> None:
    """Only Dask-backed input suppresses the below-ground warning; eager input
    reports the masked cells."""
    from unittest import mock

    temperature_lower, temperature_upper, dewpoint = _haines_dataarrays()
    mock_logger = mock.MagicMock()

    with mock.patch.object(fire._haines, "_logger", mock_logger):
        result = fire.haines_index(
            temperature_lower, temperature_upper, dewpoint, variant="low", surface_pressure_hpa=900.0
        )

    assert np.isnan(result.values).all()
    mock_logger.warning.assert_called_once()
    assert str(result.size) in mock_logger.warning.call_args.args[0]


def test_haines_xarray_non_numeric_scalar_pressure_raises() -> None:
    """A non-numeric scalar pressure is rejected the way the NumPy path rejects
    it, instead of numpy raising a raw ValueError (or silently coercing a
    numeric string)."""
    temperature_lower, temperature_upper, dewpoint = _haines_dataarrays()
    for pressure in ("900", "nope"):
        with pytest.raises(InputTypeError, match="surface_pressure_hpa") as exc_info:
            fire.haines_index(
                temperature_lower, temperature_upper, dewpoint, variant="low", surface_pressure_hpa=pressure
            )
        assert exc_info.value.actual_type is not None

    # the NumPy path rejects the same input
    with pytest.raises(InputTypeError):
        fire.haines_index(32.0, 24.0, 13.0, variant="low", surface_pressure_hpa="900")


def test_haines_xarray_non_numeric_dtype_raises() -> None:
    """The NumPy path rejects non-numeric input; the xarray path must too,
    rather than failing inside numpy or inside compute()."""
    temperature_lower, temperature_upper, dewpoint = _haines_dataarrays()
    strings = xr.DataArray(np.full((3, 2, 2), "nope"), dims=("time", "y", "x"))
    with pytest.raises(InputTypeError, match="numeric") as exc_info:
        fire.haines_index(strings, temperature_upper, dewpoint, variant="low")
    assert exc_info.value.actual_type is not None

    non_numeric_pressure = xr.DataArray(np.full((3, 2, 2), "nope"), dims=("time", "y", "x"))
    with pytest.raises(InputTypeError, match="surface_pressure_hpa"):
        fire.haines_index(
            temperature_lower,
            temperature_upper,
            dewpoint,
            variant="low",
            surface_pressure_hpa=non_numeric_pressure,
        )


def test_haines_xarray_unsupported_units_attribute_names_the_input() -> None:
    temperature_lower, temperature_upper, dewpoint = _haines_dataarrays()
    kelvin = (temperature_lower + 273.15).assign_attrs(units="rankine")
    with pytest.raises(InvalidArgumentError, match="Unsupported temperature units") as exc_info:
        fire.haines_index(kelvin, temperature_upper, dewpoint, variant="low")
    assert exc_info.value.argument_name == "temperature_lower_celsius.attrs['units']"


def test_haines_xarray_fahrenheit_temperatures_converted() -> None:
    temperature_lower, temperature_upper, dewpoint = _haines_dataarrays()
    fahrenheit = (temperature_lower * 9.0 / 5.0 + 32.0).assign_attrs(units="degF")
    assert float(fire.haines_index(fahrenheit, temperature_upper, dewpoint, variant="low").values.flat[0]) == float(
        fire.haines_index(temperature_lower, temperature_upper, dewpoint, variant="low").values.flat[0]
    )


def test_haines_xarray_temperature_kelvin_units_converted() -> None:
    temperature_lower, temperature_upper, dewpoint = _haines_dataarrays()
    kelvin = (temperature_lower + 273.15).assign_attrs(units="K")
    assert float(fire.haines_index(kelvin, temperature_upper, dewpoint, variant="low").values.flat[0]) == float(
        fire.haines_index(temperature_lower, temperature_upper, dewpoint, variant="low").values.flat[0]
    )


def test_haines_xarray_mixed_input_types_raise() -> None:
    temperature_lower, temperature_upper, _ = _haines_dataarrays()
    with pytest.raises(TypeError, match="same type"):
        fire.haines_index(temperature_lower, temperature_upper, np.array(13.0), variant="low")


def test_haines_xarray_extra_dimension_broadcasts() -> None:
    """A dimension on only one input (e.g. a member axis on one level) broadcasts."""
    temperature_lower, temperature_upper, dewpoint = _haines_dataarrays()
    members = xr.DataArray(
        np.full((3, 2, 2, 2), 24.0),
        dims=("time", "y", "x", "member"),
        coords={"member": [0, 1]},
    )
    result = fire.haines_index(temperature_lower, members, dewpoint, variant="low")
    assert set(result.dims) == {"time", "y", "x", "member"}

    expected = fire.haines_index(
        temperature_lower.values[:, :, :, None],
        members.values,
        dewpoint.values[:, :, :, None],
        variant="low",
    )
    np.testing.assert_allclose(result.transpose("time", "y", "x", "member").values, expected)


def test_haines_index_surface_pressure_nan_is_masked_not_scored() -> None:
    """A masked observation at a required level must withhold the cell, and a
    NaN delta must never fall through to the most severe score."""
    result = fire.haines_index([np.nan, 30.0], [20.0, 20.0], [10.0, 10.0], variant="low")
    assert np.isnan(result[0])
    assert np.isfinite(result[1])
