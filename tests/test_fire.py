"""Tests for the fire-weather indices: Fosberg FFWI (#808) and HDW (#809)."""

from __future__ import annotations

import logging
from unittest import mock

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from climate_indices import fire, pm_eto
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.exceptions import CoordinateValidationError, InvalidArgumentError


@pytest.fixture(scope="module", autouse=True)
def disable_logging():
    """Silence the calculation lifecycle events during these tests."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def test_public_api_is_namespaced() -> None:
    """Expose the fire module, never its functions, at package level."""
    import climate_indices

    assert climate_indices.fire is fire
    assert "fire" in climate_indices.__all__
    for name in fire.__all__:
        assert not hasattr(climate_indices, name)


def _emc(temperature_fahrenheit: float, relative_humidity_percent: float) -> float:
    return float(
        fire._equilibrium_moisture_content(
            np.asarray(temperature_fahrenheit, dtype=np.float64),
            np.asarray(relative_humidity_percent, dtype=np.float64),
        )
    )


# ------------------------------------------------------------------------------
# equilibrium moisture content


@pytest.mark.parametrize(
    ("relative_humidity", "expected"),
    [
        pytest.param(5.0, 1.235355, id="low_range"),
        pytest.param(30.0, 5.99582, id="mid_range"),
        pytest.param(80.0, 16.06068, id="high_range"),
    ],
)
def test_emc_matches_simard_on_each_range(relative_humidity: float, expected: float) -> None:
    """Hand-evaluated Simard (1968) regressions at 70 F."""
    assert _emc(70.0, relative_humidity) == pytest.approx(expected, rel=1e-12)


def test_emc_breakpoints_belong_to_the_lower_range() -> None:
    """10% and 50% use the lower regression, as in NCEP's GEMPAK code.

    The published equations are written "h < 10" and "10 < h <= 50", which
    leaves exactly 10% in neither range; GEMPAK closes both ranges from above.
    """
    just_above_10 = np.nextafter(10.0, np.inf)
    just_above_50 = np.nextafter(50.0, np.inf)

    assert _emc(70.0, 10.0) == pytest.approx(2.43842, rel=1e-12)
    assert _emc(70.0, just_above_10) == pytest.approx(2.79368, rel=1e-9)
    assert _emc(70.0, 50.0) == pytest.approx(9.19796, rel=1e-12)
    assert _emc(70.0, just_above_50) == pytest.approx(9.58815, rel=1e-9)


def test_emc_is_discontinuous_at_the_breakpoints() -> None:
    """The three regressions do not meet, so continuity cannot be asserted.

    Pinned so that nobody "fixes" the published equations into a smooth curve
    by accident. At 10% the jump even changes sign with temperature.
    """
    below_10, above_10 = _emc(32.0, 10.0), _emc(32.0, np.nextafter(10.0, np.inf))
    below_50, above_50 = _emc(32.0, 50.0), _emc(32.0, np.nextafter(50.0, np.inf))
    assert above_10 - below_10 == pytest.approx(0.697412, abs=1e-6)
    assert above_50 - below_50 == pytest.approx(0.493398, abs=1e-6)

    hot_jump = _emc(110.0, np.nextafter(10.0, np.inf)) - _emc(110.0, 10.0)
    assert hot_jump == pytest.approx(-0.0049, abs=1e-6)


# ------------------------------------------------------------------------------
# damping coefficient and index


def test_calibration_point() -> None:
    """Zero moisture and a 30 mph wind give 100, to the precision of 0.3002."""
    value = float(fire._ffwi(np.asarray(0.0), np.asarray(30.0)))
    assert value == pytest.approx(99.988881, abs=1e-6)
    assert abs(value - 100.0) < 0.02


def test_damping_changes_sign_exactly_at_30() -> None:
    """eta = (1 - x)(0.5x^2 - x + 1) with the second factor always positive."""
    x = np.linspace(0.0, 1.5, 301)
    polynomial = 1.0 - 2.0 * x + 1.5 * x**2 - 0.5 * x**3
    np.testing.assert_allclose(polynomial, (1.0 - x) * (0.5 * x**2 - x + 1.0), atol=1e-12)
    assert np.all(0.5 * x**2 - x + 1.0 > 0.0)

    damping = fire._moisture_damping(np.array([0.0, 15.0, 30.0, 40.0]))
    assert damping[0] == pytest.approx(1.0)
    assert 0.0 < damping[1] < 1.0
    assert damping[2] == pytest.approx(0.0, abs=1e-12)
    assert damping[3] == pytest.approx(0.0, abs=1e-12)


def test_saturated_cold_air_gives_zero_not_a_negative_index() -> None:
    """Below about -43 C at 100% humidity the moisture content exceeds 30."""
    assert _emc(-58.0, 100.0) > 30.0
    for cap in (True, False):
        assert float(fire.fosberg_ffwi(-50.0, 100.0, 5.0, cap_at_100=cap)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("temperature", "humidity", "wind", "expected"),
    [
        pytest.param(30.0, 15.0, 10.0, 59.242107, id="hot_dry_breezy"),
        pytest.param(25.0, 20.0, 10.0, 55.430582, id="warm_dry_breezy"),
    ],
)
def test_reference_values(temperature: float, humidity: float, wind: float, expected: float) -> None:
    assert float(fire.fosberg_ffwi(temperature, humidity, wind)) == pytest.approx(expected, rel=1e-7)


def test_cap_is_applied_by_default_and_can_be_turned_off() -> None:
    assert float(fire.fosberg_ffwi(40.0, 5.0, 25.0)) == pytest.approx(100.0)
    assert float(fire.fosberg_ffwi(40.0, 5.0, 25.0, cap_at_100=False)) == pytest.approx(172.5894, abs=1e-4)


def _gempak_pd_fosb(tmpc: np.ndarray, relh: np.ndarray, sped: np.ndarray) -> np.ndarray:
    """Transcription of NCEP GEMPAK's pd_fosb, in float32 with its own constants.

    Note this shares its Simard/GEMPAK coefficients with ``fire.py`` by
    construction, so `test_matches_ncep_gempak` below checks formula
    structure, branch selection, and unit conversion against an
    independently-written float32 implementation; it cannot catch a
    coefficient mistranscribed identically in both places. The reference
    values in `test_reference_values` and `test_emc_matches_simard_on_each_range`
    are pinned directly from these same coefficients too. An authoritative,
    independently-sourced FFWI dataset (e.g. a captured real GEMPAK run) would
    close that gap; none was available while writing this test.
    """
    f32 = np.float32
    tmpc, relh, sped = (np.asarray(a, dtype=f32) for a in (tmpc, relh, sped))
    tf = tmpc * f32(9.0 / 5.0) + f32(32.0)
    smph = sped * f32(1.9425) / f32(0.868976)
    fw = np.where(
        relh <= f32(10.0),
        f32(0.03229) + f32(0.281073) * relh - f32(0.000578) * relh * tf,
        np.where(
            relh <= f32(50.0),
            f32(2.22749) + f32(0.160107) * relh - f32(0.014784) * tf,
            f32(21.0606) + f32(0.005565) * relh * relh - f32(0.00035) * relh * tf - f32(0.483199) * relh,
        ),
    )
    fwd = fw / f32(30.0)
    damping = f32(1.0) - f32(2.0) * fwd + f32(1.5) * fwd**2 - f32(0.5) * fwd**3
    return damping * np.sqrt(f32(1.0) + smph * smph) / f32(0.3002)


def test_matches_ncep_gempak() -> None:
    """Agrees with NCEP's operational implementation behind FOSINDX.

    The tolerance covers GEMPAK's m/s to mph factor, 1.9425 / 0.868976, which is
    0.069% below the exact 1 / 0.44704. Temperatures stay above -10 C, where
    GEMPAK's unclamped moisture content cannot exceed 30.
    """
    temperature, humidity, wind = np.meshgrid(
        np.arange(-10.0, 46.0, 5.0),
        np.array([0.0, 3.0, 10.0, 10.5, 25.0, 50.0, 50.5, 70.0, 100.0]),
        np.array([0.0, 1.0, 3.0, 7.0, 12.0, 20.0, 30.0]),
        indexing="ij",
    )
    ours = fire.fosberg_ffwi(temperature, humidity, wind, cap_at_100=False)
    np.testing.assert_allclose(ours, _gempak_pd_fosb(temperature, humidity, wind), rtol=1e-3)


# ------------------------------------------------------------------------------
# inputs


def test_out_of_range_inputs_are_nan() -> None:
    result = fire.fosberg_ffwi(
        np.array([20.0, 20.0, 20.0, 20.0, np.nan, 20.0]),
        np.array([-1.0, 100.5, 40.0, 40.0, 40.0, np.nan]),
        np.array([5.0, 5.0, -0.1, 5.0, 5.0, 5.0]),
    )
    assert np.isnan(result[[0, 1, 2, 4, 5]]).all()
    assert np.isfinite(result[3])


def test_masked_inputs_are_nan() -> None:
    """Masked elements count as missing, like NaN, not as the data under the mask."""
    humidity = np.ma.masked_array([20.0, 20.0, 20.0], mask=[False, True, False])
    wind = np.ma.masked_array([5, 5, 5], mask=[False, False, True])  # integer data too
    result = fire.fosberg_ffwi(25.0, humidity, wind)
    assert not np.ma.isMaskedArray(result)
    assert np.isfinite(result[0])
    assert np.isnan(result[1:]).all()


def test_edges_of_the_valid_range_are_valid() -> None:
    result = fire.fosberg_ffwi(20.0, np.array([0.0, 100.0]), 0.0)
    assert np.isfinite(result).all()


def test_inputs_broadcast() -> None:
    assert fire.fosberg_ffwi(20.0, 30.0, 5.0).shape == ()
    grid = fire.fosberg_ffwi(np.zeros((3, 1)), np.full((1, 4), 30.0), 5.0)
    assert grid.shape == (3, 4)


def test_incompatible_shapes_raise() -> None:
    with pytest.raises(InvalidArgumentError, match="must broadcast together") as exc_info:
        fire.fosberg_ffwi(np.zeros(3), np.zeros(4), 5.0)

    assert exc_info.value.argument_name == (
        "temperature_celsius/relative_humidity_percent/wind_speed_meters_per_second"
    )
    assert exc_info.value.valid_values == "Arrays broadcastable to a common shape"
    assert "(3,)" in exc_info.value.argument_value
    assert "(4,)" in exc_info.value.argument_value


def test_accepts_plain_python_sequences() -> None:
    """Non-numpy array-likes broadcast and coerce through the public API."""
    result = fire.fosberg_ffwi([20.0, 25.0], [30, 90], [1, 10])
    assert result.shape == (2,)
    assert np.isfinite(result).all()


def test_calculation_failure_logs_and_propagates() -> None:
    """An internal failure emits calculation_failed and re-raises, not swallowed."""
    mock_logger = mock.MagicMock()
    mock_logger.bind.return_value = mock_logger

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic failure")

    with (
        mock.patch.object(fire, "_logger", mock_logger),
        mock.patch.object(fire, "_equilibrium_moisture_content", side_effect=_raise),
        pytest.raises(RuntimeError, match="synthetic failure"),
    ):
        fire.fosberg_ffwi(20.0, 30.0, 5.0)

    mock_logger.info.assert_called_once_with("calculation_started")
    failed_calls = [call for call in mock_logger.error.call_args_list if call.args[0] == "calculation_failed"]
    assert len(failed_calls) == 1
    assert failed_calls[0].kwargs["error_type"] == "RuntimeError"
    assert "synthetic failure" in failed_calls[0].kwargs["error_message"]


def test_large_array_memory_metrics_are_logged() -> None:
    """calculation_completed includes memory metrics when check_large_array_memory reports them."""
    mock_logger = mock.MagicMock()
    mock_logger.bind.return_value = mock_logger

    with (
        mock.patch.object(fire, "_logger", mock_logger),
        mock.patch.object(fire, "check_large_array_memory", return_value={"array_memory_mb": 1234.5}),
    ):
        result = fire.fosberg_ffwi(20.0, 30.0, 5.0)

    assert np.isfinite(result)
    completed_calls = [call for call in mock_logger.info.call_args_list if call.args[0] == "calculation_completed"]
    assert len(completed_calls) == 1
    assert completed_calls[0].kwargs["array_memory_mb"] == 1234.5


def test_chunked_time_axis_matches_eager() -> None:
    """Elementwise, so chunking along time changes nothing."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("dask")

    rng = np.random.default_rng(808)
    shape = (48, 3)
    dims = ("time", "station")
    temperature = xr.DataArray(rng.uniform(-5.0, 40.0, shape), dims=dims)
    humidity = xr.DataArray(rng.uniform(0.0, 100.0, shape), dims=dims)
    wind = xr.DataArray(rng.uniform(0.0, 20.0, shape), dims=dims)

    eager = fire.fosberg_ffwi(temperature.values, humidity.values, wind.values)
    chunked = xr.apply_ufunc(
        fire.fosberg_ffwi,
        temperature.chunk({"time": 12}),
        humidity.chunk({"time": 12}),
        wind.chunk({"time": 12}),
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    np.testing.assert_array_equal(chunked.compute().values, eager)


# ------------------------------------------------------------------------------
# properties

_temperature = st.floats(min_value=-60.0, max_value=55.0, allow_nan=False)
_humidity = st.floats(min_value=0.0, max_value=100.0, allow_nan=False)
_wind = st.floats(min_value=0.0, max_value=60.0, allow_nan=False)


@given(temperature=_temperature, humidity=_humidity, wind=_wind)
@settings(max_examples=200, deadline=None)
def test_capped_index_stays_within_0_and_100(temperature: float, humidity: float, wind: float) -> None:
    value = float(fire.fosberg_ffwi(temperature, humidity, wind))
    assert 0.0 <= value <= 100.0


@given(temperature=_temperature, humidity=_humidity, wind=_wind)
@settings(max_examples=200, deadline=None)
def test_uncapped_index_is_never_negative(temperature: float, humidity: float, wind: float) -> None:
    assert float(fire.fosberg_ffwi(temperature, humidity, wind, cap_at_100=False)) >= 0.0


@given(temperature=_temperature, humidity=_humidity, wind=_wind, extra=_wind)
@settings(max_examples=200, deadline=None)
def test_index_does_not_decrease_with_wind(temperature: float, humidity: float, wind: float, extra: float) -> None:
    slower = float(fire.fosberg_ffwi(temperature, humidity, wind, cap_at_100=False))
    faster = float(fire.fosberg_ffwi(temperature, humidity, wind + extra, cap_at_100=False))
    assert faster >= slower - 1e-12


# ------------------------------------------------------------------------------
# Hot-Dry-Windy Index (#809)


def _hdw(
    temperature: float,
    humidity: float,
    wind: float,
    height: float = 10.0,
) -> float:
    return float(fire.hot_dry_windy([temperature], [humidity], [wind], [height]))


@pytest.mark.parametrize(
    ("temperature", "humidity", "expected_vpd"),
    [
        pytest.param(0.0, 20.0, 4.9, id="freezing"),
        pytest.param(30.0, 20.0, 34.0, id="hot"),
    ],
)
def test_hdw_vpd_matches_srock_examples(temperature: float, humidity: float, expected_vpd: float) -> None:
    """Srock et al. (2018) state these VPD values for 20% relative humidity,
    rounded to 0.1 hPa."""
    vpd = _hdw(temperature, humidity, 1.0)
    assert vpd == pytest.approx(expected_vpd, abs=0.1)


def test_hdw_reference_value() -> None:
    """Hand-evaluated from FAO-56 Eq 11: es(30 C) = 42.437 hPa, VPD = 36.071."""
    assert _hdw(30.0, 15.0, 8.0) == pytest.approx(288.528424, rel=1e-7)


def test_hdw_takes_the_maximum_product_not_the_maximum_factors() -> None:
    """The layer maximum is over the per-level product, as the issue contract
    specifies; the level with the highest wind does not win on its own."""
    result = fire.hot_dry_windy(
        [30.0, 26.0],  # VPD 36.07 vs 23.54 hPa
        [15.0, 30.0],
        [8.0, 12.0],  # second level windier, but its product is smaller
        [10.0, 400.0],
    )
    assert float(result) == pytest.approx(36.071 * 8.0, rel=1e-3)


def test_hdw_excludes_levels_above_500_m() -> None:
    """A larger product above the layer must not leak into the index."""
    result = fire.hot_dry_windy([20.0, 40.0], [50.0, 5.0], [5.0, 30.0], [250.0, 500.5])
    assert float(result) == pytest.approx(_hdw(20.0, 50.0, 5.0), rel=1e-12)


def test_hdw_layer_top_is_inclusive() -> None:
    result = fire.hot_dry_windy([20.0, 30.0], [50.0, 15.0], [5.0, 8.0], [500.0, 250.0])
    assert float(result) == pytest.approx(288.528424, rel=1e-7)


def test_hdw_column_without_a_layer_level_is_nan() -> None:
    result = fire.hot_dry_windy(
        [[30.0, 30.0], [26.0, 26.0]],
        [[15.0, 15.0], [30.0, 30.0]],
        [[8.0, 8.0], [12.0, 12.0]],
        [[10.0, 600.0], [400.0, 900.0]],
        level_axis=0,
    )
    assert np.isfinite(result[0])
    assert np.isnan(result[1])


def test_hdw_nan_height_excludes_the_level() -> None:
    result = fire.hot_dry_windy([40.0, 20.0], [5.0, 50.0], [30.0, 5.0], [np.nan, 250.0])
    assert float(result) == pytest.approx(_hdw(20.0, 50.0, 5.0), rel=1e-12)


def test_hdw_invalid_in_layer_propagates_but_above_layer_is_ignored() -> None:
    """An invalid observation at an in-layer level poisons the column; the same
    observation above the layer is irrelevant."""
    poisoned = fire.hot_dry_windy([np.nan, 20.0], [30.0, 50.0], [5.0, 5.0], [10.0, 250.0])
    assert np.isnan(poisoned)

    out_of_range = fire.hot_dry_windy([20.0, 20.0], [101.0, 50.0], [5.0, 5.0], [10.0, 250.0])
    assert np.isnan(out_of_range)

    above_layer = fire.hot_dry_windy([20.0, np.nan], [50.0, 30.0], [5.0, 5.0], [10.0, 800.0])
    assert np.isfinite(above_layer)


def test_hdw_invalid_above_layer_does_not_warn() -> None:
    mock_logger = mock.MagicMock()
    mock_logger.bind.return_value = mock_logger

    with mock.patch.object(fire, "_logger", mock_logger):
        result = fire.hot_dry_windy([20.0, 20.0], [50.0, 101.0], [5.0, 5.0], [10.0, 800.0])

    assert np.isfinite(result)
    mock_logger.warning.assert_not_called()


def test_hdw_masked_inputs_are_nan() -> None:
    humidity = np.ma.masked_array([15.0, 30.0], mask=[True, False])
    result = fire.hot_dry_windy([30.0, 26.0], humidity, [8.0, 12.0], [10.0, 400.0])
    assert not np.ma.isMaskedArray(result)
    assert np.isnan(result)


def test_hdw_scalar_inputs_are_a_single_level_profile() -> None:
    assert float(fire.hot_dry_windy(30.0, 15.0, 8.0, 10.0)) == pytest.approx(288.528424, rel=1e-7)
    assert np.isnan(fire.hot_dry_windy(30.0, 15.0, 8.0, 600.0))


def test_hdw_level_axis_selects_the_reduced_dimension() -> None:
    temperature = np.array([[30.0, 20.0], [26.0, 18.0]])  # (level, column)
    humidity = np.array([[15.0, 50.0], [30.0, 60.0]])
    wind = np.array([[8.0, 3.0], [12.0, 4.0]])
    height = np.array([10.0, 400.0])

    result = fire.hot_dry_windy(temperature, humidity, wind, height, level_axis=0)
    transposed = fire.hot_dry_windy(temperature.T, humidity.T, wind.T, np.array([10.0, 400.0]), level_axis=-1)
    np.testing.assert_array_equal(result, transposed)
    assert result.shape == (2,)


def test_hdw_empty_level_axis_is_nan() -> None:
    result = fire.hot_dry_windy(np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)), np.empty(0), level_axis=0)
    assert result.shape == (2,)
    assert np.isnan(result).all()


def test_hdw_level_axis_out_of_range_raises() -> None:
    with pytest.raises(InvalidArgumentError, match="level_axis") as exc_info:
        fire.hot_dry_windy([30.0, 26.0], [15.0, 30.0], [8.0, 12.0], [10.0, 400.0], level_axis=2)
    assert exc_info.value.argument_name == "level_axis"


def test_hdw_incompatible_shapes_raise() -> None:
    with pytest.raises(InvalidArgumentError, match="must broadcast together") as exc_info:
        fire.hot_dry_windy(np.zeros(3), np.zeros(4), 5.0, 10.0)
    assert "(3,)" in exc_info.value.argument_value
    assert "(4,)" in exc_info.value.argument_value


def test_hdw_calculation_failure_logs_and_propagates() -> None:
    mock_logger = mock.MagicMock()
    mock_logger.bind.return_value = mock_logger

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic failure")

    with (
        mock.patch.object(fire, "_logger", mock_logger),
        mock.patch.object(fire.pm_eto, "saturation_vapor_pressure", side_effect=_raise),
        pytest.raises(RuntimeError, match="synthetic failure"),
    ):
        fire.hot_dry_windy(30.0, 15.0, 8.0, 10.0)

    failed_calls = [call for call in mock_logger.error.call_args_list if call.args[0] == "calculation_failed"]
    assert len(failed_calls) == 1
    assert failed_calls[0].kwargs["error_type"] == "RuntimeError"


def _srock_hdw_bolton(
    temperature_celsius: np.ndarray,
    relative_humidity_percent: np.ndarray,
    wind_speed: np.ndarray,
    height_agl: np.ndarray,
) -> np.ndarray:
    """Independent transcription: Bolton (1980) saturation vapor pressure."""
    es_hpa = 6.112 * np.exp(17.67 * temperature_celsius / (temperature_celsius + 243.5))
    vpd = es_hpa * (1.0 - relative_humidity_percent / 100.0)
    product = np.where(height_agl <= 500.0, vpd * wind_speed, -np.inf)
    return np.max(product, axis=0)


def test_hdw_matches_independent_reference() -> None:
    """Agrees with a Bolton-form transcription; the SVP formulas differ by <0.5%."""
    rng = np.random.default_rng(809)
    shape = (5, 4, 3)  # (level, y, x)
    temperature = rng.uniform(-5.0, 45.0, shape)
    humidity = rng.uniform(0.0, 100.0, shape)
    wind = rng.uniform(0.0, 30.0, shape)
    height = np.linspace(0.0, 800.0, shape[0])[:, None, None] * np.ones(shape[1:])

    ours = fire.hot_dry_windy(temperature, humidity, wind, height, level_axis=0)
    np.testing.assert_allclose(ours, _srock_hdw_bolton(temperature, humidity, wind, height), rtol=5e-3)


def test_hdw_chunked_time_and_space_match_eager() -> None:
    """Dask chunking along time/space changes nothing; the level axis stays whole."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("dask")

    rng = np.random.default_rng(8090)
    shape = (48, 4, 3, 5)
    dims = ("time", "y", "x", "level")
    arrays = {
        name: xr.DataArray(values, dims=dims)
        for name, values in {
            "temperature": rng.uniform(-5.0, 40.0, shape),
            "humidity": rng.uniform(0.0, 100.0, shape),
            "wind": rng.uniform(0.0, 30.0, shape),
        }.items()
    }
    height = xr.DataArray(np.linspace(0.0, 700.0, shape[-1]), dims=("level",))

    eager = fire.hot_dry_windy(
        arrays["temperature"].values, arrays["humidity"].values, arrays["wind"].values, height.values
    )
    chunked = xr.apply_ufunc(
        fire.hot_dry_windy,
        *(a.chunk({"time": 12, "x": 1}) for a in arrays.values()),
        height,
        input_core_dims=[["level"]] * 4,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    np.testing.assert_array_equal(chunked.compute().values, eager)


def _hdw_profile_dataarrays(xr, *, shape=(6, 3, 2, 4), dims=("time", "y", "x", "level")):  # noqa: ANN001, ANN202
    """Build temperature/humidity/wind/height DataArrays sharing ``dims``, level last."""
    rng = np.random.default_rng(80917)
    level_len = shape[dims.index("level")]
    arrays = {
        name: xr.DataArray(values, dims=dims)
        for name, values in {
            "temperature": rng.uniform(-5.0, 40.0, shape),
            "humidity": rng.uniform(0.0, 100.0, shape),
            "wind": rng.uniform(0.0, 30.0, shape),
        }.items()
    }
    height = xr.DataArray(np.linspace(0.0, 700.0, level_len), dims=("level",))
    return arrays["temperature"], arrays["humidity"], arrays["wind"], height


def test_hdw_xarray_matches_numpy_and_drops_level() -> None:
    """The adapter's result equals the NumPy core's, with level dropped and other dims/coords kept."""
    xr = pytest.importorskip("xarray")

    temperature, humidity, wind, height = _hdw_profile_dataarrays(xr)
    temperature = temperature.assign_coords(time=np.arange(temperature.sizes["time"]))

    result = fire.hot_dry_windy(temperature, humidity, wind, height)

    expected = fire.hot_dry_windy(temperature.values, humidity.values, wind.values, height.values, level_axis=-1)
    assert isinstance(result, xr.DataArray)
    assert "level" not in result.dims
    assert result.dims == ("time", "y", "x")
    np.testing.assert_allclose(result.values, expected)
    np.testing.assert_array_equal(result.coords["time"].values, temperature.coords["time"].values)


def test_hdw_xarray_metadata_matches_registry() -> None:
    xr = pytest.importorskip("xarray")

    temperature, humidity, wind, height = _hdw_profile_dataarrays(xr)
    result = fire.hot_dry_windy(temperature, humidity, wind, height)

    for key in ("long_name", "units", "description", "references"):
        assert result.attrs[key] == CF_METADATA["hdw"][key]


def test_hdw_xarray_chunked_matches_eager() -> None:
    """Chunking every dimension but level through the public adapter changes nothing."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("dask")

    temperature, humidity, wind, height = _hdw_profile_dataarrays(xr)
    eager = fire.hot_dry_windy(temperature, humidity, wind, height)

    chunked_inputs = (a.chunk({"time": 2, "x": 1}) for a in (temperature, humidity, wind))
    chunked = fire.hot_dry_windy(*chunked_inputs, height)
    assert chunked.chunks is not None
    np.testing.assert_allclose(chunked.compute().values, eager.values)


def test_hdw_xarray_level_chunked_raises() -> None:
    xr = pytest.importorskip("xarray")
    pytest.importorskip("dask")

    temperature, humidity, wind, height = _hdw_profile_dataarrays(xr)
    temperature = temperature.chunk({"level": 1})

    with pytest.raises(CoordinateValidationError, match="level"):
        fire.hot_dry_windy(temperature, humidity, wind, height)


def test_hdw_xarray_missing_level_dim_raises() -> None:
    xr = pytest.importorskip("xarray")

    temperature, humidity, wind, height = _hdw_profile_dataarrays(xr)
    temperature = temperature.rename({"level": "plev"})

    with pytest.raises(CoordinateValidationError, match="level"):
        fire.hot_dry_windy(temperature, humidity, wind, height)


def test_hdw_xarray_level_axis_non_default_raises() -> None:
    xr = pytest.importorskip("xarray")

    temperature, humidity, wind, height = _hdw_profile_dataarrays(xr)
    with pytest.raises(InvalidArgumentError, match="level_axis") as exc_info:
        fire.hot_dry_windy(temperature, humidity, wind, height, level_axis=0)
    assert exc_info.value.argument_name == "level_axis"


def test_hdw_xarray_mixed_input_types_raise() -> None:
    xr = pytest.importorskip("xarray")

    temperature, humidity, wind, height = _hdw_profile_dataarrays(xr)
    with pytest.raises(TypeError, match="same type"):
        fire.hot_dry_windy(temperature, humidity.values, wind, height)


def test_hdw_xarray_height_accepts_1d_list_and_nd_dataarray() -> None:
    xr = pytest.importorskip("xarray")

    temperature, humidity, wind, height = _hdw_profile_dataarrays(xr)

    from_list = fire.hot_dry_windy(temperature, humidity, wind, list(height.values))
    np.testing.assert_allclose(from_list.values, fire.hot_dry_windy(temperature, humidity, wind, height).values)

    height_nd = height.broadcast_like(temperature)
    from_nd = fire.hot_dry_windy(temperature, humidity, wind, height_nd)
    np.testing.assert_allclose(from_nd.values, fire.hot_dry_windy(temperature, humidity, wind, height).values)


def test_hdw_xarray_height_bad_shape_raises() -> None:
    xr = pytest.importorskip("xarray")

    temperature, humidity, wind, _height = _hdw_profile_dataarrays(xr)
    with pytest.raises(InvalidArgumentError, match="height_agl_meters"):
        fire.hot_dry_windy(temperature, humidity, wind, [[1.0, 2.0], [3.0, 4.0]])


def test_hdw_xarray_temperature_kelvin_units_converted() -> None:
    xr = pytest.importorskip("xarray")

    temperature, humidity, wind, height = _hdw_profile_dataarrays(xr)
    celsius_result = fire.hot_dry_windy(temperature, humidity, wind, height)

    kelvin_temperature = (temperature + 273.15).assign_attrs(units="K")
    kelvin_result = fire.hot_dry_windy(kelvin_temperature, humidity, wind, height)
    np.testing.assert_allclose(kelvin_result.values, celsius_result.values)
    assert kelvin_result.attrs["units"] == CF_METADATA["hdw"]["units"]


# ------------------------------------------------------------------------------
# HDW properties

_profiles = st.integers(min_value=1, max_value=8).flatmap(
    lambda n: st.tuples(
        st.lists(_temperature, min_size=n, max_size=n),
        st.lists(_humidity, min_size=n, max_size=n),
        st.lists(_wind, min_size=n, max_size=n),
        st.lists(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False), min_size=n, max_size=n),
    )
)


@given(profiles=_profiles)
@settings(max_examples=200, deadline=None)
def test_hdw_is_non_negative_and_bounds_the_per_level_products(
    profiles: tuple[list[float], list[float], list[float], list[float]],
) -> None:
    temperatures, humidities, winds, heights = profiles
    result = float(fire.hot_dry_windy(temperatures, humidities, winds, heights))
    if not any(h <= 500.0 for h in heights):
        assert np.isnan(result)
        return
    assert result >= 0.0
    for t, h, w, z in zip(temperatures, humidities, winds, heights, strict=True):
        if z <= 500.0:
            es_hpa = float(pm_eto.saturation_vapor_pressure(t)) * 10.0
            assert result >= es_hpa * (1.0 - h / 100.0) * w - 1e-9


@given(profiles=_profiles, extra=_temperature, extra_h=_humidity, extra_w=_wind)
@settings(max_examples=200, deadline=None)
def test_hdw_adding_a_layer_level_never_decreases_the_index(
    profiles: tuple[list[float], list[float], list[float], list[float]],
    extra: float,
    extra_h: float,
    extra_w: float,
) -> None:
    """Finer vertical sampling can only reveal worse combinations."""
    temperatures, humidities, winds, heights = profiles
    before = float(fire.hot_dry_windy(temperatures, humidities, winds, heights))
    after = float(
        fire.hot_dry_windy([*temperatures, extra], [*humidities, extra_h], [*winds, extra_w], [*heights, 250.0])
    )
    if np.isnan(before):
        assert np.isfinite(after)
    else:
        assert after >= before - 1e-12
