"""Tests for the Keetch-Byram Drought Index (#799) and its xarray adapter (#801)."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st

from climate_indices import fire
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.exceptions import (
    CoordinateValidationError,
    DataShapeError,
    InputAlignmentWarning,
    InvalidArgumentError,
)

# the corrected Equation 18 contract, restated here so a test failure points at
# the implementation rather than at a shared helper
_MAX_MM = 203.2
_THRESHOLD_MM = 5.08
_DRYING_TEMPERATURE_CELSIUS = 10.0


@pytest.fixture(scope="module", autouse=True)
def disable_logging():
    """Silence the calculation lifecycle events during these tests."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _reference_kbdi(
    precipitation_mm: list[float],
    temperature_celsius: list[float],
    mean_annual_precipitation_mm: float,
    initial_kbdi_mm: float = 0.0,
) -> list[float]:
    """Independent day-at-a-time transcription of corrected Equation 18 and the SE-38 rain rules.

    Deliberately written as a plain Python loop with ``math.exp``, sharing no
    code with the production implementation, so it checks the algebra, the
    ordering, and the wet-spell state machine rather than the vectorization.
    """
    kbdi_value = initial_kbdi_mm
    wet_spell = 0.0
    values = []
    for precipitation, temperature in zip(precipitation_mm, temperature_celsius, strict=True):
        net_rain = 0.0
        if precipitation > 0.0:
            event_total = wet_spell + precipitation
            if wet_spell > _THRESHOLD_MM:
                net_rain = precipitation
            elif event_total > _THRESHOLD_MM:
                net_rain = event_total - _THRESHOLD_MM
            wet_spell = event_total
        else:
            wet_spell = 0.0
        kbdi_value = max(0.0, kbdi_value - net_rain)
        if temperature >= _DRYING_TEMPERATURE_CELSIUS and kbdi_value < _MAX_MM:
            drying = (
                (_MAX_MM - kbdi_value)
                * (0.968 * math.exp(0.0875 * temperature + 1.5552) - 8.30)
                / (1.0 + 10.88 * math.exp(-0.001736 * mean_annual_precipitation_mm))
                * 1e-3
            )
            kbdi_value = min(_MAX_MM, kbdi_value + max(drying, 0.0))
        values.append(kbdi_value)
    return values


def _dry_series(days: int, temperature_celsius: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(days), np.full(days, temperature_celsius)


# ------------------------------------------------------------------------------
# corrected Equation 18


def test_first_day_matches_hand_evaluated_equation_18() -> None:
    """203.2 * (0.968 * exp(0.0875 * 20 + 1.5552) - 8.30) / (1 + 10.88 * exp(-1.736)) * 1e-3."""
    assert float(fire.kbdi([0.0], [20.0], 1000.0)[0]) == pytest.approx(1.25945729100979, rel=1e-12)


def test_uses_the_corrected_constant_not_the_misprinted_one() -> None:
    """The literal 0.830 of SE-38 Appendix Equation 18 is a typo; 8.30 is the corrected constant.

    Alexander (1990) shows the misprint gives 7.83 where the original tables
    give 6, so this pins the correction rather than letting it regress.
    """
    literal = (
        _MAX_MM
        * (0.968 * math.exp(0.0875 * 20.0 + 1.5552) - 0.830)
        / (1.0 + 10.88 * math.exp(-0.001736 * 1000.0))
        * 1e-3
    )
    assert float(fire.kbdi([0.0], [20.0], 1000.0)[0]) == pytest.approx(1.25945729100979, rel=1e-12)
    assert abs(literal - 1.25945729100979) > 0.4


def test_days_below_50_fahrenheit_add_no_drought_factor() -> None:
    """The source states drought development requires daily maxima of 50 F or higher."""
    precipitation, _ = _dry_series(3)
    temperatures = np.array([9.99, 10.0, -20.0])
    values = fire.kbdi(precipitation, temperatures, 1000.0, initial_kbdi=100.0)
    assert float(values[0]) == pytest.approx(100.0, abs=1e-12)
    assert float(values[1]) > 100.0
    assert float(values[2]) == pytest.approx(float(values[1]), rel=1e-15)


def test_matches_independent_reference_implementation() -> None:
    """A plain-Python reimplementation of the contract, checked day by day."""
    rng = np.random.default_rng(799)
    days = 365
    precipitation = np.where(rng.random(days) < 0.25, rng.gamma(2.0, 3.0, days), 0.0)
    temperature = rng.uniform(-5.0, 45.0, days)
    mean_annual = 850.0

    expected = _reference_kbdi(precipitation.tolist(), temperature.tolist(), mean_annual)
    values = fire.kbdi(precipitation, temperature, mean_annual)
    np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------------------
# wet spells


def test_threshold_boundary_is_exactly_5_08_mm() -> None:
    """An event of exactly 0.20 in is not effective rain; the next 0.01 mm is."""
    precipitation = np.array([2.54, 2.54, 0.01])
    temperatures = np.full(3, -20.0)
    values = fire.kbdi(precipitation, temperatures, 1000.0, initial_kbdi=100.0)
    assert float(values[0]) == pytest.approx(100.0, abs=1e-12)
    assert float(values[1]) == pytest.approx(100.0, abs=1e-12)
    assert float(values[2]) == pytest.approx(99.99, abs=1e-12)


def test_imperial_threshold_boundary_matches_the_published_examples() -> None:
    """Figure 1's 0.16 + 0.09 event nets 0.05 in, and 0.30 in nets 0.10 in."""
    boundary = fire.kbdi([0.16, 0.09], [-20.0, -20.0], 50.0, units="imperial", initial_kbdi=100.0)
    assert float(boundary[0]) == pytest.approx(100.0, abs=1e-12)
    assert float(boundary[1]) == pytest.approx(95.0, abs=1e-12)

    isolated = fire.kbdi([0.30], [-20.0], 50.0, units="imperial", initial_kbdi=100.0)
    assert float(isolated[0]) == pytest.approx(90.0, abs=1e-12)


def test_a_day_without_measurable_rain_ends_the_wet_spell() -> None:
    """Rain below the threshold only counts once the spell has already crossed it."""
    precipitation = np.array([3.0, 3.0, 2.0, 0.0, 2.0])
    temperatures = np.full(5, -20.0)
    mean_annual = 1000.0

    values = fire.kbdi(precipitation, temperatures, mean_annual, initial_kbdi=100.0)
    expected = _reference_kbdi(precipitation.tolist(), temperatures.tolist(), mean_annual, initial_kbdi_mm=100.0)
    # 3.0 + 3.0 crosses on day two netting 0.92; day three nets its full 2.0;
    # the dry day resets, so the trailing 2.0 mm is below the threshold again.
    assert [float(value) for value in values] == pytest.approx(expected)
    assert float(values[2]) == pytest.approx(97.08, abs=1e-12)
    assert float(values[4]) == pytest.approx(97.08, abs=1e-12)


def test_rain_reduces_the_index_before_the_drought_factor_is_added() -> None:
    """Same-day ordering: subtract net rain, then add today's drought factor."""
    temperatures = np.array([30.0])
    mean_annual = 1000.0
    dry = float(fire.kbdi([0.0], temperatures, mean_annual, initial_kbdi=100.0)[0])
    wet = float(fire.kbdi([20.0], temperatures, mean_annual, initial_kbdi=100.0)[0])

    after_rain = 100.0 - (20.0 - _THRESHOLD_MM)
    expected = (
        after_rain
        + (_MAX_MM - after_rain)
        * (0.968 * math.exp(0.0875 * 30.0 + 1.5552) - 8.30)
        / (1.0 + 10.88 * math.exp(-0.001736 * mean_annual))
        * 1e-3
    )
    assert wet == pytest.approx(expected, rel=1e-12)
    assert wet < dry


# ------------------------------------------------------------------------------
# units


def test_imperial_and_metric_agree_after_unit_conversion() -> None:
    rng = np.random.default_rng(799)
    days = 120
    precipitation_mm = np.where(rng.random(days) < 0.3, rng.gamma(2.0, 4.0, days), 0.0)
    temperature_celsius = rng.uniform(-5.0, 40.0, days)
    mean_annual_mm = 900.0

    metric = fire.kbdi(precipitation_mm, temperature_celsius, mean_annual_mm)
    imperial = fire.kbdi(
        precipitation_mm / 25.4,
        temperature_celsius * 9.0 / 5.0 + 32.0,
        mean_annual_mm / 25.4,
        units="imperial",
    )
    np.testing.assert_allclose(metric, imperial * 0.254, rtol=1e-9, atol=1e-9)


def test_imperial_output_is_hundredths_of_an_inch() -> None:
    """The same dry spell tops out at 203.2 mm metric and 800 imperial."""
    precipitation, temperature = _dry_series(4000, temperature_celsius=45.0)
    metric = float(fire.kbdi(precipitation, temperature, 200.0)[-1])
    imperial = float(fire.kbdi(precipitation, temperature * 9.0 / 5.0 + 32.0, 200.0 / 25.4, units="imperial")[-1])
    assert metric == pytest.approx(203.2, abs=1e-9)
    assert imperial == pytest.approx(800.0, abs=1e-9)


# ------------------------------------------------------------------------------
# bounds and monotonicity


def test_sustained_heavy_rain_drives_the_index_to_zero() -> None:
    """Rain beyond the 5.08 mm event threshold saturates at zero, before any drying."""
    precipitation, temperature = _dry_series(60, temperature_celsius=0.0)
    values = fire.kbdi(precipitation + 200.0, temperature, 500.0, initial_kbdi=150.0)
    assert float(values[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(values[-1]) == pytest.approx(0.0, abs=1e-12)


def test_long_hot_dry_spell_asymptotes_to_the_cap_without_exceeding_it() -> None:
    precipitation, temperature = _dry_series(4000, temperature_celsius=45.0)
    values = fire.kbdi(precipitation, temperature, 200.0)
    assert np.all(np.diff(values) >= 0.0)
    assert float(values[-1]) == pytest.approx(_MAX_MM, abs=1e-9)


_temperature = st.floats(min_value=-30.0, max_value=55.0, allow_nan=False)
_precipitation = st.floats(min_value=0.0, max_value=200.0, allow_nan=False)
_mean_annual = st.floats(min_value=50.0, max_value=5000.0, allow_nan=False)


@given(
    day_one_precipitation=_precipitation,
    day_two_precipitation=_precipitation,
    temperature=_temperature,
    mean_annual=_mean_annual,
)
@settings(max_examples=200, deadline=None)
def test_index_stays_within_the_cap(
    day_one_precipitation: float,
    day_two_precipitation: float,
    temperature: float,
    mean_annual: float,
) -> None:
    values = fire.kbdi(
        [day_one_precipitation, day_two_precipitation],
        [temperature, temperature],
        mean_annual,
        initial_kbdi=100.0,
    )
    assert np.all((values >= 0.0) & (values <= _MAX_MM))


@given(extra_rain=st.floats(min_value=0.0, max_value=100.0), temperature=_temperature, mean_annual=_mean_annual)
@settings(max_examples=200, deadline=None)
def test_more_rain_never_raises_the_index(extra_rain: float, temperature: float, mean_annual: float) -> None:
    drier = float(fire.kbdi([10.0], [temperature], mean_annual, initial_kbdi=100.0)[0])
    wetter = float(fire.kbdi([10.0 + extra_rain], [temperature], mean_annual, initial_kbdi=100.0)[0])
    assert wetter <= drier + 1e-12


# ------------------------------------------------------------------------------
# initialization, spin-up, and return values


def test_initial_kbdi_seeds_the_recurrence() -> None:
    precipitation, temperature = _dry_series(2, temperature_celsius=0.0)
    values = fire.kbdi(precipitation, temperature, 1000.0, initial_kbdi=137.0)
    np.testing.assert_array_equal(values, np.array([137.0, 137.0]))


def test_return_state_is_opt_in() -> None:
    precipitation, temperature = _dry_series(3)
    plain = fire.kbdi(precipitation, temperature, 1000.0)
    result = fire.kbdi(precipitation, temperature, 1000.0, return_state=True)
    assert isinstance(plain, np.ndarray)
    assert isinstance(result, fire.KBDIResult)
    np.testing.assert_array_equal(result.values, plain)
    assert isinstance(result.state, fire.KBDIState)
    assert float(result.state.kbdi) == pytest.approx(float(plain[-1]), rel=1e-15)


def test_spin_up_omits_leading_days_without_changing_the_state() -> None:
    precipitation, temperature = _dry_series(20)
    precipitation = precipitation.copy()
    precipitation[10] = 30.0
    full = fire.kbdi(precipitation, temperature, 1000.0, return_state=True)
    spun = fire.kbdi(precipitation, temperature, 1000.0, spin_up=7, return_state=True)
    np.testing.assert_array_equal(spun.values, full.values[7:])
    np.testing.assert_array_equal(spun.state.kbdi, full.state.kbdi)
    np.testing.assert_array_equal(spun.state.wet_spell_precipitation, full.state.wet_spell_precipitation)


def test_spin_up_longer_than_the_input_yields_an_empty_result() -> None:
    precipitation, temperature = _dry_series(5)
    assert fire.kbdi(precipitation, temperature, 1000.0, spin_up=9).shape == (0,)


def test_numpy_integer_configuration_is_accepted() -> None:
    precipitation, temperature = _dry_series(3)
    np.testing.assert_array_equal(
        fire.kbdi(precipitation, temperature, 1000.0, spin_up=np.int64(1)),
        fire.kbdi(precipitation, temperature, 1000.0)[1:],
    )
    gapped = precipitation.copy()
    gapped[1] = np.nan
    np.testing.assert_array_equal(
        fire.kbdi(gapped, temperature, 1000.0, nan_policy="bridge", max_gap_days=np.int64(1)),
        fire.kbdi(gapped, temperature, 1000.0, nan_policy="bridge", max_gap_days=1),
    )


def test_initial_kbdi_cannot_be_combined_with_initial_state() -> None:
    precipitation, temperature = _dry_series(2)
    state = fire.kbdi(precipitation, temperature, 1000.0, return_state=True).state
    with pytest.raises(InvalidArgumentError, match="initial_kbdi"):
        fire.kbdi(precipitation, temperature, 1000.0, initial_kbdi=10.0, initial_state=state)


def test_initial_state_units_must_match() -> None:
    precipitation, temperature = _dry_series(2)
    state = fire.kbdi(precipitation, temperature, 40.0, units="imperial", return_state=True).state
    with pytest.raises(InvalidArgumentError, match="units"):
        fire.kbdi(precipitation, temperature, 1000.0, initial_state=state)


def test_initial_state_must_be_a_kbdi_state() -> None:
    precipitation, temperature = _dry_series(2)
    with pytest.raises(InvalidArgumentError, match="KBDIState"):
        fire.kbdi(precipitation, temperature, 1000.0, initial_state="not a state")  # type: ignore[arg-type]


@pytest.mark.parametrize("initial_kbdi", [-0.1, 203.3, np.nan, np.inf])
def test_initial_kbdi_must_be_within_range(initial_kbdi: float) -> None:
    precipitation, temperature = _dry_series(2)
    with pytest.raises(InvalidArgumentError, match="initial_kbdi"):
        fire.kbdi(precipitation, temperature, 1000.0, initial_kbdi=initial_kbdi)


def test_invalid_state_bookkeeping_raises() -> None:
    precipitation, temperature = _dry_series(2)
    with pytest.raises(InvalidArgumentError, match="trailing_gap_days"):
        fire.kbdi(
            precipitation,
            temperature,
            1000.0,
            initial_state=fire.KBDIState(
                kbdi=np.asarray(10.0),
                wet_spell_precipitation=np.asarray(0.0),
                trailing_gap_days=np.asarray(0.5),
            ),
        )
    with pytest.raises(InvalidArgumentError, match="wet_spell_precipitation"):
        fire.kbdi(
            precipitation,
            temperature,
            1000.0,
            initial_state=fire.KBDIState(
                kbdi=np.asarray(10.0),
                wet_spell_precipitation=np.asarray(-1.0),
                trailing_gap_days=None,
            ),
        )


def test_nan_state_kbdi_requires_started_gap_bookkeeping() -> None:
    """A not-started recurrence holds a number, so NaN kbdi with no started cell is impossible."""
    precipitation, temperature = _dry_series(2)
    for trailing_gap_days in (None, np.asarray(-1)):
        with pytest.raises(InvalidArgumentError, match="trailing_gap_days"):
            fire.kbdi(
                precipitation,
                temperature,
                1000.0,
                initial_state=fire.KBDIState(
                    kbdi=np.asarray(np.nan),
                    wet_spell_precipitation=np.asarray(0.0),
                    trailing_gap_days=trailing_gap_days,
                ),
            )


def test_gap_poisoned_state_is_accepted() -> None:
    precipitation, temperature = _dry_series(4)
    precipitation = precipitation.copy()
    precipitation[2] = np.nan
    poisoned = fire.kbdi(precipitation, temperature, 1000.0, return_state=True).state
    assert np.isnan(float(poisoned.kbdi))
    assert int(poisoned.trailing_gap_days) >= 0
    assert np.isnan(fire.kbdi([0.0, 0.0], [20.0, 20.0], 1000.0, initial_state=poisoned)).all()


# ------------------------------------------------------------------------------
# mean annual precipitation


def test_mean_annual_precipitation_is_derived_from_a_long_record() -> None:
    days = 30 * 365
    daily_rate = 2.7397  # mm, about 1000 mm/year
    precipitation = np.full((days, 2), daily_rate)
    temperature = np.full((days, 2), 20.0)

    derived = fire.kbdi(precipitation, temperature)
    explicit = fire.kbdi(precipitation, temperature, daily_rate * 365.25)
    np.testing.assert_allclose(derived, explicit, rtol=1e-12)


def test_deriving_mean_annual_precipitation_requires_thirty_years() -> None:
    days = 30 * 365 - 1
    precipitation = np.full(days, 2.0)
    temperature = np.full(days, 20.0)
    with pytest.raises(InvalidArgumentError, match="10,950"):
        fire.kbdi(precipitation, temperature)


def test_mean_annual_precipitation_must_be_positive() -> None:
    precipitation, temperature = _dry_series(2)
    with pytest.raises(InvalidArgumentError, match="mean_annual_precipitation"):
        fire.kbdi(precipitation, temperature, 0.0)


def test_nan_mean_annual_precipitation_yields_an_all_nan_cell() -> None:
    precipitation = np.zeros((4, 2))
    temperature = np.full((4, 2), 20.0)
    values = fire.kbdi(precipitation, temperature, np.array([1000.0, np.nan]))
    assert np.isfinite(values[:, 0]).all()
    assert np.isnan(values[:, 1]).all()


def test_nan_mean_annual_precipitation_is_not_treated_as_a_missing_weather_day() -> None:
    """An unavailable static cell has no recurrence, so it cannot advance or poison gap state."""
    precipitation, temperature = _dry_series(3)
    started = fire.kbdi(precipitation, temperature, 1000.0, return_state=True).state

    unavailable = fire.kbdi(precipitation, temperature, np.nan, initial_state=started, return_state=True)
    assert np.isnan(unavailable.values).all()
    np.testing.assert_array_equal(unavailable.state.kbdi, started.kbdi)
    assert int(unavailable.state.trailing_gap_days) == 0

    bridged = fire.kbdi(
        precipitation,
        temperature,
        np.nan,
        initial_state=started,
        nan_policy="bridge",
        max_gap_days=1,
        return_state=True,
    )
    np.testing.assert_array_equal(bridged.state.kbdi, started.kbdi)
    assert int(bridged.state.trailing_gap_days) == 0

    resumed = fire.kbdi(precipitation, temperature, 1000.0, initial_state=unavailable.state)
    np.testing.assert_array_equal(
        resumed,
        fire.kbdi(precipitation, temperature, 1000.0, initial_state=started),
    )


def test_mean_annual_precipitation_broadcasts_to_cells() -> None:
    """The climate factor advances the index faster where the mean annual rainfall is higher."""
    precipitation = np.zeros((3, 2))
    temperature = np.full((3, 2), 20.0)
    values = fire.kbdi(precipitation, temperature, np.array([500.0, 2000.0]))
    assert values.shape == (3, 2)
    assert np.all(values[:, 0] < values[:, 1])


# ------------------------------------------------------------------------------
# shapes and validation


def test_time_first_shape_is_preserved() -> None:
    precipitation = np.zeros((5, 3, 2))
    temperature = np.full((5, 3, 2), 20.0)
    assert fire.kbdi(precipitation, temperature, 1000.0).shape == (5, 3, 2)


def test_scalar_inputs_have_no_time_dimension() -> None:
    with pytest.raises(DataShapeError) as exc_info:
        fire.kbdi(0.0, 20.0, 1000.0)
    assert exc_info.value.expected_shape == "(time, ...)"


def test_incompatible_shapes_raise() -> None:
    with pytest.raises(InvalidArgumentError, match="broadcast"):
        fire.kbdi(np.zeros(3), np.zeros(4), 1000.0)


def test_negative_precipitation_raises() -> None:
    with pytest.raises(InvalidArgumentError, match="precipitation"):
        fire.kbdi([0.0, -1.0], [20.0, 20.0], 1000.0)


@pytest.mark.parametrize("bound_infinity", [np.inf, -np.inf])
@pytest.mark.parametrize("argument", ["precipitation", "maximum_temperature"])
def test_infinite_weather_values_raise(bound_infinity: float, argument: str) -> None:
    """Infinity is an invalid observation, not a missing day."""
    precipitation, temperature = _dry_series(2)
    weather = {"precipitation": precipitation.tolist(), "maximum_temperature": temperature.tolist()}
    weather[argument][0] = bound_infinity
    with pytest.raises(InvalidArgumentError, match="finite"):
        fire.kbdi(weather["precipitation"], weather["maximum_temperature"], 1000.0)


def test_imperial_values_that_overflow_the_metric_conversion_raise() -> None:
    huge = np.finfo(np.float64).max / 25.0
    with pytest.raises(InvalidArgumentError, match="metric"):
        fire.kbdi([0.0, huge], [70.0, 70.0], 40.0, units="imperial")
    with pytest.raises(InvalidArgumentError, match="metric"):
        fire.kbdi(np.zeros(2), np.full(2, 70.0), huge, units="imperial")


def test_huge_finite_imperial_temperature_is_not_treated_as_missing() -> None:
    """Grouping the Fahrenheit-to-Celsius factor keeps the intermediate product finite."""
    temperature = np.finfo(np.float64).max / 5.0 * 1.0000001
    assert np.isfinite(temperature)
    result = fire.kbdi(np.zeros(2), np.full(2, temperature), 40.0, units="imperial")
    assert np.isfinite(result).all()


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        pytest.param("units", "fahrenheit", id="units"),
        pytest.param("nan_policy", "interpolate", id="nan_policy"),
        pytest.param("max_gap_days", -1, id="negative_gap"),
        pytest.param("spin_up", -1, id="negative_spin_up"),
    ],
)
def test_invalid_configuration_raises(argument: str, value: object) -> None:
    precipitation, temperature = _dry_series(3)
    with pytest.raises(InvalidArgumentError, match=argument):
        fire.kbdi(precipitation, temperature, 1000.0, **{argument: value})


@pytest.mark.parametrize(
    ("nan_policy", "max_gap_days"),
    [("propagate", 1), ("bridge", 0)],
)
def test_gap_policy_and_limit_must_be_consistent(nan_policy: str, max_gap_days: int) -> None:
    precipitation, temperature = _dry_series(3)
    with pytest.raises(InvalidArgumentError, match="max_gap_days"):
        fire.kbdi(precipitation, temperature, 1000.0, nan_policy=nan_policy, max_gap_days=max_gap_days)


# ------------------------------------------------------------------------------
# missing days: propagate


def _gapped_series() -> tuple[np.ndarray, np.ndarray]:
    precipitation, temperature = _dry_series(6)
    precipitation = precipitation.copy()
    precipitation[2] = np.nan
    return precipitation, temperature


def test_propagate_poisons_from_the_first_valid_day_after_an_interior_gap() -> None:
    precipitation, temperature = _gapped_series()
    result = fire.kbdi(precipitation, temperature, 1000.0, return_state=True)
    assert np.isfinite(result.values[:2]).all()
    assert np.isnan(result.values[2:]).all()
    assert np.isnan(float(result.state.kbdi))
    assert int(result.state.trailing_gap_days) == 0


def test_propagate_treats_a_missing_temperature_like_missing_rain() -> None:
    precipitation, temperature = _dry_series(4)
    temperature = temperature.copy()
    temperature[1] = np.nan
    values = fire.kbdi(precipitation, temperature, 1000.0)
    assert np.isfinite(values[0])
    assert np.isnan(values[1:]).all()


def test_propagate_poisons_the_final_state_after_a_trailing_gap() -> None:
    precipitation, temperature = _dry_series(4)
    precipitation = precipitation.copy()
    precipitation[3] = np.nan
    result = fire.kbdi(precipitation, temperature, 1000.0, return_state=True)
    assert np.isnan(result.values[3])
    assert np.isnan(float(result.state.kbdi))


def test_propagate_ignores_leading_missing_days_before_the_start() -> None:
    precipitation, temperature = _dry_series(4)
    precipitation = precipitation.copy()
    precipitation[:2] = np.nan
    result = fire.kbdi(precipitation, temperature, 1000.0, initial_kbdi=100.0, return_state=True)
    assert np.isnan(result.values[:2]).all()
    assert np.isfinite(result.values[2:]).all()
    unpoisoned = fire.kbdi(precipitation[2:], temperature[2:], 1000.0, initial_kbdi=100.0)
    np.testing.assert_array_equal(result.values[2:], unpoisoned)
    assert int(result.state.trailing_gap_days) == 0


def test_propagate_poisons_a_resumed_state_on_leading_missing_days() -> None:
    precipitation, temperature = _dry_series(4)
    state = fire.kbdi(precipitation, temperature, 1000.0, return_state=True).state
    resumed = fire.kbdi(
        [np.nan, 0.0, 0.0],
        list(temperature[:3]),
        1000.0,
        initial_state=state,
        return_state=True,
    )
    assert np.isnan(resumed.values).all()
    assert np.isnan(float(resumed.state.kbdi))


def test_propagate_poisons_only_the_missing_cells() -> None:
    precipitation = np.zeros((4, 2))
    temperature = np.full((4, 2), 20.0)
    precipitation[1, 0] = np.nan
    result = fire.kbdi(precipitation, temperature, 1000.0, return_state=True)
    assert np.isnan(result.values[1:, 0]).all()
    assert np.isfinite(result.values[:, 1]).all()
    assert np.isnan(float(result.state.kbdi[0]))
    assert np.isfinite(float(result.state.kbdi[1]))


def test_propagate_all_nan_input_returns_an_unstarted_state() -> None:
    days = 5
    precipitation = np.full(days, np.nan)
    temperature = np.full(days, np.nan)
    result = fire.kbdi(precipitation, temperature, 1000.0, initial_kbdi=100.0, return_state=True)
    assert np.isnan(result.values).all()
    assert float(result.state.kbdi) == pytest.approx(100.0)
    assert result.state.trailing_gap_days is None


def test_propagate_all_nan_input_poisons_a_started_state() -> None:
    precipitation, temperature = _dry_series(3)
    state = fire.kbdi(precipitation, temperature, 1000.0, return_state=True).state
    result = fire.kbdi(np.full(3, np.nan), np.full(3, np.nan), 1000.0, initial_state=state, return_state=True)
    assert np.isnan(result.values).all()
    assert np.isnan(float(result.state.kbdi))


# ------------------------------------------------------------------------------
# missing days: bridge


@pytest.mark.parametrize("gap_days", [1, 2, 3, 4])
@pytest.mark.parametrize("max_gap_days", [1, 2, 3])
def test_bridge_matrix(gap_days: int, max_gap_days: int) -> None:
    """A run within the limit resumes; the first day past the limit poisons."""
    precipitation, temperature = _dry_series(4 + gap_days + 3)
    precipitation = precipitation.copy()
    precipitation[4 : 4 + gap_days] = np.nan
    values = fire.kbdi(precipitation, temperature, 1000.0, nan_policy="bridge", max_gap_days=max_gap_days)

    assert np.isfinite(values[:4]).all()
    assert np.isnan(values[4 : 4 + gap_days]).all()
    if gap_days <= max_gap_days:
        assert np.isfinite(values[4 + gap_days :]).all()
    else:
        first_poisoned = 4 + max_gap_days
        assert np.isnan(values[first_poisoned:]).all()


def test_bridged_run_equals_running_only_the_valid_days() -> None:
    """Skipped days leave the state untouched, so the recurrence path is identical."""
    precipitation, temperature = _dry_series(12)
    precipitation = precipitation.copy()
    precipitation[2:5] = np.nan
    precipitation[9] = np.nan
    temperature = temperature.copy()
    temperature[9] = np.nan

    valid = np.isfinite(precipitation) & np.isfinite(temperature)
    bridged = fire.kbdi(
        precipitation,
        temperature,
        1000.0,
        initial_kbdi=160.0,
        nan_policy="bridge",
        max_gap_days=3,
    )
    only_valid = fire.kbdi(precipitation[valid], temperature[valid], 1000.0, initial_kbdi=160.0)
    np.testing.assert_array_equal(bridged[valid], only_valid)


def test_bridge_skips_leading_missing_days_before_the_start() -> None:
    precipitation, temperature = _dry_series(5)
    precipitation = precipitation.copy()
    precipitation[:2] = np.nan
    values = fire.kbdi(precipitation, temperature, 1000.0, nan_policy="bridge", max_gap_days=1)
    assert np.isnan(values[:2]).all()
    assert np.isfinite(values[2:]).all()


def test_bridge_trailing_gap_within_the_limit_keeps_the_last_valid_state() -> None:
    precipitation, temperature = _dry_series(6)
    precipitation = precipitation.copy()
    precipitation[4:] = np.nan
    result = fire.kbdi(
        precipitation,
        temperature,
        1000.0,
        nan_policy="bridge",
        max_gap_days=3,
        return_state=True,
    )
    assert np.isfinite(result.values[:4]).all()
    assert np.isnan(result.values[4:]).all()
    np.testing.assert_array_equal(result.state.kbdi, result.values[3])
    assert int(result.state.trailing_gap_days) == 2


def test_bridge_trailing_gap_past_the_limit_poisons_the_state() -> None:
    precipitation, temperature = _dry_series(6)
    precipitation = precipitation.copy()
    precipitation[4:] = np.nan
    result = fire.kbdi(
        precipitation,
        temperature,
        1000.0,
        nan_policy="bridge",
        max_gap_days=1,
        return_state=True,
    )
    assert np.isnan(float(result.state.kbdi))
    assert int(result.state.trailing_gap_days) == 2


def test_bridge_tracks_trailing_gap_days_per_cell() -> None:
    precipitation = np.zeros((4, 2))
    temperature = np.full((4, 2), 20.0)
    precipitation[3, 0] = np.nan
    result = fire.kbdi(
        precipitation,
        temperature,
        1000.0,
        nan_policy="bridge",
        max_gap_days=1,
        return_state=True,
    )
    np.testing.assert_array_equal(result.state.trailing_gap_days, np.array([1, 0]))
    assert np.isfinite(float(result.state.kbdi[0]))


def test_bridge_split_gap_append_poisons_on_the_same_day_as_one_shot() -> None:
    """A gap spanning an append boundary is measured against the continued count."""
    precipitation, temperature = _dry_series(10)
    precipitation = precipitation.copy()
    precipitation[2:7] = np.nan

    one_shot = fire.kbdi(
        precipitation,
        temperature,
        1000.0,
        nan_policy="bridge",
        max_gap_days=2,
        return_state=True,
    )
    first = fire.kbdi(
        precipitation[:4],
        temperature[:4],
        1000.0,
        nan_policy="bridge",
        max_gap_days=2,
        return_state=True,
    )
    assert int(first.state.trailing_gap_days) == 2
    assert np.isfinite(float(first.state.kbdi))

    second = fire.kbdi(
        precipitation[4:],
        temperature[4:],
        1000.0,
        initial_state=first.state,
        nan_policy="bridge",
        max_gap_days=2,
        return_state=True,
    )
    joined = np.concatenate((first.values, second.values))
    np.testing.assert_array_equal(joined, one_shot.values)
    assert np.isnan(joined[4])
    assert np.isnan(float(second.state.kbdi))
    np.testing.assert_array_equal(second.state.kbdi, one_shot.state.kbdi)


# ------------------------------------------------------------------------------
# append/resume round trip


def test_append_resume_round_trip_is_bitwise_identical() -> None:
    rng = np.random.default_rng(799)
    days = 60
    precipitation = np.where(rng.random(days) < 0.3, rng.gamma(2.0, 3.0, days), 0.0)
    temperature = rng.uniform(0.0, 40.0, days)
    mean_annual = 900.0

    one_shot = fire.kbdi(precipitation, temperature, mean_annual, return_state=True)
    first = fire.kbdi(precipitation[:25], temperature[:25], mean_annual, return_state=True)
    second = fire.kbdi(
        precipitation[25:],
        temperature[25:],
        mean_annual,
        initial_state=first.state,
        return_state=True,
    )

    np.testing.assert_array_equal(np.concatenate((first.values, second.values)), one_shot.values)
    np.testing.assert_array_equal(second.state.kbdi, one_shot.state.kbdi)
    np.testing.assert_array_equal(second.state.wet_spell_precipitation, one_shot.state.wet_spell_precipitation)
    np.testing.assert_array_equal(second.state.trailing_gap_days, one_shot.state.trailing_gap_days)


def test_append_resume_round_trip_is_bitwise_identical_per_cell() -> None:
    rng = np.random.default_rng(800)
    days, cells = 40, 3
    precipitation = np.where(rng.random((days, cells)) < 0.3, rng.gamma(2.0, 4.0, (days, cells)), 0.0)
    temperature = rng.uniform(0.0, 40.0, (days, cells))
    mean_annual = np.array([400.0, 900.0, 1500.0])

    one_shot = fire.kbdi(precipitation, temperature, mean_annual, return_state=True)
    first = fire.kbdi(precipitation[:17], temperature[:17], mean_annual, return_state=True)
    second = fire.kbdi(
        precipitation[17:],
        temperature[17:],
        mean_annual,
        initial_state=first.state,
        return_state=True,
    )
    np.testing.assert_array_equal(np.concatenate((first.values, second.values)), one_shot.values)
    np.testing.assert_array_equal(second.state.kbdi, one_shot.state.kbdi)


# ------------------------------------------------------------------------------
# xarray adapter (#801)


def _gridded_inputs(days: int = 120, seed: int = 801) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """A small (time, lat, lon) weather block and matching mean annual precipitation."""
    rng = np.random.default_rng(seed)
    shape = (days, 2, 3)
    precipitation = np.where(rng.random(shape) < 0.3, rng.gamma(2.0, 3.0, shape), 0.0)
    temperature = rng.uniform(-5.0, 35.0, shape)
    mean_annual = np.array([[600.0, 900.0, 1200.0], [800.0, 1000.0, 1400.0]])
    time = pd.date_range("2000-01-01", periods=days, freq="D")
    return precipitation, temperature, mean_annual, time


def _gridded_dataarrays(
    days: int = 120, seed: int = 801
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, np.ndarray, np.ndarray, np.ndarray]:
    """DataArray inputs plus the raw NumPy arrays they wrap, for equivalence checks."""
    precipitation, temperature, mean_annual, time = _gridded_inputs(days, seed)
    dims = ["time", "lat", "lon"]
    coords = {"time": time, "lat": [10.0, 20.0], "lon": [30.0, 40.0, 50.0]}
    precip_da = xr.DataArray(precipitation, dims=dims, coords=coords)
    temp_da = xr.DataArray(temperature, dims=dims, coords=coords)
    mean_annual_da = xr.DataArray(mean_annual, dims=["lat", "lon"], coords={"lat": coords["lat"], "lon": coords["lon"]})
    return precip_da, temp_da, mean_annual_da, precipitation, temperature, mean_annual


class TestKBDIXarrayEquivalence:
    """xarray and NumPy paths must agree exactly on values and final state."""

    def test_values_match_numpy_eager(self) -> None:
        precip_da, temp_da, mean_annual_da, precipitation, temperature, mean_annual = _gridded_dataarrays()
        expected = fire.kbdi(precipitation, temperature, mean_annual)
        result = fire.kbdi(precip_da, temp_da, mean_annual_da)
        assert isinstance(result, xr.DataArray)
        np.testing.assert_array_equal(result.values, expected)

    def test_values_and_state_match_numpy_dask(self) -> None:
        precip_da, temp_da, mean_annual_da, precipitation, temperature, mean_annual = _gridded_dataarrays()
        expected = fire.kbdi(precipitation, temperature, mean_annual, return_state=True)
        precip_dask = precip_da.chunk({"time": -1, "lat": 1, "lon": 1})
        temp_dask = temp_da.chunk({"time": -1, "lat": 1, "lon": 1})
        result = fire.kbdi(precip_dask, temp_dask, mean_annual_da, return_state=True)
        assert isinstance(result.values, xr.DataArray)
        np.testing.assert_array_equal(result.values.values, expected.values)
        np.testing.assert_array_equal(result.state.kbdi, expected.state.kbdi)
        np.testing.assert_array_equal(result.state.wet_spell_precipitation, expected.state.wet_spell_precipitation)

    def test_dims_and_coords_preserved(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        result = fire.kbdi(precip_da, temp_da, mean_annual_da)
        assert result.dims == precip_da.dims
        xr.testing.assert_equal(result.coords["time"], precip_da.coords["time"])
        xr.testing.assert_equal(result.coords["lat"], precip_da.coords["lat"])
        xr.testing.assert_equal(result.coords["lon"], precip_da.coords["lon"])

    def test_derived_mean_annual_precipitation_matches_numpy(self) -> None:
        """Omitting mean_annual_precipitation (derive-from-record path) matches NumPy on the xarray path too.

        This changes _kbdi_xarray's conditional optional-args list (one fewer
        apply_ufunc argument), the specific plumbing this omission exercises.
        """
        days = 30 * 365 + 5  # just over the 10,950-day minimum
        precipitation, temperature, _mean_annual, time = _gridded_inputs(days=days, seed=802)
        dims = ["time", "lat", "lon"]
        coords = {"time": time, "lat": [10.0, 20.0], "lon": [30.0, 40.0, 50.0]}
        precip_da = xr.DataArray(precipitation, dims=dims, coords=coords)
        temp_da = xr.DataArray(temperature, dims=dims, coords=coords)

        expected = fire.kbdi(precipitation, temperature)
        result = fire.kbdi(precip_da, temp_da)
        assert isinstance(result, xr.DataArray)
        np.testing.assert_array_equal(result.values, expected)


class TestKBDIXarrayCFMetadata:
    """The kbdi/kbdi_imperial registry entry must resolve from the call's `units`, not a fixed constant."""

    def test_metric_uses_kbdi_registry_entry(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        result = fire.kbdi(precip_da, temp_da, mean_annual_da, units="metric")
        assert result.attrs["long_name"] == CF_METADATA["kbdi"]["long_name"]
        assert result.attrs["units"] == CF_METADATA["kbdi"]["units"] == "mm"

    def test_imperial_uses_kbdi_imperial_registry_entry(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        mean_annual_inches_da = mean_annual_da / 25.4
        result = fire.kbdi(precip_da / 25.4, temp_da * 9.0 / 5.0 + 32.0, mean_annual_inches_da, units="imperial")
        assert result.attrs["units"] == CF_METADATA["kbdi_imperial"]["units"] == "0.01 in"
        assert result.attrs["climate_indices_variant"] == "imperial"

    def test_version_and_history_present(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        result = fire.kbdi(precip_da, temp_da, mean_annual_da)
        assert "climate_indices_version" in result.attrs
        assert "KBDI" in result.attrs["history"]


class TestKBDIXarrayUnitInference:
    """CF `units` attributes on precipitation/maximum_temperature drive conversion, not silent guessing."""

    def test_kelvin_temperature_is_converted(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        temp_kelvin = temp_da + 273.15
        temp_kelvin.attrs["units"] = "K"
        result = fire.kbdi(precip_da, temp_kelvin, mean_annual_da)
        expected = fire.kbdi(precip_da, temp_da, mean_annual_da)
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-10)

    def test_precipitation_flux_units_are_converted(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        precip_flux = precip_da / 86400.0
        precip_flux.attrs["units"] = "kg m-2 s-1"
        result = fire.kbdi(precip_flux, temp_da, mean_annual_da)
        expected = fire.kbdi(precip_da, temp_da, mean_annual_da)
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-10)

    def test_precipitation_inches_are_converted_to_metric(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        precip_inches = precip_da / 25.4
        precip_inches.attrs["units"] = "inches"
        result = fire.kbdi(precip_inches, temp_da, mean_annual_da)
        expected = fire.kbdi(precip_da, temp_da, mean_annual_da)
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-10)

    def test_custom_precipitation_attrs_survive_unit_conversion(self) -> None:
        """Output provenance comes from the caller's input, not the converted copy."""
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        precip_inches = (precip_da / 25.4).assign_attrs(units="inches", source="gridded-observations")
        result = fire.kbdi(precip_inches, temp_da, mean_annual_da)
        assert result.attrs["source"] == "gridded-observations"
        assert result.attrs["units"] == "mm"

    def test_absent_units_attribute_is_trusted_as_is(self) -> None:
        """No `units` attribute means the raw values already match the `units=` scale."""
        precip_da, temp_da, mean_annual_da, precipitation, temperature, mean_annual = _gridded_dataarrays()
        assert "units" not in temp_da.attrs
        assert "units" not in precip_da.attrs
        result = fire.kbdi(precip_da, temp_da, mean_annual_da)
        expected = fire.kbdi(precipitation, temperature, mean_annual)
        np.testing.assert_array_equal(result.values, expected)

    def test_unrecognized_temperature_units_raises(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        temp_da = temp_da.copy()
        temp_da.attrs["units"] = "rankine"
        with pytest.raises(InvalidArgumentError):
            fire.kbdi(precip_da, temp_da, mean_annual_da)

    def test_unrecognized_precipitation_units_raises(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        precip_da = precip_da.copy()
        precip_da.attrs["units"] = "furlongs"
        with pytest.raises(InvalidArgumentError):
            fire.kbdi(precip_da, temp_da, mean_annual_da)

    def test_inch_mean_annual_climatology_is_converted_to_metric(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        mean_annual_inches = (mean_annual_da / 25.4).assign_attrs(units="inches")
        result = fire.kbdi(precip_da, temp_da, mean_annual_inches)
        expected = fire.kbdi(precip_da, temp_da, mean_annual_da)
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-10)

    def test_mm_mean_annual_climatology_is_converted_to_imperial(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        precip_inches = (precip_da / 25.4).assign_attrs(units="inches")
        temp_fahrenheit = (temp_da * 9.0 / 5.0 + 32.0).assign_attrs(units="degF")
        mean_annual_mm = mean_annual_da.assign_attrs(units="mm")
        result = fire.kbdi(precip_inches, temp_fahrenheit, mean_annual_mm, units="imperial")
        expected = fire.kbdi(precip_inches, temp_fahrenheit, mean_annual_da / 25.4, units="imperial")
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-10)

    def test_year_rate_mean_annual_units_are_accepted(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        mean_annual_per_year = mean_annual_da.assign_attrs(units="mm year-1")
        expected = fire.kbdi(precip_da, temp_da, mean_annual_da)
        result = fire.kbdi(precip_da, temp_da, mean_annual_per_year)
        np.testing.assert_array_equal(result.values, expected.values)

    def test_rate_units_for_mean_annual_climatology_raise(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        rate = xr.full_like(mean_annual_da, 1e-5).assign_attrs(units="kg m-2 s-1")
        with pytest.raises(InvalidArgumentError, match="mean_annual_precipitation"):
            fire.kbdi(precip_da, temp_da, rate)


class TestKBDIXarrayDaskChunking:
    """A chunked time dimension must be rejected explicitly, never silently rechunked."""

    def test_multi_chunked_time_dimension_raises(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays(days=40)
        precip_bad = precip_da.chunk({"time": 10})
        with pytest.raises(CoordinateValidationError) as exc_info:
            fire.kbdi(precip_bad, temp_da, mean_annual_da)
        assert "time" in str(exc_info.value)

    def test_spatially_chunked_output_stays_lazy(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        precip_dask = precip_da.chunk({"time": -1, "lat": 1, "lon": 1})
        temp_dask = temp_da.chunk({"time": -1, "lat": 1, "lon": 1})
        result = fire.kbdi(precip_dask, temp_dask, mean_annual_da)
        assert result.chunks is not None

    def test_invalid_configuration_raises_before_lazy_construction(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays(days=40)
        precip_dask = precip_da.chunk({"time": -1, "lat": 1, "lon": 1})
        temp_dask = temp_da.chunk({"time": -1, "lat": 1, "lon": 1})
        with pytest.raises(InvalidArgumentError, match="nan_policy"):
            fire.kbdi(precip_dask, temp_dask, mean_annual_da, nan_policy="interpolate")

    def test_invalid_initial_state_raises_before_lazy_construction(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays(days=40)
        precip_dask = precip_da.chunk({"time": -1, "lat": 1, "lon": 1})
        temp_dask = temp_da.chunk({"time": -1, "lat": 1, "lon": 1})
        state = fire.KBDIState(
            kbdi=np.ones((2, 3)), wet_spell_precipitation=np.zeros((2, 3)), trailing_gap_days=None, units="imperial"
        )
        with pytest.raises(InvalidArgumentError, match="units"):
            fire.kbdi(precip_dask, temp_dask, mean_annual_da, initial_state=state)

    def test_state_materialization_executes_the_recurrence_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All three state fields must share one execution of the recurrence graph."""
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        precip_dask = precip_da.chunk({"time": -1, "lat": 1, "lon": 1})
        temp_dask = temp_da.chunk({"time": -1, "lat": 1, "lon": 1})
        real_kbdi = fire.kbdi
        calls: list[int] = []

        def counting_kbdi(*args: object, **kwargs: object) -> object:
            calls.append(1)
            return real_kbdi(*args, **kwargs)

        monkeypatch.setattr(fire, "kbdi", counting_kbdi)
        result = real_kbdi(precip_dask, temp_dask, mean_annual_da, return_state=True)
        assert isinstance(result, fire.KBDIResult)
        # 2 lat x 3 lon chunks: exactly one recurrence execution per chunk,
        # shared by the values and all three state fields
        assert len(calls) == 6
        assert isinstance(result.values, xr.DataArray)
        np.testing.assert_array_equal(result.values.values, result.values.values)
        assert len(calls) == 6


class TestKBDIXarraySpinUp:
    """spin_up truncates both the value array and the reattached time coordinate."""

    def test_spin_up_truncates_values_and_time_coord(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays(days=50)
        spin_up = 7
        result = fire.kbdi(precip_da, temp_da, mean_annual_da, spin_up=spin_up)
        assert result.sizes["time"] == 50 - spin_up
        assert result.coords["time"].values[0] == precip_da.coords["time"].values[spin_up]


class TestKBDIXarrayStateRoundTrip:
    """Resuming a gridded recurrence from xarray-path state reproduces the one-shot series."""

    def test_append_resume_round_trip_matches_one_shot(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays(days=60)
        split = 25
        one_shot = fire.kbdi(precip_da, temp_da, mean_annual_da, return_state=True)
        history = fire.kbdi(
            precip_da.isel(time=slice(0, split)),
            temp_da.isel(time=slice(0, split)),
            mean_annual_da,
            return_state=True,
        )
        resumed = fire.kbdi(
            precip_da.isel(time=slice(split, None)),
            temp_da.isel(time=slice(split, None)),
            mean_annual_da,
            initial_state=history.state,
            return_state=True,
        )
        joined = np.concatenate([history.values.values, resumed.values.values], axis=0)
        np.testing.assert_array_equal(joined, one_shot.values.values)
        np.testing.assert_array_equal(resumed.state.kbdi, one_shot.state.kbdi)
        np.testing.assert_array_equal(resumed.state.wet_spell_precipitation, one_shot.state.wet_spell_precipitation)

    def test_dask_chunked_resume_matches_eager_resume_per_cell(self) -> None:
        """Each spatial chunk must resume from its OWN cell's state, not a broadcast whole-grid array.

        Distinct per-cell climatology makes a per-chunk state-slicing bug (the
        wrong cell's seed applied to another cell) change values rather than
        just shift them uniformly.
        """
        precip_da, temp_da, _mean_annual_da, *_ = _gridded_dataarrays(days=60)
        rng = np.random.default_rng(4242)
        mean_annual_distinct = xr.DataArray(
            rng.uniform(400.0, 1600.0, (2, 3)),
            dims=["lat", "lon"],
            coords={"lat": precip_da.coords["lat"], "lon": precip_da.coords["lon"]},
        )
        split = 25

        history = fire.kbdi(
            precip_da.isel(time=slice(0, split)),
            temp_da.isel(time=slice(0, split)),
            mean_annual_distinct,
            return_state=True,
        )
        resumed_eager = fire.kbdi(
            precip_da.isel(time=slice(split, None)),
            temp_da.isel(time=slice(split, None)),
            mean_annual_distinct,
            initial_state=history.state,
            return_state=True,
        )

        precip_dask = precip_da.isel(time=slice(split, None)).chunk({"time": -1, "lat": 1, "lon": 1})
        temp_dask = temp_da.isel(time=slice(split, None)).chunk({"time": -1, "lat": 1, "lon": 1})
        resumed_dask = fire.kbdi(
            precip_dask, temp_dask, mean_annual_distinct, initial_state=history.state, return_state=True
        )

        assert resumed_eager.state.kbdi.std() > 1e-6, "fixture too uniform to catch a per-chunk broadcast bug"
        np.testing.assert_array_equal(resumed_dask.values.values, resumed_eager.values)
        np.testing.assert_array_equal(resumed_dask.state.kbdi, resumed_eager.state.kbdi)
        np.testing.assert_array_equal(
            resumed_dask.state.wet_spell_precipitation, resumed_eager.state.wet_spell_precipitation
        )

    def test_bridge_nan_policy_round_trip_matches_one_shot(self) -> None:
        """nan_policy='bridge' state (trailing_gap_days) resumes correctly through the xarray path."""
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays(days=40)
        precip_da = precip_da.copy()
        precip_da.values[10:13] = np.nan  # a 3-day gap, bridgeable with max_gap_days=3
        split = 15

        one_shot = fire.kbdi(precip_da, temp_da, mean_annual_da, nan_policy="bridge", max_gap_days=3, return_state=True)
        history = fire.kbdi(
            precip_da.isel(time=slice(0, split)),
            temp_da.isel(time=slice(0, split)),
            mean_annual_da,
            nan_policy="bridge",
            max_gap_days=3,
            return_state=True,
        )
        resumed = fire.kbdi(
            precip_da.isel(time=slice(split, None)),
            temp_da.isel(time=slice(split, None)),
            mean_annual_da,
            initial_state=history.state,
            nan_policy="bridge",
            max_gap_days=3,
            return_state=True,
        )
        joined = np.concatenate([history.values.values, resumed.values.values], axis=0)
        np.testing.assert_array_equal(joined, one_shot.values.values)
        np.testing.assert_array_equal(resumed.state.trailing_gap_days, one_shot.state.trailing_gap_days)

    def test_initial_kbdi_seeds_the_xarray_recurrence(self) -> None:
        """initial_kbdi on the xarray path matches an equivalent initial_state seed."""
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays(days=20)
        seed = 50.0
        via_initial_kbdi = fire.kbdi(precip_da, temp_da, mean_annual_da, initial_kbdi=seed)

        zeros_state = fire.KBDIState(
            kbdi=np.full((2, 3), seed),
            wet_spell_precipitation=np.zeros((2, 3)),
            trailing_gap_days=None,
            units="metric",
        )
        via_initial_state = fire.kbdi(precip_da, temp_da, mean_annual_da, initial_state=zeros_state)
        np.testing.assert_array_equal(via_initial_kbdi.values, via_initial_state.values)


class TestKBDIXarrayInputValidation:
    """xarray-specific validation: type matching, time dimension, and alignment."""

    def test_mismatched_input_types_raises_type_error(self) -> None:
        precip_da, _temp_da, mean_annual_da, precipitation, temperature, _mean_annual = _gridded_dataarrays()
        with pytest.raises(TypeError, match="same type"):
            fire.kbdi(precip_da, temperature, mean_annual_da)
        with pytest.raises(TypeError, match="same type"):
            fire.kbdi(precipitation, _temp_da, mean_annual_da)

    def test_invalid_configuration_raises_on_the_xarray_path(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays(days=20)
        state = fire.KBDIState(kbdi=np.array(1.0), wet_spell_precipitation=np.array(0.0), trailing_gap_days=None)
        with pytest.raises(InvalidArgumentError, match="initial_kbdi"):
            fire.kbdi(precip_da, temp_da, mean_annual_da, initial_kbdi=5.0, initial_state=state)

    def test_not_daily_time_coordinate_raises(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays(days=48)
        hourly = pd.date_range("2000-01-01", periods=48, freq="h")
        with pytest.raises(CoordinateValidationError, match="daily"):
            fire.kbdi(precip_da.assign_coords(time=hourly), temp_da, mean_annual_da)

    def test_time_dimension_without_coordinate_is_supported(self) -> None:
        """A dimension-only time axis is aligned positionally, as before."""
        precip_da, temp_da, mean_annual_da, precipitation, temperature, mean_annual = _gridded_dataarrays(days=20)
        no_time_precip = xr.DataArray(precip_da.values, dims=precip_da.dims)
        no_time_temp = xr.DataArray(temp_da.values, dims=temp_da.dims)
        result = fire.kbdi(no_time_precip, no_time_temp, mean_annual_da)
        expected = fire.kbdi(precipitation, temperature, mean_annual)
        np.testing.assert_array_equal(result.values, expected)

    def test_coordinate_less_time_pairs_with_indexed_input(self) -> None:
        """xarray's positional alignment handles one unindexed time axis."""
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays(days=20)
        no_time_precip = xr.DataArray(precip_da.values, dims=precip_da.dims)
        result = fire.kbdi(no_time_precip, temp_da, mean_annual_da)
        expected = fire.kbdi(precip_da, temp_da, mean_annual_da)
        np.testing.assert_array_equal(result.values, expected.values)

    def test_mismatched_spatial_coordinates_raise(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        shifted_temp = temp_da.assign_coords(lat=[20.0, 30.0])
        with pytest.raises(CoordinateValidationError, match="lat"):
            fire.kbdi(precip_da, shifted_temp, mean_annual_da)

    def test_missing_time_dimension_raises(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        precip_no_time = precip_da.rename({"time": "not_time"})
        with pytest.raises(CoordinateValidationError):
            fire.kbdi(precip_no_time, temp_da, mean_annual_da)

    def test_non_overlapping_time_raises(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays()
        shifted_temp = temp_da.assign_coords(time=temp_da.coords["time"] + pd.Timedelta(days=10_000))
        with pytest.raises(CoordinateValidationError):
            fire.kbdi(precip_da, shifted_temp, mean_annual_da)

    def test_partial_overlap_warns_and_uses_intersection(self) -> None:
        precip_da, temp_da, mean_annual_da, *_ = _gridded_dataarrays(days=50)
        shifted_temp = temp_da.assign_coords(time=temp_da.coords["time"] + pd.Timedelta(days=5))
        with pytest.warns(InputAlignmentWarning):
            result = fire.kbdi(precip_da, shifted_temp, mean_annual_da)
        assert result.sizes["time"] == 45

    def test_temperature_coordinate_loss_is_rejected(self) -> None:
        """Inner alignment must not silently shrink temperature's spatial coverage."""
        precip_da, temp_da, _mean_annual_da, *_ = _gridded_dataarrays(days=40)
        narrower_precip = precip_da.isel(lon=slice(0, 2))
        with pytest.raises(CoordinateValidationError, match="lon"):
            fire.kbdi(narrower_precip, temp_da, 100.0)

    def test_time_only_precipitation_broadcasts_to_the_temperature_grid(self) -> None:
        """A time-only precipitation series must adopt the temperature's spatial dims."""
        _precip_da, temp_da, _mean_annual_da, *_ = _gridded_dataarrays(days=40)
        time_only_precip = xr.DataArray(np.full(40, 5.0), dims=["time"], coords={"time": temp_da.coords["time"]})
        result = fire.kbdi(time_only_precip, temp_da, 100.0)
        expected = fire.kbdi(np.full(temp_da.shape, 5.0), temp_da.values, 100.0)
        assert result.dims == temp_da.dims
        np.testing.assert_array_equal(result.values, expected)

    def test_time_only_temperature_broadcasts_to_the_precipitation_grid(self) -> None:
        """A time-only temperature series must adopt the precipitation's spatial dims."""
        precip_da, _temp_da, _mean_annual_da, *_ = _gridded_dataarrays(days=40)
        time_only_temp = xr.DataArray(np.full(40, 30.0), dims=["time"], coords={"time": precip_da.coords["time"]})
        result = fire.kbdi(precip_da, time_only_temp, 100.0)
        expected = fire.kbdi(precip_da.values, np.full(precip_da.shape, 30.0), 100.0)
        assert result.dims == precip_da.dims
        np.testing.assert_array_equal(result.values, expected)


class TestKBDIXarrayNumpyPassthrough:
    """The new dispatch guard must not change NumPy-path behavior or return type."""

    def test_numpy_input_returns_ndarray_not_dataarray(self) -> None:
        precipitation, temperature, mean_annual, _time = _gridded_inputs()
        result = fire.kbdi(precipitation, temperature, mean_annual)
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, xr.DataArray)
