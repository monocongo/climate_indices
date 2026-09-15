"""Tests for the Keetch-Byram Drought Index (#799)."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from climate_indices import fire
from climate_indices.exceptions import DataShapeError, InvalidArgumentError

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


def test_approximates_the_published_figure_1_series() -> None:
    """The 1968 worked example, evaluated with the continuous equation instead of the tables.

    Figure 1 exercises the table workflow, which quantizes temperature into
    3 F bins, the deficit into 50-point columns, and the daily factor into
    integers. The continuous equation therefore tracks the published series
    rather than reproducing it; the tolerance absorbs the accumulated binning
    and rounding, and a regression beyond a few points means the recurrence,
    the rain order, or the constants are wrong.
    """
    precipitation_in = [
        0,
        0,
        0.66,
        0,
        0.23,
        0,
        0.16,
        0.09,
        0,
        0,
        0.08,
        0.03,
        0,
        0.22,
        0,
        0.21,
        0,
        0,
        0,
        0.01,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.25,
        0.16,
    ]
    temperature_f = [
        79,
        75,
        70,
        76,
        79,
        84,
        65,
        66,
        83,
        70,
        67,
        65,
        76,
        69,
        65,
        75,
        78,
        85,
        88,
        79,
        69,
        75,
        84,
        89,
        93,
        92,
        96,
        91,
        78,
        83,
    ]
    published = np.array(
        [
            174,
            182,
            142,
            151,
            159,
            173,
            177,
            176,
            190,
            196,
            200,
            204,
            212,
            215,
            219,
            226,
            235,
            248,
            263,
            271,
            276,
            283,
            295,
            311,
            328,
            345,
            365,
            378,
            380,
            374,
        ],
        dtype=np.float64,
    )
    values = fire.kbdi(precipitation_in, temperature_f, 50.0, units="imperial", initial_kbdi=164.0)
    np.testing.assert_allclose(values, published, atol=4.0)


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
