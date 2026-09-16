"""Tests for the CFFWIS moisture codes: ffmc, duff_moisture_code, drought_code (#803).

The frozen reference vectors in this module were produced with the NRCan
reference implementation (``cffdrs_py`` commit 0f57fcca2a6a84b69fe8d50f29d947f0be34d5f6,
2026-08-06), a port of the Canadian Forest Service code that the full
cross-implementation validation in #805 targets. They pin the equations, the
latitude-band tables, and the month-dependent day lengths; they are
regression fixtures, not an independent scientific validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from unittest import mock

import numpy as np
import pytest

from climate_indices import fire
from climate_indices.exceptions import DataShapeError, InvalidArgumentError

# the weather series behind every reference vector, with distinct dry, rainy,
# cold, and wetting days plus a month sequence that exercises all four table
# columns of the latitude bands
_REFERENCE_TEMPERATURE = np.array([24, 26, 30, -8, 12, 20, 22, 18, 27, 15, 33, 25], dtype=np.float64)
_REFERENCE_HUMIDITY = np.array([45, 30, 60, 85, 40, 55, 70, 35, 25, 90, 20, 50], dtype=np.float64)
_REFERENCE_PRECIPITATION = np.array([0, 0, 2.0, 0.6, 8.0, 1.5, 1.6, 0, 0, 12.0, 0.5, 4.0], dtype=np.float64)
_REFERENCE_WIND = np.array([2, 3, 1, 0, 5, 2, 4, 3, 6, 1, 2, 2], dtype=np.float64)
_REFERENCE_MONTH = np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)

_REFERENCE_TOLERANCE = 1e-9

_FFMC_REFERENCE = np.array(
    [
        88.0968330314,
        91.5846327021,
        80.0934343057,
        77.8130667526,
        60.388475634,
        72.4284669329,
        74.5231519876,
        86.0601420153,
        92.7319052307,
        29.672196356,
        87.3986618239,
        73.825658243,
    ]
)
_DMC_NORTH_REFERENCE = np.array(
    [
        7.69953355,
        10.03493025,
        10.1111695092,
        10.1111695092,
        6.9248831211,
        9.4245937911,
        10.5351431664,
        13.4508804064,
        17.8017298564,
        8.6141187749,
        12.7475843749,
        9.9150720108,
    ]
)
_DMC_SOUTH_REFERENCE = np.array(
    [
        9.00686705,
        13.13872275,
        13.6991526738,
        13.6991526738,
        8.2360041524,
        9.4588841924,
        9.5579488101,
        11.0863594601,
        14.0401471601,
        6.8023257414,
        11.9691577414,
        10.378056036,
    ]
)
_DC_NORTH_REFERENCE = np.array(
    [
        19.024,
        23.408,
        28.512,
        28.512,
        20.4322573341,
        26.4362573341,
        33.8002573341,
        40.7442573341,
        48.6082573341,
        34.147568139,
        40.791568139,
        40.55167628,
    ]
)
_DC_SOUTH_REFERENCE = np.array(
    [
        23.024,
        31.408,
        39.812,
        41.012,
        32.3320089377,
        35.6360089377,
        39.3000089377,
        42.2440089377,
        46.8080089377,
        30.4300738005,
        37.3240738005,
        39.8223279159,
    ]
)

# one day from the literature seed, T=25 C, RH=40 %, no rain: (latitude, expected).
# Bands: 46 N (latitude > 30), 20 N (10, 30], equator [-10, 10], 20 S [-30, -10), 40 S (< -30).
_DMC_LATITUDE_CASES = [
    (50.0, 9.67784496),
    (30.0, 8.99566404),
    (25.0, 8.99566404),
    (10.0, 8.6694036),
    (0.0, 8.6694036),
    (-10.0, 8.34314316),
    (-20.0, 8.34314316),
    (-30.0, 7.9279026),
    (-45.0, 7.9279026),
]
# (latitude, expected) for DC. Bands: north (> 20), equator [-20, 20], south (< -20).
_DC_LATITUDE_CASES = [
    (50.0, 23.204),
    (20.0, 20.704),
    (0.0, 20.704),
    (-20.0, 19.204),
    (-50.0, 19.204),
]
# one day from the literature seed at 46 N, T=25 C, RH=40 %, no rain: (month, expected)
_DMC_MONTH_CASES = [
    (1, 7.9279026),
    (4, 9.79648512),
    (7, 9.67784496),
    (10, 8.3728032),
    (12, 7.7796024),
]
_DC_MONTH_CASES = [
    (1, 19.204),
    (4, 20.454),
    (7, 23.204),
    (10, 20.204),
    (12, 19.204),
]
# one FFMC day from the 85 seed: T=25 C, RH=40 %, wind 10.8 km/h, one rain amount
_FFMC_RAIN_REFERENCE = np.array([89.1896214799, 88.7924137871, 87.2245824186, 71.8779804228])
# (temperature, expected) for the DMC temperature floor at 46 N, July
_DMC_FLOOR_REFERENCE = np.array([6.0, 6.15500496])
# (temperature, expected) for the DC midwinter evapotranspiration floor
_DC_FLOOR_REFERENCE = np.array([100.0, 100.0])


@pytest.fixture(scope="module", autouse=True)
def disable_logging():
    """Silence the calculation lifecycle events during these tests."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@dataclass
class _Weather:
    """A batch of time-first weather inputs plus the static DMC/DC inputs."""

    temperature: np.ndarray
    humidity: np.ndarray
    wind: np.ndarray
    precipitation: np.ndarray
    latitude: np.ndarray
    month: np.ndarray


def _series(days: int, *, latitude: float = 46.0, month: int = 7) -> _Weather:
    """A run of mild, rain-free days, with latitude and month as given."""
    return _Weather(
        temperature=np.full(days, 24.0),
        humidity=np.full(days, 45.0),
        wind=np.full(days, 3.0),
        precipitation=np.zeros(days),
        latitude=np.asarray(latitude, dtype=np.float64),
        month=np.full(days, month, dtype=np.int64),
    )


def _reference_weather() -> _Weather:
    return _Weather(
        temperature=_REFERENCE_TEMPERATURE.copy(),
        humidity=_REFERENCE_HUMIDITY.copy(),
        wind=_REFERENCE_WIND.copy(),
        precipitation=_REFERENCE_PRECIPITATION.copy(),
        latitude=np.asarray(46.0),
        month=_REFERENCE_MONTH.copy(),
    )


def _slice(weather: _Weather, start: int, stop: int) -> _Weather:
    return _Weather(
        temperature=weather.temperature[start:stop],
        humidity=weather.humidity[start:stop],
        wind=weather.wind[start:stop],
        precipitation=weather.precipitation[start:stop],
        latitude=weather.latitude,
        month=weather.month[start:stop],
    )


def _with_missing(weather: _Weather, start: int, stop: int | None = None) -> _Weather:
    precipitation = weather.precipitation.copy()
    precipitation[start:stop] = np.nan
    return replace(weather, precipitation=precipitation)


def _column(weather: _Weather, index: int) -> _Weather:
    """One spatial cell of a two-column weather batch as its own time series."""
    return _Weather(
        temperature=weather.temperature[:, index],
        humidity=weather.humidity[:, index],
        wind=weather.wind[:, index],
        precipitation=weather.precipitation[:, index],
        latitude=weather.latitude,
        month=weather.month[:, index],
    )


def _state_code(state: object) -> np.ndarray:
    """The code array of a state dataclass, ignoring the gap bookkeeping field."""
    for state_field in fields(state):  # type: ignore[arg-type]
        if state_field.name != "trailing_gap_days":
            return getattr(state, state_field.name)
    raise AssertionError("state has no code field")


def _run_ffmc(weather: _Weather, **options: object) -> object:
    return fire.ffmc(
        weather.temperature,
        weather.humidity,
        weather.wind,
        weather.precipitation,
        **options,
    )


def _run_dmc(weather: _Weather, **options: object) -> object:
    return fire.duff_moisture_code(
        weather.temperature,
        weather.humidity,
        weather.precipitation,
        weather.latitude,
        weather.month,
        **options,
    )


def _run_dc(weather: _Weather, **options: object) -> object:
    return fire.drought_code(
        weather.temperature,
        weather.precipitation,
        weather.latitude,
        weather.month,
        **options,
    )


_RUNNERS = [
    pytest.param(_run_ffmc, id="ffmc"),
    pytest.param(_run_dmc, id="dmc"),
    pytest.param(_run_dc, id="dc"),
]

_SEED_NAMES = [
    pytest.param(_run_ffmc, "initial_ffmc", id="ffmc"),
    pytest.param(_run_dmc, "initial_dmc", id="dmc"),
    pytest.param(_run_dc, "initial_dc", id="dc"),
]


def test_public_api_is_namespaced() -> None:
    """The three codes are exported from fire and nowhere unqualified."""
    assert {
        "ffmc",
        "duff_moisture_code",
        "drought_code",
        "FFMCState",
        "DMCState",
        "DCState",
        "FFMCResult",
        "DMCResult",
        "DCResult",
    }.issubset(set(fire.__all__))


# ------------------------------------------------------------------------------
# reference values


def test_ffmc_matches_the_reference_series() -> None:
    values = _run_ffmc(_reference_weather())
    np.testing.assert_allclose(values, _FFMC_REFERENCE, rtol=0.0, atol=_REFERENCE_TOLERANCE)


def test_dmc_matches_the_northern_reference_series() -> None:
    values = _run_dmc(_reference_weather())
    np.testing.assert_allclose(values, _DMC_NORTH_REFERENCE, rtol=0.0, atol=_REFERENCE_TOLERANCE)


def test_dmc_matches_the_southern_reference_series() -> None:
    weather = replace(_reference_weather(), latitude=np.asarray(-35.0))
    values = _run_dmc(weather)
    np.testing.assert_allclose(values, _DMC_SOUTH_REFERENCE, rtol=0.0, atol=_REFERENCE_TOLERANCE)


def test_dc_matches_the_northern_reference_series() -> None:
    values = _run_dc(_reference_weather())
    np.testing.assert_allclose(values, _DC_NORTH_REFERENCE, rtol=0.0, atol=_REFERENCE_TOLERANCE)


def test_dc_matches_the_southern_reference_series() -> None:
    weather = replace(_reference_weather(), latitude=np.asarray(-35.0))
    values = _run_dc(weather)
    np.testing.assert_allclose(values, _DC_SOUTH_REFERENCE, rtol=0.0, atol=_REFERENCE_TOLERANCE)


@pytest.mark.parametrize(("latitude", "expected"), _DMC_LATITUDE_CASES)
def test_dmc_latitude_bands_match_the_reference_values(latitude: float, expected: float) -> None:
    """Each latitude band, including its boundaries, selects its own table row."""
    weather = _series(1, latitude=latitude)
    weather.temperature[:] = 25.0
    weather.humidity[:] = 40.0
    np.testing.assert_allclose(_run_dmc(weather), expected, rtol=0.0, atol=_REFERENCE_TOLERANCE)


@pytest.mark.parametrize(("latitude", "expected"), _DC_LATITUDE_CASES)
def test_dc_latitude_bands_match_the_reference_values(latitude: float, expected: float) -> None:
    """Each latitude band, including its boundaries, selects its own adjustment."""
    weather = _series(1, latitude=latitude)
    weather.temperature[:] = 25.0
    np.testing.assert_allclose(_run_dc(weather), expected, rtol=0.0, atol=_REFERENCE_TOLERANCE)


@pytest.mark.parametrize(("month", "expected"), _DMC_MONTH_CASES)
def test_dmc_month_table_matches_the_reference_values(month: int, expected: float) -> None:
    weather = _series(1, month=month)
    weather.temperature[:] = 25.0
    weather.humidity[:] = 40.0
    np.testing.assert_allclose(_run_dmc(weather), expected, rtol=0.0, atol=_REFERENCE_TOLERANCE)


@pytest.mark.parametrize(("month", "expected"), _DC_MONTH_CASES)
def test_dc_month_table_matches_the_reference_values(month: int, expected: float) -> None:
    weather = _series(1, month=month)
    weather.temperature[:] = 25.0
    np.testing.assert_allclose(_run_dc(weather), expected, rtol=0.0, atol=_REFERENCE_TOLERANCE)


def test_northern_and_southern_hemispheres_differ() -> None:
    """The same weather must not produce the same DMC or DC across the equator."""
    northern = _reference_weather()
    southern = replace(northern, latitude=np.asarray(-35.0))
    assert not np.allclose(_run_dmc(northern), _run_dmc(southern))
    assert not np.allclose(_run_dc(northern), _run_dc(southern))


# ------------------------------------------------------------------------------
# equations and physical bounds


def test_ffmc_rain_threshold_is_exclusive() -> None:
    """Rain at exactly 0.5 mm is not yet enough to rewet the fuel."""
    weather = _series(1)
    weather.temperature[:] = 25.0
    weather.humidity[:] = 40.0
    weather.wind[:] = 3.0
    weather.precipitation[:] = 0.5
    at_threshold = _run_ffmc(weather)
    weather.precipitation[:] = 0.0
    no_rain = _run_ffmc(weather)
    np.testing.assert_array_equal(at_threshold, no_rain)


@pytest.mark.parametrize(
    ("precipitation", "expected"), list(zip([0.5, 0.6, 1.0, 10.0], _FFMC_RAIN_REFERENCE, strict=True))
)
def test_ffmc_rain_amounts_match_the_reference_values(precipitation: float, expected: float) -> None:
    weather = _series(1)
    weather.temperature[:] = 25.0
    weather.humidity[:] = 40.0
    weather.precipitation[:] = precipitation
    np.testing.assert_allclose(_run_ffmc(weather), expected, rtol=0.0, atol=_REFERENCE_TOLERANCE)


def test_dmc_temperature_floor() -> None:
    """Every day at or below -1.1 C contributes the same (floored) drying."""
    weather = _series(1)
    weather.temperature[:] = -10.0
    weather.humidity[:] = 40.0
    floored = _run_dmc(weather)
    weather.temperature[:] = -1.1
    at_floor = _run_dmc(weather)
    np.testing.assert_array_equal(floored, at_floor)
    np.testing.assert_allclose(floored, _DMC_FLOOR_REFERENCE[0], rtol=0.0, atol=_REFERENCE_TOLERANCE)
    weather.temperature[:] = 0.0
    np.testing.assert_allclose(_run_dmc(weather), _DMC_FLOOR_REFERENCE[1], rtol=0.0, atol=_REFERENCE_TOLERANCE)


def test_dc_temperature_floor_and_evapotranspiration_floor() -> None:
    """At top latitudes in midwinter the potential evapotranspiration stays at zero."""
    weather = _series(1, month=1)
    weather.temperature[:] = -10.0
    assert _run_dc(weather, initial_dc=100.0) == pytest.approx(_DC_FLOOR_REFERENCE[0])
    weather.temperature[:] = -2.8
    assert _run_dc(weather, initial_dc=100.0) == pytest.approx(_DC_FLOOR_REFERENCE[1])


def test_heavy_rain_lowers_ffmc_sharply_but_dc_slowly() -> None:
    """One heavy rain day resets the fast fine-fuel moisture but only trims the deep drought code."""
    dry = _series(30)
    dry.temperature[:] = 32.0
    dry.humidity[:] = 25.0
    flood = replace(
        dry,
        temperature=dry.temperature.copy(),
        humidity=dry.humidity.copy(),
        precipitation=dry.precipitation.copy(),
    )
    flood.temperature[-1] = 15.0
    flood.humidity[-1] = 80.0
    flood.precipitation[-1] = 20.0

    ffmc_dry = float(_run_ffmc(dry)[-1])
    ffmc_flood = float(_run_ffmc(flood)[-1])
    dc_dry = float(_run_dc(dry)[-1])
    dc_flood = float(_run_dc(flood)[-1])

    assert ffmc_dry - ffmc_flood > 25.0
    assert dc_dry * 0.5 < dc_flood <= dc_dry


def test_codes_stay_within_physical_bounds() -> None:
    rng = np.random.default_rng(803)
    days = 120
    weather = _Weather(
        temperature=rng.uniform(-40.0, 50.0, (days, 4, 3)),
        humidity=rng.uniform(0.0, 100.0, (days, 4, 3)),
        wind=rng.uniform(0.0, 20.0, (days, 4, 3)),
        precipitation=np.where(rng.random((days, 4, 3)) < 0.4, rng.uniform(0.0, 60.0, (days, 4, 3)), 0.0),
        latitude=np.linspace(-80.0, 80.0, 4)[:, None],
        month=rng.integers(1, 13, (days, 4, 3)),
    )

    ffmc_values = _run_ffmc(weather)
    assert np.all((ffmc_values >= 0.0) & (ffmc_values <= 101.0))
    assert np.all(_run_dmc(weather) >= 0.0)
    assert np.all(_run_dc(weather) >= 0.0)


def test_dmc_and_dc_grow_with_heat_and_dry_with_rain() -> None:
    base = _series(20)
    base.temperature[:] = 25.0
    base.humidity[:] = 40.0
    hotter = replace(base, temperature=np.full(20, 35.0))
    wetter = replace(base, humidity=np.full(20, 80.0))
    assert _run_dmc(hotter)[-1] > _run_dmc(base)[-1]
    assert _run_dmc(wetter)[-1] < _run_dmc(base)[-1]
    assert _run_dc(hotter)[-1] > _run_dc(base)[-1]

    rainy = replace(base, precipitation=np.full(20, 12.0))
    assert _run_ffmc(rainy)[-1] < _run_ffmc(base)[-1]
    assert _run_dmc(rainy)[-1] < _run_dmc(base)[-1]
    assert _run_dc(rainy)[-1] < _run_dc(base)[-1]


def test_nan_latitude_yields_an_all_nan_cell_and_leaves_the_state_unstarted() -> None:
    weather = _series(5)
    weather.latitude = np.asarray(np.nan)

    for runner in (_run_dmc, _run_dc):
        result = runner(weather, return_state=True)
        assert np.isnan(result.values).all()
        assert result.state.trailing_gap_days is None


# ------------------------------------------------------------------------------
# state contract


def test_return_state_is_opt_in() -> None:
    weather = _reference_weather()
    assert isinstance(_run_ffmc(weather), np.ndarray)
    assert isinstance(_run_ffmc(weather, return_state=True), fire.FFMCResult)
    assert isinstance(_run_dmc(weather), np.ndarray)
    assert isinstance(_run_dmc(weather, return_state=True), fire.DMCResult)
    assert isinstance(_run_dc(weather), np.ndarray)
    assert isinstance(_run_dc(weather, return_state=True), fire.DCResult)


@pytest.mark.parametrize(("seed_name", "seed"), [("initial_ffmc", 90.0), ("initial_dmc", 50.0), ("initial_dc", 100.0)])
def test_seed_changes_the_recurrence(seed_name: str, seed: float) -> None:
    weather = _reference_weather()
    runner = {"initial_ffmc": _run_ffmc, "initial_dmc": _run_dmc, "initial_dc": _run_dc}[seed_name]
    seeded = runner(weather, **{seed_name: seed})
    default = runner(weather)
    assert not np.allclose(seeded, default)


@pytest.mark.parametrize("runner", _RUNNERS)
def test_append_resume_round_trip_is_bitwise_identical(runner: object) -> None:
    """Resuming from the returned state must equal one continuous run (ADR-0006)."""
    weather = _reference_weather()
    first = runner(_slice(weather, 0, 5), return_state=True)
    second = runner(_slice(weather, 5, 12), initial_state=first.state, return_state=True)
    whole = runner(weather, return_state=True)
    np.testing.assert_array_equal(np.concatenate((first.values, second.values)), whole.values)
    np.testing.assert_array_equal(_state_code(second.state), _state_code(whole.state))


@pytest.mark.parametrize("runner", _RUNNERS)
def test_append_resume_round_trip_is_bitwise_identical_per_cell(runner: object) -> None:
    weather = _series(8)
    weather.temperature = np.tile(weather.temperature[:, None], (1, 3))
    weather.humidity = np.tile(weather.humidity[:, None], (1, 3))
    weather.wind = np.tile(weather.wind[:, None], (1, 3))
    weather.precipitation = np.tile(weather.precipitation[:, None], (1, 3))
    weather.month = np.tile(weather.month[:, None], (1, 3))
    weather.month[:, 1] = 1

    first = runner(_slice(weather, 0, 3), return_state=True)
    second = runner(_slice(weather, 3, 8), initial_state=first.state, return_state=True)
    whole = runner(weather, return_state=True)
    np.testing.assert_array_equal(np.concatenate((first.values, second.values)), whole.values)
    np.testing.assert_array_equal(_state_code(second.state), _state_code(whole.state))
    assert first.state.trailing_gap_days is not None
    assert whole.state.trailing_gap_days is not None


@pytest.mark.parametrize("runner", _RUNNERS)
def test_returned_state_does_not_alias_the_values(runner: object) -> None:
    """ADR-0006: the final state is copied before it is returned."""
    result = runner(_reference_weather(), return_state=True)
    assert not np.shares_memory(_state_code(result.state), result.values)


@pytest.mark.parametrize("runner", _RUNNERS)
def test_partially_valid_grid_matches_running_the_valid_cell_alone(runner: object) -> None:
    """The active-cell fast path must not change results for cells that stay valid."""
    weather = _series(6)
    weather.temperature = np.tile(weather.temperature[:, None], (1, 2))
    weather.humidity = np.tile(weather.humidity[:, None], (1, 2))
    weather.wind = np.tile(weather.wind[:, None], (1, 2))
    weather.precipitation = np.tile(weather.precipitation[:, None], (1, 2))
    weather.month = np.tile(weather.month[:, None], (1, 2))
    weather.precipitation[2, 0] = np.nan

    mixed = runner(weather)
    alone = runner(_column(weather, 1))
    np.testing.assert_array_equal(mixed[:, 1], alone)
    assert np.isnan(mixed[2:, 0]).all()


@pytest.mark.parametrize("runner", _RUNNERS)
def test_spin_up_omits_leading_days_without_changing_the_state(runner: object) -> None:
    weather = _reference_weather()
    spun = runner(weather, spin_up=3, return_state=True)
    full = runner(weather, return_state=True)
    np.testing.assert_array_equal(spun.values, full.values[3:])
    np.testing.assert_array_equal(_state_code(spun.state), _state_code(full.state))


@pytest.mark.parametrize("runner", _RUNNERS)
def test_spin_up_longer_than_the_input_yields_an_empty_result(runner: object) -> None:
    weather = _reference_weather()
    values = runner(weather, spin_up=20)
    assert values.shape == (0,)


@pytest.mark.parametrize(("runner", "seed_name"), _SEED_NAMES)
def test_seed_and_state_cannot_be_combined(runner: object, seed_name: str) -> None:
    weather = _reference_weather()
    state = runner(_slice(weather, 0, 5), return_state=True).state
    with pytest.raises(InvalidArgumentError, match=seed_name):
        runner(weather, **{seed_name: 50.0, "initial_state": state})


@pytest.mark.parametrize("runner", _RUNNERS)
def test_initial_state_must_have_the_matching_type(runner: object) -> None:
    weather = _series(3)
    mismatched = {
        _run_ffmc: fire.DMCState(dmc=np.asarray(0.0), trailing_gap_days=None),
        _run_dmc: fire.FFMCState(ffmc=np.asarray(0.0), trailing_gap_days=None),
        _run_dc: fire.FFMCState(ffmc=np.asarray(0.0), trailing_gap_days=None),
    }[runner]
    with pytest.raises(InvalidArgumentError, match="initial_state"):
        runner(weather, initial_state=mismatched)


@pytest.mark.parametrize("runner", _RUNNERS)
def test_nan_state_requires_started_gap_bookkeeping(runner: object) -> None:
    weather = _series(1)
    state_type = {_run_ffmc: fire.FFMCState, _run_dmc: fire.DMCState, _run_dc: fire.DCState}[runner]
    value_name = {_run_ffmc: "ffmc", _run_dmc: "dmc", _run_dc: "dc"}[runner]
    state = state_type(**{value_name: np.asarray(np.nan), "trailing_gap_days": np.asarray(-1)})
    with pytest.raises(InvalidArgumentError, match="NaN"):
        runner(weather, initial_state=state)


# ------------------------------------------------------------------------------
# missing days: propagate


@pytest.mark.parametrize("runner", _RUNNERS)
def test_propagate_poisons_from_the_first_valid_day_after_an_interior_gap(runner: object) -> None:
    weather = _with_missing(_series(6), 2, 3)
    result = runner(weather, return_state=True)
    assert np.isfinite(result.values[:2]).all()
    assert np.isnan(result.values[2:]).all()
    assert int(result.state.trailing_gap_days) == 0
    assert np.isnan(_state_code(result.state))


@pytest.mark.parametrize("runner", _RUNNERS)
def test_propagate_ignores_leading_missing_days_before_the_start(runner: object) -> None:
    weather = _with_missing(_series(4), 0, 2)
    result = runner(weather, return_state=True)
    assert np.isnan(result.values[:2]).all()
    assert np.isfinite(result.values[2:]).all()
    assert int(result.state.trailing_gap_days) == 0


@pytest.mark.parametrize("runner", _RUNNERS)
def test_propagate_all_nan_input_returns_an_unstarted_state(runner: object) -> None:
    weather = _with_missing(_series(5), 0)
    result = runner(weather, return_state=True)
    assert np.isnan(result.values).all()
    assert result.state.trailing_gap_days is None
    expected_seed = {_run_ffmc: 85.0, _run_dmc: 6.0, _run_dc: 15.0}[runner]
    assert float(_state_code(result.state)) == pytest.approx(expected_seed)


@pytest.mark.parametrize("runner", _RUNNERS)
def test_propagate_all_nan_input_poisons_a_started_state(runner: object) -> None:
    started = runner(_series(3), return_state=True).state
    weather = _with_missing(_series(3), 0)
    result = runner(weather, initial_state=started, return_state=True)
    assert np.isnan(result.values).all()
    assert np.isnan(_state_code(result.state))


@pytest.mark.parametrize("runner", _RUNNERS)
def test_propagate_poisons_the_final_state_after_a_trailing_gap(runner: object) -> None:
    weather = _with_missing(_series(4), 3, 4)
    result = runner(weather, return_state=True)
    assert np.isfinite(result.values[:3]).all()
    assert np.isnan(result.values[3])
    assert np.isnan(_state_code(result.state))


@pytest.mark.parametrize("runner", _RUNNERS)
def test_propagate_poisons_a_resumed_state_on_leading_missing_days(runner: object) -> None:
    state = runner(_series(3), return_state=True).state
    resumed = runner(_with_missing(_series(3), 0, 1), initial_state=state, return_state=True)
    assert np.isnan(resumed.values).all()


# ------------------------------------------------------------------------------
# missing days: bridge


@pytest.mark.parametrize("runner", _RUNNERS)
@pytest.mark.parametrize("gap_days", [1, 2, 3, 4])
@pytest.mark.parametrize("max_gap_days", [1, 2, 3])
def test_bridge_matrix(runner: object, gap_days: int, max_gap_days: int) -> None:
    """A run within the limit resumes; the first day past the limit poisons."""
    weather = _with_missing(_series(4 + gap_days + 3), 4, 4 + gap_days)
    values = runner(weather, nan_policy="bridge", max_gap_days=max_gap_days)
    assert np.isfinite(values[:4]).all()
    assert np.isnan(values[4 : 4 + gap_days]).all()
    if gap_days <= max_gap_days:
        assert np.isfinite(values[4 + gap_days :]).all()
    else:
        assert np.isnan(values[4 + max_gap_days :]).all()


@pytest.mark.parametrize("runner", _RUNNERS)
def test_bridged_run_equals_running_only_the_valid_days(runner: object) -> None:
    weather = _with_missing(_series(12), 2, 5)
    weather = _with_missing(weather, 9, 10)
    valid = np.isfinite(weather.precipitation)
    bridged = runner(weather, nan_policy="bridge", max_gap_days=3)
    only_valid = runner(
        replace(
            weather,
            precipitation=weather.precipitation[valid],
            temperature=weather.temperature[valid],
            humidity=weather.humidity[valid],
            wind=weather.wind[valid],
            month=weather.month[valid],
        )
    )
    np.testing.assert_array_equal(bridged[valid], only_valid)


@pytest.mark.parametrize("runner", _RUNNERS)
def test_bridge_skips_leading_missing_days_before_the_start(runner: object) -> None:
    weather = _with_missing(_series(5), 0, 2)
    values = runner(weather, nan_policy="bridge", max_gap_days=1)
    assert np.isnan(values[:2]).all()
    assert np.isfinite(values[2:]).all()


@pytest.mark.parametrize("runner", _RUNNERS)
@pytest.mark.parametrize("max_gap_days", [1, 2, 3])
def test_bridge_trailing_gap_at_the_limit_keeps_the_last_valid_state(runner: object, max_gap_days: int) -> None:
    weather = _with_missing(_series(3 + max_gap_days), 3, 3 + max_gap_days)
    result = runner(weather, nan_policy="bridge", max_gap_days=max_gap_days, return_state=True)
    assert np.isfinite(result.values[:3]).all()
    assert np.isnan(result.values[3:]).all()
    assert int(result.state.trailing_gap_days) == max_gap_days
    assert np.isfinite(_state_code(result.state))


@pytest.mark.parametrize("runner", _RUNNERS)
@pytest.mark.parametrize("max_gap_days", [1, 2, 3])
def test_bridge_trailing_gap_past_the_limit_poisons_the_state(runner: object, max_gap_days: int) -> None:
    weather = _with_missing(_series(4 + max_gap_days), 3, 4 + max_gap_days)
    result = runner(weather, nan_policy="bridge", max_gap_days=max_gap_days, return_state=True)
    assert np.isnan(result.values[3:]).all()
    assert int(result.state.trailing_gap_days) == max_gap_days + 1
    assert np.isnan(_state_code(result.state))


@pytest.mark.parametrize("runner", _RUNNERS)
def test_bridge_all_nan_continuation_poisons_only_after_the_limit(runner: object) -> None:
    state = runner(_series(3), return_state=True).state
    weather = _with_missing(_series(3), 0)

    within = runner(weather, initial_state=state, nan_policy="bridge", max_gap_days=3, return_state=True)
    assert np.isnan(within.values).all()
    assert int(within.state.trailing_gap_days) == 3
    assert np.isfinite(_state_code(within.state))

    past = runner(weather, initial_state=state, nan_policy="bridge", max_gap_days=2, return_state=True)
    assert np.isnan(_state_code(past.state))

    unstarted = runner(weather, nan_policy="bridge", max_gap_days=1, return_state=True)
    assert unstarted.state.trailing_gap_days is None


@pytest.mark.parametrize("runner", _RUNNERS)
def test_bridge_split_gap_append_poisons_on_the_same_day_as_one_shot(runner: object) -> None:
    """A gap that exceeds the limit only after joining two calls poisons on schedule."""
    days = 12
    weather = _with_missing(_series(days), 4, 9)
    options = {"nan_policy": "bridge", "max_gap_days": 3}
    first = runner(_slice(weather, 0, 6), return_state=True, **options)
    second = runner(_slice(weather, 6, days), initial_state=first.state, return_state=True, **options)
    whole = runner(weather, return_state=True, **options)
    joined = np.concatenate((first.values, second.values))
    assert np.isfinite(_state_code(first.state))
    assert np.isnan(joined[7])
    np.testing.assert_array_equal(joined, whole.values)
    np.testing.assert_array_equal(_state_code(second.state), _state_code(whole.state))


# ------------------------------------------------------------------------------
# input validation


def test_scalar_inputs_have_no_time_dimension() -> None:
    with pytest.raises(DataShapeError) as exc_info:
        fire.ffmc(25.0, 40.0, 3.0, 0.0)
    assert exc_info.value.expected_shape == "(time, ...)"
    with pytest.raises(DataShapeError):
        fire.duff_moisture_code(25.0, 40.0, 0.0, 46.0, 7)
    with pytest.raises(DataShapeError):
        fire.drought_code(25.0, 0.0, 46.0, 7)


def test_incompatible_shapes_raise() -> None:
    with pytest.raises(InvalidArgumentError, match="broadcast"):
        fire.ffmc(np.zeros(3), np.zeros(4), np.zeros(3), np.zeros(3))
    with pytest.raises(InvalidArgumentError, match="broadcast"):
        fire.duff_moisture_code(np.zeros(3), np.zeros(3), np.zeros(3), np.asarray([46.0, 47.0, 48.0]), 7)


def test_negative_precipitation_raises() -> None:
    weather = _series(2)
    weather.precipitation = np.asarray([0.0, -1.0])
    for runner in (_run_ffmc, _run_dmc, _run_dc):
        with pytest.raises(InvalidArgumentError, match="precipitation"):
            runner(weather)


def test_absurd_temperature_raises_instead_of_returning_a_non_finite_state() -> None:
    """A finite input whose recurrence overflows must fail, not return an unusable state."""
    weather = _series(10)
    weather.temperature[:] = np.finfo(np.float64).max
    for runner in (_run_dmc, _run_dc):
        with pytest.raises(InvalidArgumentError, match="non-finite"):
            runner(weather, return_state=True)


def test_absurd_precipitation_raises_instead_of_returning_a_non_finite_state() -> None:
    weather = _series(2)
    weather.precipitation = np.asarray([0.0, 1e308])
    with pytest.raises(InvalidArgumentError, match="non-finite"):
        _run_ffmc(weather, return_state=True)


def test_absurd_wind_raises_instead_of_becoming_a_missing_day() -> None:
    """The km/h conversion must not turn a finite wind into a missing observation."""
    weather = _series(2)
    weather.wind = np.asarray([3.0, 1e308])
    with pytest.raises(InvalidArgumentError, match="wind_speed_meters_per_second"):
        _run_ffmc(weather)


@pytest.mark.parametrize("bound_infinity", [np.inf, -np.inf])
def test_infinite_weather_values_raise(bound_infinity: float) -> None:
    weather = _series(2)
    weather.temperature = np.asarray([20.0, bound_infinity])
    for runner in (_run_ffmc, _run_dmc, _run_dc):
        with pytest.raises(InvalidArgumentError, match="finite"):
            runner(weather)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        pytest.param("nan_policy", "interpolate", id="nan_policy"),
        pytest.param("max_gap_days", -1, id="negative_gap"),
        pytest.param("spin_up", -1, id="negative_spin_up"),
    ],
)
def test_invalid_configuration_raises(argument: str, value: object) -> None:
    weather = _series(3)
    for runner in (_run_ffmc, _run_dmc, _run_dc):
        with pytest.raises(InvalidArgumentError, match=argument):
            runner(weather, **{argument: value})


@pytest.mark.parametrize(("nan_policy", "max_gap_days"), [("propagate", 1), ("bridge", 0)])
def test_gap_policy_and_limit_must_be_consistent(nan_policy: str, max_gap_days: int) -> None:
    weather = _series(3)
    with pytest.raises(InvalidArgumentError, match="max_gap_days"):
        _run_ffmc(weather, nan_policy=nan_policy, max_gap_days=max_gap_days)


@pytest.mark.parametrize("month", [0, 13, 2.5, np.nan])
def test_invalid_month_raises(month: float) -> None:
    weather = _series(3)
    weather.month = np.full(3, month)
    for runner in (_run_dmc, _run_dc):
        with pytest.raises(InvalidArgumentError, match="month"):
            runner(weather)


def test_month_may_be_a_time_axis_shared_across_cells() -> None:
    days = 5
    weather = _series(days)
    weather.temperature = np.tile(weather.temperature[:, None], (1, 3))
    weather.humidity = np.tile(weather.humidity[:, None], (1, 3))
    weather.precipitation = np.tile(weather.precipitation[:, None], (1, 3))
    weather.month = np.array([3, 3, 4, 4, 5])
    values = _run_dmc(weather)
    assert values.shape == (days, 3)
    np.testing.assert_array_equal(values[:, 0], values[:, 2])


def test_shared_time_series_broadcasts_over_spatial_fields() -> None:
    """A (time,) input shares across a gridded input instead of aligning with its final axis."""
    days = 6
    base = _series(days)
    gridded = replace(
        base,
        temperature=np.tile(base.temperature[:, None], (1, 4)),
        latitude=np.full(4, 46.0),
    )
    for runner in (_run_ffmc, _run_dmc, _run_dc):
        values = runner(gridded)
        assert values.shape == (days, 4)
        expected = runner(base)
        for column in range(4):
            np.testing.assert_allclose(values[:, column], expected)


def test_shared_calendar_months_stay_a_broadcast_view() -> None:
    """Shared scalar and (time,) months must not materialize a full grid-sized int64 copy."""
    shared = np.asarray([1, 2, 3])
    result = fire._month_array(shared, (3, 4, 5))
    assert result.shape == (3, 4, 5)
    assert 0 in result.strides
    np.testing.assert_array_equal(result[:, 0, 0], shared)
    assert fire._month_array(7, (3, 4, 5)).strides == (0, 0, 0)


def test_output_allocation_failure_emits_lifecycle_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure to allocate the recurrence output still reports started and failed."""
    real_full = np.full

    def fail_output_allocation(*args: object, **kwargs: object) -> np.ndarray:
        fill_value = args[1] if len(args) > 1 else kwargs.get("fill_value")
        if isinstance(fill_value, float) and np.isnan(fill_value):
            raise MemoryError("simulated output allocation failure")
        return real_full(*args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(fire.np, "full", fail_output_allocation)
    with mock.patch.object(fire, "_logger") as mocked_logger:
        with pytest.raises(MemoryError):
            _run_ffmc(_series(3))
    bound = mocked_logger.bind.return_value
    bound.info.assert_called_once_with("calculation_started")
    assert bound.error.call_args.args[0] == "calculation_failed"


@pytest.mark.parametrize("latitude", [91.0, -91.0, np.inf])
def test_invalid_latitude_raises(latitude: float) -> None:
    weather = _series(3)
    weather.latitude = np.asarray(latitude)
    for runner in (_run_dmc, _run_dc):
        with pytest.raises(InvalidArgumentError, match="latitude"):
            runner(weather)


@pytest.mark.parametrize(
    ("runner", "seed_name", "seed"),
    [
        pytest.param(_run_ffmc, "initial_ffmc", -1.0, id="ffmc_low"),
        pytest.param(_run_ffmc, "initial_ffmc", 102.0, id="ffmc_high"),
        pytest.param(_run_dmc, "initial_dmc", -1.0, id="dmc_low"),
        pytest.param(_run_dc, "initial_dc", -1.0, id="dc_low"),
    ],
)
def test_out_of_range_seed_raises(runner: object, seed_name: str, seed: float) -> None:
    with pytest.raises(InvalidArgumentError, match=seed_name):
        runner(_series(3), **{seed_name: seed})


def test_missing_days_count_all_three_weather_inputs() -> None:
    """NaN in any time-varying input of the code is a missing day."""
    weather = _series(4)
    weather.humidity = np.asarray([45.0, np.nan, 45.0, 45.0])
    assert np.isnan(_run_ffmc(weather)[1:]).all()
    assert np.isnan(_run_dmc(weather)[1:]).all()
    weather = _series(4)
    weather.temperature = np.asarray([24.0, np.nan, 24.0, 24.0])
    assert np.isnan(_run_dc(weather)[1:]).all()


def test_out_of_range_humidity_counts_as_a_missing_day() -> None:
    weather = _series(4)
    weather.humidity = np.asarray([45.0, 101.0, 45.0, 45.0])
    assert np.isnan(_run_ffmc(weather)[1:]).all()
    assert np.isnan(_run_dmc(weather)[1:]).all()
