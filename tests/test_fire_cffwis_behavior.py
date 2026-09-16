"""Tests for the CFFWIS behavior indices and orchestrator (#804).

The frozen reference vectors in this module were produced with the NRCan
reference implementation (``cffdrs_py`` commit 0f57fcca2a6a84b69fe8d50f29d947f0be34d5f6),
the same port the moisture-code vectors in ``test_fire_cffwis_moisture.py``
pin. Its ``cffdrs/fwi.py`` holds ``initial_spread_index``,
``buildup_index``, ``fire_weather_index``, and the DSR power transform; each
vector is the daily recurrence driven one day at a time through that whole
chain (moisture codes included) with the weather series below, the literature
seeds 85/6/15, and the latitude each case names. They pin the equations and the
latitudinal wiring; they are regression fixtures, not an independent
scientific validation (#805 owns that).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, replace
from timeit import repeat
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import fire
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.exceptions import (
    ClimateIndicesWarning,
    CoordinateValidationError,
    DataShapeError,
    InputAlignmentWarning,
    InputTypeError,
    InvalidArgumentError,
)
from climate_indices.fire import _cffwis

# the weather series behind every reference vector, with distinct dry, rainy,
# cold, and wetting days plus a month sequence that exercises all four table
# columns of the latitude bands
_REFERENCE_TEMPERATURE = np.array([24, 26, 30, -8, 12, 20, 22, 18, 27, 15, 33, 25], dtype=np.float64)
_REFERENCE_HUMIDITY = np.array([45, 30, 60, 85, 40, 55, 70, 35, 25, 90, 20, 50], dtype=np.float64)
_REFERENCE_PRECIPITATION = np.array([0, 0, 2.0, 0.6, 8.0, 1.5, 1.6, 0, 0, 12.0, 0.5, 4.0], dtype=np.float64)
_REFERENCE_WIND = np.array([2, 3, 1, 0, 5, 2, 4, 3, 6, 1, 2, 2], dtype=np.float64)
_REFERENCE_MONTH = np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)

_REFERENCE_TOLERANCE = 1e-9

_ISI_NORTH_REFERENCE = np.array(
    [
        4.6856673043,
        9.2556557376,
        1.3753725668,
        0.9267896831,
        1.0375512980,
        0.9775608106,
        1.5419459682,
        4.2041966284,
        18.7559681952,
        0.0037526106,
        4.2399812253,
        1.0369097517,
    ]
)
_BUI_NORTH_REFERENCE = np.array(
    [
        7.6940351001,
        10.0022034748,
        10.7190955043,
        10.7190955043,
        7.4973108468,
        9.9664895934,
        11.8424246784,
        14.7380768054,
        18.5863402667,
        10.5652324973,
        14.3129691800,
        12.3072161466,
    ]
)
_FWI_NORTH_REFERENCE = np.array(
    [
        4.4066362072,
        9.4984484029,
        0.8617428191,
        0.5806821900,
        0.5389331305,
        0.5886445009,
        1.1395265591,
        5.6738465546,
        21.6845864315,
        0.0023325932,
        5.6249952304,
        0.7019824980,
    ]
)
_DSR_NORTH_REFERENCE = np.array(
    [
        0.3755248190,
        1.4622247353,
        0.0209019780,
        0.0103929792,
        0.0091072162,
        0.0106465494,
        0.0342745044,
        0.5873976869,
        6.3031323799,
        0.0000005965,
        0.5784757058,
        0.0145400677,
    ]
)
_BUI_SOUTH_REFERENCE = np.array(
    [
        9.1071054070,
        13.1172339971,
        14.7283699041,
        14.9303888575,
        10.0633559828,
        11.3717468425,
        11.8879082590,
        13.3885902445,
        16.0469871124,
        8.7273700306,
        13.2864793502,
        12.5678656371,
    ]
)
_FWI_SOUTH_REFERENCE = np.array(
    [
        4.8250932736,
        10.7807961440,
        1.1911295153,
        0.7022252390,
        0.6280459555,
        0.6329157156,
        1.1492761419,
        5.3672462728,
        20.3255318035,
        0.0021059690,
        5.3887555144,
        0.7104397293,
    ]
)

# one-step cffdrs reference cases: (DMC, DC, expected BUI)
_BUI_REFERENCE_CASES = [
    (0.0, 0.0, 0.0),
    (0.0, 100.0, 0.0),
    (10.0, 0.0, 9.0550690833),
    (50.0, 500.0, 80.0),
    (100.0, 800.0, 152.3809523810),
]
# (ISI, BUI, expected FWI), spanning both BUI branches and both bb branches
_FWI_REFERENCE_CASES = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.2),
    (5.0, 50.0, 13.2633770216),
    (5.0, 80.0, 17.2256590420),
    (10.0, 81.0, 28.3854819185),
    (30.0, 150.0, 72.9003498156),
]
# (FFMC, wind km/h, expected ISI); the public function takes m/s
_ISI_REFERENCE_CASES = [
    (0.0, 0.0, 0.0000000019),
    (85.0, 0.0, 2.1050822068),
    (85.0, 20.0, 5.7670144454),
    (101.0, 50.0, 237.4563530801),
]
# (FWI, expected DSR)
_DSR_REFERENCE_CASES = [
    (0.0, 0.0),
    (1.0, 0.0272),
    (10.0, 1.6016547426),
    (100.0, 94.3124233231),
]


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


def _run_orchestrator(weather: _Weather, **options: object) -> fire.CFFWISResult:
    return fire.cffwis(
        weather.temperature,
        weather.humidity,
        weather.wind,
        weather.precipitation,
        weather.latitude,
        weather.month,
        **options,
    )


def _run_chained(weather: _Weather, **options: object) -> tuple[np.ndarray, ...]:
    """The three codes and four derived indices through separate public calls."""
    ffmc = fire.ffmc(weather.temperature, weather.humidity, weather.wind, weather.precipitation, **options)
    dmc = fire.duff_moisture_code(
        weather.temperature,
        weather.humidity,
        weather.precipitation,
        weather.latitude,
        weather.month,
        **options,
    )
    dc = fire.drought_code(weather.temperature, weather.precipitation, weather.latitude, weather.month, **options)
    isi = fire.initial_spread_index(ffmc, weather.wind)
    bui = fire.buildup_index(dmc, dc)
    fwi = fire.cffwis_fwi(isi, bui)
    dsr = fire.daily_severity_rating(fwi)
    return ffmc, dmc, dc, isi, bui, fwi, dsr


def _assert_matches_chained(weather: _Weather, **options: object) -> fire.CFFWISResult:
    """The orchestrator's seven outputs must equal the separately chained calls."""
    result = _run_orchestrator(weather, **options)
    for name, orchestrated in zip(
        ("ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr"),
        _run_chained(weather, **options),
        strict=True,
    ):
        np.testing.assert_array_equal(getattr(result, name), orchestrated)
    return result


_CODE_NAMES = ("ffmc", "dmc", "dc")


def _code_values(result: fire.CFFWISResult, code: str) -> np.ndarray:
    values = getattr(result, code)
    assert isinstance(values, np.ndarray)
    return values


def _code_state(result: fire.CFFWISResult, code: str) -> object:
    assert result.state is not None
    return getattr(result.state, code)


def _state_value(result: fire.CFFWISResult, code: str) -> np.ndarray:
    values = getattr(_code_state(result, code), code)
    assert isinstance(values, np.ndarray)
    return values


def _valid_days_weather(weather: _Weather, keep: np.ndarray) -> _Weather:
    """The valid days alone, with the same static latitude and month values."""
    return _Weather(
        temperature=weather.temperature[keep],
        humidity=weather.humidity[keep],
        wind=weather.wind[keep],
        precipitation=weather.precipitation[keep],
        latitude=weather.latitude,
        month=weather.month[keep],
    )


def test_public_api_is_namespaced() -> None:
    """The behavior indices and orchestrator are exported from fire only."""
    assert {
        "CFFWISResult",
        "CFFWISState",
        "initial_spread_index",
        "buildup_index",
        "cffwis_fwi",
        "daily_severity_rating",
        "cffwis",
    }.issubset(set(fire.__all__))


# ------------------------------------------------------------------------------
# reference values


def test_initial_spread_index_matches_the_reference_series() -> None:
    ffmc, _, _, _, _, _, _ = _run_chained(_reference_weather())
    values = fire.initial_spread_index(ffmc, _REFERENCE_WIND)
    np.testing.assert_allclose(values, _ISI_NORTH_REFERENCE, rtol=0.0, atol=_REFERENCE_TOLERANCE)


def test_buildup_index_matches_the_northern_reference_series() -> None:
    _, dmc, dc, _, _, _, _ = _run_chained(_reference_weather())
    np.testing.assert_allclose(fire.buildup_index(dmc, dc), _BUI_NORTH_REFERENCE, rtol=0.0, atol=_REFERENCE_TOLERANCE)


def test_buildup_index_matches_the_southern_reference_series() -> None:
    weather = replace(_reference_weather(), latitude=np.asarray(-35.0))
    _, dmc, dc, _, _, _, _ = _run_chained(weather)
    np.testing.assert_allclose(fire.buildup_index(dmc, dc), _BUI_SOUTH_REFERENCE, rtol=0.0, atol=_REFERENCE_TOLERANCE)


def test_cffwis_fwi_matches_the_northern_reference_series() -> None:
    ffmc, dmc, dc, _, _, _, _ = _run_chained(_reference_weather())
    isi = fire.initial_spread_index(ffmc, _REFERENCE_WIND)
    np.testing.assert_allclose(
        fire.cffwis_fwi(isi, fire.buildup_index(dmc, dc)),
        _FWI_NORTH_REFERENCE,
        rtol=0.0,
        atol=_REFERENCE_TOLERANCE,
    )


def test_cffwis_fwi_matches_the_southern_reference_series() -> None:
    weather = replace(_reference_weather(), latitude=np.asarray(-35.0))
    ffmc, dmc, dc, _, _, _, _ = _run_chained(weather)
    isi = fire.initial_spread_index(ffmc, weather.wind)
    np.testing.assert_allclose(
        fire.cffwis_fwi(isi, fire.buildup_index(dmc, dc)),
        _FWI_SOUTH_REFERENCE,
        rtol=0.0,
        atol=_REFERENCE_TOLERANCE,
    )


def test_daily_severity_rating_matches_the_reference_series() -> None:
    fwi = _run_chained(_reference_weather())[5]
    np.testing.assert_allclose(
        fire.daily_severity_rating(fwi), _DSR_NORTH_REFERENCE, rtol=0.0, atol=_REFERENCE_TOLERANCE
    )


@pytest.mark.parametrize(("dmc", "dc", "expected"), _BUI_REFERENCE_CASES)
def test_buildup_index_matches_the_reference_cases(dmc: float, dc: float, expected: float) -> None:
    assert float(fire.buildup_index(dmc, dc)) == pytest.approx(expected, abs=_REFERENCE_TOLERANCE)


@pytest.mark.parametrize(("isi", "bui", "expected"), _FWI_REFERENCE_CASES)
def test_cffwis_fwi_matches_the_reference_cases(isi: float, bui: float, expected: float) -> None:
    assert float(fire.cffwis_fwi(isi, bui)) == pytest.approx(expected, abs=_REFERENCE_TOLERANCE)


@pytest.mark.parametrize(("ffmc", "wind_kmh", "expected"), _ISI_REFERENCE_CASES)
def test_initial_spread_index_matches_the_reference_cases(ffmc: float, wind_kmh: float, expected: float) -> None:
    wind_mps = wind_kmh / 3.6
    assert float(fire.initial_spread_index(ffmc, wind_mps)) == pytest.approx(expected, abs=_REFERENCE_TOLERANCE)


@pytest.mark.parametrize(("fwi", "expected"), _DSR_REFERENCE_CASES)
def test_daily_severity_rating_matches_the_reference_cases(fwi: float, expected: float) -> None:
    assert float(fire.daily_severity_rating(fwi)) == pytest.approx(expected, abs=_REFERENCE_TOLERANCE)


# ------------------------------------------------------------------------------
# elementwise contract


def test_initial_spread_index_masks_invalid_inputs() -> None:
    values = fire.initial_spread_index([-1.0, 102.0, 85.0, np.nan, 85.0], [-1.0, 2.0, 2.0, 2.0, np.nan])
    assert np.isnan(values[[0, 1, 3, 4]]).all()
    assert np.isfinite(values[2])


def test_buildup_index_masks_negative_and_non_finite_codes() -> None:
    values = fire.buildup_index([10.0, -1.0, np.nan, np.inf], [100.0, 100.0, 100.0, 100.0])
    assert np.isnan(values[[1, 2, 3]]).all()
    assert np.isfinite(values[0])


def test_cffwis_fwi_masks_invalid_inputs() -> None:
    values = fire.cffwis_fwi([5.0, -1.0, np.nan], [50.0, 50.0, -1.0])
    assert np.isnan(values[[1, 2]]).all()
    assert np.isfinite(values[0])


def test_daily_severity_rating_masks_negative_fwi() -> None:
    values = fire.daily_severity_rating([10.0, -1.0, np.nan])
    assert np.isnan(values[[1, 2]]).all()
    assert np.isfinite(values[0])


@pytest.mark.parametrize(
    "call",
    [
        lambda: fire.initial_spread_index(np.ones((3, 2)), np.ones(3)),
        lambda: fire.buildup_index(np.ones((3, 2)), np.ones(3)),
        lambda: fire.cffwis_fwi(np.ones((3, 2)), np.ones(3)),
    ],
)
def test_derived_indices_reject_incompatible_shapes(call: object) -> None:
    with pytest.raises(InvalidArgumentError):
        call()  # type: ignore[operator]


def test_derived_indices_reject_non_numeric_inputs() -> None:
    with pytest.raises(InputTypeError):
        fire.buildup_index(np.array(["a", "b"]), np.ones(2))


# ------------------------------------------------------------------------------
# orchestrator: reference and chained equality


def test_orchestrator_matches_the_reference_series() -> None:
    result = _run_orchestrator(_reference_weather())
    for values, expected in (
        (result.isi, _ISI_NORTH_REFERENCE),
        (result.bui, _BUI_NORTH_REFERENCE),
        (result.fwi, _FWI_NORTH_REFERENCE),
        (result.dsr, _DSR_NORTH_REFERENCE),
    ):
        np.testing.assert_allclose(values, expected, rtol=0.0, atol=_REFERENCE_TOLERANCE)


def test_orchestrator_matches_individually_chained_calls_bitwise() -> None:
    _assert_matches_chained(_reference_weather())


def test_orchestrator_matches_the_southern_reference_series() -> None:
    weather = replace(_reference_weather(), latitude=np.asarray(-35.0))
    result = _assert_matches_chained(weather)
    np.testing.assert_allclose(result.bui, _BUI_SOUTH_REFERENCE, rtol=0.0, atol=_REFERENCE_TOLERANCE)
    np.testing.assert_allclose(result.fwi, _FWI_SOUTH_REFERENCE, rtol=0.0, atol=_REFERENCE_TOLERANCE)


def test_dsr_is_the_analytic_power_transform_of_fwi() -> None:
    result = _run_orchestrator(_reference_weather())
    np.testing.assert_array_equal(result.dsr, 0.0272 * result.fwi**1.77)


def test_orchestrator_rejects_a_missing_time_dimension() -> None:
    with pytest.raises(DataShapeError):
        fire.cffwis(24.0, 45.0, 3.0, 0.0, 46.0, 7)


def test_orchestrator_rejects_negative_precipitation() -> None:
    weather = _series(3)
    weather.precipitation = np.array([0.0, -1.0, 0.0])
    with pytest.raises(InvalidArgumentError):
        _run_orchestrator(weather)


# ------------------------------------------------------------------------------
# orchestrator: outputs selection


def test_outputs_default_to_all_seven() -> None:
    result = _run_orchestrator(_reference_weather())
    for name in ("ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr"):
        assert getattr(result, name) is not None


def test_outputs_subset_sets_the_rest_to_none() -> None:
    result = _run_orchestrator(_reference_weather(), outputs=["fwi", "dsr"])
    assert result.fwi is not None
    assert result.dsr is not None
    for name in ("ffmc", "dmc", "dc", "isi", "bui"):
        assert getattr(result, name) is None


def test_outputs_accepts_a_single_name_string() -> None:
    result = _run_orchestrator(_reference_weather(), outputs="isi")
    assert result.isi is not None
    assert result.fwi is None


def test_outputs_subset_matches_the_full_result() -> None:
    full = _run_orchestrator(_reference_weather())
    subset = _run_orchestrator(_reference_weather(), outputs=["dmc", "bui"])
    np.testing.assert_array_equal(subset.dmc, full.dmc)
    np.testing.assert_array_equal(subset.bui, full.bui)


def test_unselected_derived_indices_are_not_computed() -> None:
    """Skipping outputs also skips their computation, not just their return."""
    with mock.patch.object(_cffwis, "_initial_spread_index") as isi:
        result = _run_orchestrator(_reference_weather(), outputs=["ffmc"])
    isi.assert_not_called()
    assert result.ffmc is not None
    assert result.isi is None


def test_unknown_output_name_raises() -> None:
    weather = _reference_weather()
    with pytest.raises(InvalidArgumentError):
        _run_orchestrator(weather, outputs=["fwi", "nope"])


def test_mixed_type_unknown_output_names_raise_invalid_argument_error() -> None:
    """Heterogeneous bad names must not leak the sorted() comparison TypeError."""
    weather = _reference_weather()
    with pytest.raises(InvalidArgumentError):
        _run_orchestrator(weather, outputs=["ffmc", 1, None])  # type: ignore[list-item]


def test_subset_outputs_record_only_the_code_histories_they_consume() -> None:
    """A subset request stops allocating the moisture-code histories it never reads."""
    with mock.patch.object(_cffwis, "_run_cffwis_system", wraps=_cffwis._run_cffwis_system) as runner:
        _run_orchestrator(_reference_weather(), outputs=["ffmc"])
    assert runner.call_args is not None
    assert runner.call_args.kwargs["record"] == (True, False, False)
    with mock.patch.object(_cffwis, "_run_cffwis_system", wraps=_cffwis._run_cffwis_system) as runner:
        _run_orchestrator(_reference_weather(), outputs=["bui"])
    assert runner.call_args is not None
    assert runner.call_args.kwargs["record"] == (False, True, True)


def test_empty_outputs_raise() -> None:
    weather = _reference_weather()
    with pytest.raises(InvalidArgumentError):
        _run_orchestrator(weather, outputs=[])


# ------------------------------------------------------------------------------
# orchestrator: state and spin-up


def test_return_state_is_opt_in_and_nested_per_code() -> None:
    result = _run_orchestrator(_reference_weather())
    assert result.state is None
    returned = _run_orchestrator(_reference_weather(), return_state=True)
    assert isinstance(returned.state, fire.CFFWISState)
    assert isinstance(returned.state.ffmc, fire.FFMCState)
    assert isinstance(returned.state.dmc, fire.DMCState)
    assert isinstance(returned.state.dc, fire.DCState)


def test_append_resume_round_trip_is_bitwise_identical() -> None:
    """Resuming from the combined state must equal one continuous run (ADR-0006)."""
    weather = _reference_weather()
    first = _run_orchestrator(_slice(weather, 0, 5), return_state=True)
    second = _run_orchestrator(_slice(weather, 5, 12), initial_state=first.state, return_state=True)
    whole = _run_orchestrator(weather, return_state=True)
    for name in ("ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr"):
        np.testing.assert_array_equal(
            np.concatenate((getattr(first, name), getattr(second, name))),
            getattr(whole, name),
        )
    assert second.state is not None
    assert whole.state is not None
    for nested in ("ffmc", "dmc", "dc"):
        np.testing.assert_array_equal(
            getattr(getattr(second.state, nested), nested),
            getattr(getattr(whole.state, nested), nested),
        )


def test_append_resume_round_trip_is_bitwise_identical_per_cell() -> None:
    weather = _series(8)
    weather.temperature = np.tile(weather.temperature[:, None], (1, 3))
    weather.humidity = np.tile(weather.humidity[:, None], (1, 3))
    weather.wind = np.tile(weather.wind[:, None], (1, 3))
    weather.precipitation = np.tile(weather.precipitation[:, None], (1, 3))
    weather.month = np.tile(weather.month[:, None], (1, 3))
    weather.month[:, 1] = 1

    first = _run_orchestrator(_slice(weather, 0, 3), return_state=True)
    second = _run_orchestrator(_slice(weather, 3, 8), initial_state=first.state, return_state=True)
    whole = _run_orchestrator(weather, return_state=True)
    for name in ("ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr"):
        np.testing.assert_array_equal(
            np.concatenate((getattr(first, name), getattr(second, name))),
            getattr(whole, name),
        )


@pytest.mark.parametrize(
    ("seed_name", "seed"),
    [("initial_ffmc", 90.0), ("initial_dmc", 50.0), ("initial_dc", 100.0)],
)
def test_seed_changes_the_recurrence(seed_name: str, seed: float) -> None:
    weather = _reference_weather()
    component = {"initial_ffmc": "ffmc", "initial_dmc": "dmc", "initial_dc": "dc"}[seed_name]
    seeded = getattr(_run_orchestrator(weather, **{seed_name: seed}), component)
    default = getattr(_run_orchestrator(weather), component)
    assert not np.allclose(seeded, default)


def test_seed_and_initial_state_are_mutually_exclusive() -> None:
    weather = _reference_weather()
    state = _run_orchestrator(weather, return_state=True).state
    with pytest.raises(InvalidArgumentError):
        _run_orchestrator(weather, initial_dmc=50.0, initial_state=state)


def test_spin_up_omits_leading_days_without_changing_the_state() -> None:
    weather = _reference_weather()
    spun = _run_orchestrator(weather, spin_up=3, return_state=True)
    full = _run_orchestrator(weather, return_state=True)
    for name in ("ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr"):
        np.testing.assert_array_equal(getattr(spun, name), getattr(full, name)[3:])
    assert spun.state is not None
    assert full.state is not None
    for nested in ("ffmc", "dmc", "dc"):
        np.testing.assert_array_equal(
            getattr(getattr(spun.state, nested), nested), getattr(getattr(full.state, nested), nested)
        )


def test_spin_up_longer_than_the_input_yields_empty_outputs() -> None:
    result = _run_orchestrator(_reference_weather(), spin_up=20)
    assert result.fwi is not None
    assert result.fwi.shape == (0,)


def test_returned_state_does_not_alias_the_outputs() -> None:
    result = _run_orchestrator(_reference_weather(), return_state=True)
    assert result.state is not None
    assert result.ffmc is not None
    for nested in ("ffmc", "dmc", "dc"):
        state_code = getattr(getattr(result.state, nested), nested)
        assert not np.shares_memory(state_code, result.ffmc)


# ------------------------------------------------------------------------------
# gaps and per-code validity


@pytest.mark.parametrize(
    ("nan_policy", "max_gap_days"),
    [("propagate", 0), ("bridge", 1), ("bridge", 3)],
)
@pytest.mark.parametrize("missing_field", ["temperature", "humidity", "wind", "precipitation"])
def test_orchestrator_codes_match_the_single_functions_under_gaps(
    nan_policy: str, max_gap_days: int, missing_field: str
) -> None:
    """The orchestrator's per-code gap bookkeeping must mirror the single codes."""
    weather = _series(9)
    missing = getattr(weather, missing_field).copy()
    missing[3] = np.nan
    weather = replace(weather, **{missing_field: missing})
    _assert_matches_chained(weather, nan_policy=nan_policy, max_gap_days=max_gap_days)


def test_negative_wind_poisons_ffmc_but_not_dmc_or_dc() -> None:
    weather = _series(4)
    weather.wind = np.array([3.0, 3.0, -1.0, 3.0])
    result = _assert_matches_chained(weather)
    assert np.isnan(result.ffmc[2:]).all()
    assert np.isfinite(result.dmc).all()
    assert np.isfinite(result.dc).all()


def test_nan_humidity_poisons_ffmc_and_dmc_but_not_dc() -> None:
    weather = _series(4)
    weather.humidity = np.array([45.0, np.nan, 45.0, 45.0])
    result = _assert_matches_chained(weather)
    assert np.isnan(result.ffmc[1:]).all()
    assert np.isnan(result.dmc[1:]).all()
    assert np.isfinite(result.dc).all()


def test_nan_latitude_poisons_dmc_and_dc_but_not_ffmc() -> None:
    weather = _series(4)
    weather.latitude = np.asarray(np.nan)
    result = _assert_matches_chained(weather)
    assert np.isfinite(result.ffmc).all()
    assert np.isnan(result.dmc).all()
    assert np.isnan(result.dc).all()


@pytest.mark.parametrize(
    ("elementwise_field", "invalid_value"),
    [("humidity", 101.0), ("humidity", -1.0), ("wind", -1.0)],
)
def test_orchestrator_matches_the_single_functions_for_elementwise_invalid_values(
    elementwise_field: str, invalid_value: float
) -> None:
    """Humidity outside [0, 100] and negative wind are elementwise-invalid, not NaN."""
    weather = _series(6)
    values = getattr(weather, elementwise_field).copy()
    values[2] = invalid_value
    weather = replace(weather, **{elementwise_field: values})
    _assert_matches_chained(weather)


@pytest.mark.parametrize("code", _CODE_NAMES)
def test_orchestrator_all_nan_input_leaves_an_unstarted_state(code: str) -> None:
    """ADR-0007: an all-missing series never starts the recurrence."""
    weather = _with_missing(_series(4), 0)
    result = _run_orchestrator(weather, return_state=True, outputs=(code,))
    assert np.isnan(_code_values(result, code)).all()
    assert _code_state(result, code).trailing_gap_days is None


@pytest.mark.parametrize("code", _CODE_NAMES)
def test_orchestrator_all_nan_continuation_poisons_a_started_state(code: str) -> None:
    """ADR-0007: a started recurrence poisons on a missing continuation."""
    started = _run_orchestrator(_series(3), return_state=True)
    weather = _with_missing(_series(4), 0)
    result = _run_orchestrator(weather, initial_state=started.state, return_state=True, outputs=(code,))
    assert np.isnan(_code_values(result, code)).all()
    assert np.isnan(_state_value(result, code)).all()


@pytest.mark.parametrize("code", _CODE_NAMES)
def test_orchestrator_leading_missing_days_are_unbounded(code: str) -> None:
    """ADR-0007: before the recurrence starts, missing days never poison."""
    weather = _with_missing(_series(8), 0, 3)
    gapped = _run_orchestrator(weather, outputs=(code,))
    short = _run_orchestrator(_slice(weather, 3, 8), outputs=(code,))
    values = _code_values(gapped, code)
    assert np.isnan(values[:3]).all()
    np.testing.assert_array_equal(values[3:], _code_values(short, code))


@pytest.mark.parametrize("code", _CODE_NAMES)
def test_orchestrator_propagate_trailing_block_poisons_the_state(code: str) -> None:
    """ADR-0007: a trailing missing run poisons under the default policy."""
    weather = _with_missing(_series(5), 4)
    result = _run_orchestrator(weather, return_state=True, outputs=(code,))
    assert np.isnan(_code_values(result, code)[4:]).all()
    assert np.isnan(_state_value(result, code)).all()
    assert _code_state(result, code).trailing_gap_days is not None


@pytest.mark.parametrize("code", _CODE_NAMES)
@pytest.mark.parametrize("max_gap_days", [1, 2, 3])
def test_orchestrator_bridge_interior_gap_at_the_limit_is_skipped(code: str, max_gap_days: int) -> None:
    """ADR-0007: a bridged run equals running only the valid days alone."""
    length = 3 + max_gap_days + 3
    weather = _with_missing(_series(length), 3, 3 + max_gap_days)
    bridged = _run_orchestrator(weather, nan_policy="bridge", max_gap_days=max_gap_days, outputs=(code,))
    keep = np.ones(length, dtype=bool)
    keep[3 : 3 + max_gap_days] = False
    valid_only = _run_orchestrator(_valid_days_weather(weather, keep), outputs=(code,))
    values = _code_values(bridged, code)
    assert np.isnan(values[~keep]).all()
    np.testing.assert_array_equal(values[keep], _code_values(valid_only, code))


@pytest.mark.parametrize("code", _CODE_NAMES)
@pytest.mark.parametrize("max_gap_days", [1, 2, 3])
def test_orchestrator_bridge_interior_gap_past_the_limit_poisons(code: str, max_gap_days: int) -> None:
    """ADR-0007: the first run longer than max_gap_days poisons."""
    gap_length = max_gap_days + 1
    length = 3 + gap_length + 2
    weather = _with_missing(_series(length), 3, 3 + gap_length)
    result = _run_orchestrator(weather, nan_policy="bridge", max_gap_days=max_gap_days, outputs=(code,))
    values = _code_values(result, code)
    assert np.isnan(values[3:]).all()


@pytest.mark.parametrize("code", _CODE_NAMES)
@pytest.mark.parametrize("max_gap_days", [1, 2, 3])
def test_orchestrator_bridge_trailing_gap_at_the_limit_keeps_the_last_valid_state(code: str, max_gap_days: int) -> None:
    """ADR-0007: a bridged trailing gap keeps the last valid state and its count."""
    valid_days = 4
    weather = _with_missing(_series(valid_days + max_gap_days), valid_days)
    result = _run_orchestrator(
        weather,
        nan_policy="bridge",
        max_gap_days=max_gap_days,
        return_state=True,
        outputs=(code,),
    )
    values = _code_values(result, code)
    assert np.isnan(values[valid_days:]).all()
    np.testing.assert_array_equal(_state_value(result, code), values[valid_days - 1])
    assert int(_code_state(result, code).trailing_gap_days) == max_gap_days


@pytest.mark.parametrize("code", _CODE_NAMES)
@pytest.mark.parametrize("max_gap_days", [1, 2, 3])
def test_orchestrator_bridge_trailing_gap_past_the_limit_poisons(code: str, max_gap_days: int) -> None:
    """ADR-0007: a trailing run longer than max_gap_days poisons the state."""
    valid_days = 4
    weather = _with_missing(_series(valid_days + max_gap_days + 1), valid_days)
    result = _run_orchestrator(
        weather,
        nan_policy="bridge",
        max_gap_days=max_gap_days,
        return_state=True,
        outputs=(code,),
    )
    assert np.isnan(_code_values(result, code)[valid_days:]).all()
    assert np.isnan(_state_value(result, code)).all()


def test_orchestrator_bridge_split_gap_append_matches_one_shot() -> None:
    """ADR-0007: within-limit gap pieces bridged across an append boundary."""
    weather = _with_missing(_series(9), 3, 6)
    one_shot = _run_orchestrator(weather, nan_policy="bridge", max_gap_days=3, return_state=True)
    first = _run_orchestrator(_slice(weather, 0, 5), nan_policy="bridge", max_gap_days=3, return_state=True)
    second = _run_orchestrator(
        _slice(weather, 5, 9), initial_state=first.state, nan_policy="bridge", max_gap_days=3, return_state=True
    )
    for code in _CODE_NAMES:
        concatenated = np.concatenate((_code_values(first, code), _code_values(second, code)))
        np.testing.assert_array_equal(concatenated, _code_values(one_shot, code))
        # the valid tail after the bridged gap stays finite on both paths
        assert np.isfinite(_code_values(second, code)[1:]).all()
        np.testing.assert_array_equal(_state_value(second, code), _state_value(one_shot, code))


def test_orchestrator_bridge_split_gap_poisons_when_the_joined_run_exceeds_the_limit() -> None:
    """ADR-0007: a run that only exceeds the limit once joined still poisons."""
    weather = _with_missing(_series(9), 3, 6)
    one_shot = _run_orchestrator(weather, nan_policy="bridge", max_gap_days=2, return_state=True)
    first = _run_orchestrator(_slice(weather, 0, 5), nan_policy="bridge", max_gap_days=2, return_state=True)
    second = _run_orchestrator(
        _slice(weather, 5, 9), initial_state=first.state, nan_policy="bridge", max_gap_days=2, return_state=True
    )
    for code in _CODE_NAMES:
        np.testing.assert_array_equal(_code_values(second, code), _code_values(one_shot, code)[5:])
        assert np.isnan(_code_values(second, code)).all()
        assert np.isnan(_state_value(second, code)).all()


# ------------------------------------------------------------------------------
# spatial shapes


def test_gridded_orchestrator_matches_per_point_runs() -> None:
    """A multi-latitude grid must equal per-cell runs with each cell's latitude."""
    weather = _series(6)
    latitudes = np.array([[20.0, 46.0], [46.0, -35.0]])
    weather.temperature = np.tile(weather.temperature[:, None, None], (1, 2, 2))
    weather.humidity = np.tile(weather.humidity[:, None, None], (1, 2, 2))
    weather.wind = np.tile(weather.wind[:, None, None], (1, 2, 2))
    weather.precipitation = np.tile(weather.precipitation[:, None, None], (1, 2, 2))
    weather.month = np.tile(weather.month[:, None, None], (1, 2, 2))
    gridded = _run_orchestrator(replace(weather, latitude=latitudes), return_state=True)
    assert gridded.fwi is not None
    assert gridded.fwi.shape == (6, 2, 2)
    for row in range(2):
        for column in range(2):
            per_point = fire.cffwis(
                weather.temperature[:, row, column],
                weather.humidity[:, row, column],
                weather.wind[:, row, column],
                weather.precipitation[:, row, column],
                latitudes[row, column],
                weather.month[:, row, column],
            )
            assert per_point.fwi is not None
            np.testing.assert_array_equal(gridded.fwi[:, row, column], per_point.fwi)


def test_orchestrator_matches_the_single_functions_per_cell_under_gaps() -> None:
    weather = _series(7)
    weather.temperature = np.tile(weather.temperature[:, None], (1, 2))
    weather.humidity = np.tile(weather.humidity[:, None], (1, 2))
    weather.wind = np.tile(weather.wind[:, None], (1, 2))
    weather.precipitation = np.tile(weather.precipitation[:, None], (1, 2))
    weather.month = np.tile(weather.month[:, None], (1, 2))
    weather.precipitation[2, 0] = np.nan
    _assert_matches_chained(weather)


# ------------------------------------------------------------------------------
# performance


@pytest.mark.benchmark(group="cffwis-orchestrator")
def test_single_pass_orchestrator_is_faster_than_chained_calls() -> None:
    """#804 acceptance: the one-pass orchestrator measures faster than chained calls.

    This records the comparison rather than gating on wall-clock time: the
    benchmark workflow runs every ``benchmark``-marked test on every PR, and
    #812 owns the deliberate-slowdown CI guard. The correctness assertion is the
    tripwire here; the printed ratio is the acceptance evidence.
    """
    rng = np.random.default_rng(804)
    shape = (3650, 25, 25)
    temperature = 20.0 + 10.0 * rng.standard_normal(shape)
    humidity = np.clip(50.0 + 20.0 * rng.standard_normal(shape), 0.0, 100.0)
    wind = np.abs(3.0 + 2.0 * rng.standard_normal(shape))
    precipitation = np.where(rng.random(shape) < 0.2, rng.exponential(5.0, shape), 0.0)
    latitude = np.full(shape[1:], 46.0)
    month = np.arange(shape[0]) % 12 + 1

    def orchestrator() -> fire.CFFWISResult:
        return fire.cffwis(temperature, humidity, wind, precipitation, latitude, month)

    def separate() -> np.ndarray:
        ffmc = fire.ffmc(temperature, humidity, wind, precipitation)
        dmc = fire.duff_moisture_code(temperature, humidity, precipitation, latitude, month)
        dc = fire.drought_code(temperature, precipitation, latitude, month)
        isi = fire.initial_spread_index(ffmc, wind)
        bui = fire.buildup_index(dmc, dc)
        return fire.daily_severity_rating(fire.cffwis_fwi(isi, bui))

    orchestrated = orchestrator()
    np.testing.assert_array_equal(orchestrated.dsr, separate())
    # alternate the timed runs so a warming runner does not bias either path
    orchestrator_time = separate_time = float("inf")
    for _ in range(3):
        orchestrator_time = min(orchestrator_time, min(repeat(orchestrator, number=1, repeat=1)))
        separate_time = min(separate_time, min(repeat(separate, number=1, repeat=1)))
    print(
        f"cffwis one-pass {orchestrator_time:.3f}s vs chained {separate_time:.3f}s "
        f"({separate_time / orchestrator_time:.2f}x)"
    )


# ------------------------------------------------------------------------------
# xarray adapter (#807)

_GRID_LATITUDES = np.array([20.0, 46.0, -35.0])
_GRID_LONGITUDES = np.array([0.0, 10.0])
_GRID_VARIABLES = ("ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr")


@dataclass
class _GriddedInputs:
    """A (time, lat, lon) weather block plus the NumPy arrays of the equivalent core call."""

    temperature: xr.DataArray
    humidity: xr.DataArray
    wind: xr.DataArray
    precipitation: xr.DataArray
    latitude_grid: np.ndarray
    month: np.ndarray


def _gridded_inputs(days: int = 12, *, hour: int = 12) -> _GriddedInputs:
    """Deterministic gridded weather with a ``lat`` coordinate for inference.

    The series starts in late January so the inferred months span a calendar-
    month boundary, exercising the DMC/DC day-length table lookup rows.
    """
    rng = np.random.default_rng(807)
    shape = (days, _GRID_LATITUDES.size, _GRID_LONGITUDES.size)
    time = pd.date_range(f"2000-01-28 {hour:02d}:00", periods=days, freq="D")
    coords = {"time": time, "lat": _GRID_LATITUDES, "lon": _GRID_LONGITUDES}
    dims = ("time", "lat", "lon")
    temperature = 20.0 + 8.0 * rng.standard_normal(shape)
    humidity = np.clip(50.0 + 20.0 * rng.standard_normal(shape), 5.0, 100.0)
    wind = np.abs(3.0 + 2.0 * rng.standard_normal(shape))
    precipitation = np.where(rng.random(shape) < 0.25, rng.exponential(3.0, shape), 0.0)
    return _GriddedInputs(
        temperature=xr.DataArray(temperature, dims=dims, coords=coords),
        humidity=xr.DataArray(humidity, dims=dims, coords=coords),
        wind=xr.DataArray(wind, dims=dims, coords=coords),
        precipitation=xr.DataArray(precipitation, dims=dims, coords=coords),
        latitude_grid=np.broadcast_to(_GRID_LATITUDES[:, None], shape[1:]).copy(),
        month=time.month.values.astype(np.int64),
    )


def _numpy_cffwis(inputs: _GriddedInputs, **options: object) -> fire.CFFWISResult:
    return fire.cffwis(
        inputs.temperature.values,
        inputs.humidity.values,
        inputs.wind.values,
        inputs.precipitation.values,
        inputs.latitude_grid,
        inputs.month,
        **options,
    )


def _xarray_cffwis(inputs: _GriddedInputs, **options: object) -> xr.Dataset | fire.CFFWISResult:
    return fire.cffwis(inputs.temperature, inputs.humidity, inputs.wind, inputs.precipitation, **options)


def _assert_matches_numpy(inputs: _GriddedInputs, result: xr.Dataset, **options: object) -> None:
    expected = _numpy_cffwis(inputs, **options)
    for name in _GRID_VARIABLES:
        np.testing.assert_array_equal(result[name].values, getattr(expected, name))


class TestCFFWISXarrayEquivalence:
    """The xarray and NumPy paths must agree exactly on values, latitude inference, and chunking."""

    def test_dataset_variables_match_numpy(self) -> None:
        inputs = _gridded_inputs()
        result = _xarray_cffwis(inputs)
        assert isinstance(result, xr.Dataset)
        assert list(result.data_vars) == list(_GRID_VARIABLES)
        _assert_matches_numpy(inputs, result)

    def test_dims_and_coords_preserved(self) -> None:
        inputs = _gridded_inputs()
        result = _xarray_cffwis(inputs)
        assert isinstance(result, xr.Dataset)
        for variable in result.data_vars.values():
            assert variable.dims == inputs.temperature.dims
            for coord in ("time", "lat", "lon"):
                xr.testing.assert_equal(variable.coords[coord], inputs.temperature.coords[coord])

    def test_gridded_multi_latitude_matches_per_point_numpy(self) -> None:
        """Acceptance: a multi-latitude grid equals per-cell runs under each cell's latitude."""
        inputs = _gridded_inputs(days=8)
        result = _xarray_cffwis(inputs)
        assert isinstance(result, xr.Dataset)
        for row, latitude in enumerate(_GRID_LATITUDES):
            for column in range(_GRID_LONGITUDES.size):
                per_point = fire.cffwis(
                    inputs.temperature.values[:, row, column],
                    inputs.humidity.values[:, row, column],
                    inputs.wind.values[:, row, column],
                    inputs.precipitation.values[:, row, column],
                    latitude,
                    inputs.month,
                )
                for name in _GRID_VARIABLES:
                    np.testing.assert_array_equal(result[name].values[:, row, column], getattr(per_point, name))

    def test_dask_chunked_matches_eager(self) -> None:
        """Acceptance: Dask and eager results are identical, with time in a single chunk."""
        inputs = _gridded_inputs()
        eager = _xarray_cffwis(inputs)
        assert isinstance(eager, xr.Dataset)
        chunked_inputs = replace(
            inputs,
            temperature=inputs.temperature.chunk({"time": -1, "lat": 1, "lon": 1}),
            humidity=inputs.humidity.chunk({"time": -1, "lat": 1, "lon": 1}),
            wind=inputs.wind.chunk({"time": -1, "lat": 1}),
            precipitation=inputs.precipitation.chunk({"time": -1, "lat": 1, "lon": 1}),
        )
        chunked = _xarray_cffwis(chunked_inputs)
        assert isinstance(chunked, xr.Dataset)
        assert chunked["fwi"].chunks is not None
        for name in _GRID_VARIABLES:
            np.testing.assert_array_equal(chunked[name].values, eager[name].values)

    def test_time_dimension_without_coordinate_matches_numpy(self) -> None:
        """A coordinate-less time axis is aligned positionally; month must then be explicit."""
        inputs = _gridded_inputs()
        coords = {"lat": _GRID_LATITUDES, "lon": _GRID_LONGITUDES}
        bare = replace(
            inputs,
            temperature=xr.DataArray(inputs.temperature.values, dims=inputs.temperature.dims, coords=coords),
            humidity=xr.DataArray(inputs.humidity.values, dims=inputs.humidity.dims, coords=coords),
            wind=xr.DataArray(inputs.wind.values, dims=inputs.wind.dims, coords=coords),
            precipitation=xr.DataArray(inputs.precipitation.values, dims=inputs.precipitation.dims, coords=coords),
        )
        result = _xarray_cffwis(bare, month=inputs.month)
        assert isinstance(result, xr.Dataset)
        _assert_matches_numpy(inputs, result)

    def test_time_only_input_matches_numpy(self) -> None:
        inputs = _gridded_inputs(days=10)
        time = inputs.temperature.coords["time"]
        column = replace(
            inputs,
            temperature=xr.DataArray(inputs.temperature.values[:, 1, 0], dims=("time",), coords={"time": time}),
            humidity=xr.DataArray(inputs.humidity.values[:, 1, 0], dims=("time",), coords={"time": time}),
            wind=xr.DataArray(inputs.wind.values[:, 1, 0], dims=("time",), coords={"time": time}),
            precipitation=xr.DataArray(inputs.precipitation.values[:, 1, 0], dims=("time",), coords={"time": time}),
        )
        expected = fire.cffwis(
            column.temperature.values,
            column.humidity.values,
            column.wind.values,
            column.precipitation.values,
            float(_GRID_LATITUDES[1]),
            column.month,
        )
        result = _xarray_cffwis(column, latitude_degrees_north=46.0)
        assert isinstance(result, xr.Dataset)
        for name in _GRID_VARIABLES:
            np.testing.assert_array_equal(result[name].values, getattr(expected, name))


class TestCFFWISXarrayCoordinates:
    """Latitude and month come from the call or from the inputs' coordinates, never by guessing."""

    def test_latitude_inferred_from_coordinate(self) -> None:
        inputs = _gridded_inputs()
        result = _xarray_cffwis(inputs)
        assert isinstance(result, xr.Dataset)
        _assert_matches_numpy(inputs, result)

    def test_scalar_latitude_matches_uniform_numpy(self) -> None:
        inputs = _gridded_inputs()
        scalar = _xarray_cffwis(inputs, latitude_degrees_north=46.0)
        assert isinstance(scalar, xr.Dataset)
        expected = fire.cffwis(
            inputs.temperature.values,
            inputs.humidity.values,
            inputs.wind.values,
            inputs.precipitation.values,
            np.full(inputs.latitude_grid.shape, 46.0),
            inputs.month,
        )
        for name in _GRID_VARIABLES:
            np.testing.assert_array_equal(scalar[name].values, getattr(expected, name))

    def test_leading_dimension_array_like_raises(self) -> None:
        """An un-nameable 1-D latitude must be a DataArray, not guessed onto the leading axis."""
        inputs = _gridded_inputs()
        with pytest.raises(InvalidArgumentError, match="broadcast to the weather inputs' spatial shape"):
            _xarray_cffwis(inputs, latitude_degrees_north=_GRID_LATITUDES)

    def test_absent_latitude_raises(self) -> None:
        inputs = _gridded_inputs()
        without_lat = replace(
            inputs,
            temperature=inputs.temperature.drop_vars("lat"),
            humidity=inputs.humidity.drop_vars("lat"),
            wind=inputs.wind.drop_vars("lat"),
            precipitation=inputs.precipitation.drop_vars("lat"),
        )
        with pytest.raises(InvalidArgumentError, match="latitude_degrees_north is required"):
            _xarray_cffwis(without_lat, month=inputs.month)

    def test_conflicting_latitude_coordinates_raise(self) -> None:
        inputs = _gridded_inputs()

        def with_auxiliary_latitude(data: xr.DataArray, latitude: float) -> xr.DataArray:
            """A curvilinear-style 2-D latitude auxiliary coordinate (no `lat` dimension index)."""
            return data.drop_vars("lat").assign_coords(
                latitude=(("lat", "lon"), np.full((_GRID_LATITUDES.size, _GRID_LONGITUDES.size), latitude))
            )

        conflicting = replace(
            inputs,
            temperature=with_auxiliary_latitude(inputs.temperature, 20.0),
            humidity=with_auxiliary_latitude(inputs.humidity, 30.0),
            wind=with_auxiliary_latitude(inputs.wind, 20.0),
            precipitation=with_auxiliary_latitude(inputs.precipitation, 20.0),
        )
        with pytest.raises(InvalidArgumentError, match="conflicting"):
            fire.cffwis(conflicting.temperature, conflicting.humidity, conflicting.wind, conflicting.precipitation)

    def test_time_varying_latitude_raises(self) -> None:
        inputs = _gridded_inputs()
        latitude_in_time = xr.DataArray(
            np.full((inputs.temperature.sizes["time"], _GRID_LATITUDES.size), 46.0),
            dims=("time", "lat"),
            coords={"time": inputs.temperature.coords["time"], "lat": _GRID_LATITUDES},
        )
        with pytest.raises(InvalidArgumentError, match="must not vary in time"):
            fire.cffwis(
                inputs.temperature,
                inputs.humidity,
                inputs.wind,
                inputs.precipitation,
                latitude_in_time,
                inputs.month,
            )

    def test_month_inferred_from_time_coordinate(self) -> None:
        inputs = _gridded_inputs()
        result = _xarray_cffwis(inputs)
        assert isinstance(result, xr.Dataset)
        _assert_matches_numpy(inputs, result)

    def test_month_as_scalar_and_series(self) -> None:
        inputs = _gridded_inputs()
        scalar = _xarray_cffwis(inputs, month=1)
        assert isinstance(scalar, xr.Dataset)
        assert scalar.sizes["time"] == inputs.temperature.sizes["time"]
        np.testing.assert_array_equal(
            scalar["fwi"].values,
            fire.cffwis(
                inputs.temperature.values,
                inputs.humidity.values,
                inputs.wind.values,
                inputs.precipitation.values,
                inputs.latitude_grid,
                1,
            ).fwi,
        )
        series = _xarray_cffwis(inputs, month=inputs.month)
        assert isinstance(series, xr.Dataset)
        _assert_matches_numpy(inputs, series)

    def test_month_wrong_length_raises(self) -> None:
        inputs = _gridded_inputs()
        with pytest.raises(InvalidArgumentError, match="entries"):
            _xarray_cffwis(inputs, month=np.ones(inputs.temperature.sizes["time"] + 1, dtype=np.int64))

    def test_month_coordinate_mismatch_raises(self) -> None:
        inputs = _gridded_inputs()
        shifted = xr.DataArray(
            inputs.month,
            dims=("time",),
            coords={"time": inputs.temperature.coords["time"] + pd.Timedelta(days=1)},
        )
        with pytest.raises(InvalidArgumentError, match="month has no value"):
            _xarray_cffwis(inputs, month=shifted)

    def test_month_coordinate_is_reindexed_not_paired_positionally(self) -> None:
        """A labelled month series follows its dates, so a reversed series still pairs correctly."""
        inputs = _gridded_inputs()
        reversed_month = xr.DataArray(
            inputs.month[::-1].copy(), dims=("time",), coords={"time": inputs.temperature.coords["time"][::-1]}
        )
        result = _xarray_cffwis(inputs, month=reversed_month)
        expected = _xarray_cffwis(inputs)
        assert isinstance(result, xr.Dataset) and isinstance(expected, xr.Dataset)
        for name in _GRID_VARIABLES:
            np.testing.assert_array_equal(result[name].values, expected[name].values)

    def test_month_missing_without_time_coordinate_raises(self) -> None:
        inputs = _gridded_inputs()
        coords = {"lat": _GRID_LATITUDES, "lon": _GRID_LONGITUDES}
        no_time = replace(
            inputs,
            temperature=xr.DataArray(inputs.temperature.values, dims=inputs.temperature.dims, coords=coords),
            humidity=xr.DataArray(inputs.humidity.values, dims=inputs.humidity.dims, coords=coords),
            wind=xr.DataArray(inputs.wind.values, dims=inputs.wind.dims, coords=coords),
            precipitation=xr.DataArray(inputs.precipitation.values, dims=inputs.precipitation.dims, coords=coords),
        )
        with pytest.raises(InvalidArgumentError, match="month is required"):
            _xarray_cffwis(no_time)


class TestCFFWISXarrayTimeSemantics:
    """CFFWIS is a daily, noon-referenced recurrence; the adapter checks both assumptions."""

    def test_midnight_coordinate_warns_about_noon(self) -> None:
        inputs = _gridded_inputs(hour=0)
        with pytest.warns(ClimateIndicesWarning, match="noon"):
            _xarray_cffwis(inputs)

    def test_noon_coordinate_does_not_warn(self) -> None:
        inputs = _gridded_inputs(hour=12)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ClimateIndicesWarning)
            _xarray_cffwis(inputs)

    def test_sub_daily_coordinate_raises(self) -> None:
        inputs = _gridded_inputs(days=5)
        hourly = inputs.temperature.assign_coords(time=pd.date_range("2000-01-01 12:00", periods=5, freq="h"))
        with pytest.raises(CoordinateValidationError, match="daily"):
            fire.cffwis(hourly, inputs.humidity, inputs.wind, inputs.precipitation)

    def test_time_dimension_split_across_chunks_raises(self) -> None:
        inputs = _gridded_inputs()
        split = replace(inputs, temperature=inputs.temperature.chunk({"time": 6, "lat": 1, "lon": 1}))
        with pytest.raises(CoordinateValidationError, match="single chunk"):
            _xarray_cffwis(split)


class TestCFFWISXarrayOutputs:
    """Selection, metadata, spin-up, and state follow the NumPy contract on the xarray route."""

    def test_outputs_selection_returns_only_requested_variables(self) -> None:
        inputs = _gridded_inputs()
        result = _xarray_cffwis(inputs, outputs=["ffmc", "fwi"])
        assert isinstance(result, xr.Dataset)
        assert list(result.data_vars) == ["ffmc", "fwi"]
        expected = _numpy_cffwis(inputs, outputs=["ffmc", "fwi"])
        np.testing.assert_array_equal(result["ffmc"].values, expected.ffmc)
        np.testing.assert_array_equal(result["fwi"].values, expected.fwi)

    def test_single_output_selection_returns_one_variable(self) -> None:
        """A one-name selection exercises the single-output apply_ufunc shape."""
        inputs = _gridded_inputs()
        result = _xarray_cffwis(inputs, outputs="fwi")
        assert isinstance(result, xr.Dataset)
        assert list(result.data_vars) == ["fwi"]
        expected = _numpy_cffwis(inputs, outputs="fwi")
        np.testing.assert_array_equal(result["fwi"].values, expected.fwi)

    def test_registry_metadata_and_provenance(self) -> None:
        inputs = _gridded_inputs()
        result = _xarray_cffwis(inputs)
        assert isinstance(result, xr.Dataset)
        for name in _GRID_VARIABLES:
            entry = CF_METADATA[name]
            attrs = result[name].attrs
            assert attrs["long_name"] == entry["long_name"]
            assert attrs["units"] == entry["units"]
            assert attrs["references"] == entry["references"]
            assert attrs["climate_indices_variant"] == "cffwis_classic"
            assert "standard_name" not in attrs
            assert "climate_indices_version" in attrs
            assert name.upper() in attrs["history"]

    def test_input_standard_name_is_not_inherited(self) -> None:
        inputs = _gridded_inputs()
        result = fire.cffwis(
            inputs.temperature.assign_attrs(standard_name="air_temperature"),
            inputs.humidity,
            inputs.wind,
            inputs.precipitation,
        )
        assert isinstance(result, xr.Dataset)
        for name in _GRID_VARIABLES:
            assert "standard_name" not in result[name].attrs

    def test_spin_up_trims_the_time_coordinate(self) -> None:
        inputs = _gridded_inputs()
        result = _xarray_cffwis(inputs, spin_up=4)
        assert isinstance(result, xr.Dataset)
        assert result.sizes["time"] == inputs.temperature.sizes["time"] - 4
        np.testing.assert_array_equal(result["fwi"].coords["time"].values, inputs.temperature.coords["time"].values[4:])
        _assert_matches_numpy(inputs, result, spin_up=4)

    def test_dask_return_state_matches_numpy(self) -> None:
        """The state must be computed eagerly while the value arrays stay lazy."""
        inputs = _gridded_inputs(days=9)
        chunked = replace(
            inputs,
            temperature=inputs.temperature.chunk({"time": -1, "lat": 1, "lon": 1}),
            humidity=inputs.humidity.chunk({"time": -1, "lat": 1, "lon": 1}),
            wind=inputs.wind.chunk({"time": -1, "lat": 1, "lon": 1}),
            precipitation=inputs.precipitation.chunk({"time": -1, "lat": 1, "lon": 1}),
        )
        expected = _numpy_cffwis(inputs, return_state=True)
        result = _xarray_cffwis(chunked, return_state=True)
        assert isinstance(result, fire.CFFWISResult)
        assert result.state is not None and expected.state is not None
        np.testing.assert_array_equal(result.state.dmc.dmc, expected.state.dmc.dmc)
        np.testing.assert_array_equal(result.state.dc.dc, expected.state.dc.dc)
        for name in _GRID_VARIABLES:
            value = getattr(result, name)
            assert isinstance(value, xr.DataArray)
            np.testing.assert_array_equal(value.values, getattr(expected, name))

    def test_bridge_gap_matches_numpy(self) -> None:
        inputs = _gridded_inputs()
        gapped = inputs.precipitation.copy()
        gapped.values[5, 0, 0] = np.nan
        gapped_inputs = replace(inputs, precipitation=gapped)
        result = _xarray_cffwis(gapped_inputs, nan_policy="bridge", max_gap_days=2)
        assert isinstance(result, xr.Dataset)
        expected = _numpy_cffwis(gapped_inputs, nan_policy="bridge", max_gap_days=2)
        for name in _GRID_VARIABLES:
            np.testing.assert_array_equal(result[name].values, getattr(expected, name))

    def test_return_state_matches_numpy(self) -> None:
        inputs = _gridded_inputs(days=10)
        expected = _numpy_cffwis(inputs, return_state=True)
        result = _xarray_cffwis(inputs, return_state=True)
        assert isinstance(result, fire.CFFWISResult)
        for name in _GRID_VARIABLES:
            value = getattr(result, name)
            assert isinstance(value, xr.DataArray)
            np.testing.assert_array_equal(value.values, getattr(expected, name))
        assert result.state is not None and expected.state is not None
        np.testing.assert_array_equal(result.state.ffmc.ffmc, expected.state.ffmc.ffmc)
        np.testing.assert_array_equal(result.state.dmc.dmc, expected.state.dmc.dmc)
        np.testing.assert_array_equal(result.state.dc.dc, expected.state.dc.dc)
        for code in ("ffmc", "dmc", "dc"):
            result_gaps = getattr(result.state, code).trailing_gap_days
            expected_gaps = getattr(expected.state, code).trailing_gap_days
            if expected_gaps is None:
                assert result_gaps is None
            else:
                assert result_gaps is not None
                np.testing.assert_array_equal(result_gaps, expected_gaps)

    def test_initial_state_resume_matches_one_shot(self) -> None:
        """Acceptance of the append contract through the xarray route."""
        inputs = _gridded_inputs()
        split = 5
        first = fire.cffwis(
            inputs.temperature.isel(time=slice(0, split)),
            inputs.humidity.isel(time=slice(0, split)),
            inputs.wind.isel(time=slice(0, split)),
            inputs.precipitation.isel(time=slice(0, split)),
            return_state=True,
        )
        assert isinstance(first, fire.CFFWISResult) and first.state is not None
        resumed = fire.cffwis(
            inputs.temperature.isel(time=slice(split, None)),
            inputs.humidity.isel(time=slice(split, None)),
            inputs.wind.isel(time=slice(split, None)),
            inputs.precipitation.isel(time=slice(split, None)),
            initial_state=first.state,
        )
        assert isinstance(resumed, xr.Dataset)
        one_shot = _numpy_cffwis(inputs)
        for name in _GRID_VARIABLES:
            np.testing.assert_array_equal(resumed[name].values, getattr(one_shot, name)[split:])

    def test_scalar_initial_seed_changes_the_start(self) -> None:
        inputs = _gridded_inputs(days=8)
        seeded = _xarray_cffwis(inputs, initial_ffmc=60.0)
        assert isinstance(seeded, xr.Dataset)
        expected = _numpy_cffwis(inputs, initial_ffmc=60.0)
        np.testing.assert_array_equal(seeded["ffmc"].values, expected.ffmc)


class TestCFFWISXarrayUnits:
    """A CF ``units`` attribute on temperature and precipitation drives conversion, not guessing."""

    def test_kelvin_temperature_is_converted(self) -> None:
        inputs = _gridded_inputs()
        kelvin = (inputs.temperature + 273.15).assign_attrs(units="K")
        result = fire.cffwis(kelvin, inputs.humidity, inputs.wind, inputs.precipitation)
        assert isinstance(result, xr.Dataset)
        expected = _xarray_cffwis(inputs)
        assert isinstance(expected, xr.Dataset)
        np.testing.assert_allclose(result["ffmc"].values, expected["ffmc"].values, rtol=1e-10)

    def test_precipitation_inches_are_converted(self) -> None:
        inputs = _gridded_inputs()
        inches = (inputs.precipitation / 25.4).assign_attrs(units="inch")
        result = fire.cffwis(inputs.temperature, inputs.humidity, inputs.wind, inches)
        assert isinstance(result, xr.Dataset)
        expected = _xarray_cffwis(inputs)
        assert isinstance(expected, xr.Dataset)
        np.testing.assert_allclose(result["ffmc"].values, expected["ffmc"].values, rtol=1e-10)


class TestCFFWISXarrayValidation:
    """Type and alignment errors follow the fire adapter conventions."""

    def test_mixed_input_types_raise_type_error(self) -> None:
        inputs = _gridded_inputs()
        with pytest.raises(TypeError, match="same type"):
            fire.cffwis(inputs.temperature, inputs.humidity.values, inputs.wind, inputs.precipitation)

    def test_alignment_warns_and_uses_the_intersection(self) -> None:
        inputs = _gridded_inputs()
        shifted = inputs.precipitation.assign_coords(time=inputs.precipitation.coords["time"] + pd.Timedelta(days=1))
        with pytest.warns(InputAlignmentWarning):
            result = fire.cffwis(inputs.temperature, inputs.humidity, inputs.wind, shifted)
        assert isinstance(result, xr.Dataset)
        assert result.sizes["time"] == inputs.temperature.sizes["time"] - 1

    def test_disjoint_time_ranges_raise(self) -> None:
        inputs = _gridded_inputs()
        shifted = inputs.precipitation.assign_coords(time=inputs.precipitation.coords["time"] + pd.Timedelta(days=400))
        with pytest.raises(CoordinateValidationError, match="No overlapping timesteps"):
            fire.cffwis(inputs.temperature, inputs.humidity, inputs.wind, shifted)

    def test_invalid_seed_raises_eagerly(self) -> None:
        """A bad seed must fail the call, before any lazy evaluation can run."""
        inputs = _gridded_inputs()
        chunked = replace(
            inputs,
            temperature=inputs.temperature.chunk({"time": -1, "lat": 1, "lon": 1}),
            humidity=inputs.humidity.chunk({"time": -1, "lat": 1, "lon": 1}),
            wind=inputs.wind.chunk({"time": -1, "lat": 1, "lon": 1}),
            precipitation=inputs.precipitation.chunk({"time": -1, "lat": 1, "lon": 1}),
        )
        with pytest.raises(InvalidArgumentError, match="ffmc must be finite"):
            _xarray_cffwis(chunked, initial_ffmc=500.0)

    def test_negative_precipitation_raises(self) -> None:
        inputs = _gridded_inputs()
        negative = inputs.precipitation.copy()
        negative.values[0, 0, 0] = -1.0
        with pytest.raises(InvalidArgumentError, match="non-negative"):
            _xarray_cffwis(replace(inputs, precipitation=negative))

    def test_non_string_units_attribute_raises(self) -> None:
        inputs = _gridded_inputs()
        kelvin = (inputs.temperature + 273.15).assign_attrs(units=b"K")
        with pytest.raises(InvalidArgumentError, match="not a CF units string"):
            fire.cffwis(kelvin, inputs.humidity, inputs.wind, inputs.precipitation)

    def test_radian_latitude_coordinates_raise(self) -> None:
        inputs = _gridded_inputs()

        def radians(data: xr.DataArray) -> xr.DataArray:
            return data.assign_coords(lat=data.coords["lat"].assign_attrs(units="radians"))

        with pytest.raises(InvalidArgumentError, match="Unsupported latitude units"):
            _xarray_cffwis(
                replace(
                    inputs,
                    temperature=radians(inputs.temperature),
                    humidity=radians(inputs.humidity),
                    wind=radians(inputs.wind),
                    precipitation=radians(inputs.precipitation),
                )
            )

    def test_latitude_with_unknown_dimension_raises(self) -> None:
        inputs = _gridded_inputs()
        unknown_axis = xr.DataArray([40.0, 45.0], dims=("y",))
        with pytest.raises(InvalidArgumentError, match="dimensions the weather inputs do not have"):
            _xarray_cffwis(inputs, latitude_degrees_north=unknown_axis)

    def test_conflicting_dimension_sizes_raise_coordinate_error(self) -> None:
        inputs = _gridded_inputs()
        coords = {"lat": _GRID_LATITUDES, "lon": _GRID_LONGITUDES}

        def bare(data: xr.DataArray, days: int) -> xr.DataArray:
            return xr.DataArray(data.values[:days], dims=data.dims, coords=coords)

        with pytest.raises(CoordinateValidationError, match="Cannot align"):
            fire.cffwis(
                bare(inputs.temperature, 10),
                bare(inputs.humidity, 8),
                bare(inputs.wind, 10),
                bare(inputs.precipitation, 10),
            )
