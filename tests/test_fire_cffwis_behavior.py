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
from dataclasses import dataclass, replace
from timeit import repeat
from unittest import mock

import numpy as np
import pytest

from climate_indices import fire
from climate_indices.exceptions import DataShapeError, InputTypeError, InvalidArgumentError
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
    with pytest.raises(InvalidArgumentError):
        _run_orchestrator(_reference_weather(), outputs=["fwi", "nope"])


def test_empty_outputs_raise() -> None:
    with pytest.raises(InvalidArgumentError):
        _run_orchestrator(_reference_weather(), outputs=[])


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
    assert second.state is not None and whole.state is not None
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
    state = _run_orchestrator(_reference_weather(), return_state=True).state
    with pytest.raises(InvalidArgumentError):
        _run_orchestrator(_reference_weather(), initial_dmc=50.0, initial_state=state)


def test_spin_up_omits_leading_days_without_changing_the_state() -> None:
    weather = _reference_weather()
    spun = _run_orchestrator(weather, spin_up=3, return_state=True)
    full = _run_orchestrator(weather, return_state=True)
    for name in ("ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr"):
        np.testing.assert_array_equal(getattr(spun, name), getattr(full, name)[3:])
    assert spun.state is not None and full.state is not None
    for nested in ("ffmc", "dmc", "dc"):
        np.testing.assert_array_equal(
            getattr(getattr(spun.state, nested), nested), getattr(getattr(full.state, nested), nested)
        )


def test_spin_up_longer_than_the_input_yields_empty_outputs() -> None:
    result = _run_orchestrator(_reference_weather(), spin_up=20)
    assert result.fwi is not None and result.fwi.shape == (0,)


def test_returned_state_does_not_alias_the_outputs() -> None:
    result = _run_orchestrator(_reference_weather(), return_state=True)
    assert result.state is not None and result.ffmc is not None
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


def test_bridge_trailing_gap_state_matches_the_one_shot_run() -> None:
    """ADR-0007: a bridged trailing gap leaves the last valid state plus its count."""
    weather = _with_missing(_series(6), 4)
    result = _run_orchestrator(weather, nan_policy="bridge", max_gap_days=3, return_state=True)
    assert result.dmc is not None and np.isnan(result.dmc[4:]).all()
    assert result.state is not None
    for nested in ("ffmc", "dmc", "dc"):
        state = getattr(result.state, nested)
        assert state.trailing_gap_days is not None
        assert int(state.trailing_gap_days) == 2


def test_bridge_split_gap_append_matches_one_shot() -> None:
    """A gap split across an append boundary stays bridged (ADR-0007)."""
    weather = _with_missing(_series(8), 4)
    one_shot = _run_orchestrator(weather, nan_policy="bridge", max_gap_days=3, return_state=True)
    first = _run_orchestrator(_slice(weather, 0, 5), nan_policy="bridge", max_gap_days=3, return_state=True)
    second = _run_orchestrator(
        _slice(weather, 5, 8), initial_state=first.state, nan_policy="bridge", max_gap_days=3, return_state=True
    )
    assert second.dmc is not None and one_shot.dmc is not None
    np.testing.assert_array_equal(second.dmc, one_shot.dmc[5:])
    assert second.state is not None and one_shot.state is not None
    for nested in ("ffmc", "dmc", "dc"):
        second_state = getattr(second.state, nested)
        one_shot_state = getattr(one_shot.state, nested)
        np.testing.assert_array_equal(second_state.__dict__[nested], one_shot_state.__dict__[nested])
        assert second_state.trailing_gap_days is not None and one_shot_state.trailing_gap_days is not None
        np.testing.assert_array_equal(second_state.trailing_gap_days, one_shot_state.trailing_gap_days)


def test_bridge_gap_past_the_limit_poisons() -> None:
    weather = _with_missing(_series(6), 3)
    result = _assert_matches_chained(weather, nan_policy="bridge", max_gap_days=1)
    assert np.isnan(result.ffmc[3:]).all()


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
    assert gridded.fwi is not None and gridded.fwi.shape == (6, 2, 2)
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
    """#804 acceptance: one pass over the recurrences beats three separate passes."""
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
    orchestrator_time = min(repeat(orchestrator, number=1, repeat=3))
    separate_time = min(repeat(separate, number=1, repeat=3))
    assert orchestrator_time < separate_time
