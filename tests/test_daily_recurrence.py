"""Shared-contract tests for the one daily-recurrence runtime (#1220).

ADR-0007 (missing-day policy) and ADR-0006 (recursive state) apply to every
stateful daily index. Those contracts used to be re-asserted per index, each
with its own copy of the gap matrix and its own monkeypatch of a private
function. This module is the one parametrized gate over all six recurrences for
the shared runtime: gap-policy routing, the identical ``trailing_gap_days``
bound, the bitwise append round trip, and the core propagate/bridge matrix. The
exhaustive ADR-0007 rows (all-NaN, leading/trailing blocks, the split-append
that exceeds the limit only once joined) remain parametrized per index in the
family test modules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from climate_indices import _recurrence, fire
from climate_indices.exceptions import InvalidArgumentError
from climate_indices.flood._antecedent import (
    APIResult,
    APIState,
    antecedent_precipitation_index,
)

_N_DAYS = 8
_BASE = np.array([20.0, 25.0, 30.0, 15.0, 22.0, 28.0, 12.0, 18.0])
_LATITUDE = np.asarray(45.0)
_MONTH = np.asarray([1, 1, 2, 2, 3, 3, 4, 4])
_OVERFLOW_GAP = np.asarray(1e19)


def _gapped(days: list[int]) -> np.ndarray:
    series = _BASE.copy()
    series[days] = np.nan
    return series


@dataclass(frozen=True)
class _Case:
    name: str
    run: Callable[..., object]
    state_with_gaps: Callable[[], object]


def _ffmc(series: np.ndarray, month: np.ndarray, **kwargs: object) -> object:
    del month
    return fire.ffmc(series, series, series, series, **kwargs)


def _dmc(series: np.ndarray, month: np.ndarray, **kwargs: object) -> object:
    return fire.duff_moisture_code(series, series, series, _LATITUDE, month, **kwargs)


def _dc(series: np.ndarray, month: np.ndarray, **kwargs: object) -> object:
    return fire.drought_code(series, series, _LATITUDE, month, **kwargs)


def _kbdi(series: np.ndarray, month: np.ndarray, **kwargs: object) -> object:
    del month
    return fire.kbdi(series, series, 1000.0, **kwargs)


def _api(series: np.ndarray, month: np.ndarray, **kwargs: object) -> object:
    del month
    return antecedent_precipitation_index(series, 0.9, **kwargs)


def _cffwis(series: np.ndarray, month: np.ndarray, **kwargs: object) -> object:
    kwargs.setdefault("outputs", ["ffmc"])
    return fire.cffwis(series, series, series, series, _LATITUDE, month, **kwargs)


def _cffwis_state_with_gaps() -> fire.CFFWISState:
    return fire.CFFWISState(
        ffmc=fire.FFMCState(ffmc=np.asarray(85.0), trailing_gap_days=_OVERFLOW_GAP),
        dmc=fire.DMCState(dmc=np.asarray(6.0), trailing_gap_days=_OVERFLOW_GAP),
        dc=fire.DCState(dc=np.asarray(15.0), trailing_gap_days=_OVERFLOW_GAP),
    )


_CASES = [
    _Case(
        "ffmc",
        _ffmc,
        lambda: fire.FFMCState(ffmc=np.asarray(85.0), trailing_gap_days=_OVERFLOW_GAP),
    ),
    _Case(
        "dmc",
        _dmc,
        lambda: fire.DMCState(dmc=np.asarray(6.0), trailing_gap_days=_OVERFLOW_GAP),
    ),
    _Case(
        "dc",
        _dc,
        lambda: fire.DCState(dc=np.asarray(15.0), trailing_gap_days=_OVERFLOW_GAP),
    ),
    _Case(
        "kbdi",
        _kbdi,
        lambda: fire.KBDIState(
            kbdi=np.asarray(0.0),
            wet_spell_precipitation=np.asarray(0.0),
            trailing_gap_days=_OVERFLOW_GAP,
        ),
    ),
    _Case("api", _api, lambda: APIState(api=np.asarray(0.0), trailing_gap_days=_OVERFLOW_GAP)),
    _Case("cffwis", _cffwis, _cffwis_state_with_gaps),
]

_IDS = [case.name for case in _CASES]


def _outcome(case: _Case, result: object) -> tuple[np.ndarray, object]:
    """Reduce a recurrence's return value to its ``(values, state)`` pair."""
    if isinstance(result, fire.CFFWISResult):
        assert result.ffmc is not None
        return result.ffmc, result.state
    if isinstance(result, APIResult):
        assert isinstance(result.values, np.ndarray)
        return result.values, result.state
    if hasattr(result, "values") and hasattr(result, "state"):
        assert isinstance(result.values, np.ndarray)
        return result.values, result.state
    assert isinstance(result, np.ndarray)
    return result, None


@pytest.mark.parametrize(("nan_policy", "max_gap_days"), [("propagate", 0), ("bridge", 1)])
@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_gap_policy_routes_through_the_one_runtime(
    case: _Case,
    monkeypatch: pytest.MonkeyPatch,
    nan_policy: str,
    max_gap_days: int,
) -> None:
    """ADR-0007: every recurrence runs the single day policy in _recurrence, forwarding the caller's options."""
    calls: list[dict[str, object]] = []
    shared = _recurrence._apply_gap_policy

    def recording_helper(*args: object, **kwargs: object) -> object:
        calls.append(kwargs)
        return shared(*args, **kwargs)

    monkeypatch.setattr(_recurrence, "_apply_gap_policy", recording_helper)

    case.run(_gapped([2]), _MONTH, nan_policy=nan_policy, max_gap_days=max_gap_days)

    assert calls, f"{case.name} never reached the shared gap policy"
    assert all(call["nan_policy"] == nan_policy and call["max_gap_days"] == max_gap_days for call in calls)
    if case.name != "cffwis":
        # every day routes through the policy; the cffwis orchestrator takes its
        # all-valid fast path on the fully valid days and only reaches the policy
        # on the missing day, once per component
        assert len(calls) == _N_DAYS


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_trailing_gap_days_rejects_int64_overflow(case: _Case) -> None:
    """Every recurrence applies the same gap-count bound, above 2**63."""
    overflow_state = case.state_with_gaps()
    with pytest.raises(InvalidArgumentError, match="trailing_gap_days"):
        case.run(_BASE, _MONTH, initial_state=overflow_state)


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_resume_round_trip_is_bitwise(case: _Case) -> None:
    """ADR-0006: a run split at an append boundary reproduces the continuous run bit for bit."""
    continuous = _outcome(case, case.run(_BASE, _MONTH, return_state=True))[0]

    first = _outcome(case, case.run(_BASE[:4], _MONTH[:4], return_state=True))
    second = _outcome(case, case.run(_BASE[4:], _MONTH[4:], initial_state=first[1], return_state=True))

    assert np.array_equal(np.concatenate([first[0], second[0]]), continuous)


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_propagate_poisons_from_an_interior_gap(case: _Case) -> None:
    """ADR-0007 propagate: a missing day poisons the started recurrence onward."""
    values = _outcome(case, case.run(_gapped([3]), _MONTH))[0]

    assert np.isfinite(values[:3]).all()
    assert np.isnan(values[3:]).all()


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_bridge_skips_a_gap_within_the_limit(case: _Case) -> None:
    """ADR-0007 bridge: a run no longer than max_gap_days resumes from the last valid state."""
    values = _outcome(case, case.run(_gapped([3]), _MONTH, nan_policy="bridge", max_gap_days=1))[0]

    assert np.isnan(values[3])
    assert np.isfinite(values[:3]).all()
    assert np.isfinite(values[4:]).all()


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_bridge_poisons_once_the_limit_is_exceeded(case: _Case) -> None:
    """ADR-0007 bridge: the first run past max_gap_days poisons the rest."""
    values = _outcome(case, case.run(_gapped([3, 4]), _MONTH, nan_policy="bridge", max_gap_days=1))[0]

    assert np.isfinite(values[:3]).all()
    assert np.isnan(values[3:]).all()


def test_kbdi_time_series_temperature_broadcasts_along_time() -> None:
    """A (time,) temperature series spans cells via the shared left-aligned broadcast."""
    precipitation = np.zeros((3, 3))
    temperature = np.array([20.0, 30.0, 40.0])

    along_time = fire.kbdi(precipitation, temperature, 800.0)
    per_day = fire.kbdi(precipitation, np.broadcast_to(temperature[:, None], (3, 3)), 800.0)

    assert np.array_equal(along_time, per_day)


def test_kbdi_spatial_temperature_without_a_time_axis_raises() -> None:
    """A spatial-only temperature field cannot broadcast against a (time, cells) grid."""
    with pytest.raises(InvalidArgumentError):
        fire.kbdi(np.zeros((6, 3)), np.full(3, 20.0), 1000.0)
