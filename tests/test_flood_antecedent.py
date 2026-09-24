"""Antecedent Precipitation Index recursion and append contract."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from climate_indices import flood
from climate_indices.exceptions import InputTypeError, InvalidArgumentError

api = flood.antecedent_precipitation_index


def test_source_recursion_and_closed_form() -> None:
    # Kohler & Linsley (1951), Eq. (3): today's rain is added after decay.
    np.testing.assert_array_equal(api([4, 0, 2], 0.5), [4, 2, 3])
    values = api(np.ones(100) * 2, 0.5)
    assert values[-1] == pytest.approx(2 / (1 - 0.5))
    np.testing.assert_array_equal(api([4, 0, 2], 0.5, spin_up=1), [2, 3])


def test_state_round_trip_and_year_layout() -> None:
    values = np.array([2, 3, 0, 4, 1, 5], dtype=float)
    first = api(values[:3], 0.8, return_state=True)
    second = api(values[3:], 0.8, initial_state=first.state)
    np.testing.assert_array_equal(np.concatenate((first.values, second)), api(values, 0.8))
    by_year = api(values.reshape(2, 3), 0.8)
    assert by_year.shape == (2, 3)
    np.testing.assert_array_equal(by_year.reshape(-1), api(values, 0.8))
    np.testing.assert_array_equal(api(values.reshape(2, 3), 0.8, spin_up=2), api(values, 0.8, spin_up=2))


@pytest.mark.parametrize(("policy", "limit"), [("propagate", 0), ("bridge", 2), ("bridge", 1)])
def test_gap_policy_across_append_boundary(policy: str, limit: int) -> None:
    values = np.array([np.nan, 2, np.nan, np.nan, 4, 1])
    full = api(values, 0.5, nan_policy=policy, max_gap_days=limit)
    first = api(values[:3], 0.5, nan_policy=policy, max_gap_days=limit, return_state=True)
    tail = api(values[3:], 0.5, nan_policy=policy, max_gap_days=limit, initial_state=first.state)
    np.testing.assert_array_equal(np.concatenate((first.values, tail)), full)
    assert np.isnan(full[0])
    assert np.isnan(full[2:4]).all()
    assert np.isnan(full[4]) == (policy == "propagate" or limit == 1)


def test_leading_missing_days_leave_state_unstarted() -> None:
    blank = api([np.nan], 0.5, return_state=True)
    assert blank.state.trailing_gap_days is None
    assert api([3], 0.5, initial_state=blank.state)[0] == pytest.approx(3)


@pytest.mark.parametrize("run", [1, 2, 3])
def test_bridged_run_equals_recurrence_over_valid_days_alone(run: int) -> None:
    # ADR-0007: a bridged day is "no change", so it neither decays nor adds.
    rain = np.array([3.0, 1.0, *[np.nan] * run, 4.0, 2.0])
    bridged = api(rain, 0.8, nan_policy="bridge", max_gap_days=run)
    valid = np.isfinite(rain)
    np.testing.assert_array_equal(bridged[valid], api(rain[valid], 0.8))
    assert np.isnan(bridged[~valid]).all()


@pytest.mark.parametrize("limit", [1, 3])
@pytest.mark.parametrize("over", [False, True])
def test_interior_and_trailing_run_at_and_beyond_limit(limit: int, over: bool) -> None:
    run = limit + over
    rain = np.array([2.0, *[np.nan] * run, 5.0])
    interior = api(rain, 0.5, nan_policy="bridge", max_gap_days=limit)
    assert np.isnan(interior[-1]) == over
    trailing = api(rain[:-1], 0.5, nan_policy="bridge", max_gap_days=limit, return_state=True).state
    assert np.isnan(float(trailing.api)) == over
    assert int(trailing.trailing_gap_days) == run
    if not over:
        assert float(trailing.api) == pytest.approx(2.0)


def test_separate_runs_each_within_limit_do_not_accumulate() -> None:
    out = api([1.0, np.nan, 2.0, np.nan, 3.0], 0.5, nan_policy="bridge", max_gap_days=1)
    assert np.isfinite(out[[0, 2, 4]]).all()


@pytest.mark.parametrize(("policy", "limit"), [("propagate", 0), ("bridge", 2)])
def test_split_after_valid_day_then_leading_missing_days_in_tail(policy: str, limit: int) -> None:
    rain = np.array([1.0, 2.0, np.nan, np.nan, np.nan, 3.0])
    one = api(rain, 0.5, nan_policy=policy, max_gap_days=limit)
    first = api(rain[:2], 0.5, nan_policy=policy, max_gap_days=limit, return_state=True)
    assert int(first.state.trailing_gap_days) == 0
    tail = api(rain[2:], 0.5, nan_policy=policy, max_gap_days=limit, initial_state=first.state)
    np.testing.assert_array_equal(np.concatenate((first.values, tail)), one)


def test_all_missing_call_on_unstarted_and_started_state() -> None:
    blank = api(np.full(4, np.nan), 0.5, return_state=True)
    assert np.isnan(blank.values).all()
    assert blank.state.trailing_gap_days is None
    started = api([1.0], 0.5, return_state=True).state
    poisoned = api(np.full(2, np.nan), 0.5, initial_state=started, return_state=True)
    assert np.isnan(poisoned.values).all()
    assert np.isnan(float(poisoned.state.api))
    bridged = api(
        np.full(2, np.nan), 0.5, initial_state=started, nan_policy="bridge", max_gap_days=2, return_state=True
    )
    assert float(bridged.state.api) == pytest.approx(1.0)
    assert int(bridged.state.trailing_gap_days) == 2


def test_masked_values_are_missing_not_their_fill_data() -> None:
    masked = np.ma.masked_array([1.0, -9999.0, 2.0], mask=[False, True, False])
    np.testing.assert_array_equal(api(masked, 0.5, nan_policy="bridge", max_gap_days=1), [1.0, np.nan, 2.5])
    assert np.isnan(api(masked, 0.5)[1:]).all()


def test_spin_up_beyond_length_is_empty_but_state_advances() -> None:
    result = api([1.0, 2.0], 0.5, spin_up=5, return_state=True)
    assert result.values.shape == (0,)
    assert float(result.state.api) == pytest.approx(2.5)


def test_missing_days_inside_spin_up_still_obey_gap_policy() -> None:
    rain = np.array([1.0, np.nan, np.nan, 2.0, 3.0])
    assert np.isnan(api(rain, 0.5, spin_up=1)[1:]).all()
    bridged = api(rain, 0.5, nan_policy="bridge", max_gap_days=2)
    np.testing.assert_array_equal(api(rain, 0.5, spin_up=3, nan_policy="bridge", max_gap_days=2), bridged[3:])
    assert np.isnan(api(rain, 0.5, spin_up=3, nan_policy="bridge", max_gap_days=1)).all()


@pytest.mark.parametrize("shape", [(5, 0, 3), (5, 3, 0), (5, 0, 0)])
def test_empty_spatial_axis_returns_empty_block(shape: tuple[int, ...]) -> None:
    result = api(np.ones(shape), 0.5, return_state=True)
    assert result.values.shape == shape
    assert result.state.api.shape == shape[1:]


@pytest.mark.parametrize("width", [np.int8, np.uint8, np.uint64])
def test_numpy_integer_spin_up_of_any_width(width: type[np.integer]) -> None:
    np.testing.assert_array_equal(api(np.ones(400), 0.5, spin_up=width(3)), api(np.ones(400), 0.5, spin_up=3))
    assert api(np.ones(5), 0.5, spin_up=width(10)).shape == (0,)


def test_spatial_cells_and_state_are_independent() -> None:
    values = np.array([[1, np.nan], [np.nan, 2], [3, 3], [1, 1]])[:, :, None]
    result = api(values, 0.5, return_state=True)
    np.testing.assert_array_equal(result.values[:, 0, 0], [1, np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(result.values[:, 1, 0], [np.nan, 2, 4, 3])
    result.state.api[1, 0] = 99
    assert result.values[-1, 1, 0] == pytest.approx(3)


@pytest.mark.parametrize("cell_axis", [12, 366])
def test_spatial_block_with_calendar_length_cell_axis_needs_declaration(cell_axis: int) -> None:
    block = np.ones((3, cell_axis, 2))
    with pytest.raises(ValueError, match="spatial_time_major=True"):
        api(block, 0.5)
    declared = api(block, 0.5, spatial_time_major=True)
    assert declared.shape == block.shape
    np.testing.assert_allclose(declared[-1], 1.75)


@pytest.mark.parametrize(("policy", "limit"), [("propagate", 0), ("bridge", 1)])
def test_multi_cell_state_with_unstarted_and_poisoned_cells_resumes_bitwise(policy: str, limit: int) -> None:
    nan = np.nan
    rain = np.array([[1, nan, 2], [nan, nan, nan], [3, nan, 1], [nan, 4, 5], [2, 1, nan]], dtype=float)[:, :, None]
    options = {"nan_policy": policy, "max_gap_days": limit}
    one = api(rain, 0.7, return_state=True, **options)
    for cut in range(len(rain) + 1):
        head = api(rain[:cut], 0.7, return_state=True, **options)
        tail = api(rain[cut:], 0.7, initial_state=head.state, return_state=True, **options)
        np.testing.assert_array_equal(np.concatenate((head.values, tail.values)), one.values)
        np.testing.assert_array_equal(tail.state.api, one.state.api)
        np.testing.assert_array_equal(tail.state.trailing_gap_days, one.state.trailing_gap_days)


def test_resume_does_not_mutate_supplied_state() -> None:
    rain = np.ones((3, 2, 2))
    state = api(rain, 0.5, return_state=True).state
    api_before, gaps_before = state.api.copy(), state.trailing_gap_days.copy()
    first = api(rain, 0.5, initial_state=state)
    np.testing.assert_array_equal(state.api, api_before)
    np.testing.assert_array_equal(state.trailing_gap_days, gaps_before)
    np.testing.assert_array_equal(api(rain, 0.5, initial_state=state), first)


@pytest.mark.parametrize("k", [0, 1, -0.1, np.nan, np.inf, True, Fraction(1, 2), np.array(0.5)], ids=repr)
def test_rejects_invalid_decay(k: object) -> None:
    with pytest.raises(InvalidArgumentError, match="k must be a real scalar"):
        api([1], k)


@pytest.mark.parametrize(
    ("values", "error"),
    [([-1], InvalidArgumentError), ([np.inf], InvalidArgumentError), (["rain"], InputTypeError)],
)
def test_rejects_invalid_precipitation(values: list, error: type[Exception]) -> None:
    with pytest.raises(error, match="precipitation"):
        api(values, 0.5)


@pytest.mark.parametrize("kwargs", [{"nan_policy": "bridge"}, {"max_gap_days": 2}, {"spin_up": -1}])
def test_rejects_invalid_configuration(kwargs: dict) -> None:
    with pytest.raises(InvalidArgumentError):
        api([1], 0.5, **kwargs)


def test_rejects_negative_api_state() -> None:
    state = api([2], 0.5, return_state=True).state
    negative = replace(state, api=np.array(-1.0))
    with pytest.raises(InvalidArgumentError, match="initial_state"):
        api([1], 0.5, initial_state=negative)


@pytest.mark.parametrize("gap", [-2, 1.5, np.inf, 2**63, np.iinfo(np.int64).max])
def test_rejects_invalid_gap_state(gap: float) -> None:
    state = api([2], 0.5, return_state=True).state
    bad = replace(state, trailing_gap_days=np.array(gap))
    with pytest.raises(InvalidArgumentError, match="trailing_gap_days"):
        api([1], 0.5, initial_state=bad)


@pytest.mark.parametrize("gaps", [None, np.array(-1)])
def test_rejects_missing_api_in_unstarted_cell(gaps: np.ndarray | None) -> None:
    state = api([2], 0.5, return_state=True).state
    bad = replace(state, api=np.array(np.nan), trailing_gap_days=gaps)
    with pytest.raises(InvalidArgumentError, match="initial_state.api"):
        api([1], 0.5, initial_state=bad)


def test_rejects_initial_state_of_wrong_type() -> None:
    with pytest.raises(InvalidArgumentError, match="initial_state"):
        api([1], 0.5, initial_state="not a state")  # type: ignore[arg-type]


def test_rejects_non_numeric_state_field_without_naming_another_family() -> None:
    state = flood.APIState(api=np.array("3"), trailing_gap_days=None)
    with pytest.raises(InputTypeError, match="must be numeric") as excinfo:
        api([1], 0.5, initial_state=state)
    assert "fire" not in str(excinfo.value).lower()
