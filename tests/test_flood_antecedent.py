"""Antecedent Precipitation Index recursion and append contract."""

from dataclasses import replace

import numpy as np
import pytest

from climate_indices import flood
from climate_indices.exceptions import InputTypeError, InvalidArgumentError


def test_source_recursion_and_closed_form() -> None:
    # Kohler & Linsley (1951), Eq. (3): today's rain is added after decay.
    np.testing.assert_array_equal(flood.antecedent_precipitation_index([4, 0, 2], 0.5), [4, 2, 3])
    values = flood.antecedent_precipitation_index(np.ones(100) * 2, 0.5)
    assert values[-1] == pytest.approx(2 / (1 - 0.5))
    np.testing.assert_array_equal(flood.antecedent_precipitation_index([4, 0, 2], 0.5, spin_up=1), [2, 3])


def test_state_round_trip_and_year_layout() -> None:
    values = np.array([2, 3, 0, 4, 1, 5], dtype=float)
    first = flood.antecedent_precipitation_index(values[:3], 0.8, return_state=True)
    second = flood.antecedent_precipitation_index(values[3:], 0.8, initial_state=first.state)
    np.testing.assert_array_equal(
        np.concatenate((first.values, second)), flood.antecedent_precipitation_index(values, 0.8)
    )
    by_year = flood.antecedent_precipitation_index(values.reshape(2, 3), 0.8)
    assert by_year.shape == (2, 3)
    np.testing.assert_array_equal(by_year.reshape(-1), flood.antecedent_precipitation_index(values, 0.8))
    np.testing.assert_array_equal(
        flood.antecedent_precipitation_index(values.reshape(2, 3), 0.8, spin_up=2),
        flood.antecedent_precipitation_index(values, 0.8, spin_up=2),
    )


def test_gap_policy_across_append_boundary() -> None:
    values = np.array([np.nan, 2, np.nan, np.nan, 4, 1])
    for policy, limit in [("propagate", 0), ("bridge", 2), ("bridge", 1)]:
        full = flood.antecedent_precipitation_index(values, 0.5, nan_policy=policy, max_gap_days=limit)
        first = flood.antecedent_precipitation_index(
            values[:3], 0.5, nan_policy=policy, max_gap_days=limit, return_state=True
        )
        tail = flood.antecedent_precipitation_index(
            values[3:], 0.5, nan_policy=policy, max_gap_days=limit, initial_state=first.state
        )
        np.testing.assert_array_equal(np.concatenate((first.values, tail)), full)
        assert np.isnan(full[0]) and np.isnan(full[2:4]).all()
        assert np.isnan(full[4]) == (policy == "propagate" or limit == 1)
    blank = flood.antecedent_precipitation_index([np.nan], 0.5, return_state=True)
    assert blank.state.trailing_gap_days is None
    assert flood.antecedent_precipitation_index([3], 0.5, initial_state=blank.state)[0] == 3


def test_spatial_cells_and_state_are_independent() -> None:
    values = np.array([[1, np.nan], [np.nan, 2], [3, 3], [1, 1]])[:, :, None]
    result = flood.antecedent_precipitation_index(values, 0.5, return_state=True)
    np.testing.assert_array_equal(result.values[:, 0, 0], [1, np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(result.values[:, 1, 0], [np.nan, 2, 4, 3])
    result.state.api[1, 0] = 99
    assert result.values[-1, 1, 0] == 3
    with pytest.raises(ValueError, match="spatial_time_major=True"):
        flood.antecedent_precipitation_index(np.ones((3, 12, 2)), 0.5)
    assert flood.antecedent_precipitation_index(np.ones((3, 12, 2)), 0.5, spatial_time_major=True).shape == (3, 12, 2)


@pytest.mark.parametrize("k", [0, 1, -0.1, np.nan, np.inf, True])
def test_rejects_invalid_decay(k: float) -> None:
    with pytest.raises(InvalidArgumentError, match="k"):
        flood.antecedent_precipitation_index([1], k)


@pytest.mark.parametrize("values", [[-1], [np.inf], ["rain"]])
def test_rejects_invalid_precipitation(values: list) -> None:
    with pytest.raises((InvalidArgumentError, InputTypeError), match="precipitation|numeric"):
        flood.antecedent_precipitation_index(values, 0.5)


def test_rejects_invalid_configuration_and_state() -> None:
    for kwargs in ({"nan_policy": "bridge"}, {"max_gap_days": 2}, {"spin_up": -1}):
        with pytest.raises(InvalidArgumentError):
            flood.antecedent_precipitation_index([1], 0.5, **kwargs)
    state = flood.antecedent_precipitation_index([2], 0.5, return_state=True).state
    with pytest.raises(InvalidArgumentError, match="initial_state"):
        flood.antecedent_precipitation_index([1], 0.5, initial_state=replace(state, api=np.array(-1.0)))
    for gap in (-2, 1.5, np.inf, 2**63):
        with pytest.raises(InvalidArgumentError, match="trailing_gap_days"):
            flood.antecedent_precipitation_index(
                [1], 0.5, initial_state=replace(state, trailing_gap_days=np.array(gap))
            )
