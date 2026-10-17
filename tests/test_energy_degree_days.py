"""NumPy degree-day kernels (#1164); the CPC oracle lives in test_energy_reference.py."""

from __future__ import annotations

import numpy as np
import pytest

from climate_indices import energy
from climate_indices.exceptions import InputTypeError, InvalidArgumentError


def test_cooling_matches_cpc_published_seven_day_example() -> None:
    temperature = (np.array([67, 65, 70, 74, 78, 65, 68], dtype=np.float64) - 32) * (5 / 9)
    actual = energy.cooling_degree_days(temperature, base_temperature=(65 - 32) * (5 / 9))
    np.testing.assert_allclose(actual, np.array([2, 0, 5, 9, 13, 0, 3]) * (5 / 9), rtol=0.0, atol=1e-12)


def test_energy_is_heating_plus_cooling_with_distinct_bases() -> None:
    temperature = np.array([5.0, 15.0, 25.0])
    actual = energy.energy_degree_days(temperature, heating_base_temperature=18.0, cooling_base_temperature=24.0)
    np.testing.assert_array_equal(actual, [13.0, 3.0, 1.0])


def test_missing_values_propagate_and_masked_arrays_are_accepted() -> None:
    temperature = np.ma.masked_array([10.0, 20.0, 30.0], mask=[False, True, False])
    actual = energy.heating_degree_days(temperature, base_temperature=18.0)
    assert actual[0] == 8.0
    assert np.isnan(actual[1])
    assert actual[2] == 0.0


def test_non_numeric_temperature_is_rejected() -> None:
    with pytest.raises(InputTypeError):
        energy.cooling_degree_days(["cold", "warm"], base_temperature=18.0)


def test_infinite_inputs_are_rejected() -> None:
    with pytest.raises(InvalidArgumentError):
        energy.heating_degree_days([np.inf], base_temperature=18.0)
    with pytest.raises(InvalidArgumentError):
        energy.heating_degree_days([10.0], base_temperature=np.inf)
