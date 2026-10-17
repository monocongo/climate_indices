"""NumPy degree-day kernels (#1164); the validation oracle lives in test_energy_reference.py.

This module keeps a thin default-lane check that reads the committed CPC fixture
rather than restating its numbers, so the oracle has one source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from climate_indices import energy
from climate_indices.exceptions import InputTypeError, InvalidArgumentError

FIXTURE = Path(__file__).parent / "fixture" / "energy_cpc" / "reference.json"


def _load() -> dict[str, Any]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_cooling_matches_cpc_published_seven_day_example() -> None:
    example = _load()["published_cdd"]
    temperature = (np.asarray(example["temperature_f"], dtype=np.float64) - 32) * (5 / 9)
    actual = energy.cooling_degree_days(temperature, base_temperature=(example["base_f"] - 32) * (5 / 9))
    np.testing.assert_allclose(actual, np.asarray(example["cdd_f_day"]) * (5 / 9), rtol=0.0, atol=1e-12)


def test_energy_is_heating_plus_cooling_with_distinct_bases() -> None:
    temperature = np.array([5.0, 15.0, 25.0])
    actual = energy.energy_degree_days(temperature, heating_base_temperature=18.0, cooling_base_temperature=24.0)
    np.testing.assert_array_equal(actual, [13.0, 3.0, 1.0])


def test_missing_values_propagate_for_masked_and_unmasked_input() -> None:
    masked = energy.heating_degree_days(
        np.ma.masked_array([10.0, 20.0, 30.0], mask=[False, True, False]), base_temperature=18.0
    )
    assert masked[0] == 8.0
    assert np.isnan(masked[1])
    assert masked[2] == 0.0
    assert np.isnan(energy.cooling_degree_days([10.0, np.nan, 30.0], base_temperature=18.0)[1])


def test_non_numeric_temperature_is_rejected() -> None:
    with pytest.raises(InputTypeError):
        energy.cooling_degree_days(["cold", "warm"], base_temperature=18.0)


def test_infinite_temperature_is_rejected() -> None:
    with pytest.raises(InvalidArgumentError):
        energy.heating_degree_days([np.inf], base_temperature=18.0)


@pytest.mark.parametrize("base", [np.nan, np.inf, [18.0], np.array([18.0, 19.0]), "18"])
def test_non_scalar_or_non_finite_base_is_rejected(base: object) -> None:
    with pytest.raises(InvalidArgumentError):
        energy.heating_degree_days([10.0], base_temperature=base)  # type: ignore[arg-type]
