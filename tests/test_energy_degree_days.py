"""NumPy degree-day kernels (#1164); the validation oracle lives in test_energy_reference.py.

This module keeps a thin default-lane check that reads the committed CPC fixture
rather than restating its numbers, so the oracle has one source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest

from climate_indices import energy
from climate_indices.energy import _degree_days
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


@pytest.mark.parametrize(
    ("heating_base", "cooling_base", "expected"),
    [(18.0, 24.0, [13.0, 3.0, 0.0, 1.0]), (18.0, 18.0, [13.0, 3.0, 2.0, 7.0]), (24.0, 18.0, [19.0, 9.0, 6.0, 7.0])],
)
def test_energy_is_heating_plus_cooling(heating_base: float, cooling_base: float, expected: list[float]) -> None:
    temperature = np.array([5.0, 15.0, 20.0, 25.0])
    actual = energy.energy_degree_days(
        temperature, heating_base_temperature=heating_base, cooling_base_temperature=cooling_base
    )
    np.testing.assert_array_equal(actual, expected)


def test_energy_validates_temperature_once_for_masked_integer_grid() -> None:
    temperature = np.ma.masked_array([[5, 20], [25, 0]], dtype=np.int16, mask=[[False, False], [False, True]])
    with patch.object(_degree_days, "_validated_temperature", wraps=_degree_days._validated_temperature) as validate:
        actual = energy.energy_degree_days(temperature, heating_base_temperature=18.0, cooling_base_temperature=24.0)
    validate.assert_called_once_with(temperature)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, [[13.0, 0.0], [1.0, np.nan]])
    np.testing.assert_array_equal(temperature.data, [[5, 20], [25, 0]])


@pytest.mark.parametrize("temperature", [5.0, np.ma.masked_array(5.0, mask=True)])
def test_energy_scalar_returns_zero_dimensional_array(temperature: npt.ArrayLike) -> None:
    actual = energy.energy_degree_days(temperature, heating_base_temperature=18.0, cooling_base_temperature=24.0)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == ()
    np.testing.assert_array_equal(actual, np.nan if np.ma.is_masked(temperature) else 13.0)


@pytest.mark.parametrize("temperature", [[np.nan, 20.0], np.array([], dtype=np.float32)])
def test_energy_preserves_missing_and_empty_inputs(temperature: npt.ArrayLike) -> None:
    actual = energy.energy_degree_days(temperature, heating_base_temperature=18.0, cooling_base_temperature=24.0)
    np.testing.assert_array_equal(actual, [np.nan, 0.0] if np.size(temperature) else [])


@pytest.mark.parametrize("temperature, error", [(["cold"], InputTypeError), ([np.inf], InvalidArgumentError)])
def test_energy_rejects_invalid_temperature(temperature: npt.ArrayLike, error: type[Exception]) -> None:
    with pytest.raises(error):
        energy.energy_degree_days(temperature, heating_base_temperature=18.0, cooling_base_temperature=24.0)


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


@pytest.mark.parametrize(
    ("kind", "base_name"),
    [
        ("heating", "base_temperature"),
        ("cooling", "base_temperature"),
        ("energy", "heating_base_temperature"),
        ("energy", "cooling_base_temperature"),
    ],
)
@pytest.mark.parametrize(
    "base",
    [np.nan, np.inf, -np.inf, [18.0], np.array([18.0, 19.0]), "18", np.ma.masked, np.ma.masked_array(18.0, mask=True)],
)
def test_non_scalar_non_finite_or_masked_base_is_rejected(kind: str, base_name: str, base: object) -> None:
    kwargs: dict[str, object] = (
        {"heating_base_temperature": 18.0, "cooling_base_temperature": 24.0} if kind == "energy" else {}
    )
    kwargs[base_name] = base
    func = getattr(energy, f"{kind}_degree_days")
    with pytest.raises(InvalidArgumentError, match=base_name):
        func([10.0], **kwargs)


def test_unmasked_scalar_base_is_accepted() -> None:
    base: Any = np.ma.masked_array(18.0, mask=False)
    actual = energy.heating_degree_days([10.0], base_temperature=base)
    np.testing.assert_array_equal(actual, [8.0])
