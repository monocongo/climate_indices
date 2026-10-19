"""SRLI and the residual-load helper (#1172).

The SEI oracle for SRLI lives in test_energy_sei_reference.py; these tests pin what the
wrapper adds: the empirical-only restriction, signed input kept as given, and the
residual-load arithmetic and its input checks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from climate_indices import compute, energy, indices
from climate_indices.exceptions import InputTypeError, InvalidArgumentError

MONTHLY = compute.Periodicity.monthly
ARGS = (3, 1990, 1990, 2019, MONTHLY)


@pytest.fixture
def signed() -> np.ndarray:
    """Thirty years of monthly residual load, with a negative and a positive share."""
    series = np.random.default_rng(1172).normal(0.0, 100.0, 30 * 12)
    assert (series < 0).any()
    assert (series > 0).any()
    return series


def test_srli_is_the_empirical_seam_without_clipping(signed: np.ndarray) -> None:
    expected = indices.standardized_index(
        signed, 3, indices.Distribution.empirical, 1990, 1990, 2019, MONTHLY, clip_negatives=False
    )
    np.testing.assert_array_equal(energy.srli(signed, *ARGS), expected)


def test_srli_keeps_negatives_by_rank(signed: np.ndarray) -> None:
    """Adding a constant leaves every rank, so a score that moved would mean a negative was clipped."""
    shifted = signed + 1000.0
    assert (shifted > 0).all()
    np.testing.assert_array_equal(energy.srli(signed, *ARGS), energy.srli(shifted, *ARGS))
    clipped = energy.srli(np.clip(signed, 0.0, None), *ARGS)
    assert not np.allclose(energy.srli(signed, *ARGS), clipped, equal_nan=True)


def test_srli_high_residual_load_scores_high(signed: np.ndarray) -> None:
    series = signed.copy()
    series[-1] = signed.max() + 1.0
    assert energy.srli(series, 1, 1990, 1990, 2019, MONTHLY)[-1] > 1.28


@pytest.mark.parametrize("distribution", [d for d in indices.Distribution if d is not indices.Distribution.empirical])
def test_srli_rejects_parametric_distributions(signed: np.ndarray, distribution: indices.Distribution) -> None:
    with pytest.raises(InvalidArgumentError) as error:
        energy.srli(signed, *ARGS, distribution=distribution)
    assert error.value.argument_name == "distribution"
    assert "empirical" in str(error.value)


def test_srli_forwards_output_scale(signed: np.ndarray) -> None:
    probability = energy.srli(signed, *ARGS, output_scale="probability")
    finite = probability[np.isfinite(probability)]
    assert finite.size > 0
    assert ((finite > 0) & (finite < 1)).all()


def test_srli_aggregation_changes_value_not_rank(signed: np.ndarray) -> None:
    summed = energy.srli(signed, *ARGS, aggregation="sum")
    averaged = energy.srli(signed, *ARGS, aggregation="mean")
    np.testing.assert_allclose(summed, averaged, rtol=0, atol=1e-12)


def test_residual_load_subtracts_capacity_weighted_production() -> None:
    demand = np.array([100.0, 80.0, 60.0])
    wind = np.array([0.5, 1.0, 0.0])
    solar = np.array([0.2, 0.0, 0.5])
    result = energy.residual_load(demand, [wind, solar], [100.0, 50.0])
    np.testing.assert_allclose(result, [100 - 50 - 10, 80 - 100 - 0, 60 - 0 - 25], rtol=0, atol=1e-12)


def test_residual_load_is_signed_and_takes_time_varying_capacity() -> None:
    demand = np.array([10.0, 10.0])
    result = energy.residual_load(demand, [np.array([1.0, 1.0])], [np.array([5.0, 20.0])])
    np.testing.assert_array_equal(result, [5.0, -10.0])


def test_residual_load_propagates_missing_values_and_leaves_inputs_alone() -> None:
    demand = np.ma.array([10.0, 10.0, 10.0], mask=[False, True, False])
    factor = np.array([0.5, 0.5, np.nan])
    original = (demand.copy(), factor.copy())
    result = energy.residual_load(demand, [factor], [4.0])
    np.testing.assert_array_equal(result, [8.0, np.nan, np.nan])
    assert result.dtype == np.float64
    np.testing.assert_array_equal(demand.mask, original[0].mask)
    np.testing.assert_array_equal(factor, original[1])


def test_residual_load_feeds_srli(signed: np.ndarray) -> None:
    demand = 500.0 + signed
    factor = np.full(signed.shape, 0.25)
    residual = energy.residual_load(demand, [factor], [2000.0])
    np.testing.assert_allclose(residual, signed, rtol=0, atol=1e-9)
    assert np.isfinite(energy.srli(residual, *ARGS)[12:]).all()


@pytest.mark.parametrize(
    ("factors", "capacities", "name"),
    [
        ([np.array([1.01, 0.5])], [1.0], "capacity_factors"),
        ([np.array([-0.01, 0.5])], [1.0], "capacity_factors"),
        ([np.array([0.5, 0.5])], [-1.0], "installed_capacities"),
        ([np.array([0.5, 0.5])], [np.nan], "installed_capacities"),
        ([np.array([0.5, 0.5])], [np.inf], "installed_capacities"),
        ([np.array([0.5, np.inf])], [1.0], "capacity_factors"),
        ([np.array([0.5, 0.5])], [1.0, 2.0], "installed_capacities"),
        ([], [], "capacity_factors"),
        ([np.array([0.5, 0.5, 0.5])], [1.0], "capacity_factors"),
        ([np.ones((2, 2))], [1.0], "capacity_factors"),
    ],
)
def test_residual_load_rejects_unusable_assumptions(factors: list[Any], capacities: list[Any], name: str) -> None:
    with pytest.raises(InvalidArgumentError) as error:
        energy.residual_load(np.array([1.0, 1.0]), factors, capacities)
    assert error.value.argument_name == name


@pytest.mark.parametrize("argument", ["demand", "capacity_factors", "installed_capacities"])
def test_residual_load_rejects_non_numeric_input(argument: str) -> None:
    arguments: dict[str, Any] = {
        "demand": np.array([1.0, 1.0]),
        "capacity_factors": [np.array([0.5, 0.5])],
        "installed_capacities": [1.0],
    }
    arguments[argument] = ["a", "b"] if argument == "demand" else [np.array(["a", "b"])]
    with pytest.raises(InputTypeError):
        energy.residual_load(**arguments)
