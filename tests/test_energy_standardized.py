"""SREPI and the standardized demand index as wrappers over the standardize seam (#1171).

The SEI oracle for both lives in test_energy_sei_reference.py; these tests pin what
the wrappers add: defaults, the distribution restriction, and the threshold constants.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from climate_indices import compute, energy, indices
from climate_indices.exceptions import InvalidArgumentError

MONTHLY = compute.Periodicity.monthly
ARGS = (3, 1985, 1990, 2019, MONTHLY)  # the data start differs from the Calibration Period start


@pytest.fixture
def series() -> np.ndarray:
    return np.random.default_rng(0).uniform(0.05, 0.95, 35 * 12)


def _seam(
    values: np.ndarray,
    aggregation: compute.Aggregation,
    distribution: indices.Distribution = indices.Distribution.empirical,
) -> np.ndarray:
    return indices.standardized_index(
        values, 3, distribution, 1985, 1990, 2019, MONTHLY, aggregation=aggregation, clip_negatives=False
    )


def test_srepi_defaults_to_empirical_mean(series: np.ndarray) -> None:
    np.testing.assert_array_equal(energy.srepi(series, *ARGS), _seam(series, "mean"))


def test_demand_index_defaults_to_empirical(series: np.ndarray) -> None:
    np.testing.assert_array_equal(energy.standardized_demand_index(series, *ARGS), _seam(series, "sum"))


def test_beta_capacity_factor_matches_the_seam(series: np.ndarray) -> None:
    actual = energy.srepi(series, *ARGS, distribution=indices.Distribution.beta)
    expected = _seam(series, "mean", indices.Distribution.beta)
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual[2:]).all()


def test_demand_index_sums_windows_by_default(series: np.ndarray) -> None:
    """Rank scoring cannot tell sum from mean, but a beta fit rejects a sum above 1."""
    with pytest.raises(InvalidArgumentError):
        energy.standardized_demand_index(series, *ARGS, distribution=indices.Distribution.beta)


def test_output_scale_reaches_the_seam(series: np.ndarray) -> None:
    probability = energy.srepi(series, *ARGS, output_scale="probability")
    assert np.nanmin(probability) >= 0.0
    assert np.nanmax(probability) <= 1.0


def test_beta_rejects_accumulated_values_outside_unit_range(series: np.ndarray) -> None:
    with pytest.raises(InvalidArgumentError):
        energy.srepi(series, *ARGS, distribution=indices.Distribution.beta, aggregation="sum")


@pytest.mark.parametrize("function", [energy.srepi, energy.standardized_demand_index])
@pytest.mark.parametrize("distribution", [indices.Distribution.gamma, indices.Distribution.pearson])
def test_parametric_families_other_than_beta_are_rejected(
    function: Callable[..., np.ndarray], distribution: indices.Distribution, series: np.ndarray
) -> None:
    with pytest.raises(InvalidArgumentError, match="empirical and beta"):
        function(series, *ARGS, distribution=distribution)


def test_negative_input_is_ranked_not_clipped() -> None:
    values = np.random.default_rng(1).normal(size=35 * 12)
    np.testing.assert_array_equal(
        energy.standardized_demand_index(values, 1, 1985, 1990, 2019, MONTHLY),
        indices.standardized_index(
            values, 1, indices.Distribution.empirical, 1985, 1990, 2019, MONTHLY, clip_negatives=False
        ),
    )


def test_drought_thresholds_are_the_paper_percentiles() -> None:
    assert dict(energy.SREPI_DROUGHT_THRESHOLDS) == {"moderate": -1.28, "severe": -1.64, "extreme": -1.96}
    with pytest.raises(TypeError):
        energy.SREPI_DROUGHT_THRESHOLDS["moderate"] = 0.0  # type: ignore[index]
