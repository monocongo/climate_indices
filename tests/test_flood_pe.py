"""Fixed-window effective precipitation contract."""

import numpy as np
import pytest

from climate_indices import flood
from climate_indices.exceptions import InvalidArgumentError


def test_pe_matches_double_sum_for_series_and_spatial_block() -> None:
    rng = np.random.default_rng(1105)
    values = rng.uniform(0, 30, (17, 3, 2))
    duration = 5
    actual = flood.effective_precipitation(values, duration=duration)
    expected = np.full(values.shape, np.nan)
    for day in range(duration - 1, values.shape[0]):
        for length in range(1, duration + 1):
            if length == 1:
                expected[day] = 0
            expected[day] += np.sum(values[day - length + 1 : day + 1], axis=0) / length
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-13, equal_nan=True)
    np.testing.assert_allclose(
        flood.effective_precipitation(values[:, 0, 0], duration=duration),
        actual[:, 0, 0],
        rtol=1e-14,
        atol=1e-13,
    )


def test_pe_missing_window_recovers_per_cell_and_year_layout() -> None:
    values = np.ones((2, 4))
    values[0, 3] = np.nan
    result = flood.effective_precipitation(values, duration=3)
    assert np.isnan(result.reshape(-1)[[0, 1, 3, 4, 5]]).all()
    np.testing.assert_allclose(result.reshape(-1)[[2, 6, 7]], [3, 3, 3])
    block = np.ones((6, 2, 1))
    block[2, 0, 0] = np.nan
    actual = flood.effective_precipitation(block, duration=3)
    np.testing.assert_allclose(actual[:, 1, 0], [np.nan, np.nan, 3, 3, 3, 3], equal_nan=True)
    assert np.isnan(actual[2:5, 0, 0]).all()
    assert actual[5, 0, 0] == pytest.approx(3)


def test_pe_short_window_and_ambiguous_shape() -> None:
    assert np.isnan(flood.effective_precipitation([1, 2], duration=3)).all()
    block = np.ones((4, 12, 2))
    with pytest.raises(ValueError, match="spatial_time_major=True"):
        flood.effective_precipitation(block, duration=2)
    assert flood.effective_precipitation(block, duration=2, spatial_time_major=True).shape == block.shape


@pytest.mark.parametrize("duration", [0, -1, 1.5, True])
def test_pe_rejects_invalid_duration(duration: object) -> None:
    with pytest.raises(InvalidArgumentError, match="duration"):
        flood.effective_precipitation([1, 2], duration=duration)  # type: ignore[arg-type]


@pytest.mark.parametrize("values", [[1, -1], [1, np.inf]])
def test_pe_rejects_invalid_precipitation(values: list[float]) -> None:
    with pytest.raises(InvalidArgumentError, match="precipitation"):
        flood.effective_precipitation(values, duration=2)
