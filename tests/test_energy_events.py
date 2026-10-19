"""Energy event statistics use the generic run-theory contract."""

from pathlib import Path

import numpy as np
import pytest

from climate_indices import compute, energy, indices
from climate_indices.runs import Direction, RunStatistics


def test_energy_events_report_runs_and_period_statistics() -> None:
    result = energy.energy_drought_events(
        [2, 0, 3, 0, 0, 4],
        threshold=1.0,
        direction="above",
        min_duration=2,
        max_gap=1,
        periods=[2020, 2020, 2021, 2021, 2021, 2021],
    )

    assert isinstance(result, RunStatistics)
    np.testing.assert_array_equal(result.events.start_index, [0])
    np.testing.assert_array_equal(result.events.end_index, [2])
    np.testing.assert_array_equal(result.count, [1, 0])
    np.testing.assert_array_equal(result.total_duration, [3, 0])
    # Integer deviations and these binary fractions are exactly representable.
    np.testing.assert_allclose(result.max_magnitude, [3.0, 0.0], rtol=0, atol=0)
    np.testing.assert_allclose(result.frequency, [0.5, 0.0], rtol=0, atol=0)


@pytest.mark.parametrize(
    "source, direction",
    [("spi", "below"), ("capacity_mean_1_normal", "below"), ("residual_sum_1_normal", "above")],
)
def test_events_accept_spi_and_saved_srepi_srli_index_series(source: str, direction: Direction) -> None:
    if source == "spi":
        rain = np.random.default_rng(1173).gamma(2, 20, 40 * 12)
        index = indices.spi(rain, 1, indices.Distribution.gamma, 1980, 1980, 2019, compute.Periodicity.monthly)
    else:
        # Saved SEI outputs cover energy indices before #1171/#1172 expose their wrappers.
        with np.load(Path(__file__).parent / "fixture" / "energy_sei" / "reference.npz") as saved:
            index = saved[source]
    threshold = -1.0 if direction == "below" else 1.0
    result = energy.energy_drought_events(index, threshold=threshold, direction=direction)

    expected = index < threshold if direction == "below" else index > threshold
    observed = np.zeros(index.size, dtype=bool)
    for start, end in zip(result.events.start_index, result.events.end_index, strict=True):
        observed[start : end + 1] = True
    np.testing.assert_array_equal(observed, expected)
    # Count starts directly, independently of the vectorized run detector.
    count = sum(active and (i == 0 or not expected[i - 1]) for i, active in enumerate(expected))
    assert len(result.events) == count > 0
    np.testing.assert_array_equal(result.count, [count])
    np.testing.assert_array_equal(result.valid_count, [np.count_nonzero(~np.isnan(index))])
    np.testing.assert_allclose(result.frequency, [count / result.valid_count[0]], rtol=0, atol=0)
