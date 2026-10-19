"""Hand-calculated period summaries for generic run-theory events."""

import numpy as np
import pytest

from climate_indices import runs
from climate_indices.exceptions import DataShapeError, InputTypeError, InvalidArgumentError


def test_period_summaries_assign_complete_events_to_onset_period() -> None:
    values = [-2, 0, -3, -4, 0, -2, np.nan, 0, -3, 0, np.nan, np.nan]
    periods = [2020, 2020, 2020, 2021, 2021, 2021, 2021, 2022, 2022, 2022, 2023, 2023]

    result = runs.summarize_runs(values, periods=periods, max_gap=1)

    np.testing.assert_array_equal(result.events.start_index, [0, 8])
    np.testing.assert_array_equal(result.events.end_index, [5, 8])
    np.testing.assert_array_equal(result.period, [2020, 2021, 2022, 2023])
    np.testing.assert_array_equal(result.count, [1, 0, 1, 0])
    np.testing.assert_array_equal(result.total_duration, [6, 0, 1, 0])
    np.testing.assert_array_equal(result.valid_count, [3, 3, 3, 0])
    # These integer deviations are exact in float64.
    np.testing.assert_allclose(result.max_magnitude, [7, 0, 2, 0], rtol=0, atol=0)
    np.testing.assert_allclose(result.frequency, [1 / 3, 0, 1 / 3, np.nan], rtol=0, atol=0)


@pytest.mark.parametrize("values", [[], [np.nan], np.ma.masked_array([0, 0], mask=True)])
def test_no_observations_have_zero_counts_and_undefined_frequency(values: np.ndarray) -> None:
    result = runs.summarize_runs(values)
    assert len(result.events) == 0
    np.testing.assert_array_equal(result.period, [0])
    np.testing.assert_array_equal(result.count, [0])
    np.testing.assert_array_equal(result.total_duration, [0])
    np.testing.assert_array_equal(result.max_magnitude, [0.0])
    np.testing.assert_array_equal(result.valid_count, [0])
    assert np.isnan(result.frequency[0])


@pytest.mark.parametrize("min_duration, count, duration, maximum", [(1, 2, 2, 3), (2, 0, 0, 0)])
def test_summaries_use_retained_events_and_all_observed_steps(
    min_duration: int, count: int, duration: int, maximum: int
) -> None:
    result = runs.summarize_runs([-2, 0, -4, np.nan], min_duration=min_duration)
    np.testing.assert_array_equal(result.count, [count])
    np.testing.assert_array_equal(result.total_duration, [duration])
    np.testing.assert_array_equal(result.max_magnitude, [maximum])
    np.testing.assert_array_equal(result.valid_count, [3])
    np.testing.assert_array_equal(result.frequency, [count / 3])
    for array in (result.count, result.total_duration, result.valid_count):
        assert array.dtype == np.int64
    assert result.max_magnitude.dtype == np.float64
    assert result.frequency.dtype == np.float64


@pytest.mark.parametrize(
    "periods",
    [
        np.array([2, 2, 1]),
        np.array(["b", "b", "a"]),
        np.array([b"b", b"b", b"a"]),
        np.array(["2021-01", "2021-01", "2020-01"], dtype="datetime64[M]"),
        np.array([2, 2, 1], dtype="timedelta64[D]"),
        np.ma.array([2, 2, 1], mask=False),
    ],
)
def test_period_labels_keep_dtype_and_sort_unique_groups(periods: np.ndarray) -> None:
    result = runs.summarize_runs([-2, 0, -3], periods=periods)
    np.testing.assert_array_equal(result.period, np.sort(np.unique(periods)))
    assert result.period.dtype == periods.dtype
    np.testing.assert_array_equal(result.count, [1, 1])
    np.testing.assert_array_equal(result.max_magnitude, [2.0, 1.0])
    np.testing.assert_array_equal(result.frequency, [1.0, 0.5])


def test_repeated_period_labels_form_one_group_without_splitting_runs() -> None:
    result = runs.summarize_runs([-2, -3, -4], periods=[2, 1, 2])
    np.testing.assert_array_equal(result.events.duration, [3])
    np.testing.assert_array_equal(result.count, [0, 1])
    np.testing.assert_array_equal(result.total_duration, [0, 3])
    np.testing.assert_array_equal(result.valid_count, [1, 2])


def test_explicit_empty_periods_produce_empty_summary_arrays() -> None:
    result = runs.summarize_runs([], periods=[])
    for array in (result.period, result.count, result.total_duration, result.max_magnitude, result.frequency):
        assert array.size == 0


def test_masked_values_are_not_frequency_observations() -> None:
    values = np.ma.array([-2, -100, 0, np.nan], mask=[False, True, False, False])
    result = runs.summarize_runs(values, max_gap=4)
    np.testing.assert_array_equal(result.valid_count, [2])
    np.testing.assert_array_equal(result.count, [1])
    np.testing.assert_array_equal(result.frequency, [0.5])


@pytest.mark.parametrize("values", [1, [[-2, -3]]])
def test_summary_rejects_non_series_inputs(values: np.ndarray) -> None:
    with pytest.raises(DataShapeError, match="one-dimensional"):
        runs.summarize_runs(values)


@pytest.mark.parametrize("periods", [2020, [2020], [[2020, 2021]]])
def test_period_shape_must_match_values(periods: np.ndarray) -> None:
    with pytest.raises(DataShapeError, match="periods"):
        runs.summarize_runs([-2, -3], periods=periods)


@pytest.mark.parametrize(
    "periods",
    [
        [np.nan, 1],
        [np.inf, 1],
        np.array(["NaT", "2000"], dtype="datetime64[Y]"),
        np.array(["NaT", "1"], dtype="timedelta64[D]"),
        np.ma.array([0, 1], mask=[False, True]),
    ],
)
def test_period_labels_cannot_be_missing_or_infinite(periods: np.ndarray) -> None:
    with pytest.raises(InvalidArgumentError, match="periods"):
        runs.summarize_runs([-2, -3], periods=periods)


def test_mixed_string_and_nan_labels_are_an_ordinary_nan_label() -> None:
    # NumPy turns the NaN into the text "nan" before summarize_runs sees it.
    result = runs.summarize_runs([-2, -3], periods=np.array(["2020", np.nan]))

    assert result.period.tolist() == ["2020", "nan"]


@pytest.mark.parametrize("periods", [[1j, 2j], np.array([None, 1], dtype=object)])
def test_period_labels_reject_unsupported_dtypes(periods: np.ndarray) -> None:
    with pytest.raises(InputTypeError, match="periods"):
        runs.summarize_runs([-2, -3], periods=periods)
