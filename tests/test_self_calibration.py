"""Unit tests for the scPDSI self-calibration statistics module.

Expected values are hand-derived from the Wells, Goddard & Wilhite (2004)
reference algorithm as implemented in the reference C++ (used for algorithm
reference only). Each test's comment shows the derivation.
"""

import numpy as np
import pytest

from climate_indices import self_calibration
from climate_indices.exceptions import InvalidArgumentError


class TestKthSmallest:
    def test_returns_kth_smallest_one_indexed(self):
        values = np.array([5.0, 1.0, 4.0, 2.0, 3.0])

        assert self_calibration.kth_smallest(values, 1) == pytest.approx(1.0)
        assert self_calibration.kth_smallest(values, 3) == pytest.approx(3.0)
        assert self_calibration.kth_smallest(values, 5) == pytest.approx(5.0)

    def test_returns_nan_for_out_of_range_k(self):
        # the reference returns its MISSING sentinel for k < 1 or k > size;
        # we return NaN
        values = np.array([5.0, 1.0, 4.0, 2.0, 3.0])

        assert np.isnan(self_calibration.kth_smallest(values, 0))
        assert np.isnan(self_calibration.kth_smallest(values, 6))

    def test_does_not_mutate_the_caller_s_array(self):
        values = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
        original = values.copy()

        self_calibration.kth_smallest(values, 2)

        np.testing.assert_array_equal(values, original)


class TestNanSafePercentile:
    def test_lower_and_upper_percentiles_over_fifty_values(self):
        values = np.arange(1.0, 51.0)

        # k = int(0.02 * 50) = 1 -> smallest value
        assert self_calibration.nan_safe_percentile(values, 0.02) == pytest.approx(1.0)
        # k = int(0.98 * 50) = 49 -> 49th smallest
        assert self_calibration.nan_safe_percentile(values, 0.98) == pytest.approx(49.0)

    def test_index_is_truncated_not_rounded(self):
        values = np.arange(1.0, 11.0)

        # k = int(0.25 * 10) = int(2.5) = 2, so the 2nd smallest -- not the 3rd,
        # and not numpy.percentile's interpolated 3.25
        assert self_calibration.nan_safe_percentile(values, 0.25) == pytest.approx(2.0)

    def test_missing_values_are_dropped_before_ranking(self):
        values = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0])

        # 5 values survive; k = int(0.6 * 5) = 3 -> 3rd smallest of [1, 2, 3, 4, 5]
        assert self_calibration.nan_safe_percentile(values, 0.6) == pytest.approx(3.0)

    def test_returns_nan_when_every_value_is_missing(self):
        assert np.isnan(self_calibration.nan_safe_percentile(np.array([np.nan, np.nan]), 0.5))

    def test_returns_nan_when_the_computed_index_truncates_to_zero(self):
        values = np.arange(1.0, 6.0)

        # k = int(0.02 * 5) = 0, below the 1-indexed range
        assert np.isnan(self_calibration.nan_safe_percentile(values, 0.02))

    @pytest.mark.parametrize("fraction", [-0.1, 1.5])
    def test_rejects_a_fraction_outside_the_unit_interval(self, fraction):
        with pytest.raises(InvalidArgumentError):
            self_calibration.nan_safe_percentile(np.arange(1.0, 11.0), fraction)


class TestExtremeZSum:
    def test_dry_side_returns_the_most_negative_rolling_sum(self):
        z = np.array([-1.0, -5.0, -2.0, 0.0, -3.0])

        # rolling 2-period sums: -6, -7, -2, -3 -> most negative is -7
        result = self_calibration.extreme_z_sum(z, 2, self_calibration.DRY_SIGN)

        assert result == pytest.approx(-7.0)

    def test_wet_side_discards_a_freak_anomaly(self):
        z = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 97.0])

        # rolling 2-period sums: 2, 3, 4, 5, 100.
        # threshold = 98th percentile = kth smallest with k = int(0.98 * 5) = 4,
        # i.e. 5.0. The 100 sum fails the filter (100 / 5 = 20.0 >= 1.25) and is
        # discarded; 5.0 passes (5 / 5 = 1.0) and is the largest survivor.
        result = self_calibration.extreme_z_sum(z, 2, self_calibration.WET_SIGN)

        assert result == pytest.approx(5.0)

    def test_dry_side_keeps_the_anomaly_the_wet_side_would_discard(self):
        # exact mirror image of the wet case above: rolling sums -2, -3, -4, -5, -100
        z = np.array([-1.0, -1.0, -2.0, -2.0, -3.0, -97.0])

        # the dry side is deliberately unfiltered in the reference algorithm, so
        # it returns the -100 outlier rather than the -5 the wet side would keep
        result = self_calibration.extreme_z_sum(z, 2, self_calibration.DRY_SIGN)

        assert result == pytest.approx(-100.0)

    def test_missing_periods_are_skipped_rather_than_treated_as_zero(self):
        with_missing = np.array([1.0, 1.0, np.nan, 2.0, 2.0, 3.0, 97.0])
        without_missing = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 97.0])

        result = self_calibration.extreme_z_sum(with_missing, 2, self_calibration.WET_SIGN)

        # a NaN mid-slide leaves the window and running sum untouched, so the
        # result is identical to the series with the NaN simply removed
        assert result == pytest.approx(self_calibration.extreme_z_sum(without_missing, 2, self_calibration.WET_SIGN))
        assert result == pytest.approx(5.0)

    def test_missing_periods_during_the_initial_fill_are_retried(self):
        z = np.array([1.0, np.nan, 1.0, 2.0, 2.0, 3.0, 97.0])

        # the NaN must not consume one of the window's two slots, so the first
        # window is [1.0, 1.0] and the rolling sums are 2, 3, 4, 5, 100 again
        result = self_calibration.extreme_z_sum(z, 2, self_calibration.WET_SIGN)

        assert result == pytest.approx(5.0)

    def test_rejects_an_invalid_sign(self):
        with pytest.raises(InvalidArgumentError):
            self_calibration.extreme_z_sum(np.arange(10.0), 3, 0)

    def test_rejects_a_nonpositive_window_length(self):
        with pytest.raises(InvalidArgumentError):
            self_calibration.extreme_z_sum(np.arange(10.0), 0, self_calibration.WET_SIGN)
