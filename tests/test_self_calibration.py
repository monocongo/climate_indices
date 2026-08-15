"""Unit tests for the scPDSI self-calibration statistics module.

Expected values are hand-derived from the Wells, Goddard & Wilhite (2004)
reference algorithm as implemented in the reference C++ (used for algorithm
reference only). Each test's comment shows the derivation.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from climate_indices import self_calibration
from climate_indices.exceptions import InsufficientDataError, InvalidArgumentError


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

    def test_does_not_mutate_the_callers_array(self):
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
        values = np.arange(1.0, 11.0)

        with pytest.raises(InvalidArgumentError):
            self_calibration.nan_safe_percentile(values, fraction)


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
        z = np.arange(10.0)

        with pytest.raises(InvalidArgumentError):
            self_calibration.extreme_z_sum(z, 3, 0)

    def test_rejects_a_nonpositive_window_length(self):
        z = np.arange(10.0)

        with pytest.raises(InvalidArgumentError):
            self_calibration.extreme_z_sum(z, 0, self_calibration.WET_SIGN)

    def test_rejects_a_fractional_window_length(self):
        z = np.arange(10.0)

        with pytest.raises(InvalidArgumentError) as exc_info:
            self_calibration.extreme_z_sum(z, 2.5, self_calibration.WET_SIGN)

        assert exc_info.value.argument_name == "window_length"
        assert exc_info.value.argument_value == "2.5"
        assert exc_info.value.valid_values == "a positive integer"

    def test_accepts_an_integer_valued_float_window_length(self):
        # window_length is deliberately checked with float(x).is_integer()
        # rather than isinstance(x, int), so an integer-valued float like 2.0
        # must still be accepted, not just rejected when fractional
        z = np.array([1.0, 2.0])

        result = self_calibration.extreme_z_sum(z, 2.0, self_calibration.WET_SIGN)

        assert result == pytest.approx(3.0)

    def test_integer_valued_float_window_length_is_normalized_in_error_metadata(self):
        z = np.array([1.0, 2.0])

        with pytest.raises(InsufficientDataError) as exc_info:
            self_calibration.extreme_z_sum(z, 5.0, self_calibration.WET_SIGN)

        assert exc_info.value.required_count == 5
        assert isinstance(exc_info.value.required_count, int)

    @pytest.mark.parametrize("sign", [self_calibration.WET_SIGN, self_calibration.DRY_SIGN])
    def test_raises_insufficient_data_when_series_is_shorter_than_one_window(self, sign):
        # only 2 non-missing values are available, so a 5-period window can
        # never fully form on either side -- there is no sentinel value that
        # safely stands in for "no complete window was ever seen", so this
        # must raise rather than return a floor/NaN placeholder
        z = np.array([1.0, 2.0])

        with pytest.raises(InsufficientDataError) as exc_info:
            self_calibration.extreme_z_sum(z, 5, sign)

        assert exc_info.value.non_zero_count == 2
        assert exc_info.value.required_count == 5

    @pytest.mark.parametrize("sign", [self_calibration.WET_SIGN, self_calibration.DRY_SIGN])
    def test_raises_insufficient_data_when_missing_values_prevent_a_full_window(self, sign):
        # 3 non-missing values total, none of which can ever fill a 5-period
        # window even though the raw series has 5 entries
        z = np.array([1.0, np.nan, 2.0, np.nan, 3.0])

        with pytest.raises(InsufficientDataError) as exc_info:
            self_calibration.extreme_z_sum(z, 5, sign)

        assert exc_info.value.non_zero_count == 3
        assert exc_info.value.required_count == 5

    def test_wet_side_returns_the_largest_survivor_rather_than_the_threshold(self):
        z = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 11.0])

        # window_length=1 makes each rolling sum equal its own Z value, so the
        # filter's behaviour can be pinned directly. 10 sums, so the threshold
        # is the kth smallest with k = int(0.98 * 10) = 9, i.e. 10.0. Everything
        # clears the tolerance (11 / 10 = 1.1 < 1.25), so the answer is the
        # largest survivor, 11.0 -- not the threshold itself.
        # This also pins the percentile: an 80th-percentile threshold would be
        # k = int(0.8 * 10) = 8 -> 8.0, admitting only values below 10.0 and
        # yielding 8.0 instead.
        result = self_calibration.extreme_z_sum(z, 1, self_calibration.WET_SIGN)

        assert result == pytest.approx(11.0)

    @pytest.mark.parametrize(
        ("tail", "expected"),
        [(12.4, 12.4), (12.5, 10.0), (12.6, 10.0)],
    )
    def test_wet_side_tolerance_boundary_is_exclusive(self, tail, expected):
        z = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, tail])

        # the threshold is 10.0 in all three cases, so the tail's ratio to it is
        # 1.24, 1.25 and 1.26 respectively. The comparison is strictly less-than
        # against a tolerance of 1.25, so only 1.24 survives; at exactly 1.25 the
        # tail is discarded and the largest survivor falls back to 10.0.
        result = self_calibration.extreme_z_sum(z, 1, self_calibration.WET_SIGN)

        assert result == pytest.approx(expected)

    def test_wet_side_counts_the_rolling_sum_from_the_initial_window_fill(self):
        z = np.array([11.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])

        # same values as the test above, with the largest moved to the front so
        # it is the sum produced by the initial window fill rather than by the
        # slide. Failing to record that first sum would drop 11.0 from the
        # candidates and return 10.0.
        result = self_calibration.extreme_z_sum(z, 1, self_calibration.WET_SIGN)

        assert result == pytest.approx(11.0)

    def test_wet_side_skips_filtering_when_no_percentile_is_available(self):
        z = np.array([7.0])

        # a single rolling sum gives k = int(0.98 * 1) = 0, below the 1-indexed
        # range, so no threshold exists. The reference divides by its
        # missing-value sentinel here, which admits every candidate -- so the
        # sum survives unfiltered rather than being rejected to the 0.0 floor.
        result = self_calibration.extreme_z_sum(z, 1, self_calibration.WET_SIGN)

        assert result == pytest.approx(7.0)

    def test_wet_side_rejects_everything_when_the_threshold_is_zero(self):
        z = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0])

        # k = int(0.98 * 10) = 9 -> the 9th smallest is 0.0. The reference's
        # division by zero yields infinity, which fails the ratio test for every
        # candidate, so the result falls back to the 0.0 floor -- and, crucially,
        # no ZeroDivisionError escapes.
        result = self_calibration.extreme_z_sum(z, 1, self_calibration.WET_SIGN)

        assert result == pytest.approx(0.0)

    def test_wet_side_applies_the_ratio_test_to_a_nonzero_threshold(self):
        threshold = 1e-10
        z = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, threshold, 1.0])

        # k = int(0.98 * 10) = 9 -> the tiny positive value is the threshold.
        # It survives with a ratio of 1.0 while the 1.0 outlier is rejected. A
        # zero check with a nonzero absolute tolerance would incorrectly reject
        # the threshold too and return the 0.0 floor.
        result = self_calibration.extreme_z_sum(z, 1, self_calibration.WET_SIGN)

        assert result == pytest.approx(threshold)


class TestLeastSquaresFit:
    def test_recovers_an_exact_dry_line_without_trimming(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([-11.0, -12.0, -13.0, -14.0, -15.0])

        slope, intercept = self_calibration.least_squares_fit(x, y, self_calibration.DRY_SIGN)

        # correlation is -1, so sign * correlation = 1 >= 0.85 and nothing is
        # trimmed. SSXY / SSX = -10 / 10 = -1. Every residual is -10, and since
        # intercept = y[i] - slope * x[i] *is* the residual, anchoring on any
        # tied point gives the same intercept = y[0] - slope * x[0] = -11 + 1 = -10.
        assert slope == pytest.approx(-1.0)
        assert intercept == pytest.approx(-10.0)

    def test_recovers_an_exact_wet_line_without_trimming(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([11.0, 12.0, 13.0, 14.0, 15.0])

        slope, intercept = self_calibration.least_squares_fit(x, y, self_calibration.WET_SIGN)

        assert slope == pytest.approx(1.0)
        assert intercept == pytest.approx(10.0)

    def test_intercept_is_anchored_on_the_most_extreme_residual(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([11.0, 12.0, 15.0, 14.0, 15.0])

        slope, intercept = self_calibration.least_squares_fit(x, y, self_calibration.WET_SIGN)

        # SSX = 10, SSXY = 10 -> slope = 1.0. Correlation is 0.8704 >= 0.85, so
        # no trimming. Residuals y - slope*x are 10, 10, 12, 10, 10; index 2 is
        # the most extreme, so intercept = y[2] - slope*x[2] = 15 - 3 = 12.
        # A textbook OLS intercept would be ybar - slope*xbar = 13.4 - 3 = 10.4.
        assert slope == pytest.approx(1.0)
        assert intercept == pytest.approx(12.0)

    def test_trims_trailing_points_until_the_correlation_clears_the_tolerance(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([-11.0, -12.0, -13.0, -14.0, -25.0])

        slope, intercept = self_calibration.least_squares_fit(x, y, self_calibration.DRY_SIGN)

        # Over all five points the correlation is -0.8321, so sign*correlation
        # = 0.8321 < 0.85 and the last point is dropped. The remaining four are
        # perfectly collinear (correlation -1), giving slope = -5/5 = -1 -- not
        # the -3 the untrimmed five-point fit would have produced. Residuals are
        # all -10, and since intercept = y[i] - slope*x[i] *is* the residual,
        # anchoring on any tied point gives the same intercept =
        # y[0] - slope*x[0] = -11 + 1 = -10.
        assert slope == pytest.approx(-1.0)
        assert intercept == pytest.approx(-10.0)

    def test_trimming_stops_at_four_retained_points(self):
        x = np.arange(1.0, 11.0)
        y = np.array([-11.0, -12.0, -13.0, -14.0, 50.0, -60.0, 70.0, -80.0, 90.0, -100.0])

        slope, intercept = self_calibration.least_squares_fit(x, y, self_calibration.DRY_SIGN)

        # The trailing six points are pure noise, so the correlation never
        # clears the tolerance and trimming runs to its floor. The four points
        # that survive are the collinear leading ones, giving exactly the fit
        # they define on their own.
        assert slope == pytest.approx(-1.0)
        assert intercept == pytest.approx(-10.0)

    def test_rejects_an_invalid_sign(self):
        x = np.arange(5.0)
        y = np.arange(5.0)

        with pytest.raises(InvalidArgumentError):
            self_calibration.least_squares_fit(x, y, 0)

    def test_rejects_mismatched_input_lengths(self):
        x = np.arange(5.0)
        y = np.arange(4.0)

        with pytest.raises(InvalidArgumentError):
            self_calibration.least_squares_fit(x, y, self_calibration.WET_SIGN)

    def test_rejects_fewer_points_than_the_trimming_floor(self):
        x = np.arange(3.0)
        y = np.arange(3.0)

        with pytest.raises(InvalidArgumentError):
            self_calibration.least_squares_fit(x, y, self_calibration.WET_SIGN)

    def test_intercept_keeps_the_initial_anchor_for_a_line_through_the_origin(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.0, 6.0, 8.0])

        slope, intercept = self_calibration.least_squares_fit(x, y, self_calibration.WET_SIGN)

        # Every residual is exactly zero, and the anchor search only moves on a
        # residual strictly more extreme than its initial 0.0 -- so the anchor
        # stays at the reference's initial (x[0], 0.0), giving
        # intercept = 0.0 - slope * x[0] = -2.0 rather than the 0.0 an
        # intuitive reading would expect. This is faithful to the reference and
        # is pinned here precisely because it looks like a bug.
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(-2.0)


class TestDurationFactors:
    def test_window_lengths_match_the_reference_monthly_set(self):
        assert self_calibration.DURATION_FACTOR_WINDOW_LENGTHS == (
            3,
            6,
            9,
            12,
            18,
            24,
            30,
            36,
            42,
            48,
        )

    @pytest.mark.parametrize("sign", [1, -1])
    def test_normalizes_the_raw_fit_by_sign_times_four(self, sign):
        rng = np.random.default_rng(7)
        z = rng.normal(0.0, 1.5, size=600)

        window_lengths = np.array(self_calibration.DURATION_FACTOR_WINDOW_LENGTHS, dtype=float)
        extreme_sums = np.array(
            [
                self_calibration.extreme_z_sum(z, length, sign)
                for length in self_calibration.DURATION_FACTOR_WINDOW_LENGTHS
            ]
        )
        raw_slope, raw_intercept = self_calibration.least_squares_fit(window_lengths, extreme_sums, sign)

        slope, intercept = self_calibration.duration_factors(z, sign)

        assert slope == pytest.approx(raw_slope / (sign * 4.0))
        assert intercept == pytest.approx(raw_intercept / (sign * 4.0))

        # guard against the assertions above passing vacuously on a zero fit
        assert raw_slope != pytest.approx(0.0)
        assert raw_intercept != pytest.approx(0.0)
        assert slope != pytest.approx(raw_slope)

    def test_dry_factors_are_plausible_for_a_realistic_series(self):
        rng = np.random.default_rng(7)
        z = rng.normal(0.0, 1.5, size=600)

        slope, intercept = self_calibration.duration_factors(z, self_calibration.DRY_SIGN)

        # the PDSI recursion divides by (m + b) and weights by b / (m + b), so a
        # usable dry fit needs a positive sum and a weighting fraction below 1 --
        # Palmer's fixed national values are m = 0.309, b = 2.691 for comparison
        assert slope + intercept > 0.0
        assert 0.0 < intercept / (slope + intercept) < 1.0

    def test_rejects_an_invalid_sign(self):
        z = np.arange(600.0)

        with pytest.raises(InvalidArgumentError):
            self_calibration.duration_factors(z, 0)


class TestModuleBoundaries:
    def test_does_not_import_palmer(self):
        """The dependency edge runs palmer -> self_calibration only (issue #718)."""
        source = Path(self_calibration.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported.add(module)
                imported.update(f"{module}.{alias.name}" for alias in node.names)

        offenders = sorted(name for name in imported if "palmer" in name)
        assert not offenders, f"self_calibration must not import palmer: {offenders}"

    def test_public_surface_is_exported(self):
        assert set(self_calibration.__all__) == {
            "DRY_SIGN",
            "DURATION_FACTOR_WINDOW_LENGTHS",
            "WET_SIGN",
            "duration_factors",
            "extreme_z_sum",
            "kth_smallest",
            "least_squares_fit",
            "nan_safe_percentile",
        }
        for name in self_calibration.__all__:
            assert hasattr(self_calibration, name), f"__all__ names a missing attribute: {name}"
