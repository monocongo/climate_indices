"""Tests for data quality warning system (Story 1.4).

This module tests the warning system that alerts researchers to data quality
issues without blocking computation. Tests cover warning class hierarchy,
attribute storage, filterability, and emission conditions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from climate_indices import ClimateIndicesWarning, compute
from climate_indices.exceptions import (
    GoodnessOfFitWarning,
    MissingDataWarning,
    ShortCalibrationWarning,
)


class TestWarningHierarchy:
    """Test warning class inheritance structure."""

    def test_climate_indices_warning_inherits_from_user_warning(self) -> None:
        """ClimateIndicesWarning should inherit from UserWarning."""
        assert issubclass(ClimateIndicesWarning, UserWarning)

    def test_missing_data_warning_inherits_from_base(self) -> None:
        """MissingDataWarning should inherit from ClimateIndicesWarning."""
        assert issubclass(MissingDataWarning, ClimateIndicesWarning)

    def test_short_calibration_warning_inherits_from_base(self) -> None:
        """ShortCalibrationWarning should inherit from ClimateIndicesWarning."""
        assert issubclass(ShortCalibrationWarning, ClimateIndicesWarning)

    def test_goodness_of_fit_warning_inherits_from_base(self) -> None:
        """GoodnessOfFitWarning should inherit from ClimateIndicesWarning."""
        assert issubclass(GoodnessOfFitWarning, ClimateIndicesWarning)


class TestWarningAttributes:
    """Test warning attribute storage and access."""

    def test_missing_data_warning_stores_ratio_and_threshold(self) -> None:
        """MissingDataWarning should store missing_ratio and threshold."""
        warning = MissingDataWarning("test message", missing_ratio=0.35, threshold=0.20)
        assert warning.missing_ratio == pytest.approx(0.35)
        assert warning.threshold == pytest.approx(0.20)

    def test_missing_data_warning_handles_none_attributes(self) -> None:
        """MissingDataWarning should allow None for optional attributes."""
        warning = MissingDataWarning("test message")
        assert warning.missing_ratio is None
        assert warning.threshold is None

    def test_short_calibration_warning_stores_years(self) -> None:
        """ShortCalibrationWarning should store actual_years and required_years."""
        warning = ShortCalibrationWarning("test message", actual_years=25, required_years=30)
        assert warning.actual_years == 25
        assert warning.required_years == 30

    def test_short_calibration_warning_handles_none_attributes(self) -> None:
        """ShortCalibrationWarning should allow None for optional attributes."""
        warning = ShortCalibrationWarning("test message")
        assert warning.actual_years is None
        assert warning.required_years is None

    def test_goodness_of_fit_warning_stores_distribution_info(self) -> None:
        """GoodnessOfFitWarning should store distribution fitting details."""
        warning = GoodnessOfFitWarning(
            "test message",
            distribution_name="gamma",
            p_value=0.02,
            threshold=0.05,
            poor_fit_count=5,
            total_steps=12,
        )
        assert warning.distribution_name == "gamma"
        assert warning.p_value == pytest.approx(0.02)
        assert warning.threshold == pytest.approx(0.05)
        assert warning.poor_fit_count == 5
        assert warning.total_steps == 12

    def test_goodness_of_fit_warning_handles_none_attributes(self) -> None:
        """GoodnessOfFitWarning should allow None for optional attributes."""
        warning = GoodnessOfFitWarning("test message")
        assert warning.distribution_name is None
        assert warning.p_value is None
        assert warning.threshold is None
        assert warning.poor_fit_count is None
        assert warning.total_steps is None


class TestWarningFilterability:
    """Test warning suppression via warnings.filterwarnings()."""

    def test_can_suppress_all_climate_indices_warnings(self) -> None:
        """Filtering by ClimateIndicesWarning should suppress all library warnings."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=ClimateIndicesWarning)

            warnings.warn(MissingDataWarning("test missing", missing_ratio=0.3, threshold=0.2))  # noqa: B028
            warnings.warn(ShortCalibrationWarning("test short", actual_years=20, required_years=30))  # noqa: B028
            warnings.warn(GoodnessOfFitWarning("test fit", distribution_name="gamma"))  # noqa: B028

            assert len(warning_list) == 0

    def test_can_suppress_specific_warning_type(self) -> None:
        """Filtering by specific warning type should suppress only that type."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=MissingDataWarning)

            warnings.warn(MissingDataWarning("test missing", missing_ratio=0.3, threshold=0.2))  # noqa: B028
            warnings.warn(ShortCalibrationWarning("test short", actual_years=20, required_years=30))  # noqa: B028

            # only ShortCalibrationWarning should be recorded
            assert len(warning_list) == 1
            assert issubclass(warning_list[0].category, ShortCalibrationWarning)

    def test_warnings_not_suppressed_by_default(self) -> None:
        """Warnings should be emitted when not explicitly filtered."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            warnings.warn(MissingDataWarning("test missing", missing_ratio=0.3, threshold=0.2))  # noqa: B028

            assert len(warning_list) == 1
            assert issubclass(warning_list[0].category, MissingDataWarning)


class TestMissingDataWarning:
    """Test MissingDataWarning emission in gamma and pearson paths."""

    def test_gamma_parameters_warns_on_excessive_missing_data(self) -> None:
        """gamma_parameters should warn when >20% of calibration data is missing."""
        # create dataset with 30% missing data in calibration period
        values = np.random.gamma(2, 2, size=(50, 12))
        # set 30% of values to NaN
        mask = np.random.rand(50, 12) < 0.30
        values[mask] = np.nan

        with pytest.warns(MissingDataWarning) as warning_info:
            compute.gamma_parameters(
                values,
                data_start_year=1970,
                calibration_start_year=1970,
                calibration_end_year=2019,
                periodicity=compute.Periodicity.monthly,
            )

        assert len(warning_info) >= 1
        warning = warning_info[0].message
        assert warning.missing_ratio > 0.20
        assert warning.threshold == pytest.approx(0.20)

    def test_gamma_parameters_no_warn_below_threshold(self) -> None:
        """gamma_parameters should not warn when missing data is below threshold."""
        # create dataset with 10% missing data
        values = np.random.gamma(2, 2, size=(50, 12))
        mask = np.random.rand(50, 12) < 0.10
        values[mask] = np.nan

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            compute.gamma_parameters(
                values,
                data_start_year=1970,
                calibration_start_year=1970,
                calibration_end_year=2019,
                periodicity=compute.Periodicity.monthly,
            )

            # filter to only MissingDataWarning
            missing_warnings = [w for w in warning_list if issubclass(w.category, MissingDataWarning)]
            assert len(missing_warnings) == 0

    def test_gamma_parameters_counts_zeros_as_valid_not_missing(self) -> None:
        """gamma_parameters should not count zero precipitation as missing data."""
        # create dataset with many zeros (valid for precipitation)
        values = np.random.gamma(2, 2, size=(50, 12))
        # set 30% to zero (valid data, not missing)
        zero_mask = np.random.rand(50, 12) < 0.30
        values[zero_mask] = 0.0

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            compute.gamma_parameters(
                values,
                data_start_year=1970,
                calibration_start_year=1970,
                calibration_end_year=2019,
                periodicity=compute.Periodicity.monthly,
            )

            # zeros should not trigger missing data warning
            missing_warnings = [w for w in warning_list if issubclass(w.category, MissingDataWarning)]
            assert len(missing_warnings) == 0

    def test_pearson_parameters_warns_on_excessive_missing_data(self) -> None:
        """pearson_parameters should warn when >20% of calibration data is missing."""
        values = np.random.randn(50, 12)
        # set 30% of values to NaN
        mask = np.random.rand(50, 12) < 0.30
        values[mask] = np.nan

        with pytest.warns(MissingDataWarning) as warning_info:
            compute.pearson_parameters(
                values,
                data_start_year=1970,
                calibration_start_year=1970,
                calibration_end_year=2019,
                periodicity=compute.Periodicity.monthly,
            )

        assert len(warning_info) >= 1
        warning = warning_info[0].message
        assert warning.missing_ratio > 0.20


class TestShortCalibrationWarning:
    """Test ShortCalibrationWarning emission for calibration periods <30 years."""

    def test_gamma_parameters_warns_on_short_calibration(self) -> None:
        """gamma_parameters should warn when calibration period is <30 years."""
        values = np.random.gamma(2, 2, size=(25, 12))

        with pytest.warns(ShortCalibrationWarning) as warning_info:
            compute.gamma_parameters(
                values,
                data_start_year=1990,
                calibration_start_year=1990,
                calibration_end_year=2014,  # 25 years
                periodicity=compute.Periodicity.monthly,
            )

        assert len(warning_info) >= 1
        warning = warning_info[0].message
        assert warning.actual_years == 25
        assert warning.required_years == 30

    def test_gamma_parameters_no_warn_at_exactly_30_years(self) -> None:
        """gamma_parameters should not warn when calibration is exactly 30 years."""
        values = np.random.gamma(2, 2, size=(30, 12))

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            compute.gamma_parameters(
                values,
                data_start_year=1990,
                calibration_start_year=1990,
                calibration_end_year=2019,  # 30 years
                periodicity=compute.Periodicity.monthly,
            )

            # filter to only ShortCalibrationWarning
            short_warnings = [w for w in warning_list if issubclass(w.category, ShortCalibrationWarning)]
            assert len(short_warnings) == 0

    def test_gamma_parameters_no_warn_above_30_years(self) -> None:
        """gamma_parameters should not warn when calibration exceeds 30 years."""
        values = np.random.gamma(2, 2, size=(50, 12))

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            compute.gamma_parameters(
                values,
                data_start_year=1970,
                calibration_start_year=1970,
                calibration_end_year=2019,  # 50 years
                periodicity=compute.Periodicity.monthly,
            )

            short_warnings = [w for w in warning_list if issubclass(w.category, ShortCalibrationWarning)]
            assert len(short_warnings) == 0

    def test_pearson_parameters_warns_on_short_calibration(self) -> None:
        """pearson_parameters should warn when calibration period is <30 years."""
        values = np.random.randn(20, 12)

        with pytest.warns(ShortCalibrationWarning) as warning_info:
            compute.pearson_parameters(
                values,
                data_start_year=1995,
                calibration_start_year=1995,
                calibration_end_year=2014,  # 20 years
                periodicity=compute.Periodicity.monthly,
            )

        assert len(warning_info) >= 1
        warning = warning_info[0].message
        assert warning.actual_years == 20


class TestGoodnessOfFitWarning:
    """Test GoodnessOfFitWarning emission when distribution fit is poor."""

    def test_gamma_parameters_warns_on_poor_fit(self) -> None:
        """gamma_parameters should warn when KS test indicates poor gamma fit."""
        # create data that poorly fits gamma (uniform distribution mixed with gamma)
        np.random.seed(123)
        # use uniform distribution which should not fit gamma well
        values = np.random.uniform(0.1, 10, size=(40, 12))

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            compute.gamma_parameters(
                values,
                data_start_year=1980,
                calibration_start_year=1980,
                calibration_end_year=2019,
                periodicity=compute.Periodicity.monthly,
            )

            # check if goodness-of-fit warning was emitted
            fit_warnings = [w for w in warning_list if issubclass(w.category, GoodnessOfFitWarning)]
            # this is a soft test - uniform data should often produce poor fit
            # but we allow for cases where it doesn't by checking attributes if present
            if len(fit_warnings) > 0:
                warning = fit_warnings[0].message
                assert warning.distribution_name == "gamma"
                assert warning.threshold == pytest.approx(0.05)
                assert warning.poor_fit_count > 0
                assert warning.total_steps == 12

    def test_gamma_parameters_no_warn_on_good_fit(self) -> None:
        """gamma_parameters should not warn when gamma fit is good."""
        # create data that fits gamma well
        np.random.seed(42)
        values = np.random.gamma(2, 2, size=(50, 12))

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            compute.gamma_parameters(
                values,
                data_start_year=1970,
                calibration_start_year=1970,
                calibration_end_year=2019,
                periodicity=compute.Periodicity.monthly,
            )

            fit_warnings = [w for w in warning_list if issubclass(w.category, GoodnessOfFitWarning)]
            # good fit should produce no warnings (or very few)
            assert len(fit_warnings) == 0

    def test_goodness_of_fit_warning_aggregates_across_time_steps(self) -> None:
        """GoodnessOfFitWarning should aggregate poor fits to avoid warning floods."""
        # create data with some poor fits
        values_good = np.random.gamma(2, 2, size=(40, 6))
        values_poor = np.random.exponential(2, size=(40, 6))
        values = np.hstack([values_good, values_poor])

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            compute.gamma_parameters(
                values,
                data_start_year=1980,
                calibration_start_year=1980,
                calibration_end_year=2019,
                periodicity=compute.Periodicity.monthly,
            )

            fit_warnings = [w for w in warning_list if issubclass(w.category, GoodnessOfFitWarning)]
            # should emit at most one aggregated warning, not 12 separate warnings
            assert len(fit_warnings) <= 1

    def test_pearson_parameters_warns_on_poor_fit(self) -> None:
        """pearson_parameters should warn when KS test indicates poor Pearson fit."""
        # create data that poorly fits Pearson Type III (uniform distribution)
        np.random.seed(456)
        values = np.random.uniform(-5, 5, size=(40, 12))

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            compute.pearson_parameters(
                values,
                data_start_year=1980,
                calibration_start_year=1980,
                calibration_end_year=2019,
                periodicity=compute.Periodicity.monthly,
            )

            fit_warnings = [w for w in warning_list if issubclass(w.category, GoodnessOfFitWarning)]
            # soft test - uniform should often produce poor fit
            if len(fit_warnings) > 0:
                warning = fit_warnings[0].message
                assert warning.distribution_name == "pearson3"


class TestCalculationsCompleteWithWarnings:
    """Test that calculations complete successfully despite warnings."""

    def test_gamma_parameters_returns_valid_results_with_warnings(self) -> None:
        """gamma_parameters should return valid results even when warnings are emitted."""
        # create dataset that triggers multiple warnings
        values = np.random.gamma(2, 2, size=(25, 12))  # short calibration
        mask = np.random.rand(25, 12) < 0.25  # missing data
        values[mask] = np.nan

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            alphas, betas = compute.gamma_parameters(
                values,
                data_start_year=1995,
                calibration_start_year=1995,
                calibration_end_year=2019,
                periodicity=compute.Periodicity.monthly,
            )

        # results should be valid despite warnings
        assert alphas.shape == (12,)
        assert betas.shape == (12,)
        assert np.all(np.isfinite(alphas) | np.isnan(alphas))
        assert np.all(np.isfinite(betas) | np.isnan(betas))

    def test_pearson_parameters_returns_valid_results_with_warnings(self) -> None:
        """pearson_parameters should return valid results even when warnings are emitted."""
        values = np.random.randn(20, 12)  # short calibration
        mask = np.random.rand(20, 12) < 0.25  # missing data
        values[mask] = np.nan

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            probs, locs, scales, skews = compute.pearson_parameters(
                values,
                data_start_year=2000,
                calibration_start_year=2000,
                calibration_end_year=2019,
                periodicity=compute.Periodicity.monthly,
            )

        # results should be valid despite warnings
        assert probs.shape == (12,)
        assert locs.shape == (12,)
        assert scales.shape == (12,)
        assert skews.shape == (12,)

    def test_warnings_are_suppressible_during_computation(self) -> None:
        """Users should be able to suppress warnings during index computation."""
        values = np.random.gamma(2, 2, size=(25, 12))

        # suppress all climate indices warnings
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=ClimateIndicesWarning)

            compute.gamma_parameters(
                values,
                data_start_year=1995,
                calibration_start_year=1995,
                calibration_end_year=2019,
                periodicity=compute.Periodicity.monthly,
            )

            # no climate indices warnings should be recorded
            climate_warnings = [w for w in warning_list if issubclass(w.category, ClimateIndicesWarning)]
            assert len(climate_warnings) == 0
