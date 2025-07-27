"""
Test cases specifically for zero precipitation scenarios and distribution fallback logic.

This module tests the fixes implemented for GitHub issue #582, which addresses
SPI computation failures when dealing with extensive zero precipitation values.
"""

import logging
import warnings

import numpy as np
import pytest

from climate_indices import compute, indices
from climate_indices.compute import HIGH_FAILURE_RATE_THRESHOLD
from climate_indices.lmoments import MIN_VALUES_FOR_LMOMENTS

# Enable logging to capture warnings during tests
logging.disable(logging.NOTSET)
logging.basicConfig(level=logging.WARNING)


class TestZeroPrecipitationFix:
    """Test suite for zero precipitation handling in SPI computation."""

    def test_spi_with_extensive_zeros_robustness(self):
        """
        Test that SPI computation is robust when encountering extensive zero precipitation periods.

        This test simulates the scenario described in GitHub issue #582 where
        grid cells have zero precipitation for 3+ consecutive months. Prior to the fix,
        this would cause computation failures. This test verifies the fix works correctly.
        """
        # Create a dataset representing an extremely dry region
        # 30 years of data (1990-2019) with extensive zero precipitation
        years = 30
        months_per_year = 12
        total_months = years * months_per_year

        # Create precipitation data simulating CHIRPs data with extensive zeros
        # Many calendar months will have < 4 non-zero values across all years
        precip_data = np.zeros(total_months)

        # Add sparse precipitation to simulate realistic dry region
        np.random.seed(42)  # For reproducible test results
        for month_idx in range(0, 12):
            # Some months have very little precipitation (simulating dry season)
            if month_idx in [0, 1, 2, 6, 7, 8, 9, 10, 11]:  # 9 dry months
                # Only 1-3 years get precipitation in these months
                if np.random.random() > 0.7:  # 30% chance this month gets any precipitation at all
                    year_indices = np.random.choice(years, size=np.random.randint(1, 4), replace=False)
                    for year_idx in year_indices:
                        precip_data[year_idx * months_per_year + month_idx] = np.random.exponential(scale=5.0)
            else:
                # Wet season months (3, 4, 5) get more regular precipitation
                year_indices = np.random.choice(years, size=np.random.randint(8, 15), replace=False)
                for year_idx in year_indices:
                    precip_data[year_idx * months_per_year + month_idx] = np.random.exponential(scale=20.0)

        data_start_year = 1990
        calibration_start_year = 1990
        calibration_end_year = 2019
        scale = 3  # 3-month SPI as mentioned in the GitHub issue

        # Test with both distributions - both should complete without crashing
        for distribution in [indices.Distribution.gamma, indices.Distribution.pearson]:
            try:
                spi_values = indices.spi(
                    values=precip_data,
                    scale=scale,
                    distribution=distribution,
                    data_start_year=data_start_year,
                    calibration_year_initial=calibration_start_year,
                    calibration_year_final=calibration_end_year,
                    periodicity=compute.Periodicity.monthly,
                )

                # Verify that computation completed without errors
                assert len(spi_values) == total_months, (
                    f"SPI output length should match input length for {distribution.value} distribution"
                )

                # After the initial scale-1 months, we should have some SPI values
                # This is the key requirement: the computation should not fail completely
                valid_spi = spi_values[scale - 1 :]
                non_nan_count = np.count_nonzero(~np.isnan(valid_spi))

                # We should get at least some valid values, even if not all
                assert non_nan_count > 0, (
                    f"SPI computation should produce some valid values even with extensive zeros "
                    f"for {distribution.value} distribution"
                )

                # Verify SPI values are within expected range [-3.09, 3.09]
                valid_values = valid_spi[~np.isnan(valid_spi)]
                if len(valid_values) > 0:
                    assert np.all(valid_values >= -3.09), f"SPI values should be >= -3.09 for {distribution.value}"
                    assert np.all(valid_values <= 3.09), f"SPI values should be <= 3.09 for {distribution.value}"

            except Exception as e:
                pytest.fail(
                    f"SPI computation failed with extensive zeros for {distribution.value} distribution: {e}. "
                    f"This indicates the GitHub issue #582 fix is not working properly."
                )

    def test_spi_all_zeros_single_month(self):
        """
        Test SPI computation when a specific calendar month has all zeros
        in the calibration period.
        """
        # Create 30 years of monthly data
        years = 30
        months_per_year = 12
        total_months = years * months_per_year

        # Start with some baseline precipitation
        np.random.seed(123)
        precip_data = np.random.exponential(scale=10.0, size=total_months).reshape(years, months_per_year)

        # Make July (month index 6) completely dry for all years
        # This simulates a region where a specific month never receives precipitation
        precip_data[:, 6] = 0.0

        # Flatten back to 1D for SPI computation
        precip_data_flat = precip_data.flatten()

        data_start_year = 1990
        calibration_start_year = 1990
        calibration_end_year = 2019
        scale = 1  # 1-month SPI to test monthly effects

        # Test with both distributions
        for distribution in [indices.Distribution.gamma, indices.Distribution.pearson]:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")

                spi_values = indices.spi(
                    values=precip_data_flat,
                    scale=scale,
                    distribution=distribution,
                    data_start_year=data_start_year,
                    calibration_year_initial=calibration_start_year,
                    calibration_year_final=calibration_end_year,
                    periodicity=compute.Periodicity.monthly,
                )

                # Verify that computation completed without errors
                assert len(spi_values) == total_months, (
                    f"SPI output length should match input length for {distribution.value} distribution"
                )

                # Check that July months have valid handling (may be NaN due to all zeros)
                july_indices = list(range(6, total_months, 12))  # Every July
                july_spi = spi_values[july_indices]

                # July should either have consistent NaN values or extreme drought values
                # This verifies that the zero precipitation is handled gracefully
                july_non_nan = july_spi[~np.isnan(july_spi)]
                if len(july_non_nan) > 0:
                    # If values are computed, they should indicate extreme drought
                    assert np.all(july_non_nan <= -2.0), (
                        f"July SPI values should indicate extreme drought for {distribution.value} distribution"
                    )

    def test_pearson_parameters_insufficient_data_handling(self):
        """
        Test that Pearson parameters function handles insufficient non-zero data
        gracefully by raising an informative exception when too many time steps fail.
        """
        # Create a very sparse dataset with insufficient non-zero values
        # 10 years of monthly data with only 2 non-zero values per calendar month
        years = 10
        months_per_year = 12

        precip_data = np.zeros((years, months_per_year))

        # Add only 2 non-zero values per calendar month (insufficient for L-moments)
        # This is less than the required minimum for Pearson Type III fitting
        for month in range(months_per_year):
            # Only 2 years have precipitation in each month
            precip_data[0, month] = 10.0
            precip_data[1, month] = 15.0
            # All other years have zero for this month

        data_start_year = 2000
        calibration_start_year = 2000
        calibration_end_year = 2009

        # This should complete and issue a warning due to high failure rate
        import io
        import logging

        # Create a log capture handler
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger("climate_indices.compute")
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

        try:
            probabilities_of_zero, locs, scales, skews = compute.pearson_parameters(
                values=precip_data,
                data_start_year=data_start_year,
                calibration_start_year=calibration_start_year,
                calibration_end_year=calibration_end_year,
                periodicity=compute.Periodicity.monthly,
            )

            # Check log output for high failure rate warning
            log_output = log_capture.getvalue()

            # Verify that warning was issued about high failure rate
            assert "High failure rate for Pearson Type III distribution fitting" in log_output
            assert "100.0% failure rate" in log_output
            assert "extensive zero precipitation patterns" in log_output
            assert "better handled by Gamma distribution" in log_output

        finally:
            # Clean up
            logger.removeHandler(handler)
            handler.close()

        # Verify that default values are returned for months with insufficient data
        # Most months should have default parameter values (0.0)
        zero_count = np.count_nonzero(locs == 0.0)
        assert zero_count >= months_per_year * HIGH_FAILURE_RATE_THRESHOLD, (
            "Most months should have default loc parameters due to insufficient data"
        )

        # Verify that the function returns arrays of correct size
        assert len(probabilities_of_zero) == months_per_year
        assert len(locs) == months_per_year
        assert len(scales) == months_per_year
        assert len(skews) == months_per_year

    def test_gamma_distribution_handles_zeros_correctly(self):
        """
        Test that Gamma distribution correctly handles datasets with many zeros
        and produces valid SPI values.
        """
        # Create a dataset typical of semi-arid regions
        # Some months with significant precipitation, others completely dry
        years = 25
        months_per_year = 12
        total_months = years * months_per_year

        np.random.seed(456)
        precip_data = np.zeros(total_months)

        # Simulate seasonal precipitation pattern (only wet season gets rain)
        for year in range(years):
            year_start = year * months_per_year
            # Wet season: months 5-9 (May-September) get precipitation
            wet_season_start = year_start + 4  # May (0-indexed)
            wet_season_end = year_start + 9  # September

            wet_months = range(wet_season_start, wet_season_end + 1)
            for month in wet_months:
                if np.random.random() > 0.3:  # 70% chance of precipitation in wet season
                    precip_data[month] = np.random.gamma(shape=2.0, scale=20.0)

            # Dry season: very occasional light precipitation
            dry_months = list(range(year_start, wet_season_start)) + list(
                range(wet_season_end + 1, year_start + months_per_year)
            )
            for month in dry_months:
                if np.random.random() > 0.9:  # 10% chance of light precipitation in dry season
                    precip_data[month] = np.random.exponential(scale=2.0)

        data_start_year = 1995
        calibration_start_year = 1995
        calibration_end_year = 2019
        scale = 6  # 6-month SPI

        # Test Gamma distribution with this challenging dataset
        spi_values = indices.spi(
            values=precip_data,
            scale=scale,
            distribution=indices.Distribution.gamma,
            data_start_year=data_start_year,
            calibration_year_initial=calibration_start_year,
            calibration_year_final=calibration_end_year,
            periodicity=compute.Periodicity.monthly,
        )

        # Verify that SPI computation completed successfully
        assert len(spi_values) == total_months, "SPI output should match input length"

        # Check that we have valid SPI values after the initial scale period
        valid_period = spi_values[scale - 1 :]
        non_nan_count = np.count_nonzero(~np.isnan(valid_period))

        assert non_nan_count > total_months * 0.7, (
            "Gamma distribution should successfully compute SPI for most of the time series"
        )

        # Verify SPI values are within valid range
        valid_spi = valid_period[~np.isnan(valid_period)]
        assert np.all(valid_spi >= -3.09) and np.all(valid_spi <= 3.09), (
            "All SPI values should be within the valid range [-3.09, 3.09]"
        )

        # Check that wet and dry seasons show expected SPI patterns
        # Wet season should have higher (less negative) SPI values on average
        wet_season_indices = []
        dry_season_indices = []

        for year in range(years):
            year_start = (year * months_per_year) + scale - 1  # Adjust for scale offset
            if year_start + months_per_year <= len(valid_period):
                # Wet season SPI (centered around July-August)
                wet_season_indices.extend(range(year_start + 6, year_start + 9))
                # Dry season SPI (centered around December-February)
                dry_season_indices.extend([year_start + 11, year_start + 0, year_start + 1])

        # Filter for valid indices and non-NaN values
        wet_season_indices = [i for i in wet_season_indices if i < len(valid_period)]
        dry_season_indices = [i for i in dry_season_indices if i < len(valid_period)]

        wet_season_spi = valid_period[wet_season_indices]
        dry_season_spi = valid_period[dry_season_indices]

        # Remove NaN values
        wet_season_spi = wet_season_spi[~np.isnan(wet_season_spi)]
        dry_season_spi = dry_season_spi[~np.isnan(dry_season_spi)]

        if len(wet_season_spi) > 0 and len(dry_season_spi) > 0:
            # Wet season should generally have higher SPI values than dry season
            wet_median = np.median(wet_season_spi)
            dry_median = np.median(dry_season_spi)

            assert wet_median > dry_median, "Wet season should have higher median SPI than dry season"

    def test_lmoments_error_message_enhancement(self):
        """
        Test that enhanced error messages are provided when L-moments
        computation fails due to insufficient data.
        """
        # Create a time series with only 2 non-NaN values (insufficient for L-moments)
        # L-moments counts non-NaN values, not non-zero values
        insufficient_data = np.array([5.0, 10.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        # This should raise a ValueError with our enhanced error message
        with pytest.raises(ValueError) as exc_info:
            from climate_indices import lmoments

            lmoments.fit(insufficient_data)

        error_message = str(exc_info.value)

        # Verify that the enhanced error message provides helpful context
        assert "Insufficient number of values to perform sample L-moments estimation" in error_message
        assert f"2 non-NaN values found (minimum {MIN_VALUES_FOR_LMOMENTS} required)" in error_message
        assert "dry regions with extensive zero precipitation" in error_message
        assert "Consider using Gamma distribution" in error_message

    @pytest.mark.parametrize("distribution", [indices.Distribution.gamma, indices.Distribution.pearson])
    def test_spi_robustness_with_mixed_zero_patterns(self, distribution):
        """
        Test SPI computation robustness with various zero precipitation patterns
        for both Gamma and Pearson distributions.
        """
        # Create datasets with different zero patterns
        years = 20
        months_per_year = 12
        total_months = years * months_per_year

        test_patterns = {
            "seasonal_zeros": self._create_seasonal_zero_pattern(years, months_per_year),
            "random_zeros": self._create_random_zero_pattern(years, months_per_year, zero_probability=0.4),
            "clustered_zeros": self._create_clustered_zero_pattern(years, months_per_year),
        }

        data_start_year = 2000
        calibration_start_year = 2000
        calibration_end_year = 2019
        scale = 3

        for pattern_name, precip_data in test_patterns.items():
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")

                try:
                    spi_values = indices.spi(
                        values=precip_data,
                        scale=scale,
                        distribution=distribution,
                        data_start_year=data_start_year,
                        calibration_year_initial=calibration_start_year,
                        calibration_year_final=calibration_end_year,
                        periodicity=compute.Periodicity.monthly,
                    )

                    # Verify basic properties
                    assert len(spi_values) == total_months, (
                        f"SPI output length mismatch for {pattern_name} with {distribution.value}"
                    )

                    # Check that we get some valid values
                    valid_spi = spi_values[~np.isnan(spi_values)]
                    if len(valid_spi) > 0:
                        assert np.all(valid_spi >= -3.09) and np.all(valid_spi <= 3.09), (
                            f"SPI values out of range for {pattern_name} with {distribution.value}"
                        )

                except Exception as e:
                    pytest.fail(f"SPI computation failed for {pattern_name} with {distribution.value}: {e}")

    def _create_seasonal_zero_pattern(self, years, months_per_year):
        """Create precipitation data with seasonal zero patterns."""
        total_months = years * months_per_year
        np.random.seed(789)
        precip_data = np.zeros(total_months)

        for year in range(years):
            year_start = year * months_per_year
            # Only months 3-8 can have precipitation
            for month in range(3, 9):
                if np.random.random() > 0.2:
                    precip_data[year_start + month] = np.random.exponential(scale=15.0)

        return precip_data

    def _create_random_zero_pattern(self, years, months_per_year, zero_probability):
        """Create precipitation data with random zero patterns."""
        total_months = years * months_per_year
        np.random.seed(101112)
        precip_data = np.random.exponential(scale=12.0, size=total_months)

        # Randomly set some months to zero
        zero_mask = np.random.random(total_months) < zero_probability
        precip_data[zero_mask] = 0.0

        return precip_data

    def _create_clustered_zero_pattern(self, years, months_per_year):
        """Create precipitation data with clustered zero patterns (drought periods)."""
        total_months = years * months_per_year
        np.random.seed(131415)
        precip_data = np.random.exponential(scale=8.0, size=total_months)

        # Create drought periods (consecutive zeros)
        drought_starts = np.random.choice(range(0, total_months - 12, 6), size=years // 3, replace=False)

        for start in drought_starts:
            drought_length = np.random.randint(3, 8)
            end = min(start + drought_length, total_months)
            precip_data[start:end] = 0.0

        return precip_data
