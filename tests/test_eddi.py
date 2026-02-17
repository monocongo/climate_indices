"""Tests for the EDDI (Evaporative Demand Drought Index) implementation.

Covers the Hastings inverse-normal approximation, empirical ranking logic,
input validation, edge cases, and integration with v2.4.0 structured exceptions.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from climate_indices import compute, indices
from climate_indices.exceptions import DataShapeError, InvalidArgumentError


# ---------------------------------------------------------------------------
# Hastings inverse normal approximation
# ---------------------------------------------------------------------------


class TestHastingsInverseNormal:
    """Tests for _hastings_inverse_normal helper."""

    def test_accuracy_vs_scipy(self) -> None:
        """Hastings approximation should match scipy.stats.norm.ppf within 5e-4."""
        probabilities = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        z_hastings = indices._hastings_inverse_normal(probabilities)
        z_scipy = stats.norm.ppf(probabilities)
        np.testing.assert_allclose(z_hastings, z_scipy, atol=5e-4)

    def test_symmetry(self) -> None:
        """z(p) should equal -z(1-p) for symmetric pairs."""
        p_low = np.array([0.1, 0.2, 0.3])
        p_high = 1.0 - p_low
        z_low = indices._hastings_inverse_normal(p_low)
        z_high = indices._hastings_inverse_normal(p_high)
        np.testing.assert_allclose(z_low, -z_high, atol=1e-6)


# ---------------------------------------------------------------------------
# EDDI function - validation and edge cases
# ---------------------------------------------------------------------------


class TestEddiValidation:
    """Tests for EDDI input validation and error handling."""

    def test_all_nans(self) -> None:
        """All-NaN input should return all-NaN output without error."""
        pet = np.full(120, np.nan)
        result = indices.eddi(
            pet, scale=1, data_start_year=2000,
            calibration_year_initial=2000, calibration_year_final=2009,
            periodicity=compute.Periodicity.monthly,
        )
        assert result.shape == pet.shape
        assert np.all(np.isnan(result))

    def test_negative_values_clipped(self) -> None:
        """Negative PET values should be clipped to zero (not raise)."""
        rng = np.random.default_rng(42)
        pet = rng.normal(50.0, 30.0, size=360)
        # ensure some negatives exist
        assert np.any(pet < 0)
        result = indices.eddi(
            pet, scale=1, data_start_year=1990,
            calibration_year_initial=1990, calibration_year_final=2019,
            periodicity=compute.Periodicity.monthly,
        )
        assert result.shape == pet.shape

    def test_invalid_array_shape_raises_datashapeerror(self) -> None:
        """2-D input should raise DataShapeError, not a generic error."""
        pet_2d = np.ones((10, 12))
        with pytest.raises(DataShapeError, match="1-D"):
            indices.eddi(
                pet_2d, scale=1, data_start_year=2000,
                calibration_year_initial=2000, calibration_year_final=2009,
                periodicity=compute.Periodicity.monthly,
            )

    def test_invalid_periodicity_raises_invalidargumenterror(self) -> None:
        """Non-enum periodicity should raise InvalidArgumentError."""
        pet = np.ones(120)
        with pytest.raises(InvalidArgumentError, match="periodicity"):
            indices.eddi(
                pet, scale=1, data_start_year=2000,
                calibration_year_initial=2000, calibration_year_final=2009,
                periodicity="monthly",  # type: ignore[arg-type]
            )

    def test_calibration_start_before_data(self) -> None:
        """calibration_year_initial before data_start_year should raise."""
        pet = np.ones(120)
        with pytest.raises(InvalidArgumentError, match="Calibration start year"):
            indices.eddi(
                pet, scale=1, data_start_year=2000,
                calibration_year_initial=1999, calibration_year_final=2009,
                periodicity=compute.Periodicity.monthly,
            )

    def test_calibration_end_after_data(self) -> None:
        """calibration_year_final beyond data extent should raise."""
        pet = np.ones(120)
        with pytest.raises(InvalidArgumentError, match="Calibration end year"):
            indices.eddi(
                pet, scale=1, data_start_year=2000,
                calibration_year_initial=2000, calibration_year_final=2020,
                periodicity=compute.Periodicity.monthly,
            )

    def test_calibration_start_after_end(self) -> None:
        """calibration start > end should raise."""
        pet = np.ones(120)
        with pytest.raises(InvalidArgumentError, match="Calibration start year"):
            indices.eddi(
                pet, scale=1, data_start_year=2000,
                calibration_year_initial=2009, calibration_year_final=2000,
                periodicity=compute.Periodicity.monthly,
            )


# ---------------------------------------------------------------------------
# EDDI function - core computation
# ---------------------------------------------------------------------------


class TestEddiComputation:
    """Tests for EDDI computational correctness."""

    @pytest.fixture()
    def synthetic_pet(self) -> np.ndarray:
        """30 years of monthly synthetic PET with seasonal cycle."""
        rng = np.random.default_rng(123)
        n_years = 30
        months = np.tile(np.arange(12), n_years)
        # seasonal cycle + noise
        seasonal = 50.0 + 30.0 * np.sin(2.0 * np.pi * months / 12.0)
        noise = rng.normal(0, 5.0, size=n_years * 12)
        return np.maximum(seasonal + noise, 0.0)

    def test_basic_computation_scale1(self, synthetic_pet: np.ndarray) -> None:
        """EDDI at scale=1 should return valid z-scores."""
        result = indices.eddi(
            synthetic_pet, scale=1, data_start_year=1990,
            calibration_year_initial=1990, calibration_year_final=2019,
            periodicity=compute.Periodicity.monthly,
        )
        assert result.shape == synthetic_pet.shape
        # z-scores should be in clipped range
        valid = ~np.isnan(result)
        assert np.all(result[valid] >= -3.09)
        assert np.all(result[valid] <= 3.09)

    def test_basic_computation_scale6(self, synthetic_pet: np.ndarray) -> None:
        """EDDI at scale=6 should have more NaNs at the start (from sum_to_scale)."""
        result_1 = indices.eddi(
            synthetic_pet, scale=1, data_start_year=1990,
            calibration_year_initial=1990, calibration_year_final=2019,
            periodicity=compute.Periodicity.monthly,
        )
        result_6 = indices.eddi(
            synthetic_pet, scale=6, data_start_year=1990,
            calibration_year_initial=1990, calibration_year_final=2019,
            periodicity=compute.Periodicity.monthly,
        )
        # scale=6 should have more leading NaNs
        nans_1 = np.sum(np.isnan(result_1))
        nans_6 = np.sum(np.isnan(result_6))
        assert nans_6 >= nans_1

    def test_empirical_ranking_monotonicity(self) -> None:
        """Higher PET in the same month should produce higher EDDI z-scores."""
        n_years = 30
        # create perfectly monotonic PET for each month
        pet = np.zeros(n_years * 12)
        for month in range(12):
            for yr in range(n_years):
                pet[yr * 12 + month] = float(yr + 1)

        result = indices.eddi(
            pet, scale=1, data_start_year=1990,
            calibration_year_initial=1990, calibration_year_final=2019,
            periodicity=compute.Periodicity.monthly,
        )

        # for each month, z-scores should be monotonically non-decreasing
        for month in range(12):
            month_values = result[month::12]
            valid = ~np.isnan(month_values)
            valid_z = month_values[valid]
            if len(valid_z) > 1:
                assert np.all(np.diff(valid_z) >= -1e-10), (
                    f"Month {month}: z-scores not monotonic: {valid_z}"
                )

    def test_daily_periodicity(self) -> None:
        """EDDI should work with daily periodicity (366 steps/year)."""
        rng = np.random.default_rng(99)
        # 5 years of daily data
        pet = rng.uniform(1.0, 10.0, size=5 * 366)
        result = indices.eddi(
            pet, scale=1, data_start_year=2015,
            calibration_year_initial=2015, calibration_year_final=2019,
            periodicity=compute.Periodicity.daily,
        )
        assert result.shape == pet.shape

    def test_insufficient_climatology(self) -> None:
        """With < 3 valid calibration values per step, output should be NaN."""
        # only 2 years of data -- less than 3 valid values per month
        pet = np.ones(24)
        result = indices.eddi(
            pet, scale=1, data_start_year=2000,
            calibration_year_initial=2000, calibration_year_final=2001,
            periodicity=compute.Periodicity.monthly,
        )
        # all should be NaN because we only have 2 years
        assert np.all(np.isnan(result))

    def test_mixed_nan_positions(self) -> None:
        """NaN values scattered through the input should propagate correctly."""
        rng = np.random.default_rng(77)
        pet = rng.uniform(10.0, 100.0, size=360)
        # scatter NaNs
        nan_idx = rng.choice(360, size=36, replace=False)
        pet[nan_idx] = np.nan

        result = indices.eddi(
            pet, scale=1, data_start_year=1990,
            calibration_year_initial=1990, calibration_year_final=2019,
            periodicity=compute.Periodicity.monthly,
        )
        assert result.shape == pet.shape
        # original NaN positions should still be NaN
        assert np.all(np.isnan(result[nan_idx]))
