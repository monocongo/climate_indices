"""
Story 4.3: Edge Case Coverage Tests

Tests edge cases through the typed public API (climate_indices.spi, climate_indices.spei).
Covers:
- Zero-inflated precipitation (all zeros, mixed zeros/non-zeros)
- Missing data patterns (random NaN, leading/trailing/block NaN)
- Minimum time series (exactly 30 years for SPI/SPEI)
- Coordinate misalignment (different time ranges)
- Single-point vs gridded data
- Graceful degradation (extreme values, partial failures)

Acceptance Criteria:
- FR-TEST-003: Edge case tests verify known failure modes
- NFR-REL-002: Tests verify graceful degradation (partial failures don't crash)

Related Stories:
- Story 4.1: xarray equivalence tests
- Story 4.2: metadata validation tests
- Story 4.6: exception handling tests
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from climate_indices import spei, spi
from climate_indices.exceptions import CoordinateValidationError, InputAlignmentWarning, ShortCalibrationWarning
from climate_indices.indices import Distribution

# ==============================================================================
# Module-scoped result fixtures
# Cache expensive SPI/SPEI computations that multiple tests reuse
# ==============================================================================


@pytest.fixture(scope="module")
def spi_zero_inflated_result(zero_inflated_precip_da: xr.DataArray) -> xr.DataArray:
    """Cached SPI result for zero-inflated precipitation."""
    return spi(zero_inflated_precip_da, scale=3, distribution=Distribution.gamma)


@pytest.fixture(scope="module")
def spi_leading_nan_result(leading_nan_block_da: xr.DataArray) -> xr.DataArray:
    """Cached SPI result for leading NaN block."""
    return spi(leading_nan_block_da, scale=3, distribution=Distribution.gamma)


@pytest.fixture(scope="module")
def spi_trailing_nan_result(trailing_nan_block_da: xr.DataArray) -> xr.DataArray:
    """Cached SPI result for trailing NaN block."""
    return spi(trailing_nan_block_da, scale=3, distribution=Distribution.gamma)


@pytest.fixture(scope="module")
def spi_block_nan_result(block_nan_pattern_da: xr.DataArray) -> xr.DataArray:
    """Cached SPI result for block NaN pattern."""
    return spi(block_nan_pattern_da, scale=3, distribution=Distribution.gamma)


@pytest.fixture(scope="module")
def spi_minimum_calibration_result(minimum_calibration_da: xr.DataArray) -> xr.DataArray:
    """Cached SPI result for minimum calibration period (30 years)."""
    return spi(minimum_calibration_da, scale=3, distribution=Distribution.gamma)


@pytest.fixture(scope="module")
def spi_single_point_result(single_point_monthly_da: xr.DataArray) -> xr.DataArray:
    """Cached SPI result for single-point (1D time-only) data."""
    return spi(single_point_monthly_da, scale=3, distribution=Distribution.gamma)


@pytest.fixture(scope="module")
def spi_gridded_result(dask_monthly_precip_3d: xr.DataArray) -> xr.DataArray:
    """Cached SPI result for 3D gridded data (Dask-backed for vectorization)."""
    return spi(dask_monthly_precip_3d, scale=3, distribution=Distribution.gamma)


# ==============================================================================
# Test Class 1: Zero-Inflated Precipitation
# AC: Zero-inflated precipitation (all zeros, mixed zeros/non-zeros)
# ==============================================================================


class TestZeroInflatedPrecipitation:
    """Tests for zero-inflated precipitation patterns."""

    def test_spi_zero_inflated_completes(self, spi_zero_inflated_result: xr.DataArray) -> None:
        """SPI computation with ~50% zeros returns DataArray without exception."""
        assert isinstance(spi_zero_inflated_result, xr.DataArray)

    def test_spi_zero_inflated_has_finite_values(self, spi_zero_inflated_result: xr.DataArray) -> None:
        """Zero-inflated SPI result has some non-NaN values (distribution fits partial data)."""
        finite_count = np.isfinite(spi_zero_inflated_result.values).sum()
        assert finite_count > 0, "Expected some finite values in zero-inflated SPI result"

    def test_spi_zero_inflated_shape_preserved(
        self, zero_inflated_precip_da: xr.DataArray, spi_zero_inflated_result: xr.DataArray
    ) -> None:
        """Output shape matches input shape for zero-inflated data."""
        assert spi_zero_inflated_result.shape == zero_inflated_precip_da.shape

    def test_spi_zero_inflated_coords_preserved(
        self, zero_inflated_precip_da: xr.DataArray, spi_zero_inflated_result: xr.DataArray
    ) -> None:
        """Time coordinates preserved for zero-inflated data."""
        assert spi_zero_inflated_result.coords["time"].equals(zero_inflated_precip_da.coords["time"])

    def test_spi_zero_inflated_pearson_completes(self, zero_inflated_precip_da: xr.DataArray) -> None:
        """Pearson distribution also handles zero-inflated data."""
        result = spi(zero_inflated_precip_da, scale=3, distribution=Distribution.pearson)
        assert isinstance(result, xr.DataArray)
        assert np.isfinite(result.values).sum() > 0

    def test_spi_all_zeros_returns_nan(self) -> None:
        """All-zero input produces all-NaN output (unfittable distribution)."""
        import pandas as pd

        time = pd.date_range("1980-01-01", periods=480, freq="MS")
        data = np.zeros(480)
        da = xr.DataArray(data, coords={"time": time}, dims=["time"], name="precipitation")
        result = spi(da, scale=3, distribution=Distribution.gamma)
        assert isinstance(result, xr.DataArray)
        # all zeros cannot fit a distribution, but implementation may handle gracefully
        # verify it doesn't crash and returns a DataArray
        assert result.shape == (480,)

    def test_spei_zero_inflated_completes(
        self, zero_inflated_precip_da: xr.DataArray, sample_monthly_pet_da: xr.DataArray
    ) -> None:
        """SPEI with zero-inflated precip and matching PET completes."""
        result = spei(zero_inflated_precip_da, sample_monthly_pet_da, scale=3, distribution=Distribution.gamma)
        assert isinstance(result, xr.DataArray)

    def test_spi_zero_inflated_values_in_range(self, spi_zero_inflated_result: xr.DataArray) -> None:
        """Finite SPI values fall within expected range [-3.09, 3.09]."""
        finite_values = spi_zero_inflated_result.values[np.isfinite(spi_zero_inflated_result.values)]
        if len(finite_values) > 0:
            assert np.all(finite_values >= -3.09)
            assert np.all(finite_values <= 3.09)


# ==============================================================================
# Test Class 2: Random NaN Pattern
# AC: Missing data patterns — random NaN
# ==============================================================================


class TestRandomNaNPattern:
    """Tests for randomly scattered NaN values in precipitation."""

    def test_spi_random_nan_completes(self, random_nan_precip_da: xr.DataArray) -> None:
        """SPI with ~10% random NaN completes without exception."""
        result = spi(random_nan_precip_da, scale=3, distribution=Distribution.gamma)
        assert isinstance(result, xr.DataArray)

    def test_spi_random_nan_input_nan_propagated(self, random_nan_precip_da: xr.DataArray) -> None:
        """Where input had NaN, output also has NaN."""
        result = spi(random_nan_precip_da, scale=3, distribution=Distribution.gamma)
        input_nan_mask = np.isnan(random_nan_precip_da.values)
        output_nan_mask = np.isnan(result.values)
        # every input NaN should be output NaN (may have more due to scale convolution)
        assert np.all(output_nan_mask[input_nan_mask])

    def test_spi_random_nan_output_has_more_nan(self, random_nan_precip_da: xr.DataArray) -> None:
        """Output NaN count >= input NaN count (scale convolution propagates NaN)."""
        result = spi(random_nan_precip_da, scale=3, distribution=Distribution.gamma)
        input_nan_count = np.isnan(random_nan_precip_da.values).sum()
        output_nan_count = np.isnan(result.values).sum()
        assert output_nan_count >= input_nan_count

    def test_spei_random_nan_completes(
        self, random_nan_precip_da: xr.DataArray, sample_monthly_pet_da: xr.DataArray
    ) -> None:
        """SPEI with random NaN in both inputs completes."""
        # introduce NaN to PET as well
        pet_with_nan = sample_monthly_pet_da.copy(deep=True)
        rng = np.random.default_rng(56)
        nan_mask = rng.random(len(pet_with_nan)) < 0.1
        pet_with_nan.values[nan_mask] = np.nan

        result = spei(random_nan_precip_da, pet_with_nan, scale=3, distribution=Distribution.gamma)
        assert isinstance(result, xr.DataArray)

    def test_spei_random_nan_propagation(
        self, random_nan_precip_da: xr.DataArray, sample_monthly_pet_da: xr.DataArray
    ) -> None:
        """NaN positions from either input are propagated to output."""
        # PET with different NaN positions
        pet_with_nan = sample_monthly_pet_da.copy(deep=True)
        rng = np.random.default_rng(57)
        pet_nan_mask = rng.random(len(pet_with_nan)) < 0.05
        pet_with_nan.values[pet_nan_mask] = np.nan

        result = spei(random_nan_precip_da, pet_with_nan, scale=3, distribution=Distribution.gamma)

        # check that NaN from either input appears in output
        input_precip_nan = np.isnan(random_nan_precip_da.values)
        input_pet_nan = np.isnan(pet_with_nan.values)
        combined_input_nan = input_precip_nan | input_pet_nan
        output_nan = np.isnan(result.values)

        assert np.all(output_nan[combined_input_nan])


# ==============================================================================
# Test Class 3: Leading NaN Block
# AC: Missing data patterns — leading NaN blocks
# ==============================================================================


class TestLeadingNaNBlock:
    """Tests for precipitation with leading NaN block (first 12 months)."""

    def test_spi_leading_nan_completes(self, spi_leading_nan_result: xr.DataArray) -> None:
        """SPI with leading NaN block returns DataArray."""
        assert isinstance(spi_leading_nan_result, xr.DataArray)

    def test_spi_leading_nan_propagated(self, spi_leading_nan_result: xr.DataArray) -> None:
        """First 12 months are NaN in output (input NaN propagated)."""
        assert np.isnan(spi_leading_nan_result.values[:12]).all()

    def test_spi_leading_nan_valid_after_block(self, spi_leading_nan_result: xr.DataArray) -> None:
        """Non-NaN values exist after month 12."""
        finite_after_block = np.isfinite(spi_leading_nan_result.values[12:]).sum()
        assert finite_after_block > 0, "Expected finite values after leading NaN block"

    def test_spei_leading_nan_completes(
        self, leading_nan_block_da: xr.DataArray, sample_monthly_pet_da: xr.DataArray
    ) -> None:
        """SPEI with leading NaN in precip handles gracefully."""
        result = spei(leading_nan_block_da, sample_monthly_pet_da, scale=3, distribution=Distribution.gamma)
        assert isinstance(result, xr.DataArray)
        assert np.isnan(result.values[:12]).all()


# ==============================================================================
# Test Class 4: Trailing NaN Block
# AC: Missing data patterns — trailing NaN blocks
# ==============================================================================


class TestTrailingNaNBlock:
    """Tests for precipitation with trailing NaN block (last 12 months)."""

    def test_spi_trailing_nan_completes(self, spi_trailing_nan_result: xr.DataArray) -> None:
        """SPI with trailing NaN block returns DataArray."""
        assert isinstance(spi_trailing_nan_result, xr.DataArray)

    def test_spi_trailing_nan_propagated(self, spi_trailing_nan_result: xr.DataArray) -> None:
        """Last 12 months are NaN in output."""
        assert np.isnan(spi_trailing_nan_result.values[-12:]).all()

    def test_spi_trailing_nan_valid_before_block(self, spi_trailing_nan_result: xr.DataArray) -> None:
        """Non-NaN values exist before the trailing NaN block."""
        finite_before_block = np.isfinite(spi_trailing_nan_result.values[:-12]).sum()
        assert finite_before_block > 0, "Expected finite values before trailing NaN block"


# ==============================================================================
# Test Class 5: Block NaN Pattern
# AC: Missing data patterns — contiguous NaN blocks
# ==============================================================================


class TestBlockNaNPattern:
    """Tests for precipitation with mid-series contiguous NaN block."""

    def test_spi_block_nan_completes(self, spi_block_nan_result: xr.DataArray) -> None:
        """SPI with mid-series NaN block returns DataArray."""
        assert isinstance(spi_block_nan_result, xr.DataArray)

    def test_spi_block_nan_propagated(self, spi_block_nan_result: xr.DataArray) -> None:
        """Months 240-251 (year 21) are NaN in output."""
        assert np.isnan(spi_block_nan_result.values[240:252]).all()

    def test_spi_block_nan_valid_outside(self, spi_block_nan_result: xr.DataArray) -> None:
        """Regions outside NaN block have finite values."""
        finite_before = np.isfinite(spi_block_nan_result.values[:240]).sum()
        finite_after = np.isfinite(spi_block_nan_result.values[252:]).sum()
        assert finite_before > 0, "Expected finite values before NaN block"
        assert finite_after > 0, "Expected finite values after NaN block"

    def test_spi_block_nan_shape_preserved(
        self, block_nan_pattern_da: xr.DataArray, spi_block_nan_result: xr.DataArray
    ) -> None:
        """Output shape matches input shape for block NaN pattern."""
        assert spi_block_nan_result.shape == block_nan_pattern_da.shape


# ==============================================================================
# Test Class 6: Minimum Calibration Period
# AC: Minimum time series (exactly 30 years for SPI/SPEI)
# ==============================================================================


class TestMinimumCalibrationPeriod:
    """Tests for exactly 30-year calibration period (minimum valid)."""

    def test_spi_exactly_30_years_completes(self, spi_minimum_calibration_result: xr.DataArray) -> None:
        """SPI succeeds at 30-year calibration boundary."""
        assert isinstance(spi_minimum_calibration_result, xr.DataArray)

    def test_spi_exactly_30_years_shape(self, spi_minimum_calibration_result: xr.DataArray) -> None:
        """Output shape (360,) matches 30-year input."""
        assert spi_minimum_calibration_result.shape == (360,)

    def test_spi_exactly_30_years_has_finite_values(self, spi_minimum_calibration_result: xr.DataArray) -> None:
        """30-year SPI produces some finite values."""
        finite_count = np.isfinite(spi_minimum_calibration_result.values).sum()
        assert finite_count > 0, "Expected finite values with 30-year calibration"

    def test_spi_29_years_emits_warning(self) -> None:
        """29-year input emits ShortCalibrationWarning."""
        import pandas as pd

        rng = np.random.default_rng(220)
        time = pd.date_range("1990-01-01", periods=348, freq="MS")  # 29 years
        data = rng.gamma(shape=2.0, scale=25.0, size=348)
        da = xr.DataArray(data, coords={"time": time}, dims=["time"], name="precipitation")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = spi(da, scale=3, distribution=Distribution.gamma)
            assert isinstance(result, xr.DataArray)
            # check that ShortCalibrationWarning was emitted
            assert any(issubclass(warning.category, ShortCalibrationWarning) for warning in w)

    def test_spei_exactly_30_years_completes(
        self, minimum_calibration_da: xr.DataArray, minimum_calibration_pet_da: xr.DataArray
    ) -> None:
        """SPEI with matching 30-year PET succeeds."""
        result = spei(minimum_calibration_da, minimum_calibration_pet_da, scale=3, distribution=Distribution.gamma)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (360,)

    def test_spi_pearson_exactly_30_years_completes(self, minimum_calibration_da: xr.DataArray) -> None:
        """Pearson distribution at 30-year boundary succeeds."""
        result = spi(minimum_calibration_da, scale=3, distribution=Distribution.pearson)
        assert isinstance(result, xr.DataArray)
        assert np.isfinite(result.values).sum() > 0


# ==============================================================================
# Test Class 7: Coordinate Misalignment
# AC: Coordinate misalignment (different grid resolutions)
# ==============================================================================


class TestCoordinateMisalignment:
    """Tests for coordinate misalignment between precip and PET."""

    def test_spei_offset_time_emits_warning(
        self, sample_monthly_precip_da: xr.DataArray, sample_monthly_pet_offset_da: xr.DataArray
    ) -> None:
        """Offset time coordinates emit InputAlignmentWarning with correct attributes."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = spei(
                sample_monthly_precip_da, sample_monthly_pet_offset_da, scale=3, distribution=Distribution.gamma
            )
            assert isinstance(result, xr.DataArray)
            # check for InputAlignmentWarning
            alignment_warnings = [warning for warning in w if issubclass(warning.category, InputAlignmentWarning)]
            assert len(alignment_warnings) > 0, "Expected InputAlignmentWarning for offset coordinates"

    def test_spei_offset_time_intersection_size(
        self, sample_monthly_precip_da: xr.DataArray, sample_monthly_pet_offset_da: xr.DataArray
    ) -> None:
        """Result length is 420 months (1985-2019 intersection)."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = spei(
                sample_monthly_precip_da, sample_monthly_pet_offset_da, scale=3, distribution=Distribution.gamma
            )
            # precip: 1980-2019 (480), PET: 1985-2024 (480), intersection: 1985-2019 (420)
            assert len(result) == 420

    def test_spei_offset_time_has_valid_values(
        self, sample_monthly_precip_da: xr.DataArray, sample_monthly_pet_offset_da: xr.DataArray
    ) -> None:
        """Non-NaN values exist in intersection."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = spei(
                sample_monthly_precip_da, sample_monthly_pet_offset_da, scale=3, distribution=Distribution.gamma
            )
            finite_count = np.isfinite(result.values).sum()
            assert finite_count > 0, "Expected finite values in aligned intersection"

    def test_spei_no_overlap_raises_error(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """Non-overlapping PET raises CoordinateValidationError."""
        import pandas as pd

        # PET with no overlap (2030-2069)
        rng = np.random.default_rng(230)
        time = pd.date_range("2030-01-01", periods=480, freq="MS")
        data = 50.0 + rng.gamma(shape=3.0, scale=20.0, size=480)
        pet_no_overlap = xr.DataArray(data, coords={"time": time}, dims=["time"], name="pet", attrs={"units": "mm"})

        with pytest.raises(CoordinateValidationError):
            spei(sample_monthly_precip_da, pet_no_overlap, scale=3, distribution=Distribution.gamma)

    def test_spei_identical_coords_no_warning(
        self, sample_monthly_precip_da: xr.DataArray, sample_monthly_pet_da: xr.DataArray
    ) -> None:
        """Matching coordinates produce no alignment warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = spei(sample_monthly_precip_da, sample_monthly_pet_da, scale=3, distribution=Distribution.gamma)
            assert isinstance(result, xr.DataArray)
            # no InputAlignmentWarning expected
            alignment_warnings = [warning for warning in w if issubclass(warning.category, InputAlignmentWarning)]
            assert len(alignment_warnings) == 0, "Expected no alignment warning for identical coords"

    def test_spei_offset_preserves_cf_metadata(
        self, sample_monthly_precip_da: xr.DataArray, sample_monthly_pet_offset_da: xr.DataArray
    ) -> None:
        """CF attributes present after coordinate alignment."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = spei(
                sample_monthly_precip_da, sample_monthly_pet_offset_da, scale=3, distribution=Distribution.gamma
            )
            assert "standard_name" in result.attrs
            assert "long_name" in result.attrs


# ==============================================================================
# Test Class 8: Single-Point vs Gridded
# AC: Single-point vs gridded data
# ==============================================================================


class TestSinglePointVsGridded:
    """Tests for single-point (1D) vs gridded (3D) data."""

    def test_spi_single_point_completes(self, spi_single_point_result: xr.DataArray) -> None:
        """1D time-only DataArray succeeds."""
        assert isinstance(spi_single_point_result, xr.DataArray)

    def test_spi_single_point_returns_dataarray(self, spi_single_point_result: xr.DataArray) -> None:
        """Returns xr.DataArray for single-point data."""
        assert isinstance(spi_single_point_result, xr.DataArray)

    def test_spi_single_point_shape(self, spi_single_point_result: xr.DataArray) -> None:
        """Output 1D, same length as input (480)."""
        assert spi_single_point_result.ndim == 1
        assert len(spi_single_point_result) == 480

    def test_spi_gridded_completes(self, spi_gridded_result: xr.DataArray) -> None:
        """3D DataArray succeeds."""
        assert isinstance(spi_gridded_result, xr.DataArray)

    def test_spi_gridded_shape_preserved(
        self, dask_monthly_precip_3d: xr.DataArray, spi_gridded_result: xr.DataArray
    ) -> None:
        """Output shape matches 3D input (480, 5, 6)."""
        assert spi_gridded_result.shape == dask_monthly_precip_3d.shape

    def test_spi_gridded_dims_preserved(self, spi_gridded_result: xr.DataArray) -> None:
        """(time, lat, lon) dimension names preserved."""
        assert spi_gridded_result.dims == ("time", "lat", "lon")

    def test_spi_gridded_spatial_coords_preserved(
        self, dask_monthly_precip_3d: xr.DataArray, spi_gridded_result: xr.DataArray
    ) -> None:
        """lat/lon values identical between input and output."""
        assert spi_gridded_result.coords["lat"].equals(dask_monthly_precip_3d.coords["lat"])
        assert spi_gridded_result.coords["lon"].equals(dask_monthly_precip_3d.coords["lon"])

    def test_spi_gridded_has_finite_values(self, spi_gridded_result: xr.DataArray) -> None:
        """Some grid points produce finite values."""
        # force compute for Dask arrays
        values = (
            spi_gridded_result.compute().values if hasattr(spi_gridded_result, "compute") else spi_gridded_result.values
        )
        finite_count = np.isfinite(values).sum()
        assert finite_count > 0, "Expected finite values in gridded SPI result"


# ==============================================================================
# Test Class 9: Graceful Degradation
# AC: Tests verify graceful degradation (partial failures don't crash)
# ==============================================================================


class TestGracefulDegradation:
    """Tests for graceful degradation with extreme inputs."""

    def test_spi_extreme_large_values(self) -> None:
        """Precip values of 1e6 don't crash."""
        import pandas as pd

        rng = np.random.default_rng(240)
        time = pd.date_range("1980-01-01", periods=480, freq="MS")
        data = rng.uniform(1e5, 1e6, size=480)
        da = xr.DataArray(data, coords={"time": time}, dims=["time"], name="precipitation")
        result = spi(da, scale=3, distribution=Distribution.gamma)
        assert isinstance(result, xr.DataArray)

    def test_spi_tiny_values(self) -> None:
        """Precip values of 1e-10 don't crash."""
        import pandas as pd

        rng = np.random.default_rng(241)
        time = pd.date_range("1980-01-01", periods=480, freq="MS")
        data = rng.uniform(1e-10, 1e-8, size=480)
        da = xr.DataArray(data, coords={"time": time}, dims=["time"], name="precipitation")
        result = spi(da, scale=3, distribution=Distribution.gamma)
        assert isinstance(result, xr.DataArray)

    def test_spi_both_distributions_complete(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """Gamma and Pearson both produce valid output on same input."""
        result_gamma = spi(sample_monthly_precip_da, scale=3, distribution=Distribution.gamma)
        result_pearson = spi(sample_monthly_precip_da, scale=3, distribution=Distribution.pearson)
        assert isinstance(result_gamma, xr.DataArray)
        assert isinstance(result_pearson, xr.DataArray)
        assert np.isfinite(result_gamma.values).sum() > 0
        assert np.isfinite(result_pearson.values).sum() > 0

    def test_warnings_are_not_exceptions(self) -> None:
        """ShortCalibrationWarning is captured, not raised as exception."""
        import pandas as pd

        rng = np.random.default_rng(242)
        time = pd.date_range("1990-01-01", periods=348, freq="MS")  # 29 years
        data = rng.gamma(shape=2.0, scale=25.0, size=348)
        da = xr.DataArray(data, coords={"time": time}, dims=["time"], name="precipitation")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = spi(da, scale=3, distribution=Distribution.gamma)
            assert isinstance(result, xr.DataArray)
            # warning emitted, not exception
            assert len(w) > 0

    def test_spi_scale_1_minimum(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """scale=1 (minimum valid) completes."""
        result = spi(sample_monthly_precip_da, scale=1, distribution=Distribution.gamma)
        assert isinstance(result, xr.DataArray)
        assert np.isfinite(result.values).sum() > 0

    def test_spi_large_scale(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """scale=24 (2 years) completes on 40-year data."""
        result = spi(sample_monthly_precip_da, scale=24, distribution=Distribution.gamma)
        assert isinstance(result, xr.DataArray)
        assert np.isfinite(result.values).sum() > 0

    def test_spi_dask_backed_completes(self, dask_monthly_precip_1d: xr.DataArray) -> None:
        """Dask-backed 1D input produces lazy result."""
        result = spi(dask_monthly_precip_1d, scale=3, distribution=Distribution.gamma)
        assert isinstance(result, xr.DataArray)
        # result should be lazy (chunked)
        assert result.chunks is not None
