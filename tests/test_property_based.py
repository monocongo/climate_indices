"""Property-based tests using Hypothesis for climate indices.

This module implements property-based testing for climate indices to verify
mathematical invariants and properties that should hold across a wide range
of input data. Tests use Hypothesis to generate random valid climate data
and check that computed indices satisfy fundamental constraints.

Satisfies FR-TEST-005 (Property-Based Testing).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

# load logging suppression fixture for Hypothesis tests
pytest_plugins = ["tests.helpers.logging"]

from climate_indices import compute, eto, indices, palmer
from climate_indices.exceptions import GoodnessOfFitWarning, MissingDataWarning, ShortCalibrationWarning

# import shared Hypothesis strategies
from tests.helpers.strategies import (
    daily_temperature_triplet,
    monthly_precipitation_array,
    monthly_temperature_array,
    precip_with_uniform_offset,
    valid_latitude,
    valid_scale,
)

# ============================================================================
# Group A: Boundedness Tests
# ============================================================================


@pytest.mark.parametrize("distribution", [indices.Distribution.gamma, indices.Distribution.pearson])
@given(precip=monthly_precipitation_array(), scale=valid_scale())
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_spi_boundedness(precip: np.ndarray, scale: int, distribution: indices.Distribution) -> None:
    """Verify SPI values fall within valid range [-3.09, 3.09].

    Property: For standardized indices fitted to normal distribution,
    all non-NaN values should fall within approximately ±3.09 standard
    deviations (this captures ~99.9% of values).
    """
    # suppress expected warnings for random data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ShortCalibrationWarning)
        warnings.filterwarnings("ignore", category=GoodnessOfFitWarning)
        warnings.filterwarnings("ignore", category=MissingDataWarning)

        result = indices.spi(
            precip,
            scale=scale,
            distribution=distribution,
            periodicity=compute.Periodicity.monthly,
            data_start_year=1950,
            calibration_year_initial=1950,
            calibration_year_final=1950 + len(precip) // 12 - 1,
        )

    # check non-NaN values are bounded
    valid_values = result[~np.isnan(result)]
    if len(valid_values) > 0:
        assert np.all(valid_values >= indices._FITTED_INDEX_VALID_MIN), "SPI values below valid minimum"
        assert np.all(valid_values <= indices._FITTED_INDEX_VALID_MAX), "SPI values above valid maximum"


@given(precip=monthly_precipitation_array(), pet_array=monthly_precipitation_array(), scale=valid_scale())
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_spei_boundedness(precip: np.ndarray, pet_array: np.ndarray, scale: int) -> None:
    """Verify SPEI values fall within valid range [-3.09, 3.09].

    Property: SPEI, like SPI, is fitted to normal distribution and should
    have all non-NaN values within ±3.09 standard deviations.
    """
    # ensure arrays are same length
    min_length = min(len(precip), len(pet_array))
    precip = precip[:min_length]
    pet_array = pet_array[:min_length]

    # suppress expected warnings for random data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ShortCalibrationWarning)
        warnings.filterwarnings("ignore", category=GoodnessOfFitWarning)
        warnings.filterwarnings("ignore", category=MissingDataWarning)

        result = indices.spei(
            precip,
            pet_array,
            scale=scale,
            distribution=indices.Distribution.gamma,
            periodicity=compute.Periodicity.monthly,
            data_start_year=1950,
            calibration_year_initial=1950,
            calibration_year_final=1950 + min_length // 12 - 1,
        )

    # check non-NaN values are bounded
    valid_values = result[~np.isnan(result)]
    if len(valid_values) > 0:
        assert np.all(valid_values >= indices._FITTED_INDEX_VALID_MIN), "SPEI values below valid minimum"
        assert np.all(valid_values <= indices._FITTED_INDEX_VALID_MAX), "SPEI values above valid maximum"


@given(temperature=monthly_temperature_array(), latitude=valid_latitude())
@settings(max_examples=30, deadline=None)
def test_pet_thornthwaite_non_negative(temperature: np.ndarray, latitude: float) -> None:
    """Verify PET (Thornthwaite) is always non-negative.

    Property: Potential evapotranspiration represents energy available
    for evaporation and must be >= 0 by physical definition.
    """
    result = eto.eto_thornthwaite(temperature, latitude, data_start_year=1950)

    # check non-NaN values are non-negative
    valid_values = result[~np.isnan(result)]
    if len(valid_values) > 0:
        assert np.all(valid_values >= 0.0), "PET Thornthwaite contains negative values"


@given(temps=daily_temperature_triplet(), latitude=valid_latitude())
@settings(max_examples=30, deadline=None)
def test_pet_hargreaves_non_negative(temps: tuple[np.ndarray, np.ndarray, np.ndarray], latitude: float) -> None:
    """Verify PET (Hargreaves) is non-negative for valid inputs.

    Property: PET is always >= 0 when tmean + 17.8 > 0 (valid input range).
    """
    tmin, tmax, tmean = temps

    # hargreaves requires tmean + 17.8 > 0
    valid_indices = (tmean + 17.8) > 0
    if not np.any(valid_indices):
        # skip if no valid values
        return

    result = eto.eto_hargreaves(tmin, tmax, tmean, latitude)

    # check non-NaN values at valid indices are non-negative
    valid_values = result[valid_indices & ~np.isnan(result)]
    if len(valid_values) > 0:
        assert np.all(valid_values >= 0.0), "PET Hargreaves contains negative values at valid indices"


# ============================================================================
# Group B: Monotonicity Tests
# ============================================================================


@given(precip_pair=precip_with_uniform_offset(), scale=valid_scale())
@settings(max_examples=50, deadline=None)
def test_sum_to_scale_monotonicity(precip_pair: tuple[np.ndarray, np.ndarray], scale: int) -> None:
    """Verify sum_to_scale preserves monotonicity.

    Property: If a[i] >= b[i] for all i, then sum(a)[i] >= sum(b)[i] for all i.
    """
    lower, higher = precip_pair

    # ensure arrays are same length
    min_length = min(len(lower), len(higher))
    lower = lower[:min_length]
    higher = higher[:min_length]

    # compute sliding sums
    sum_lower = compute.sum_to_scale(lower, scale)
    sum_higher = compute.sum_to_scale(higher, scale)

    # check monotonicity is preserved
    valid_indices = ~(np.isnan(sum_lower) | np.isnan(sum_higher))
    if np.any(valid_indices):
        assert np.all(sum_higher[valid_indices] >= sum_lower[valid_indices]), (
            "sum_to_scale violated monotonicity property"
        )


@given(
    v1=st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
    v2=st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
    alpha=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    beta=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, deadline=None)
def test_gamma_cdf_monotonicity(v1: float, v2: float, alpha: float, beta: float) -> None:
    """Verify gamma CDF is monotonically increasing.

    Property: For CDF F(x), if x1 < x2 then F(x1) <= F(x2).
    This is fundamental to the SPI gamma transformation.
    """
    # import scipy for gamma CDF
    from scipy import stats

    # ensure v1 < v2
    if v1 > v2:
        v1, v2 = v2, v1
    elif v1 == v2:
        # skip equal values
        return

    # compute CDFs
    cdf1 = stats.gamma.cdf(v1, a=alpha, scale=beta)
    cdf2 = stats.gamma.cdf(v2, a=alpha, scale=beta)

    assert cdf1 <= cdf2, f"Gamma CDF not monotonic: cdf({v1}) = {cdf1}, cdf({v2}) = {cdf2}"


@given(temp_base=monthly_temperature_array())
@settings(max_examples=20, deadline=None)
def test_pet_thornthwaite_increases_with_temperature(temp_base: np.ndarray) -> None:
    """Verify PET increases with higher temperatures.

    Property: Higher temperatures should result in higher total annual PET.
    Tests annual sum rather than pointwise to account for non-linear effects.

    Note: Thornthwaite clips negative temps to 0°C before calculating the heat
    index, which can create non-monotonic behavior when comparing very cold base
    temps vs. warmer temps. We filter out cases with significant sub-freezing
    temps to avoid this edge case.
    """
    # skip test cases where more than 25% of months are below freezing
    # as the temperature clipping creates non-monotonic heat index behavior
    fraction_below_freezing = np.sum(temp_base < 0) / len(temp_base)
    assume(fraction_below_freezing < 0.25)

    # create higher temperature array
    temp_higher = temp_base + 5.0

    # use fixed latitude for comparison
    latitude = 40.0

    pet_base = eto.eto_thornthwaite(temp_base, latitude, data_start_year=1950)
    pet_higher = eto.eto_thornthwaite(temp_higher, latitude, data_start_year=1950)

    # compare annual totals (sum over complete years only)
    num_complete_years = len(pet_base) // 12
    if num_complete_years > 0:
        # reshape to (years, 12) and sum over months
        pet_base_annual = pet_base[: num_complete_years * 12].reshape(-1, 12).sum(axis=1)
        pet_higher_annual = pet_higher[: num_complete_years * 12].reshape(-1, 12).sum(axis=1)

        # check that higher temperature produces higher annual PET
        # use strict inequality since we added 5 degrees everywhere
        assert np.all(pet_higher_annual > pet_base_annual), (
            "Higher temperature did not produce higher annual PET (Thornthwaite)"
        )


# ============================================================================
# Group C: Symmetry Tests
# ============================================================================


@given(scale=valid_scale())
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_spi_mean_approximately_zero(scale: int) -> None:
    """Verify SPI mean is approximately zero for full-record calibration.

    Property: When calibration period equals data period, SPI should
    have mean near zero (standardized to N(0,1)). Use gamma-distributed
    input for better fit quality.
    """
    # generate gamma-distributed precipitation (fits gamma model well)
    num_years = 40
    rng = np.random.default_rng(12345)
    precip = rng.gamma(shape=2.0, scale=50.0, size=num_years * 12)

    # suppress expected warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ShortCalibrationWarning)
        warnings.filterwarnings("ignore", category=GoodnessOfFitWarning)
        warnings.filterwarnings("ignore", category=MissingDataWarning)

        result = indices.spi(
            precip,
            scale=scale,
            distribution=indices.Distribution.gamma,
            periodicity=compute.Periodicity.monthly,
            data_start_year=1950,
            calibration_year_initial=1950,
            calibration_year_final=1950 + num_years - 1,
        )

    # check mean is approximately zero (generous tolerance for random data)
    valid_values = result[~np.isnan(result)]
    if len(valid_values) > 10:
        mean_spi = np.mean(valid_values)
        assert abs(mean_spi) < 0.5, f"SPI mean {mean_spi} is too far from zero"


@given(precip=monthly_precipitation_array(), scale=valid_scale())
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_spi_output_shape_matches_input(precip: np.ndarray, scale: int) -> None:
    """Verify SPI output shape matches input shape.

    Property: Shape preservation - output array should have same length as input.
    """
    # suppress expected warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ShortCalibrationWarning)
        warnings.filterwarnings("ignore", category=GoodnessOfFitWarning)
        warnings.filterwarnings("ignore", category=MissingDataWarning)

        result = indices.spi(
            precip,
            scale=scale,
            distribution=indices.Distribution.gamma,
            periodicity=compute.Periodicity.monthly,
            data_start_year=1950,
            calibration_year_initial=1950,
            calibration_year_final=1950 + len(precip) // 12 - 1,
        )

    assert result.shape == precip.shape, f"Shape mismatch: input {precip.shape}, output {result.shape}"


# ============================================================================
# Group D: PDSI Boundedness
# ============================================================================


@pytest.mark.slow
@given(
    precip=monthly_precipitation_array(num_years=10),  # use fewer years for slow PDSI
    pet_array=monthly_precipitation_array(num_years=10),
    awc=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_pdsi_bounded_range(precip: np.ndarray, pet_array: np.ndarray, awc: float) -> None:
    """Verify PDSI falls within expected range [-12, 12].

    Property: Palmer Drought Severity Index typically ranges from
    approximately -10 (extreme drought) to +10 (extreme wetness),
    though values outside this range are theoretically possible.
    Use [-12, 12] as conservative bounds.
    """
    # ensure arrays are same length
    min_length = min(len(precip), len(pet_array))
    precip = precip[:min_length]
    pet_array = pet_array[:min_length]

    # pdsi expects inches - convert from mm if needed (assume input is mm, divide by 25.4)
    precip_inches = precip / 25.4
    pet_inches = pet_array / 25.4

    # suppress expected warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        pdsi_values, phdi, pmdi, zindex, fitting_params = palmer.pdsi(
            precip_inches,
            pet_inches,
            awc,
            data_start_year=1950,
            calibration_year_initial=1950,
            calibration_year_final=1950 + min_length // 12 - 1,
        )

    # check non-NaN values are within expected range
    valid_values = pdsi_values[~np.isnan(pdsi_values)]
    if len(valid_values) > 0:
        # filter out infinite values that may occur with edge-case data
        finite_values = valid_values[np.isfinite(valid_values)]
        if len(finite_values) > 0:
            assert np.all(finite_values >= -12.0), "PDSI values below expected minimum -12"
            assert np.all(finite_values <= 12.0), "PDSI values above expected maximum 12"


# ============================================================================
# Group E: Edge Cases
# ============================================================================


@given(length=st.integers(min_value=360, max_value=600), scale=valid_scale())
@settings(max_examples=20, deadline=None)
def test_spi_all_nan_returns_all_nan(length: int, scale: int) -> None:
    """Verify SPI returns all NaN when input is all NaN.

    Property: All-NaN input should produce all-NaN output (early return path).
    """
    precip = np.full(length, np.nan)

    # suppress expected warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ShortCalibrationWarning)
        warnings.filterwarnings("ignore", category=GoodnessOfFitWarning)
        warnings.filterwarnings("ignore", category=MissingDataWarning)

        result = indices.spi(
            precip,
            scale=scale,
            distribution=indices.Distribution.gamma,
            periodicity=compute.Periodicity.monthly,
            data_start_year=1950,
            calibration_year_initial=1950,
            calibration_year_final=1950 + length // 12 - 1,
        )

    assert np.all(np.isnan(result)), "All-NaN input did not produce all-NaN output"


@given(length=st.integers(min_value=360, max_value=600), scale=valid_scale())
@settings(max_examples=10, deadline=None)
def test_spi_all_zeros_does_not_crash(length: int, scale: int) -> None:
    """Verify SPI handles all-zero input without crashing.

    Property: All-zero input may return NaN or -inf, but should not raise exception.
    """
    precip = np.zeros(length)

    # suppress expected warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ShortCalibrationWarning)
        warnings.filterwarnings("ignore", category=GoodnessOfFitWarning)
        warnings.filterwarnings("ignore", category=MissingDataWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        try:
            result = indices.spi(
                precip,
                scale=scale,
                distribution=indices.Distribution.gamma,
                periodicity=compute.Periodicity.monthly,
                data_start_year=1950,
                calibration_year_initial=1950,
                calibration_year_final=1950 + length // 12 - 1,
            )
            # if we get here, no exception was raised - good!
            assert result is not None, "SPI returned None for all-zero input"
        except Exception as e:
            pytest.fail(f"SPI crashed on all-zero input: {e}")
