import logging
from unittest import mock

import numpy as np
import pytest

from climate_indices import compute, indices
from climate_indices.exceptions import PeriodicityError

# disable logging messages
logging.disable(logging.CRITICAL)

UNEXPECTED_PEARSON3_MESSAGE = "Transformed Pearson Type III fitted values not computed as expected"
UNEXPECTED_SLIDING_SUMS_MESSAGE = "Sliding sums not computed as expected"


@pytest.mark.usefixtures(
    "precips_mm_monthly",
    "precips_mm_daily",
    "data_year_start_monthly",
    "data_year_end_monthly",
    "data_year_start_daily",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
    "calibration_year_start_daily",
    "calibration_year_end_daily",
    "transformed_gamma_monthly",
    "transformed_gamma_daily",
)
def test_transform_fitted_gamma(
    precips_mm_monthly,
    precips_mm_daily,
    data_year_start_monthly,
    data_year_end_monthly,
    data_year_start_daily,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
    calibration_year_start_daily,
    calibration_year_end_daily,
    transformed_gamma_monthly,
    transformed_gamma_daily,
):
    """
    Test for the compute.transform_fitted_gamma() function
    """

    # confirm that an input array of all NaNs results in the same array returned
    all_nans = np.full(precips_mm_monthly.shape, np.nan)
    computed_values = compute.transform_fitted_gamma(
        all_nans,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        compute.Periodicity.monthly,
    )
    np.testing.assert_allclose(
        computed_values,
        all_nans,
        equal_nan=True,
        err_msg="Gamma fit/transform not handling all-NaN arrays as expected",
    )

    # compute sigmas of transformed (normalized) values fitted to a gamma
    # distribution, using the full period of record as the calibration period
    computed_values = compute.transform_fitted_gamma(
        precips_mm_monthly,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        compute.Periodicity.monthly,
    )
    np.testing.assert_allclose(
        computed_values,
        transformed_gamma_monthly,
        err_msg="Transformed gamma fitted monthly values not computed as expected",
    )

    # compute sigmas of transformed (normalized) values fitted to a gamma
    # distribution, using the full period of record as the calibration period
    computed_values = compute.transform_fitted_gamma(
        precips_mm_daily.flatten(),
        data_year_start_daily,
        calibration_year_start_daily,
        calibration_year_end_daily,
        compute.Periodicity.daily,
    )

    # Check that non-NaN values in fixture match
    mask_valid_fixture = ~np.isnan(transformed_gamma_daily)
    np.testing.assert_allclose(
        computed_values[mask_valid_fixture],
        transformed_gamma_daily[mask_valid_fixture],
        atol=0.001,
        err_msg="Transformed gamma fitted daily values mismatch on valid fixture values",
    )

    # Check that values where input was zero are NOT NaN in computed result
    # and are finite (either a real number or -inf for extreme drought)
    # Note: Zero precipitation can result in positive SPI when zeros are historically
    # common for that time step (high probability of zero means zero is "normal")
    mask_zeros = precips_mm_daily == 0
    spi_for_zeros = computed_values[mask_zeros]
    assert not np.any(np.isnan(spi_for_zeros)), "Computed SPI should not be NaN for zero precipitation"
    # SPI values should be real numbers or -inf (not +inf or NaN)
    assert np.all(spi_for_zeros < np.inf), "SPI for zero precipitation should not be +infinity"

    # confirm that we can call with a calibration period out of the valid range
    # and as a result use the full period of record as the calibration period instead
    computed_values = compute.transform_fitted_gamma(
        precips_mm_monthly,
        data_year_start_monthly,
        1500,
        2500,
        compute.Periodicity.monthly,
    )
    np.testing.assert_allclose(
        computed_values,
        transformed_gamma_monthly,
        atol=0.001,
        equal_nan=True,
        err_msg=UNEXPECTED_PEARSON3_MESSAGE,
    )

    # if we provide a 1-D array then we need to provide a corresponding
    # time series type, confirm we can't use an invalid type
    flat_array = precips_mm_monthly.flatten()
    np.testing.assert_raises(
        PeriodicityError,
        compute.transform_fitted_gamma,
        flat_array,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        "invalid_value",
    )
    np.testing.assert_raises(
        PeriodicityError,
        compute.transform_fitted_gamma,
        flat_array,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        None,
    )

    # confirm that an input array which is not 1-D or 2-D will raise an error
    with pytest.raises(ValueError):
        compute.transform_fitted_gamma(
            np.zeros((9, 8, 7, 6), dtype=float),
            data_year_start_daily,
            calibration_year_start_daily,
            calibration_year_end_daily,
            compute.Periodicity.monthly,
        )


def test_transform_fitted_gamma_all_zeros_produces_finite_spi():
    """
    Test that all-zero precipitation produces finite SPI values, not NaN.

    When all precipitation values are zero, SPI should indicate extreme drought
    (negative infinity or large negative values), not NaN.
    """
    # one year of daily data (366 days)
    n_years = 1
    n_days_per_year = 366
    values = np.zeros((n_years, n_days_per_year), dtype=float)

    result = compute.transform_fitted_gamma(
        values,
        data_start_year=2000,
        calibration_start_year=2000,
        calibration_end_year=2000,
        periodicity=compute.Periodicity.daily,
    )

    # all-zero input should not produce NaN
    assert not np.any(np.isnan(result)), "SPI should not be NaN when all inputs are zero"

    # all-zero input should indicate extreme drought (negative values)
    assert np.all(result < 0), "SPI for all-zero precipitation should be negative (extreme drought)"


@pytest.mark.usefixtures(
    "precips_mm_monthly",
    "precips_mm_daily",
    "data_year_start_monthly",
    "data_year_end_monthly",
    "data_year_start_daily",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
    "calibration_year_start_daily",
    "calibration_year_end_daily",
    "gamma_monthly",
    "gamma_daily",
)
def test_gamma_parameters(
    precips_mm_monthly,
    precips_mm_daily,
    data_year_start_monthly,
    data_year_end_monthly,
    data_year_start_daily,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
    calibration_year_start_daily,
    calibration_year_end_daily,
    gamma_monthly,
    gamma_daily,
):
    """
    Test for the compute.gamma_parameters() function
    """

    # confirm that an input array of all NaNs results in the same array returned
    all_nans = np.full(precips_mm_monthly.shape, np.nan)
    nan_alphas = np.full(shape=(12,), fill_value=np.nan)
    nan_betas = np.full(shape=(12,), fill_value=np.nan)
    alphas, betas = compute.gamma_parameters(
        all_nans,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        compute.Periodicity.monthly,
    )
    assert np.allclose(alphas, nan_alphas, equal_nan=True)
    assert np.allclose(betas, nan_betas, equal_nan=True)

    computed_values = compute.gamma_parameters(
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
    )

    np.testing.assert_allclose(
        computed_values,
        gamma_monthly,
        equal_nan=True,
        err_msg="Monthly gamma fitting parameters not being computed as expected",
    )

    computed_values = compute.gamma_parameters(
        precips_mm_daily,
        data_year_start_daily,
        calibration_year_start_daily,
        calibration_year_end_daily,
        compute.Periodicity.daily,
    )

    np.testing.assert_allclose(
        computed_values,
        gamma_daily,
        equal_nan=True,
        err_msg="Daily gamma fitting parameters not being computed as expected",
    )


@pytest.mark.usefixtures(
    "precips_mm_monthly",
    "precips_mm_daily",
    "data_year_start_monthly",
    "data_year_end_monthly",
    "data_year_start_daily",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
    "calibration_year_start_daily",
    "calibration_year_end_daily",
    "transformed_pearson3",
    "transformed_pearson3_monthly_fullperiod",
)
def test_transform_fitted_pearson(
    precips_mm_monthly,
    precips_mm_daily,
    data_year_start_monthly,
    data_year_end_monthly,
    data_year_start_daily,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
    calibration_year_start_daily,
    calibration_year_end_daily,
    transformed_pearson3,
    transformed_pearson3_monthly_fullperiod,
):
    """
    Test for the compute.transform_fitted_pearson() function
    """

    # confirm that an input array of all NaNs results in the same array returned
    all_nans = np.full(precips_mm_monthly.shape, np.nan)
    computed_values = compute.transform_fitted_pearson(
        all_nans,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        compute.Periodicity.monthly,
    )
    np.testing.assert_allclose(
        computed_values,
        all_nans,
        equal_nan=True,
        err_msg="Pearson fit/transform not handling all-NaN arrays as expected",
    )

    # compute sigmas of transformed (normalized) values
    # fitted to a Pearson Type III distribution
    computed_values = compute.transform_fitted_pearson(
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
    )
    expected_values = transformed_pearson3
    np.testing.assert_allclose(
        computed_values,
        expected_values,
        atol=0.001,
        err_msg=UNEXPECTED_PEARSON3_MESSAGE,
    )

    # confirm that an input array of all NaNs will return the same array
    all_nans = np.full(precips_mm_monthly.shape, np.nan)
    computed_values = compute.transform_fitted_pearson(
        all_nans,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
    )
    np.testing.assert_allclose(
        computed_values,
        all_nans,
        equal_nan=True,
        err_msg=UNEXPECTED_PEARSON3_MESSAGE,
    )

    # confirm that we can call with a calibration period outside of valid range
    # and as a result use the full period of record as the calibration period instead
    computed_values = compute.transform_fitted_pearson(
        precips_mm_monthly,
        data_year_start_monthly,
        1500,
        2500,
        compute.Periodicity.monthly,
    )
    np.testing.assert_allclose(
        computed_values.flatten(),
        transformed_pearson3_monthly_fullperiod,
        atol=0.001,
        equal_nan=True,
        err_msg=UNEXPECTED_PEARSON3_MESSAGE,
    )

    # confirm that we can call with daily values and not raise an error
    compute.transform_fitted_pearson(
        precips_mm_daily,
        data_year_start_daily,
        calibration_year_start_daily,
        calibration_year_end_daily,
        compute.Periodicity.daily,
    )

    # confirm that we get expected errors when
    # using invalid time series type arguments
    flattened_precips = precips_mm_monthly.flatten()
    with pytest.raises(PeriodicityError, match="requires a corresponding periodicity") as missing_periodicity:
        compute.transform_fitted_pearson(
            flattened_precips,
            data_year_start_monthly,
            calibration_year_start_monthly,
            calibration_year_end_monthly,
            None,
        )
    assert missing_periodicity.value.periodicity_value == "None"

    with pytest.raises(PeriodicityError, match="Unsupported periodicity argument") as unsupported_periodicity:
        compute.transform_fitted_pearson(
            flattened_precips,
            data_year_start_monthly,
            calibration_year_start_monthly,
            calibration_year_end_monthly,
            "unsupported_type",
        )
    assert unsupported_periodicity.value.periodicity_value == "unsupported_type"

    # confirm that an input array which is not 1-D or 2-D will raise an error
    with pytest.raises(ValueError):
        compute.transform_fitted_pearson(
            np.zeros((9, 8, 7, 6), dtype=float),
            data_year_start_daily,
            calibration_year_start_daily,
            calibration_year_end_daily,
            compute.Periodicity.monthly,
        )


@pytest.mark.usefixtures(
    "precips_mm_monthly",
    "data_year_start_monthly",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
)
def test_pearson_parameters(
    precips_mm_monthly,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
):
    """
    Test for the compute._pearson3_fitting_values() function
    """
    np.testing.assert_raises(
        ValueError,
        compute.pearson_parameters,
        np.array(
            [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0], [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.7]],
        ),
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
    )
    np.testing.assert_raises(
        PeriodicityError,
        compute.pearson_parameters,
        None,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        None,
    )

    # try using a subset of the precipitation dataset (1897 - 1915, year indices 2 - 20)
    computed_values = compute.pearson_parameters(
        precips_mm_monthly[2:21, :],
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
    )
    expected_probs_of_zero = np.zeros((12,))
    expected_locs = np.array(
        [
            48.539987664499996,
            53.9852487665,
            44.284745065842102,
            62.583727384894736,
            125.72157689160528,
            182.03053042784214,
            159.00575657926319,
            170.92269736865791,
            189.8925781252895,
            155.13420024692104,
            72.953125000026319,
            43.31532689144737,
        ]
    )
    expected_scales = np.array(
        [
            33.781507724523095,
            43.572151699968387,
            40.368173442404107,
            44.05329691434887,
            60.10621716019174,
            59.343178125457186,
            49.228795303727473,
            66.775653341386999,
            65.362977393206421,
            94.467597091088265,
            72.63706898364299,
            34.250906049301463,
        ]
    )
    expected_skews = np.array(
        [
            0.76530966976335302,
            1.2461447518219784,
            2.275517179222323,
            0.8069305098698194,
            -0.6783037020197018,
            1.022194696224529,
            0.40876120732817578,
            1.2372551346168916,
            0.73881116931924118,
            0.91911763257003465,
            2.3846715887263725,
            1.4700559294571962,
        ]
    )
    np.testing.assert_allclose(
        computed_values,
        (expected_probs_of_zero, expected_locs, expected_scales, expected_skews),
        atol=0.001,
        equal_nan=True,
        err_msg="Failed to accurately compute Pearson Type III fitting values",
    )

    # add some zeros in order to exercise the parts where it gets a percentage of zeros
    precips_mm = np.array(precips_mm_monthly, copy=True)
    precips_mm[0, 1] = 0.0
    precips_mm[3, 4] = 0.0
    precips_mm[14, 9] = 0.0
    precips_mm[2, 5] = 0.0
    precips_mm[8, 3] = 0.0
    precips_mm[7, 11] = 0.0
    precips_mm[3, 9] = 0.0
    precips_mm[11, 4] = 0.0
    precips_mm[13, 5] = 0.0
    computed_values = compute.pearson_parameters(
        precips_mm,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
    )
    expected_probs_of_zero = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    expected_locs = np.array(
        [
            45.07,
            51.86,
            62.71,
            57.52,
            101.46,
            195.25,
            145.89,
            188.22,
            203.33,
            125.49,
            65.22,
            41.81,
        ]
    )
    expected_scales = np.array(
        [
            45.67,
            49.47,
            35.25,
            38.97,
            59.21,
            117.08,
            56.17,
            68.33,
            72.65,
            85.53,
            53.99,
            35.14,
        ]
    )
    expected_skews = np.array([2.28, 1.98, 0.74, 0.87, 0.21, 0.97, 1.2, 1.29, 1.06, 1.58, 1.86, 1.85])
    np.testing.assert_allclose(
        computed_values,
        (expected_probs_of_zero, expected_locs, expected_scales, expected_skews),
        atol=0.01,
        equal_nan=True,
        err_msg="Failed to accurately compute Pearson Type III fitting values",
    )


def test_sum_to_scale():
    """
    Test for the compute.sum_to_scale() function
    """

    # test an input array with no missing values
    values = np.array([3.0, 4, 6, 2, 1, 3, 5, 8, 5])
    computed_values = compute.sum_to_scale(values, 3)
    expected_values = np.array([np.nan, np.nan, 13, 12, 9, 6, 9, 16, 18])
    np.testing.assert_allclose(
        computed_values,
        expected_values,
        err_msg=UNEXPECTED_SLIDING_SUMS_MESSAGE,
    )
    computed_values = compute.sum_to_scale(values, 4)
    expected_values = np.array([np.nan, np.nan, np.nan, 15, 13, 12, 11, 17, 21])
    np.testing.assert_allclose(
        computed_values,
        expected_values,
        err_msg=UNEXPECTED_SLIDING_SUMS_MESSAGE,
    )

    # test an input array with missing values on the end
    values = np.array([3, 4, 6, 2, 1, 3, 5, 8, 5, np.nan, np.nan, np.nan])
    computed_values = compute.sum_to_scale(values, 3)
    expected_values = np.array([np.nan, np.nan, 13, 12, 9, 6, 9, 16, 18, np.nan, np.nan, np.nan])
    np.testing.assert_allclose(
        computed_values,
        expected_values,
        err_msg="Sliding sums not computed as expected when missing values appended to end of input array",
    )

    # test an input array with missing values within the array
    values = np.array([3, 4, 6, 2, 1, 3, 5, np.nan, 8, 5, 6])
    computed_values = compute.sum_to_scale(values, 3)
    expected_values = np.array([np.nan, np.nan, 13, 12, 9, 6, 9, np.nan, np.nan, np.nan, 19])
    np.testing.assert_allclose(
        computed_values,
        expected_values,
        err_msg="Sliding sums not computed as expected when missing values appended to end of input array",
    )

    test_values = np.array([1.0, 5, 7, 2, 3, 4, 9, 6, 3, 8])
    sum_by2 = np.array([np.nan, 6, 12, 9, 5, 7, 13, 15, 9, 11])
    sum_by4 = np.array([np.nan, np.nan, np.nan, 15, 17, 16, 18, 22, 22, 26])
    sum_by6 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 22, 30, 31, 27, 33])
    np.testing.assert_equal(
        compute.sum_to_scale(test_values, 2),
        sum_by2,
        err_msg=UNEXPECTED_SLIDING_SUMS_MESSAGE,
    )
    np.testing.assert_equal(
        compute.sum_to_scale(test_values, 4),
        sum_by4,
        err_msg=UNEXPECTED_SLIDING_SUMS_MESSAGE,
    )
    np.testing.assert_equal(
        compute.sum_to_scale(test_values, 6),
        sum_by6,
        err_msg=UNEXPECTED_SLIDING_SUMS_MESSAGE,
    )


def test_periodicity_period_length():
    """
    Each periodicity reports the number of time steps in one year of data.
    """
    assert compute.Periodicity.monthly.period_length == 12
    assert compute.Periodicity.daily.period_length == 366


def test_prepare_scaled_flattens_clips_and_reshapes():
    """
    2-D input is flattened before summing, negatives are clipped, and the result is reshaped.
    """
    values = np.arange(24, dtype=float).reshape(2, 12)
    values[0, 0] = -5.0

    computed = compute.prepare_scaled(values, 3, compute.Periodicity.monthly)

    assert computed.shape == (2, 12)
    # the sum crosses the year boundary, i.e. the input was flattened before scaling
    assert computed[1, 0] == 10.0 + 11.0 + 12.0
    # the negative value was clipped to zero rather than summed
    assert computed[0, 2] == 0.0 + 1.0 + 2.0
    np.testing.assert_array_equal(
        computed,
        compute.prepare_scaled(values.flatten(), 3, compute.Periodicity.monthly),
    )

    # clipping is optional, and nothing else about the preparation changes
    unclipped = compute.prepare_scaled(values, 3, compute.Periodicity.monthly, clip_negatives=False)
    assert unclipped[0, 2] == -5.0 + 1.0 + 2.0


def test_prepare_scaled_sums_over_the_scale():
    """
    The scale sums each time step, and the reshape to (years, periods) is optional.
    """
    values = np.arange(1.0, 25.0)

    unreshaped = compute.prepare_scaled(values, 3, compute.Periodicity.monthly, reshape=False)
    np.testing.assert_array_equal(unreshaped, compute.sum_to_scale(values, 3))
    assert unreshaped.shape == (24,)

    reshaped = compute.prepare_scaled(values, 3, compute.Periodicity.monthly)
    assert reshaped.shape == (2, 12)


def test_prepare_scaled_fills_masked_values_with_nan():
    """
    Masked values are missing values, so they come back as NaN rather than as raw data.
    """
    values = np.ma.array(np.arange(24, dtype=float), mask=False)
    values.mask[3] = True

    # scale == 1 returns the still-masked values, which the seam makes explicit
    computed = compute.prepare_scaled(values, 1, compute.Periodicity.monthly)

    assert not np.ma.isMaskedArray(computed)
    assert computed.shape == (2, 12)
    assert np.isnan(computed[0, 3])
    assert computed[0, 4] == 4.0


def test_scale_values_delegates_to_prepare_scaled():
    """
    The public scaling wrapper keeps its contract by delegating to the shared seam.
    """
    values = np.array([-2.0, 3.0, 4.0, 5.0] * 6).reshape(2, 12)

    scaled = compute.scale_values(values, 3, compute.Periodicity.monthly)

    assert scaled.shape == (2, 12)
    assert scaled[0, 2] == 0.0 + 3.0 + 4.0  # the negative value was clipped before summing
    np.testing.assert_array_equal(
        scaled,
        compute.prepare_scaled(values, 3, compute.Periodicity.monthly),
    )


def test_prepare_scaled_returns_all_missing_input_unreshaped():
    """
    All-missing input is handed back flattened and un-reshaped so that callers can short-circuit.
    """
    computed = compute.prepare_scaled(np.full((2, 12), np.nan), 3, compute.Periodicity.monthly)
    assert computed.ndim == 1
    assert np.all(np.isnan(computed))

    computed_masked = compute.prepare_scaled(
        np.ma.array(np.zeros((2, 12)), mask=True),
        3,
        compute.Periodicity.monthly,
    )
    assert np.ma.isMaskedArray(computed_masked)
    assert computed_masked.ndim == 1
    assert computed_masked.mask.all()


def test_prepare_scaled_rejects_unsupported_shapes():
    """
    Input with no time axis, and ambiguous spatial input, raise a ValueError.

    Three or more dimensions are read as a time-major (time, *cells) block, except when
    the first cell axis is itself the period length: that shape is equally readable as a
    (years, periods, *cells) array, so it has to be declared (#923).
    """
    with pytest.raises(ValueError, match="Invalid shape of input array"):
        compute.prepare_scaled(np.array(0.0), 1, compute.Periodicity.monthly)

    with pytest.raises(ValueError, match="ambiguous"):
        compute.prepare_scaled(np.zeros((24, 12, 2)), 1, compute.Periodicity.monthly)

    # unambiguous spatial input is read as time-major (time, *cells) without a declaration
    spatial = compute.prepare_scaled(np.zeros((24, 2, 2)), 1, compute.Periodicity.monthly)
    assert spatial.shape == (2, 12, 2, 2)

    # and the ambiguous shape folds the same way once it is declared
    declared = compute.prepare_scaled(
        np.zeros((24, 12, 2)),
        1,
        compute.Periodicity.monthly,
        spatial_time_major=True,
    )
    assert declared.shape == (2, 12, 12, 2)


def test_prepare_scaled_rejects_unsupported_periodicity_when_unreshaped():
    """
    An invalid periodicity must be rejected even with reshape=False, since
    reshape_values() -- the only other periodicity check -- is skipped in that case.
    """
    with pytest.raises(PeriodicityError, match="Invalid periodicity argument") as error:
        compute.prepare_scaled(np.arange(12, dtype=float), 3, "monthly", reshape=False)

    assert error.value.periodicity_value == "monthly"


def test_gamma_parameters_all_missing_rejects_invalid_periodicity():
    """
    The all-missing early return must not skip periodicity validation, which is
    the only check standing between the caller and a silently reshape-free result.
    """
    with pytest.raises(PeriodicityError, match="Unsupported periodicity") as error:
        compute.gamma_parameters(np.full((2, 12), np.nan), 2000, 2000, 2001, None)

    assert error.value.periodicity_value == "None"


def test_reshape_values_rejects_unsupported_periodicity():
    """The shared reshape helper rejects a periodicity it cannot reshape to."""
    with pytest.raises(PeriodicityError, match="Invalid periodicity argument") as error:
        compute.reshape_values(np.arange(24, dtype=float), "monthly")

    assert error.value.periodicity_value == "monthly"


def test_prepare_scaled_clips_negatives_alongside_missing_values():
    """
    A negative value must be clipped even when the array also contains unmasked NaN
    or masked entries, since np.amin/np.nanmin either miss the negative (a NaN in the
    array makes np.amin return NaN, so `NaN < 0.0` is False) or reach under the mask.
    """
    # scale == 1 short-circuits sum_to_scale, so the negative reaches the output
    # unmodified if it isn't clipped -- this is the path the bug hid on
    values = np.array([-3.0, np.nan, 2.0])
    computed = compute.prepare_scaled(values, 1, compute.Periodicity.monthly, reshape=False)
    assert computed[0] == 0.0
    assert np.isnan(computed[1])

    # masked entries must not be mistaken for the negative, or reported as clipped
    masked = np.ma.array([-3.0, 5.0, 2.0], mask=[False, True, False])
    computed_masked = compute.prepare_scaled(masked, 1, compute.Periodicity.monthly, reshape=False)
    assert computed_masked[0] == 0.0
    assert np.isnan(computed_masked[1])

    # the summed path (scale > 1) must also clip before summing, with a NaN elsewhere
    # in the array (np.amin would return NaN here, hiding the negative under the bug)
    with_nan = np.array([-3.0, 4.0, np.nan, 2.0])
    summed = compute.prepare_scaled(with_nan, 2, compute.Periodicity.monthly, reshape=False)
    assert summed[1] == 0.0 + 4.0


def test_fit_and_standardize_dispatches_on_distribution():
    """
    Each distribution is fitted and transformed by its own transform, with no
    parameters to normalize first.
    """
    values = np.arange(1.0, 121.0).reshape(10, 12)

    gamma = compute.fit_and_standardize(
        values, indices.Distribution.gamma, 2000, 2000, 2009, compute.Periodicity.monthly
    )
    np.testing.assert_array_equal(
        gamma,
        compute.transform_fitted_gamma(values, 2000, 2000, 2009, compute.Periodicity.monthly),
    )

    pearson = compute.fit_and_standardize(
        values, indices.Distribution.pearson, 2000, 2000, 2009, compute.Periodicity.monthly
    )
    np.testing.assert_array_equal(
        pearson,
        compute.transform_fitted_pearson(values, 2000, 2000, 2009, compute.Periodicity.monthly),
    )


def test_fit_and_standardize_normalizes_fitting_parameter_keys():
    """
    Fitting parameters are normalized: the canonical spellings and the deprecated
    aliases that mean the same thing are both used, an explicit None for a canonical
    key means "fit this parameter from the data", and a canonical key that is None
    defers to a deprecated alias that carries a value.
    """
    values = np.arange(1.0, 121.0).reshape(10, 12)
    alphas = np.full(12, 4.0)
    betas = np.full(12, 8.0)
    with_supplied = compute.transform_fitted_gamma(values, 2000, 2000, 2009, compute.Periodicity.monthly, alphas, betas)
    from_the_data = compute.transform_fitted_gamma(values, 2000, 2000, 2009, compute.Periodicity.monthly)

    # the supplied parameters are distinguishable from the fitted ones, so a seam that
    # ignored fitting_params would fail the assertions below
    assert not np.array_equal(with_supplied, from_the_data)

    parameters = (
        {"alpha": alphas, "beta": betas},
        {"alphas": alphas, "betas": betas},
        {"alpha": None, "alphas": alphas, "beta": None, "betas": betas},
    )
    for fitting_params in parameters:
        computed = compute.fit_and_standardize(
            values,
            indices.Distribution.gamma,
            2000,
            2000,
            2009,
            compute.Periodicity.monthly,
            fitting_params,
        )
        np.testing.assert_array_equal(computed, with_supplied)

    explicit_none = compute.fit_and_standardize(
        values,
        indices.Distribution.gamma,
        2000,
        2000,
        2009,
        compute.Periodicity.monthly,
        {"alpha": None, "beta": None},
    )
    np.testing.assert_array_equal(explicit_none, from_the_data)

    # a key absent from a partial parameter set reads as None rather than raising KeyError:
    # gamma refits from the data, and Pearson Type III still reports its incomplete set
    partial_gamma = compute.fit_and_standardize(
        values,
        indices.Distribution.gamma,
        2000,
        2000,
        2009,
        compute.Periodicity.monthly,
        {"alpha": alphas},
    )
    np.testing.assert_array_equal(partial_gamma, from_the_data)

    with pytest.raises(ValueError, match="either none or all"):
        compute.fit_and_standardize(
            values,
            indices.Distribution.pearson,
            2000,
            2000,
            2009,
            compute.Periodicity.monthly,
            {"prob_zero": np.full(12, 0.1)},
        )


def test_fit_and_standardize_falls_back_to_gamma_only_when_asked():
    """
    A failed Pearson Type III fit falls back to gamma when the caller asked for the
    fall back, and propagates the failure to the caller when it did not.
    """
    values = np.arange(1.0, 121.0).reshape(10, 12)
    failed_pearson = mock.patch(
        "climate_indices.compute.transform_fitted_pearson",
        side_effect=compute.DistributionFittingError("Pearson failed", distribution_name="pearson3"),
    )

    with failed_pearson:
        with pytest.raises(compute.DistributionFittingError):
            compute.fit_and_standardize(
                values, indices.Distribution.pearson, 2000, 2000, 2009, compute.Periodicity.monthly
            )

    with failed_pearson:
        fell_back = compute.fit_and_standardize(
            values,
            indices.Distribution.pearson,
            2000,
            2000,
            2009,
            compute.Periodicity.monthly,
            fallback_to_gamma=True,
        )

    # a failed fit leaves the scaled values in place, so the fall back fits gamma to those
    np.testing.assert_array_equal(
        fell_back,
        compute.transform_fitted_gamma(values, 2000, 2000, 2009, compute.Periodicity.monthly),
    )


def test_fit_and_standardize_falls_back_when_pearson_leaves_excessive_nans():
    """
    A Pearson Type III result that is mostly missing counts as a fitting failure and
    falls back to gamma.
    """
    values = np.arange(1.0, 121.0).reshape(10, 12)

    with mock.patch("climate_indices.compute.transform_fitted_pearson", return_value=np.full(values.shape, np.nan)):
        with mock.patch("climate_indices.compute.transform_fitted_gamma", return_value=np.ones(values.shape)) as gamma:
            computed = compute.fit_and_standardize(
                values,
                indices.Distribution.pearson,
                2000,
                2000,
                2009,
                compute.Periodicity.monthly,
                fallback_to_gamma=True,
            )

    assert gamma.call_count == 1
    np.testing.assert_array_equal(computed, np.ones(values.shape))
