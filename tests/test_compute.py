import logging

import numpy as np
import pytest

from climate_indices import compute

# ------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)


# ------------------------------------------------------------------------------
@pytest.mark.usefixtures("precips_mm_monthly",
    "precips_mm_daily",
    "data_year_start_monthly",
    "data_year_end_monthly",
    "data_year_start_daily",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
    "calibration_year_start_daily",
    "calibration_year_end_daily",
    "transformed_gamma_monthly",
    "transformed_gamma_daily")
def test_transform_fitted_gamma(precips_mm_monthly,
        precips_mm_daily,
        data_year_start_monthly,
        data_year_end_monthly,
        data_year_start_daily,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        calibration_year_start_daily,
        calibration_year_end_daily,
        transformed_gamma_monthly,
        transformed_gamma_daily):
    """
    Test for the compute.transform_fitted_gamma() function
    """

    # confirm that an input array of all NaNs results in the same array returned
    all_nans = np.full(precips_mm_monthly.shape, np.NaN)
    computed_values = compute.transform_fitted_gamma(all_nans,
                                                     data_year_start_monthly,
                                                     data_year_start_monthly,
                                                     data_year_end_monthly,
                                                     compute.Periodicity.monthly)
    np.testing.assert_allclose(computed_values,
                               all_nans,
                               equal_nan=True,
                               err_msg="Gamma fit/transform not handling "
                                       "all-NaN arrays as expected")

    # compute sigmas of transformed (normalized) values fitted to a gamma
    # distribution, using the full period of record as the calibration period
    computed_values = compute.transform_fitted_gamma(precips_mm_monthly,
                                                     data_year_start_monthly,
                                                     data_year_start_monthly,
                                                     data_year_end_monthly,
                                                     compute.Periodicity.monthly)
    np.testing.assert_allclose(computed_values,
                               transformed_gamma_monthly,
                               err_msg="Transformed gamma fitted monthly "
                                       "values not computed as expected")

    # compute sigmas of transformed (normalized) values fitted to a gamma
    # distribution, using the full period of record as the calibration period
    computed_values = compute.transform_fitted_gamma(precips_mm_daily.flatten(),
                                                     data_year_start_daily,
                                                     calibration_year_start_daily,
                                                     calibration_year_end_daily,
                                                     compute.Periodicity.daily)
    np.testing.assert_allclose(computed_values,
                               transformed_gamma_daily,
                               atol=0.001,
                               equal_nan=True,
                               err_msg="Transformed gamma fitted daily "
                                       "values not computed as expected")

    # confirm that we can call with a calibration period outside of valid range
    # and as a result use the full period of record as the calibration period instead
    computed_values = compute.transform_fitted_gamma(precips_mm_monthly,
                                                     data_year_start_monthly,
                                                     1500,
                                                     2500,
                                                     compute.Periodicity.monthly)
    np.testing.assert_allclose(computed_values,
                               transformed_gamma_monthly,
                               atol=0.001,
                               equal_nan=True,
                               err_msg="Transformed Pearson Type III "
                                       "fitted values not computed as expected")

    # if we provide a 1-D array then we need to provide a corresponding
    # time series type, confirm we can't use an invalid type
    flat_array = precips_mm_monthly.flatten()
    np.testing.assert_raises(ValueError,
                             compute.transform_fitted_gamma,
                             flat_array,
                             data_year_start_monthly,
                             calibration_year_start_monthly,
                             calibration_year_end_monthly,
                             "invalid_value")
    np.testing.assert_raises(ValueError,
                             compute.transform_fitted_gamma,
                             flat_array,
                             data_year_start_monthly,
                             calibration_year_start_monthly,
                             calibration_year_end_monthly,
                             None)

    # confirm that an input array which is not 1-D or 2-D will raise an error
    pytest.raises(ValueError,
                  compute.transform_fitted_gamma,
                  np.zeros((9, 8, 7, 6), dtype=float),
                  data_year_start_daily,
                  calibration_year_start_daily,
                  calibration_year_end_daily,
                  compute.Periodicity.monthly)


# ------------------------------------------------------------------------------
@pytest.mark.usefixtures("precips_mm_monthly",
                         "precips_mm_daily",
                         "data_year_start_monthly",
                         "data_year_end_monthly",
                         "data_year_start_daily",
                         "calibration_year_start_monthly",
                         "calibration_year_end_monthly",
                         "calibration_year_start_daily",
                         "calibration_year_end_daily",
                         "transformed_pearson3",
                         "transformed_pearson3_monthly_fullperiod")
def test_transform_fitted_pearson(precips_mm_monthly,
                                  precips_mm_daily,
                                  data_year_start_monthly,
                                  data_year_end_monthly,
                                  data_year_start_daily,
                                  calibration_year_start_monthly,
                                  calibration_year_end_monthly,
                                  calibration_year_start_daily,
                                  calibration_year_end_daily,
                                  transformed_pearson3,
                                  transformed_pearson3_monthly_fullperiod):
    """
    Test for the compute.transform_fitted_pearson() function
    """

    # confirm that an input array of all NaNs results in the same array returned
    all_nans = np.full(precips_mm_monthly.shape, np.NaN)
    computed_values = compute.transform_fitted_pearson(all_nans,
                                                       data_year_start_monthly,
                                                       data_year_start_monthly,
                                                       data_year_end_monthly,
                                                       compute.Periodicity.monthly)
    np.testing.assert_allclose(
        computed_values,
        all_nans,
        equal_nan=True,
        err_msg="Pearson fit/transform not handling all-NaN arrays as expected")

    # compute sigmas of transformed (normalized) values
    # fitted to a Pearson Type III distribution
    computed_values = compute.transform_fitted_pearson(precips_mm_monthly,
                                                       data_year_start_monthly,
                                                       calibration_year_start_monthly,
                                                       calibration_year_end_monthly,
                                                       compute.Periodicity.monthly)
    expected_values = transformed_pearson3
    np.testing.assert_allclose(
        computed_values,
        expected_values,
        atol=0.001,
        err_msg="Transformed Pearson Type III fitted "
                "values not computed as expected")

    # confirm that an input array of all NaNs will return the same array
    all_nans = np.full(precips_mm_monthly.shape, np.NaN)
    computed_values = compute.transform_fitted_pearson(
        all_nans,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly)
    np.testing.assert_allclose(
        computed_values,
        all_nans,
        equal_nan=True,
        err_msg="Transformed Pearson Type III fitted "
                "values not computed as expected")

    # confirm that we can call with a calibration period outside of valid range
    # and as a result use the full period of record as the calibration period instead
    computed_values = compute.transform_fitted_pearson(precips_mm_monthly,
                                                       data_year_start_monthly,
                                                       1500,
                                                       2500,
                                                       compute.Periodicity.monthly)
    np.testing.assert_allclose(computed_values.flatten(),
                               transformed_pearson3_monthly_fullperiod,
                               atol=0.001,
                               equal_nan=True,
                               err_msg="Transformed Pearson Type III fitted "
                                       "values not computed as expected")

    # confirm that we can call with daily values and not raise an error
    compute.transform_fitted_pearson(precips_mm_daily,
                                     data_year_start_daily,
                                     calibration_year_start_daily,
                                     calibration_year_end_daily,
                                     compute.Periodicity.daily)

    # confirm that we get expected errors when
    # using invalid time series type arguments
    pytest.raises(ValueError,
                  compute.transform_fitted_pearson,
                  precips_mm_monthly.flatten(),
                  data_year_start_monthly,
                  calibration_year_start_monthly,
                  calibration_year_end_monthly,
                  None)
    pytest.raises(ValueError,
                  compute.transform_fitted_pearson,
                  precips_mm_monthly.flatten(),
                  data_year_start_monthly,
                  calibration_year_start_monthly,
                  calibration_year_end_monthly,
                  "unsupported_type")

    # confirm that an input array which is not 1-D or 2-D will raise an error
    pytest.raises(ValueError,
                  compute.transform_fitted_pearson,
                  np.zeros((9, 8, 7, 6), dtype=float),
                  data_year_start_daily,
                  calibration_year_start_daily,
                  calibration_year_end_daily,
                  compute.Periodicity.monthly)


# ------------------------------------------------------------------------------
@pytest.mark.usefixtures("precips_mm_monthly")
def test_pearson3_fitting_values(precips_mm_monthly):
    """
    Test for the compute._pearson3_fitting_values() function
    """
    # provide some bogus inputs to make sure these raise expected errors
    np.testing.assert_raises(ValueError,
                             compute._pearson3_fitting_values,
                             np.array([1.0, 0.0, 0.0]))
    np.testing.assert_raises(ValueError,
                             compute._pearson3_fitting_values,
                             np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
    np.testing.assert_raises(ValueError,
                             compute._pearson3_fitting_values,
                             np.array(
                                 [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0],
                                  [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.7]]
                             ))
    np.testing.assert_raises(ValueError,
                             compute._pearson3_fitting_values,
                             np.array([np.NaN]*9))
    np.testing.assert_raises(AttributeError, compute._pearson3_fitting_values, None)

    # try using a subset of the precipitation dataset (1897 - 1915, year indices 2 - 20)
    computed_values = compute._pearson3_fitting_values(precips_mm_monthly[2:21, :])
    expected_values = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
        ],
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
        ],
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
        ],
    ]
    )
    np.testing.assert_allclose(computed_values,
                               expected_values,
                               atol=0.001,
                               equal_nan=True,
                               err_msg="Failed to accurately compute "
                                       "Pearson Type III fitting values")

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
    computed_values = compute._pearson3_fitting_values(precips_mm)
    expected_values = np.array([
        [0.0, 0.008, 0.0, 0.008, 0.0164, 0.0164, 0.0, 0.0, 0.0, 0.0164, 0.0, 0.008],
        [45.85, 46.35, 48.32, 67.64, 121.17, 184.13, 154.97, 170.29, 196.43, 153.53, 58.40, 38.86],
        [38.87, 35.33, 34.32, 50.26, 73.52, 100.18, 50.63, 63.07, 75.26, 93.67, 48.75, 33.01],
        [1.76, 1.25, 1.17, 1.19, 0.76, 0.83, 0.18, 0.996, 0.83, 1.16, 1.85, 1.81]
    ])
    np.testing.assert_allclose(computed_values,
                               expected_values,
                               atol=0.01,
                               equal_nan=True,
                               err_msg="Failed to accurately compute "
                                       "Pearson Type III fitting values")


# ------------------------------------------------------------------------------
def test_sum_to_scale():
    """
    Test for the compute.sum_to_scale() function
    """

    # test an input array with no missing values
    values = np.array([3.0, 4, 6, 2, 1, 3, 5, 8, 5])
    computed_values = compute.sum_to_scale(values, 3)
    expected_values = np.array([np.NaN, np.NaN, 13, 12, 9, 6, 9, 16, 18])
    np.testing.assert_allclose(computed_values,
                               expected_values,
                               err_msg="Sliding sums not computed as expected")
    computed_values = compute.sum_to_scale(values, 4)
    expected_values = np.array([np.NaN, np.NaN, np.NaN, 15, 13, 12, 11, 17, 21])
    np.testing.assert_allclose(computed_values,
                               expected_values,
                               err_msg="Sliding sums not computed as expected")

    # test an input array with missing values on the end
    values = np.array([3, 4, 6, 2, 1, 3, 5, 8, 5, np.NaN, np.NaN, np.NaN])
    computed_values = compute.sum_to_scale(values, 3)
    expected_values = np.array(
        [np.NaN, np.NaN, 13, 12, 9, 6, 9, 16, 18, np.NaN, np.NaN, np.NaN]
    )
    np.testing.assert_allclose(computed_values,
                               expected_values,
                               err_msg="Sliding sums not computed as expected when "
                                       "missing values appended to end of input array")

    # test an input array with missing values within the array
    values = np.array([3, 4, 6, 2, 1, 3, 5, np.NaN, 8, 5, 6])
    computed_values = compute.sum_to_scale(values, 3)
    expected_values = np.array(
        [np.NaN, np.NaN, 13, 12, 9, 6, 9, np.NaN, np.NaN, np.NaN, 19]
    )
    np.testing.assert_allclose(computed_values,
                               expected_values,
                               err_msg="Sliding sums not computed as expected when "
                                       "missing values appended to end of input array")

    test_values = np.array([1.0, 5, 7, 2, 3, 4, 9, 6, 3, 8])
    sum_by2 = np.array([np.NaN, 6, 12, 9, 5, 7, 13, 15, 9, 11])
    sum_by4 = np.array([np.NaN, np.NaN, np.NaN, 15, 17, 16, 18, 22, 22, 26])
    sum_by6 = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 22, 30, 31, 27, 33])
    np.testing.assert_equal(compute.sum_to_scale(test_values, 2),
                            sum_by2,
                            err_msg="Sliding sums not computed as expected")
    np.testing.assert_equal(compute.sum_to_scale(test_values, 4),
                            sum_by4,
                            err_msg="Sliding sums not computed as expected")
    np.testing.assert_equal(compute.sum_to_scale(test_values, 6),
                            sum_by6,
                            err_msg="Sliding sums not computed as expected")
