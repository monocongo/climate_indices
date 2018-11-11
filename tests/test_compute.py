import logging
import math
import numpy as np
import unittest

from tests import fixtures
from climate_indices import compute

# ----------------------------------------------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)


# ------------------------------------------------------------------------------------------------------------------------------------------
class ComputeTestCase(fixtures.FixturesTestCase):
    """
    Tests for `compute.py`.
    """

    # # ---------------------------------------------------------------------------------------
    # def test_error_function(self):
    #     """
    #     Test for the compute._error_function() function
    #     """
    #
    #     self.assertEqual(compute._error_function(0.0),
    #                      0.0,
    #                      msg='Failed to accurately compute error function')
    #
    #     self.assertEqual(compute._error_function(5.0),
    #                      0.9999999999992313,
    #                      msg='Failed to accurately compute error function')
    #
    #     self.assertEqual(compute._error_function(-5.0),
    #                      -0.9999999999992313,
    #                      msg='Failed to accurately compute error function')
    #
    #     self.assertEqual(compute._error_function(5.5),
    #                      0.9999999999999941,
    #                      msg='Failed to accurately compute error function')
    #
    #     self.assertEqual(compute._error_function(0.17),
    #                      0.5949962306009045,
    #                      msg='Failed to accurately compute error function')
    #
    #     self.assertEqual(compute._error_function(-0.07),
    #                      -0.5394288598854453,
    #                      msg='Failed to accurately compute error function')
    #
    #     self.assertEqual(compute._error_function(50.0),
    #                      1.0,
    #                      msg='Failed to accurately compute error function')
    #
    #     self.assertEqual(compute._error_function(-50.0),
    #                      -1.0,
    #                      msg='Failed to accurately compute error function')
    #
    #     self.assertEqual(compute._error_function(-2.0),
    #                      -0.9976611325094764,
    #                      msg='Failed to accurately compute error function')
    #
    # # ---------------------------------------------------------------------------------------
    # @staticmethod
    # def test_estimate_lmoments():
    #     """
    #     Test for the compute._estimate_lmoments() function
    #     """
    #
    #     # provide some bogus inputs to at least make sure these raise expected errors
    #     np.testing.assert_raises(ValueError, compute._estimate_lmoments, [1.0, 0.0, 0.0])
    #     np.testing.assert_raises(ValueError, compute._estimate_lmoments, [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
    #     np.testing.assert_raises(TypeError, compute._estimate_lmoments, None)
    #
    #     values = [0.8, 0.7, 0.6, 0.8, 0.7, 0.6, 0.8, 0.7, 0.6, 0.8, 0.8, 0.7, 0.6, 0.8, 0.7, 0.6, 0.7, 0.6]
    #     lmoments_expected = [0.7, 0.0470588235294, -9.43689570931e-15]
    #     lmoments_computed = compute._estimate_lmoments(values)
    #     np.testing.assert_allclose(lmoments_expected,
    #                                lmoments_computed,
    #                                atol=0.001,
    #                                err_msg='Failed to accurately estimate L-moments')
    #
    # # ---------------------------------------------------------------------------------------
    # @staticmethod
    # def test_estimate_pearson3_parameters():
    #     """
    #     Test for the compute._estimate_pearson3_parameters() function
    #     """
    #     # provide some bogus inputs to at least make sure these raise expected errors
    #     np.testing.assert_raises(ValueError, compute._estimate_pearson3_parameters, [1.0, 0.0, 0.0])
    #     np.testing.assert_raises(ValueError, compute._estimate_pearson3_parameters, [1.0, 1.0, 5.0])
    #     np.testing.assert_raises(ValueError, compute._estimate_pearson3_parameters, [1.0, -1.0, 1.0])
    #     np.testing.assert_raises(ValueError, compute._estimate_pearson3_parameters, [1.0, -1.0, 1e-7])
    #
    # # ---------------------------------------------------------------------------------------
    # def test_pearson3cdf(self):
    #     """
    #     Test for the compute._pearson3cdf() function
    #     """
    #
    #     self.assertTrue(math.isnan(compute._pearson3cdf(5.0, [1.0, -1.0, 0.0])),
    #                     msg='Failed to accurately compute Pearson Type III CDF')
    #
    #     self.assertEqual(compute._pearson3cdf(5.0, [1.0, 1.0, 1e-7]),
    #                      0.9999841643790834,
    #                      msg='Failed to accurately compute Pearson Type III CDF')
    #
    #     self.assertEqual(compute._pearson3cdf(7.7, [1.0, 501.0, 0.0]),
    #                      0.752667498611228,
    #                      msg='Failed to accurately compute Pearson Type III CDF')
    #
    #     self.assertEqual(compute._pearson3cdf(7.7, [1.0, 501.0, -10.0]),
    #                      0.10519432662999628,
    #                      msg='Failed to accurately compute Pearson Type III CDF')
    #
    #     self.assertEqual(compute._pearson3cdf(1e-6, [441.0, 501.0, 30.0]),
    #                      0.0005,  # value corresponding to trace value
    #                      msg='Failed to accurately compute Pearson Type III CDF')
    #
    # # ---------------------------------------------------------------------------------------
    # def test_pearson_fit_ufunc(self):
    #     """
    #     Test for the compute._pearson_fit_ufunc() function
    #     """
    #
    #     self.assertTrue(math.isnan(compute._pearson_fit_ufunc(np.NaN, 1.0, -1.0, 0.0, 0.0)),
    #                     msg='Failed to accurately compute error function')
    #
    #     self.assertTrue(math.isnan(compute._pearson_fit_ufunc(5.0, 1.0, -1.0, 0.0, 0.0)),
    #                     msg='Failed to accurately compute error function')
    #
    #     self.assertEqual(compute._pearson_fit_ufunc(7.7, 1.0, 501.0, 0.0, 0.07),
    #                      0.7387835329883602,
    #                      msg='Failed to accurately compute error function')

    # ---------------------------------------------------------------------------------------
    def test_pearson3_fitting_values(self):
        """
        Test for the compute._pearson3_fitting_values() function
        """
        # provide some bogus inputs to make sure these raise expected errors
        np.testing.assert_raises(
            ValueError, compute._pearson3_fitting_values, np.array([1.0, 0.0, 0.0])
        )
        np.testing.assert_raises(
            ValueError,
            compute._pearson3_fitting_values,
            np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        )
        np.testing.assert_raises(
            ValueError,
            compute._pearson3_fitting_values,
            np.array(
                [
                    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0],
                    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.7],
                ]
            ),
        )
        np.testing.assert_raises(
            ValueError,
            compute._pearson3_fitting_values,
            np.array(
                [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]
            ),
        )
        np.testing.assert_raises(AttributeError, compute._pearson3_fitting_values, None)

        # try using a subset of the precipitation dataset (1897 - 1915, year indices 2 - 20)
        computed_values = compute._pearson3_fitting_values(
            self.fixture_precips_mm_monthly[2:21, :]
        )
        expected_values = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    48.539_987_664_499_996,
                    53.985_248_766_5,
                    44.284_745_065_842_102,
                    62.583_727_384_894_736,
                    125.721_576_891_605_28,
                    182.030_530_427_842_14,
                    159.005_756_579_263_19,
                    170.922_697_368_657_91,
                    189.892_578_125_289_5,
                    155.134_200_246_921_04,
                    72.953_125_000_026_319,
                    43.315_326_891_447_37,
                ],
                [
                    33.781_507_724_523_095,
                    43.572_151_699_968_387,
                    40.368_173_442_404_107,
                    44.053_296_914_348_87,
                    60.106_217_160_191_74,
                    59.343_178_125_457_186,
                    49.228_795_303_727_473,
                    66.775_653_341_386_999,
                    65.362_977_393_206_421,
                    94.467_597_091_088_265,
                    72.637_068_983_642_99,
                    34.250_906_049_301_463,
                ],
                [
                    0.765_309_669_763_353_02,
                    1.246_144_751_821_978_4,
                    2.275_517_179_222_323,
                    0.806_930_509_869_819_4,
                    -0.678_303_702_019_701_8,
                    1.022_194_696_224_529,
                    0.408_761_207_328_175_78,
                    1.237_255_134_616_891_6,
                    0.738_811_169_319_241_18,
                    0.919_117_632_570_034_65,
                    2.384_671_588_726_372_5,
                    1.470_055_929_457_196_2,
                ],
            ]
        )
        np.testing.assert_allclose(
            computed_values,
            expected_values,
            atol=0.001,
            equal_nan=True,
            err_msg="Failed to accurately compute Pearson Type III fitting values",
        )

        # add some zeros in order to exercise the parts where it gets a percentage of zeros
        precips_mm = np.array(self.fixture_precips_mm_monthly, copy=True)
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
        expected_values = np.array(
            [
                [
                    0.0,
                    0.008,
                    0.0,
                    0.008,
                    0.0164,
                    0.0164,
                    0.0,
                    0.0,
                    0.0,
                    0.0164,
                    0.0,
                    0.008,
                ],
                [
                    45.85,
                    46.35,
                    48.32,
                    67.64,
                    121.17,
                    184.13,
                    154.97,
                    170.29,
                    196.43,
                    153.53,
                    58.40,
                    38.86,
                ],
                [
                    38.87,
                    35.33,
                    34.32,
                    50.26,
                    73.52,
                    100.18,
                    50.63,
                    63.07,
                    75.26,
                    93.67,
                    48.75,
                    33.01,
                ],
                [
                    1.76,
                    1.25,
                    1.17,
                    1.19,
                    0.76,
                    0.83,
                    0.18,
                    0.996,
                    0.83,
                    1.16,
                    1.85,
                    1.81,
                ],
            ]
        )
        np.testing.assert_allclose(
            computed_values,
            expected_values,
            atol=0.01,
            equal_nan=True,
            err_msg="Failed to accurately compute Pearson Type III fitting values",
        )

    # ---------------------------------------------------------------------------------------
    @staticmethod
    def test_sum_to_scale():
        """
        Test for the compute.sum_to_scale() function
        """

        # test an input array with no missing values
        values = np.array([3, 4, 6, 2, 1, 3, 5, 8, 5])
        computed_values = compute.sum_to_scale(values, 3)
        expected_values = np.array([np.NaN, np.NaN, 13, 12, 9, 6, 9, 16, 18])
        np.testing.assert_allclose(
            computed_values,
            expected_values,
            err_msg="Sliding sums not computed as expected",
        )

        # test an input array with missing values on the end
        values = np.array([3, 4, 6, 2, 1, 3, 5, 8, 5, np.NaN, np.NaN, np.NaN])
        computed_values = compute.sum_to_scale(values, 3)
        expected_values = np.array(
            [np.NaN, np.NaN, 13, 12, 9, 6, 9, 16, 18, np.NaN, np.NaN, np.NaN]
        )
        np.testing.assert_allclose(
            computed_values,
            expected_values,
            err_msg="Sliding sums not computed as expected when missing values appended to end of input array",
        )

        # test an input array with missing values within the array
        values = np.array([3, 4, 6, 2, 1, 3, 5, np.NaN, 8, 5, 6])
        computed_values = compute.sum_to_scale(values, 3)
        expected_values = np.array(
            [np.NaN, np.NaN, 13, 12, 9, 6, 9, np.NaN, np.NaN, np.NaN, 19]
        )
        np.testing.assert_allclose(
            computed_values,
            expected_values,
            err_msg="Sliding sums not computed as expected when missing "
            "values appended to end of input array",
        )

    # ----------------------------------------------------------------------------------------
    def test_transform_fitted_gamma(self):
        """
        Test for the compute.transform_fitted_gamma() function
        """

        # confirm that an input array of all NaNs results in the same array returned
        all_nans = np.full(self.fixture_precips_mm_monthly.shape, np.NaN)
        computed_values = compute.transform_fitted_gamma(
            all_nans,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_end_monthly,
            compute.Periodicity.monthly,
        )
        np.testing.assert_allclose(
            computed_values,
            all_nans,
            equal_nan=True,
            err_msg="Gamma fit/transform not handling all-NaN arrays as expected",
        )

        # compute sigmas of transformed (normalized) values fitted to a gamma distribution,
        # using the full period of record as the calibration period
        computed_values = compute.transform_fitted_gamma(
            self.fixture_precips_mm_monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_end_monthly,
            compute.Periodicity.monthly,
        )
        np.testing.assert_allclose(
            computed_values,
            self.fixture_transformed_gamma_monthly,
            err_msg="Transformed gamma fitted monthly values not computed as expected",
        )

        # compute sigmas of transformed (normalized) values fitted to a gamma distribution,
        # using the full period of record as the calibration period
        computed_values = compute.transform_fitted_gamma(
            self.fixture_precips_mm_daily.flatten(),
            self.fixture_data_year_start_daily,
            self.fixture_calibration_year_start_daily,
            self.fixture_calibration_year_end_daily,
            compute.Periodicity.daily,
        )
        np.testing.assert_allclose(
            computed_values,
            self.fixture_transformed_gamma_daily,
            atol=0.001,
            equal_nan=True,
            err_msg="Transformed gamma fitted daily values not computed as expected",
        )

        # confirm that we can call with a calibration period outside of valid range
        # and as a result use the full period of record as the calibration period instead
        computed_values = compute.transform_fitted_gamma(
            self.fixture_precips_mm_monthly,
            self.fixture_data_year_start_monthly,
            1500,
            2500,
            compute.Periodicity.monthly,
        )
        np.testing.assert_allclose(
            computed_values,
            self.fixture_transformed_gamma_monthly,
            atol=0.001,
            equal_nan=True,
            err_msg="Transformed Pearson Type III fitted values not computed as expected",
        )

        # if we provide a 1-D array then we need to provide a corresponding time series type,
        # confirm we can't use an invalid type
        flat_array = self.fixture_precips_mm_monthly.flatten()
        np.testing.assert_raises(
            ValueError,
            compute.transform_fitted_gamma,
            flat_array,
            self.fixture_data_year_start_monthly,
            self.fixture_calibration_year_start_monthly,
            self.fixture_calibration_year_end_monthly,
            "invalid_value",
        )
        np.testing.assert_raises(
            ValueError,
            compute.transform_fitted_gamma,
            flat_array,
            self.fixture_data_year_start_monthly,
            self.fixture_calibration_year_start_monthly,
            self.fixture_calibration_year_end_monthly,
            None,
        )

        # confirm that an input array which is not 1-D or 2-D will raise an error
        self.assertRaises(
            ValueError,
            compute.transform_fitted_gamma,
            np.zeros((9, 8, 7, 6), dtype=float),
            self.fixture_data_year_start_daily,
            self.fixture_calibration_year_start_daily,
            self.fixture_calibration_year_end_daily,
            compute.Periodicity.monthly,
        )

    # ---------------------------------------------------------------------------------------
    def test_transform_fitted_pearson(self):
        """
        Test for the compute.transform_fitted_pearson() function
        """

        # confirm that an input array of all NaNs results in the same array returned
        all_nans = np.full(self.fixture_precips_mm_monthly.shape, np.NaN)
        computed_values = compute.transform_fitted_pearson(
            all_nans,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_end_monthly,
            compute.Periodicity.monthly,
        )
        np.testing.assert_allclose(
            computed_values,
            all_nans,
            equal_nan=True,
            err_msg="Pearson fit/transform not handling all-NaN arrays as expected",
        )

        # compute sigmas of transformed (normalized) values fitted to a Pearson Type III distribution
        computed_values = compute.transform_fitted_pearson(
            self.fixture_precips_mm_monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_calibration_year_start_monthly,
            self.fixture_calibration_year_end_monthly,
            compute.Periodicity.monthly,
        )
        expected_values = self.fixture_transformed_pearson3
        np.testing.assert_allclose(
            computed_values,
            expected_values,
            atol=0.001,
            err_msg="Transformed Pearson Type III fitted values not computed as expected",
        )

        # confirm that an input array of all NaNs will return the same array
        all_nans = np.full(self.fixture_precips_mm_monthly.shape, np.NaN)
        computed_values = compute.transform_fitted_pearson(
            all_nans,
            self.fixture_data_year_start_monthly,
            self.fixture_calibration_year_start_monthly,
            self.fixture_calibration_year_end_monthly,
            compute.Periodicity.monthly,
        )
        np.testing.assert_allclose(
            computed_values,
            all_nans,
            equal_nan=True,
            err_msg="Transformed Pearson Type III fitted values not computed as expected",
        )

        # confirm that we can call with a calibration period outside of valid range
        # and as a result use the full period of record as the calibration period instead
        computed_values = compute.transform_fitted_pearson(
            self.fixture_precips_mm_monthly,
            self.fixture_data_year_start_monthly,
            1500,
            2500,
            compute.Periodicity.monthly,
        )
        np.testing.assert_allclose(
            computed_values.flatten(),
            self.fixture_transformed_pearson3_monthly_fullperiod,
            atol=0.001,
            equal_nan=True,
            err_msg="Transformed Pearson Type III fitted values not computed as expected",
        )

        # confirm that we can call with daily values and not raise an error
        compute.transform_fitted_pearson(
            self.fixture_precips_mm_daily,
            self.fixture_data_year_start_daily,
            self.fixture_calibration_year_start_daily,
            self.fixture_calibration_year_end_daily,
            compute.Periodicity.daily,
        )

        # confirm that we get expected errors when using invalid time series type arguments
        self.assertRaises(
            ValueError,
            compute.transform_fitted_pearson,
            self.fixture_precips_mm_monthly.flatten(),
            self.fixture_data_year_start_monthly,
            self.fixture_calibration_year_start_monthly,
            self.fixture_calibration_year_end_monthly,
            None,
        )
        self.assertRaises(
            ValueError,
            compute.transform_fitted_pearson,
            self.fixture_precips_mm_monthly.flatten(),
            self.fixture_data_year_start_monthly,
            self.fixture_calibration_year_start_monthly,
            self.fixture_calibration_year_end_monthly,
            "unsupported_type",
        )

        # confirm that an input array which is not 1-D or 2-D will raise an error
        self.assertRaises(
            ValueError,
            compute.transform_fitted_pearson,
            np.zeros((9, 8, 7, 6), dtype=float),
            self.fixture_data_year_start_daily,
            self.fixture_calibration_year_start_daily,
            self.fixture_calibration_year_end_daily,
            compute.Periodicity.monthly,
        )


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
