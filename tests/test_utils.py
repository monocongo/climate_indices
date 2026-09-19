import logging

import numpy as np
import pytest

from climate_indices import utils
from climate_indices.exceptions import DataShapeError

# disable logging messages
logging.disable(logging.CRITICAL)


# Tests for `climate_indices.utils.py`


# ------------------------------------------------------------------------------
def test_compute_days():
    # Test for the utils.compute_days() function

    days_array = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    results = utils.compute_days(1800, 12, 1, 1800)
    np.testing.assert_allclose(
        days_array,
        results,
        err_msg="Fahrenheit to Celsius conversion failed",
        atol=0.01,
        equal_nan=True,
    )

    days_array = np.array(
        [
            18443,
            18474,
            18505,
            18535,
            18566,
            18596,
            18627,
            18658,
            18686,
            18717,
            18747,
            18778,
            18808,
            18839,
            18870,
            18900,
            18931,
            18961,
            18992,
            19023,
        ]
    )
    results = utils.compute_days(1850, 20, 7, 1800)
    np.testing.assert_allclose(
        days_array,
        results,
        err_msg="Fahrenheit to Celsius conversion failed",
        atol=0.01,
        equal_nan=True,
    )


# ------------------------------------------------------------------------------
def test_count_zeros_and_non_missings():
    # Test for the utils.count_zeros_and_non_missings() function

    # messages used multiple times below
    zero_count_error = "Failed to correctly count zero values"
    non_missing_count_error = "Failed to correctly count non-missing values"

    # vanilla use case
    values_list = [3, 4, 0, 2, 3.1, 5, np.nan, 8, 5, 6, 0.0, np.nan, 5.6, 2]
    values = np.array(values_list)
    zeros, non_missings = utils.count_zeros_and_non_missings(values)
    if zeros != 2:
        raise AssertionError(zero_count_error)
    if non_missings != 12:
        raise AssertionError(non_missing_count_error)

    # test with lists
    values = values_list
    zeros, non_missings = utils.count_zeros_and_non_missings(values)
    if zeros != 2:
        raise AssertionError(zero_count_error)
    if non_missings != 12:
        raise AssertionError(non_missing_count_error)
    values = [[3, 4, 0, 2, 3.1, 5, np.nan], [8, 5, 6, 0.0, np.nan, 5.6, 2]]
    zeros, non_missings = utils.count_zeros_and_non_missings(values)
    if zeros != 2:
        raise AssertionError(zero_count_error)
    if non_missings != 12:
        raise AssertionError(non_missing_count_error)

    # using a list that can't be converted
    # into an array should result in a TypeError
    values = [1, 2, 3, 0, "abcxyz"]
    np.testing.assert_raises(TypeError, utils.count_zeros_and_non_missings, values)


# ------------------------------------------------------------------------------
def test_is_data_valid():
    # Test for the utils.is_data_valid() function

    valid_array = np.full((12,), 1.0)
    invalid_array = np.full((12,), np.nan)
    if not utils.is_data_valid(valid_array):
        raise AssertionError()
    if utils.is_data_valid(invalid_array):
        raise AssertionError()
    if utils.is_data_valid(["bad", "data"]):
        raise AssertionError()
    if not utils.is_data_valid(np.ma.masked_array(valid_array)):
        raise AssertionError()


# ------------------------------------------------------------------------------
def test_gregorian_length_as_366day():
    # Test for the utils.sign_change() function

    assert utils.gregorian_length_as_366day(365, 1980) == 365
    assert utils.gregorian_length_as_366day(366, 1980) == 366
    assert utils.gregorian_length_as_366day(731, 1980) == 732
    assert utils.gregorian_length_as_366day(732, 1980) == 733
    assert utils.gregorian_length_as_366day(1096, 1980) == 1098
    assert utils.gregorian_length_as_366day(1855, 1980) == 1858


# ------------------------------------------------------------------------------
def test_sign_change():
    # Test for the utils.sign_change() function

    a = np.array([1.0, 2.0, 3.0, -4])
    b = np.array([1.0, -2.0, -3.0, -4])
    c = utils.sign_change(a, b)
    np.testing.assert_equal(c, np.array([False, True, True, False]), "Sign changes not detected as expected")

    a = np.array([1.0, 2.0, 3.0, -4])
    b = np.array([[1.0, -2.0], [-3.0, -4]])
    c = utils.sign_change(a, b)
    np.testing.assert_equal(c, np.array([False, True, True, False]), "Sign changes not detected as expected")

    # make sure that the function croaks with a ValueError
    np.testing.assert_raises(
        ValueError,
        utils.sign_change,
        np.array([1.0, 2.0, 3.0, -4]),
        np.array([1.0, 2.0, 3.0]),
    )
    np.testing.assert_raises(
        ValueError,
        utils.sign_change,
        np.array([1.0, 2.0, 3.0]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )


# ------------------------------------------------------------------------------
def test_reshape_to_2d():
    # Test for the utils.reshape_to_2d() function

    # an array of monthly values
    values_1d = np.array(
        [
            3,
            4,
            6,
            2,
            1,
            3,
            5,
            8,
            5,
            6,
            3,
            4,
            6,
            2,
            1,
            3,
            5,
            8,
            5,
            6,
            3,
            4,
            6,
            2,
            1,
            3,
            5,
            8,
            5,
            6,
        ],
        dtype=float,
    )

    # the expected rearrangement of the above values
    # from 1-D to 2-D if using 12 as the second axis size
    values_2d_by_12_expected = np.array(
        [
            [3, 4, 6, 2, 1, 3, 5, 8, 5, 6, 3, 4],
            [6, 2, 1, 3, 5, 8, 5, 6, 3, 4, 6, 2],
            [1, 3, 5, 8, 5, 6, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )

    # exercise the function
    values_2d_reshaped = utils.reshape_to_2d(values_1d, 12)

    # verify that the function performed as expected
    np.testing.assert_equal(
        values_2d_by_12_expected,
        values_2d_reshaped,
        "Not rearranging the 1-D array into " + "2-D year increments of 12 as expected",
    )

    # the expected rearrangement of the above values from 1-D to 2-D if using 8 as the second axis size
    values_2d_by_8_expected = np.array(
        [
            [3, 4, 6, 2, 1, 3, 5, 8],
            [5, 6, 3, 4, 6, 2, 1, 3],
            [5, 8, 5, 6, 3, 4, 6, 2],
            [1, 3, 5, 8, 5, 6, np.nan, np.nan],
        ]
    )

    # exercise the function
    values_2d_reshaped = utils.reshape_to_2d(values_1d, 8)

    # verify that the function performed as expected
    np.testing.assert_equal(
        values_2d_by_8_expected,
        values_2d_reshaped,
        "Not rearranging the 1-D array " + "into 2-D increments of 8 as expected",
    )

    # a 3-D array that should be returned as-is if using 12 as the second axis size
    values_2d = np.array(
        [
            [3, 4, 6, 2, 1, 3, 5, 8, 5, 6, 3, 4],
            [6, 2, 1, 3, 5, 8, 5, 6, 3, 4, 6, 2],
            [1, 3, 5, 8, 5, 6, 3, 5, 1, 2, 8, 4],
        ]
    )

    # exercise the function
    values_2d_reshaped = utils.reshape_to_2d(values_2d, 12)

    # verify that the function performed as expected
    np.testing.assert_equal(
        values_2d,
        values_2d_reshaped,
        "Not returning an already valid 2-D array as expected",
    )

    # a 2-D array that's in an invalid shape for the function
    values_2d = np.array(
        [
            [3, 4, 6, 2, 1, 3, 5, 3, 4],
            [6, 2, 1, 3, 5, 8, 5, 6, 2],
            [1, 3, 5, 8, 5, 6, 3, 8, 4],
        ]
    )

    # make sure that the function croaks with a ValueError when expected
    np.testing.assert_raises(ValueError, utils.reshape_to_2d, values_2d, 12)
    np.testing.assert_raises(ValueError, utils.reshape_to_2d, values_2d.reshape((3, 3, 3)), 6)


# ------------------------------------------------------------------------------
def test_reshape_to_divs_years_months():
    # Test for the utils.reshape_to_divs_years_months() function

    # an array of monthly values
    values_1d = np.array(
        [
            3,
            4,
            6,
            2,
            1,
            3,
            5,
            8,
            5,
            6,
            3,
            4,
            6,
            2,
            1,
            3,
            5,
            8,
            5,
            6,
            3,
            4,
            6,
            2,
            1,
            3,
            5,
            8,
            5,
            6,
        ]
    )

    # verify that the function performed as expected
    np.testing.assert_raises(ValueError, utils.reshape_to_divs_years_months, values_1d)

    # array of values for a single division, as 2-D
    row_1 = [3, 4, 6, 2, 1, 3, 5, 8, 5, 6, 3, 4]
    row_2 = [6, 2, 1, 3, 5, 8, 5, 6, 3, 4, 6, 2]
    values_2d = np.array([row_1, row_2])

    # the expected rearrangement of the above values from 2-D to 3-D
    values_3d_expected = np.array([[row_1], [row_2]])

    # exercise the function
    values_3d_computed = utils.reshape_to_divs_years_months(values_2d)

    np.testing.assert_equal(
        values_3d_computed,
        values_3d_expected,
        "Not rearranging the 1-D array months " + "into 2-D year increments as expected",
    )

    # a 3-D array that should be returned as-is
    values_3d = np.array(
        [
            [
                [3, 4, 6, 2, 1, 3, 5, 8, 5, 6, 3, 4],
                [6, 2, 1, 3, 5, 8, 5, 6, 3, 4, 6, 2],
                [1, 3, 5, 8, 5, 6, 3, 5, 1, 2, 8, 4],
            ],
            [
                [6, 2, 8, 3, 2, 1, 9, 6, 3, 4, 9, 8],
                [3, 1, 6, 2, 7, 3, 5, 8, 5, 6, 3, 4],
                [4, 2, 1, 7, 2, 8, 5, 6, 3, 4, 7, 9],
            ],
        ]
    )

    # exercise the function
    values_3d_reshaped = utils.reshape_to_divs_years_months(values_3d)

    # verify that the function performed as expected
    np.testing.assert_equal(values_3d, values_3d_reshaped, "Not returning a valid 2-D array as expected")

    # a 2-D array that's in an invalid shape for the function
    values_2d = np.array(
        [
            [3, 4, 6, 2, 1, 3, 5, 3, 4],
            [6, 2, 1, 3, 5, 9, 3, 6, 2],
            [1, 3, 5, 8, 9, 6, 3, 8, 4],
        ]
    )

    # make sure that the function croaks with a ValueError
    # whenever it gets a mis-shaped array
    np.testing.assert_raises(ValueError, utils.reshape_to_divs_years_months, values_1d)
    np.testing.assert_raises(ValueError, utils.reshape_to_divs_years_months, values_2d)
    np.testing.assert_raises(ValueError, utils.reshape_to_divs_years_months, np.reshape(values_2d, (3, 3, 3)))


# ------------------------------------------------------------------------------
def test_rmse():
    # Test for the utils.rmse() function

    vals1 = np.array([32, 212, 100, 98.6, 150, -15])
    vals2 = np.array([35, 216, 90, 88.6, 153, -12])
    computed_rmse = utils.rmse(vals1, vals2)
    expected_rmse = 6.364

    # verify that the function performed as expected
    if computed_rmse != pytest.approx(expected_rmse, abs=1e-3):
        raise AssertionError("Incorrect root mean square error (RMSE)")


# ------------------------------------------------------------------------------
def test_transform_to_gregorian():
    # Test for the utils.transform_to_gregorian() function

    # an array of 366 values, representing a year with 366-days, such as a leap year
    values_366 = np.array(range(366))

    # an array of 365 values, representing a year with 365 days, with the value
    # for all days after Feb 28th matching to those in the 366-day array
    values_365 = np.array(range(365))
    values_365[59:] = [x + 1 for x in values_365[59:]]

    # exercise the function with the 366-day year array,
    # using a non-leap year argument (1971)
    values_365_computed = utils.transform_to_gregorian(values_366, 1971)

    np.testing.assert_equal(
        values_365_computed,
        values_365,
        "Not transforming the 1-D array of 366-days " + "into a corresponding 365 day array",
    )

    # exercise the function with the 366-day year array,
    # using a leap year argument (1972)
    values_366_computed = utils.transform_to_gregorian(values_366, 1972)

    np.testing.assert_equal(
        values_366_computed,
        values_366,
        "Not transforming the 1-D array of 366-days into a corresponding 366-day array",
    )

    # make sure that the function croaks with a ValueError
    # whenever it gets invalid array arguments
    np.testing.assert_raises(ValueError, utils.transform_to_gregorian, values_365, 1972)
    np.testing.assert_raises(ValueError, utils.transform_to_gregorian, np.ones((2, 10)), 1972)

    # make sure that the function croaks with
    # a ValueError whenever it gets invalid year arguments
    np.testing.assert_raises(ValueError, utils.transform_to_gregorian, values_366, -1972)
    np.testing.assert_raises(TypeError, utils.transform_to_gregorian, values_366, 45.7)
    np.testing.assert_raises(TypeError, utils.transform_to_gregorian, values_366, "obviously wrong")


# ------------------------------------------------------------------------------
def test_transform_to_366day():
    # Test for the utils.transform_to_366day() function

    # an array of 366 values, representing a year
    # with 366-days, such as a leap year
    values_366 = np.array(range(366))

    # an array of 366 values, representing a year with 366-days, as a non-leap
    # year with the Feb 29th value an average of the Feb. 28th and Mar. 1st values
    values_366_faux_feb29 = np.array(range(366), dtype=float)
    values_366_faux_feb29[59] = 58.5
    values_366_faux_feb29[60:] = [x - 1 for x in values_366_faux_feb29[60:]]

    # an array of 365 values, representing a year with 365 days
    values_365 = np.array(range(365))

    # exercise the function with the 366-day year array,
    # using a non-leap year argument (1971)
    values_366_computed = utils.transform_to_366day(values_365, 1971, 1)

    np.testing.assert_equal(
        values_366_computed,
        values_366_faux_feb29,
        "Not transforming the 1-D array of 365 days " + "into a corresponding 366-day array as expected",
    )

    # exercise the function with the 366-day year array,
    # using a leap year argument (1972)
    values_366_computed = utils.transform_to_366day(values_366, 1972, 1)

    np.testing.assert_equal(
        values_366_computed,
        values_366,
        "Not transforming the 1-D array of 366-days " + "into a corresponding 366-day array",
    )

    # a final year may be partial; its missing days are padded with NaN
    partial_leap_year = utils.transform_to_366day(np.arange(50, dtype=float), 1972, 1)
    assert partial_leap_year.size == 366
    np.testing.assert_array_equal(partial_leap_year[:50], np.arange(50))
    assert np.all(np.isnan(partial_leap_year[50:]))

    partial_january = utils.transform_to_366day(np.ones(31), 2025, 1)
    assert partial_january.size == 366
    np.testing.assert_array_equal(partial_january[:31], np.ones(31))
    assert np.all(np.isnan(partial_january[31:]))

    # make sure that the function croaks with a ValueError
    # whenever it gets invalid array arguments
    np.testing.assert_raises(ValueError, utils.transform_to_366day, np.ones((2, 10)), 1972, 1)

    # make sure that the function croaks with a ValueError whenever it gets invalid year arguments
    np.testing.assert_raises(ValueError, utils.transform_to_366day, values_365, -1972, 1)
    np.testing.assert_raises(TypeError, utils.transform_to_366day, values_365, 45.7, 1)
    np.testing.assert_raises(TypeError, utils.transform_to_366day, values_365, "obviously wrong", 1)

    # make sure that the function croaks with a ValueError whenever it gets invalid total years arguments
    np.testing.assert_raises(ValueError, utils.transform_to_366day, values_365, 1972, -5)
    np.testing.assert_raises(TypeError, utils.transform_to_366day, values_365, 1972, 4.9)
    np.testing.assert_raises(ValueError, utils.transform_to_366day, values_365, 1972, 24)

    # an array longer than the declared span, or an empty array, is rejected
    np.testing.assert_raises(ValueError, utils.transform_to_366day, np.ones(366), 2001, 1)
    np.testing.assert_raises(ValueError, utils.transform_to_366day, np.array([]), 1972, 1)


def test_daily_calendar_plan_from_year_span():
    """from_year_span distributes observed values across the declared Gregorian years."""
    assert utils.DailyCalendarPlan.from_year_span(1971, 2, 730) == utils.DailyCalendarPlan(1971, (365, 365))
    assert utils.DailyCalendarPlan.from_year_span(1972, 1, 366) == utils.DailyCalendarPlan(1972, (366,))
    # 2020 is a leap year followed by 100 observed days of 2021
    assert utils.DailyCalendarPlan.from_year_span(2020, 2, 466) == utils.DailyCalendarPlan(2020, (366, 100))
    # a length shorter than the first year is absorbed by the first year
    assert utils.DailyCalendarPlan.from_year_span(2019, 2, 100) == utils.DailyCalendarPlan(2019, (100, 0))
    # trailing years beyond the data contribute nothing, and negative lengths clamp to zero
    assert utils.DailyCalendarPlan.from_year_span(2019, 3, 400) == utils.DailyCalendarPlan(2019, (365, 35, 0))
    assert utils.DailyCalendarPlan.from_year_span(2020, 2, -3) == utils.DailyCalendarPlan(2020, (0, 0))


def test_tolerance():
    lons, dlon = np.linspace(-180.0, 180.0, 250, retstep=True)
    tol = utils.get_tolerance(lons)
    tolerance_smaller = "Tolerance must always come out smaller than the coordinate delta"
    tolerance_greater = "Tolerance must always be greater than zero"
    assert tol > 0, tolerance_greater
    assert tol < abs(dlon), tolerance_smaller

    lats, dlat = np.linspace(25.0, -15.0, 50, retstep=True)
    tol = utils.get_tolerance(lats)
    assert tol > 0, tolerance_greater
    assert tol < abs(dlat), tolerance_smaller

    meshlat, meshlon = np.meshgrid(lats, lons)
    tol = utils.get_tolerance(meshlat)
    assert tol > 0, tolerance_greater
    assert tol < abs(dlat), tolerance_smaller
    # Tricky situation because np.diff() on this grid returns all zeros.
    # Not the greatest situation, but we can at least allow a tiny bit of tolerance.
    # Would be nice to be smarter about this situation.
    tol = utils.get_tolerance(meshlon)
    assert tol > 0, tolerance_greater
    assert tol < abs(dlon), tolerance_smaller

    # singleton dimensions have no spacing to derive a tolerance from
    tol = utils.get_tolerance(np.array([25.0]))
    assert tol > 0, tolerance_greater


class TestDailyCalendarPlanShapeContract:
    """Verify DailyCalendarPlan rejects slices that are not whole time series."""

    @staticmethod
    def _plan():
        """Build a two-year plan covering a non-leap year and a leap year."""
        # 2019 is not a leap year, 2020 is; the final year is partial
        return utils.DailyCalendarPlan(2019, (365, 100))

    def test_to_all_leap_rejects_wrong_length(self) -> None:
        """Converting a slice of the wrong length raises with shape context."""
        plan = self._plan()

        with pytest.raises(DataShapeError) as exc_info:
            plan.to_all_leap(np.zeros(plan.original_length - 1))

        assert exc_info.value.expected_shape == f"({plan.original_length},)"
        assert exc_info.value.actual_shape == (plan.original_length - 1,)

    def test_to_all_leap_rejects_multidimensional_input(self) -> None:
        """A block whose leading axis is not a whole time series raises, not reshapes."""
        plan = self._plan()

        with pytest.raises(DataShapeError) as exc_info:
            plan.to_all_leap(np.zeros((2, plan.original_length)))

        assert exc_info.value.actual_shape == (2, plan.original_length)

    def test_to_all_leap_preserves_trailing_cell_dimensions(self) -> None:
        """A 2-D block of complete time-series slices is converted cell-wise."""
        plan = self._plan()
        values = np.zeros((plan.original_length, 3))
        values[:, 0] = np.arange(1, plan.original_length + 1)

        converted = plan.to_all_leap(values)

        assert converted.shape == (plan.all_leap_length, 3)
        # cell 0 is a non-leap year followed by a partial leap year: the synthetic
        # February 29 of 2019 is the mean of February 28 and March 1
        assert converted[59, 0] == 59.5
        assert converted[59, 1] == 0.0
        np.testing.assert_array_equal(converted[60:366, 0], np.arange(60, 366))
        np.testing.assert_array_equal(converted[366:466, 0], np.arange(366, 466))
        assert np.all(np.isnan(converted[466:, 0]))

    def test_to_gregorian_rejects_wrong_length(self) -> None:
        """Restoring a slice that is not a whole 366-day series raises."""
        plan = self._plan()

        with pytest.raises(DataShapeError) as exc_info:
            plan.to_gregorian(np.zeros(plan.all_leap_length - 1))

        assert exc_info.value.expected_shape == f"({plan.all_leap_length},)"
        assert exc_info.value.actual_shape == (plan.all_leap_length - 1,)

    def test_to_gregorian_rejects_multidimensional_input(self) -> None:
        """A block whose leading axis is not a whole 366-day series raises, not reshapes."""
        plan = self._plan()

        with pytest.raises(DataShapeError) as exc_info:
            plan.to_gregorian(np.zeros((2, plan.all_leap_length)))

        assert exc_info.value.actual_shape == (2, plan.all_leap_length)

    def test_gregorian_round_trip_preserves_trailing_cell_dimensions(self) -> None:
        """Cells of a 2-D block survive the leap-calendar round trip."""
        plan = self._plan()
        values = np.arange(1, plan.original_length + 1, dtype=float)
        block = np.stack([values, values * 2.0], axis=1)

        restored = plan.to_gregorian(plan.to_all_leap(block))

        assert restored.shape == block.shape
        np.testing.assert_array_equal(restored, block)

    def test_round_trip_restores_original_values_including_partial_final_year(self) -> None:
        """to_gregorian inverts to_all_leap exactly, partial final year included."""
        plan = self._plan()
        values = np.arange(1, plan.original_length + 1, dtype=float)

        np.testing.assert_array_equal(plan.to_gregorian(plan.to_all_leap(values)), values)
