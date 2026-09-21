"""Various utility/convenience functions

Every name in ``__all__`` is supported public API: helpers with no in-package
caller (``compute_days``, ``is_data_valid``, ``rmse``, ``sign_change``,
``transform_to_366day``, ``transform_to_gregorian``,
``gregorian_length_as_366day``, ``reshape_to_divs_years_months``) are kept for
downstream users rather than deprecated, since removing them would break
callers without offering an in-library replacement.
"""

import calendar
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from climate_indices.exceptions import DataShapeError
from climate_indices.logging_config import get_logger as _get_structlog_logger

# module-level structlog logger
_logger = _get_structlog_logger(__name__)

# declare the function names that should be included in the public API for this module
__all__ = [
    "DailyCalendarPlan",
    "compute_days",
    "count_zeros_and_non_missings",
    "get_logger",
    "get_tolerance",
    "gregorian_length_as_366day",
    "is_data_valid",
    "reshape_to_2d",
    "reshape_to_divs_years_months",
    "rmse",
    "sign_change",
    "transform_to_366day",
    "transform_to_gregorian",
]


def compute_days(
    initial_year: int,
    total_months: int,
    initial_month: int = 1,
    units_start_year: int = 1800,
) -> np.ndarray:
    """
    Computes the "number of days" equivalent for regular, incremental monthly
    time steps given an initial year/month. Useful when using "days since
    <start_date>" as time units within a NetCDF dataset.

    :param initial_year: the initial year from which the day values
        should start, i.e. the first value in the output array will correspond
        to the number of days between January of this initial year since January
        of the units start year
    :param initial_month: the month within the initial year from which the day
        values should start, with 1: January, 2: February, etc.
    :param total_months: the total number of monthly increments (time steps
        measured in days) to be computed
    :param units_start_year: the start year from which the monthly increments
        are computed, with time steps measured in days since January of this
        starting year
    :return: an array of time step increments, measured in days since midnight
        of January 1st of the units start year
    :rtype: ndarray of ints
    """

    # compute an offset from which the day values should begin
    start_date = datetime(units_start_year, 1, 1)

    # initialize the list of day values we'll build
    days = np.empty(total_months, dtype=int)

    # loop over all time steps (months)
    for i in range(total_months):
        years = int((i + initial_month - 1) / 12)  # the number of years since the initial year
        months = int((i + initial_month - 1) % 12)  # the number of months since January

        # cook up a datetime object for the current time step (month)
        current_date = datetime(initial_year + years, 1 + months, 1)

        # get the number of days since the initial date
        days[i] = (current_date - start_date).days

    return days


def count_zeros_and_non_missings(
    values: np.ndarray,
) -> tuple[int, int]:
    """
    Given an input array of values return a count of the zeros
    and non-missing values. Missing values assumed to be numpy.NaNs.

    :param values: array like object (numpy array, most likely)
    :return: two int scalars: 1) the count of zeros, and
        2) the count of non-missing values
    """

    # make sure we have a numpy array
    values = np.array(values)

    # count the number of zeros and non-missing (non-NaN) values
    zeros = values.size - np.count_nonzero(values)
    non_missings = np.count_nonzero(~np.isnan(values))

    return int(zeros), int(non_missings)


def get_logger(name: str, level: int) -> logging.Logger:
    """
    Sets up a basic, global _logger

    :param name:
    :param level:
    :return:
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d  %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def get_tolerance(dim: np.ndarray) -> float:
    """
    dynamic threshold absolute tolerance parameter np.allclose
    derived from (smallest) absolute grid size along dimension dim.
    Always greater than zero.
    """
    if dim.size < 2:
        # a singleton dimension has no spacing, so np.diff() is empty
        return float(np.finfo(np.float64).resolution)
    tol = np.abs(np.diff(dim)).min() / 10
    return float(max(tol, np.finfo(tol.dtype).resolution))


def sign_change(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Given two same-sized arrays of floats return an array of booleans indicating
    if a sign change occurs at the corresponding index.

    :param a: array of floats
    :param b: array of floats
    :return: array of booleans of same size as input arrays
    """

    if a.size != b.size:
        raise ValueError("Mismatched input arrays")

    # use the shape of the first array as the shape of the array we'll return
    original_shape = a.shape

    # get the sign value for each element
    sign_a = np.sign(a.flatten())
    sign_b = np.sign(b.flatten())

    # sign change between the two where values unequal
    sign_changes = sign_a != sign_b

    return np.reshape(sign_changes, original_shape)


def is_data_valid(
    data: np.ndarray,
) -> bool:
    """
    Returns if an array is valid or not, i.e. a supported array type
    (ndarray or MaskArray) which is not all-NaN.

    :param data: data object, expected as either numpy.ndarray or numpy.ma.MaskArray
    :return: True if array is non-NaN for at least one element
        and is an array type valid for processing by other modules
    :rtype: boolean
    """

    # make sure we're not dealing with all NaN values
    if np.ma.isMaskedArray(data):
        # TODO fix this, there is no ndarray.count according to PyCharm's warning, use another approach for this flag
        valid_flag = bool(data.count())

    elif isinstance(data, np.ndarray):
        valid_flag = not np.all(np.isnan(data))
    else:
        _logger.warning("validation_warning", reason="invalid_data_type")  # type: ignore[unreachable]
        valid_flag = False

    return valid_flag


def rmse(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Root mean square error

    :param predictions: np.ndarray
    :param targets: np.ndarray
    :return: float
    """
    return float(np.sqrt(((predictions - targets) ** 2).mean()))


def reshape_to_2d(
    values: np.ndarray,
    second_axis_length: int,
) -> np.ndarray:
    """
    :param values: an 1-D numpy.ndarray of values
    :param second_axis_length:
    :return: the original values reshaped to 2-D, with shape
        [int(original length / second axis length), second axis length]
    :rtype: 2-D numpy.ndarray of floats
    """

    # if we've been passed a 2-D array with valid shape then let it pass through
    shape = values.shape
    if len(shape) == 2:
        if shape[1] == second_axis_length:
            # data is already in the shape we want, return it unaltered
            return values
        else:
            message = (
                "Values array has an invalid shape (2-D but second " + f"dimension not {second_axis_length}): {shape}"
            )
            _logger.error(
                "array_reshape_error",
                operation="reshape_to_2d",
                reason="invalid_2d_shape",
                actual_shape=str(shape),
                expected_second_dim=second_axis_length,
            )
            raise ValueError(message)

    # otherwise make sure that we've been passed a flat (1-D) array of values
    elif len(shape) != 1:
        message = f"Values array has an invalid shape (not 1-D or 2-D): {shape}"
        _logger.error(
            "array_reshape_error",
            operation="reshape_to_2d",
            reason="invalid_dimensionality",
            actual_shape=str(shape),
        )
        raise ValueError(message)

    # pad the end of the original array in order
    # to have an ordinal increment, if necessary
    final_year_values = shape[0] % second_axis_length
    if final_year_values > 0:
        pads = second_axis_length - final_year_values
        values = np.pad(values, pad_width=(0, pads), mode="constant", constant_values=np.nan)

    # we should have an ordinal number of years now
    # (ordinally divisible by second_axis_length)
    first_axis_length = int(values.shape[0] / second_axis_length)

    # return the reshaped array
    result = np.reshape(values, (first_axis_length, second_axis_length))
    _logger.debug(
        "array_reshaped",
        operation="reshape_to_2d",
        input_shape=str(shape),
        output_shape=str(result.shape),
    )
    return result


def reshape_to_divs_years_months(
    monthly_values: np.ndarray,
) -> np.ndarray:
    """
    :param monthly_values: an 2-D numpy.ndarray of monthly values,
        assumed to start at January of the first year for each division,
        with dimension 0: division, dimension 1: months (0 to total months - 1)
    :return: the original monthly values reshaped to 3-D (divisions, years, 12),
        within each division each row maps to a year, with each column of
        the row matching to the corresponding calendar month
    :rtype: 3-D numpy.ndarray of floats
    """

    # if we've been passed a 3-D array with valid shape then let it pass through
    shape = monthly_values.shape
    if len(shape) == 3:
        if shape[2] == 12:
            # data is already in the shape we want, return it unaltered
            return monthly_values
        else:
            message = "Values array has an invalid shape (3-D but " + "third dimension is not 12): " + str(shape)
            _logger.error(
                "array_reshape_error",
                operation="reshape_to_divs_years_months",
                reason="invalid_3d_shape",
                actual_shape=str(shape),
            )
            raise ValueError(message)

    # otherwise make sure that we've been passed in a 2-D array of values
    elif len(shape) != 2:
        message = "Values array has an invalid shape (not 2-D or 3-D): " + str(shape)
        _logger.error(
            "array_reshape_error",
            operation="reshape_to_divs_years_months",
            reason="invalid_dimensionality",
            actual_shape=str(shape),
        )
        raise ValueError(message)

    # otherwise make sure that we've been passed in a 2-D array
    # of values with the final dimension size == 12
    elif shape[1] != 12:
        message = "Values array has an invalid shape (second dimension " + "should be 12, but is not): " + str(shape)
        _logger.error(
            "array_reshape_error",
            operation="reshape_to_divs_years_months",
            reason="invalid_second_dimension",
            actual_shape=str(shape),
        )
        raise ValueError(message)

    # we should have an ordinal number of years now (ordinally divisible by 12)
    total_years = int(monthly_values.shape[1] / 12)

    # reshape from (months) to (years, 12) in order
    # to have one year of months per row
    return np.reshape(monthly_values, (shape[0], total_years, 12))


def gregorian_length_as_366day(
    length_gregorian: int,
    year_start: int,
) -> int:
    """
    Return the number of values a Gregorian span occupies in a 366-day year layout.

    A trailing partial year is counted with only its observed days, not padded;
    ``DailyCalendarPlan.all_leap_length`` is the padded length for the same span.

    :param length_gregorian: the number of Gregorian days in the span
    :param year_start: the Gregorian year of the first day
    :return: the number of 366-day-layout values
    """
    year = year_start
    remaining = length_gregorian
    length_366day = 0
    while remaining > 0:
        if calendar.isleap(year):
            days_in_current_year = 366
        else:
            days_in_current_year = 365

        if remaining >= days_in_current_year:
            length_366day += 366
        else:
            length_366day += remaining

        remaining -= days_in_current_year
        year += 1

    return length_366day


@dataclass(frozen=True)
class DailyCalendarPlan:
    """
    Map Gregorian daily values to the NumPy core's 366-day calendar positions.

    The values are assumed to begin on January 1 of ``year_start`` and to step
    one calendar day at a time; the plan is a positional mapping and validates
    only the total length, not the timestamps themselves.

    :param year_start: the Gregorian year of the first value
    :param observed_days_by_year: the number of Gregorian values falling in each year
    """

    year_start: int
    observed_days_by_year: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject year counts that cannot describe one contiguous daily series."""
        for offset, observed_days in enumerate(self.observed_days_by_year):
            days_in_year = 366 if calendar.isleap(self.year_start + offset) else 365
            if not 0 <= observed_days <= days_in_year:
                raise ValueError(
                    f"Invalid observed days for {self.year_start + offset}: "
                    f"{observed_days} is outside 0..{days_in_year}"
                )
            if offset < len(self.observed_days_by_year) - 1 and observed_days != days_in_year:
                raise ValueError(
                    f"Invalid observed days for {self.year_start + offset}: "
                    f"only the final year may be partial, got {observed_days} of {days_in_year}"
                )

    @classmethod
    def from_year_span(cls, year_start: int, total_years: int, observed_length: int) -> "DailyCalendarPlan":
        """
        Plan the conversion of ``observed_length`` Gregorian daily values spanning ``total_years`` years.

        Values are distributed across the years in calendar order. Only the
        final year may be partial; its missing positions are held as NaN when
        the plan converts to the 366-day layout.

        Args:
            year_start: the Gregorian year of the first value.
            total_years: the number of Gregorian years the values span.
            observed_length: the number of Gregorian daily values.

        Returns:
            The plan for this span.

        Raises:
            ValueError: if ``observed_length`` is negative or exceeds the days
                in the declared span, or if the span cannot be distributed
                with only a partial final year.
        """
        if observed_length < 0:
            raise ValueError("Invalid observed length: must not be negative")
        year_capacities = tuple(
            366 if calendar.isleap(year) else 365 for year in range(year_start, year_start + total_years)
        )
        if observed_length > sum(year_capacities):
            raise ValueError("Invalid observed length: exceeds the days in the declared span")
        remaining = observed_length
        observed_days: list[int] = []
        for days_in_year in year_capacities:
            observed_days.append(min(remaining, days_in_year))
            remaining -= observed_days[-1]
        return cls(year_start, tuple(observed_days))

    @property
    def original_length(self) -> int:
        """Return the number of observed Gregorian days."""
        return sum(self.observed_days_by_year)

    @property
    def all_leap_length(self) -> int:
        """Return the number of values required by the 366-day NumPy core."""
        return len(self.observed_days_by_year) * 366

    def to_all_leap(self, values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Insert synthetic February 29 values while retaining a partial final year.

        Args:
            values: an array whose leading axis is one complete Gregorian record.

        Returns:
            The values in the 366-day layout, padded with NaN.

        Raises:
            DataShapeError: if the leading axis length is not ``original_length``.
        """
        source = np.asarray(values)
        if source.ndim < 1 or source.shape[0] != self.original_length:
            raise DataShapeError(
                "Daily calendar transformation requires one complete time-series slice",
                expected_shape=f"({self.original_length},)",
                actual_shape=source.shape,
            )

        transformed = np.full((self.all_leap_length, *source.shape[1:]), np.nan, dtype=float)
        source_index = 0
        target_index = 0

        for year_offset, observed_days in enumerate(self.observed_days_by_year):
            year = self.year_start + year_offset
            source_year = source[source_index : source_index + observed_days]

            if calendar.isleap(year):
                transformed[target_index : target_index + observed_days] = source_year
            else:
                days_before_february_29 = min(observed_days, 59)
                transformed[target_index : target_index + days_before_february_29] = source_year[
                    :days_before_february_29
                ]
                if observed_days > 59:
                    transformed[target_index + 59] = (source_year[58] + source_year[59]) / 2
                    transformed[target_index + 60 : target_index + observed_days + 1] = source_year[59:]

            source_index += observed_days
            target_index += 366

        return transformed

    def to_gregorian(self, values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Remove synthetic February 29 values and trim to observed Gregorian days.

        Args:
            values: an array whose leading axis is one complete 366-day record.

        Returns:
            The values in the Gregorian layout, trimmed to the observed days.

        Raises:
            DataShapeError: if the leading axis length is not ``all_leap_length``.
        """
        source = np.asarray(values)
        if source.ndim < 1 or source.shape[0] != self.all_leap_length:
            raise DataShapeError(
                "Daily calendar restoration requires one complete 366-day time-series slice",
                expected_shape=f"({self.all_leap_length},)",
                actual_shape=source.shape,
            )

        restored = np.full((self.original_length, *source.shape[1:]), np.nan, dtype=float)
        source_index = 0
        target_index = 0

        for year_offset, observed_days in enumerate(self.observed_days_by_year):
            year = self.year_start + year_offset
            if calendar.isleap(year):
                restored[target_index : target_index + observed_days] = source[
                    source_index : source_index + observed_days
                ]
            else:
                days_before_february_29 = min(observed_days, 59)
                restored[target_index : target_index + days_before_february_29] = source[
                    source_index : source_index + days_before_february_29
                ]
                if observed_days > 59:
                    restored[target_index + 59 : target_index + observed_days] = source[
                        source_index + 60 : source_index + observed_days + 1
                    ]

            source_index += 366
            target_index += observed_days

        return restored


def transform_to_366day(
    original: np.ndarray,
    year_start: int,
    total_years: int,
) -> np.ndarray:
    """
    Takes an array of daily values with only actual leap years represented
    as 366 day years (non-leap years with 365 days) and converts it to an array
    of daily values represented as containing full 366-day years as if each year
    is a leap year with computed/faux values for the Feb. 29th of each
    non-leap year.

    For example if provided an input array representing two years,
    we expect/assume that it will contain 730 elements if neither of the years
    represented are leap years (as indicated by the year start argument),
    or 731 elements if either of the two years is a leap year (i.e. a year with
    366 days). The resulting/transformed array will contain 732 elements -- 366
    for the leap year plus 366 for the non-leap year, with the element
    that corresponds to Feb. 29th in the non-leap year having a value that's an
    average of the Feb 28th and Mar. 1st values.

    Every year but the last must be complete; a partial final year is padded
    with NaN to a whole 366-day year.

    :param original: 1-D array of daily values
    :param year_start: the year corresponding to the initial year of the input
        array, used to determine whether  each increment of daily values
        represents an actual leap year
    :param total_years: the total number of years represented by the input array
    :return: 1-D array of values with size (total_years * 366)
    """
    # the Gregorian values are laid out one year after another, with only the
    # final year permitted to be partial; the plan pads it to a whole 366-day year
    if len(original.shape) > 1:
        message = "Invalid input array: only 1-D arrays are supported"
        _logger.error(
            "array_transformation_error",
            operation="transform_to_366day",
            reason="only_1d_supported",
            actual_shape=str(original.shape),
        )
        raise ValueError(message)

    if year_start < 1:
        raise ValueError("Invalid year start: years must be positive")
    if total_years < 0:
        raise ValueError("Invalid total years: must not be negative")
    if total_years == 0:
        # preserved legacy behavior: no years requested, no values returned
        return np.full((0,), np.nan)
    if len(original) == 0:
        raise ValueError("Invalid input array: an empty array cannot represent any year")

    plan = DailyCalendarPlan.from_year_span(year_start, total_years, len(original))
    if plan.original_length != len(original):
        # more values than the declared span can hold
        raise ValueError("Incompatible shapes")
    for offset, observed_days in enumerate(plan.observed_days_by_year[:-1]):
        days_in_year = 366 if calendar.isleap(year_start + offset) else 365
        if observed_days != days_in_year:
            # only the final year may be short; an earlier gap means the input
            # length contradicts the declared span
            raise ValueError("Incompatible shapes")

    all_leap = plan.to_all_leap(original)

    _logger.debug(
        "array_transformation_completed",
        operation="transform_to_366day",
        input_size=original.size,
        output_size=all_leap.size,
    )
    return all_leap


def transform_to_gregorian(
    original: np.ndarray,
    year_start: int,
) -> np.ndarray:
    """
    Takes an array of daily values represented as full 366-day years (as if each
    year is a leap year with fill/faux values for the Feb. 29th of each non-leap
    year) and converts it to an array of daily values with only actual leap
    years represented as 366 day years.

    For example if provided an input array representing two years,
    we expect/assume that it will contain 732 elements corresponding to two
    years with 366 days. Two possible transformation results are possible:

    1) If the start year or the following year is a leap year then
    the resulting/transformed array will contain 731 elements (366 for
    the leap year plus 365 for the non-leap year), with the element
    that corresponded to Feb. 29th in the non-leap year removed.

    2) If both years represented are non-leap years, as determined by
    the starting year argument, then the resulting/transformed array will
    contain 730 elements (365 days for both non-leap years), with the
    elements that corresponded to Feb. 29th removed.

    :param original: 1-D array of daily values, total size should be
        a multiple of 366
    :param year_start: the year corresponding to the initial year (first 366
        values) of the input array, used to determine whether each 366
        increment of daily values represents an actual leap year
    """
    # the input is the NumPy core's 366-day layout, one whole year per 366 values
    if len(original.shape) > 1:
        message = "Invalid input array: only 1-D arrays are supported"
        _logger.error(message)
        raise ValueError(message)
    if original.size % 366 != 0:
        message = "Invalid input array: only 1-D arrays containing " + "multiples of 366 days are supported"
        _logger.error(message)
        raise ValueError(message)
    if not isinstance(year_start, (int, np.integer)):
        raise TypeError("Invalid year start: year must be an integer")
    if year_start < 1:
        raise ValueError("Invalid year start: years must be positive")

    total_years = int(original.size / 366)
    # the input's 366-day layout gives every year a full Gregorian complement
    plan = DailyCalendarPlan(
        year_start,
        tuple(366 if calendar.isleap(year) else 365 for year in range(year_start, year_start + total_years)),
    )
    gregorian = plan.to_gregorian(original)

    _logger.debug(
        "array_transformation_completed",
        operation="transform_to_gregorian",
        input_size=original.size,
        output_size=gregorian.size,
    )
    return gregorian
