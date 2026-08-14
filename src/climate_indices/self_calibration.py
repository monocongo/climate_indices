"""Statistics for the self-calibrating Palmer Drought Severity Index (scPDSI).

Pure functions implementing the Wells, Goddard & Wilhite (2004) self-calibration
procedure's statistical building blocks: the rolling-window extreme Z-index sum,
the correlation-adaptive least-squares duration-factor fit, and the order
statistic the reference implementation uses for percentile rescaling.

Every function here is a faithful port of the reference algorithm, including
behaviours that differ from what an idiomatic numpy implementation would do
(truncating 1-indexed order statistics rather than interpolating percentiles,
an asymmetric outlier filter applied to the wet side only, and a regression
intercept anchored on the most extreme residual rather than on the mean). Those
divergences are deliberate and are noted at each site; changing them would break
numeric parity with the published algorithm.

These functions are independent of the ``data``-dict machinery in
``climate_indices.palmer`` and take no calibration-period arguments -- callers
slice the Z-index series to the calibration window before calling.
"""

import math
from collections import deque

import numpy as np

from climate_indices.exceptions import InvalidArgumentError

__all__ = [
    "DRY_SIGN",
    "DURATION_FACTOR_WINDOW_LENGTHS",
    "WET_SIGN",
    "duration_factors",
    "extreme_z_sum",
    "kth_smallest",
    "least_squares_fit",
    "nan_safe_percentile",
]

# Sign conventions used throughout: the self-calibration procedure fits wet and
# dry duration factors separately, and several formulas are sign-weighted so
# that "more extreme" means larger for wet spells and smaller for dry ones.
WET_SIGN = 1
DRY_SIGN = -1

# A wet-side rolling sum is treated as a "freak anomaly" -- and excluded from
# the duration-factor fit -- once its ratio to the 98th-percentile rolling sum
# reaches this tolerance.
_EXTREME_PERCENTILE = 0.98
_REASONABLE_TOLERANCE = 1.25

# The duration-factor regression drops trailing points until the sign-weighted
# correlation clears this tolerance, down to a floor of four retained points.
_CORRELATION_TOLERANCE = 0.85
_MIN_REGRESSION_POINTS = 4

# Spell durations, in months, sampled by the duration-factor regression. The
# reference implementation also carries weekly sets; Palmer indices here are
# monthly-only, so only the monthly set is ported.
DURATION_FACTOR_WINDOW_LENGTHS: tuple[int, ...] = (3, 6, 9, 12, 18, 24, 30, 36, 42, 48)

# The fitted duration-factor line represents a PDSI value of +/-4, the
# conventional extreme threshold; normalizing by it rescales the regression into
# the slope/intercept pair the PDSI recursion consumes.
_PDSI_ANCHOR = 4.0


def _validate_sign(sign: int) -> None:
    """Reject a spell sign that is neither wet nor dry.

    Args:
        sign: The spell sign to validate.

    Raises:
        InvalidArgumentError: If sign is not WET_SIGN or DRY_SIGN.
    """
    if sign not in (WET_SIGN, DRY_SIGN):
        raise InvalidArgumentError(
            f"invalid spell sign: {sign}",
            argument_name="sign",
            argument_value=str(sign),
            valid_values=f"{WET_SIGN} (wet) or {DRY_SIGN} (dry)",
        )


def kth_smallest(values: np.ndarray, k: int) -> float:
    """Select the kth smallest value, 1-indexed.

    Ports the reference implementation's ``kthLargest()``, which despite its
    name selects in ascending order (k=1 is the minimum, k=size is the maximum)
    and returns its missing-value sentinel for an out-of-range k. No missing
    value handling is done here -- see :func:`nan_safe_percentile` for that.

    Args:
        values: The values to select from.
        k: The 1-indexed rank to select.

    Returns:
        The kth smallest value, or NaN if k is outside [1, len(values)].
    """
    array = np.asarray(values, dtype=float)
    if k < 1 or k > array.size:
        return float("nan")
    # np.partition returns a new array, so the caller's array is left untouched
    return float(np.partition(array, k - 1)[k - 1])


def nan_safe_percentile(values: np.ndarray, fraction: float) -> float:
    """Select the percentile order statistic the reference implementation uses.

    Ports ``safe_percentile()``: drop missing values, then take the
    ``int(fraction * count)``th smallest of what remains. The index is
    *truncated*, and the result is an actual data value -- this is deliberately
    not ``numpy.percentile``, which interpolates between neighbouring values and
    would shift the self-calibration ratios.

    Args:
        values: The values to rank; NaN entries are treated as missing.
        fraction: The percentile as a fraction within [0.0, 1.0].

    Returns:
        The selected value, or NaN if no value qualifies -- either because every
        input was missing, or because the truncated index came out as 0 (which
        happens when the surviving count is small relative to the fraction).

    Raises:
        InvalidArgumentError: If fraction is outside [0.0, 1.0].
    """
    if not 0.0 <= fraction <= 1.0:
        raise InvalidArgumentError(
            f"percentile fraction outside the unit interval: {fraction}",
            argument_name="fraction",
            argument_value=str(fraction),
            valid_values="a float within [0.0, 1.0]",
        )

    array = np.asarray(values, dtype=float)
    present = array[~np.isnan(array)]
    return kth_smallest(present, int(fraction * present.size))


def _highest_reasonable(sums: np.ndarray, sign: int) -> float:
    """Select the most extreme rolling sum that is not a freak anomaly.

    Applies the reference implementation's outlier filter: among the sums whose
    own sign matches the spell being fitted, keep only those whose ratio to the
    98th-percentile sum stays below the reasonableness tolerance, and return the
    most extreme survivor.

    Args:
        sums: The rolling sums to filter.
        sign: Only WET_SIGN is ever passed in practice -- :func:`extreme_z_sum`
            returns before calling this on the dry side, since the reference's
            dry path is deliberately unfiltered. ``_EXTREME_PERCENTILE`` is
            hard-wired to the wet side's 98th-percentile threshold; the
            reference's dry-side filter, if it were wired up here, would need
            a 2nd-percentile threshold instead, so passing DRY_SIGN would
            silently apply the wrong filter rather than the reference's "no
            filter at all".

    Returns:
        The most extreme surviving sum, or 0.0 if nothing survives.
    """
    threshold = nan_safe_percentile(sums, _EXTREME_PERCENTILE)

    highest = 0.0
    for value in sums:
        if sign * value <= 0.0:
            continue
        if math.isnan(threshold):
            # Too few sums for a percentile to exist. The reference divides by
            # its missing-value sentinel here, which lets every candidate
            # through, so the filter effectively does not apply.
            is_reasonable = True
        elif math.isclose(threshold, 0.0, rel_tol=0.0, abs_tol=0.0):
            # Use an exact zero tolerance here: the percentile is an observed
            # value rather than an interpolated result, and a nonzero threshold
            # must still participate in the reference's ratio test.
            is_reasonable = False
        else:
            is_reasonable = (value / threshold) < _REASONABLE_TOLERANCE
        if is_reasonable and sign * value > sign * highest:
            highest = value
    return highest


def extreme_z_sum(z_values: np.ndarray, window_length: int, sign: int) -> float:
    """Compute the representative extreme rolling Z-index sum for one window length.

    Ports ``get_Z_sum()``. Slides a window of ``window_length`` non-missing
    Z-index values across the series and picks one representative extreme sum,
    asymmetrically by spell sign: the dry side takes the single most negative
    rolling sum unfiltered, while the wet side discards freak anomalies first
    (see :func:`_highest_reasonable`). The asymmetry is a property of the
    published algorithm, not an oversight.

    Missing periods never enter the window: during the initial fill they are
    retried rather than consuming a slot, and during the slide they leave the
    window and the running sum untouched.

    Callers are responsible for restricting ``z_values`` to the calibration
    period before calling.

    Args:
        z_values: The Z-index series, in chronological order; NaN means missing.
        window_length: The number of non-missing periods in the rolling window.
        sign: WET_SIGN or DRY_SIGN.

    Returns:
        The representative extreme rolling sum for this window length. On the
        wet side this is 0.0 when no rolling sum survives the anomaly filter,
        or when the series has fewer than ``window_length`` non-missing
        values and no complete window ever forms. On the dry side the latter
        case returns NaN instead, since the dry path has no 0.0 floor to fall
        back on.

    Raises:
        InvalidArgumentError: If sign is invalid or window_length is not a
            positive integer.
    """
    _validate_sign(sign)
    if window_length < 1 or not float(window_length).is_integer():
        raise InvalidArgumentError(
            f"invalid rolling window length: {window_length}",
            argument_name="window_length",
            argument_value=str(window_length),
            valid_values="a positive integer",
        )

    series = np.asarray(z_values, dtype=float)
    window: deque[float] = deque()
    running = 0.0
    index = 0

    # fill the initial window, retrying past missing periods
    while len(window) < window_length and index < series.size:
        value = float(series[index])
        index += 1
        if not math.isnan(value):
            running += value
            window.append(value)

    if len(window) < window_length:
        # the series ended before a complete window could be formed, so there
        # is no rolling sum to report -- not even the partial one just
        # accumulated, which would be a shorter-than-window_length sum
        return 0.0 if sign == WET_SIGN else float("nan")

    extreme = running
    sums = np.empty(series.size + 1, dtype=float)
    sum_count = 1
    sums[0] = running

    # slide the window forward one non-missing period at a time. The subtract-
    # then-add ordering below mirrors the reference's accumulation order and is
    # deliberate: reassociating it as `running += value - old_value` changes
    # the rounding, which PR4 compares against a C++ oracle at ATOL=5e-5.
    while index < series.size:
        value = float(series[index])
        index += 1
        if not math.isnan(value):
            running -= window.popleft()
            running += value
            window.append(value)
            sums[sum_count] = running
            sum_count += 1
        if sign * running > sign * extreme:
            extreme = running

    if sign == DRY_SIGN:
        return extreme
    return _highest_reasonable(sums[:sum_count], sign)


def least_squares_fit(x: np.ndarray, y: np.ndarray, sign: int) -> tuple[float, float]:
    """Fit a line by the reference implementation's adaptive least squares.

    Ports ``LeastSquares()``, which differs from textbook OLS in two ways. It
    trims trailing points until the sign-weighted correlation clears 0.85 (down
    to a floor of four retained points), and it anchors the intercept on the
    retained point with the most extreme sign-weighted residual rather than on
    the sample means. Both are deliberate properties of the published
    self-calibration procedure.

    Args:
        x: The independent variable (window lengths, in the duration-factor fit).
        y: The dependent variable (extreme Z-index sums), same length as x.
        sign: WET_SIGN or DRY_SIGN, selecting the direction of "most extreme".

    Returns:
        A tuple of (slope, intercept) for the fitted line.

    Raises:
        InvalidArgumentError: If sign is invalid, the inputs differ in length,
            or fewer than four points were supplied.
        ZeroDivisionError: If the retained ``y`` values are all exactly 0.0,
            leaving the sum-of-squares ``ss_y`` exactly zero.
        ValueError: If the retained ``y`` values are all equal to some other
            constant. Because the sums of squares are accumulated
            sequentially rather than via a pairwise-summation reduction,
            ``ss_y`` typically comes out very slightly negative rather than
            exactly zero in this case, so ``math.sqrt(ss_y)`` raises
            ``math domain error`` before the division is reached. Both cases
            are deliberately unguarded here -- see #720.
    """
    _validate_sign(sign)

    abscissa = np.asarray(x, dtype=float)
    ordinate = np.asarray(y, dtype=float)
    if abscissa.size != ordinate.size:
        raise InvalidArgumentError(
            f"mismatched input lengths: {abscissa.size} and {ordinate.size}",
            argument_name="y",
            argument_value=str(ordinate.size),
            valid_values=f"an array of length {abscissa.size}, matching x",
        )
    if abscissa.size < _MIN_REGRESSION_POINTS:
        raise InvalidArgumentError(
            f"too few points to fit: {abscissa.size}",
            argument_name="x",
            argument_value=str(abscissa.size),
            valid_values=f"at least {_MIN_REGRESSION_POINTS} points",
        )

    # accumulate sequentially rather than via ndarray.sum(), whose pairwise
    # summation would reassociate the additions and perturb the low-order bits
    count = abscissa.size
    sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0
    for index in range(count):
        this_x = float(abscissa[index])
        this_y = float(ordinate[index])
        sum_x += this_x
        sum_y += this_y
        sum_x2 += this_x * this_x
        sum_y2 += this_y * this_y
        sum_xy += this_x * this_y

    ss_x = sum_x2 - (sum_x * sum_x) / count
    ss_y = sum_y2 - (sum_y * sum_y) / count
    ss_xy = sum_xy - (sum_x * sum_y) / count
    correlation = ss_xy / (math.sqrt(ss_x) * math.sqrt(ss_y))

    # drop points from the end until the fit correlates well enough, or until
    # only the minimum number of points is left
    last = count - 1
    while sign * correlation < _CORRELATION_TOLERANCE and last > _MIN_REGRESSION_POINTS - 1:
        this_x = float(abscissa[last])
        this_y = float(ordinate[last])
        sum_x -= this_x
        sum_y -= this_y
        sum_x2 -= this_x * this_x
        sum_y2 -= this_y * this_y
        sum_xy -= this_x * this_y

        ss_x = sum_x2 - (sum_x * sum_x) / last
        ss_y = sum_y2 - (sum_y * sum_y) / last
        ss_xy = sum_xy - (sum_x * sum_y) / last
        correlation = ss_xy / (math.sqrt(ss_x) * math.sqrt(ss_y))
        last -= 1

    slope = ss_xy / ss_x

    # translate the line through the retained point whose sign-weighted residual
    # is most extreme. The initial anchor of (x[0], 0.0) is the reference's, and
    # survives whenever no residual is strictly more extreme than zero -- which
    # happens for retained points that are collinear through the origin.
    retained = last + 1
    max_residual = 0.0
    anchor_x = float(abscissa[0])
    anchor_y = 0.0
    for index in range(retained):
        residual = float(ordinate[index]) - slope * float(abscissa[index])
        if sign * residual > sign * max_residual:
            max_residual = residual
            anchor_x = float(abscissa[index])
            anchor_y = float(ordinate[index])

    return float(slope), float(anchor_y - slope * anchor_x)


def duration_factors(z_values: np.ndarray, sign: int) -> tuple[float, float]:
    """Fit the wet or dry duration factors for one location's Z-index series.

    Ports ``CalcDurFact()``, the self-calibration step that replaces Palmer's
    (1965) fixed national duration-factor constants with values fitted to the
    location at hand. One representative extreme rolling Z sum is taken per
    spell duration, a line is fitted through those ten points, and the result is
    normalized against a PDSI value of +/-4.

    Callers are responsible for restricting ``z_values`` to the calibration
    period before calling.

    Args:
        z_values: The raw Z-index series for the calibration period, in
            chronological order; NaN means missing.
        sign: WET_SIGN to fit wet-spell factors, DRY_SIGN for dry-spell factors.

    Returns:
        A tuple of (m, b) -- the duration-factor slope and intercept, in the
        form the PDSI recursion consumes as wetm/wetb or drym/dryb.

        Callers that go on to divide by ``m + b`` (as the PDSI recursion
        does) should not assume that sum is positive. On the wet side,
        :func:`extreme_z_sum` floors to 0.0 whenever nothing survives its
        anomaly filter for a given window length; a calibration window
        skewed dry can floor several of the ten window lengths, and those
        0.0s are fed to the regression as if they were real data, which can
        pull the fitted slope negative. Symmetrically, the dry side's
        unfiltered extreme sums can come out positive for an all-wet series.
        Both are faithful to the reference; a consumer must handle
        ``m + b <= 0`` rather than assume it away.

    Raises:
        InvalidArgumentError: If sign is invalid.
    """
    _validate_sign(sign)

    series = np.asarray(z_values, dtype=float)
    window_lengths = np.array(DURATION_FACTOR_WINDOW_LENGTHS, dtype=float)
    extreme_sums = np.array([extreme_z_sum(series, length, sign) for length in DURATION_FACTOR_WINDOW_LENGTHS])

    slope, intercept = least_squares_fit(window_lengths, extreme_sums, sign)

    anchor = sign * _PDSI_ANCHOR
    return slope / anchor, intercept / anchor
