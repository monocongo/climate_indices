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

import numpy as np

from climate_indices.exceptions import InvalidArgumentError

__all__ = [
    "DRY_SIGN",
    "WET_SIGN",
    "extreme_z_sum",
    "kth_smallest",
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


def _highest_reasonable(sums: list[float], sign: int) -> float:
    """Select the most extreme rolling sum that is not a freak anomaly.

    Applies the reference implementation's outlier filter: among the sums whose
    own sign matches the spell being fitted, keep only those whose ratio to the
    98th-percentile sum stays below the reasonableness tolerance, and return the
    most extreme survivor.

    Args:
        sums: The rolling sums to filter.
        sign: WET_SIGN or DRY_SIGN, selecting which direction counts as extreme.

    Returns:
        The most extreme surviving sum, or 0.0 if nothing survives.
    """
    threshold = nan_safe_percentile(np.array(sums, dtype=float), _EXTREME_PERCENTILE)

    highest = 0.0
    for value in sums:
        if sign * value <= 0.0:
            continue
        if math.isnan(threshold):
            # Too few sums for a percentile to exist. The reference divides by
            # its missing-value sentinel here, which lets every candidate
            # through, so the filter effectively does not apply.
            is_reasonable = True
        elif threshold == 0.0:
            # The reference's division yields infinity, failing the ratio test
            # for every candidate.
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
        wet side this is 0.0 when no rolling sum survives the anomaly filter.

    Raises:
        InvalidArgumentError: If sign is invalid or window_length is not positive.
    """
    _validate_sign(sign)
    if window_length < 1:
        raise InvalidArgumentError(
            f"invalid rolling window length: {window_length}",
            argument_name="window_length",
            argument_value=str(window_length),
            valid_values="a positive integer",
        )

    series = np.asarray(z_values, dtype=float)
    window: list[float] = []
    running = 0.0
    index = 0

    # fill the initial window, retrying past missing periods
    while len(window) < window_length and index < series.size:
        value = float(series[index])
        index += 1
        if not math.isnan(value):
            running += value
            window.append(value)

    extreme = running
    sums = [running]

    # slide the window forward one non-missing period at a time. The subtract-
    # then-add ordering below mirrors the reference's accumulation order and is
    # deliberate: reassociating it as `running += value - window.pop(0)` changes
    # the rounding, which PR4 compares against a C++ oracle at ATOL=5e-5.
    while index < series.size:
        value = float(series[index])
        index += 1
        if not math.isnan(value):
            running -= window.pop(0)
            running += value
            window.append(value)
            sums.append(running)
        if sign * running > sign * extreme:
            extreme = running

    if sign == DRY_SIGN:
        return extreme
    return _highest_reasonable(sums, sign)
