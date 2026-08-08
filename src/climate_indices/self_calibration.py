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

import numpy as np

from climate_indices.exceptions import InvalidArgumentError

__all__ = [
    "DRY_SIGN",
    "WET_SIGN",
    "kth_smallest",
    "nan_safe_percentile",
]

# Sign conventions used throughout: the self-calibration procedure fits wet and
# dry duration factors separately, and several formulas are sign-weighted so
# that "more extreme" means larger for wet spells and smaller for dry ones.
WET_SIGN = 1
DRY_SIGN = -1


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
