"""Computation of L-moments used for Pearson Type-III distribution fitting"""

import logging
from math import exp, lgamma, pi, sqrt

import numpy as np
from scipy import special

from climate_indices import utils

# declare the function names that should be included in the public API for this module
__all__ = ["fit"]

# Retrieve logger and set desired logging level
_logger = utils.get_logger(__name__, logging.WARN)

# Configuration constants for L-moments computation
# Minimum number of non-NaN values required for L-moments estimation
MIN_VALUES_FOR_LMOMENTS = 4

# Pearson Type III parameter-estimation coefficients from the Hosking RC20525
# 'pearson3' subroutine, shared by the single-series and cell-axis fits
# (c1, c2, c3, d1, d2, d3, d4, d5, d6)
_PEARSON3_COEFFICIENTS = (
    0.2906,
    0.1882,
    0.0442,
    0.36067,
    -0.59567,
    0.25361,
    -2.78861,
    2.56096,
    -0.77045,
)


def fit(timeseries: np.ndarray) -> dict[str, float]:
    """
    Returns the L-Moments fit (loc, scale, skew) corresponding to the
    input array of values.

    :param timeseries:
    :return:
    """

    # estimate the L-moments of the values array
    lmoments = _estimate_lmoments(timeseries)

    # validate the L-Moments
    if (lmoments[1] <= 0.0) or (abs(lmoments[2]) >= 1.0):
        message = "Unable to calculate Pearson Type III parameters " + "due to invalid L-moments"
        _logger.error(message)
        raise ValueError(message)

    return _estimate_pearson3_parameters(lmoments)


def _estimate_pearson3_parameters(lmoments: np.ndarray) -> dict[str, float]:
    """
    Estimate parameters via L-moments for the Pearson Type III distribution,
    based on Fortran code written for inclusion in IBM Research Report RC20525,
    'FORTRAN ROUTINES FOR USE WITH THE METHOD OF L-MOMENTS, VERSION 3'
    by J. R. M. Hosking, IBM Research Division, T. J. Watson Research Center,
    Yorktown Heights, NY 10598

    This is a Python translation of the original Fortran subroutine
    named 'pearson3'.

    :param lmoments: 3-element, 1-D (flat) array containing the first
        three L-moments (lambda-1, lambda-2, and tau-3)
    :return the Pearson Type III parameters corresponding to the input L-moments
    :rtype: a 3-element, 1-D (flat) numpy array of floats (loc, scale, skew)
    """

    c1, c2, c3, d1, d2, d3, d4, d5, d6 = _PEARSON3_COEFFICIENTS
    t3 = abs(lmoments[2])  # L-skewness?

    # ensure the validity of the L-moments
    if (lmoments[1] <= 0) or (t3 >= 1):
        message = "Unable to calculate Pearson Type III parameters due to invalid L-moments"
        _logger.error(message)
        raise ValueError(message)

    # initialize the output values
    # loc, scale, skew

    # the first Pearson Type III parameter is the same as the first L-moment
    loc = lmoments[0]

    # # the first Pearson Type III parameter is the same as the first L-moment

    if t3 <= 1e-6:
        # skewness is effectively zero
        scale = lmoments[1] * sqrt(pi)
        skew = 0.0

    else:
        if t3 < 0.333333333:
            t = pi * 3 * t3 * t3
            alpha = (1.0 + (c1 * t)) / (t * (1.0 + (t * (c2 + (t * c3)))))
        else:
            t = 1.0 - t3
            alpha = t * (d1 + (t * (d2 + (t * d3)))) / (1.0 + (t * (d4 + (t * (d5 + (t * d6))))))

        alpha_root = sqrt(alpha)
        beta = sqrt(pi) * lmoments[1] * exp(lgamma(alpha) - lgamma(alpha + 0.5))
        scale = beta * alpha_root

        # the sign of the third L-moment determines
        # the sign of the third Pearson Type III parameter
        if lmoments[2] < 0:
            skew = -2.0 / alpha_root
        else:
            skew = 2.0 / alpha_root

    return {"loc": loc, "skew": skew, "scale": scale}


def fit_spatial(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the L-Moments fits (loc, scale, skew) for every cell of an array whose
    first axis is the sample axis.

    Cell-axis counterpart of :func:`fit`: instead of raising on the first cell whose
    sample is too short or whose L-moments are invalid, the invalid cells are marked
    in the returned validity mask so the caller can apply its own fallback.

    :param values: array of samples with shape (samples, *cells)
    :return: tuple of (loc, scale, skew, valid), each array shaped like values.shape[1:]
    """
    lmoments, valid = _estimate_lmoments_spatial(values)
    locs, scales, skews = _estimate_pearson3_parameters_spatial(lmoments, valid)
    return locs, scales, skews, valid


def _estimate_pearson3_parameters_spatial(
    lmoments: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cell-axis counterpart of :func:`_estimate_pearson3_parameters`.

    Every branch is evaluated for every cell with NumPy operations and the result is
    masked by the validity of the cell's L-moments, rather than returning early.

    :param lmoments: array of the first three L-moments, shaped (3, *cells)
    :param valid: boolean array shaped (*cells) marking usable L-moments
    :return: tuple of (loc, scale, skew) arrays shaped (*cells); invalid cells are zero
    """
    c1, c2, c3, d1, d2, d3, d4, d5, d6 = _PEARSON3_COEFFICIENTS
    locs = lmoments[0]
    second_lmoment = lmoments[1]
    t3 = np.abs(lmoments[2])
    valid = valid & (second_lmoment > 0) & (t3 < 1.0)

    zero_skew = t3 <= 1e-6
    low_skew = (~zero_skew) & (t3 < 0.333333333)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_low = pi * 3 * t3 * t3
        alpha_low = (1.0 + (c1 * t_low)) / (t_low * (1.0 + (t_low * (c2 + (t_low * c3)))))
        t_high = 1.0 - t3
        alpha_high = (
            t_high * (d1 + (t_high * (d2 + (t_high * d3)))) / (1.0 + (t_high * (d4 + (t_high * (d5 + (t_high * d6))))))
        )
        alpha = np.where(zero_skew, 0.0, np.where(low_skew, alpha_low, alpha_high))
        alpha_root = np.sqrt(alpha)
        beta = np.sqrt(pi) * second_lmoment * np.exp(special.gammaln(alpha) - special.gammaln(alpha + 0.5))
        scales = np.where(zero_skew, second_lmoment * sqrt(pi), beta * alpha_root)
        skews = np.where(zero_skew, 0.0, np.where(lmoments[2] < 0, -2.0 / alpha_root, 2.0 / alpha_root))

    return (
        np.where(valid, locs, 0.0),
        np.where(valid, scales, 0.0),
        np.where(valid, skews, 0.0),
    )


def _estimate_lmoments_spatial(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Cell-axis counterpart of :func:`_estimate_lmoments`.

    The per-series accumulator is a sequential loop over the sorted sample ranks; the
    same loop runs here with every cell evaluated at once, so each cell's additions
    happen in the same order as the single-series fit.

    :param values: array of samples with shape (samples, *cells)
    :return: tuple of (lmoments, valid) with lmoments shaped (3, *cells) and valid
        shaped (*cells)
    """
    values = np.asarray(values, dtype=float)
    number_of_values = np.count_nonzero(~np.isnan(values), axis=0)
    sorted_values = np.sort(values, axis=0)

    sample_count, *cell_shape = values.shape
    sums = np.zeros((3, *cell_shape))
    ranks = np.arange(sample_count)
    in_sample = ranks.reshape((sample_count,) + (1,) * len(cell_shape)) < number_of_values
    for rank in ranks:
        ranked_value = np.where(in_sample[rank], sorted_values[rank], 0.0)
        sums[0] = sums[0] + ranked_value
        first_term = ranked_value * rank
        sums[1] = sums[1] + first_term
        sums[2] = sums[2] + first_term * (rank - 1)

    counts = number_of_values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sums[0] = sums[0] / counts
        y = counts - 1.0
        z_val = counts * y
        sums[1] = sums[1] / z_val
        y = y - 1.0
        z_val = z_val * y
        sums[2] = sums[2] / z_val

    k = 3
    p0 = -1.0
    for _ in range(2):
        ak = float(k)
        p0 = -p0
        p = p0
        temp = p * sums[0]
        for i in range(1, k):
            ai = float(i)
            p = -p * (ak + ai - 1.0) * (ak - ai) / (ai * ai)
            temp = temp + (p * sums[i])
        sums[k - 1] = temp
        k = k - 1

    lmoments = np.zeros((3, *cell_shape))
    valid = number_of_values >= MIN_VALUES_FOR_LMOMENTS
    valid = valid & (sums[1] != 0)
    lmoments[0] = np.where(valid, sums[0], 0.0)
    lmoments[1] = np.where(valid, sums[1], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lmoments[2] = np.where(valid, sums[2] / sums[1], 0.0)
    return lmoments, valid


def _estimate_lmoments(
    values: np.ndarray,
) -> np.ndarray:
    """
    Estimate sample L-moments, based on Fortran code written for inclusion
    in IBM Research Report RC20525,
    'FORTRAN ROUTINES FOR USE WITH THE METHOD OF L-MOMENTS, VERSION 3'
    by J. R. M. Hosking, IBM Research Division,
    T. J. Watson Research Center, Yorktown Heights, NY 10598, Version 3 August 1996.

    Documentation on the original Fortran routines found here:
        https://rdrr.io/cran/nsRFA/man/HW.original.html

    This is a Python translation of the original Fortran subroutine SAMLMR()
    and which has been optimized for calculating only the first three L-moments.

    :param values: 1-D (flattened) array of float values
    :return: an estimate of the first three sample L-moments
    :rtype: 1-D numpy array of floats (the first three sample L-moments
        corresponding to the input values)
    """

    # we need to have at least four values in order
    # to make a sample L-moments estimation
    number_of_values = np.count_nonzero(~np.isnan(values))
    if number_of_values < MIN_VALUES_FOR_LMOMENTS:
        message = (
            "Insufficient number of values to perform sample L-moments estimation: "
            f"{number_of_values} non-NaN values found (minimum {MIN_VALUES_FOR_LMOMENTS} required). "
            "This commonly occurs in dry regions with extensive zero precipitation. "
            "Consider using Gamma distribution instead of Pearson Type III for such areas."
        )
        _logger.warning(message)
        raise ValueError(message)

    # sort the values into ascending order
    values = np.sort(values)

    sums = np.zeros((3,))

    for i in range(1, number_of_values + 1):
        z = i
        term = values[i - 1]
        sums[0] = sums[0] + term
        for j in range(1, 3):
            z -= 1
            term = term * z
            sums[j] = sums[j] + term

    y = float(number_of_values)
    z_val: float = float(number_of_values)
    sums[0] = sums[0] / z_val
    for j in range(1, 3):
        y = y - 1.0
        z_val = z_val * y
        sums[j] = sums[j] / z_val

    k = 3
    p0 = -1.0
    for _ in range(2):
        ak = float(k)
        p0 = -p0
        p = p0
        temp = p * sums[0]
        for i in range(1, k):
            ai = i
            p = -p * (ak + ai - 1.0) * (ak - ai) / (ai * ai)
            temp = temp + (p * sums[i])
        sums[k - 1] = temp
        k = k - 1

    lmoments = np.zeros((3,))
    if sums[1] != 0:
        lmoments[0] = sums[0]
        lmoments[1] = sums[1]
        lmoments[2] = sums[2] / sums[1]

    return lmoments
