import logging
from math import exp, lgamma, pi, sqrt

import numpy as np
from climate_indices import utils

# ------------------------------------------------------------------------------
# Retrieve logger and set desired logging level
_logger = utils.get_logger(__name__, logging.WARN)


# ------------------------------------------------------------------------------
def fit(timeseries: np.ndarray) -> dict:
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
        message = "Unable to calculate Pearson Type III parameters " + \
                  "due to invalid L-moments"
        _logger.error(message)
        raise ValueError(message)

    return _estimate_pearson3_parameters(lmoments)


# ------------------------------------------------------------------------------
def _estimate_pearson3_parameters(lmoments: np.ndarray) -> dict:
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

    c1 = 0.2906
    c2 = 0.1882
    c3 = 0.0442
    d1 = 0.36067
    d2 = -0.59567
    d3 = 0.25361
    d4 = -2.78861
    d5 = 2.56096
    d6 = -0.77045
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
    # pearson3_parameters = np.zeros((3,))

    # # the first Pearson Type III parameter is the same as the first L-moment
    # pearson3_parameters[0] = lmoments[0]

    if t3 <= 1e-6:
        # skewness is effectively zero
        scale = lmoments[1] * sqrt(pi)
        skew = 0.0
        # pearson3_parameters[1] = lmoments[1] * sqrt(pi)

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
        # pearson3_parameters[1] = beta * alpha_root

        # the sign of the third L-moment determines
        # the sign of the third Pearson Type III parameter
        if lmoments[2] < 0:
            skew = -2.0 / alpha_root
            # pearson3_parameters[2] = -2.0 / alpha_root
        else:
            skew = 2.0 / alpha_root
            # pearson3_parameters[2] = 2.0 / alpha_root

    # return pearson3_parameters
    return {"loc": loc, "skew": skew, "scale": scale}


# ------------------------------------------------------------------------------
# @numba.jit
def _estimate_lmoments(values: np.ndarray) -> np.ndarray:
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
    if number_of_values < 4:
        message = "Insufficient number of values to perform sample L-moments estimation"
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
    z = float(number_of_values)
    sums[0] = sums[0] / z
    for j in range(1, 3):
        y = y - 1.0
        z = z * y
        sums[j] = sums[j] / z

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
