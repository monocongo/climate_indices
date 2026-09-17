"""Compute palmer drought indices"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from structlog.stdlib import BoundLogger

from climate_indices import _palmer_wells, self_calibration, utils
from climate_indices._palmer_duration import DurationFactors
from climate_indices.exceptions import ConvergenceError
from climate_indices.logging_config import get_logger

_logger = get_logger(__name__)

# declare the function names that should be included in the public API for this module
__all__ = ["pdsi", "scpdsi"]

AWCTOP = 1.0
K8_SIZE = 40


class _PalmerResult(NamedTuple):
    """Output of a prepared Palmer calculation: the four indices and their parameters.

    ``pdsi`` is the index the calculation produced: PDSI for :func:`pdsi`,
    scPDSI for :func:`scpdsi`. The standard path derives the PMDI through the
    statement recursion (``wplm``); scPDSI takes it from the Wells recursion
    (``pmdi``).
    """

    pdsi: np.ndarray
    phdi: np.ndarray
    pmdi: np.ndarray
    zindex: np.ndarray
    params: dict[str, Any] | None


@dataclass
class _PalmerPrepared:
    """The Palmer inputs, water balance, and calibration results shared by both indices.

    Written by ``_prepare_palmer_data`` (plus the index-specific K-factor stage)
    and read-only afterwards, so no recursion stage can overwrite an input.
    """

    # input record and calibration configuration
    precips: np.ndarray
    pet: np.ndarray
    awc: float
    awc_bot: float
    n_years: int
    n_calb_years: int
    calibration_year_initial_idx: int
    calibration_year_final_idx: int
    calibrate: bool

    # water balance: monthly arrays and calibration-period monthly sums
    spdat: np.ndarray
    pldat: np.ndarray
    prdat: np.ndarray
    rdat: np.ndarray
    tldat: np.ndarray
    etdat: np.ndarray
    rodat: np.ndarray
    sssdat: np.ndarray
    ssudat: np.ndarray
    psum: np.ndarray
    spsum: np.ndarray
    petsum: np.ndarray
    plsum: np.ndarray
    prsum: np.ndarray
    rsum: np.ndarray
    tlsum: np.ndarray
    etsum: np.ndarray
    rosum: np.ndarray

    # CAFEC coefficients, moisture-demand ratio, and Z-index weighting factors
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    delta: np.ndarray
    trat: np.ndarray
    ak: np.ndarray

    # duration factors
    wetm: float
    wetb: float
    drym: float
    dryb: float


@dataclass
class _PalmerRecursion:
    """Mutable per-location recursion state and the arrays the recursion fills.

    Constructed from a prepared struct by ``_initialize_recursion``. The
    month-carry scalars that a statement assigns before reading default to
    zero, so the struct exists ahead of the recursion that fills them.
    """

    # recursion state: the K8 window, per-month candidates, and the current severity
    indexj: np.ndarray
    indexm: np.ndarray
    sx: np.ndarray
    sx1: np.ndarray
    sx2: np.ndarray
    sx3: np.ndarray
    ppr: np.ndarray
    px1: np.ndarray
    px2: np.ndarray
    px3: np.ndarray
    x: np.ndarray

    # arrays the recursion and the CAFEC stage write, and the results built from them
    z: np.ndarray
    pdsi: np.ndarray
    phdi: np.ndarray
    wplm: np.ndarray

    # loop control, assigned by the _calc_zindex driver before the recursion runs
    k8: int = 0
    k8max: int = 0
    year: int = 0
    month: int = 0

    # month-carry state a statement assigns before reading it; zero until then
    iass: int = 0
    v: float = 0.0
    pro: float = 0.0
    x1: float = 0.0
    x2: float = 0.0
    x3: float = 0.0
    ze: float = 0.0
    ud: float = 0.0
    uw: float = 0.0
    pv: float = 0.0


def _select_duration_factors(prepared: _PalmerPrepared, state: _PalmerRecursion) -> tuple[float, float]:
    """
    Select the wet or dry duration factors based on the sign of the
    currently-established spell's severity (X3).

    X3 equal to zero means that no wet or dry spell is established. It is
    assigned the wet factors to preserve the recursion's historical
    non-negative tie-break. This choice is immaterial with Palmer's identical
    wet and dry defaults, but must remain explicit when scPDSI supplies distinct
    factors.

    :param prepared: the prepared Palmer inputs
    :param state: the mutable recursion state
    :return a tuple of (m, b) - the duration-factor slope and intercept
    :rtype: tuple[float, float]
    """
    if state.x3 >= 0:
        return prepared.wetm, prepared.wetb
    return prepared.drym, prepared.dryb


def _get_awc_bot(awc: float) -> float:
    """
    Calculate available water capcity in bottom layer

    :param awc: available water capacity (total), in inches
    :return available water capacity (under layer), in inches
    :rtype: float
    """
    return max(awc - AWCTOP, 0.0)


def _calc_potential_loss(
    pet: float,
    ss: float,
    su: float,
    awc: float,
) -> float:
    """
    Calculate potential loss

    :param pet: potential evapotranspiration
    :param ss: surface layer water content, in inches
    :param su: under layer water content, in inches
    :param awc: available water capacity (total), in inches
    :return potential loss
    :rtype: float
    """
    awc_bot = _get_awc_bot(awc)
    if ss >= pet:
        return pet
    return min(ss + su, ((pet - ss) * su) / (awc_bot + AWCTOP) + ss)


def _calc_recharge(
    p: float,
    pet: float,
    ss: float,
    su: float,
    awc: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Calculate recharge, runoff, residual moisture, loss
    to both surface and under layers

    Depends on the starting moisture content and values of
    precipitation and evaporation.

    :param p: preciptiation, in inches
    :param pet: potential evapotranspiration
    :param ss: surface layer water content, in inches
    :param su: under layer water content, in inches
    :param awc: available water capacity (total), in inches
    :return a tuple of floats
        - et: evapotranspiration
        - tl: total loss
        - r: recharge
        - ro: runoff
        - sss: surface layer water content, in inches
        - ssu: under layer water content, in inches
    """
    awc_bot = _get_awc_bot(awc)

    # precipitation exceeds potential evaporation
    if p >= pet:
        et = pet
        tl = 0.0

        # excess precipitation recharges under layer as well as upper
        if (p - pet) > (AWCTOP - ss):
            rs = AWCTOP - ss
            sss = AWCTOP

            # both layers can take the entire excess
            if (p - pet - rs) < (awc_bot - su):
                ru = p - pet - rs
                ro = 0.0

            # some runoff occurs
            else:
                ru = awc_bot - su
                ro = p - pet - rs - ru

            ssu = su + ru
            r = rs + ru

        # only top layer recharged
        else:
            r = p - pet
            sss = ss + p - pet
            ssu = su
            ro = 0.0
    # evaporation exceeds precipitation
    else:
        r = 0.0

        # evaporation from surface layer only
        if ss >= (pet - p):
            sl = pet - p
            sss = ss - sl
            ul = 0.0
            ssu = su

        # evaporation from both layers
        else:
            sl = ss
            sss = 0.0
            ul = min(su, (pet - p - sl) * su / awc)
            ssu = su - ul

        tl = sl + ul
        ro = 0.0
        et = p + sl + ul

    return et, tl, r, ro, sss, ssu


def _calc_cafec_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    both_zero: float = 1.0,
) -> np.ndarray:
    """
    Calculate a CAFEC coefficient as the ratio of two summed water balance terms

    :param numerator: the numerator sums
    :param denominator: the denominator sums
    :param both_zero: value to use when the numerator and denominator are both zero
    :return the per-month ratios
    :rtype: np.ndarray
    """
    values = np.zeros(denominator.shape)
    for idx, den in enumerate(denominator):
        if den != 0:
            values[idx] = numerator[idx] / den
        elif numerator[idx] == 0:
            values[idx] = both_zero
    return values


def _calc_water_balances(prepared: _PalmerPrepared) -> None:
    """
    Perform water balance calculations

    :param prepared: the prepared Palmer inputs
    """
    ss = AWCTOP
    su = prepared.awc_bot
    for year in range(prepared.n_years):
        for month in range(12):
            p = prepared.precips[year, month]
            pet = prepared.pet[year, month]
            sp = ss + su
            pr = prepared.awc_bot + AWCTOP - sp

            # Get potential loss
            pl = _calc_potential_loss(pet, ss, su, prepared.awc)

            # Calculate recharge, runoff, residual moisture, loss to both
            # surface and under layers, depending on starting moisture
            # content and values of precipitation and evaporation
            et, tl, r, ro, sss, ssu = _calc_recharge(p, pet, ss, su, prepared.awc)

            # update sums
            if prepared.calibration_year_initial_idx <= year <= prepared.calibration_year_final_idx:
                prepared.psum[month] += p
                prepared.spsum[month] += sp
                prepared.petsum[month] += pet
                prepared.plsum[month] += pl
                prepared.prsum[month] += pr
                prepared.rsum[month] += r
                prepared.tlsum[month] += tl
                prepared.etsum[month] += et
                prepared.rosum[month] += ro

            # set data
            prepared.spdat[year, month] = sp
            prepared.pldat[year, month] = pl
            prepared.prdat[year, month] = pr
            prepared.rdat[year, month] = r
            prepared.tldat[year, month] = tl
            prepared.etdat[year, month] = et
            prepared.rodat[year, month] = ro
            prepared.sssdat[year, month] = sss
            prepared.ssudat[year, month] = ssu

            # update soil moisture
            ss = sss
            su = ssu


def _calc_cafec_coefficients(prepared: _PalmerPrepared) -> None:
    """
    Calculate CAFEC Coefficients

    :param prepared: the prepared Palmer inputs
    """
    prepared.alpha = _calc_cafec_ratio(prepared.etsum, prepared.petsum)
    prepared.beta = _calc_cafec_ratio(prepared.rsum, prepared.prsum)
    prepared.gamma = _calc_cafec_ratio(prepared.rosum, prepared.spsum)
    prepared.delta = _calc_cafec_ratio(prepared.tlsum, prepared.plsum, both_zero=0.0)


def _calc_zindex_factors(prepared: _PalmerPrepared) -> None:
    """
    Calculate Z-Index weighting factors (variable AK)

    trat is the 'T' ratio of average moisture demand
    to average moisture supply in month M

    :param prepared: the prepared Palmer inputs
    """
    prepared.trat = (prepared.petsum + prepared.rsum + prepared.rosum) / (prepared.psum + prepared.tlsum)


def _calc_k_prime_and_dbar(prepared: _PalmerPrepared) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate monthly mean absolute departures (dbar) and raw K-prime factors

    :param prepared: the prepared Palmer inputs
    """
    sabsd = np.zeros((12,))
    for year in range(prepared.calibration_year_initial_idx, prepared.calibration_year_final_idx + 1):
        for month in range(12):
            phat = (
                prepared.alpha[month] * prepared.pet[year, month]
                + prepared.beta[month] * prepared.prdat[year, month]
                + prepared.gamma[month] * prepared.spdat[year, month]
                - prepared.delta[month] * prepared.pldat[year, month]
            )
            sabsd[month] += abs(prepared.precips[year, month] - phat)

    dbar = sabsd / prepared.n_calb_years
    return dbar, 1.5 * np.log10((prepared.trat + 2.8) / dbar) + 0.5


def _calc_kfactors(prepared: _PalmerPrepared) -> None:
    """
    Calculate K Factors

    Reread monthly parameters for calculation of the 'K' monthly
    weighting factors used in z-index calculation

    :param prepared: the prepared Palmer inputs
    """
    dbar, akhat = _calc_k_prime_and_dbar(prepared)
    swtd = np.sum(dbar * akhat)
    prepared.ak = 17.67 * akhat / swtd


def _calc_scpdsi_k_factors(prepared: _PalmerPrepared) -> None:
    """Calculate the unnormalized monthly K-prime factors for scPDSI."""
    with np.errstate(divide="ignore", invalid="ignore"):
        _, k_prime = _calc_k_prime_and_dbar(prepared)
    if not np.all(np.isfinite(k_prime)):
        raise ConvergenceError(
            "scPDSI K-prime calibration produced non-finite values",
            algorithm="scPDSI K-prime calibration",
        )
    prepared.ak = k_prime


def _calc_cafec_zindex(prepared: _PalmerPrepared, state: _PalmerRecursion, year: int, month: int) -> float:
    """
    Calculate one month's CAFEC (climatically appropriate for existing
    conditions) precipitation and raw Z-index, writing the Z-index into the
    recursion state.

    The standard PDSI recursion (_calc_zindex) and the scPDSI recursion
    (_calc_scpdsi_raw_zindex) compute these identically; only the recurrences
    downstream of them differ.

    :param prepared: the prepared Palmer inputs
    :param state: the mutable recursion state
    :param year: row index into the monthly arrays
    :param month: month index, 0 = January
    :return the CAFEC precipitation value, returned so tests can pin its exact
            term grouping
    :rtype: float
    """
    cafec: float = (
        prepared.alpha[month] * prepared.pet[year, month]
        + prepared.beta[month] * prepared.prdat[year, month]
        + prepared.gamma[month] * prepared.spdat[year, month]
        - prepared.delta[month] * prepared.pldat[year, month]
    )
    state.z[year, month] = prepared.ak[month] * (prepared.precips[year, month] - cafec)
    return cafec


def _calc_scpdsi_raw_zindex(prepared: _PalmerPrepared, state: _PalmerRecursion) -> None:
    """Calculate raw Z-index values for the entire input record."""
    for year in range(prepared.n_years):
        for month in range(12):
            _calc_cafec_zindex(prepared, state, year, month)


def _calibration_values(prepared: _PalmerPrepared, values: np.ndarray) -> np.ndarray:
    """Return the flattened inclusive calibration-period portion of an array."""
    first = prepared.calibration_year_initial_idx * 12
    final = (prepared.calibration_year_final_idx + 1) * 12
    return np.asarray(values).reshape(-1)[first:final]


def _rescale_scpdsi_zindex(z_values: np.ndarray, dry_percentile: float, wet_percentile: float) -> np.ndarray:
    """Apply one sign-specific, cumulative scPDSI Z-index rescaling pass."""
    if (
        not np.isfinite(dry_percentile)
        or dry_percentile >= 0.0
        or not np.isfinite(wet_percentile)
        or wet_percentile <= 0.0
    ):
        raise ConvergenceError(
            "scPDSI calibration produced invalid dry/wet percentile anchors",
            algorithm="scPDSI percentile calibration",
        )

    dry_ratio = -4.0 / dry_percentile
    wet_ratio = 4.0 / wet_percentile
    if not np.isfinite(dry_ratio) or not np.isfinite(wet_ratio):
        raise ConvergenceError(
            "scPDSI percentile rescaling produced non-finite ratios",
            algorithm="scPDSI percentile calibration",
        )
    return np.where(z_values < 0.0, z_values * dry_ratio, z_values * wet_ratio)


def _case(prob: float, x1: float, x2: float, x3: float) -> float:
    """
    Select the preliminary (or near-real time) PDSI

    Selects the PDSI from the given x values
    defined below and the probability (prob) of ending either a
    drought or wet spell.

    :param prob: the probability of ending either a drought
                 or wet spell
    :param x1: Index for incipient wet spells (always positive)
    :param x2: Index for incipient dry spells (always negative)
    :param x3: severity index for an established wet spell (positive)
               or drought (negative)
    :returns the selected pdsi (either preliminary or final)
    :rtype: float
    """

    # if x3 = 0 the index is near normal and either a dry or wet spell
    # exists. Choose the largest absolute value of x1 or x2
    if x3 == 0:
        if abs(x1) > abs(x2):
            return x1
        return x2

    # A weather spell is established and palm = x3 is final
    if (prob <= 0) or (prob >= 100):
        return x3

    pro = prob / 100
    if x3 <= 0:
        return (1.0 - pro) * x3 + pro * x1

    return (1.0 - pro) * x3 + pro * x2


def _record_index_values(state: _PalmerRecursion, year: int, month: int, value: float) -> None:
    """
    Record one month's PDSI, PHDI, and PMDI

    Used both when no spell is open (k8 == 0), where ``value`` is this
    month's preliminary X value, and when a spell closes and ``_assign``
    flushes the backtracked trail, where ``value`` is the assigned severity.

    :param state: the mutable recursion state
    :param year: row index into the monthly arrays
    :param month: month index, 0 = January
    :param value: the PDSI value to record for this month
    """
    state.pdsi[year, month] = value
    state.phdi[year, month] = state.px3[year, month]
    if state.px3[year, month] == 0:
        state.phdi[year, month] = value
    state.wplm[year, month] = _case(
        state.ppr[year, month],
        state.px1[year, month],
        state.px2[year, month],
        state.px3[year, month],
    )


def _backtrack_assigned_values(state: _PalmerRecursion) -> None:
    """
    Backtrack through the x1/x2 trail arrays

    Stores the assigned x1 (or x2) in sx until it is zero, then switches to
    the other until it is zero, etc.

    :param state: the mutable recursion state
    """
    isave = state.iass
    for i in range(state.k8 - 1, -1, -1):
        if isave == 2:
            if state.sx2[i] == 0:
                isave = 1
                state.sx[i] = state.sx1[i]
            else:
                isave = 2
                state.sx[i] = state.sx2[i]
        else:
            if state.sx1[i] == 0:
                isave = 2
                state.sx[i] = state.sx2[i]
            else:
                isave = 1
                state.sx[i] = state.sx1[i]


def _assign(state: _PalmerRecursion) -> None:
    """
    Assign x values

    :param state: the mutable recursion state
    """
    year = state.year
    month = state.month
    state.sx[state.k8] = state.x[year, month]
    if state.k8 == 0:
        _record_index_values(state, year, month, state.x[year, month])
        return

    # use all x3 values
    if state.iass == 3:
        state.sx[: state.k8] = state.sx3[: state.k8]
    else:
        _backtrack_assigned_values(state)

    # proper assignments to array sx have been made, output the mess
    for idx in range(state.k8 + 1):
        j = int(state.indexj[idx])
        m = int(state.indexm[idx])
        _record_index_values(state, j, m, state.sx[idx])
    state.k8 = 0
    # k8max is deliberately not reset here: it is the high-water mark
    # _finish_up reads once the whole recursion ends, not per-spell state.


def _statement_220(state: _PalmerRecursion) -> None:
    """
    Save this month's calculated variables (v,pro,x1,x2,x3) for
    use with next month's data

    Translated from statement 220 in NCEI's pdi.f

    :param state: the mutable recursion state
    """
    year = state.year
    month = state.month
    state.v = state.pv
    state.pro = state.ppr[year, month]
    state.x1 = state.px1[year, month]
    state.x2 = state.px2[year, month]
    state.x3 = state.px3[year, month]


def _statement_210(prepared: _PalmerPrepared, state: _PalmerRecursion) -> None:
    """
    prob(end) returns to 0. A possible abatement has fizzled out,
    so we accept all stored values of x3

    Translated from statement 210 in NCEI's pdi.f

    :param prepared: the prepared Palmer inputs
    :param state: the mutable recursion state
    """
    year = state.year
    month = state.month
    state.pv = 0.0
    state.px1[year, month] = 0.0
    state.px2[year, month] = 0.0
    state.ppr[year, month] = 0.0
    m, b = _select_duration_factors(prepared, state)
    state.px3[year, month] = DurationFactors.weighting_fraction(m, b) * state.x3 + state.z[year, month] / (m + b)
    state.x[year, month] = state.px3[year, month]

    if state.k8 == 0:
        _record_index_values(state, year, month, state.x[year, month])
    else:
        state.iass = 3
        _assign(state)

    _statement_220(state)


def _statement_200(prepared: _PalmerPrepared, state: _PalmerRecursion) -> None:
    """
    Continue x1 and x2 calculations
    if either indicates the start of a new wet or drought,
    and if the last wet or drought has ended, use x1 or x2
    as the new x3

    Translated from statement 200 in NCEI's pdi.f

    :param prepared: the prepared Palmer inputs
    :param state: the mutable recursion state
    """
    year = state.year
    month = state.month
    wetm, wetb = prepared.wetm, prepared.wetb
    state.px1[year, month] = max(
        0, DurationFactors.weighting_fraction(wetm, wetb) * state.x1 + state.z[year, month] / (wetm + wetb)
    )

    # if no existing wet spell or drought
    # x1 becomes the new x3
    if (state.px1[year, month] >= 1) and (state.px3[year, month] == 0):
        state.x[year, month] = state.px1[year, month]
        state.px3[year, month] = state.px1[year, month]
        state.px1[year, month] = 0
        state.iass = 1
        _assign(state)
        _statement_220(state)
        return

    drym, dryb = prepared.drym, prepared.dryb
    state.px2[year, month] = min(
        0.0, DurationFactors.weighting_fraction(drym, dryb) * state.x2 + state.z[year, month] / (drym + dryb)
    )

    # if no existing wet spell or drought x2 becomes the new x3
    if (state.px2[year, month] <= -1) and (state.px3[year, month] == 0):
        state.x[year, month] = state.px2[year, month]
        state.px3[year, month] = state.px2[year, month]
        state.px2[year, month] = 0.0
        state.iass = 2
        _assign(state)
        _statement_220(state)
        return

    # No established drought (wet spell), but x3 = 0
    # so either (nonzero) x1 or x2 must be used as x3
    if state.px3[year, month] == 0:
        if state.px1[year, month] == 0:
            state.x[year, month] = state.px2[year, month]
            state.iass = 2
            _assign(state)
            _statement_220(state)
            return

        if state.px2[year, month] == 0:
            state.x[year, month] = state.px1[year, month]
            state.iass = 1
            _assign(state)
            _statement_220(state)
            return

    # at this point there is no determed value to assign to x,
    # all the values of x1, x2, and x3 are saved. Ata a later
    # time x3 will reach a value where it is the value of x (pdsi).
    # At that time, the assign method backtracs through choosing
    # the appropriate x1 or x2 to be that month's x.
    if state.k8 >= state.sx.shape[0] + 1:
        vals = [0] * (state.k8 - state.sx.shape[0] + 2)
        state.sx = np.append(state.sx, vals)
        state.sx1 = np.append(state.sx1, vals)
        state.sx2 = np.append(state.sx2, vals)
        state.sx3 = np.append(state.sx3, vals)
        state.indexj = np.append(state.indexj, vals)
        state.indexm = np.append(state.indexm, vals)

    state.sx1[state.k8] = state.px1[year, month]
    state.sx2[state.k8] = state.px2[year, month]
    state.sx3[state.k8] = state.px3[year, month]
    state.x[year, month] = state.px3[year, month]
    state.k8 += 1
    state.k8max = state.k8

    _statement_220(state)


def _statement_190(prepared: _PalmerPrepared, state: _PalmerRecursion) -> None:
    """
    drought or wet continues, calculate prob(end) (variable ze)

    Translated from statement 190 in NCEI's pdi.f

    :param prepared: the prepared Palmer inputs
    :param state: the mutable recursion state
    """
    year = state.year
    month = state.month
    if state.pro == 100:
        q = state.ze
    else:
        q = state.ze + state.v

    state.ppr[year, month] = (state.pv / q) * 100

    if state.ppr[year, month] >= 100:
        state.ppr[year, month] = 100
        state.px3[year, month] = 0
    else:
        m, b = _select_duration_factors(prepared, state)
        state.px3[year, month] = DurationFactors.weighting_fraction(m, b) * state.x3 + state.z[year, month] / (m + b)

    _statement_200(prepared, state)


def _statement_180(prepared: _PalmerPrepared, state: _PalmerRecursion) -> None:
    """
    drought abatement is possible

    Translated from statement 180 in NCEI's pdi.f

    :param prepared: the prepared Palmer inputs
    :param state: the mutable recursion state
    """
    year = state.year
    month = state.month
    state.uw = state.z[year, month] + 0.15
    state.pv = state.uw + max(state.v, 0.0)

    # During a drought, PV <= 0 implies prob(end) has returned to 0
    if state.pv <= 0:
        _statement_210(prepared, state)
        return

    m, b = prepared.drym, prepared.dryb
    state.ze = -b * state.x3 - 0.5 * (m + b)
    _statement_190(prepared, state)


def _statement_170(prepared: _PalmerPrepared, state: _PalmerRecursion) -> None:
    """
    Wet spell abatement is possible

    Translated from statement 170 in NCEI's pdi.f

    :param prepared: the prepared Palmer inputs
    :param state: the mutable recursion state
    """
    year = state.year
    month = state.month
    state.ud = state.z[year, month] - 0.15
    state.pv = state.ud + min(state.v, 0.0)

    # During a wet spell, PV >= 0 implies prob(end) has returned to 0
    if state.pv >= 0:
        _statement_210(prepared, state)
        return

    m, b = prepared.wetm, prepared.wetb
    state.ze = -b * state.x3 + 0.5 * (m + b)
    _statement_190(prepared, state)


def _step_established_spell(prepared: _PalmerPrepared, state: _PalmerRecursion, year: int, month: int) -> bool:
    """
    Handle a month where no abatement is underway (pro is 0 or 100)

    :param prepared: the prepared Palmer inputs
    :param state: the mutable recursion state
    :param year: row index into the monthly arrays
    :param month: month index, 0 = January
    :returns: True if this month was dispatched to a statement here, False
              to have the caller fall through to its own default
    """
    # End of drought or wet
    if -0.5 <= state.x3 <= 0.5:
        state.pv = 0.0
        state.ppr[year, month] = 0.0
        state.px3[year, month] = 0.0
        # check for new wet or drought start
        _statement_200(prepared, state)
        return True
    # We are in a wet spell
    elif state.x3 > 0.5:
        # The wet spell intensifies
        if state.z[year, month] >= 0.15:
            _statement_210(prepared, state)
        # The wet spell starts to abate (and may end)
        else:
            _statement_170(prepared, state)
        return True
    # We are in a drought
    elif state.x3 < -0.5:
        # The drought intensifies
        if state.z[year, month] <= -0.15:
            _statement_210(prepared, state)
        # The drought starts to abate (and may end)
        else:
            _statement_180(prepared, state)
        return True
    return False


def _advance_month(prepared: _PalmerPrepared, state: _PalmerRecursion, year: int, month: int) -> None:
    """
    Advance the Z-index recursion by one month

    Rereads monthly parameters for calculation of the 'K' monthly weighting
    factors used in z-index calculation, then dispatches to the
    established-spell logic (no abatement underway) or the
    abatement-in-progress logic.

    :param prepared: the prepared Palmer inputs
    :param state: the mutable recursion state
    :param year: row index into the monthly arrays
    :param month: month index, 0 = January
    """
    state.year = year
    state.month = month
    k8 = int(state.k8)
    state.indexj[k8] = year
    state.indexm[k8] = month
    state.ze = 0.0
    state.ud = 0.0
    state.uw = 0.0
    _calc_cafec_zindex(prepared, state, year, month)

    # No abatement underway, wet or drought will end if -.5 <= X3 <= .5
    if (state.pro == 100) or (state.pro == 0):
        if _step_established_spell(prepared, state, year, month):
            return
    # Abatement is underway
    else:
        # We are in a wet spell
        if state.x3 > 0:
            _statement_170(prepared, state)
            return
        # We are in a drought
        elif state.x3 <= 0:
            _statement_180(prepared, state)
            return

    _statement_170(prepared, state)


def _calc_zindex(prepared: _PalmerPrepared, state: _PalmerRecursion) -> None:
    """
    Calculate Z Index

    :param prepared: the prepared Palmer inputs
    :param state: the mutable recursion state
    """
    for year in range(prepared.n_years):
        for month in range(12):
            _advance_month(prepared, state, year, month)


def _finish_up(state: _PalmerRecursion) -> None:
    """
    Wet spell abatement is possible

    :param state: the mutable recursion state
    """
    for k8 in range(state.k8max):
        i = int(state.indexj[k8])
        j = int(state.indexm[k8])
        i_end = state.pdsi.shape[0] - 1
        state.pdsi[i, j] = state.x[i, j]
        state.phdi[i, j] = state.px3[i, j]

        if state.px3[i, j] == 0:
            state.phdi[i, j] = state.x[i, j]

        state.wplm[i, j] = _case(
            state.ppr[i_end, 11],
            state.px1[i_end, 11],
            state.px2[i_end, 11],
            state.px3[i_end, 11],
        )


def _validate_fitting_params(prepared: _PalmerPrepared, fitting_params: dict[str, Any] | None) -> None:
    """
    Validate the fitting parameters

    :param prepared: the prepared Palmer inputs
    :param fitting_params: dictionary of the fitted parameters
    """
    if fitting_params is None:
        prepared.calibrate = True
        return

    # each coefficient must be a numeric one-dimensional vector with exactly one
    # value per month; anything else (missing, non-numeric, or two-dimensional)
    # leaves the calibration flag set so the coefficients are fitted from data
    names = ("alpha", "beta", "gamma", "delta")
    coefficients: list[np.ndarray] = []
    for name in names:
        try:
            values = np.asarray(fitting_params.get(name), dtype=float)
        except (TypeError, ValueError):
            break
        if values.shape != (12,):
            break
        coefficients.append(values)

    prepared.calibrate = len(coefficients) != len(names)
    if not prepared.calibrate:
        prepared.alpha = coefficients[0]
        prepared.beta = coefficients[1]
        prepared.gamma = coefficients[2]
        prepared.delta = coefficients[3]


def _validate_calibration_period(
    data_start_year: int,
    n_years: int,
    calibration_year_initial: int,
    calibration_year_final: int,
) -> None:
    """Ensure the inclusive calibration period is represented by the input record."""
    data_final_year = data_start_year + n_years - 1
    if (
        calibration_year_initial > calibration_year_final
        or calibration_year_initial < data_start_year
        or calibration_year_final > data_final_year
    ):
        raise ValueError(
            "calibration period must be an inclusive interval within the input data years "
            f"[{data_start_year}, {data_final_year}]"
        )


def _initialize_prepared(
    precips: np.ndarray,
    pet: np.ndarray,
    awc: float,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict[str, Any] | None = None,
) -> _PalmerPrepared:
    """
    Initialize the prepared inputs

    :param precips: time series of monthly precipitation values, in inches
    :param pet: time series of monthly PET values, in inches
    :param awc: available water capacity (soil constant), in inches
    :param data_start_year: initial year of the input precipitation and PET datasets,
                            both of which are assumed to start in January of this year
    :param calibration_year_initial: initial year of the calibration period
    :param calibration_year_final: final year of the calibration period
    :param fitting_params: dictionary of the fitted parameters
    :return the initialized prepared inputs
    :rtype: _PalmerPrepared
    """
    # reshape precipitation values to (years, 12)
    precips = utils.reshape_to_2d(precips, 12)
    pet = utils.reshape_to_2d(pet, 12)
    n_years = int(precips.shape[0])
    _validate_calibration_period(
        data_start_year,
        n_years,
        calibration_year_initial,
        calibration_year_final,
    )

    # duration factors default to Palmer's fixed national values and are read by
    # the standard PDSI recursion through _select_duration_factors. scPDSI does not
    # override these fields: it passes its per-location fitted factors straight to
    # _palmer_wells.calculate. ``calibrate`` is settled by _validate_fitting_params,
    # and the CAFEC coefficients, moisture-demand ratio, and Z-index factors are
    # filled by the stage that owns them before anything reads them.
    duration_factors = DurationFactors.from_defaults()
    prepared = _PalmerPrepared(
        precips=precips,
        pet=pet,
        awc=awc,
        awc_bot=_get_awc_bot(awc),
        n_years=n_years,
        n_calb_years=calibration_year_final - calibration_year_initial + 1,
        calibration_year_initial_idx=calibration_year_initial - data_start_year,
        calibration_year_final_idx=calibration_year_final - data_start_year,
        calibrate=True,
        spdat=np.full((n_years, 12), np.nan),
        pldat=np.full((n_years, 12), np.nan),
        prdat=np.full((n_years, 12), np.nan),
        rdat=np.full((n_years, 12), np.nan),
        tldat=np.full((n_years, 12), np.nan),
        etdat=np.full((n_years, 12), np.nan),
        rodat=np.full((n_years, 12), np.nan),
        sssdat=np.full((n_years, 12), np.nan),
        ssudat=np.full((n_years, 12), np.nan),
        psum=np.zeros((12,)),
        spsum=np.zeros((12,)),
        petsum=np.zeros((12,)),
        plsum=np.zeros((12,)),
        prsum=np.zeros((12,)),
        rsum=np.zeros((12,)),
        tlsum=np.zeros((12,)),
        etsum=np.zeros((12,)),
        rosum=np.zeros((12,)),
        alpha=np.full((12,), np.nan),
        beta=np.full((12,), np.nan),
        gamma=np.full((12,), np.nan),
        delta=np.full((12,), np.nan),
        trat=np.full((12,), np.nan),
        ak=np.full((12,), np.nan),
        wetm=duration_factors.wetm,
        wetb=duration_factors.wetb,
        drym=duration_factors.drym,
        dryb=duration_factors.dryb,
    )

    _validate_fitting_params(prepared, fitting_params)

    return prepared


def _initialize_recursion(prepared: _PalmerPrepared) -> _PalmerRecursion:
    """
    Construct the zeroed recursion state for one calculation over the prepared record.

    :param prepared: the prepared Palmer inputs
    :return the initialized recursion state
    :rtype: _PalmerRecursion
    """
    n_years = prepared.n_years
    return _PalmerRecursion(
        indexj=np.full((K8_SIZE,), np.nan),
        indexm=np.full((K8_SIZE,), np.nan),
        sx=np.zeros((K8_SIZE,)),
        sx1=np.zeros((K8_SIZE,)),
        sx2=np.zeros((K8_SIZE,)),
        sx3=np.zeros((K8_SIZE,)),
        ppr=np.zeros((n_years, 12)),
        px1=np.zeros((n_years, 12)),
        px2=np.zeros((n_years, 12)),
        px3=np.zeros((n_years, 12)),
        x=np.zeros((n_years, 12)),
        z=np.full((n_years, 12), np.nan),
        pdsi=np.full((n_years, 12), np.nan),
        phdi=np.full((n_years, 12), np.nan),
        wplm=np.full((n_years, 12), np.nan),
    )


def _bind_palmer_log(
    index_type: str,
    precips: np.ndarray,
    awc: float,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
) -> BoundLogger:
    """Bind the structured context shared by Palmer calculations."""
    return _logger.bind(
        index_type=index_type,
        awc=awc,
        data_start_year=data_start_year,
        calibration_year_initial=calibration_year_initial,
        calibration_year_final=calibration_year_final,
        input_shape=precips.shape,
        input_elements=precips.size,
    )


def _prepare_palmer_data(
    precips: np.ndarray,
    pet: np.ndarray,
    awc: float,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict[str, Any] | None,
    log: BoundLogger,
) -> tuple[_PalmerPrepared, int]:
    """Validate inputs and run the water-balance/CAFEC stages shared by Palmer indices."""
    if np.any(precips < 0.0):
        log.warning("negative_values_clipped", field="precips")
        precips = np.clip(precips, a_min=0.0, a_max=None)

    original_length = precips.size
    prepared = _initialize_prepared(
        precips=precips,
        pet=pet,
        awc=awc,
        data_start_year=data_start_year,
        calibration_year_initial=calibration_year_initial,
        calibration_year_final=calibration_year_final,
        fitting_params=fitting_params,
    )
    _calc_water_balances(prepared)
    if prepared.calibrate:
        _calc_cafec_coefficients(prepared)
    _calc_zindex_factors(prepared)
    return prepared, original_length


def _calculate_pdsi_prepared(prepared: _PalmerPrepared, original_length: int) -> _PalmerResult:
    """Complete standard PDSI after the shared Palmer preparation stages."""
    _calc_kfactors(prepared)
    state = _initialize_recursion(prepared)
    _calc_zindex(prepared, state)
    _finish_up(state)

    pdsi_result = state.pdsi.flatten()[0:original_length]
    phdi = state.phdi.flatten()[0:original_length]
    wplm = state.wplm.flatten()[0:original_length]
    z = state.z.flatten()[0:original_length]
    params = {
        "alpha": prepared.alpha,
        "beta": prepared.beta,
        "gamma": prepared.gamma,
        "delta": prepared.delta,
    }
    return _PalmerResult(pdsi_result, phdi, wplm, z, params)


def _calculate_scpdsi_prepared(prepared: _PalmerPrepared, original_length: int) -> _PalmerResult:
    """Complete self-calibrating PDSI after shared Palmer preparation."""
    _calc_scpdsi_k_factors(prepared)
    state = _initialize_recursion(prepared)
    _calc_scpdsi_raw_zindex(prepared, state)

    z_values = state.z.reshape(-1)
    calibration_z = _calibration_values(prepared, z_values)
    wetm, wetb = self_calibration.duration_factors(calibration_z, self_calibration.WET_SIGN)
    drym, dryb = self_calibration.duration_factors(calibration_z, self_calibration.DRY_SIGN)

    recursion = _palmer_wells.calculate(
        z_values,
        wetm=wetm,
        wetb=wetb,
        drym=drym,
        dryb=dryb,
    )
    for _ in range(3):
        calibration_pdsi = _calibration_values(prepared, recursion.pdsi)
        dry_percentile = self_calibration.nan_safe_percentile(calibration_pdsi, 0.02)
        wet_percentile = self_calibration.nan_safe_percentile(calibration_pdsi, 0.98)
        z_values = _rescale_scpdsi_zindex(z_values, dry_percentile, wet_percentile)
        recursion = _palmer_wells.calculate(
            z_values,
            wetm=wetm,
            wetb=wetb,
            drym=drym,
            dryb=dryb,
        )

    params: dict[str, Any] = {
        "alpha": prepared.alpha,
        "beta": prepared.beta,
        "gamma": prepared.gamma,
        "delta": prepared.delta,
    }
    params.update(wetm=wetm, wetb=wetb, drym=drym, dryb=dryb)
    return _PalmerResult(
        recursion.pdsi[:original_length],
        recursion.phdi[:original_length],
        recursion.pmdi[:original_length],
        z_values[:original_length],
        params,
    )


def _palmer_calculation(
    index_type: str,
    calculate_prepared: Callable[[_PalmerPrepared, int], _PalmerResult],
    precips: np.ndarray,
    pet: np.ndarray,
    awc: float,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict[str, Any] | None,
) -> _PalmerResult:
    """Run validation, shared setup, logging, and one Palmer calculation."""
    log = _bind_palmer_log(
        index_type,
        precips,
        awc,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()

    try:
        if precips.size != pet.size:
            message = "Incompatible precipitation and PET arrays"
            log.error("validation_failed", reason=message)
            raise ValueError(message)

        if np.any(np.isinf(precips)) or np.any(np.isinf(pet)):
            message = "precipitation and PET arrays cannot contain infinite values"
            log.error("validation_failed", reason=message)
            raise ValueError(message)

        all_missing = (isinstance(precips, np.ma.MaskedArray) and precips.mask.all()) or np.all(np.isnan(precips))
        if all_missing:
            _validate_calibration_period(
                data_start_year,
                int(utils.reshape_to_2d(precips, 12).shape[0]),
                calibration_year_initial,
                calibration_year_final,
            )
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info(
                "calculation_completed",
                duration_ms=round(duration_ms, 2),
                result="all_missing",
            )
            return _PalmerResult(precips, precips, precips, precips, None)

        prepared, original_length = _prepare_palmer_data(
            precips,
            pet,
            awc,
            data_start_year,
            calibration_year_initial,
            calibration_year_final,
            fitting_params,
            log,
        )
        result = calculate_prepared(prepared, original_length)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "calculation_completed",
            duration_ms=round(duration_ms, 2),
            output_elements=result.pdsi.size,
        )
        return result
    except Exception:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.error("calculation_failed", duration_ms=round(duration_ms, 2))
        raise


def pdsi(
    precips: np.ndarray,
    pet: np.ndarray,
    awc: float,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    """
    Compute the Palmer Drought Severity Index (PDSI),
    Palmer Hydrological Drought Index (PHDI),
    Palmer Modified Drought Index (PMDI), and
    Palmer Z-Index.

    Args:
        precips: Time series of monthly precipitation values, in inches.
        pet: Time series of monthly PET values, in inches.
        awc: Available water capacity (soil constant), in inches.
        data_start_year: Initial year of the input precipitation and PET
            datasets, both of which are assumed to start in January of this
            year.
        calibration_year_initial: Initial year of the calibration period.
        calibration_year_final: Final year of the calibration period.
        fitting_params: Dictionary of the fitted parameters.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
            A five-item tuple containing NumPy arrays of PDSI, PHDI, PMDI, and
            Z-Index values, respectively, and a dictionary containing the
            fitted ``alpha``, ``beta``, ``gamma``, and ``delta`` parameters.
            For all-missing input, the parameter dictionary is ``None``.
    """

    # _palmer_calculation emits calculation_started, calculation_completed,
    # and calculation_failed lifecycle events for this public entry point.
    return _palmer_calculation(
        "pdsi",
        _calculate_pdsi_prepared,
        precips,
        pet,
        awc,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        fitting_params,
    )


def scpdsi(
    precips: np.ndarray,
    pet: np.ndarray,
    awc: float,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    """Compute self-calibrating Palmer drought indices.

    The water balance and CAFEC coefficients are shared with :func:`pdsi`.
    Duration factors, K-prime factors, and percentile scaling are calibrated
    from the requested calibration period, while the resulting factors are
    applied to the full monthly record.

    Args:
        precips: Monthly precipitation values in inches.
        pet: Monthly potential evapotranspiration values in inches.
        awc: Available water capacity in inches.
        data_start_year: First calendar year represented by the inputs, which
            are assumed to begin in January.
        calibration_year_initial: First year of the inclusive calibration
            period.
        calibration_year_final: Final year of the inclusive calibration period.
        fitting_params: Optional CAFEC coefficients to reuse. Valid ``alpha``,
            ``beta``, ``gamma``, and ``delta`` arrays follow :func:`pdsi`'s
            behavior; duration factors are always recalibrated.

    Returns:
        A tuple containing scPDSI, scPHDI, scPMDI, the cumulatively calibrated
        Z-index, and fitted parameters. The parameter dictionary contains
        ``alpha``, ``beta``, ``gamma``, ``delta``, ``wetm``, ``wetb``,
        ``drym``, and ``dryb``. All-missing input returns four same-length
        missing arrays and ``None``.

    Raises:
        ValueError: If precipitation and PET have different lengths.
        InsufficientDataError: If the calibration period cannot supply a
            complete duration-factor fitting window.
        ConvergenceError: If a numerical calibration stage produces unusable
            factors, percentile anchors, or recurrence denominators. The
            duration-factor fit must yield contracting recurrence coefficients;
            a short or climatologically skewed calibration period can pull a
            fitted slope non-positive and trigger this (see
            :func:`climate_indices.self_calibration.duration_factors`).
    """
    return _palmer_calculation(
        "scpdsi",
        _calculate_scpdsi_prepared,
        precips,
        pet,
        awc,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        fitting_params,
    )


# TODO(v2.5.0): implement palmer_xarray() wrapper using Pattern C
# (stack/unpack workaround for xarray Issue #1815, see architecture.md)
