"""Compute palmer drought indices"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from structlog.stdlib import BoundLogger

from climate_indices import _palmer_wells, compute, self_calibration, utils
from climate_indices._palmer_duration import DurationFactors
from climate_indices.exceptions import ConvergenceError
from climate_indices.logging_config import get_logger

_logger = get_logger(__name__)

# declare the function names that should be included in the public API for this module
__all__ = ["pdsi", "scpdsi"]

AWCTOP = 1.0


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
    awc: float | np.ndarray
    awc_bot: float | np.ndarray
    n_years: int
    n_calb_years: int
    calibration_year_initial_idx: int
    calibration_year_final_idx: int
    calibrate: bool

    # the trailing cell axis: n_cells == 1 and cell_shape == () for a single
    # location (1-D or (years, 12) input); n_cells == prod(cell_shape) for a
    # spatial_time_major block. precips/pet/awc and every array below always
    # carry this axis, so the recursion has one code path for both.
    n_cells: int
    cell_shape: tuple[int, ...]

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
    month-carry fields a statement assigns before reading default to zero, so
    the struct exists ahead of the recursion that fills them.

    Every field below carries the same trailing ``(n_cells,)`` (or
    ``(n_months, n_cells)`` / ``(n_years, 12, n_cells)``) cell axis as
    :class:`_PalmerPrepared`, including the "scalar" month-carry state
    (``v``, ``pro``, ``x1``, ``x2``, ``x3``, ``iass``, ``k8``, ...): each is a
    length-``n_cells`` array so a single location (``n_cells == 1``) and a
    spatial block share one code path. ``year``/``month`` stay plain ints --
    every cell in a block shares the same calendar step, so they are the
    Python loop indices ``_calc_zindex`` advances, not per-cell state.
    """

    # recursion state: the K8 window (indexed by month-of-record, not the
    # fixed K8_SIZE historical bound -- see _initialize_recursion), per-month
    # candidates, and the current severity
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
    k8: np.ndarray
    k8max: np.ndarray
    year: int
    month: int

    # month-carry state a statement assigns before reading it; zero until then
    iass: np.ndarray
    v: np.ndarray
    pro: np.ndarray
    x1: np.ndarray
    x2: np.ndarray
    x3: np.ndarray
    ze: np.ndarray
    ud: np.ndarray
    uw: np.ndarray
    pv: np.ndarray


def _select_duration_factors(prepared: _PalmerPrepared, state: _PalmerRecursion) -> tuple[np.ndarray, np.ndarray]:
    """
    Select the wet or dry duration factors based on the sign of the
    currently-established spell's severity (X3), per cell.

    X3 equal to zero means that no wet or dry spell is established. It is
    assigned the wet factors to preserve the recursion's historical
    non-negative tie-break. With Palmer's identical wet and dry defaults the
    choice is unobservable; a distinct duration-factor override makes it
    observable, and the tie-break is kept deliberately.

    :param prepared: the prepared Palmer inputs
    :param state: the mutable recursion state
    :return a tuple of (m, b) arrays - the duration-factor slope and intercept, per cell
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    m = np.where(state.x3 >= 0, prepared.wetm, prepared.drym)
    b = np.where(state.x3 >= 0, prepared.wetb, prepared.dryb)
    return m, b


def _py_max(a: float | np.ndarray, b: float | np.ndarray) -> np.ndarray:
    """``max(a, b)`` matching Python's builtin comparison order, not ``np.maximum``.

    Python's ``max(a, b)`` returns ``b`` only if ``b > a``, so a NaN in ``b`` never
    wins (its comparison is always False) while a NaN in ``a`` always loses unless
    ``b`` also fails to compare greater -- an asymmetry ``np.maximum`` does not
    have (it propagates NaN from either operand). The recursion below calls Python's
    builtin at several call sites with data-dependent NaN possible in either
    position, so replicating this exact rule is required for bit-for-bit
    equivalence with the per-location path.
    """
    return np.where(np.asarray(b) > a, b, a)


def _py_min(a: float | np.ndarray, b: float | np.ndarray) -> np.ndarray:
    """``min(a, b)`` matching Python's builtin comparison order; see :func:`_py_max`."""
    return np.where(np.asarray(b) < a, b, a)


def _get_awc_bot(awc: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate available water capcity in bottom layer

    :param awc: available water capacity (total), in inches
    :return available water capacity (under layer), in inches
    :rtype: float | np.ndarray
    """
    return _py_max(awc - AWCTOP, 0.0)


def _calc_potential_loss(
    pet: float | np.ndarray,
    ss: float | np.ndarray,
    su: float | np.ndarray,
    awc: float | np.ndarray,
) -> float | np.ndarray:
    """
    Calculate potential loss

    :param pet: potential evapotranspiration
    :param ss: surface layer water content, in inches
    :param su: under layer water content, in inches
    :param awc: available water capacity (total), in inches
    :return potential loss
    :rtype: float | np.ndarray
    """
    awc_bot = _get_awc_bot(awc)
    candidate = _py_min(ss + su, ((pet - ss) * su) / (awc_bot + AWCTOP) + ss)
    return np.where(ss >= pet, pet, candidate)


def _calc_recharge(
    p: float | np.ndarray,
    pet: float | np.ndarray,
    ss: float | np.ndarray,
    su: float | np.ndarray,
    awc: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate recharge, runoff, residual moisture, loss
    to both surface and under layers

    Depends on the starting moisture content and values of
    precipitation and evaporation.

    Every branch below is elementwise on ``p``/``pet``/``ss``/``su`` -- both
    the "precipitation exceeds PET" and "PET exceeds precipitation" arms, and
    their nested sub-branches, are computed for every cell and combined with
    ``np.where`` rather than taken by a Python ``if``, so the same code path
    handles a single location (0-d/scalar arrays) and a block of cells.

    :param p: preciptiation, in inches
    :param pet: potential evapotranspiration
    :param ss: surface layer water content, in inches
    :param su: under layer water content, in inches
    :param awc: available water capacity (total), in inches
    :return a tuple of arrays
        - et: evapotranspiration
        - tl: total loss
        - r: recharge
        - ro: runoff
        - sss: surface layer water content, in inches
        - ssu: under layer water content, in inches
    """
    awc_bot = _get_awc_bot(awc)
    p = np.asarray(p, dtype=float)
    pet = np.asarray(pet, dtype=float)
    ss = np.asarray(ss, dtype=float)
    su = np.asarray(su, dtype=float)

    # --- precipitation exceeds potential evaporation ---
    excess = p - pet
    rs = AWCTOP - ss
    both_layers_take_it_all = (excess - rs) < (awc_bot - su)
    ru_both = excess - rs
    ru_runoff = awc_bot - su
    ru = np.where(both_layers_take_it_all, ru_both, ru_runoff)
    ro_recharge_case = np.where(both_layers_take_it_all, 0.0, excess - rs - ru)

    under_recharged = excess > (AWCTOP - ss)
    r_recharge_case = np.where(under_recharged, rs + ru, excess)
    sss_recharge_case = np.where(under_recharged, AWCTOP, ss + excess)
    ssu_recharge_case = np.where(under_recharged, su + ru, su)
    ro_recharge_case = np.where(under_recharged, ro_recharge_case, 0.0)
    et_recharge_case = pet
    tl_recharge_case = np.zeros_like(p)

    # --- evaporation exceeds precipitation ---
    deficit = pet - p
    sl_both = ss
    ul_both = _py_min(su, (deficit - sl_both) * su / awc)
    surface_only = ss >= deficit
    sl = np.where(surface_only, deficit, sl_both)
    ul = np.where(surface_only, 0.0, ul_both)
    sss_evap_case = np.where(surface_only, ss - sl, 0.0)
    ssu_evap_case = np.where(surface_only, su, su - ul)
    tl_evap_case = sl + ul
    et_evap_case = p + sl + ul
    r_evap_case = np.zeros_like(p)
    ro_evap_case = np.zeros_like(p)

    precip_exceeds = p >= pet
    et = np.where(precip_exceeds, et_recharge_case, et_evap_case)
    tl = np.where(precip_exceeds, tl_recharge_case, tl_evap_case)
    r = np.where(precip_exceeds, r_recharge_case, r_evap_case)
    ro = np.where(precip_exceeds, ro_recharge_case, ro_evap_case)
    sss = np.where(precip_exceeds, sss_recharge_case, sss_evap_case)
    ssu = np.where(precip_exceeds, ssu_recharge_case, ssu_evap_case)

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
    # Exact zero is the reference's "this month contributed nothing to the
    # calibration sums" sentinel, not a float-equality accident: a tolerance
    # would replace the reference's substitution -- both_zero when both sums
    # vanish (1.0 for alpha/beta/gamma, 0.0 for delta), 0.0 when only the
    # denominator does -- with arithmetic on near-zero denominators.
    den_nonzero = denominator != 0  # NOSONAR
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = numerator / denominator
    return np.where(den_nonzero, ratio, np.where(numerator == 0, both_zero, 0.0))  # NOSONAR


def _calc_water_balances(prepared: _PalmerPrepared) -> None:
    """
    Perform water balance calculations

    :param prepared: the prepared Palmer inputs
    """
    ss: float | np.ndarray = AWCTOP
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
    sabsd = np.zeros((12, prepared.n_cells))
    for year in range(prepared.calibration_year_initial_idx, prepared.calibration_year_final_idx + 1):
        for month in range(12):
            phat = (
                prepared.alpha[month] * prepared.pet[year, month]
                + prepared.beta[month] * prepared.prdat[year, month]
                + prepared.gamma[month] * prepared.spdat[year, month]
                - prepared.delta[month] * prepared.pldat[year, month]
            )
            sabsd[month] += np.abs(prepared.precips[year, month] - phat)

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
    # Reduce over the calendar-month axis only: each cell is normalized
    # independently, never pooled with its neighbours. Summing axis 0 of the
    # (12, n_cells) array directly would use numpy's strided-reduction path,
    # which associates differently -- by up to 1 ULP -- from the pairwise
    # summation numpy uses for a standalone contiguous (12,) array (the
    # legacy per-location shape); the recursion branches on exact float
    # comparisons, so that ULP can cascade into a different result. Reducing
    # a C-contiguous (n_cells, 12) transpose puts the sum on the fast,
    # contiguous axis instead, reproducing the per-location value exactly.
    swtd = np.sum(np.ascontiguousarray((dbar * akhat).T), axis=-1)
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


def _case(prob: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    """
    Select the preliminary (or near-real time) PDSI, for every cell.

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
    :rtype: np.ndarray
    """
    # if x3 = 0 the index is near normal and either a dry or wet spell
    # exists. Choose the largest absolute value of x1 or x2
    near_normal = np.where(np.abs(x1) > np.abs(x2), x1, x2)

    # A weather spell is established and palm = x3 is final
    pro = prob / 100.0
    interpolated = np.where(x3 <= 0, (1.0 - pro) * x3 + pro * x1, (1.0 - pro) * x3 + pro * x2)
    established = np.where((prob <= 0) | (prob >= 100), x3, interpolated)

    # x3 is assigned 0.0 exactly when no spell is established, so its exact
    # zero -- not a tolerance -- is what selects the near-normal value.
    return np.where(x3 == 0, near_normal, established)  # NOSONAR


def _record_index_values(
    state: _PalmerRecursion,
    years: np.ndarray,
    months: np.ndarray,
    values: np.ndarray,
    cell_ids: np.ndarray,
) -> None:
    """
    Record PDSI, PHDI, and PMDI for a set of (cell, month) entries.

    Used both when no spell is open (k8 == 0), where ``values`` is this
    month's preliminary X value for each entry's cell, and when a spell
    closes and ``_assign`` flushes the backtracked trail, where ``values``
    is the assigned severity. Vectorized over an explicit list of
    (year, month, cell) triples rather than a single (year, month) pair, so
    the same function serves an immediate single-month record (years/months
    constant, one entry per resolving cell) and a multi-month backtracked
    flush (years/months vary per cell, one entry per (cell,
    historical-month-in-its-spell) pair).

    :param state: the mutable recursion state
    :param years: row index into the monthly arrays, one per entry
    :param months: month index (0 = January), one per entry
    :param values: the PDSI value to record, one per entry
    :param cell_ids: which cell each entry belongs to
    """
    if cell_ids.size == 0:
        return
    px3_here = state.px3[years, months, cell_ids]
    state.pdsi[years, months, cell_ids] = values
    # No established spell (px3 exactly 0.0) means PHDI has no severity of its
    # own and falls back to the PDSI value recorded for this period.
    state.phdi[years, months, cell_ids] = np.where(px3_here == 0, values, px3_here)  # NOSONAR
    state.wplm[years, months, cell_ids] = _case(
        state.ppr[years, months, cell_ids],
        state.px1[years, months, cell_ids],
        state.px2[years, months, cell_ids],
        px3_here,
    )


def _backtrack_assigned_values(state: _PalmerRecursion, active: np.ndarray) -> None:
    """
    Backtrack through the x1/x2 trail arrays, for every active cell.

    Stores the assigned x1 (or x2) in sx until it is zero, then switches to
    the other until it is zero, etc. Each step's choice depends on the
    previous step's, so this stays a Python loop over the K8 window -- bounded
    by the largest k8 among the active cells this month, not by the record
    length -- vectorized across cells within each step.

    :param state: the mutable recursion state
    :param active: which cells backtrack this call (iass in {1, 2}, k8 > 0)
    """
    if not np.any(active):
        return
    isave = np.where(active, state.iass, 0)
    max_k8 = int(state.k8[active].max())
    for i in range(max_k8 - 1, -1, -1):
        step = active & (i < state.k8)
        if not np.any(step):
            continue
        # sx1/sx2 hold 0.0 exactly where the trail has no candidate from that
        # index; the exact test is what switches the backtracking between them.
        use_sx1_branch = isave == 2
        sx2_zero = state.sx2[i] == 0  # NOSONAR
        branch_isave_a = np.where(sx2_zero, 1, 2)
        branch_sx_a = np.where(sx2_zero, state.sx1[i], state.sx2[i])
        sx1_zero = state.sx1[i] == 0  # NOSONAR
        branch_isave_b = np.where(sx1_zero, 2, 1)
        branch_sx_b = np.where(sx1_zero, state.sx2[i], state.sx1[i])
        new_isave = np.where(use_sx1_branch, branch_isave_a, branch_isave_b)
        new_sx = np.where(use_sx1_branch, branch_sx_a, branch_sx_b)
        isave = np.where(step, new_isave, isave)
        state.sx[i] = np.where(step, new_sx, state.sx[i])


def _flush_spells(state: _PalmerRecursion, flush: np.ndarray, cells: np.ndarray) -> None:
    """
    Output the PDSI/PHDI/PMDI entries for the cells whose spell closes this month.

    :param state: the mutable recursion state
    :param flush: which cells close an open spell this month (k8 > 0)
    :param cells: every cell index, parallel to ``flush``
    """
    use_all_x3 = flush & (state.iass == 3)
    backtrack = flush & ~use_all_x3

    # use all x3 values
    if np.any(use_all_x3):
        max_k8_x3 = int(state.k8[use_all_x3].max())
        for idx in range(max_k8_x3):
            step = use_all_x3 & (idx < state.k8)
            if np.any(step):
                state.sx[idx] = np.where(step, state.sx3[idx], state.sx[idx])
    if np.any(backtrack):
        _backtrack_assigned_values(state, backtrack)

    # proper assignments to array sx have been made, output the mess
    max_k8_flush = int(state.k8[flush].max())
    for idx in range(max_k8_flush + 1):
        step = flush & (idx <= state.k8)
        if not np.any(step):
            continue
        step_cells = cells[step]
        _record_index_values(
            state,
            state.indexj[idx][step],
            state.indexm[idx][step],
            state.sx[idx][step],
            step_cells,
        )


def _assign(state: _PalmerRecursion, active: np.ndarray) -> None:
    """
    Assign x values, for every active cell.

    :param state: the mutable recursion state
    :param active: which cells this call resolves this month (the caller has
        already set ``state.iass`` to 1, 2, or 3 for these cells)
    """
    if not np.any(active):
        return
    y, m = state.year, state.month
    cells = np.arange(state.k8.shape[0])
    state.sx[state.k8[active], cells[active]] = state.x[y, m][active]

    # k8 is an integer count of months pending a spell flush, not a computed
    # float, so this is an integer test rather than a float comparison.
    direct = active & (state.k8 == 0)
    flush = active & (state.k8 > 0)

    if np.any(direct):
        direct_cells = cells[direct]
        n = direct_cells.size
        _record_index_values(
            state,
            np.full(n, y),
            np.full(n, m),
            state.x[y, m][direct],
            direct_cells,
        )

    if np.any(flush):
        _flush_spells(state, flush, cells)

    state.k8 = np.where(active, 0, state.k8)
    # k8max is deliberately not reset here: it is the high-water mark
    # _finish_up reads once the whole recursion ends, not per-spell state.


def _statement_220(state: _PalmerRecursion, active: np.ndarray) -> None:
    """
    Save this month's calculated variables (v,pro,x1,x2,x3) for
    use with next month's data, for every active cell.

    Translated from statement 220 in NCEI's pdi.f

    :param state: the mutable recursion state
    :param active: which cells this call updates
    """
    if not np.any(active):
        return
    y, m = state.year, state.month
    state.v = np.where(active, state.pv, state.v)
    state.pro = np.where(active, state.ppr[y, m], state.pro)
    state.x1 = np.where(active, state.px1[y, m], state.x1)
    state.x2 = np.where(active, state.px2[y, m], state.x2)
    state.x3 = np.where(active, state.px3[y, m], state.x3)


def _statement_210(prepared: _PalmerPrepared, state: _PalmerRecursion, active: np.ndarray) -> None:
    """
    prob(end) returns to 0. A possible abatement has fizzled out,
    so we accept all stored values of x3, for every active cell.

    Translated from statement 210 in NCEI's pdi.f. Always resolves through
    ``_assign`` with iass=3: ``_assign``'s own k8==0 branch already performs
    the direct record the scalar recursion inlined here, so there is no
    separate direct-record path to keep in sync.

    :param prepared: the prepared Palmer inputs
    :param state: the mutable recursion state
    :param active: which cells this call updates
    """
    if not np.any(active):
        return
    y, m = state.year, state.month
    state.pv = np.where(active, 0.0, state.pv)
    state.px1[y, m] = np.where(active, 0.0, state.px1[y, m])
    state.px2[y, m] = np.where(active, 0.0, state.px2[y, m])
    state.ppr[y, m] = np.where(active, 0.0, state.ppr[y, m])
    m_factor, b_factor = _select_duration_factors(prepared, state)
    px3_new = DurationFactors.weighting_fraction(m_factor, b_factor) * state.x3 + state.z[y, m] / (m_factor + b_factor)
    state.px3[y, m] = np.where(active, px3_new, state.px3[y, m])
    state.x[y, m] = np.where(active, state.px3[y, m], state.x[y, m])

    state.iass = np.where(active, 3, state.iass)
    _assign(state, active)
    _statement_220(state, active)


def _statement_200(prepared: _PalmerPrepared, state: _PalmerRecursion, active: np.ndarray) -> None:
    """
    Continue x1 and x2 calculations
    if either indicates the start of a new wet or drought,
    and if the last wet or drought has ended, use x1 or x2
    as the new x3, for every active cell.

    Translated from statement 200 in NCEI's pdi.f. The four early-return
    branches and the deferral fallthrough are mutually exclusive outcomes,
    computed for every active cell and combined by mask, since a Python early
    return cannot resolve one cell's branch independently of its neighbour's.
    Each branch's write is masked to that branch alone, so a later branch's
    condition -- which, like the original's sequential ``if``, reads state a
    prior branch may have written -- sees an unmodified value for any cell
    the prior branch did not touch.

    :param prepared: the prepared Palmer inputs
    :param state: the mutable recursion state
    :param active: which cells this call updates
    """
    if not np.any(active):
        return
    y, m = state.year, state.month
    wetm, wetb = prepared.wetm, prepared.wetb
    px1_computed = DurationFactors.weighting_fraction(wetm, wetb) * state.x1 + state.z[y, m] / (wetm + wetb)
    px1_new = np.where(px1_computed > 0, px1_computed, 0.0)
    state.px1[y, m] = np.where(active, px1_new, state.px1[y, m])

    # px3 exactly 0.0 means no spell is established, and px1/px2 exactly 0.0
    # mean no incipient wet/dry index exists to promote to x3; the recursions
    # above clamp to those zeros exactly rather than interpolating to them.
    # if no existing wet spell or drought, x1 becomes the new x3
    branch1 = active & (state.px1[y, m] >= 1) & (state.px3[y, m] == 0)  # NOSONAR
    state.px3[y, m] = np.where(branch1, state.px1[y, m], state.px3[y, m])
    state.x[y, m] = np.where(branch1, state.px1[y, m], state.x[y, m])
    state.px1[y, m] = np.where(branch1, 0.0, state.px1[y, m])
    state.iass = np.where(branch1, 1, state.iass)

    drym, dryb = prepared.drym, prepared.dryb
    px2_computed = DurationFactors.weighting_fraction(drym, dryb) * state.x2 + state.z[y, m] / (drym + dryb)
    px2_new = np.where(px2_computed < 0, px2_computed, 0.0)
    state.px2[y, m] = np.where(active & ~branch1, px2_new, state.px2[y, m])

    # if no existing wet spell or drought, x2 becomes the new x3
    branch2 = active & ~branch1 & (state.px2[y, m] <= -1) & (state.px3[y, m] == 0)  # NOSONAR
    state.px3[y, m] = np.where(branch2, state.px2[y, m], state.px3[y, m])
    state.x[y, m] = np.where(branch2, state.px2[y, m], state.x[y, m])
    state.px2[y, m] = np.where(branch2, 0.0, state.px2[y, m])
    state.iass = np.where(branch2, 2, state.iass)

    # No established drought (wet spell), but x3 = 0, so either (nonzero) x1
    # or x2 must be used as x3
    resolved = branch1 | branch2
    px3_still_zero = active & ~resolved & (state.px3[y, m] == 0)  # NOSONAR
    branch3 = px3_still_zero & (state.px1[y, m] == 0)  # NOSONAR
    state.x[y, m] = np.where(branch3, state.px2[y, m], state.x[y, m])
    state.iass = np.where(branch3, 2, state.iass)

    branch4 = px3_still_zero & ~branch3 & (state.px2[y, m] == 0)  # NOSONAR
    state.x[y, m] = np.where(branch4, state.px1[y, m], state.x[y, m])
    state.iass = np.where(branch4, 1, state.iass)

    assign_mask = branch1 | branch2 | branch3 | branch4
    _assign(state, assign_mask)

    # at this point there is no determined value to assign to x for the
    # remaining cells: all the values of x1, x2, and x3 are saved. At a later
    # time x3 will reach a value where it is the value of x (pdsi). At that
    # time, _assign backtracks through choosing the appropriate x1 or x2 to
    # be that month's x.
    defer = active & ~assign_mask
    if np.any(defer):
        cells = np.arange(state.k8.shape[0])
        defer_cells = cells[defer]
        rows = state.k8[defer]
        state.sx1[rows, defer_cells] = state.px1[y, m][defer]
        state.sx2[rows, defer_cells] = state.px2[y, m][defer]
        state.sx3[rows, defer_cells] = state.px3[y, m][defer]
        state.x[y, m] = np.where(defer, state.px3[y, m], state.x[y, m])
        state.k8 = np.where(defer, state.k8 + 1, state.k8)
        state.k8max = np.where(defer, state.k8, state.k8max)

    _statement_220(state, active)


def _statement_190(prepared: _PalmerPrepared, state: _PalmerRecursion, active: np.ndarray) -> None:
    """
    drought or wet continues, calculate prob(end) (variable ze), for every
    active cell.

    Translated from statement 190 in NCEI's pdi.f

    :param prepared: the prepared Palmer inputs
    :param state: the mutable recursion state
    :param active: which cells this call updates
    """
    if not np.any(active):
        return
    y, m = state.year, state.month
    # pro is 100.0 exactly where ppr was clamped to that endpoint; the exact
    # test selects the certain-end form of q.
    q = np.where(state.pro == 100, state.ze, state.ze + state.v)  # NOSONAR
    with np.errstate(divide="ignore", invalid="ignore"):
        ppr_new = (state.pv / q) * 100

    m_factor, b_factor = _select_duration_factors(prepared, state)
    px3_candidate = DurationFactors.weighting_fraction(m_factor, b_factor) * state.x3 + state.z[y, m] / (
        m_factor + b_factor
    )
    over = ppr_new >= 100
    ppr_final = np.where(over, 100.0, ppr_new)
    px3_final = np.where(over, 0.0, px3_candidate)
    state.ppr[y, m] = np.where(active, ppr_final, state.ppr[y, m])
    state.px3[y, m] = np.where(active, px3_final, state.px3[y, m])

    _statement_200(prepared, state, active)


def _statement_180(prepared: _PalmerPrepared, state: _PalmerRecursion, active: np.ndarray) -> None:
    """
    drought abatement is possible, for every active cell.

    Translated from statement 180 in NCEI's pdi.f

    :param prepared: the prepared Palmer inputs
    :param state: the mutable recursion state
    :param active: which cells this call updates
    """
    if not np.any(active):
        return
    y, m = state.year, state.month
    uw_new = state.z[y, m] + 0.15
    pv_new = uw_new + _py_max(state.v, 0.0)
    state.uw = np.where(active, uw_new, state.uw)
    state.pv = np.where(active, pv_new, state.pv)

    # During a drought, PV <= 0 implies prob(end) has returned to 0
    fizzled = active & (state.pv <= 0)
    _statement_210(prepared, state, fizzled)

    continuing = active & ~fizzled
    m_factor, b_factor = prepared.drym, prepared.dryb
    ze_new = -b_factor * state.x3 - 0.5 * (m_factor + b_factor)
    state.ze = np.where(continuing, ze_new, state.ze)
    _statement_190(prepared, state, continuing)


def _statement_170(prepared: _PalmerPrepared, state: _PalmerRecursion, active: np.ndarray) -> None:
    """
    Wet spell abatement is possible, for every active cell.

    Translated from statement 170 in NCEI's pdi.f

    :param prepared: the prepared Palmer inputs
    :param state: the mutable recursion state
    :param active: which cells this call updates
    """
    if not np.any(active):
        return
    y, m = state.year, state.month
    ud_new = state.z[y, m] - 0.15
    pv_new = ud_new + _py_min(state.v, 0.0)
    state.ud = np.where(active, ud_new, state.ud)
    state.pv = np.where(active, pv_new, state.pv)

    # During a wet spell, PV >= 0 implies prob(end) has returned to 0
    fizzled = active & (state.pv >= 0)
    _statement_210(prepared, state, fizzled)

    continuing = active & ~fizzled
    m_factor, b_factor = prepared.wetm, prepared.wetb
    ze_new = -b_factor * state.x3 + 0.5 * (m_factor + b_factor)
    state.ze = np.where(continuing, ze_new, state.ze)
    _statement_190(prepared, state, continuing)


def _advance_month(prepared: _PalmerPrepared, state: _PalmerRecursion, year: int, month: int) -> None:
    """
    Advance the Z-index recursion by one month, for every cell at once.

    Rereads monthly parameters for calculation of the 'K' monthly weighting
    factors used in z-index calculation, then dispatches every cell to the
    established-spell logic (no abatement underway) or the
    abatement-in-progress logic. Every cell shares this same calendar step
    (``year``/``month``); the six masks below partition every cell into
    exactly one of the four statement calls, mirroring the scalar
    recursion's ``_step_established_spell`` dispatch (including its NaN
    fallthrough, reachable only when ``state.x3`` is NaN) and its abatement
    branch.

    :param prepared: the prepared Palmer inputs
    :param state: the mutable recursion state
    :param year: row index into the monthly arrays
    :param month: month index, 0 = January
    """
    state.year = year
    state.month = month
    cells = np.arange(state.k8.shape[0])
    state.indexj[state.k8, cells] = year
    state.indexm[state.k8, cells] = month
    state.ze = np.zeros_like(state.ze)
    state.ud = np.zeros_like(state.ud)
    state.uw = np.zeros_like(state.uw)
    _calc_cafec_zindex(prepared, state, year, month)

    z = state.z[year, month]
    # pro takes its endpoints exactly -- clamped to 100.0, reset to 0.0 -- and
    # either endpoint means a spell is established; values between them are
    # abatement.
    established = (state.pro == 100) | (state.pro == 0)  # NOSONAR
    abating = ~established

    # End of drought or wet
    spell_ended = established & (state.x3 >= -0.5) & (state.x3 <= 0.5)
    # We are in a wet spell
    wet = established & (state.x3 > 0.5)
    # We are in a drought
    dry = established & (state.x3 < -0.5)
    # The wet/drought spell intensifies
    wet_intensify = wet & (z >= 0.15)
    dry_intensify = dry & (z <= -0.15)
    # The wet/drought spell starts to abate (and may end)
    wet_abate = wet & ~wet_intensify
    dry_abate = dry & ~dry_intensify
    # a NaN x3 satisfies none of spell_ended/wet/dry (every comparison
    # against NaN is False), matching the scalar dispatch's own fallthrough
    nan_x3_fallback = established & ~spell_ended & ~wet & ~dry

    # Abatement is underway; a NaN x3 also takes the wet path here, matching
    # the "no abatement" branch's NaN fallthrough above
    abating_wet_or_nan = abating & ((state.x3 > 0) | np.isnan(state.x3))
    abating_dry = abating & ~abating_wet_or_nan

    # check for new wet or drought start
    y, m = year, month
    state.pv = np.where(spell_ended, 0.0, state.pv)
    state.ppr[y, m] = np.where(spell_ended, 0.0, state.ppr[y, m])
    state.px3[y, m] = np.where(spell_ended, 0.0, state.px3[y, m])
    _statement_200(prepared, state, spell_ended)
    _statement_210(prepared, state, wet_intensify | dry_intensify)
    _statement_170(prepared, state, wet_abate | nan_x3_fallback | abating_wet_or_nan)
    _statement_180(prepared, state, dry_abate | abating_dry)


def _calc_zindex(prepared: _PalmerPrepared, state: _PalmerRecursion) -> None:
    """
    Calculate Z Index

    The only remaining Python loop is over the shared calendar record (every
    cell advances the same month together); no loop runs over the cell axis.

    :param prepared: the prepared Palmer inputs
    :param state: the mutable recursion state
    """
    for year in range(prepared.n_years):
        for month in range(12):
            _advance_month(prepared, state, year, month)


def _finish_up(state: _PalmerRecursion) -> None:
    """
    Flush any spell still open when the record ends, for every cell.

    Whatever px3/x value was computed at deferral time is written out
    directly -- there is no later month to trigger ``_assign``'s backtrack
    selection between x1 and x2 -- using the final month's probability state
    for every leftover entry, per cell.

    :param state: the mutable recursion state
    """
    if not np.any(state.k8max > 0):
        return
    cells = np.arange(state.k8.shape[0])
    i_end = state.pdsi.shape[0] - 1
    max_k8max = int(state.k8max.max())
    final_wplm = _case(
        state.ppr[i_end, 11],
        state.px1[i_end, 11],
        state.px2[i_end, 11],
        state.px3[i_end, 11],
    )

    for k8 in range(max_k8max):
        step = k8 < state.k8max
        if not np.any(step):
            continue
        step_cells = cells[step]
        i = state.indexj[k8][step]
        j = state.indexm[k8][step]
        x_val = state.x[i, j, step_cells]
        px3_val = state.px3[i, j, step_cells]
        state.pdsi[i, j, step_cells] = x_val
        # the same no-established-spell fallback as _record_index_values
        state.phdi[i, j, step_cells] = np.where(px3_val == 0, x_val, px3_val)  # NOSONAR
        state.wplm[i, j, step_cells] = final_wplm[step_cells]


def _reshape_palmer_input(values: np.ndarray, spatial_time_major: bool) -> tuple[np.ndarray, tuple[int, ...]]:
    """
    Reshape a Palmer input to (years, 12, n_cells).

    A single location (1-D or (years, 12) input) is reshaped exactly as
    before and given a trailing cell axis of length 1, so the recursion has
    one code path for a single location and a spatial block. Three or more
    dimensions are read as a time-major ``(time, *cells)`` block per
    ADR-0009 -- ``compute._prepare_input_shape`` raises for an undeclared
    ambiguous shape -- with its trailing cell dimensions flattened to
    ``n_cells`` and folded onto a (years, 12) axis by
    ``compute._reshape_time_major``, the same helper ``eto.py`` reuses for
    its own spatial block.

    :param values: the input array
    :param spatial_time_major: declares an ambiguous 3+-D block as time-major
    :return: the reshaped array and the original cell_shape (() for a single location)
    """
    values = np.asarray(values)
    if values.ndim <= 2:
        reshaped = utils.reshape_to_2d(values, 12)
        return reshaped.reshape(*reshaped.shape, 1), ()
    block = compute._prepare_input_shape(values, spatial_time_major)
    cell_shape = block.shape[1:]
    n_cells = int(np.prod(cell_shape))
    flat = block.reshape(block.shape[0], n_cells)
    folded = compute._reshape_time_major(flat, compute.Periodicity.monthly)
    return folded, cell_shape


def _reshape_palmer_awc(awc: float | np.ndarray, cell_shape: tuple[int, ...], n_cells: int) -> float | np.ndarray:
    """Flatten a per-cell AWC to (n_cells,); a scalar AWC passes through unchanged."""
    if cell_shape == ():
        return awc
    return np.broadcast_to(np.asarray(awc, dtype=float), cell_shape).reshape(n_cells)


def _trim_time_major(values: np.ndarray, original_length: int) -> np.ndarray:
    """
    Fold a (years, 12, n_cells) recursion output back to (time, n_cells),
    dropping the calendar padding ``_reshape_palmer_input`` may have added.

    The recursion state always carries a cell axis (n_cells == 1 for a single
    location), so this fold is unconditional.

    :param values: a recursion output array, shape (years, 12, n_cells)
    :param original_length: the un-padded time length to trim to
    """
    return values.reshape(-1, values.shape[-1])[:original_length]


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
    awc: float | np.ndarray,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict[str, Any] | None = None,
    spatial_time_major: bool = False,
) -> _PalmerPrepared:
    """
    Initialize the prepared inputs

    :param precips: time series of monthly precipitation values, in inches,
        or a time-major (time, *cells) spatial block
    :param pet: time series of monthly PET values, in inches, matching precips' shape
    :param awc: available water capacity (soil constant), in inches; a scalar
        or an array broadcastable to the block's cell shape
    :param data_start_year: initial year of the input precipitation and PET datasets,
                            both of which are assumed to start in January of this year
    :param calibration_year_initial: initial year of the calibration period
    :param calibration_year_final: final year of the calibration period
    :param fitting_params: dictionary of the fitted parameters
    :param spatial_time_major: declares an ambiguous 3+-D precips/pet as a
        time-major spatial block per ADR-0009
    :return the initialized prepared inputs
    :rtype: _PalmerPrepared
    """
    # reshape precipitation and PET to (years, 12, n_cells); n_cells == 1 and
    # cell_shape == () for a single location
    precips, cell_shape = _reshape_palmer_input(precips, spatial_time_major)
    pet, _ = _reshape_palmer_input(pet, spatial_time_major)
    n_years = int(precips.shape[0])
    n_cells = int(precips.shape[2])
    awc = _reshape_palmer_awc(awc, cell_shape, n_cells)
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
        n_cells=n_cells,
        cell_shape=cell_shape,
        spdat=np.full((n_years, 12, n_cells), np.nan),
        pldat=np.full((n_years, 12, n_cells), np.nan),
        prdat=np.full((n_years, 12, n_cells), np.nan),
        rdat=np.full((n_years, 12, n_cells), np.nan),
        tldat=np.full((n_years, 12, n_cells), np.nan),
        etdat=np.full((n_years, 12, n_cells), np.nan),
        rodat=np.full((n_years, 12, n_cells), np.nan),
        sssdat=np.full((n_years, 12, n_cells), np.nan),
        ssudat=np.full((n_years, 12, n_cells), np.nan),
        psum=np.zeros((12, n_cells)),
        spsum=np.zeros((12, n_cells)),
        petsum=np.zeros((12, n_cells)),
        plsum=np.zeros((12, n_cells)),
        prsum=np.zeros((12, n_cells)),
        rsum=np.zeros((12, n_cells)),
        tlsum=np.zeros((12, n_cells)),
        etsum=np.zeros((12, n_cells)),
        rosum=np.zeros((12, n_cells)),
        alpha=np.full((12, n_cells), np.nan),
        beta=np.full((12, n_cells), np.nan),
        gamma=np.full((12, n_cells), np.nan),
        delta=np.full((12, n_cells), np.nan),
        trat=np.full((12, n_cells), np.nan),
        ak=np.full((12, n_cells), np.nan),
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

    The K8 window (``indexj``/``indexm``/``sx``/``sx1``/``sx2``/``sx3``) is
    preallocated to the full month count rather than the historical
    ``K8_SIZE`` bound: a spell cannot outlast the record, this removes the
    scalar recursion's runtime ``np.append`` growth, and unlike that growth
    it is safe to size once for every cell rather than per cell.

    :param prepared: the prepared Palmer inputs
    :return the initialized recursion state
    :rtype: _PalmerRecursion
    """
    n_years = prepared.n_years
    n_cells = prepared.n_cells
    n_months = n_years * 12
    return _PalmerRecursion(
        indexj=np.zeros((n_months, n_cells), dtype=int),
        indexm=np.zeros((n_months, n_cells), dtype=int),
        sx=np.zeros((n_months, n_cells)),
        sx1=np.zeros((n_months, n_cells)),
        sx2=np.zeros((n_months, n_cells)),
        sx3=np.zeros((n_months, n_cells)),
        ppr=np.zeros((n_years, 12, n_cells)),
        px1=np.zeros((n_years, 12, n_cells)),
        px2=np.zeros((n_years, 12, n_cells)),
        px3=np.zeros((n_years, 12, n_cells)),
        x=np.zeros((n_years, 12, n_cells)),
        z=np.full((n_years, 12, n_cells), np.nan),
        pdsi=np.full((n_years, 12, n_cells), np.nan),
        phdi=np.full((n_years, 12, n_cells), np.nan),
        wplm=np.full((n_years, 12, n_cells), np.nan),
        k8=np.zeros((n_cells,), dtype=int),
        k8max=np.zeros((n_cells,), dtype=int),
        year=0,
        month=0,
        iass=np.zeros((n_cells,), dtype=int),
        v=np.zeros((n_cells,)),
        pro=np.zeros((n_cells,)),
        x1=np.zeros((n_cells,)),
        x2=np.zeros((n_cells,)),
        x3=np.zeros((n_cells,)),
        ze=np.zeros((n_cells,)),
        ud=np.zeros((n_cells,)),
        uw=np.zeros((n_cells,)),
        pv=np.zeros((n_cells,)),
    )


def _bind_palmer_log(
    index_type: str,
    precips: np.ndarray,
    awc: float | np.ndarray,
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


_DURATION_FACTOR_PARAM_NAMES = ("wetm", "wetb", "drym", "dryb")


def _duration_factor_override(fitting_params: dict[str, Any] | None) -> DurationFactors | None:
    """
    Read the optional duration-factor override from the caller's fitting parameters.

    ``pdsi()`` accepts the same ``wetm``/``wetb``/``drym``/``dryb`` keys scPDSI returns,
    so a caller can run the standard recursion with custom duration factors instead of
    Palmer's fixed national constants. Supplying only some of the four is a caller
    error rather than a silent partial default. scPDSI never calls this: its duration
    factors are always self-calibrated, and it ignores these keys.

    :param fitting_params: the caller's fitting parameters, if any
    :return: the validated override, or None when no duration factors were supplied
    :raises ValueError: if only some of the four keys were supplied, or a supplied
        value is not a finite scalar
    :raises ConvergenceError: if the override does not yield contracting recurrence
        coefficients, per :meth:`DurationFactors.from_fitted`
    """
    if fitting_params is None:
        return None
    supplied = [name for name in _DURATION_FACTOR_PARAM_NAMES if name in fitting_params]
    if not supplied:
        return None
    missing = [name for name in _DURATION_FACTOR_PARAM_NAMES if name not in fitting_params]
    if missing:
        raise ValueError(
            f"duration-factor override requires all of wetm, wetb, drym, and dryb; missing: {', '.join(missing)}"
        )
    values = []
    for name in _DURATION_FACTOR_PARAM_NAMES:
        try:
            value = np.asarray(fitting_params[name], dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"duration-factor override {name} must be a finite scalar") from error
        if value.ndim != 0 or not np.isfinite(value):
            raise ValueError(f"duration-factor override {name} must be a finite scalar")
        values.append(float(value))
    try:
        return DurationFactors.from_fitted(*values)
    except ConvergenceError as error:
        # the shared validation names the Wells lineage and the scPDSI
        # calibration; attribute the failure to this pdsi-only override
        raise ConvergenceError(
            f"invalid duration-factor override for the standard PDSI recursion: {error}",
            algorithm="PDSI duration-factor override",
            underlying_error=error,
        ) from error


def _prepare_palmer_data(
    precips: np.ndarray,
    pet: np.ndarray,
    awc: float | np.ndarray,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict[str, Any] | None,
    log: BoundLogger,
    spatial_time_major: bool = False,
    duration_factors: DurationFactors | None = None,
) -> tuple[_PalmerPrepared, int]:
    """Validate inputs and run the water-balance/CAFEC stages shared by Palmer indices."""
    if np.any(precips < 0.0):
        log.warning("negative_values_clipped", field="precips")
        precips = np.clip(precips, a_min=0.0, a_max=None)

    # for a spatial block, "original length" is the time axis alone, not
    # every cell's worth of it
    original_length = precips.shape[0] if precips.ndim > 2 else precips.size
    prepared = _initialize_prepared(
        precips=precips,
        pet=pet,
        awc=awc,
        data_start_year=data_start_year,
        calibration_year_initial=calibration_year_initial,
        calibration_year_final=calibration_year_final,
        fitting_params=fitting_params,
        spatial_time_major=spatial_time_major,
    )
    # _initialize_prepared can only set Palmer's fixed defaults, and scPDSI
    # bypasses these fields entirely (it passes its fitted factors straight to
    # _palmer_wells.calculate), so a pdsi()-only override lands here, before any
    # recursion stage reads them.
    if duration_factors is not None:
        prepared.wetm = duration_factors.wetm
        prepared.wetb = duration_factors.wetb
        prepared.drym = duration_factors.drym
        prepared.dryb = duration_factors.dryb
    _calc_water_balances(prepared)
    if prepared.calibrate:
        _calc_cafec_coefficients(prepared)
    _calc_zindex_factors(prepared)
    return prepared, original_length


def _palmer_cafec_params(prepared: _PalmerPrepared) -> dict[str, Any]:
    """
    The alpha/beta/gamma/delta CAFEC parameters, in the caller's shape.

    ``prepared.alpha`` etc. are always 1-D ``(12,)`` when ``fitting_params``
    was supplied (they are the caller's own arrays, shared across every
    cell) and otherwise ``(12, n_cells)``, which is reshaped to
    ``(12, *cell_shape)`` for a spatial block or squeezed back to ``(12,)``
    for a single location, matching :func:`pdsi`'s pre-existing contract.
    """
    alpha, beta, gamma, delta = prepared.alpha, prepared.beta, prepared.gamma, prepared.delta
    if alpha.ndim == 1:
        return {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta}
    if prepared.cell_shape:
        return {
            "alpha": alpha.reshape(12, *prepared.cell_shape),
            "beta": beta.reshape(12, *prepared.cell_shape),
            "gamma": gamma.reshape(12, *prepared.cell_shape),
            "delta": delta.reshape(12, *prepared.cell_shape),
        }
    return {"alpha": alpha[:, 0], "beta": beta[:, 0], "gamma": gamma[:, 0], "delta": delta[:, 0]}


def _calculate_pdsi_prepared(prepared: _PalmerPrepared, original_length: int) -> _PalmerResult:
    """Complete standard PDSI after the shared Palmer preparation stages."""
    _calc_kfactors(prepared)
    state = _initialize_recursion(prepared)
    _calc_zindex(prepared, state)
    _finish_up(state)

    pdsi_result = _trim_time_major(state.pdsi, original_length)
    phdi = _trim_time_major(state.phdi, original_length)
    wplm = _trim_time_major(state.wplm, original_length)
    z = _trim_time_major(state.z, original_length)
    if prepared.cell_shape:
        pdsi_result = pdsi_result.reshape(original_length, *prepared.cell_shape)
        phdi = phdi.reshape(original_length, *prepared.cell_shape)
        wplm = wplm.reshape(original_length, *prepared.cell_shape)
        z = z.reshape(original_length, *prepared.cell_shape)
    else:
        pdsi_result = pdsi_result[:, 0]
        phdi = phdi[:, 0]
        wplm = wplm[:, 0]
        z = z[:, 0]
    params = _palmer_cafec_params(prepared)
    params.update(wetm=prepared.wetm, wetb=prepared.wetb, drym=prepared.drym, dryb=prepared.dryb)
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
    # a fixed three rescaling passes, not an iteration to a fixed point; reported
    # in the result parameters so callers and tests can pin the count
    rescale_passes = 3
    for _ in range(rescale_passes):
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

    params: dict[str, Any] = _palmer_cafec_params(prepared)
    params.update(wetm=wetm, wetb=wetb, drym=drym, dryb=dryb, rescale_passes=rescale_passes)
    return _PalmerResult(
        recursion.pdsi[:original_length],
        recursion.phdi[:original_length],
        recursion.pmdi[:original_length],
        z_values[:original_length],
        params,
    )


def _fill_masked_with_nan(values: np.ndarray) -> np.ndarray:
    """
    Replace a masked array with a plain float array whose masked elements are NaN.

    The reshaper and the recursion call ``np.asarray``, which drops a mask and
    exposes the backing values underneath, so a masked grid cell would be
    computed from -- and publish -- the data the caller marked missing.
    Converting once at the shared calculation entry makes a masked element
    indistinguishable from NaN input everywhere downstream, including the
    per-cell all-missing test.

    :param values: the input array
    :return: the input unchanged, or the masked input's data with NaN under its mask
    """
    if not np.ma.isMaskedArray(values):
        return values
    return np.ma.filled(values.astype(float), np.nan)


def _mask_fully_missing_cells(precips: np.ndarray, result: _PalmerResult) -> _PalmerResult:
    """
    NaN out any cell whose entire time series is missing, in a spatial block.

    A standalone call on an all-missing series takes the ``all_missing``
    shortcut above and returns NaN outputs directly. Inside a block that is
    NOT entirely missing, that shortcut never fires, so a fully-missing cell
    instead runs the recursion like any other: Python's ``max(0, ...)``/
    ``min(0.0, ...)`` calls in ``_statement_200`` (replicated exactly by
    :func:`_py_max`/:func:`_py_min` for bit-for-bit equivalence) silently
    turn a NaN Z-index into 0 rather than propagating it, so an unmasked
    fully-missing cell would read back as a misleadingly ordinary near-zero
    PDSI instead of missing data -- the wrong answer for a real grid's
    ocean or no-data cells. This reproduces the standalone shortcut's NaN
    result for exactly those cells, leaving every other cell untouched.

    :param precips: the original, un-reshaped spatial block, (time, *cells)
    :param result: the block result to mask
    """
    fully_missing = np.all(np.isnan(precips), axis=0)
    if not np.any(fully_missing):
        return result
    return _PalmerResult(
        np.where(fully_missing, np.nan, result.pdsi),
        np.where(fully_missing, np.nan, result.phdi),
        np.where(fully_missing, np.nan, result.pmdi),
        np.where(fully_missing, np.nan, result.zindex),
        result.params,
    )


def _palmer_calculation(
    index_type: str,
    calculate_prepared: Callable[[_PalmerPrepared, int], _PalmerResult],
    precips: np.ndarray,
    pet: np.ndarray,
    awc: float | np.ndarray,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict[str, Any] | None,
    spatial_time_major: bool = False,
    use_fitting_duration_factors: bool = False,
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
        # resolved inside the try so a malformed override emits the same
        # calculation_started/calculation_failed lifecycle as other input errors
        duration_factors = _duration_factor_override(fitting_params) if use_fitting_duration_factors else None
        # equal element counts are not enough to pair a spatial block: (time, 2, 3)
        # and (time, 3, 2) have the same size but flatten their cells in different
        # spatial order, and a block's time axis is not recoverable from size alone.
        # 1-D and 2-D input stays interchangeable -- (time,) and (years, 12) are the
        # same series folded the same way.
        precips_shape = np.shape(precips)
        pet_shape = np.shape(pet)
        if precips_shape != pet_shape and (precips.size != pet.size or len(precips_shape) > 2 or len(pet_shape) > 2):
            message = "Incompatible precipitation and PET arrays"
            log.error("validation_failed", reason=message)
            raise ValueError(message)

        precips = _fill_masked_with_nan(precips)
        pet = _fill_masked_with_nan(pet)

        if np.any(np.isinf(precips)) or np.any(np.isinf(pet)):
            message = "precipitation and PET arrays cannot contain infinite values"
            log.error("validation_failed", reason=message)
            raise ValueError(message)

        all_missing = np.all(np.isnan(precips))
        if all_missing:
            reshaped, _ = _reshape_palmer_input(precips, spatial_time_major)
            _validate_calibration_period(
                data_start_year,
                int(reshaped.shape[0]),
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
            spatial_time_major=spatial_time_major,
            duration_factors=duration_factors,
        )
        result = calculate_prepared(prepared, original_length)
        if precips.ndim > 2:
            result = _mask_fully_missing_cells(precips, result)
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
    awc: float | np.ndarray,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict[str, Any] | None = None,
    spatial_time_major: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    """
    Compute the Palmer Drought Severity Index (PDSI),
    Palmer Hydrological Drought Index (PHDI),
    Palmer Modified Drought Index (PMDI), and
    Palmer Z-Index.

    Args:
        precips: Time series of monthly precipitation values, in inches, or
            a time-major spatial block with shape ``(time, *cells)`` and three
            or more dimensions, per ADR-0009 and ADR-0011. A block whose first
            cell axis is a calendar period length (12 or 366) is ambiguous
            with a ``(years, periods, *cells)`` array and requires
            ``spatial_time_major=True``.
        pet: Time series of monthly PET values, in inches, matching precips'
            shape.
        awc: Available water capacity (soil constant), in inches. A scalar,
            or an array broadcastable to precips' cell shape for a spatial
            block.
        data_start_year: Initial year of the input precipitation and PET
            datasets, both of which are assumed to start in January of this
            year.
        calibration_year_initial: Initial year of the calibration period.
        calibration_year_final: Final year of the calibration period.
        fitting_params: Dictionary of the fitted parameters. For a spatial
            block, each coefficient is still (12,), shared across every cell.
            Supplying all four of ``wetm``, ``wetb``, ``drym``, and ``dryb``
            overrides Palmer's fixed national duration factors with those
            scalars (validated like scPDSI's calibrated factors); supplying
            only some of the four raises :class:`ValueError`.
        spatial_time_major: Declares a three-or-more-dimensional precips/pet
            as a time-major spatial block, per ADR-0009.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
            A five-item tuple containing NumPy arrays of PDSI, PHDI, PMDI, and
            Z-Index values, respectively, and a dictionary containing the
            fitted ``alpha``, ``beta``, ``gamma``, and ``delta`` parameters
            plus the effective ``wetm``, ``wetb``, ``drym``, and ``dryb``
            duration factors (Palmer's defaults unless overridden). Passing
            that dictionary back as ``fitting_params`` reproduces the run; a
            spatial block's returned coefficient arrays are re-fit rather
            than reused, which is numerically equivalent.
            For all-missing input, the parameter dictionary is ``None``. A
            spatial block's outputs keep precips' cell shape, and the
            parameter arrays gain the same trailing shape unless
            ``fitting_params`` was supplied.

    Raises:
        ValueError: If precipitation and PET have incompatible shapes, if
            precips/pet contains infinite values, if the calibration period
            is not contained in the data years, or if only some of the
            ``wetm``/``wetb``/``drym``/``dryb`` override keys were supplied
            or a supplied value is not a finite scalar.
        ConvergenceError: If a supplied duration-factor override does not
            yield contracting recurrence coefficients.
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
        spatial_time_major=spatial_time_major,
        use_fitting_duration_factors=True,
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
            behavior; duration factors are always recalibrated, so the
            ``wetm``/``wetb``/``drym``/``dryb`` override keys :func:`pdsi`
            accepts are ignored here.

    Returns:
        A tuple containing scPDSI, scPHDI, scPMDI, the cumulatively calibrated
        Z-index, and fitted parameters. The parameter dictionary contains
        ``alpha``, ``beta``, ``gamma``, ``delta``, ``wetm``, ``wetb``,
        ``drym``, ``dryb``, and ``rescale_passes`` (the fixed count of Z-index
        rescaling passes). All-missing input returns four same-length
        missing arrays and ``None``.

    Raises:
        ValueError: If precipitation and PET have different lengths, or if
            precips/pet is a spatial block (three or more dimensions).
            scPDSI runs the Wells backtracking recursion once per cell plus
            per-location duration-factor fits, so it stays on the
            per-location path -- see ADR-0011 -- while :func:`pdsi` vectorizes
            across a block's cells.
        InsufficientDataError: If the calibration period cannot supply a
            complete duration-factor fitting window.
        ConvergenceError: If a numerical calibration stage produces unusable
            factors, percentile anchors, or recurrence denominators. The
            duration-factor fit must yield contracting recurrence coefficients;
            a short or climatologically skewed calibration period can pull a
            fitted slope non-positive and trigger this (see
            :func:`climate_indices.self_calibration.duration_factors`).
    """
    if np.ndim(precips) > 2 or np.ndim(pet) > 2:
        raise ValueError(
            "scpdsi() does not support a spatial block (three or more dimensions); "
            "self-calibrating PDSI runs per location -- see ADR-0011. Use pdsi() "
            "with spatial_time_major=True for vectorized gridded PDSI."
        )
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
