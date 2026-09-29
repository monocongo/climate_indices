"""NCEI ``pdi.f``-lineage Palmer spell recursion for the standard PDSI family.

:mod:`climate_indices.palmer` prepares the water balance, CAFEC coefficients, and
the K-factor-weighted Z-index series; this module owns the spell recursion those
feed.  It is the sibling of :mod:`climate_indices._palmer_wells`: both take a
precomputed Z-index series and a validated :class:`PdiDurationFactors`/
:class:`~climate_indices._palmer_duration.DurationFactors`, and differ only in the
spell-selection rules and in how each lineage expresses its duration-factor
weighting fraction.  The ``pdi.f`` lineage divides (``c = b / (m + b)``); the
Wells lineage subtracts the complement.  The forms agree in exact arithmetic but
may differ bitwise, and the recursions branch on exact comparisons, so each
module keeps its own expression and its own validation rather than sharing one.
"""

from dataclasses import dataclass

import numpy as np

from climate_indices.exceptions import ConvergenceError


@dataclass(frozen=True)
class PdiResult:
    """Arrays produced by one complete ``pdi.f`` recursion pass."""

    pdsi: np.ndarray
    phdi: np.ndarray
    pmdi: np.ndarray


@dataclass(frozen=True)
class PdiDurationFactors:
    """Validated duration factors for the ``pdi.f``-lineage spell recursion.

    The ``pdi.f`` recurrence forms only the two weighting fractions
    ``c = b / (m + b)`` (one per spell sign) and the two denominators ``m + b``;
    it never builds the cross coefficient the Wells lineage needs.  Validation
    therefore covers exactly those terms, so a factor set the ``pdi.f`` recursion
    can use is accepted even when the Wells coefficients derived from it would
    not be contractions.
    """

    wetm: float
    wetb: float
    drym: float
    dryb: float

    @classmethod
    def from_fitted(cls, wetm: float, wetb: float, drym: float, dryb: float) -> "PdiDurationFactors":
        """Validate fitted duration factors against the terms the ``pdi.f`` recursion uses.

        Both denominators must be finite and strictly positive, and both weighting
        fractions ``c = b / (m + b)`` must be contractions (``|c| < 1``) so the
        unclamped ``px3`` recurrence cannot diverge.  This is deliberately narrower
        than :meth:`climate_indices._palmer_duration.DurationFactors.from_fitted`,
        which also rejects a non-contracting Wells cross coefficient that this
        lineage never computes.

        :param wetm: wet duration-factor slope
        :param wetb: wet duration-factor intercept
        :param drym: dry duration-factor slope
        :param dryb: dry duration-factor intercept
        :raises ConvergenceError: if a denominator is non-finite or non-positive, or a
            weighting fraction is non-finite or has magnitude >= 1
        """
        wet_denominator = wetm + wetb
        dry_denominator = drym + dryb
        if (
            not np.isfinite(wet_denominator)
            or wet_denominator <= 0.0
            or not np.isfinite(dry_denominator)
            or dry_denominator <= 0.0
        ):
            raise ConvergenceError(
                "invalid fitted duration factors for the standard PDSI recursion",
                algorithm="PDSI duration-factor calibration",
            )
        for name, intercept, denominator in (
            ("wetc", wetb, wet_denominator),
            ("dryc", dryb, dry_denominator),
        ):
            coefficient = intercept / denominator
            if not np.isfinite(coefficient) or abs(coefficient) >= 1.0:
                raise ConvergenceError(
                    f"invalid fitted duration factors for the standard PDSI recursion: "
                    f"{name} = {coefficient!r} is non-finite or has magnitude >= 1",
                    algorithm="PDSI duration-factor calibration",
                )
        return cls(wetm=wetm, wetb=wetb, drym=drym, dryb=dryb)


def _weighting_fraction(m: float | np.ndarray, b: float | np.ndarray) -> float | np.ndarray:
    """The ``pdi.f`` weighting fraction ``c = b / (m + b)``.

    ``m``/``b`` are scalars for a single location and length-``n_cells`` arrays for
    a vectorized block; the zero check covers every element.  The division form is
    load-bearing: it is not bitwise equal to the Wells complement form, and the
    recursion branches on exact comparisons.

    :raises ValueError: if the duration factors sum to zero for any element
    """
    denominator = m + b
    if bool(np.any(denominator == 0)):
        raise ValueError("duration-factor slope and intercept must not sum to zero")
    return b / denominator


@dataclass
class _State:
    """Mutable per-location recursion state and the arrays the recursion fills.

    Every field carries the same trailing ``(n_cells,)`` (or
    ``(n_months, n_cells)`` / ``(n_years, 12, n_cells)``) cell axis, including the
    "scalar" month-carry state, so a single location (``n_cells == 1``) and a
    spatial block share one code path.  ``year``/``month`` stay plain ints.
    """

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
    z: np.ndarray
    pdsi: np.ndarray
    phdi: np.ndarray
    wplm: np.ndarray
    k8: np.ndarray
    k8max: np.ndarray
    year: int
    month: int
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


def _initialize_state(z_values: np.ndarray) -> _State:
    """Construct the zeroed recursion state around a precomputed Z-index series.

    The K8 window (``indexj``/``indexm``/``sx``/``sx1``/``sx2``/``sx3``) is
    preallocated to the full month count: a spell cannot outlast the record, and
    unlike the scalar recursion's runtime growth it is safe to size once for every
    cell rather than per cell.
    """
    n_years, _, n_cells = z_values.shape
    n_months = n_years * 12
    return _State(
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
        z=z_values,
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


def _select_duration_factors(factors: PdiDurationFactors, state: _State) -> tuple[np.ndarray, np.ndarray]:
    """Select the wet or dry duration factors based on the sign of X3, per cell.

    X3 equal to zero means that no wet or dry spell is established. It is
    assigned the wet factors to preserve the recursion's historical
    non-negative tie-break.
    """
    m = np.where(state.x3 >= 0, factors.wetm, factors.drym)
    b = np.where(state.x3 >= 0, factors.wetb, factors.dryb)
    return m, b


def _py_max(a: float | np.ndarray, b: float | np.ndarray) -> np.ndarray:
    """``max(a, b)`` matching Python's builtin comparison order, not ``np.maximum``.

    Python's ``max(a, b)`` returns ``b`` only if ``b > a``, so a NaN in ``b`` never
    wins while a NaN in ``a`` always loses unless ``b`` also fails to compare
    greater -- an asymmetry ``np.maximum`` does not have.  The recursion below calls
    Python's builtin with data-dependent NaN possible in either position, so
    replicating this exact rule is required for bit-for-bit equivalence.
    """
    return np.where(np.asarray(b) > a, b, a)


def _py_min(a: float | np.ndarray, b: float | np.ndarray) -> np.ndarray:
    """``min(a, b)`` matching Python's builtin comparison order; see :func:`_py_max`."""
    return np.where(np.asarray(b) < a, b, a)


def _case(prob: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    """Select the preliminary (or near-real time) PDSI, for every cell.

    :param prob: the probability of ending either a drought or wet spell
    :param x1: index for incipient wet spells (always positive)
    :param x2: index for incipient dry spells (always negative)
    :param x3: severity index for an established wet spell or drought
    """
    near_normal = np.where(np.abs(x1) > np.abs(x2), x1, x2)

    pro = prob / 100.0
    interpolated = np.where(x3 <= 0, (1.0 - pro) * x3 + pro * x1, (1.0 - pro) * x3 + pro * x2)
    established = np.where((prob <= 0) | (prob >= 100), x3, interpolated)

    # x3 is assigned 0.0 exactly when no spell is established, so its exact
    # zero -- not a tolerance -- is what selects the near-normal value.
    return np.where(x3 == 0, near_normal, established)  # NOSONAR


def _record_index_values(
    state: _State,
    years: np.ndarray,
    months: np.ndarray,
    values: np.ndarray,
    cell_ids: np.ndarray,
) -> None:
    """Record PDSI, PHDI, and PMDI for a set of (cell, month) entries."""
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


def _backtrack_assigned_values(state: _State, active: np.ndarray) -> None:
    """Backtrack through the x1/x2 trail arrays, for every active cell."""
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


def _flush_spells(state: _State, flush: np.ndarray, cells: np.ndarray) -> None:
    """Output the PDSI/PHDI/PMDI entries for the cells whose spell closes this month."""
    use_all_x3 = flush & (state.iass == 3)
    backtrack = flush & ~use_all_x3

    if np.any(use_all_x3):
        max_k8_x3 = int(state.k8[use_all_x3].max())
        for idx in range(max_k8_x3):
            step = use_all_x3 & (idx < state.k8)
            if np.any(step):
                state.sx[idx] = np.where(step, state.sx3[idx], state.sx[idx])
    if np.any(backtrack):
        _backtrack_assigned_values(state, backtrack)

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


def _assign(state: _State, active: np.ndarray) -> None:
    """Assign x values, for every active cell."""
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


def _statement_220(state: _State, active: np.ndarray) -> None:
    """Save this month's variables (v, pro, x1, x2, x3) for next month, per active cell."""
    if not np.any(active):
        return
    y, m = state.year, state.month
    state.v = np.where(active, state.pv, state.v)
    state.pro = np.where(active, state.ppr[y, m], state.pro)
    state.x1 = np.where(active, state.px1[y, m], state.x1)
    state.x2 = np.where(active, state.px2[y, m], state.x2)
    state.x3 = np.where(active, state.px3[y, m], state.x3)


def _statement_210(factors: PdiDurationFactors, state: _State, active: np.ndarray) -> None:
    """Prob(end) returns to 0; accept all stored x3 values, per active cell."""
    if not np.any(active):
        return
    y, m = state.year, state.month
    state.pv = np.where(active, 0.0, state.pv)
    state.px1[y, m] = np.where(active, 0.0, state.px1[y, m])
    state.px2[y, m] = np.where(active, 0.0, state.px2[y, m])
    state.ppr[y, m] = np.where(active, 0.0, state.ppr[y, m])
    m_factor, b_factor = _select_duration_factors(factors, state)
    px3_new = _weighting_fraction(m_factor, b_factor) * state.x3 + state.z[y, m] / (m_factor + b_factor)
    state.px3[y, m] = np.where(active, px3_new, state.px3[y, m])
    state.x[y, m] = np.where(active, state.px3[y, m], state.x[y, m])

    state.iass = np.where(active, 3, state.iass)
    _assign(state, active)
    _statement_220(state, active)


def _statement_200(factors: PdiDurationFactors, state: _State, active: np.ndarray) -> None:
    """Continue x1/x2, and promote a new x3 when the previous spell has ended."""
    if not np.any(active):
        return
    y, m = state.year, state.month
    wetm, wetb = factors.wetm, factors.wetb
    px1_computed = _weighting_fraction(wetm, wetb) * state.x1 + state.z[y, m] / (wetm + wetb)
    px1_new = np.where(px1_computed > 0, px1_computed, 0.0)
    state.px1[y, m] = np.where(active, px1_new, state.px1[y, m])

    # px3 exactly 0.0 means no spell is established, and px1/px2 exactly 0.0
    # mean no incipient wet/dry index exists to promote to x3; the recursions
    # above clamp to those zeros exactly rather than interpolating to them.
    branch1 = active & (state.px1[y, m] >= 1) & (state.px3[y, m] == 0)  # NOSONAR
    state.px3[y, m] = np.where(branch1, state.px1[y, m], state.px3[y, m])
    state.x[y, m] = np.where(branch1, state.px1[y, m], state.x[y, m])
    state.px1[y, m] = np.where(branch1, 0.0, state.px1[y, m])
    state.iass = np.where(branch1, 1, state.iass)

    drym, dryb = factors.drym, factors.dryb
    px2_computed = _weighting_fraction(drym, dryb) * state.x2 + state.z[y, m] / (drym + dryb)
    px2_new = np.where(px2_computed < 0, px2_computed, 0.0)
    state.px2[y, m] = np.where(active & ~branch1, px2_new, state.px2[y, m])

    branch2 = active & ~branch1 & (state.px2[y, m] <= -1) & (state.px3[y, m] == 0)  # NOSONAR
    state.px3[y, m] = np.where(branch2, state.px2[y, m], state.px3[y, m])
    state.x[y, m] = np.where(branch2, state.px2[y, m], state.x[y, m])
    state.px2[y, m] = np.where(branch2, 0.0, state.px2[y, m])
    state.iass = np.where(branch2, 2, state.iass)

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
    # time x3 will reach a value where it is the value of x (pdsi).
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


def _statement_190(factors: PdiDurationFactors, state: _State, active: np.ndarray) -> None:
    """A drought or wet spell continues; calculate prob(end) (ze), per active cell."""
    if not np.any(active):
        return
    y, m = state.year, state.month
    # pro is 100.0 exactly where ppr was clamped to that endpoint; the exact
    # test selects the certain-end form of q.
    q = np.where(state.pro == 100, state.ze, state.ze + state.v)  # NOSONAR
    with np.errstate(divide="ignore", invalid="ignore"):
        ppr_new = (state.pv / q) * 100

    m_factor, b_factor = _select_duration_factors(factors, state)
    px3_candidate = _weighting_fraction(m_factor, b_factor) * state.x3 + state.z[y, m] / (m_factor + b_factor)
    over = ppr_new >= 100
    ppr_final = np.where(over, 100.0, ppr_new)
    px3_final = np.where(over, 0.0, px3_candidate)
    state.ppr[y, m] = np.where(active, ppr_final, state.ppr[y, m])
    state.px3[y, m] = np.where(active, px3_final, state.px3[y, m])

    _statement_200(factors, state, active)


def _statement_180(factors: PdiDurationFactors, state: _State, active: np.ndarray) -> None:
    """Drought abatement is possible, for every active cell."""
    if not np.any(active):
        return
    y, m = state.year, state.month
    uw_new = state.z[y, m] + 0.15
    pv_new = uw_new + _py_max(state.v, 0.0)
    state.uw = np.where(active, uw_new, state.uw)
    state.pv = np.where(active, pv_new, state.pv)

    # During a drought, PV <= 0 implies prob(end) has returned to 0
    fizzled = active & (state.pv <= 0)
    _statement_210(factors, state, fizzled)

    continuing = active & ~fizzled
    m_factor, b_factor = factors.drym, factors.dryb
    ze_new = -b_factor * state.x3 - 0.5 * (m_factor + b_factor)
    state.ze = np.where(continuing, ze_new, state.ze)
    _statement_190(factors, state, continuing)


def _statement_170(factors: PdiDurationFactors, state: _State, active: np.ndarray) -> None:
    """Wet spell abatement is possible, for every active cell."""
    if not np.any(active):
        return
    y, m = state.year, state.month
    ud_new = state.z[y, m] - 0.15
    pv_new = ud_new + _py_min(state.v, 0.0)
    state.ud = np.where(active, ud_new, state.ud)
    state.pv = np.where(active, pv_new, state.pv)

    # During a wet spell, PV >= 0 implies prob(end) has returned to 0
    fizzled = active & (state.pv >= 0)
    _statement_210(factors, state, fizzled)

    continuing = active & ~fizzled
    m_factor, b_factor = factors.wetm, factors.wetb
    ze_new = -b_factor * state.x3 + 0.5 * (m_factor + b_factor)
    state.ze = np.where(continuing, ze_new, state.ze)
    _statement_190(factors, state, continuing)


def _advance_month(factors: PdiDurationFactors, state: _State, year: int, month: int) -> None:
    """Advance the spell recursion by one month, for every cell at once.

    Every cell shares this same calendar step (``year``/``month``); the six masks
    below partition every cell into exactly one of the four statement calls,
    mirroring the scalar recursion's dispatch (including its NaN fallthrough,
    reachable only when ``state.x3`` is NaN) and its abatement branch.
    """
    state.year = year
    state.month = month
    cells = np.arange(state.k8.shape[0])
    state.indexj[state.k8, cells] = year
    state.indexm[state.k8, cells] = month
    state.ze = np.zeros_like(state.ze)
    state.ud = np.zeros_like(state.ud)
    state.uw = np.zeros_like(state.uw)

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
    _statement_200(factors, state, spell_ended)
    _statement_210(factors, state, wet_intensify | dry_intensify)
    _statement_170(factors, state, wet_abate | nan_x3_fallback | abating_wet_or_nan)
    _statement_180(factors, state, dry_abate | abating_dry)


def _finish_up(state: _State) -> None:
    """Flush any spell still open when the record ends, for every cell."""
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


def calculate(z_values: np.ndarray, factors: PdiDurationFactors) -> PdiResult:
    """Run the ``pdi.f`` Palmer recursion over a precomputed Z-index series.

    The Z series is the K-factor-weighted departure computed by
    :mod:`climate_indices.palmer`; this function owns only the spell recursion
    downstream of it, matching :func:`climate_indices._palmer_wells.calculate`.

    Args:
        z_values: K-factor-weighted Z-index values, shaped ``(years, 12, n_cells)``
            with NaN denoting a missing period.
        factors: the validated duration factors the recurrence uses.

    Returns:
        The final PDSI, PHDI, and PMDI series, each shaped like ``z_values``.
    """
    z = np.asarray(z_values, dtype=float)
    if z.ndim != 3:
        raise ValueError(f"z_values must have shape (years, 12, n_cells), got {z.shape}")
    state = _initialize_state(z)
    n_years = z.shape[0]
    for year in range(n_years):
        for month in range(12):
            _advance_month(factors, state, year, month)
    _finish_up(state)
    return PdiResult(pdsi=state.pdsi, phdi=state.phdi, pmdi=state.wplm)
