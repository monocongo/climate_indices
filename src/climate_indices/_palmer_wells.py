"""Wells-lineage Palmer recursion used by self-calibrating PDSI.

The standard :mod:`climate_indices.palmer` recursion follows NCEI's ``pdi.f``
lineage.  Self-calibrating PDSI instead uses the spell-selection rules described
by Wells, Goddard, and Hayes (2004), so the two state machines intentionally
remain isolated.
"""

from dataclasses import dataclass

import numpy as np

from climate_indices.exceptions import ConvergenceError

_TOLERANCE = 1e-5
_BUG = 0
_SPELL_THRESHOLD = 0.5


@dataclass(frozen=True)
class WellsResult:
    """Arrays produced by one complete Wells recursion pass."""

    pdsi: np.ndarray
    phdi: np.ndarray
    pmdi: np.ndarray
    x1: np.ndarray
    x2: np.ndarray
    x3: np.ndarray
    probability: np.ndarray


@dataclass
class _State:
    """Mutable scalar state carried between non-missing periods."""

    x1: float = 0.0
    x2: float = 0.0
    x3: float = 0.0
    v: float = 0.0
    probability: float = 0.0


@dataclass(frozen=True)
class _Factors:
    """Validated duration factors and their derived recurrence constants."""

    wetm: float
    wetb: float
    drym: float
    dryb: float
    wet_denominator: float
    dry_denominator: float
    wetc: float
    dryc: float
    dry_spell_c: float


@dataclass(frozen=True)
class _Tentative:
    """Candidate values retained until a spell path becomes unambiguous."""

    index: int
    x1: float
    x2: float
    x3: float


@dataclass(frozen=True)
class _Transition:
    """One period's candidate state before it is persisted."""

    x1: float
    x2: float
    x3: float = 0.0
    v: float = 0.0
    probability: float = 0.0
    selection: int = 0
    spell_terminated: bool = False


def _is_exact_zero(value: float) -> bool:
    """Preserve Wells' intentional exact-zero branching semantics."""
    # Exact zero selects and backtracks Wells candidates.
    return value == 0.0  # NOSONAR


def _validated_factors(wetm: float, wetb: float, drym: float, dryb: float) -> _Factors:
    """Validate fitted duration factors and derive the Wells recurrence coefficients.

    ``dry_coefficient_denominator`` (``drym + wetb``) may be negative, pushing
    ``|dryc| > 1``; that is bounded because ``dryc`` only feeds the clamped x2
    recurrence (``min(0.0, ...)``) and spell-establishment resets, unlike
    ``wetc``/``dry_spell_c`` below, which feed the unclamped x3 recurrence and
    must be contractions. The magnitude check below still rejects ``|dryc| >=
    1`` defensively, for uniformity with ``wetc``/``dry_spell_c``.
    """
    wet_denominator = wetm + wetb
    dry_denominator = drym + dryb
    dry_coefficient_denominator = drym + wetb
    if (
        not np.isfinite(wet_denominator)
        or wet_denominator <= 0.0
        or not np.isfinite(dry_denominator)
        or dry_denominator <= 0.0
        or not np.isfinite(dry_coefficient_denominator)
        or _is_exact_zero(dry_coefficient_denominator)
    ):
        raise ConvergenceError(
            "invalid fitted duration factors for the Wells recursion",
            algorithm="scPDSI duration-factor calibration",
        )

    wetc = 1.0 - wetm / wet_denominator
    # Wells' published implementation intentionally uses the wet intercept in
    # this coefficient, while the dry Z contribution below uses drym + dryb.
    # dry_coefficient_denominator (drym + wetb) may be negative; see the
    # function docstring above for why that stays bounded rather than being
    # rejected outright.
    dryc = 1.0 - drym / dry_coefficient_denominator
    dry_spell_c = 1.0 - drym / dry_denominator
    # wetc and dry_spell_c drive the unclamped x3 recurrence and must be
    # contractions; dryc drives the clamped x2 recurrence but is checked too,
    # for uniformity. Real-data calibration keeps all three well under 1.0
    # (max observed |wetc|=0.983, |dryc|=0.981, |dry_spell_c|=0.982 across
    # 344 nClimDiv divisions).
    for name, value in (("wetc", wetc), ("dryc", dryc), ("dry_spell_c", dry_spell_c)):
        if not np.isfinite(value) or abs(value) >= 1.0:
            raise ConvergenceError(
                f"invalid fitted duration factors for the Wells recursion: {name} = {value!r} is non-finite or has magnitude >= 1",
                algorithm="scPDSI duration-factor calibration",
            )
    return _Factors(
        wetm=wetm,
        wetb=wetb,
        drym=drym,
        dryb=dryb,
        wet_denominator=wet_denominator,
        dry_denominator=dry_denominator,
        wetc=wetc,
        dryc=dryc,
        dry_spell_c=dry_spell_c,
    )


def _candidate_values(state: _State, z: float, factors: _Factors) -> tuple[float, float]:
    x1 = max(0.0, factors.wetc * state.x1 + z / factors.wet_denominator)
    x2 = min(0.0, factors.dryc * state.x2 + z / factors.dry_denominator)
    return x1, x2


def _backtrack(pdsi: np.ndarray, tentative: list[_Tentative], selection: int) -> None:
    """Resolve tentative periods using Wells' alternating zero-candidate rule."""
    if selection == 3:
        for values in tentative:
            pdsi[values.index] = values.x3
        tentative.clear()
        return

    for values in reversed(tentative):
        if selection == 2:
            if _is_exact_zero(values.x2):
                selection = 1
                pdsi[values.index] = values.x1
            else:
                pdsi[values.index] = values.x2
        else:
            if _is_exact_zero(values.x1):
                selection = 2
                pdsi[values.index] = values.x2
            else:
                pdsi[values.index] = values.x1
    tentative.clear()


def _pmdi(probability: float, x1: float, x2: float, x3: float) -> float:
    if _is_exact_zero(x3):
        if abs(x1) > abs(x2):
            return x1
        return x2
    if probability <= 0.0 or probability >= 100.0:
        return x3

    fraction = probability / 100.0
    if x3 <= 0.0:
        return (1.0 - fraction) * x3 + fraction * x1
    return (1.0 - fraction) * x3 + fraction * x2


def _abatement_transition(
    state: _State,
    z_value: float,
    factors: _Factors,
    transition: _Transition,
    *,
    wet_spell: bool,
    coefficient: float,
    denominator: float,
    slope: float,
    direction: float,
    abatement_z: float,
) -> _Transition:
    """Advance an established spell while abatement is possible."""
    tolerance_adjustment = (1 - _BUG) * _TOLERANCE
    if wet_spell:
        carry = min(state.v + tolerance_adjustment, 0.0)
    else:
        carry = max(state.v - tolerance_adjustment, 0.0)
    new_v = z_value - direction * abatement_z + carry

    if direction * new_v >= 0.0:
        return _Transition(
            x1=0.0,
            x2=0.0,
            x3=coefficient * state.x3 + z_value / denominator,
            selection=3,
        )

    own_intercept = factors.wetb if wet_spell else factors.dryb
    ze = direction * _SPELL_THRESHOLD * (slope + own_intercept) - own_intercept * state.x3
    q = ze if state.probability >= 100.0 - _TOLERANCE else ze + state.v
    if _is_exact_zero(q) or not np.isfinite(q):
        raise ConvergenceError(
            "Wells recursion could not calculate an abatement probability",
            algorithm="scPDSI Wells recursion",
        )

    new_probability = (new_v / q) * 100.0
    if new_probability >= 100.0 - _TOLERANCE:
        return _Transition(
            x1=transition.x1,
            x2=transition.x2,
            v=new_v,
            probability=100.0,
            spell_terminated=True,
        )

    return _Transition(
        x1=transition.x1,
        x2=transition.x2,
        x3=coefficient * state.x3 + z_value / denominator,
        v=new_v,
        probability=new_probability,
    )


def _continue_spell(state: _State, z_value: float, factors: _Factors, transition: _Transition) -> _Transition:
    """Continue or abate an established wet or dry spell."""
    if _is_exact_zero(state.x3):
        return transition

    wet_spell = state.x3 >= 0.0
    coefficient = factors.wetc if wet_spell else factors.dry_spell_c
    denominator = factors.wet_denominator if wet_spell else factors.dry_denominator
    slope = factors.wetm if wet_spell else factors.drym
    direction = 1.0 if wet_spell else -1.0
    # Generalizing Palmer's recurrence replaces the fixed 0.15
    # effective-moisture threshold with half the fitted slope.
    abatement_z = slope * _SPELL_THRESHOLD
    abatement_underway = state.probability not in (0.0, 100.0)

    if not abatement_underway and direction * z_value >= abatement_z:
        return _Transition(
            x1=0.0,
            x2=0.0,
            x3=coefficient * state.x3 + z_value / denominator,
            selection=3,
        )

    return _abatement_transition(
        state,
        z_value,
        factors,
        transition,
        wet_spell=wet_spell,
        coefficient=coefficient,
        denominator=denominator,
        slope=slope,
        direction=direction,
        abatement_z=abatement_z,
    )


def _establish_spell(transition: _Transition) -> _Transition:
    """Apply Wells' candidate selection and spell-establishment tie-breaks."""
    if not _is_exact_zero(transition.x3):
        return transition

    if transition.x1 >= _SPELL_THRESHOLD:
        return _Transition(
            x1=0.0,
            x2=transition.x2,
            x3=transition.x1,
            v=transition.v if transition.spell_terminated else 0.0,
            probability=transition.probability if transition.spell_terminated else 0.0,
            selection=1,
            spell_terminated=transition.spell_terminated,
        )
    if transition.x2 <= -_SPELL_THRESHOLD:
        return _Transition(
            x1=transition.x1,
            x2=0.0,
            x3=transition.x2,
            v=transition.v if transition.spell_terminated else 0.0,
            probability=transition.probability if transition.spell_terminated else 0.0,
            selection=2,
            spell_terminated=transition.spell_terminated,
        )
    if _is_exact_zero(transition.x1):
        return _Transition(
            x1=transition.x1,
            x2=transition.x2,
            v=transition.v,
            probability=transition.probability,
            selection=2,
            spell_terminated=transition.spell_terminated,
        )
    if _is_exact_zero(transition.x2):
        return _Transition(
            x1=transition.x1,
            x2=transition.x2,
            v=transition.v,
            probability=transition.probability,
            selection=1,
            spell_terminated=transition.spell_terminated,
        )
    return transition


def _assign_period(
    pdsi: np.ndarray,
    tentative: list[_Tentative],
    index: int,
    transition: _Transition,
) -> None:
    """Assign or defer one period's final PDSI value."""
    if not transition.selection:
        pdsi[index] = transition.x3
        tentative.append(
            _Tentative(
                index=index,
                x1=transition.x1,
                x2=transition.x2,
                x3=transition.x3,
            )
        )
        return

    _backtrack(pdsi, tentative, transition.selection)
    if transition.selection == 1:
        pdsi[index] = transition.x3 if not _is_exact_zero(transition.x3) else transition.x1
    elif transition.selection == 2:
        pdsi[index] = transition.x3 if not _is_exact_zero(transition.x3) else transition.x2
    else:
        pdsi[index] = transition.x3


def calculate(
    z_values: np.ndarray,
    *,
    wetm: float,
    wetb: float,
    drym: float,
    dryb: float,
) -> WellsResult:
    """Run the Wells Palmer recursion over a Z-index series.

    Missing periods are emitted as missing in every result and do not advance
    the recursion state.

    Args:
        z_values: Chronological Z-index values; NaN denotes a missing period.
        wetm: Wet duration-factor slope.
        wetb: Wet duration-factor intercept.
        drym: Dry duration-factor slope.
        dryb: Dry duration-factor intercept.

    Returns:
        The final PDSI-family series and the state arrays used to derive them.

    Raises:
        ConvergenceError: If fitted factors make a recurrence denominator
            non-finite, non-positive, or zero as applicable, or a Z-index
            value is infinite.
    """
    factors = _validated_factors(wetm, wetb, drym, dryb)
    z = np.asarray(z_values, dtype=float).reshape(-1)
    if np.any(np.isinf(z)):
        raise ConvergenceError(
            "Wells recursion received non-finite Z-index values",
            algorithm="scPDSI Wells recursion",
        )
    size = z.size
    pdsi = np.full(size, np.nan)
    x1_values = np.full(size, np.nan)
    x2_values = np.full(size, np.nan)
    x3_values = np.full(size, np.nan)
    probability_values = np.full(size, np.nan)

    state = _State()
    tentative: list[_Tentative] = []

    for index, z_value in enumerate(z):
        if np.isnan(z_value):
            continue

        z_value = float(z_value)
        new_x1, new_x2 = _candidate_values(state, z_value, factors)
        transition = _Transition(x1=new_x1, x2=new_x2)
        transition = _continue_spell(state, z_value, factors, transition)
        transition = _establish_spell(transition)
        _assign_period(pdsi, tentative, index, transition)

        state = _State(
            x1=transition.x1,
            x2=transition.x2,
            x3=transition.x3,
            v=transition.v,
            probability=transition.probability,
        )
        x1_values[index] = transition.x1
        x2_values[index] = transition.x2
        x3_values[index] = transition.x3
        probability_values[index] = transition.probability

    phdi = np.where(np.isnan(pdsi), np.nan, np.where(np.equal(x3_values, 0.0), pdsi, x3_values))
    pmdi = np.full(size, np.nan)
    for index in range(size):
        if not np.isnan(z[index]):
            pmdi[index] = _pmdi(
                probability_values[index],
                x1_values[index],
                x2_values[index],
                x3_values[index],
            )

    return WellsResult(
        pdsi=pdsi,
        phdi=phdi,
        pmdi=pmdi,
        x1=x1_values,
        x2=x2_values,
        x3=x3_values,
        probability=probability_values,
    )
