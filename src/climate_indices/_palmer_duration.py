"""Duration-factor contract shared by the standard PDSI and scPDSI recursions.

A duration-factor pair ``(m, b)`` fixes how much previously accumulated spell
severity each Palmer recursion carries forward, and which share of the
climatically expected moisture it expects back. The two lineages express that
fraction differently: the NCEI ``pdi.f`` lineage followed by
:mod:`climate_indices.palmer` divides it out (``c = b / (m + b)``), while the
Wells lineage followed by :mod:`climate_indices._palmer_wells` subtracts the
complement (``c = 1 - m / (m + b)``). The forms agree in exact arithmetic but
may differ in the last bit of the float, and both recursions branch on exact
comparisons against spell severity, so each lineage keeps its own expression
here rather than perturbing the other's state machine.

Palmer's (1965) fixed national duration factors come from the published ``p``
and ``q`` constants; self-calibrating PDSI replaces them with per-location
values fitted by :func:`climate_indices.self_calibration.duration_factors`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_indices.exceptions import ConvergenceError

# Palmer's (1965) fixed national duration-factor constants, published as p and q.
_PALMER_P = 0.897
_PALMER_Q = 1.0 / 3.0


@dataclass(frozen=True)
class DurationFactors:
    """Validated duration factors and the recurrence coefficients they imply."""

    wetm: float
    wetb: float
    drym: float
    dryb: float
    wet_denominator: float
    dry_denominator: float
    wetc: float
    dryc: float
    dry_spell_c: float

    @classmethod
    def from_defaults(cls) -> DurationFactors:
        """Palmer's (1965) fixed national duration factors, derived from p and q.

        Standard PDSI uses these directly; scPDSI replaces them with per-location
        fitted values.
        """
        m = (1.0 - _PALMER_P) / _PALMER_Q
        b = _PALMER_P / _PALMER_Q
        return cls.from_fitted(m, b, m, b)

    @classmethod
    def from_fitted(cls, wetm: float, wetb: float, drym: float, dryb: float) -> DurationFactors:
        """Validate fitted duration factors and derive the Wells recurrence coefficients.

        All three derived coefficients must be contractions (``|c| < 1``). ``wetc``
        and ``dry_spell_c`` feed the unclamped x3 recurrence, so that requirement is
        immediate. ``dryc`` feeds only the x2 recurrence, whose ``min(0.0, ...)``
        clamp in ``_palmer_wells._candidate_values`` is one-sided: it caps x2 at zero
        but does not bound its magnitude. While an opposite-sign spell is established
        (``x3 != 0``), ``_palmer_wells._establish_spell`` returns early without
        resetting x2, so across consecutive abatement periods x2 keeps recurring
        through ``dryc`` unbounded in magnitude before being captured into x3 when the
        spell abates. (The non-abating branches -- ``_continue_spell`` and the
        ``direction * new_v >= 0`` arm of ``_abatement_transition`` -- do reset x2 to
        zero, so the exposure is a run of abatement periods, not the whole spell.)
        ``dryc``'s check is therefore required, not defensive uniformity.

        Taken together with the denominator checks below, the magnitude checks imply
        ``wetm > 0`` and ``drym > 0`` (since ``wetc == wetb / wet_denominator`` and
        ``dry_spell_c == dryb / dry_denominator``), which in turn makes
        ``dry_coefficient_denominator = drym + wetb <= 0`` unreachable: a negative
        denominator would force ``|dryc| = |wetb / (drym + wetb)| >= 1``. The
        exact-zero check on it below is thus the only case it can still catch.
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
            # mirrors _palmer_wells._is_exact_zero, inlined because that module
            # imports this one; a negative cross denominator is not rejected here.
            or dry_coefficient_denominator == 0.0  # NOSONAR
        ):
            raise ConvergenceError(
                "invalid fitted duration factors for the Wells recursion",
                algorithm="scPDSI duration-factor calibration",
            )

        wetc = 1.0 - wetm / wet_denominator
        # Wells' published implementation intentionally uses the wet intercept in
        # this coefficient, while the dry Z contribution below uses drym + dryb.
        dryc = 1.0 - drym / dry_coefficient_denominator
        dry_spell_c = 1.0 - drym / dry_denominator
        # All three must be contractions -- see the function docstring above.
        # Real-data calibration keeps them well under 1.0 (max observed
        # |wetc|=0.98284, |dryc|=0.98087, |dry_spell_c|=0.98212 across 344 nClimDiv
        # divisions), i.e. only ~1.7% of margin against this threshold. That margin
        # is pinned by test_fitted_duration_factor_coefficients_stay_contractions in
        # tests/test_scpdsi.py, so erosion fails a test rather than surfacing here.
        for name, value in (("wetc", wetc), ("dryc", dryc), ("dry_spell_c", dry_spell_c)):
            if not np.isfinite(value) or abs(value) >= 1.0:
                raise ConvergenceError(
                    f"invalid fitted duration factors for the Wells recursion: {name} = {value!r} is non-finite or has magnitude >= 1",
                    algorithm="scPDSI duration-factor calibration",
                )
        return cls(
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

    @staticmethod
    def weighting_fraction(m: float, b: float) -> float:
        """The duration-factor weighting fraction ``c = b / (m + b)`` as the pdi.f lineage computes it.

        :param m: duration-factor slope
        :param b: duration-factor intercept
        :return the weighting fraction c = b / (m + b)
        :rtype: float
        :raises ValueError: if the duration factors sum to zero
        """
        denominator = m + b
        if denominator == 0:
            raise ValueError("duration-factor slope and intercept must not sum to zero")
        return b / denominator
