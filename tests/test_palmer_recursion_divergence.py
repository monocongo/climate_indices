"""Pin both Palmer recursions on the shortest Z-Index sequences that make them disagree (REFIMPL-29, #1631).

Each case is a minimal reproducing sequence for one divergence class, found by
``scripts/find_palmer_minimal_sequences.py`` and described in
``docs/research/reference-papers/palmer-recursion-divergence.md``.  The class name is the set
of code-level differences (E establishment threshold, T termination at |X3| <= 0.5, A abatement
trigger, C candidate carried at establishment, F end-of-record flush) that must all be
harmonized for the disagreement to disappear.

The tests characterize behavior: they pin what ``_palmer_pdi.calculate`` (NCEI ``pdi.f``
lineage) and ``_palmer_wells.calculate`` (Wells lineage) return on the same Z-Index series at
Palmer's national duration factors.  They do not say which recursion is right, and they must be
updated, not relaxed, if either recursion is deliberately changed.  Sequences are left-padded
with zeros to whole years, which both recursions treat as no-ops.
"""

import numpy as np
import pytest

from climate_indices.drought import _palmer_pdi, _palmer_wells
from climate_indices.drought._palmer_duration import DurationFactors
from climate_indices.drought._palmer_pdi import PdiDurationFactors

# class -> (active Z sequence, {month index in the padded series: (pdi.f PDSI, Wells PDSI)})
_CASES = {
    "A": (
        [2.0, 2.0, -1.0, 2.0, 0.15, -2.0, 3.0],
        {9: (1.292550277, 0.05)},
    ),
    "A+C": (
        [
            -2.0,
            -2.7,
            -2.8,
            -3.0,
            -4.0,
            -3.0,
            -0.6,
            2.0,
            0.0,
            3.1,
            1.0,
            1.0,
            -2.5,
            -0.2,
            2.0,
            -3.0,
            -1.0,
            -0.15,
            6.0,
            -1.0,
        ],
        {19: (-1.0, -1.057085828), 20: (-1.230333333, -1.281539321), 21: (-1.153609, -0.05)},
    ),
    "C": (
        [-2.0, -4.0, -2.0, -3.0, 1.4, 2.4, 1.0, -0.2, -1.6, -2.0, -2.0, 2.0, 2.2, -3.0, 1.0],
        {22: (-1.0, -1.062988044), 23: (-0.563666667, -0.620166942)},
    ),
    "E": (
        [-1.0, -1.0, -0.0, 3.0, 1.0],
        {9: (-0.567203, 0.0)},
    ),
    "E+A": (
        [2.2459814106903577, -0.3607725973830373],
        {11: (0.0, -0.120257532)},
    ),
    "E+A+C": (
        [
            -1.3,
            -0.8,
            -1.7,
            -5.0,
            -1.0,
            -1.43,
            0.7,
            -1.35,
            -2.3,
            -1.4,
            -2.0,
            -0.55,
            -0.9,
            -0.43,
            2.73,
            -2.43,
            -1.45,
            1.08,
            1.7,
            -1.84,
            -0.31,
            -0.9544272116171671,
            1.25,
            1.59,
            -1.14,
            -1.76,
            -0.9142391457331976,
            0.81,
            1.4,
            -1.0,
            1.0,
        ],
        {
            22: (-2.10792987, 0.36),
            23: (-1.324146426, 0.889586667),
            24: (-1.801092678, -0.613333333),
            25: (-1.718913465, -0.653493333),
            26: (-1.860007782, -0.904325924),
            27: (-1.251760314, 0.416666667),
            28: (-0.592829002, 0.90375),
            29: (-0.911767614, -0.38),
            30: (-1.404522217, -0.927526667),
            31: (-1.56460281, -1.136737802),
            32: (-1.133448721, 0.27),
            33: (-0.550036836, 0.708856667),
            34: (-0.826716375, -0.517911534),
            35: (0.0, -0.131233313),
        },
    ),
    "E+C": (
        [-1.0, -1.0, -3.0, -2.0, -1.0, -1.0, 2.5, 1.5, -1.6, -1.0, -2.0, 1.0, -1.0, 3.0, -2.0, 0.2],
        {22: (-0.666666667, -0.700273494), 23: (0.0, -0.561478657)},
    ),
    "E+F": (
        [-2.0, -0.5, 0.0, 1.0],
        {10: (-0.685906, 0.0), 11: (0.0, 0.333333333)},
    ),
    "E+T": (
        [1.5, -2.0, -0.3, 2.0],
        {9: (-0.666666667, -0.218166667), 10: (-0.698, -0.2956955), 11: (0.666666667, 0.401427803)},
    ),
    "E+T+A": (
        [-0.5, 0.3, -1.4, 2.29, -0.34, -1.2, -0.76, -0.36, 0.34, -0.645071353700617, 2.0, 0.0, -1.0, -1.0],
        {
            13: (0.763333333, 0.304905188),
            14: (-0.113333333, 0.16016662),
            15: (-0.50166, -0.256330542),
            16: (-0.703322353, -0.483261829),
            17: (-0.750880151, -0.553485861),
            18: (-0.560206162, 0.113333333),
            19: (-0.717528712, -0.215023785),
            21: (0.598, 0.0),
            22: (0.0, -0.333333333),
        },
    ),
    "E+T+C": (
        [
            -1.0,
            -2.0,
            -2.0,
            -4.0,
            -2.0,
            4.0,
            1.0,
            0.6,
            -1.0,
            -0.6,
            0.3,
            -1.7,
            0.5,
            0.8,
            0.6,
            -1.1,
            0.2,
            -0.7,
            2.0,
            0.4,
            0.0,
        ],
        {
            9: (-0.961758174, 1.529333333),
            10: (-0.662697082, 1.571812),
            11: (-0.927772616, 1.076582031),
            12: (-1.032212036, 0.765694082),
            13: (-0.825894197, 0.786827591),
            14: (-1.307493761, 0.139117683),
            15: (-1.006155237, 0.291455228),
            16: (-0.635854581, 0.528102006),
            17: (0.0, 0.673707499),
            18: (0.0, -0.421130643),
            19: (0.0, -0.31108752),
            20: (0.0, -0.512378839),
            21: (0.682526666, 0.207062848),
            22: (0.745559753, 0.319068708),
            23: (0.668767098, 0.286204631),
        },
    ),
    "F": (
        [-2.0, -1.0, -1.3, -2.0, 3.0, 3.0],
        {10: (-0.618839086, 1.0)},
    ),
}


def _run(active: list[float]) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate([np.zeros(-len(active) % 12), np.asarray(active, dtype=float)])
    wells_factors = DurationFactors.from_defaults()
    pdi_factors = PdiDurationFactors.from_fitted(
        wells_factors.wetm, wells_factors.wetb, wells_factors.drym, wells_factors.dryb
    )
    pdi = _palmer_pdi.calculate(padded.reshape(-1, 12, 1), pdi_factors).pdsi.reshape(-1)
    wells = _palmer_wells.calculate(padded, factors=wells_factors).pdsi
    return pdi, wells


@pytest.mark.parametrize("label", sorted(_CASES))
def test_minimal_sequence_pins_both_recursions(label: str) -> None:
    active, expected = _CASES[label]
    pdi, wells = _run(active)

    differing = np.flatnonzero(np.abs(pdi - wells) > 1e-9)
    assert differing.tolist() == sorted(expected)
    for index, (pdi_value, wells_value) in expected.items():
        assert pdi[index] == pytest.approx(pdi_value, abs=1e-8)
        assert wells[index] == pytest.approx(wells_value, abs=1e-8)
