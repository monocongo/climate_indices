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
        [-1.0, -2.2, -0.15, 6.0, 3.0],
        {9: (-0.976003, -0.05)},
    ),
    "A+C": (
        [
            -1.7,
            -2.7,
            -3.0,
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
            4.0,
            3.0,
            -1.760804724971848,
            -0.5618104471028332,
            0.43,
            -2.58,
            -2.5934196789941963,
            -0.34,
            -0.2912800684270723,
            -2.6709524502381403,
            -2.9,
            4.02790594538576,
            -3.45,
            -0.48,
            -0.9809322896576268,
            -0.6024294186238482,
            0.6455441156158037,
            -2.8036402656626023,
            -0.28,
            1.71,
            -1.6833150012862512,
            1.44,
            1.75,
            -0.2665657954567833,
            -1.38,
            1.25,
            -2.0,
            -1.0,
            -2.4,
            2.6,
        ],
        {
            25: (-1.70744784, 0.57),
            26: (-2.092685713, -0.561105),
            27: (-1.397139085, 0.48),
            28: (-0.669900426, 1.013893333),
            29: (-0.689755947, 0.820607055),
            30: (-1.078711084, 0.276084528),
            31: (-0.550937176, 0.664314488),
            32: (-1.160857314, -0.727166697),
            33: (-1.374622344, -0.98560186),
            34: (-2.033036242, -1.684084869),
            35: (-0.956966843, -0.643957461),
        },
    ),
    "E+C": (
        [-1.0, -1.0, -3.0, -2.0, -1.0, -1.0, 2.5, 1.5, -1.6, -1.0, -2.0, 1.0, -1.0, 2.9, -0.4, 0.6, -2.3, 0.0],
        {20: (0.733766667, -0.196840161), 21: (0.8581887, 0.2), 22: (0.0, -0.766666667), 23: (0.0, -0.6877)},
    ),
    "E+F": (
        [-2.0, -0.5, 0.0, 1.0],
        {10: (-0.685906, 0.0), 11: (0.0, 0.333333333)},
    ),
    "E+T": (
        [-1.65, 2.0, 1.0, 3.0],
        {9: (0.666666667, 0.173316667), 10: (0.931333333, 0.488798383), 11: (1.835406, 1.43845215)},
    ),
    "E+T+A": (
        [-0.6, -1.0, 1.7, 1.0, 2.3, -1.44, 1.74, -0.5, -3.0, 1.0],
        {
            4: (0.566666667, 0.106744867),
            5: (0.841633333, 0.429083479),
            6: (1.521611767, 1.151554547),
            7: (0.884885755, -0.48),
            8: (1.373742522, 0.58),
        },
    ),
    "E+T+C": (
        [
            -1.0,
            -2.0,
            -1.4,
            -1.1,
            -1.3,
            1.3,
            -2.0,
            0.0,
            -2.0,
            -0.0,
            3.0,
            -0.52,
            -0.8,
            0.8,
            0.0762726169170272,
            -1.3,
            1.4,
            1.0,
            2.0,
        ],
        {
            15: (-0.672041853, 1.0),
            16: (-0.776154875, 0.723666667),
            17: (-0.96287759, 0.382462333),
            18: (-0.597034531, 0.60973538),
            19: (0.0, 0.572356841),
            20: (0.0, -0.500643138),
            21: (0.538490132, 0.017589771),
            22: (0.816358982, 0.349111358),
            23: (1.398940674, 0.979819555),
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
