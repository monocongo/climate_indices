"""NaN-ordering helpers shared by the pdi.f Palmer code.

Both the water-balance stage in :mod:`climate_indices.palmer` and the pdi.f spell
recursion in :mod:`climate_indices._palmer_pdi` call Python's builtin ``max``/
``min`` with data-dependent NaN possible in either operand.  Replicating the
builtin's comparison order exactly is required for bit-for-bit equivalence with
the per-location reference, so the helpers live in one leaf module both import
rather than in either caller.
"""

import numpy as np


def _py_max(a: float | np.ndarray, b: float | np.ndarray) -> np.ndarray:
    """``max(a, b)`` matching Python's builtin comparison order, not ``np.maximum``.

    Python's ``max(a, b)`` returns ``b`` only if ``b > a``, so a NaN in ``b`` never
    wins while a NaN in ``a`` always loses unless ``b`` also fails to compare
    greater -- an asymmetry ``np.maximum`` does not have.
    """
    return np.where(np.asarray(b) > a, b, a)


def _py_min(a: float | np.ndarray, b: float | np.ndarray) -> np.ndarray:
    """``min(a, b)`` matching Python's builtin comparison order; see :func:`_py_max`."""
    return np.where(np.asarray(b) < a, b, a)
