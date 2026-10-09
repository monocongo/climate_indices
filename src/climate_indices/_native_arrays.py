"""Array views and extension checks shared by the native dispatch modules.

``climate_indices.fire._native`` and ``climate_indices.flood._native`` hand the
Rust kernels (``docs/architecture.md``) time-first arrays as ``(days, cells)``
blocks and per-cell vectors. The helpers here build those as views, so the
binding's own copy before it releases the GIL is the only full-size copy of an
input whose cells merge into one axis. Each dispatch module decides what
happens to a layout that cannot be viewed that way: fire keeps it on the Python
path, and flood lets the reshape copy it. Each also keeps its own ``_native``
attribute, so tests can disable one family's kernels without the other's.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt

# The calendar and gap-count arrays a kernel takes: Python's ``intp`` (the fire
# day-length band) and the ``int64`` months and gap counts.
_Counts = npt.NDArray[np.int64] | npt.NDArray[np.intp]


def _with_kernels(native: ModuleType | None, *names: str) -> ModuleType | None:
    """The extension when it exposes every named attribute, else None.

    A stale installed extension can import successfully yet predate a kernel
    (``docs/architecture.md``). Dispatching into it would raise
    ``AttributeError`` instead of leaving the computation on its Python path,
    so a missing kernel declines the whole call.
    """
    if native is None or not all(hasattr(native, name) for name in names):
        return None
    return native


def _cells(shape: tuple[int, ...]) -> int:
    """The number of cells in a time-first array's trailing spatial shape."""
    return int(np.prod(shape, dtype=np.intp))


def _merges_cells(array: npt.NDArray[Any]) -> bool:
    """Whether a time-first array's spatial axes reshape to one cell axis without a copy.

    A broadcast or transposed view merges; a sliced or otherwise strided one
    does not, and reshaping it would add a full-size copy the caller's memory
    accounting never sees, so such a layout stays on the Python path.
    """
    axes = [(length, stride) for length, stride in zip(array.shape[1:], array.strides[1:], strict=True) if length != 1]
    return all(outer == length * inner for (_, outer), (length, inner) in zip(axes, axes[1:], strict=False))


def _block(array: npt.NDArray[np.float64], days: int, cells: int) -> npt.NDArray[np.float64]:
    """One time-first float64 input as the ``(days, cells)`` view a kernel takes."""
    return np.asarray(array, dtype=np.float64).reshape(days, cells)


def _flat(array: npt.NDArray[np.float64], cells: int) -> npt.NDArray[np.float64]:
    """One per-cell float64 array as the ``(cells,)`` vector a kernel takes."""
    return np.asarray(array, dtype=np.float64).reshape(cells)


def _mask(array: npt.NDArray[np.bool_], days: int, cells: int) -> npt.NDArray[np.bool_]:
    """One time-first validity mask as the ``(days, cells)`` view a kernel takes."""
    return np.asarray(array, dtype=np.bool_).reshape(days, cells)


def _flags_flat(array: npt.NDArray[np.bool_], cells: int) -> npt.NDArray[np.bool_]:
    """One per-cell boolean array as the ``(cells,)`` vector a kernel takes."""
    return np.asarray(array, dtype=np.bool_).reshape(cells)


def _counts_flat(array: _Counts, cells: int) -> npt.NDArray[np.int64]:
    """One per-cell calendar or gap-count array as the ``(cells,)`` vector a kernel takes."""
    return np.asarray(array, dtype=np.int64).reshape(cells)
