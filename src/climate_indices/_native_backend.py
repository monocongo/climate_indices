"""The one seam between the Python package and the optional Rust extension.

``climate_indices._native`` (``docs/architecture.md``, "Optional Rust Backend")
is imported here and nowhere else. Each family that dispatches to it copies
:data:`EXTENSION` into its own module attribute ``_native``: ``compute``,
``eto``, ``pm_eto``, ``palmer``, ``climate_indices.fire._native``,
``climate_indices.flood._native``, ``climate_indices.compound._bootstrap``, and
``climate_indices.runs``. That attribute is the family's switch.
Pointing it at None keeps the family on its Python implementation, and pointing
it at a recorder instruments it, without touching any other family; tests do
both through ``disable_native`` and ``NATIVE_FAMILIES`` in ``tests/conftest.py``.

This module holds the rules every family shares: whether the extension has the
kernels a call needs, whether the operands reach it as they are, and how the
extension's own errors become the ones the Python path raises. Each family keeps
its own eligibility rule and its marshalling. A runtime error from a kernel is
never caught and retried in Python (ADR-0017).
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

import numpy as np

from climate_indices._recurrence import _raise_non_finite
from climate_indices.exceptions import ConvergenceError

try:
    # without the extension every family runs its pure-Python implementation
    from climate_indices import _native as _extension
except ImportError:
    _extension = None  # type: ignore[assignment]

#: The optional Rust extension, or None when this install is pure Python. It is
#: typed as optional, so each family's fallback branch stays visible to the type
#: checker.
EXTENSION: ModuleType | None = _extension


def kernels(native: ModuleType | None, *names: str) -> ModuleType | None:
    """``native`` when it is loaded and exposes every named attribute, else None.

    A stale installed extension can import successfully yet predate a kernel.
    Dispatching into it would raise ``AttributeError`` instead of leaving the
    computation on its Python path, so a missing kernel declines the whole call.
    """
    if native is None or not all(hasattr(native, name) for name in names):
        return None
    return native


def plain_float64(array: np.ndarray) -> bool:
    """Whether an operand is a plain, aligned float64 ``ndarray``.

    A masked array or ``DataArray`` is not plain: the NumPy path reads its mask or
    labels, which a bare buffer would drop.
    """
    return (
        type(array) is np.ndarray
        and array.dtype == np.float64
        and array.flags.aligned
        and array.ctypes.data % array.dtype.alignment == 0
    )


def policy_allows() -> bool:
    """Whether Rust can preserve the caller's floating-point reporting policy."""
    return (
        all(policy == "ignore" for policy in np.geterr().values())
        and not getattr(sys.flags, "context_aware_warnings", False)
        and not any(
            action == "error" and issubclass(RuntimeWarning, category) for action, _, category, _, _ in warnings.filters
        )
    )


def takes(*arrays: np.ndarray) -> bool:
    """Whether the extension can take every operand as it is without changing reporting.

    The layouts are checked first, so the reporting policy is read once per call.
    """
    return all(plain_float64(array) for array in arrays) and policy_allows()


@contextmanager
def converging(native: ModuleType, algorithm: str) -> Iterator[None]:
    """Raise the extension's ``NoConvergenceError`` as the ``ConvergenceError`` the Python path raises."""
    try:
        yield
    except native.NoConvergenceError as error:
        raise ConvergenceError(str(error), algorithm=algorithm) from None


@contextmanager
def finite_results(native: ModuleType, index_type: str) -> Iterator[None]:
    """Raise the extension's ``NonFiniteResultError`` as the error the Python recurrence driver raises."""
    try:
        yield
    except native.NonFiniteResultError as error:
        _raise_non_finite(index_type, error)
