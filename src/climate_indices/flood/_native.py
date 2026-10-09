"""The optional Rust kernels behind the flood family.

``crates/climate-core`` computes effective precipitation, the EDI and Flood
Index standardizations, and the Antecedent Precipitation Index recurrence
(``docs/architecture.md``). This module is the flood package's only native
boundary: it decides whether the extension can take an entry point's prepared
arrays, reshapes them into the time-first ``(time, cells)`` blocks the kernels
read, and builds the callable :mod:`climate_indices._recurrence` runs in place
of its day loop for the API.

Nothing here holds an algorithm. The Python implementations in ``_pe.py``,
``_edi.py``, ``_if.py``, and ``_antecedent.py`` stay the parity oracles, and
every input this module cannot hand to the extension unchanged returns None so
its caller stays on the Python path. Validation, calibration-period resolution,
and the all-leap layout stay in those modules.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import numpy.typing as npt

from climate_indices import compute
from climate_indices._native_arrays import _block, _cells, _counts_flat, _flags_flat, _flat, _mask, _with_kernels
from climate_indices._recurrence import _MAX_NATIVE_OPTION, NativeRecurrence, _raise_non_finite

try:
    # the optional Rust kernels (docs/architecture.md); without the extension the
    # flood indices run the Python implementations they always have
    from climate_indices import _native
except ImportError:
    _native = None  # type: ignore[assignment]

_API_INDEX_TYPE = "antecedent_precipitation_index"


def _native_module() -> ModuleType | None:
    """The optional Rust extension, or None when this install is pure Python.

    Every guard reads the module through this accessor: a successful import is
    typed as always present, which would make the fallback branches look
    unreachable to the type checker.
    """
    return _native


def _kernel_module(*names: str) -> ModuleType | None:
    """The extension when it exposes every named attribute, else None (see :func:`_with_kernels`)."""
    return _with_kernels(_native_module(), *names)


def _time_first_block(array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | None:
    """A ``(time, *cells)`` array as a ``(time, cells)`` block, or None when the kernels cannot take it.

    The block is a view when the cells merge into one axis, and otherwise the
    reshape copies it; only the fire recurrences decline such a layout. The
    public entry points validate their input with ``np.ma.asarray``, which
    already makes it C-contiguous, so from those the block is always a view.
    """
    if not compute._native_float64(array):
        return None
    rows = array.shape[0]
    columns = _cells(array.shape[1:])
    if rows == 0 or columns == 0:
        return None
    return _block(array, rows, columns)


def effective_precipitation(series: npt.NDArray[np.float64], duration: int) -> npt.NDArray[np.float64] | None:
    """The Rust effective precipitation of a time-first series, or None to stay on the Python path.

    The kernel reproduces ``correlate1d``'s window, its operation order, and the
    leading ``duration - 1`` NaN days, so the result replaces the Python block
    whole.
    """
    native = _kernel_module("effective_precipitation")
    block = None if native is None else _time_first_block(series)
    if native is None or block is None:
        return None
    result: npt.NDArray[np.float64] = native.effective_precipitation(block, int(duration))
    return result.reshape(series.shape)


def edi(years: npt.NDArray[np.float64], calibration_start: int, calibration_end: int) -> npt.NDArray[np.float64] | None:
    """The Rust EDI of a ``(years, 366, *cells)`` block, or None to stay on the Python path.

    ``calibration_start`` and ``calibration_end`` are the Calibration Period's
    year rows, end exclusive, as ``_edi.py`` resolved them.
    """
    native = _kernel_module("edi")
    block = None if native is None else _time_first_block(years)
    if native is None or block is None:
        return None
    result: npt.NDArray[np.float64] = native.edi(block, int(calibration_start), int(calibration_end))
    return result.reshape(years.shape)


def flood_index(
    series: npt.NDArray[np.float64], first_start: int, calibration_years: int
) -> npt.NDArray[np.float64] | None:
    """The Rust Flood Index of a time-first PE series, or None to stay on the Python path.

    ``first_start`` is the first day of the first calibration annual period and
    ``calibration_years`` the number of consecutive periods, as ``_if.py``
    resolved them. The kernel's calibration sums follow NumPy's axis-0 order for
    the same ``(years, cells)`` sample, which depends on the cell count.
    """
    native = _kernel_module("flood_index")
    block = None if native is None else _time_first_block(series)
    if native is None or block is None:
        return None
    result: npt.NDArray[np.float64] = native.flood_index(block, int(first_start), int(calibration_years))
    return result.reshape(series.shape)


def api_recurrence(
    precipitation: npt.NDArray[np.float64],
    k: float,
    api: npt.NDArray[np.float64],
    weather_valid: npt.NDArray[np.bool_],
    static_valid: npt.NDArray[np.bool_],
    trailing_gap_days: npt.NDArray[np.int64],
) -> NativeRecurrence | None:
    """The Rust API kernel, or None to stay on the Python path.

    The kernel takes the recurrence's prepared, time-first arrays and runs the
    whole time axis per cell, including the ADR-0007 gap bookkeeping, through
    the same ``recurrence::run`` driver the fire codes use. A decay constant
    whose type would promote the Python step beyond float64 (``np.longdouble``)
    stays on the Python path, as does any array the extension cannot take.
    """
    native = _kernel_module("antecedent_precipitation_index", "NonFiniteResultError")
    if native is None or np.result_type(k, np.float64) != np.float64:
        return None
    if not all(compute._native_float64(array) for array in (precipitation, api)):
        return None
    days = precipitation.shape[0]
    cells = _cells(precipitation.shape[1:])
    if days == 0 or cells == 0:
        return None

    decay = float(k)

    def native_run(
        shape: tuple[int, ...] | None,
        spin_up: int,
        nan_policy: str,
        max_gap_days: int,
    ) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.int64] | None] | None:
        # public validation accepts any non-negative integer; one the binding
        # cannot represent stays on the Python path instead of overflowing
        if spin_up > _MAX_NATIVE_OPTION or max_gap_days > _MAX_NATIVE_OPTION:
            return None
        # built here, inside the runner's guarded region, so a copy a layout
        # needs is reported through the recurrence's failure events
        try:
            history, final_api, final_gaps = native.antecedent_precipitation_index(
                precipitation_mm=_block(precipitation, days, cells),
                k=decay,
                initial_api=_flat(api, cells),
                weather_valid=_mask(weather_valid, days, cells),
                static_valid=_flags_flat(static_valid, cells),
                trailing_gap_days=_counts_flat(trailing_gap_days, cells),
                spin_up=spin_up,
                nan_policy=nan_policy,
                max_gap_days=max_gap_days,
                record=shape is not None,
            )
        except native.NonFiniteResultError as exc:
            _raise_non_finite(_API_INDEX_TYPE, exc)

        # the kernel returns whole arrays; the driver and the returned APIState
        # read the component's own state and gap arrays, so write them back there
        api[...] = final_api.reshape(api.shape)
        if final_gaps is not None:
            final_gaps = final_gaps.reshape(trailing_gap_days.shape)
            trailing_gap_days[...] = final_gaps
            final_gaps = final_gaps.copy()
        # the kernel built the one history; return it in the public shape instead
        # of copying it into a second full-size array
        recorded = None if history is None or shape is None else history.reshape(shape)
        return recorded, final_gaps

    return native_run
