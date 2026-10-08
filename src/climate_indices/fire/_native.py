"""The optional Rust kernels behind the fire-weather recurrences.

``crates/climate-core`` runs the KBDI and CFFWIS moisture-code daily recursions
over a whole time axis per cell (``docs/architecture.md``). This module is the
fire package's only native boundary: it decides whether the extension can take
a recurrence's prepared arrays, builds the callable that
:mod:`climate_indices._recurrence` runs in place of its day loop, and translates
the extension's errors into the ones the Python driver raises.

Nothing here holds an algorithm. The Python implementations in ``_kbdi.py`` and
``_cffwis_codes.py`` stay the parity oracles, and every layout this module
cannot hand to the extension unchanged stays on the Python path. The kernels
take views of the time-first arrays: the binding's own copy before it releases
the GIL is the only full-size copy of an input, so a broadcast month series or
season mask is never materialized here first.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING, Any, NoReturn, cast

import numpy as np
import numpy.typing as npt

from climate_indices import compute
from climate_indices._recurrence import NativeRecurrence
from climate_indices.exceptions import InvalidArgumentError

if TYPE_CHECKING:
    from climate_indices.fire._cffwis_codes import _CodeInputs, _MoistureCode

try:
    # the optional Rust kernels (docs/architecture.md); without the extension the
    # fire recurrences run the Python steps they always have
    from climate_indices import _native
except ImportError:
    _native = None  # type: ignore[assignment]

# The widest recurrence option the bindings can represent: ``spin_up`` crosses
# as a ``usize`` and ``max_gap_days`` as an ``i64``, while public validation
# accepts any non-negative Python integer. A wider value stays on the Python
# path instead of failing argument conversion.
_MAX_NATIVE_OPTION = int(np.iinfo(np.int64).max)

# The calendar and gap-count arrays a kernel takes: Python's ``intp`` (the day
# length band) and the ``int64`` months and gap counts.
_Counts = npt.NDArray[np.int64] | npt.NDArray[np.intp]


def _native_module() -> ModuleType | None:
    """The optional Rust extension, or None when this install is pure Python.

    Every guard reads the module through this accessor: a successful import is
    typed as always present, which would make the fallback branches look
    unreachable to the type checker.
    """
    return _native


def _kernel_module(*names: str) -> ModuleType | None:
    """The extension when it exposes every named kernel, else None.

    A stale installed extension can import successfully yet predate a kernel
    (``docs/architecture.md``). Dispatching into it would raise
    ``AttributeError`` instead of leaving the recurrence on its Python steps, so
    a missing kernel or the shared ``NonFiniteResultError`` declines the whole
    call.
    """
    native = _native_module()
    if native is None or not all(hasattr(native, name) for name in (*names, "NonFiniteResultError")):
        return None
    return native


def _raise_non_finite(index_type: str, underlying: Exception) -> NoReturn:
    """Raise the error the Python driver raises for a non-finite step result.

    ``_advance_component`` rejects a daily update that is not finite although
    its inputs were, and the kernel reports the same condition; raising the
    original error type from here keeps the two paths indistinguishable.
    """
    raise InvalidArgumentError(
        f"{index_type} produced a non-finite value from finite inputs.",
        argument_name=index_type,
        argument_value="non-finite result",
        valid_values="Finite inputs whose result stays within float64",
    ) from underlying


def _cells(shape: tuple[int, ...]) -> int:
    """The number of cells in a time-first array's trailing spatial shape."""
    return int(np.prod(shape, dtype=np.intp))


def _merges_cells(array: npt.NDArray[Any]) -> bool:
    """Whether a time-first array's spatial axes reshape to one cell axis without a copy.

    A broadcast or transposed view merges; a sliced or otherwise strided one
    does not, and reshaping it would add a full-size copy the recurrence's
    memory metrics never see, so such a layout stays on the Python path.
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


def _optional_mask(array: npt.NDArray[np.bool_] | None, days: int, cells: int) -> npt.NDArray[np.bool_] | None:
    """An optional ``(days, cells)`` mask, or None when the recurrence has none."""
    return None if array is None else _mask(array, days, cells)


def _counts(array: _Counts, days: int, cells: int) -> npt.NDArray[np.int64]:
    """One time-first calendar array as the ``(days, cells)`` view a kernel takes."""
    return np.asarray(array, dtype=np.int64).reshape(days, cells)


def _counts_flat(array: _Counts, cells: int) -> npt.NDArray[np.int64]:
    """One per-cell calendar or gap-count array as the ``(cells,)`` vector a kernel takes."""
    return np.asarray(array, dtype=np.int64).reshape(cells)


def _flags_flat(array: npt.NDArray[np.bool_], cells: int) -> npt.NDArray[np.bool_]:
    """One per-cell boolean array as the ``(cells,)`` vector a kernel takes."""
    return np.asarray(array, dtype=np.bool_).reshape(cells)


def _recorded_history(
    history: npt.NDArray[np.float64] | None,
    values_out: npt.NDArray[np.float64] | None,
) -> npt.NDArray[np.float64] | None:
    """The kernel's recorded history in the public shape, or None when none was recorded.

    Returning the kernel's own array avoids a second full-size history: the
    caller's pre-allocated slot is replaced instead of filled by an extra copy.
    """
    if history is None or values_out is None:
        return None
    return history.reshape(values_out.shape)


def _final_gaps(
    gaps: npt.NDArray[np.int64] | None,
    shape: tuple[int, ...],
) -> npt.NDArray[np.int64] | None:
    """The kernel's trailing gap counts in the public shape, detached from the kernel's buffer."""
    return None if gaps is None else gaps.reshape(shape).copy()


@dataclass(frozen=True)
class _MoistureArrays:
    """One moisture code's inputs, held until the guarded kernel call builds its views."""

    index_type: str
    temperature: npt.NDArray[np.float64]
    precipitation: npt.NDArray[np.float64]
    state_value: npt.NDArray[np.float64]
    weather_valid: npt.NDArray[np.bool_]
    static_valid: npt.NDArray[np.bool_]
    trailing_gap_days: npt.NDArray[np.int64]
    days: int
    cells: int
    humidity: npt.NDArray[np.float64] | None = None
    wind: npt.NDArray[np.float64] | None = None
    months: _Counts | None = None
    band: _Counts | None = None
    table: npt.NDArray[np.float64] | None = None
    in_season: npt.NDArray[np.bool_] | None = None


@dataclass(frozen=True)
class _KbdiArrays:
    """KBDI's inputs, held until the guarded kernel call builds its views."""

    precipitation: npt.NDArray[np.float64]
    temperature: npt.NDArray[np.float64]
    mean_annual: npt.NDArray[np.float64]
    kbdi_value: npt.NDArray[np.float64]
    wet_spell: npt.NDArray[np.float64]
    weather_valid: npt.NDArray[np.bool_]
    static_valid: npt.NDArray[np.bool_]
    trailing_gap_days: npt.NDArray[np.int64]
    days: int
    cells: int


def _run_moisture_kernel(
    native: ModuleType,
    arrays: _MoistureArrays,
    values_out: npt.NDArray[np.float64] | None,
    spin_up: int,
    nan_policy: str,
    max_gap_days: int,
) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.int64] | None] | None:
    """Run one moisture code's whole time axis, or None to leave it on the Python path.

    The array conversions and the kernel call both happen here so they run
    inside the shared runner's guarded region: an allocation failure reports the
    recurrence lifecycle instead of escaping before it starts. An option wider
    than the bindings can represent also returns None, keeping the Python path.
    """
    if spin_up > _MAX_NATIVE_OPTION or max_gap_days > _MAX_NATIVE_OPTION:
        return None
    days = arrays.days
    cells = arrays.cells
    common: dict[str, Any] = {
        "temperature_celsius": _block(arrays.temperature, days, cells),
        "precipitation_mm": _block(arrays.precipitation, days, cells),
        "weather_valid": _mask(arrays.weather_valid, days, cells),
        "static_valid": _flags_flat(arrays.static_valid, cells),
        "in_season": _optional_mask(arrays.in_season, days, cells),
        "trailing_gap_days": _counts_flat(arrays.trailing_gap_days, cells),
        "spin_up": spin_up,
        "nan_policy": nan_policy,
        "max_gap_days": max_gap_days,
        "record": values_out is not None,
    }
    try:
        if arrays.index_type == "ffmc":
            history, final, final_gaps = native.ffmc(
                relative_humidity_percent=_block(cast(npt.NDArray[np.float64], arrays.humidity), days, cells),
                wind_speed_kilometers_per_hour=_block(cast(npt.NDArray[np.float64], arrays.wind), days, cells),
                initial_ffmc=_flat(arrays.state_value, cells),
                **common,
            )
        elif arrays.index_type == "duff_moisture_code":
            history, final, final_gaps = native.duff_moisture_code(
                relative_humidity_percent=_block(cast(npt.NDArray[np.float64], arrays.humidity), days, cells),
                day_length_table=arrays.table,
                months=_counts(cast(_Counts, arrays.months), days, cells),
                day_length_band=_counts_flat(cast(_Counts, arrays.band), cells),
                initial_dmc=_flat(arrays.state_value, cells),
                **common,
            )
        else:
            history, final, final_gaps = native.drought_code(
                day_length_table=arrays.table,
                months=_counts(cast(_Counts, arrays.months), days, cells),
                day_length_band=_counts_flat(cast(_Counts, arrays.band), cells),
                initial_dc=_flat(arrays.state_value, cells),
                **common,
            )
    except native.NonFiniteResultError as exc:  # pragma: no cover - defensive parity with the Python driver
        _raise_non_finite(arrays.index_type, exc)

    # the kernel returns whole arrays; the driver reads the component's own
    # state and gap arrays, so write the results back where it looks for them
    arrays.state_value[...] = final.reshape(arrays.state_value.shape)
    gap_shape = arrays.trailing_gap_days.shape
    if final_gaps is not None:
        arrays.trailing_gap_days[...] = final_gaps.reshape(gap_shape)
    return _recorded_history(history, values_out), _final_gaps(final_gaps, gap_shape)


def _run_kbdi_kernel(
    native: ModuleType,
    arrays: _KbdiArrays,
    values_out: npt.NDArray[np.float64] | None,
    spin_up: int,
    nan_policy: str,
    max_gap_days: int,
) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.int64] | None] | None:
    """Run KBDI's whole time axis, or None to leave it on the Python path.

    Shares the guarded-region conversion and option-domain policy of
    :func:`_run_moisture_kernel`.
    """
    if spin_up > _MAX_NATIVE_OPTION or max_gap_days > _MAX_NATIVE_OPTION:
        return None
    days = arrays.days
    cells = arrays.cells
    try:
        history, final_kbdi, final_wet_spell, final_gaps = native.kbdi(
            precipitation_mm=_block(arrays.precipitation, days, cells),
            maximum_temperature_celsius=_block(arrays.temperature, days, cells),
            mean_annual_precipitation_mm=_flat(arrays.mean_annual, cells),
            initial_kbdi=_flat(arrays.kbdi_value, cells),
            initial_wet_spell_precipitation=_flat(arrays.wet_spell, cells),
            weather_valid=_mask(arrays.weather_valid, days, cells),
            static_valid=_flags_flat(arrays.static_valid, cells),
            in_season=None,
            trailing_gap_days=_counts_flat(arrays.trailing_gap_days, cells),
            spin_up=spin_up,
            nan_policy=nan_policy,
            max_gap_days=max_gap_days,
            record=values_out is not None,
        )
    except native.NonFiniteResultError as exc:  # pragma: no cover - defensive parity with the Python driver
        _raise_non_finite("kbdi", exc)

    arrays.kbdi_value[...] = final_kbdi.reshape(arrays.kbdi_value.shape)
    arrays.wet_spell[...] = final_wet_spell.reshape(arrays.wet_spell.shape)
    gap_shape = arrays.trailing_gap_days.shape
    if final_gaps is not None:
        arrays.trailing_gap_days[...] = final_gaps.reshape(gap_shape)
    return _recorded_history(history, values_out), _final_gaps(final_gaps, gap_shape)


def moisture_code_recurrence(
    code: _MoistureCode,
    inputs: _CodeInputs,
    state_value: npt.NDArray[np.float64],
    weather_valid: npt.NDArray[np.bool_],
    static_valid: npt.NDArray[np.bool_],
    trailing_gap_days: npt.NDArray[np.int64],
    in_season: npt.NDArray[np.bool_] | None,
) -> NativeRecurrence | None:
    """The Rust kernel for one moisture code, or None to stay on the Python path.

    The kernel takes the code's prepared, time-first arrays and runs the whole
    time axis per cell, including the ADR-0007 gap bookkeeping and the ADR-0010
    seasonal carry. Anything the extension cannot take as it is — an array that
    is not a plain, aligned float64 ``ndarray``, a masked array, an empty axis,
    or a time-first array whose cells cannot be viewed as one axis — leaves the
    recurrence to the Python driver, which is the parity oracle. The prepared
    input views are built at call time, inside the shared runner's guarded
    region.
    """
    native = _kernel_module("ffmc", "duff_moisture_code", "drought_code")
    if native is None:
        return None
    temperature = inputs.temperature
    precipitation = inputs.precipitation
    float64_inputs = [state_value, temperature, precipitation]
    if inputs.humidity is not None:
        float64_inputs.append(inputs.humidity)
    if inputs.wind_kilometers_per_hour is not None:
        float64_inputs.append(inputs.wind_kilometers_per_hour)
    if not all(compute._native_float64(array) for array in float64_inputs):
        return None

    days = temperature.shape[0]
    cells = _cells(temperature.shape[1:])
    if days == 0 or cells == 0:
        return None
    months = inputs.months
    band = inputs.day_length_band
    table = code.day_length_table
    if code.index_type == "ffmc":
        if inputs.humidity is None or inputs.wind_kilometers_per_hour is None:
            return None
    elif months is None or band is None or table is None:
        return None
    blocks = (
        temperature,
        precipitation,
        inputs.humidity,
        inputs.wind_kilometers_per_hour,
        weather_valid,
        in_season,
        months,
    )
    if not all(_merges_cells(array) for array in blocks if array is not None):
        return None

    arrays = _MoistureArrays(
        index_type=code.index_type,
        temperature=temperature,
        precipitation=precipitation,
        state_value=state_value,
        weather_valid=weather_valid,
        static_valid=static_valid,
        trailing_gap_days=trailing_gap_days,
        days=days,
        cells=cells,
        humidity=inputs.humidity,
        wind=inputs.wind_kilometers_per_hour,
        months=months,
        band=band,
        table=table,
        in_season=in_season,
    )
    return partial(_run_moisture_kernel, native, arrays)


def kbdi_recurrence(
    precipitation_mm: npt.NDArray[np.float64],
    maximum_temperature_celsius: npt.NDArray[np.float64],
    mean_annual_precipitation_mm: npt.NDArray[np.float64],
    weather_valid: npt.NDArray[np.bool_],
    static_valid: npt.NDArray[np.bool_],
    kbdi_value: npt.NDArray[np.float64],
    wet_spell_precipitation: npt.NDArray[np.float64],
    trailing_gap_days: npt.NDArray[np.int64],
) -> NativeRecurrence | None:
    """The Rust KBDI kernel, or None to stay on the Python path.

    KBDI carries two per-cell state values (the index and its wet spell) and has
    no seasonal mask; everything else matches
    :func:`moisture_code_recurrence`.
    """
    native = _kernel_module("kbdi")
    if native is None:
        return None
    if not all(
        compute._native_float64(array)
        for array in (
            kbdi_value,
            wet_spell_precipitation,
            precipitation_mm,
            maximum_temperature_celsius,
            mean_annual_precipitation_mm,
        )
    ):
        return None

    days = precipitation_mm.shape[0]
    cells = _cells(precipitation_mm.shape[1:])
    if days == 0 or cells == 0:
        return None
    if not all(_merges_cells(array) for array in (precipitation_mm, maximum_temperature_celsius, weather_valid)):
        return None
    arrays = _KbdiArrays(
        precipitation=precipitation_mm,
        temperature=maximum_temperature_celsius,
        mean_annual=mean_annual_precipitation_mm,
        kbdi_value=kbdi_value,
        wet_spell=wet_spell_precipitation,
        weather_valid=weather_valid,
        static_valid=static_valid,
        trailing_gap_days=trailing_gap_days,
        days=days,
        cells=cells,
    )
    return partial(_run_kbdi_kernel, native, arrays)
