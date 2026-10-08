"""The optional Rust kernels behind the fire-weather recurrences.

``crates/climate-core`` runs the KBDI and CFFWIS moisture-code daily recursions
over a whole time axis per cell (``docs/architecture.md``). This module is the
fire package's only native boundary: it decides whether the extension can take
a recurrence's prepared arrays, builds the callable that
:mod:`climate_indices._recurrence` runs in place of its day loop, and translates
the extension's errors into the ones the Python driver raises.

Nothing here holds an algorithm. The Python implementations in ``_kbdi.py`` and
``_cffwis_codes.py`` stay the parity oracles, and every layout this module
cannot hand to the extension unchanged stays on the Python path.
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, NoReturn

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


def _native_module() -> ModuleType | None:
    """The optional Rust extension, or None when this install is pure Python.

    Every guard reads the module through this accessor: a successful import is
    typed as always present, which would make the fallback branches look
    unreachable to the type checker.
    """
    return _native


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


def _block(array: npt.NDArray[np.float64], days: int, cells: int) -> npt.NDArray[np.float64]:
    """One time-first float64 input as the contiguous ``(days, cells)`` block a kernel takes."""
    return np.ascontiguousarray(array, dtype=np.float64).reshape(days, cells)


def _flat(array: npt.NDArray[np.float64], cells: int) -> npt.NDArray[np.float64]:
    """One per-cell float64 array as the contiguous ``(cells,)`` vector a kernel takes."""
    return np.ascontiguousarray(array, dtype=np.float64).reshape(cells)


def _mask(array: npt.NDArray[np.bool_], days: int, cells: int) -> npt.NDArray[np.bool_]:
    """One time-first validity mask as the contiguous ``(days, cells)`` block a kernel takes."""
    return np.ascontiguousarray(array, dtype=np.bool_).reshape(days, cells)


def _optional_mask(array: npt.NDArray[np.bool_] | None, days: int, cells: int) -> npt.NDArray[np.bool_] | None:
    """An optional ``(days, cells)`` mask, or None when the recurrence has none."""
    return None if array is None else _mask(array, days, cells)


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
    is not a plain, aligned float64 ``ndarray``, a masked array, or an empty
    axis — leaves the recurrence to the Python driver, which is the parity
    oracle.
    """
    native = _native_module()
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
    humidity = None if inputs.humidity is None else _block(inputs.humidity, days, cells)
    wind = None if inputs.wind_kilometers_per_hour is None else _block(inputs.wind_kilometers_per_hour, days, cells)
    months = None if inputs.months is None else inputs.months.reshape(days, cells).astype(np.int64)
    band = None if inputs.day_length_band is None else inputs.day_length_band.reshape(cells).astype(np.int64)
    table = code.day_length_table
    if code.index_type != "ffmc" and (months is None or band is None or table is None):
        return None

    index_type = code.index_type
    initial = _flat(state_value, cells)
    temperature_celsius = _block(temperature, days, cells)
    precipitation_mm = _block(precipitation, days, cells)
    valid = _mask(weather_valid, days, cells)
    static = np.ascontiguousarray(static_valid, dtype=np.bool_).reshape(cells)
    season = _optional_mask(in_season, days, cells)
    gap_days = np.ascontiguousarray(trailing_gap_days, dtype=np.int64).reshape(cells)

    def native_run(
        values_out: npt.NDArray[np.float64] | None,
        spin_up: int,
        nan_policy: str,
        max_gap_days: int,
    ) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.int64] | None] | None:
        record = values_out is not None
        try:
            if index_type == "ffmc":
                assert humidity is not None and wind is not None
                history, final, final_gaps = native.ffmc(
                    temperature_celsius=temperature_celsius,
                    relative_humidity_percent=humidity,
                    wind_speed_kilometers_per_hour=wind,
                    precipitation_mm=precipitation_mm,
                    initial_ffmc=initial,
                    weather_valid=valid,
                    static_valid=static,
                    in_season=season,
                    trailing_gap_days=gap_days,
                    spin_up=spin_up,
                    nan_policy=nan_policy,
                    max_gap_days=max_gap_days,
                    record=record,
                )
            elif index_type == "duff_moisture_code":
                assert humidity is not None and months is not None and band is not None and table is not None
                history, final, final_gaps = native.duff_moisture_code(
                    temperature_celsius=temperature_celsius,
                    relative_humidity_percent=humidity,
                    precipitation_mm=precipitation_mm,
                    day_length_table=table,
                    months=months,
                    day_length_band=band,
                    initial_dmc=initial,
                    weather_valid=valid,
                    static_valid=static,
                    in_season=season,
                    trailing_gap_days=gap_days,
                    spin_up=spin_up,
                    nan_policy=nan_policy,
                    max_gap_days=max_gap_days,
                    record=record,
                )
            else:
                assert months is not None and band is not None and table is not None
                history, final, final_gaps = native.drought_code(
                    temperature_celsius=temperature_celsius,
                    precipitation_mm=precipitation_mm,
                    day_length_table=table,
                    months=months,
                    day_length_band=band,
                    initial_dc=initial,
                    weather_valid=valid,
                    static_valid=static,
                    in_season=season,
                    trailing_gap_days=gap_days,
                    spin_up=spin_up,
                    nan_policy=nan_policy,
                    max_gap_days=max_gap_days,
                    record=record,
                )
        except native.NonFiniteResultError as exc:  # pragma: no cover - defensive parity with the Python driver
            _raise_non_finite(index_type, exc)

        # the kernel returns whole arrays; the driver reads the component's own
        # state and gap arrays, so write the results back where it looks for them
        state_value[...] = final.reshape(state_value.shape)
        gap_shape = trailing_gap_days.shape
        if final_gaps is not None:
            trailing_gap_days[...] = final_gaps.reshape(gap_shape)
        if values_out is not None and history is not None:
            values_out[...] = history.reshape(values_out.shape)
        return values_out, None if final_gaps is None else final_gaps.reshape(gap_shape).copy()

    return native_run


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
    native = _native_module()
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
    precipitation = _block(precipitation_mm, days, cells)
    temperature = _block(maximum_temperature_celsius, days, cells)
    mean_annual = _flat(mean_annual_precipitation_mm, cells)
    initial_kbdi = _flat(kbdi_value, cells)
    initial_wet_spell = _flat(wet_spell_precipitation, cells)
    valid = _mask(weather_valid, days, cells)
    static = np.ascontiguousarray(static_valid, dtype=np.bool_).reshape(cells)
    gap_days = np.ascontiguousarray(trailing_gap_days, dtype=np.int64).reshape(cells)

    def native_run(
        values_out: npt.NDArray[np.float64] | None,
        spin_up: int,
        nan_policy: str,
        max_gap_days: int,
    ) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.int64] | None] | None:
        record = values_out is not None
        try:
            history, final_kbdi, final_wet_spell, final_gaps = native.kbdi(
                precipitation_mm=precipitation,
                maximum_temperature_celsius=temperature,
                mean_annual_precipitation_mm=mean_annual,
                initial_kbdi=initial_kbdi,
                initial_wet_spell_precipitation=initial_wet_spell,
                weather_valid=valid,
                static_valid=static,
                in_season=None,
                trailing_gap_days=gap_days,
                spin_up=spin_up,
                nan_policy=nan_policy,
                max_gap_days=max_gap_days,
                record=record,
            )
        except native.NonFiniteResultError as exc:  # pragma: no cover - defensive parity with the Python driver
            _raise_non_finite("kbdi", exc)

        kbdi_value[...] = final_kbdi.reshape(kbdi_value.shape)
        wet_spell_precipitation[...] = final_wet_spell.reshape(wet_spell_precipitation.shape)
        gap_shape = trailing_gap_days.shape
        if final_gaps is not None:
            trailing_gap_days[...] = final_gaps.reshape(gap_shape)
        if values_out is not None and history is not None:
            values_out[...] = history.reshape(values_out.shape)
        return values_out, None if final_gaps is None else final_gaps.reshape(gap_shape).copy()

    return native_run
