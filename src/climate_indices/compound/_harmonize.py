"""Declared-input harmonization: thresholded hazards on observed, comparable time steps.

``harmonize()`` turns declared probability or Boolean-indicator series into Boolean
exceedance and validity arrays on one time axis. It never invents an observation:

- an absent date, NaN, or masked value is *invalid*, never ``False``, and interior
  gaps stay in place instead of being closed;
- daily inputs are thresholded first, then reduced to calendar months with an explicit
  ``"any"`` or ``"all"`` rule, and a month with one unobserved civil day is invalid;
- probabilities are never averaged, interpolated, clipped, or converted from z-scores.

Dates are ``datetime64`` (proleptic Gregorian). Monthly labels identify calendar months
and are normalized to the first of the month; real February 29 dates are ordinary days.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices.exceptions import (
    ClimateIndicesWarning,
    CoordinateValidationError,
    DataShapeError,
    InputAlignmentWarning,
    InputTypeError,
    InvalidArgumentError,
)
from climate_indices.logging_config import get_logger, log_calculation_failure
from climate_indices.runs import _VALID_DIRECTIONS, Direction, _as_finite_threshold

_logger = get_logger(__name__)

Cadence: TypeAlias = Literal["daily", "monthly"]
InputScale: TypeAlias = Literal["probability", "indicator"]
Aggregation: TypeAlias = Literal["any", "all"]
Join: TypeAlias = Literal["exact", "inner"]

_CADENCES = ("daily", "monthly")
_AGGREGATIONS = ("any", "all")
_JOINS = ("exact", "inner")
_UNKNOWN = "unknown"
_MANIFEST_SCHEMA_VERSION = 1

# an indicator source's public declaration; its physical threshold and comparator live in the manifest
_INDICATOR_THRESHOLD = 0.5
_INDICATOR_DIRECTION = "above"


@dataclass(frozen=True, eq=False)
class AlignedHazards:
    """Boolean hazard occurrence and validity on one observed time axis.

    Instances compare by identity: the fields are arrays. Every array is read-only
    and was built by :func:`harmonize`; mutate neither it nor ``metadata``.

    Args:
        names: Source names in input order; the leading axis of the arrays.
        time: Observed time labels, shape ``(time,)``, as ``datetime64[D]``. A
            monthly label is the first day of its calendar month.
        cadence: ``"daily"`` or ``"monthly"``; the step of ``time``.
        exceedance: ``True`` where a valid observation meets the hazard condition,
            shape ``(hazard, time, *cells)``. It is ``False`` wherever ``valid`` is.
        valid: ``True`` where the hazard was observed, same shape as ``exceedance``.
            An unobserved position is invalid, not a non-exceedance.
        metadata: Immutable, JSON-compatible manifest of every declaration, date range,
            reduction, dropped boundary, and known unknown that the arrays depend on.
    """

    names: tuple[str, ...]
    time: npt.NDArray[np.datetime64]
    cadence: Cadence
    exceedance: npt.NDArray[np.bool_]
    valid: npt.NDArray[np.bool_]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _Prepared:
    """One source on its own complete grid at the target cadence."""

    name: str
    hazard: npt.NDArray[np.bool_]  # (steps, *cells); False wherever invalid
    valid: npt.NDArray[np.bool_]
    first: int  # target-unit index (days or months since 1970-01) of step 0
    entry: dict[str, Any]  # manifest entry, extended with the dropped boundaries on alignment
    calibration: tuple[int, int] | None

    @property
    def last(self) -> int:
        """Return the inclusive target-unit index of the final step."""
        return self.first + int(self.hazard.shape[0]) - 1


def _freeze(value: Any) -> Any:
    """Return ``value`` with every nested dict as a read-only mapping and list as a tuple."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _check_keys(argument: str, mapping: object, names: tuple[str, ...], *, subset: bool = False) -> Mapping[str, Any]:
    """Require ``mapping`` to be keyed by exactly (or a subset of) the input names."""
    if not isinstance(mapping, Mapping):
        raise InvalidArgumentError(
            f"{argument} must be a mapping keyed by source name.",
            argument_name=argument,
            argument_value=type(mapping).__name__,
            valid_values="Mapping[str, ...]",
        )
    unexpected = [repr(key) for key in mapping if key not in names]
    missing = [] if subset else [repr(name) for name in names if name not in mapping]
    if unexpected or missing:
        raise InvalidArgumentError(
            f"{argument} keys must {'be a subset of' if subset else 'match'} the input names {list(names)}; "
            f"unexpected: {unexpected}, missing: {missing}.",
            argument_name=argument,
            valid_values=f"keys {'within' if subset else 'equal to'} the input names",
        )
    return mapping


def _check_choice(argument: str, value: object, valid: tuple[str, ...]) -> None:
    """Require ``value`` to be one of the ``valid`` strings."""
    if not isinstance(value, str) or value not in valid:
        raise InvalidArgumentError(
            f"{argument} must be one of {valid}.",
            argument_name=argument,
            argument_value=repr(value),
            valid_values=", ".join(repr(choice) for choice in valid),
        )


def _check_scale(name: str, scale: object) -> None:
    """Require a declared probability or indicator scale, rejecting z-score declarations."""
    if scale in ("normal", "bounded"):
        raise InvalidArgumentError(
            f"input_scale[{name!r}]={scale!r} is not supported: a z-score inside [0, 1] cannot be detected "
            "numerically and clipped scores cannot recover the fitted tail. Request "
            'output_scale="probability" upstream, or declare a Boolean "indicator".',
            argument_name="input_scale",
            argument_value=repr(scale),
            valid_values="'probability' or 'indicator'",
        )
    _check_choice(f"input_scale[{name!r}]", scale, ("probability", "indicator"))


def _declared_threshold(name: str, scale: str, threshold: object, direction: object) -> float:
    """Validate one source's threshold and direction declaration and return the threshold."""
    _check_choice(f"directions[{name!r}]", direction, _VALID_DIRECTIONS)
    try:
        value = _as_finite_threshold(threshold)
    except InvalidArgumentError as exc:
        raise InvalidArgumentError(
            f"thresholds[{name!r}] must be a finite number.",
            argument_name="thresholds",
            argument_value=repr(threshold),
            valid_values="a finite int or float",
        ) from exc
    if scale == "indicator":
        if value != _INDICATOR_THRESHOLD or direction != _INDICATOR_DIRECTION:
            raise InvalidArgumentError(
                f"Boolean indicator {name!r} must declare threshold {_INDICATOR_THRESHOLD} and direction "
                f"{_INDICATOR_DIRECTION!r}; record its physical threshold and comparator in source_metadata.",
                argument_name="thresholds",
                argument_value=f"{threshold!r}, {direction!r}",
                valid_values=f"{_INDICATOR_THRESHOLD}, {_INDICATOR_DIRECTION!r}",
            )
    elif not 0.0 <= value <= 1.0:
        raise InvalidArgumentError(
            f"thresholds[{name!r}] must lie in [0, 1] for a probability input.",
            argument_name="thresholds",
            argument_value=repr(threshold),
            valid_values="0 <= threshold <= 1",
        )
    return value


def _hazard_and_validity(
    name: str,
    values: npt.ArrayLike,
    scale: str,
    threshold: float,
    direction: str,
    validity: object,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Apply the declared threshold to one source, returning ``(hazard, valid)`` with ``hazard`` False where invalid."""
    if isinstance(values, xr.DataArray):
        raise InputTypeError(
            f"inputs[{name!r}] is an xarray.DataArray; harmonize() accepts NumPy arrays and explicit dates only.",
            expected_type=np.ndarray,
            actual_type=type(values),
        )
    masked: np.ma.MaskedArray[Any, Any] = np.ma.asanyarray(values)
    array = np.ma.getdata(masked)
    valid = ~np.ma.getmaskarray(masked)
    if array.ndim == 0:
        raise DataShapeError(f"inputs[{name!r}] must have a time axis.", expected_shape="(time, ...)", actual_shape=())

    if scale == "indicator":
        if array.dtype.kind != "b":
            raise InputTypeError(
                f"inputs[{name!r}] is declared an indicator and must have Boolean dtype, not {array.dtype}.",
                expected_type=np.bool_,
                actual_type=array.dtype.type,
            )
        hazard = array.copy()
    else:
        if array.dtype.kind not in "iuf":
            raise InputTypeError(
                f"inputs[{name!r}] is declared a probability and must be a real numeric array, not {array.dtype}.",
                expected_type=float,
                actual_type=array.dtype.type,
            )
        # float64 holds every int and float32 value exactly, so the strict comparison tests the stored value
        data = array.astype(np.float64)
        valid &= ~np.isnan(data)
        if np.any(valid & ~((data >= 0.0) & (data <= 1.0))):  # also catches +/-inf
            raise InvalidArgumentError(
                f"inputs[{name!r}] holds probabilities outside [0, 1] or infinite values; they are not clipped.",
                argument_name="inputs",
                valid_values="values in [0, 1], NaN, or masked",
            )
        hazard = data > threshold if direction == "above" else data < threshold

    if validity is not None:
        mask = np.asarray(validity)
        if mask.dtype.kind != "b":
            raise InputTypeError(
                f"validity[{name!r}] must have Boolean dtype, not {mask.dtype}.",
                expected_type=np.bool_,
                actual_type=mask.dtype.type,
            )
        if mask.shape != array.shape:
            raise DataShapeError(
                f"validity[{name!r}] must match the shape of inputs[{name!r}].",
                expected_shape=str(array.shape),
                actual_shape=mask.shape,
            )
        valid &= mask
    return hazard & valid, valid


def _observed_index(name: str, raw: object, cadence: str, steps: int) -> npt.NDArray[np.int64]:
    """Validate one source's dates and return them as days or months since 1970-01-01, strictly increasing."""
    label = f"time[{name!r}]"
    dates = np.asarray(raw)
    if dates.dtype.kind != "M":
        raise InputTypeError(
            f"{label} must be a datetime64 array; cftime, timezone-aware, string, and numeric dates are rejected "
            "and calendars are never converted.",
            expected_type=np.datetime64,
            actual_type=dates.dtype.type,
        )
    if dates.ndim != 1 or dates.shape[0] != steps:
        raise DataShapeError(
            f"{label} must be one-dimensional with one date per time step of inputs[{name!r}].",
            expected_shape=f"({steps},)",
            actual_shape=dates.shape,
        )
    if np.isnat(dates).any():
        raise CoordinateValidationError(f"{label} contains NaT.", coordinate_name=label, reason="NaT")
    index = dates.astype("datetime64[D]" if cadence == "daily" else "datetime64[M]").astype(np.int64)
    step = np.diff(index)
    if np.any(step < 0):
        raise CoordinateValidationError(f"{label} must be increasing.", coordinate_name=label, reason="unordered")
    if np.any(step == 0):
        unit = "civil day" if cadence == "daily" else "calendar month"
        raise CoordinateValidationError(
            f"{label} holds more than one observation in a {unit}.", coordinate_name=label, reason="duplicate"
        )
    return index


def _on_grid(
    positions: npt.NDArray[np.int64], steps: int, hazard: npt.NDArray[np.bool_], valid: npt.NDArray[np.bool_]
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Place observations on a complete grid of ``steps``; every absent position is invalid."""
    cells = hazard.shape[1:]
    grid_hazard = np.zeros((steps, *cells), dtype=np.bool_)
    grid_valid = np.zeros((steps, *cells), dtype=np.bool_)
    grid_hazard[positions] = hazard
    grid_valid[positions] = valid
    return grid_hazard, grid_valid


def _days_to_months(
    day_index: npt.NDArray[np.int64],
    hazard: npt.NDArray[np.bool_],
    valid: npt.NDArray[np.bool_],
    aggregation: str,
) -> tuple[int, npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Reduce daily hazard occurrence to calendar months; a month is valid only when every civil day is."""
    days = day_index.astype("datetime64[D]")
    first_month = days[0].astype("datetime64[M]")
    last_month = days[-1].astype("datetime64[M]")
    month_starts = np.arange(first_month, last_month + 1).astype("datetime64[D]")
    grid_start = month_starts[0]
    steps = int((last_month + 1).astype("datetime64[D]").astype(np.int64) - grid_start.astype(np.int64))
    grid_hazard, grid_valid = _on_grid(day_index - grid_start.astype(np.int64), steps, hazard, valid)
    # every month has at least 28 days, so no reduceat segment is empty
    offsets = (month_starts - grid_start).astype(np.int64)
    month_valid = np.logical_and.reduceat(grid_valid, offsets, axis=0)
    reducer = np.logical_or if aggregation == "any" else np.logical_and
    month_hazard = reducer.reduceat(grid_hazard, offsets, axis=0) & month_valid
    return int(first_month.astype(np.int64)), month_hazard, month_valid


def _source_metadata(name: str, metadata: object) -> tuple[dict[str, Any], tuple[int, int] | None]:
    """Return a JSON-round-tripped copy of one source's metadata and its known Calibration Period."""
    try:
        copied: dict[str, Any] = json.loads(json.dumps(dict(metadata) if metadata is not None else {}, allow_nan=False))  # type: ignore[call-overload]  # a non-mapping raises below
    except (TypeError, ValueError) as exc:
        raise InvalidArgumentError(
            f"source_metadata[{name!r}] must be a JSON-compatible mapping.",
            argument_name="source_metadata",
            valid_values="a mapping of JSON strings, numbers, booleans, null, lists, and mappings",
        ) from exc
    initial = copied.get("calibration_year_initial")
    final = copied.get("calibration_year_final")
    if initial is None and final is None:
        return copied, None
    if type(initial) is not int or type(final) is not int or initial > final:
        raise InvalidArgumentError(
            f"source_metadata[{name!r}] must give calibration_year_initial and calibration_year_final together "
            "as integer years with initial <= final.",
            argument_name="source_metadata",
            argument_value=f"{initial!r}, {final!r}",
            valid_values="both integer years, initial <= final",
        )
    return copied, (initial, final)


def _prepare_source(
    name: str,
    values: npt.ArrayLike,
    *,
    time: object,
    cadence: str,
    scale: str,
    threshold: object,
    direction: object,
    aggregation: str | None,
    target_cadence: str,
    validity: object,
    metadata: object,
    spatial_time_major: bool,
) -> _Prepared:
    """Validate one source and place it on its own complete grid at the target cadence."""
    _check_choice(f"cadence[{name!r}]", cadence, _CADENCES)
    _check_scale(name, scale)
    threshold_value = _declared_threshold(name, scale, threshold, direction)
    hazard, valid = _hazard_and_validity(name, values, scale, threshold_value, str(direction), validity)
    if hazard.ndim > 1 and not spatial_time_major:
        raise DataShapeError(
            f"inputs[{name!r}] has shape {hazard.shape}; a gridded input is read as (time, *cells) only when "
            "declared with spatial_time_major=True. Legacy (years, periods) layouts are never guessed.",
            expected_shape="(time,) or (time, *cells) with spatial_time_major=True",
            actual_shape=hazard.shape,
        )
    if hazard.shape[0] == 0:
        raise DataShapeError(
            f"inputs[{name!r}] has no time steps.", expected_shape="(time>=1, ...)", actual_shape=hazard.shape
        )
    index = _observed_index(name, time, cadence, hazard.shape[0])
    copied, calibration = _source_metadata(name, metadata)

    if cadence == "daily" and target_cadence == "monthly":
        first, grid_hazard, grid_valid = _days_to_months(index, hazard, valid, str(aggregation))
    else:
        first = int(index[0])
        grid_hazard, grid_valid = _on_grid(index - first, int(index[-1]) - first + 1, hazard, valid)

    label_unit = "datetime64[D]" if cadence == "daily" else "datetime64[M]"
    entry: dict[str, Any] = {
        "input_scale": scale,
        "threshold": threshold_value,
        "direction": direction,
        "cadence": cadence,
        "aggregation": aggregation,
        "time_start": str(index[0].astype(label_unit)),
        "time_end": str(index[-1].astype(label_unit)),
        "n_observed": int(index.shape[0]),
        "calibration_period": list(calibration) if calibration else _UNKNOWN,
        "scale": copied.get("scale", _UNKNOWN),
    }
    if scale == "indicator":
        entry["physical_threshold"] = copied.get("physical_threshold", _UNKNOWN)
        entry["comparator"] = copied.get("comparator", _UNKNOWN)
    entry["source_metadata"] = copied
    return _Prepared(name, grid_hazard, grid_valid, first, entry, calibration)


def _check_aggregation(
    aggregation: object, names: tuple[str, ...], cadence: Mapping[str, Any], target_cadence: str
) -> Mapping[str, Any]:
    """Require a reducer for each daily source of a monthly target, and none anywhere else."""
    reducers = _check_keys("aggregation", {} if aggregation is None else aggregation, names, subset=True)
    for name in names:
        reducer = reducers.get(name)
        needs_reducer = target_cadence == "monthly" and cadence[name] == "daily"
        if needs_reducer:
            _check_choice(f"aggregation[{name!r}]", reducer, _AGGREGATIONS)
        elif reducer is not None:
            raise InvalidArgumentError(
                f"aggregation[{name!r}] is an irrelevant reducer: only daily sources of a monthly target are reduced.",
                argument_name="aggregation",
                argument_value=repr(reducer),
                valid_values="no entry for this source",
            )
    return reducers


def _align(prepared: list[_Prepared], target_cadence: str, join: str) -> tuple[int, int, int]:
    """Return the retained inclusive ``(first, last)`` target-unit range and the total dropped step count."""
    firsts = [source.first for source in prepared]
    lasts = [source.last for source in prepared]
    if join == "exact":
        if len(set(firsts)) != 1 or len(set(lasts)) != 1:
            ranges = {source.name: (source.entry["time_start"], source.entry["time_end"]) for source in prepared}
            raise CoordinateValidationError(
                f'join="exact" requires identical normalized coverage at the {target_cadence} cadence; '
                f'sources span {ranges}. Pass join="inner" to restrict to the common overlap.',
                coordinate_name="time",
                reason="coverage mismatch",
            )
        return firsts[0], lasts[0], 0
    first, last = max(firsts), min(lasts)
    if first > last:
        raise CoordinateValidationError(
            "The sources share no time range, so there is nothing to align.",
            coordinate_name="time",
            reason="empty overlap",
        )
    dropped = 0
    for source in prepared:
        leading, trailing = first - source.first, source.last - last
        source.entry["dropped_leading"] = leading
        source.entry["dropped_trailing"] = trailing
        dropped += leading + trailing
    return first, last, dropped


def _harmonize(
    inputs: Mapping[str, npt.ArrayLike],
    *,
    time: Mapping[str, npt.ArrayLike],
    cadence: Mapping[str, Cadence],
    input_scale: Mapping[str, InputScale],
    thresholds: Mapping[str, float],
    directions: Mapping[str, Direction],
    target_cadence: Cadence,
    aggregation: Mapping[str, Aggregation] | None,
    validity: Mapping[str, npt.ArrayLike] | None,
    join: Join,
    source_metadata: Mapping[str, Mapping[str, Any]] | None,
    spatial_time_major: bool,
) -> AlignedHazards:
    """Validate, threshold, reduce, and align; see :func:`harmonize`."""
    if not isinstance(inputs, Mapping) or not inputs:
        raise InvalidArgumentError(
            "inputs must be a non-empty mapping of source name to array.", argument_name="inputs"
        )
    names = tuple(inputs)
    if not all(isinstance(name, str) for name in names):
        raise InvalidArgumentError("inputs keys must be strings.", argument_name="inputs")
    for argument, mapping in (
        ("time", time),
        ("cadence", cadence),
        ("input_scale", input_scale),
        ("thresholds", thresholds),
        ("directions", directions),
    ):
        _check_keys(argument, mapping, names)
    validity_by_name = _check_keys("validity", {} if validity is None else validity, names, subset=True)
    metadata_by_name = _check_keys(
        "source_metadata", {} if source_metadata is None else source_metadata, names, subset=True
    )
    _check_choice("target_cadence", target_cadence, _CADENCES)
    _check_choice("join", join, _JOINS)
    if not isinstance(spatial_time_major, bool):
        raise InvalidArgumentError(
            "spatial_time_major must be a bool.",
            argument_name="spatial_time_major",
            argument_value=repr(spatial_time_major),
        )
    for name in names:
        _check_choice(f"cadence[{name!r}]", cadence[name], _CADENCES)
    if target_cadence == "daily" and any(cadence[name] == "monthly" for name in names):
        raise InvalidArgumentError(
            'A monthly source cannot be expanded to a daily target; use target_cadence="monthly" or provide daily data.',
            argument_name="target_cadence",
            argument_value="daily",
            valid_values="'monthly' when any source is monthly",
        )
    reducers = _check_aggregation(aggregation, names, cadence, target_cadence)

    prepared = [
        _prepare_source(
            name,
            inputs[name],
            time=time[name],
            cadence=cadence[name],
            scale=input_scale[name],
            threshold=thresholds[name],
            direction=directions[name],
            aggregation=reducers.get(name),
            target_cadence=target_cadence,
            validity=validity_by_name.get(name),
            metadata=metadata_by_name.get(name),
            spatial_time_major=spatial_time_major,
        )
        for name in names
    ]
    cells = {source.hazard.shape[1:] for source in prepared}
    if len(cells) != 1:
        raise DataShapeError(
            "Every source must have the same cell shape; broadcasting a point across a grid is never implicit.",
            expected_shape="one common (*cells)",
            actual_shape=None,
        )

    known = {source.name: source.calibration for source in prepared if source.calibration is not None}
    if len(set(known.values())) > 1:
        raise InvalidArgumentError(
            f"Sources were calibrated on conflicting periods {known}; compound statistics need one Calibration Period.",
            argument_name="source_metadata",
            valid_values="identical calibration_year_initial/final across sources, or none",
        )

    first, last, dropped = _align(prepared, target_cadence, join)
    exceedance = np.stack([source.hazard[first - source.first : last - source.first + 1] for source in prepared])
    valid = np.stack([source.valid[first - source.first : last - source.first + 1] for source in prepared])
    steps = np.arange(first, last + 1, dtype=np.int64)
    time_labels = (
        steps.astype("datetime64[D]")
        if target_cadence == "daily"
        else steps.astype("datetime64[M]").astype("datetime64[D]")
    )
    for array in (exceedance, valid, time_labels):
        array.setflags(write=False)

    undeclared = [
        source.name
        for source in prepared
        if source.entry["input_scale"] == "probability" and source.calibration is None
    ]
    if undeclared:
        warnings.warn(
            ClimateIndicesWarning(
                f"Calibration Period metadata is missing for probability sources {undeclared}; it is recorded as "
                f"{_UNKNOWN!r} and calibration compatibility is not checked."
            ),
            stacklevel=3,
        )
    if dropped:
        warnings.warn(
            InputAlignmentWarning(
                f'join="inner" dropped {dropped} boundary time steps in total to keep the common overlap.',
                original_size=max(source.hazard.shape[0] for source in prepared),
                aligned_size=last - first + 1,
                dropped_count=dropped,
            ),
            stacklevel=3,
        )

    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "calendar": "proleptic_gregorian",
        "cadence": target_cadence,
        "join": join,
        "time_start": str(time_labels[0]) if target_cadence == "daily" else str(time_labels[0].astype("datetime64[M]")),
        "time_end": str(time_labels[-1]) if target_cadence == "daily" else str(time_labels[-1].astype("datetime64[M]")),
        "n_steps": int(steps.shape[0]),
        "missingness": (
            "An absent date, NaN, or masked value is invalid, not False, and interior gaps stay in place. A reduced "
            "month is valid only when every civil day is valid."
        ),
        "sources": {source.name: source.entry for source in prepared},
    }
    return AlignedHazards(names, time_labels, target_cadence, exceedance, valid, _freeze(manifest))


def harmonize(
    inputs: Mapping[str, npt.ArrayLike],
    *,
    time: Mapping[str, npt.ArrayLike],
    cadence: Mapping[str, Cadence],
    input_scale: Mapping[str, InputScale],
    thresholds: Mapping[str, float],
    directions: Mapping[str, Direction],
    target_cadence: Cadence,
    aggregation: Mapping[str, Aggregation] | None = None,
    validity: Mapping[str, npt.ArrayLike] | None = None,
    join: Join = "exact",
    source_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    spatial_time_major: bool = False,
) -> AlignedHazards:
    """Threshold declared inputs and align their observed hazard opportunities.

    Each source declares how to read its values: ``"probability"`` inputs are
    thresholded with a strict ``>`` or ``<`` in the declared direction, and
    ``"indicator"`` inputs are already Boolean hazard occurrence. A declared
    probability is a trust boundary: a z-score that happens to lie in ``[0, 1]``
    cannot be detected, and nothing here proves the values are uniform PIT values
    or a good fit. Probabilities are compared in float64 after exact widening, so
    a float32 value that prints as the threshold is not equal to it.

    Daily sources are thresholded first. With ``target_cadence="monthly"`` each
    is then reduced per calendar month with ``aggregation[name]``: ``"any"`` asks
    whether the month holds at least one hazard day and ``"all"`` whether every day
    meets the condition. A month is valid only if every civil day, including a real
    February 29, was observed; a partial month is invalid even when an observed day
    exceeds the threshold. The result holds hazard masks, never re-labelled
    probabilities. Monthly sources are never expanded to days.

    Args:
        inputs: Source name to array shaped ``(time,)``, or ``(time, *cells)`` when
            ``spatial_time_major=True``. NaN and masked elements are unknown.
        time: Source name to its observation dates (``datetime64``, strictly
            increasing). A monthly label is normalized to the first of its month.
        cadence: Source name to ``"daily"`` or ``"monthly"``.
        input_scale: Source name to ``"probability"`` or ``"indicator"``. ``"normal"``
            and ``"bounded"`` are rejected; request probabilities upstream.
        thresholds: Source name to a finite threshold in ``[0, 1]``. An indicator
            must declare ``0.5``; record its physical threshold in ``source_metadata``.
        directions: Source name to ``"above"`` or ``"below"``. An indicator must
            declare ``"above"``.
        target_cadence: ``"daily"`` or ``"monthly"``, the step of the result.
        aggregation: Source name to ``"any"`` or ``"all"`` for each daily source of
            a monthly target, and no entry for any other source.
        validity: Optional source name to a Boolean array shaped like its input, marking
            observed positions. It is combined with NaN and mask validity.
        join: ``"exact"`` requires identical coverage at the target cadence.
            ``"inner"`` keeps the common time range and warns about dropped boundaries.
        source_metadata: Optional source name to a JSON-compatible mapping. The keys
            ``calibration_year_initial`` and ``calibration_year_final`` (both or
            neither), ``scale`` (Timescale), and, for an indicator, ``physical_threshold``
            and ``comparator`` are recorded in the manifest; everything is copied.
        spatial_time_major: Declare that arrays with more than one dimension are
            ``(time, *cells)`` blocks. Every source must have the same cell shape.

    Returns:
        An :class:`AlignedHazards` whose ``exceedance`` and ``valid`` arrays are new
        and read-only, shaped ``(hazard, time, *cells)``. The caller's arrays are not
        modified.

    Raises:
        InvalidArgumentError: For an inconsistent or unsupported declaration, a
            probability outside ``[0, 1]``, a reducer that does not apply, a monthly
            source with a daily target, or sources with conflicting known Calibration
            Periods.
        InputTypeError: For a dtype that does not match the declared scale, or dates
            that are not ``datetime64``.
        DataShapeError: For an undeclared gridded layout, mismatched cell shapes, or
            dates or validity that do not match an input.
        CoordinateValidationError: For NaT, duplicate, or unordered dates, mismatched
            coverage under ``join="exact"``, or an empty overlap under ``join="inner"``.

    Warns:
        ClimateIndicesWarning: When a probability source has no Calibration Period
            metadata; it is recorded as ``"unknown"``.
        InputAlignmentWarning: When ``join="inner"`` drops boundary time steps.

    Example:
        >>> import numpy as np
        >>> from climate_indices import compound
        >>> months = np.arange("2000-01", "2000-05", dtype="datetime64[M]")
        >>> aligned = compound.harmonize(
        ...     {"drought": np.array([0.04, 0.50, 0.03, np.nan]), "heat": np.array([True, False, True, True])},
        ...     time={"drought": months, "heat": months},
        ...     cadence={"drought": "monthly", "heat": "monthly"},
        ...     input_scale={"drought": "probability", "heat": "indicator"},
        ...     thresholds={"drought": 0.1, "heat": 0.5},
        ...     directions={"drought": "below", "heat": "above"},
        ...     target_cadence="monthly",
        ...     source_metadata={"drought": {"calibration_year_initial": 1991, "calibration_year_final": 2020}},
        ... )
        >>> aligned.exceedance.astype(int).tolist()
        [[1, 0, 1, 0], [1, 0, 1, 1]]
        >>> aligned.valid.astype(int).tolist()
        [[1, 1, 1, 0], [1, 1, 1, 1]]
    """
    log = _logger.bind(
        index_type="compound_harmonize",
        sources=tuple(inputs) if isinstance(inputs, Mapping) else (),
        target_cadence=target_cadence,
        join=join,
    )
    log.info("calculation_started")
    t0 = perf_counter()
    try:
        result = _harmonize(
            inputs,
            time=time,
            cadence=cadence,
            input_scale=input_scale,
            thresholds=thresholds,
            directions=directions,
            target_cadence=target_cadence,
            aggregation=aggregation,
            validity=validity,
            join=join,
            source_metadata=source_metadata,
            spatial_time_major=spatial_time_major,
        )
    except Exception as exc:
        log_calculation_failure(log, exc)
        raise
    log.info(
        "calculation_completed",
        duration_ms=round((perf_counter() - t0) * 1000.0, 2),
        output_shape=result.exceedance.shape,
    )
    return result
