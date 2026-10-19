"""Eager numeric compound-event catalogs; all matching stays in the NumPy core."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import fields
from typing import Any, cast

import numpy as np
import xarray as xr

from climate_indices.compound._events import CompoundEventSet, EventMode, _compound_events
from climate_indices.compound._harmonize import AlignedHazards, Cadence
from climate_indices.compound._statistics import _validate_hazards
from climate_indices.compound._xarray import _MANIFEST, _count, _dataset_attrs, _prepare, _Prepared, _route
from climate_indices.exceptions import DimensionMismatchError, InputTypeError, InvalidArgumentError
from climate_indices.runs import _resolve_options, identify_runs

_EVENT_FIELDS = tuple(field.name for field in fields(CompoundEventSet) if field.name not in ("mode", "metadata"))
_CELL_COUNTS = ("n_valid", "n_first_runs", "n_second_runs", "n_pairs", "n_unknown_bridges")
_RESERVED = frozenset((*_EVENT_FIELDS, *_CELL_COUNTS, "event", "cell", "cell_index", "start_time", "end_time"))
# The event columns hold about 100 bytes per row and concatenation holds a second copy, so this caps a
# catalog near 200 MB at peak; a larger match fails instead of allocating until the process is exhausted
_MAX_EVENTS = 1_000_000
_LONG_NAMES = {
    "cell_index": "Cell Index",
    "first_run_id": "First Source Run Index",
    "second_run_id": "Second Source Run Index",
    "start_index": "Interval Start Step",
    "end_index": "Interval End Step",
    "overlap_duration": "Overlap Duration",
    "lag": "Sequential Lag",
    "first_left_censored": "First Source Left Censored",
    "first_right_censored": "First Source Right Censored",
    "second_left_censored": "Second Source Left Censored",
    "second_right_censored": "Second Source Right Censored",
    "n_valid": "Shared Valid Step Count",
    "n_first_runs": "First Source Retained Run Count",
    "n_second_runs": "Second Source Retained Run Count",
    "n_pairs": "Matched Pair Count",
    "n_unknown_bridges": "Rejected Unknown Bridge Count",
}
# a boundary declaration that would dangle once the referenced variable is dropped
_DROPPED_ATTRS = frozenset({"bounds", "coordinates", "grid_mapping", "axis", "cf_role"})


def _catalog(
    hazards: xr.Dataset,
    *,
    first: str,
    second: str,
    mode: EventMode,
    lag_window: tuple[int, int] | None,
    min_duration: int,
) -> xr.Dataset:
    """Validate an eager subset, identify runs on shared support, and concatenate per-cell pairs."""
    prepared, duration, source_metadata, plan = _validated_request(
        hazards, first, second, min_duration, mode=mode, lag_window=lag_window
    )
    grid, coords, cells = _cell_coords(hazards, prepared)
    selected = [prepared.names.index(first), prepared.names.index(second)]
    exceedance = prepared.exceedance.isel(hazard=selected).transpose("hazard", "time", *grid).values
    valid = prepared.valid.isel(hazard=selected).transpose("hazard", "time", *grid).values
    exceedance = exceedance.reshape(2, prepared.day.size, cells)
    valid = valid.reshape(2, prepared.day.size, cells)
    columns: dict[str, list[np.ndarray]] = {name: [] for name in (*_EVENT_FIELDS, "cell_index")}
    counts = {name: np.zeros(cells, dtype=np.int64) for name in _CELL_COUNTS}
    emitted = 0
    for cell in range(cells):
        shared = valid[:, :, cell].all(axis=0)
        runs = [
            identify_runs(
                np.where(shared, exceedance[source, :, cell], np.nan),
                threshold=0.5,
                direction="above",
                min_duration=duration,
            )
            for source in range(2)
        ]
        matched = _compound_events(
            *runs,
            time=prepared.day,
            valid=shared,
            cadence=prepared.cadence,
            mode=mode,
            lag_window=lag_window,
            source_metadata=source_metadata,
            max_pairs=_MAX_EVENTS - emitted,
            pair_limit=_MAX_EVENTS,
        )
        emitted += len(matched)
        for name in _EVENT_FIELDS:
            columns[name].append(getattr(matched, name))
        columns["cell_index"].append(np.full(len(matched), cell, dtype=np.int64))
        for name, value in zip(
            _CELL_COUNTS,
            (shared.sum(), len(runs[0]), len(runs[1]), len(matched), matched.metadata["n_unknown_bridges"]),
            strict=True,
        ):
            counts[name][cell] = value
    variables = _event_variables(columns, plan)
    for name, values in counts.items():
        variables[name] = ("cell", values, _count(_LONG_NAMES[name], _descriptions(name)))
    result: xr.Dataset = xr.Dataset(variables, coords=coords)
    result = result.assign_coords(event=np.arange(result.sizes["event"], dtype=np.int64))
    for boundary in ("start", "end"):
        result[f"{boundary}_time"] = (
            "event",
            prepared.day[result[f"{boundary}_index"].values].astype("datetime64[s]"),
            {
                "long_name": f"Reported pair {boundary} time",
                "description": "Inclusive boundary of the reported interval.",
            },
        )
    manifest = dict(plan.metadata)
    manifest["grid"] = dict(plan.metadata["grid"])
    # counts belong to cells, not to the zero-cell validation match
    for name in ("n_runs", "n_pairs", "n_unknown_bridges"):
        manifest.pop(name)
    for name in ("n_valid", "n_unknown"):
        manifest["grid"].pop(name)
    manifest.update(
        prepared_hazards=prepared.metadata,
        spatial_dimensions=list(grid),
        cell_order="C order over spatial_dimensions; source-run IDs are local to each cell after min_duration filtering.",
        observation_support="Both selected hazards must be valid; runs are identified on this shared mask.",
        censoring_support="Censor flags use the shared support, so a boundary is censored when the neighboring step is unknown in either selected source.",
        counts="Per-cell variables retain support, retained-run, pair, and rejected-bridge counts, including empty cells.",
        row_limit=f"A catalog holds at most {_MAX_EVENTS} event rows; a larger match raises InvalidArgumentError.",
    )
    result.attrs = _dataset_attrs("Compound event catalog", cadence=prepared.cadence, mode=mode)
    result.attrs[_MANIFEST] = json.dumps(manifest, allow_nan=False)
    return result


def _validated_request(
    hazards: xr.Dataset,
    first: str,
    second: str,
    min_duration: int,
    *,
    mode: EventMode,
    lag_window: tuple[int, int] | None,
) -> tuple[_Prepared, int, dict[str, dict[str, Any]], CompoundEventSet]:
    """Check the eager input and the two-source request, then validate the options on an empty match."""
    if not isinstance(hazards, xr.Dataset):
        raise InputTypeError(
            "hazards must be a prepared xarray.Dataset.", expected_type=xr.Dataset, actual_type=type(hazards)
        )
    # check every variable before reading values, including auxiliary coordinates and mixed backends
    if any(variable.chunks is not None for variable in hazards.variables.values()):
        raise InvalidArgumentError(
            "Event catalogs require eager input. Select a bounded subset, then call .compute() explicitly before "
            "compound_events_xarray(); this function never computes or loads a Dask-backed variable or coordinate.",
            argument_name="hazards",
            valid_values="an in-memory prepared Dataset",
        )
    prepared = _prepare(hazards)
    try:
        json.dumps(prepared.metadata, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise InvalidArgumentError(
            "hazards.attrs['compound_manifest'] must be a JSON object of finite numbers, strings, booleans, null, "
            "lists, and mappings, as compound.harmonize() writes it.",
            argument_name="hazards",
        ) from exc
    if len(set(prepared.names)) != len(prepared.names):
        repeated = sorted(name for name, count in Counter(prepared.names).items() if count > 1)
        raise InvalidArgumentError(
            f"hazards['hazard'] labels must be unique so each selected name identifies one source; {repeated} "
            "appears more than once.",
            argument_name="hazards",
            valid_values="distinct hazard coordinate labels",
        )
    for role, name in (("first", first), ("second", second)):
        if not isinstance(name, str) or name not in prepared.names:
            raise InvalidArgumentError(
                f"{role} must name one prepared hazard in {list(prepared.names)}.",
                argument_name=role,
                argument_value=repr(name),
                valid_values=repr(prepared.names),
            )
    if first == second:
        raise InvalidArgumentError("first and second must name different hazards.", argument_name="second")
    # validate a two-source stand-in: the statistics' two-or-three-source limit does not apply here,
    # because the catalog reads only the two selected hazards from a possibly larger prepared Dataset
    _validate_hazards(
        AlignedHazards(
            (first, second),
            prepared.day,
            cast(Cadence, prepared.cadence),
            np.zeros((2, prepared.day.size, 0), dtype=np.bool_),
            np.zeros((2, prepared.day.size, 0), dtype=np.bool_),
            prepared.metadata,
        )
    )
    _, duration, _ = _resolve_options(0.5, "above", min_duration, 0)
    source_metadata: dict[str, dict[str, Any]] = {
        role: {"name": name, "threshold": 0.5, "direction": "above", "min_duration": duration, "max_gap": 0}
        for role, name in (("first", first), ("second", second))
    }
    # the empty match validates dates, mode, and lag options even for a zero-cell subset
    empty_runs = identify_runs([], threshold=0.5, direction="above", min_duration=duration)
    plan = _compound_events(
        empty_runs,
        empty_runs,
        time=prepared.day,
        valid=np.zeros(prepared.day.size, dtype=bool),
        cadence=prepared.cadence,
        mode=mode,
        lag_window=lag_window,
        source_metadata=source_metadata,
    )
    return prepared, duration, source_metadata, plan


def _cell_coords(hazards: xr.Dataset, prepared: _Prepared) -> tuple[tuple[str, ...], dict[str, Any], int]:
    """Return the spatial grid, the cell coordinate table, and the flattened cell count."""
    grid = tuple(str(dim) for dim in prepared.exceedance.dims if dim not in ("hazard", "time"))
    clashes = _RESERVED.intersection(grid)
    if clashes:
        raise DimensionMismatchError(
            "Spatial dimension names collide with event catalog fields; rename those dimensions first.",
            expected_dims="spatial dimensions distinct from catalog fields",
            actual_dims=grid,
            reason="reserved_dimension_name",
        )
    shape = tuple(prepared.exceedance.sizes[dim] for dim in grid)
    cells = int(np.prod(shape, dtype=np.int64))
    lookup = np.unravel_index(np.arange(cells), shape) if grid else ()
    coords: dict[str, Any] = {"cell": np.arange(cells, dtype=np.int64)}
    for dim, indices in zip(grid, lookup, strict=True):
        values = np.asarray(hazards.indexes[dim].values)
        if values.dtype.kind == "O" and all(isinstance(value, str) for value in values):
            values = values.astype(str)
        if values.dtype.kind not in "biufUSM":
            raise InputTypeError(
                "Cell labels must be integer, float, bool, string, bytes, or datetime64 values; object, timedelta, "
                "and MultiIndex labels are unsupported.",
                expected_type=np.ndarray,
                actual_type=values.dtype.type,
            )
        attrs = {key: value for key, value in hazards[dim].attrs.items() if key not in _DROPPED_ATTRS}
        coords[dim] = ("cell", values[indices], attrs)
    return grid, coords, cells


def _event_variables(columns: dict[str, list[np.ndarray]], plan: CompoundEventSet) -> dict[str, Any]:
    """Concatenate the per-cell event columns, falling back to the empty match for absent dtypes."""
    variables: dict[str, Any] = {}
    for name, parts in columns.items():
        empty = np.empty(0, dtype=np.int64) if name == "cell_index" else getattr(plan, name)
        values = np.concatenate(parts) if parts else empty
        attrs = _count(_LONG_NAMES[name], _descriptions(name))
        if name.endswith("censored"):
            values = values.astype(np.int8)
            attrs.update(flag_values=np.array([0, 1], dtype=np.int8), flag_meanings="not_censored censored")
        variables[name] = ("event", values, attrs)
    return variables


def _descriptions(name: str) -> str:
    """Describe units and index scope without claiming a CF feature type or joint severity."""
    if name == "cell_index":
        return "Zero-based lookup into the cell coordinate table."
    if name.endswith("run_id"):
        return "Zero-based source-run position within this cell after min_duration filtering."
    if name.endswith("censored"):
        return "The source boundary touches the record edge or a step unknown in either selected source."
    if name in ("start_index", "end_index"):
        return "Zero-based inclusive reported boundary on the prepared time axis."
    if name == "overlap_duration":
        return "Shared cadence steps (days or calendar months); zero for sequential pairs."
    if name == "lag":
        return "Second onset minus first end in cadence steps (days or calendar months); adjacency is one; concurrent is NaN."
    return {
        "n_valid": "Steps with both selected hazards observed in this cell.",
        "n_first_runs": "Retained first-source runs on shared validity after min_duration filtering.",
        "n_second_runs": "Retained second-source runs on shared validity after min_duration filtering.",
        "n_pairs": "Matched source-run pairs in this cell; not independent observations.",
        "n_unknown_bridges": "Sequential candidate pairs rejected because their separating interval crosses an unknown step.",
    }[name]


def compound_events_xarray(
    hazards: xr.Dataset,
    *,
    first: str,
    second: str,
    mode: EventMode,
    lag_window: tuple[int, int] | None,
    min_duration: int,
) -> xr.Dataset:
    """Apply the common beta warning and lifecycle policy to the eager catalog route."""
    return _route(
        "compound_events_xarray",
        {"mode": mode},
        lambda queue: _catalog(
            hazards, first=first, second=second, mode=mode, lag_window=lag_window, min_duration=min_duration
        ),
        "first_run_id",
    )
