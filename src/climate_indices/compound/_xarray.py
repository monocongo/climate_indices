"""Beta xarray Dataset routes for the compound statistics (ADR-0022, ADR-0012).

Every route runs the private NumPy core on one spatial block at a time through a single
shape-changing :func:`xarray.apply_ufunc`, so there is one scientific implementation and
this module owns only labels, metadata, and the Dask contract:

- every option, coordinate, backend, and chunk shape is validated before a graph is built;
- a Dask input must keep ``time`` (and ``hazard``) in one chunk and every spatial dimension
  identically chunked across inputs; nothing is rechunked, broadcast, loaded, or computed;
- value-dependent errors surface inside the numerical task, so under Dask they appear when
  computation starts, and the NumPy support warnings are left to ``quality_flags``;
- the eager manifest comes from running the NumPy core on a zero-cell stand-in, which also
  raises every NumPy validation error at the call site.
"""

from __future__ import annotations

import copy
import json
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.compound._bootstrap import (
    _FLAGS as _BOOTSTRAP_FLAGS,
)
from climate_indices.compound._bootstrap import (
    Alternative,
    Statistic,
    _bootstrap,
    _checked_options,
)
from climate_indices.compound._harmonize import (
    _CADENCES,
    _MISSING,
    Aggregation,
    AlignedHazards,
    Cadence,
    InputScale,
    Join,
    _check_choice,
    _check_keys,
    _harmonize,
    _replay,
)
from climate_indices.compound._statistics import (
    _FLAG_MEANINGS,
    _GROUPINGS,
    GroupBy,
    _check_lag_window,
    _check_min_samples,
    _conditional_probability,
    _joint_statistics,
)
from climate_indices.exceptions import (
    BetaFeatureWarning,
    CoordinateValidationError,
    DimensionMismatchError,
    InputTypeError,
    InvalidArgumentError,
)
from climate_indices.logging_config import get_logger, log_calculation_failure
from climate_indices.runs import Direction
from climate_indices.validation import validate_dask_chunks, validate_time_dimension
from climate_indices.xarray_adapter import _validate_supported_calendar, build_output_attrs

_logger = get_logger(__name__)

_TIME, _HAZARD, _MONTH, _SOURCE = "time", "hazard", "month", "source"
_MANIFEST = "compound_manifest"
_BETA_MESSAGE = "The compound xarray route is beta; the NumPy API is stable (ADR-0012)."
# the prepared Dataset's dimension names are part of its contract, so no spatial dimension may reuse them
_RESERVED = frozenset({_TIME, _HAZARD, _MONTH, _SOURCE})
_DAY_UNIT = "datetime64[D]"
# xarray holds dates in seconds at the finest, so datetime outputs are declared and returned in that unit
_RESULT_DATE_UNIT = "datetime64[s]"


def _route(
    index_type: str,
    fields: Mapping[str, Any],
    build: Callable[[list[Warning]], xr.Dataset],
    shape_of: str,
) -> xr.Dataset:
    """Run one beta route: lifecycle logs, one ``BetaFeatureWarning``, and the NumPy warnings at the caller.

    ``build`` receives a list that collects every warning the NumPy cores raise, and each is replayed
    here, attributed to the caller of the public function. xarray adds frames of its own between a core
    and the caller, so no fixed stack level could reach the caller from inside the core, and capturing the
    warnings with ``warnings.catch_warnings`` instead would be unsafe across threads. Lazy completion means
    the graph was built, not that any worker has computed; a task-time failure is therefore never a
    ``calculation_failed`` event.
    """
    log = _logger.bind(index_type=index_type, route="xarray", **fields)
    log.info("calculation_started")
    t0 = perf_counter()
    queued: list[Warning] = []
    try:
        # the frames to skip are this function, the route function, and the public function
        warnings.warn(_BETA_MESSAGE, BetaFeatureWarning, stacklevel=4)
        result = build(queued)
        _replay(queued, 4)
    except Exception as exc:
        log_calculation_failure(log, exc)
        raise
    shape = result[shape_of]
    log.info(
        "calculation_completed",
        duration_ms=round((perf_counter() - t0) * 1000.0, 2),
        output_shape=shape.shape,
        # Dataset.chunks lists dimensions only, so it is empty when every variable is 0-d
        evaluation="eager" if shape.chunks is None else "graph_construction",
    )
    return result


def _check_backends(arrays: Mapping[str, xr.DataArray]) -> bool:
    """Return whether every array is Dask-backed; mixed backends are rejected, never reconciled."""
    lazy = {name: array.chunks is not None for name, array in arrays.items()}
    if len(set(lazy.values())) > 1:
        raise InvalidArgumentError(
            f"The inputs mix Dask-backed {[name for name, flag in lazy.items() if flag]} and in-memory "
            f"{[name for name, flag in lazy.items() if not flag]} arrays. Chunk the in-memory arrays with "
            ".chunk() or compute the Dask arrays explicitly; nothing is converted automatically.",
            argument_name="inputs",
            valid_values="all Dask-backed or all in memory",
        )
    return all(lazy.values())


def _validate_grid(arrays: Mapping[str, xr.DataArray], *, core: tuple[str, ...], lazy: bool) -> tuple[str, ...]:
    """Require one labeled, identically chunked spatial grid and return its dimension names.

    ``core`` names the dimensions each array must carry whole (time, and hazard for statistics);
    every other dimension is spatial. A Dask input must hold each core dimension in one chunk.
    """
    reference = next(iter(arrays.values()))
    grid = tuple(str(dim) for dim in reference.dims if dim not in core)
    clashes = sorted(_RESERVED & set(grid))
    if clashes:
        raise DimensionMismatchError(
            f"Spatial dimensions {clashes} collide with the result's reserved dimension names {sorted(_RESERVED)}.",
            expected_dims="no spatial dimension named hazard, time, month, or source",
            actual_dims=grid,
            coordinate_name=clashes[0],
            reason="reserved_dimension_name",
        )
    for name, array in arrays.items():
        for dim in core:
            if dim not in array.dims:
                raise DimensionMismatchError(
                    f"{name!r} has no {dim!r} dimension; its dimensions are {list(array.dims)}.",
                    expected_dims=core,
                    actual_dims=tuple(array.dims),
                    coordinate_name=dim,
                    reason="missing_dimension",
                )
        dims = tuple(str(dim) for dim in array.dims if dim not in core)
        if set(dims) != set(grid):
            raise DimensionMismatchError(
                f"{name!r} has spatial dimensions {list(dims)} but the first input has {list(grid)}; "
                "broadcasting a point across a grid is never implicit.",
                expected_dims=grid,
                actual_dims=dims,
                reason="mismatched_spatial_dimensions",
            )
        for dim in dims:
            if dim not in array.indexes:
                raise CoordinateValidationError(
                    f"Spatial dimension {dim!r} of {name!r} needs coordinate labels; a dimension-only grid "
                    "cannot be checked against the other inputs.",
                    coordinate_name=dim,
                    reason="missing_coordinate_labels",
                )
            if not array.indexes[dim].equals(reference.indexes[dim]):
                raise CoordinateValidationError(
                    f"Spatial coordinate {dim!r} of {name!r} differs from the first input; grids are never "
                    "regridded, subset, or aligned implicitly.",
                    coordinate_name=dim,
                    reason="mismatched_coordinates",
                )
        if lazy:
            for dim in core:
                validate_dask_chunks(array, dim)
            for dim in dims:
                if array.chunks[array.dims.index(dim)] != reference.chunks[reference.dims.index(dim)]:  # type: ignore[index]
                    raise CoordinateValidationError(
                        f"Spatial dimension {dim!r} of {name!r} is chunked differently from the first input; "
                        f"rechunk explicitly with .chunk({{'{dim}': ...}}) so every input shares one boundary set. "
                        "Nothing is rechunked automatically.",
                        coordinate_name=dim,
                        reason="incompatible_spatial_chunks",
                    )
    return grid


def _time_values(name: str, array: xr.DataArray, time_dim: str) -> npt.NDArray[np.datetime64]:
    """Return one array's eager Gregorian time labels; a dimension-only time carries no calendar."""
    if time_dim not in array.indexes:
        raise CoordinateValidationError(
            f"{name!r} needs a labeled {time_dim!r} coordinate; a dimension-only time is rejected because no "
            "cadence or calendar can be read from it.",
            coordinate_name=time_dim,
            reason="missing_coordinate_labels",
        )
    if array.coords[time_dim].dtype.kind in "iuf":
        raise CoordinateValidationError(
            f"{name!r} has numeric {time_dim!r} values; the compound routes need decoded datetime64 labels. "
            "Open files with decode_times=True, and note that cftime and non-Gregorian calendars are not supported.",
            coordinate_name=time_dim,
            reason="undecoded_time",
        )
    _validate_supported_calendar(array.coords[time_dim])
    return np.asarray(array.indexes[time_dim].values)


def _registry(key: str, **extra: Any) -> dict[str, Any]:
    """Return one registry entry's attributes, plus bounds defined beside the adapter."""
    return {**CF_METADATA[key], **extra}


def _count(long_name: str, description: str, units: str = "1") -> dict[str, Any]:
    """Return the attributes of one count, flag, or audit variable; none has a CF standard name."""
    return {"long_name": long_name, "units": units, "description": description}


def _flags(meanings: Mapping[str, int]) -> dict[str, Any]:
    """Return CF bit-field attributes for a quality-flag variable."""
    return {
        "long_name": "Compound Quality Flags",
        "units": "1",
        "flag_masks": np.array(list(meanings.values()), dtype=np.int64),
        "flag_meanings": " ".join(meanings),
        "description": "Bitwise OR of the named conditions; zero means none applies.",
    }


def _dataset_attrs(label: str, **common: str) -> dict[str, Any]:
    """Return version, history, and scalar choices, inheriting nothing from a source array."""
    return {**build_output_attrs(xr.DataArray(0), index_name=label), **common}


# --- harmonize -----------------------------------------------------------------------------------


def _harmonize_block(
    *blocks: npt.NDArray[Any],
    names: tuple[str, ...],
    mask_names: tuple[str, ...],
    times: Mapping[str, Any],
    **declared: Any,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Run the NumPy harmonization on one block, with time moved from last to first and back."""
    values = {name: np.moveaxis(block, -1, 0) for name, block in zip(names, blocks[: len(names)], strict=True)}
    masks = {name: np.moveaxis(block, -1, 0) for name, block in zip(mask_names, blocks[len(names) :], strict=True)}
    aligned = _harmonize(values, time=times, validity=masks or None, spatial_time_major=True, warn=None, **declared)
    # the NumPy result is read-only; a Dataset variable should be an ordinary writable array
    return (
        np.array(np.moveaxis(aligned.exceedance, (0, 1), (-2, -1))),
        np.array(np.moveaxis(aligned.valid, (0, 1), (-2, -1))),
    )


def _harmonize_dataarrays(
    inputs: Mapping[str, Any],
    *,
    time: Any,
    cadence: Mapping[str, Cadence],
    input_scale: Mapping[str, InputScale],
    thresholds: Mapping[str, float],
    directions: Mapping[str, Direction],
    target_cadence: Cadence,
    aggregation: Mapping[str, Aggregation] | None,
    validity: Mapping[str, Any] | None,
    join: Join,
    source_metadata: Mapping[str, Mapping[str, Any]] | None,
    spatial_time_major: bool,
    time_dim: str,
    queue: list[Warning],
) -> xr.Dataset:
    """Validate, plan eagerly from labels and dtypes, and build the per-block harmonization."""
    if not isinstance(time_dim, str):
        raise InvalidArgumentError(
            "time_dim must be a string.", argument_name="time_dim", argument_value=repr(time_dim)
        )
    names = tuple(inputs)
    foreign = [name for name, array in inputs.items() if not isinstance(array, xr.DataArray)]
    if foreign:
        raise InputTypeError(
            f"inputs mixes xarray.DataArray with other types ({foreign}); pass all DataArrays or all NumPy arrays.",
            expected_type=xr.DataArray,
            actual_type=type(inputs[foreign[0]]),
        )
    if time is not _MISSING or spatial_time_major:
        raise InvalidArgumentError(
            "time and spatial_time_major apply only to NumPy inputs; DataArray inputs carry their dates in the "
            f"{time_dim!r} coordinate and their layout in named dimensions.",
            argument_name="time" if time is not _MISSING else "spatial_time_major",
        )
    masks = _check_keys("validity", {} if validity is None else validity, names, subset=True)
    foreign_masks = [name for name, mask in masks.items() if not isinstance(mask, xr.DataArray)]
    if foreign_masks:
        raise InputTypeError(
            f"validity mixes xarray.DataArray inputs with other types ({foreign_masks}).",
            expected_type=xr.DataArray,
            actual_type=type(masks[foreign_masks[0]]),
        )
    arrays = {**inputs, **{f"validity[{name!r}]": mask for name, mask in masks.items()}}
    lazy = _check_backends(arrays)
    for array in arrays.values():
        validate_time_dimension(array, time_dim)
    _validate_grid(arrays, core=(time_dim,), lazy=lazy)

    times = {name: _time_values(name, array, time_dim) for name, array in inputs.items()}
    for name, mask in masks.items():
        if not np.array_equal(_time_values(f"validity[{name!r}]", mask, time_dim), times[name]):
            raise CoordinateValidationError(
                f"validity[{name!r}] must share the time coordinate of inputs[{name!r}].",
                coordinate_name=time_dim,
                reason="mismatched_coordinates",
            )

    # the manifest and the harmonization warnings depend on dates, dtypes, and declarations, never on values
    plan = _harmonize(
        {name: np.zeros(array.sizes[time_dim], dtype=array.dtype) for name, array in inputs.items()},
        time=times,
        cadence=cadence,
        input_scale=input_scale,
        thresholds=thresholds,
        directions=directions,
        target_cadence=target_cadence,
        aggregation=aggregation,
        validity={name: np.zeros(mask.sizes[time_dim], dtype=mask.dtype) for name, mask in masks.items()} or None,
        join=join,
        source_metadata=source_metadata,
        spatial_time_major=False,
        warn=queue,
    )
    for name, array in inputs.items():
        declared = array.attrs.get("output_scale")
        if input_scale[name] == "probability" and declared not in (None, "probability"):
            raise InvalidArgumentError(
                f"inputs[{name!r}] declares output_scale={declared!r} but is declared a probability: a z-score "
                'inside [0, 1] cannot be detected numerically. Request output_scale="probability" upstream.',
                argument_name="input_scale",
                argument_value=repr(input_scale[name]),
                valid_values="probability inputs with output_scale='probability' or no output_scale",
            )

    declared_choices = copy.deepcopy(
        {
            "cadence": cadence,
            "input_scale": input_scale,
            "thresholds": thresholds,
            "directions": directions,
            "target_cadence": target_cadence,
            "aggregation": aggregation,
            "join": join,
            "source_metadata": source_metadata,
        }
    )
    mask_names = tuple(masks)
    # apply_ufunc merges, and so compares, the coordinates of every input; a Dask-backed auxiliary
    # coordinate would be computed while the graph is built, and conflicting ones would be dropped
    # silently. Index coordinates (grid labels and their attributes) are all a result needs.
    bare = [array.reset_coords(drop=True) for array in (*inputs.values(), *masks.values())]
    exceedance, valid = xr.apply_ufunc(
        _harmonize_block,
        *bare,
        input_core_dims=[[time_dim]] * (len(names) + len(mask_names)),
        output_core_dims=[[_HAZARD, _TIME], [_HAZARD, _TIME]],
        # sources may cover different dates; the retained range is the plan's, not any input's
        exclude_dims={time_dim},
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {_HAZARD: len(names), _TIME: int(plan.time.shape[0])}},
        output_dtypes=[np.bool_, np.bool_],
        # keeps the grid coordinates' CF attributes; every variable's attributes are replaced below
        keep_attrs=True,
        kwargs={"names": names, "mask_names": mask_names, "times": times, **declared_choices},
    )
    result: xr.Dataset = xr.Dataset({"exceedance": exceedance, "valid": valid}).transpose(_HAZARD, _TIME, ...)
    result = result.assign_coords(
        {
            _HAZARD: (_HAZARD, list(names), {"long_name": "Hazard source name"}),
            _TIME: (_TIME, plan.time, {"long_name": "Observed time label"}),
        }
    )
    result["exceedance"].attrs = _count(
        "Compound Hazard Exceedance",
        "True where a valid observation meets the declared hazard condition; False wherever valid is False.",
    )
    result["valid"].attrs = _count(
        "Compound Hazard Validity",
        "True where the hazard was observed; an unobserved step is invalid, not a non-exceedance.",
    )
    result.attrs = _dataset_attrs("Compound hazard harmonization", cadence=plan.cadence)
    result.attrs[_MANIFEST] = json.dumps(plan.metadata)
    return result


def harmonize(
    inputs: Mapping[str, Any],
    *,
    time: Any,
    cadence: Mapping[str, Cadence],
    input_scale: Mapping[str, InputScale],
    thresholds: Mapping[str, float],
    directions: Mapping[str, Direction],
    target_cadence: Cadence,
    aggregation: Mapping[str, Aggregation] | None,
    validity: Mapping[str, Any] | None,
    join: Join,
    source_metadata: Mapping[str, Mapping[str, Any]] | None,
    spatial_time_major: bool,
    time_dim: str,
) -> xr.Dataset:
    """Prepare labeled hazard and validity arrays without evaluating a grid; see :func:`harmonize`."""
    return _route(
        "compound_harmonize",
        {"n_sources": len(inputs), "target_cadence": target_cadence, "join": join},
        lambda queue: _harmonize_dataarrays(
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
            time_dim=time_dim,
            queue=queue,
        ),
        "exceedance",
    )


# --- statistics ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Prepared:
    """A validated prepared Dataset: its labels, manifest, and the two arrays the cores read."""

    names: tuple[str, ...]
    day: npt.NDArray[np.datetime64]
    cadence: str
    metadata: dict[str, Any]
    exceedance: xr.DataArray
    valid: xr.DataArray
    lazy: bool

    def standin(self) -> AlignedHazards:
        """Return a zero-cell :class:`AlignedHazards`: the NumPy core validates and describes it without any data."""
        empty = np.zeros((len(self.names), self.day.shape[0], 0), dtype=np.bool_)
        return AlignedHazards(self.names, self.day, self.cadence, empty, empty, self.metadata)  # type: ignore[arg-type]


def _prepare(dataset: xr.Dataset) -> _Prepared:
    """Validate a prepared Dataset's variables, labels, backends, chunks, and embedded manifest."""
    missing = [name for name in ("exceedance", "valid") if name not in dataset.data_vars]
    if missing:
        raise InvalidArgumentError(
            f"hazards must be a Dataset from compound.harmonize(); it has no {missing} variable.",
            argument_name="hazards",
            valid_values="a Dataset with exceedance and valid variables",
        )
    arrays = {"exceedance": dataset["exceedance"], "valid": dataset["valid"]}
    for name, array in arrays.items():
        if array.dtype != np.bool_:
            raise InputTypeError(
                f"hazards[{name!r}] must have Boolean dtype, not {array.dtype}.",
                expected_type=np.bool_,
                actual_type=array.dtype.type,
            )
    lazy = _check_backends(arrays)
    for array in arrays.values():
        validate_time_dimension(array, _TIME)
    _validate_grid(arrays, core=(_HAZARD, _TIME), lazy=lazy)
    if _HAZARD not in dataset.indexes:
        raise CoordinateValidationError(
            "hazards needs labeled hazard names in its 'hazard' coordinate.",
            coordinate_name=_HAZARD,
            reason="missing_coordinate_labels",
        )
    names = tuple(str(name) for name in dataset.indexes[_HAZARD].values)
    time = _time_values("hazards", arrays["exceedance"], _TIME)
    day = time.astype(_DAY_UNIT)
    if np.any(~np.isnat(time) & (time != day)):
        raise CoordinateValidationError(
            "hazards.time must label whole civil days; a prepared Dataset holds one label per step.",
            coordinate_name=_TIME,
            reason="sub_daily_labels",
        )
    cadence = dataset.attrs.get("cadence")
    _check_choice("hazards.attrs['cadence']", cadence, _CADENCES)
    metadata: dict[str, Any] = {}
    if _MANIFEST in dataset.attrs:
        try:
            metadata = json.loads(dataset.attrs[_MANIFEST])
            if not isinstance(metadata, dict):
                raise TypeError(type(metadata).__name__)
        except (TypeError, ValueError) as exc:
            raise InvalidArgumentError(
                f"hazards.attrs[{_MANIFEST!r}] must be a JSON object, as compound.harmonize() writes it.",
                argument_name="hazards",
            ) from exc
    # auxiliary and scalar coordinates are dropped for the same reason as in harmonize()
    exceedance, valid = (arrays[name].reset_coords(drop=True) for name in ("exceedance", "valid"))
    return _Prepared(names, day, str(cadence), metadata, exceedance, valid, lazy)


@dataclass(frozen=True)
class _Field:
    """One NumPy result field: its output dimensions, dtype, and Dataset attributes."""

    name: str
    attrs: dict[str, Any]
    lead: tuple[str, ...] = ()  # core dimensions before month, in NumPy order
    dtype: Any = np.float64
    monthly: bool = True


def _statistics_block(
    exceedance: npt.NDArray[np.bool_],
    valid: npt.NDArray[np.bool_],
    *,
    compute: Callable[..., Any],
    names: tuple[str, ...],
    time: npt.NDArray[np.datetime64],
    cadence: str,
    metadata: Mapping[str, Any],
    groupby: str,
    fields: tuple[tuple[str, int], ...],
    **options: Any,
) -> tuple[npt.NDArray[Any], ...]:
    """Run one NumPy core on a block; ``fields`` names each result array and its leading group axes."""
    hazards = AlignedHazards(
        names,
        time,
        cadence,  # type: ignore[arg-type]
        np.moveaxis(exceedance, (-2, -1), (0, 1)),
        np.moveaxis(valid, (-2, -1), (0, 1)),
        metadata,
    )
    result = compute(hazards, groupby=groupby, **options)
    # xarray wants the core dimensions last; the NumPy layout has them first
    arrays = (
        np.array(np.moveaxis(getattr(result, name), tuple(range(lead)), tuple(range(-lead, 0))))
        for name, lead in fields
    )
    return tuple(array.astype(_RESULT_DATE_UNIT) if array.dtype.kind == "M" else array for array in arrays)


def _statistics_dataset(
    prepared: _Prepared,
    compute: Callable[..., Any],
    fields: tuple[_Field, ...],
    *,
    groupby: GroupBy,
    manifest: Mapping[str, Any],
    hazard_names: tuple[str, ...],
    label: str,
    options: Mapping[str, Any],
    queue: list[Warning],
) -> xr.Dataset:
    """Apply one NumPy core across the grid and label its results like the NumPy dataclass.

    In-memory input runs the core inline, so its support warnings are appended to ``queue``; a lazy graph runs
    them on worker threads, where they would repeat per tile, and drops them for ``quality_flags`` to carry.
    """
    grouped = groupby == "month"
    leads = [(*field.lead, *((_MONTH,) if grouped and field.monthly else ())) for field in fields]
    sizes = {_MONTH: 12, _HAZARD: len(hazard_names)}
    outputs = xr.apply_ufunc(
        _statistics_block,
        prepared.exceedance.rename({_HAZARD: _SOURCE}),
        prepared.valid.rename({_HAZARD: _SOURCE}),
        input_core_dims=[[_SOURCE, _TIME]] * 2,
        output_core_dims=[list(lead) for lead in leads],
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {dim: sizes[dim] for lead in leads for dim in lead}},
        output_dtypes=[field.dtype for field in fields],
        keep_attrs=True,
        kwargs={
            "compute": compute,
            "names": prepared.names,
            "time": prepared.day,
            "cadence": prepared.cadence,
            "metadata": prepared.metadata,
            "groupby": groupby,
            "fields": tuple((field.name, len(lead)) for field, lead in zip(fields, leads, strict=True)),
            # the lazy route leaves the value-dependent support conditions to quality_flags
            "warn": None if prepared.lazy else queue,
            **options,
        },
    )
    variables: dict[str, xr.DataArray] = {}
    for field, lead, output in zip(fields, leads, outputs, strict=True):
        variable = output.transpose(*lead, ...)
        # replace, never merge; the module-level attrs hold arrays (flag_masks), so each Dataset copies them
        variable.attrs = copy.deepcopy(field.attrs)
        variables[field.name] = variable
    coords: dict[str, Any] = {}
    if grouped:
        coords[_MONTH] = (_MONTH, np.arange(1, 13), {"long_name": "Calendar month, 1 = January"})
    if any(_HAZARD in lead for lead in leads):
        coords[_HAZARD] = (_HAZARD, list(hazard_names), {"long_name": "Hazard source name"})
    result: xr.Dataset = xr.Dataset(variables).assign_coords(coords)
    result.attrs = _dataset_attrs(label, cadence=prepared.cadence, groupby=groupby)
    result.attrs[_MANIFEST] = json.dumps(manifest)
    return result


_JOINT_FIELDS = (
    _Field(
        "n_valid", _count("Compound Eligible Step Count", "Steps where every selected source is valid."), dtype=np.int64
    ),
    _Field(
        "n_joint",
        _count("Compound Joint Step Count", "Eligible steps where every selected hazard occurs."),
        dtype=np.int64,
    ),
    _Field(
        "n_marginal",
        _count("Compound Marginal Step Count", "Eligible steps where each source hazard occurs."),
        (_HAZARD,),
        np.int64,
    ),
    _Field(
        "marginal_frequency",
        _registry("compound_marginal_probability", valid_min=0.0, valid_max=1.0),
        (_HAZARD,),
    ),
    _Field("joint_probability", _registry("compound_joint_probability", valid_min=0.0, valid_max=1.0)),
    _Field("independent_probability", _registry("compound_independent_probability", valid_min=0.0, valid_max=1.0)),
    _Field(
        "likelihood_multiplication_factor",
        _registry("compound_likelihood_multiplication_factor", valid_min=0.0),
    ),
    _Field("quality_flags", _flags(_FLAG_MEANINGS), dtype=np.int64),
)

_CONDITIONAL_FIELDS = (
    _Field(
        "n_anchors", _count("Compound Eligible Anchor Count", "Anchors with a complete target window."), dtype=np.int64
    ),
    _Field(
        "n_positive",
        _count("Compound Positive Anchor Count", "Eligible anchors where the given hazard occurs."),
        dtype=np.int64,
    ),
    _Field(
        "n_response",
        _count("Compound Response Anchor Count", "Eligible anchors whose window holds a target occurrence."),
        dtype=np.int64,
    ),
    _Field(
        "n_success",
        _count("Compound Success Anchor Count", "Positive anchors whose window holds a target occurrence."),
        dtype=np.int64,
    ),
    _Field("conditional_probability", _registry("compound_conditional_probability", valid_min=0.0, valid_max=1.0)),
    _Field(
        "response_window_frequency",
        _registry("compound_response_window_probability", valid_min=0.0, valid_max=1.0),
    ),
    _Field("quality_flags", _flags(_FLAG_MEANINGS), dtype=np.int64),
)

_ESTIMATE_KEY = {
    "joint_probability": "compound_joint_probability",
    "likelihood_multiplication_factor": "compound_likelihood_multiplication_factor",
    "conditional_probability": "compound_conditional_probability",
}


def _bootstrap_fields(statistic: str) -> tuple[_Field, ...]:
    """Return the bootstrap result fields, with the estimate described by the selected statistic."""
    bounds = (
        {"valid_min": 0.0} if statistic == "likelihood_multiplication_factor" else {"valid_min": 0.0, "valid_max": 1.0}
    )
    estimate = _registry(_ESTIMATE_KEY[statistic], **bounds)

    def variant(suffix: str, role: str) -> dict[str, Any]:
        """Describe one output by its role, then by the statistic it is a value or a bound of."""
        return {
            **estimate,
            "long_name": f"{estimate['long_name']} {suffix}",
            "description": f"{role} {estimate['description']}",
        }

    return (
        _Field(
            "estimate",
            variant(
                "on the Inference Segment",
                "Estimate on the inference segment, each cell's longest run of complete calendar years. The statistic:",
            ),
        ),
        _Field(
            "full_record_estimate",
            variant(
                "on the Full Record",
                "Estimate on every eligible original observation, including partial years. The statistic:",
            ),
        ),
        _Field(
            "confidence_low",
            variant(
                "Lower Confidence Bound",
                "Lower bound of the paired-block percentile interval for the statistic:",
            ),
        ),
        _Field(
            "confidence_high",
            variant(
                "Upper Confidence Bound",
                "Upper bound of the paired-block percentile interval for the statistic:",
            ),
        ),
        _Field(
            "p_value",
            _count(
                "Compound Null-Test p-value",
                "Inclusive null-test p-value, (1 + extreme) / (1 + valid resamples); local and unadjusted, "
                "never field significance.",
            )
            | {"valid_min": 0.0, "valid_max": 1.0},
        ),
        _Field(
            "n_valid",
            _count("Compound Eligible Step Count", "Eligible steps on the inference segment."),
            dtype=np.int64,
        ),
        _Field(
            "n_marginal",
            _count(
                "Compound Channel Occurrence Count",
                "Occurrences per channel on the inference segment; for conditional inference the second "
                "channel is the target's response window.",
            ),
            (_HAZARD,),
            np.int64,
        ),
        _Field(
            "n_joint",
            _count("Compound Success Count", "Joint or conditional successes on the segment."),
            dtype=np.int64,
        ),
        _Field(
            "expected_null_successes",
            _count(
                "Compound Expected Null Successes", "Expected successes under independence, tested by the null guard."
            ),
        ),
        _Field(
            "n_interval_resamples",
            _count("Compound Interval Resample Count", "Valid paired resamples behind the interval."),
            dtype=np.int64,
        ),
        _Field(
            "n_null_resamples",
            _count("Compound Null Resample Count", "Valid independent resamples behind the p-value."),
            dtype=np.int64,
        ),
        _Field(
            "n_years",
            _count("Compound Inference Years", "Complete calendar years in the segment.", "year"),
            dtype=np.int64,
            monthly=False,
        ),
        _Field(
            "block_length_years",
            _count("Compound Block Length", "Moving-block length.", "year"),
            dtype=np.int64,
            monthly=False,
        ),
        _Field(
            "inference_start",
            {"long_name": "Compound Inference Start", "description": "First January of the segment; NaT if none."},
            dtype=np.dtype(_RESULT_DATE_UNIT),
            monthly=False,
        ),
        _Field(
            "inference_end",
            {"long_name": "Compound Inference End", "description": "Last December of the segment; NaT if none."},
            dtype=np.dtype(_RESULT_DATE_UNIT),
            monthly=False,
        ),
        _Field("quality_flags", _flags(_BOOTSTRAP_FLAGS), dtype=np.int64),
    )


def joint_statistics(hazards: xr.Dataset, *, groupby: GroupBy, min_samples: int) -> xr.Dataset:
    """Count hazards on one eligible support across a prepared Dataset; see :func:`joint_statistics`."""

    def build(queue: list[Warning]) -> xr.Dataset:
        prepared = _prepare(hazards)
        _check_choice("groupby", groupby, _GROUPINGS)
        minimum = _check_min_samples(min_samples)
        reference = _joint_statistics(prepared.standin(), groupby=groupby, min_samples=minimum, warn=None)
        return _statistics_dataset(
            prepared,
            _joint_statistics,
            _JOINT_FIELDS,
            groupby=groupby,
            manifest=reference.metadata,
            hazard_names=prepared.names,
            label="Compound joint statistics",
            options={"min_samples": minimum},
            queue=queue,
        )

    return _route(
        "compound_joint_statistics",
        {"n_sources": hazards.sizes.get(_HAZARD, 0), "groupby": groupby, "min_samples": min_samples},
        build,
        "n_valid",
    )


def conditional_probability(
    hazards: xr.Dataset, *, given: str, target: str, lag_window: tuple[int, int], groupby: GroupBy, min_samples: int
) -> xr.Dataset:
    """Condition one hazard window on another across a prepared Dataset; see :func:`conditional_probability`."""

    def build(queue: list[Warning]) -> xr.Dataset:
        prepared = _prepare(hazards)
        _check_choice("groupby", groupby, _GROUPINGS)
        window = _check_lag_window(lag_window)
        minimum = _check_min_samples(min_samples)
        options: dict[str, Any] = {"given": given, "target": target, "lag_window": window, "min_samples": minimum}
        reference = _conditional_probability(prepared.standin(), groupby=groupby, warn=None, **options)
        return _statistics_dataset(
            prepared,
            _conditional_probability,
            _CONDITIONAL_FIELDS,
            groupby=groupby,
            manifest=reference.metadata,
            hazard_names=prepared.names,
            label="Compound conditional probability",
            options=options,
            queue=queue,
        )

    return _route(
        "compound_conditional_probability",
        {
            "n_sources": hazards.sizes.get(_HAZARD, 0),
            "groupby": groupby,
            "lag_window": lag_window,
            "min_samples": min_samples,
        },
        build,
        "n_anchors",
    )


def bootstrap_dependence(
    hazards: xr.Dataset,
    *,
    statistic: Statistic,
    given: str | None,
    target: str | None,
    lag_window: tuple[int, int],
    groupby: GroupBy,
    block_length_years: int | None,
    n_resamples: int,
    confidence_level: float,
    alternative: Alternative,
    seed: int,
) -> xr.Dataset:
    """Estimate monthly uncertainty across a prepared Dataset; see :func:`bootstrap_dependence`."""

    def build(queue: list[Warning]) -> xr.Dataset:
        prepared = _prepare(hazards)
        window, level = _checked_options(
            statistic=statistic,
            groupby=groupby,
            alternative=alternative,
            lag_window=lag_window,
            block_length_years=block_length_years,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            seed=seed,
        )
        options: dict[str, Any] = {
            "statistic": statistic,
            "given": given,
            "target": target,
            "lag_window": window,
            "block_length_years": block_length_years,
            "n_resamples": n_resamples,
            "confidence_level": level,
            "alternative": alternative,
            "seed": seed,
        }
        reference = _bootstrap(prepared.standin(), groupby=groupby, warn=None, **options)
        return _statistics_dataset(
            prepared,
            _bootstrap,
            _bootstrap_fields(statistic),
            groupby=groupby,
            manifest=reference.metadata,
            hazard_names=tuple(reference.metadata["source_names"]),
            label="Compound bootstrap dependence",
            options=options,
            queue=queue,
        )

    return _route(
        "compound_bootstrap_dependence",
        {
            "statistic": statistic,
            "n_sources": hazards.sizes.get(_HAZARD, 0),
            "groupby": groupby,
            "seed": seed,
            "n_resamples": n_resamples,
            "block_length_years": block_length_years,
        },
        build,
        "estimate",
    )
