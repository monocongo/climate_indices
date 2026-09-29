"""One xarray adapter for the stateful daily recurrences (KBDI, CFFWIS, API).

KBDI, CFFWIS and the antecedent precipitation index each run a daily
recurrence over a ``(*spatial, time)`` array and carry typed state between
calls. Before this module each adapter hand-rolled the same pipeline: validate
every time axis, inner-join the inputs without silently dropping cells, read
the shared spatial topology, partition static state operands to the weather
chunks, drive :func:`xarray.apply_ufunc`, and restore the trimmed time
coordinate. The three copies had drifted -- KBDI broadcast the whole grid and
lost its time coordinate, the antecedent index and CFFWIS disagreed on how a
state operand reached a worker -- so the policy now lives here once.

The adapter is deliberately not the stateless :func:`xarray_adapter`: a
stateful recurrence must keep the whole ``time`` axis in one chunk (ADR-0003,
ADR-0006) and returns state fields alongside the values, neither of which the
stateless adapter expresses.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import xarray as xr

from climate_indices._units import _validate_daily_time_coordinate
from climate_indices.exceptions import CoordinateValidationError, InputAlignmentWarning
from climate_indices.validation import validate_dask_chunks, validate_time_dimension, validate_time_monotonicity

NamedInput = tuple[str, xr.DataArray]
ExtraInput = tuple[xr.DataArray, "str | None"]


@dataclass(frozen=True)
class StatefulAlignment:
    """The aligned topology one recurrence runs over.

    ``spatial_dims``/``spatial_shape`` come from the union of the primary
    inputs' dimensions (the same union ``xr.broadcast`` would produce) without
    broadcasting any data; ``spatial_chunks`` is one of the primary inputs'
    spatial chunkings, so a static state operand can be partitioned alongside
    the weather tiles without adding new Dask boundaries. ``aligned_inputs`` are
    the primary inputs after the shared inner join, so a callback that resolves
    a coordinate from them sees the same labels in the same order.
    """

    spatial_dims: tuple[str, ...]
    spatial_shape: tuple[int, ...]
    spatial_chunks: dict[str, tuple[int, ...]]
    output_dims: tuple[str, ...]
    time_source: xr.DataArray | None
    time_length: int
    time_dim: str
    aligned_inputs: tuple[xr.DataArray, ...]

    @property
    def internal_spatial_shape(self) -> tuple[int, ...]:
        """``(1,)`` for a time-only input, so state validation sees one cell."""
        return self.spatial_shape or (1,)


@dataclass(frozen=True)
class StatefulAdapterOutput:
    """The raw apply_ufunc outputs plus the topology they were computed on."""

    outputs: tuple[xr.DataArray, ...]
    alignment: StatefulAlignment


def broadcast_topology(inputs: Sequence[xr.DataArray]) -> tuple[tuple[str, ...], dict[str, int]]:
    """Return the dims and sizes ``xr.broadcast`` would produce, without broadcasting data.

    The broadcast dims come in order of appearance across the inputs, exactly
    as ``xarray.core.variable._unified_dims`` orders them; ``xr.align`` has
    already matched the sizes of the shared dims.
    """
    dims: list[str] = []
    sizes: dict[str, int] = {}
    for data in inputs:
        for dim in data.dims:
            name = str(dim)
            if name not in sizes:
                dims.append(name)
                sizes[name] = data.sizes[dim]
    return tuple(dims), sizes


def spatial_chunk_targets(inputs: Sequence[xr.DataArray], spatial_dims: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    """A shared spatial chunking the Dask-backed inputs carry, per dimension.

    Static spatial operands (seeds, resumed state) are partitioned to this
    chunking so ``apply_ufunc`` hands a worker a tile of its own instead of the
    whole grid. The coarsest shared chunking is chosen so the operand adds no
    new Dask boundaries; ``apply_ufunc`` still unifies every input to the
    finest of their chunkings on its own. Empty when no input is Dask-backed.
    """
    targets: dict[str, tuple[int, ...]] = {}
    for dim in spatial_dims:
        chunkings = [
            data.chunks[data.dims.index(dim)] for data in inputs if data.chunks is not None and dim in data.dims
        ]
        if chunkings:
            targets[dim] = min(chunkings, key=len)
    return targets


def validate_time_axes(inputs: Sequence[xr.DataArray], *, time_dim: str) -> None:
    """Validate each input's time axis: known dimension, monotonic, daily, single chunk."""
    for data in inputs:
        validate_time_dimension(data, time_dim)
        # a dimension-only time axis carries no cadence metadata: xarray aligns
        # it positionally, so monotonicity and daily checks apply only to real
        # coordinates
        if time_dim in data.coords:
            validate_time_monotonicity(data.coords[time_dim])
            _validate_daily_time_coordinate(data, time_dim)
        validate_dask_chunks(data, time_dim)


def align_time_series_inputs(
    inputs: Sequence[NamedInput],
    *,
    time_dim: str,
    index_display_name: str,
) -> tuple[xr.DataArray, ...]:
    """Inner-join the time-series inputs, protecting spatial coverage and reporting time drops.

    A shared spatial dimension that loses coordinates is an error, while a
    shortened time axis is the documented intersection and only warns. All
    inputs are aligned in one call so a partial mismatch cannot pair one
    input's grid with another's.
    """
    data_arrays = tuple(data for _, data in inputs)
    shared_spatial_dims = sorted(
        {
            str(dim)
            for first in range(len(data_arrays))
            for second in range(first + 1, len(data_arrays))
            for dim in data_arrays[first].dims
            if dim in data_arrays[second].dims and dim != time_dim
        }
    )
    original_sizes = {
        str(dim): [(name, data.sizes[dim]) for name, data in inputs if dim in data.sizes] for dim in shared_spatial_dims
    }
    try:
        aligned = xr.align(*data_arrays, join="inner")
    except xr.AlignmentError as exc:
        raise CoordinateValidationError(
            message=(
                f"Cannot align the {index_display_name} inputs: their dimension sizes or coordinate labels "
                f"conflict ({exc}). Give the inputs matching spatial shapes and a shared time axis."
            ),
            coordinate_name=time_dim,
            reason="alignment_conflict",
        ) from exc
    for dim in shared_spatial_dims:
        positions = [index for index, (_, data) in enumerate(inputs) if dim in data.sizes]
        if any(aligned[position].sizes[dim] != inputs[position][1].sizes[dim] for position in positions):
            raise CoordinateValidationError(
                message=(
                    f"Input alignment dropped coordinates along non-time dimension '{dim}': "
                    + ", ".join(f"{name} had {size}" for name, size in original_sizes[dim])
                    + "; after the inner join they have "
                    + ", ".join(f"{inputs[position][0]} {aligned[position].sizes[dim]}" for position in positions)
                    + f". Subset or align the inputs explicitly; {index_display_name} never reduces "
                    "spatial coverage silently."
                ),
                coordinate_name=dim,
                reason="non_time_alignment_dropped_coordinates",
            )
    aligned_length = aligned[0].sizes[time_dim]
    original_length = max(data.sizes[time_dim] for _, data in inputs)
    # an already-empty record yields an empty result; only an alignment that
    # empties a non-empty input is an error
    if aligned_length == 0 and original_length > 0:
        raise CoordinateValidationError(
            message=(
                f"No overlapping timesteps found across the {index_display_name} inputs along "
                f"'{time_dim}'. Cannot compute {index_display_name}."
            ),
            coordinate_name=time_dim,
            reason="empty_intersection_after_alignment",
        )
    if aligned_length < original_length:
        warnings.warn(
            InputAlignmentWarning(
                message=(
                    "Input alignment: "
                    + ", ".join(f"{name} had {data.sizes[time_dim]} timesteps" for name, data in inputs)
                    + f". After inner join, {aligned_length} remain."
                ),
                original_size=original_length,
                aligned_size=aligned_length,
                dropped_count=original_length - aligned_length,
            ),
            # points one frame above the adapter helper, at the public index
            # function, so a caller's warning filters see it
            stacklevel=4,
        )
    return tuple(aligned)


def restore_time_coordinates(
    result: xr.DataArray,
    *,
    source: xr.DataArray | None,
    time_dim: str,
    spin_up: int,
) -> xr.DataArray:
    """Reattach every time-dependent coordinate the excluded time dimension dropped.

    ``apply_ufunc`` with ``exclude_dims={time_dim}`` drops each coordinate that
    varies along ``time``. Slicing the coordinate objects (rather than their
    bare values) keeps their CF attributes, and copying the source's other
    time-dependent coordinates keeps a caller's ``doy``/``month`` bookkeeping.
    The slice is ``spin_up`` plus the result's own time length, so a kernel
    that trims its own spin-up (as every recurrence here does) stays in step.
    """
    if source is None:
        return result
    start = slice(spin_up, spin_up + result.sizes[time_dim])
    restored: xr.DataArray = result.assign_coords(
        {name: coord.isel({time_dim: start}) for name, coord in source.coords.items() if time_dim in coord.dims}
    )
    return restored


def _validate_extra_time_lengths(
    extras: Sequence[ExtraInput],
    alignment: StatefulAlignment,
    *,
    index_display_name: str,
) -> None:
    """Reject an extra time series whose length differs from the aligned inputs'.

    An extra time series (month) shares the aligned time coordinate, so it must
    not introduce a second time length. CFFWIS resolves its month before this
    point, so the guard serves external adapter callers.
    """
    time_dim = alignment.time_dim
    for data, core_dim in extras:
        if core_dim == time_dim and data.sizes.get(time_dim) not in (None, alignment.time_length):
            raise CoordinateValidationError(
                message=(
                    f"An extra {index_display_name} input varies along '{time_dim}' with "
                    f"{data.sizes[time_dim]} entries but the aligned inputs have {alignment.time_length}."
                ),
                coordinate_name=time_dim,
                reason="extra_input_time_length_mismatch",
            )


def stateful_recurrence_xarray(
    inputs: Sequence[NamedInput],
    kernel: Callable[..., Any],
    *,
    time_dim: str,
    spin_up: int,
    output_core_dims: Sequence[str | None],
    output_dtypes: Sequence[type],
    index_display_name: str,
    build_extra_inputs: Callable[[StatefulAlignment], Sequence[ExtraInput]] | None = None,
    kernel_kwargs: Mapping[str, Any] | None = None,
) -> StatefulAdapterOutput:
    """Run one stateful recurrence over every spatial block of the inputs.

    ``inputs`` are the time-varying primary inputs, aligned with the shared
    policy above. ``kernel`` receives each primary input's block with its core
    dimension last, then each extra input's block (also core-dimension-last for
    a time-varying extra), and returns one array per ``output_core_dims``
    entry. ``build_extra_inputs`` runs after alignment so a static operand can
    be wrapped to the resolved topology and chunked to the weather tiles.
    """
    if len(inputs) < 1:
        raise ValueError("stateful_recurrence_xarray requires at least one time-series input")
    if len(output_core_dims) != len(output_dtypes):
        raise ValueError("output_core_dims and output_dtypes must have the same length")

    validate_time_axes([data for _, data in inputs], time_dim=time_dim)
    aligned = align_time_series_inputs(inputs, time_dim=time_dim, index_display_name=index_display_name)

    output_dims, broadcast_sizes = broadcast_topology(aligned)
    spatial_dims = tuple(dim for dim in output_dims if dim != time_dim)
    spatial_shape = tuple(broadcast_sizes[dim] for dim in spatial_dims)
    alignment = StatefulAlignment(
        spatial_dims=spatial_dims,
        spatial_shape=spatial_shape,
        spatial_chunks=spatial_chunk_targets(aligned, spatial_dims),
        output_dims=output_dims,
        time_source=next((data for data in aligned if time_dim in data.coords), None),
        time_length=broadcast_sizes[time_dim],
        time_dim=time_dim,
        aligned_inputs=aligned,
    )

    extras: tuple[ExtraInput, ...] = tuple(build_extra_inputs(alignment)) if build_extra_inputs is not None else ()
    extra_arrays = tuple(data for data, _ in extras)
    _validate_extra_time_lengths(extras, alignment, index_display_name=index_display_name)

    output_time_length = max(alignment.time_length - spin_up, 0)
    input_core_dims: list[list[str]] = [[time_dim] for _ in aligned]
    input_core_dims += [[core_dim] if core_dim is not None else [] for _, core_dim in extras]
    output_core_dims_list = [[core_dim] if core_dim is not None else [] for core_dim in output_core_dims]

    apply_results = xr.apply_ufunc(
        kernel,
        *aligned,
        *extra_arrays,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims_list,
        exclude_dims={time_dim},
        vectorize=False,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {time_dim: output_time_length}},
        output_dtypes=list(output_dtypes),
        kwargs=dict(kernel_kwargs or {}),
    )
    results = tuple(apply_results) if isinstance(apply_results, tuple) else (apply_results,)

    finalized: list[xr.DataArray] = []
    for result, core_dim in zip(results, output_core_dims, strict=True):
        if core_dim is not None and result.dims != output_dims:
            result = result.transpose(*output_dims)
        if core_dim == time_dim:
            result = restore_time_coordinates(
                result,
                source=alignment.time_source,
                time_dim=time_dim,
                spin_up=spin_up,
            )
        finalized.append(result)
    return StatefulAdapterOutput(outputs=tuple(finalized), alignment=alignment)
