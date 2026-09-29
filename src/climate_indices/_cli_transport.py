"""Shared-array transport for the CLI's multiprocessing kernels.

The CLI copies each input variable into process-shared memory once, splits the
leading axis across workers, runs one NumPy kernel per chunk, and reads the
results back out. Those four steps used to be spread across module globals, an
untyped per-worker parameter dictionary, and three hand-written workers. This
module owns them behind one interface: *run this kernel over these named inputs
and return these named outputs*.

The store is an instance, not a module global, so two transports in one process
cannot reuse each other's buffers. A worker reads the store through the module
reference :func:`_initialize_worker` sets from ``Pool(initializer=...)`` (a
shared ``multiprocessing.Array`` cannot be pickled through the task queue); the
inline executor sets and then restores it around its map.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import xarray as xr

from climate_indices.utils import DailyCalendarPlan
from climate_indices.validation import DatasetLayout, expected_dimensions

# the store keys that are not input variable names: the latitude companion, the
# single result buffer, and the five Palmer result buffers
LATITUDE_ARRAY_KEY = "lat"
RESULT_ARRAY_KEY = "result_array"
PALMER_RESULT_KEYS = (
    "result_array_pdsi",
    "result_array_phdi",
    "result_array_pmdi",
    "result_array_zindex",
    "result_array_scpdsi",
)

# the order the transport stores a time-carrying variable in: time-last, which
# is where _TIME_AXIS_INDEX expects it and what the kernels index. copy_in
# transposes any other accepted order to this one.
_CANONICAL_DIMENSIONS: dict[DatasetLayout, tuple[Hashable, ...]] = {
    DatasetLayout.GRID: ("lat", "lon", "time"),
    DatasetLayout.DIVISIONS: ("division", "time"),
    DatasetLayout.TIMESERIES: ("time",),
}

# the time-major orders a time-carrying variable may arrive in, per layout
_TIME_MAJOR_DIMENSIONS: dict[DatasetLayout, tuple[tuple[Hashable, ...], ...]] = {
    DatasetLayout.GRID: (("time", "lat", "lon"),),
    DatasetLayout.DIVISIONS: (("time", "division"),),
    DatasetLayout.TIMESERIES: (),
}

# every order the transport reads for a layout, canonical first
_TRANSPORT_DIMENSIONS: dict[DatasetLayout, tuple[tuple[Hashable, ...], ...]] = {
    layout: (_CANONICAL_DIMENSIONS[layout], *_TIME_MAJOR_DIMENSIONS[layout]) for layout in _CANONICAL_DIMENSIONS
}

# the axis each input type's time dimension lies along
_TIME_AXIS_INDEX: dict[DatasetLayout, int] = {
    DatasetLayout.GRID: 2,
    DatasetLayout.DIVISIONS: 1,
    DatasetLayout.TIMESERIES: 0,
}


def canonical_dimensions(layout: DatasetLayout) -> tuple[Hashable, ...]:
    """The time-last order the transport stores a time-carrying variable in."""
    return _CANONICAL_DIMENSIONS[layout]


def transport_dimensions(layout: DatasetLayout) -> tuple[tuple[Hashable, ...], ...]:
    """The time-carrying dimension orders the transport reads for this layout."""
    return _TRANSPORT_DIMENSIONS[layout]


def accepted_dimensions(layout: DatasetLayout) -> tuple[tuple[Hashable, ...], ...]:
    """
    Every dimension order a variable in a dataset of this layout may use.

    The data variables are limited to the orders the shared-array transport and
    the kernels can read, and a layout's per-location companions -- such as the
    division latitudes -- are fixed per location, without a time dimension.

    param layout: the dataset layout the dimensions are accepted for
    return: the accepted dimension orders, in storage order
    """
    return transport_dimensions(layout) + (expected_dimensions(layout, includes_time=False) or ())


@dataclass
class _SharedArray:
    """One process-shared buffer and the shape its values are mapped onto."""

    values: Any
    shape: tuple[int, ...]


class SharedArrayStore:
    """The CLI's named process-shared arrays, owned by one invocation."""

    def __init__(self) -> None:
        self._arrays: dict[str, _SharedArray] = {}

    def allocate(self, name: str, shape: tuple[int, ...]) -> None:
        """Create a zeroed shared buffer under ``name``."""
        self._arrays[name] = _SharedArray(
            values=multiprocessing.Array("d", int(np.prod(shape))),
            shape=shape,
        )

    def view(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        """Return the named buffer's values as a NumPy array of ``shape``."""
        shared = self._arrays[name].values
        values: np.ndarray = np.frombuffer(shared.get_obj()).reshape(shape)
        return values

    def write(self, name: str, values: np.ndarray) -> None:
        """Allocate a buffer and copy ``values`` into it."""
        self.allocate(name, values.shape)
        np.copyto(self.view(name, values.shape), values)

    def read(self, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
        """Return a copy of the named buffer's values."""
        buffer = self.view(name, shape or self._arrays[name].shape)
        copied: np.ndarray = buffer.copy()
        return copied

    def shape(self, name: str) -> tuple[int, ...]:
        """The shape the named buffer's values are mapped onto."""
        return self._arrays[name].shape

    def __contains__(self, name: str) -> bool:
        return name in self._arrays


@dataclass(frozen=True)
class WorkItem:
    """One kernel chunk: a kernel, its named operands, and the row span it owns."""

    kernel: Callable[..., Any]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    coordinate_input: bool
    layout: DatasetLayout
    arguments: Mapping[str, Any]
    start: int
    end: int | None


# the store a spawned worker reads; set once per pool by _initialize_worker
_worker_store: SharedArrayStore | None = None


def _initialize_worker(store: SharedArrayStore) -> None:
    """Pool initializer: hand each worker the process-inherited store."""
    global _worker_store
    _worker_store = store


def _require_worker_store() -> SharedArrayStore:
    assert _worker_store is not None, "the transport store must be initialised in the worker process"
    return _worker_store


def run_along_axis(item: WorkItem) -> None:
    """
    Apply ``item.kernel`` along the time axis of one chunk of a single input.

    Like :func:`numpy.apply_along_axis`, but reading and writing the shared
    store. Suitable as a :meth:`multiprocessing.Pool.map` target.
    """
    store = _require_worker_store()
    input_name = item.input_names[0]
    output_name = item.output_names[0]
    shape = store.shape(input_name)

    sub_array = store.view(input_name, shape)[item.start : item.end]
    axis_index = _TIME_AXIS_INDEX[item.layout]
    computed_array = np.apply_along_axis(item.kernel, axis=axis_index, arr=sub_array, parameters=item.arguments)

    np.copyto(store.view(output_name, shape)[item.start : item.end], computed_array)


def run_along_axis_double(item: WorkItem) -> None:
    """
    Apply ``item.kernel`` across the time axis of two chunks, one per input.

    The second input is a coordinate fixed per row rather than a per-cell value
    when ``item.coordinate_input`` is set. Suitable as a
    :meth:`multiprocessing.Pool.map` target.
    """
    store = _require_worker_store()
    first_array_key, second_array_key = item.input_names
    output_name = item.output_names[0]

    shape = store.shape(output_name)
    # a coordinate input has one value per row rather than per cell
    second_shape = (shape[0],) if item.coordinate_input else shape
    sub_array_1 = store.view(first_array_key, shape)[item.start : item.end]
    sub_array_2 = store.view(second_array_key, second_shape)[item.start : item.end]
    computed_array = store.view(output_name, shape)[item.start : item.end]

    for i, (x, y) in enumerate(zip(sub_array_1, sub_array_2, strict=False)):
        if item.layout == DatasetLayout.GRID:
            for j in range(x.shape[0]):
                second_value = y if item.coordinate_input else y[j]
                computed_array[i, j] = item.kernel(x[j], second_value, parameters=item.arguments)
        elif item.layout == DatasetLayout.DIVISIONS:
            computed_array[i] = item.kernel(x, y, parameters=item.arguments)
        else:
            raise ValueError(f"Unsupported input type: '{item.layout}'")


def run_palmers(item: WorkItem) -> None:
    """
    Apply the Palmer kernel across one chunk of the Palmer inputs.

    A grid chunk is computed in one vectorized call over the whole
    ``(lat_chunk, lon, time)`` block with a private ``spatial_time_major=True``,
    so the standard indices are read per ADR-0009 while the kernel loops scPDSI
    per location (ADR-0011); multiprocessing still parallelizes across chunks
    (ADR-0002). A divisions chunk stays on the per-location loop.
    """
    store = _require_worker_store()
    precip_array_key, pet_array_key, awc_array_key = item.input_names
    output_keys = item.output_names

    shape = store.shape(output_keys[0])
    sub_array_precip = store.view(precip_array_key, shape)[item.start : item.end]
    sub_array_pet = store.view(pet_array_key, shape)[item.start : item.end]
    # available water capacity is fixed per location, without a time dimension
    awc_shape: tuple[Any, ...] = (shape[0], shape[1]) if item.layout == DatasetLayout.GRID else (shape[0],)
    sub_array_awc = store.view(awc_array_key, awc_shape)[item.start : item.end]

    args = item.arguments

    pdsi = store.view(output_keys[0], shape)[item.start : item.end]
    phdi = store.view(output_keys[1], shape)[item.start : item.end]
    pmdi = store.view(output_keys[2], shape)[item.start : item.end]
    zindex = store.view(output_keys[3], shape)[item.start : item.end]
    scpdsi = store.view(output_keys[4], shape)[item.start : item.end]

    if item.layout == DatasetLayout.GRID:
        # sub_array_precip/pet are (lat_chunk, lon, time); pdsi() wants a
        # time-major (time, *cells) block
        precip_block = np.moveaxis(sub_array_precip, -1, 0)
        pet_block = np.moveaxis(sub_array_pet, -1, 0)
        block_args = {**args, "spatial_time_major": True}
        block_pdsi, block_phdi, block_pmdi, block_zindex, block_scpdsi = item.kernel(
            precip_block,
            pet_block,
            sub_array_awc,
            parameters=block_args,
        )
        np.copyto(pdsi, np.moveaxis(block_pdsi, 0, -1))
        np.copyto(phdi, np.moveaxis(block_phdi, 0, -1))
        np.copyto(pmdi, np.moveaxis(block_pmdi, 0, -1))
        np.copyto(zindex, np.moveaxis(block_zindex, 0, -1))
        np.copyto(scpdsi, np.moveaxis(block_scpdsi, 0, -1))
    else:  # divisions
        for i, (precip, pet, awc) in enumerate(zip(sub_array_precip, sub_array_pet, sub_array_awc, strict=False)):
            pdsi[i], phdi[i], pmdi[i], zindex[i], scpdsi[i] = item.kernel(precip, pet, awc, parameters=args)


class Executor(Protocol):
    """The dispatch seam: a pool of processes and an in-process map."""

    def map(self, worker: Callable[[WorkItem], None], items: Sequence[WorkItem]) -> None: ...


class InlineExecutor:
    """Run every work item in-process, for tests and single-process runs."""

    def __init__(self, store: SharedArrayStore) -> None:
        self._store = store

    def map(self, worker: Callable[[WorkItem], None], items: Sequence[WorkItem]) -> None:
        global _worker_store
        previous, _worker_store = _worker_store, self._store
        try:
            for item in items:
                worker(item)
        finally:
            _worker_store = previous


class PoolExecutor:
    """Run work items across a ``multiprocessing.Pool``."""

    def __init__(self, store: SharedArrayStore, processes: int) -> None:
        self._store = store
        self._processes = processes

    def map(self, worker: Callable[[WorkItem], None], items: Sequence[WorkItem]) -> None:
        with multiprocessing.Pool(
            processes=self._processes,
            initializer=_initialize_worker,
            initargs=(self._store,),
        ) as pool:
            pool.map(worker, items)


class Transport:
    """Copy inputs into shared memory, run kernels over chunks, read results."""

    def __init__(self, processes: int) -> None:
        self.store = SharedArrayStore()
        # a one-CPU host's default all-but-one count is zero, which would make
        # _partition divide by zero rather than run the single process
        self.processes = max(1, processes)

    def copy_in(
        self,
        dataset: xr.Dataset,
        var_names: list[str],
        calendar_plan: DailyCalendarPlan | None,
        layout: DatasetLayout,
    ) -> tuple[int, ...]:
        """
        Copy each named variable into the shared store and return the output shape.

        A daily input is first converted to 366-day years; a time-free companion
        such as a division latitude keeps its shape. A grid's output shape is the
        last variable's shape; a divisions output shape comes from the 2-D
        variable rather than a 1-D companion.
        """
        canonical = canonical_dimensions(layout)
        output_shape: tuple[int, ...] | None = None
        for var_name in var_names:
            variable = dataset[var_name]
            # a time-major variable is transposed to the canonical time-last
            # storage order before the kernels index it
            if variable.ndim == len(canonical) and tuple(variable.dims) != canonical:
                variable = variable.transpose(*canonical)
            if calendar_plan is not None and "time" in variable.dims:
                var_values = np.apply_along_axis(calendar_plan.to_all_leap, len(variable.dims) - 1, variable.values)
            else:
                var_values = variable.values

            self.store.write(var_name, var_values)

            if layout == DatasetLayout.DIVISIONS:
                # a divisions output shape comes from its 2-D variable, not a 1-D companion
                if len(var_values.shape) == 2:
                    output_shape = var_values.shape
            else:
                output_shape = var_values.shape

            # drop the variable from this local view (we're assuming this frees the memory)
            dataset = dataset.drop_vars(names=[var_name])

        assert output_shape is not None, "No variables processed; output shape is unknown"
        return output_shape

    def execute(
        self,
        *,
        kernel: Callable[..., Any],
        input_names: tuple[str, ...],
        output_names: tuple[str, ...],
        coordinate_input: bool,
        layout: DatasetLayout,
        arguments: Mapping[str, Any],
        worker: Callable[[WorkItem], None],
        executor: Executor | None = None,
    ) -> None:
        """Partition the leading axis and run ``worker`` over each chunk."""
        shape = self.store.shape(output_names[0])
        items = _partition(
            shape[0], self.processes, kernel, input_names, output_names, coordinate_input, layout, arguments
        )
        if executor is None:
            executor = InlineExecutor(self.store) if self.processes == 1 else PoolExecutor(self.store, self.processes)
        executor.map(worker, items)

    def read(self, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
        return self.store.read(name, shape)

    def shape(self, name: str) -> tuple[int, ...]:
        return self.store.shape(name)

    def allocate(self, name: str, shape: tuple[int, ...]) -> None:
        self.store.allocate(name, shape)

    def write(self, name: str, values: np.ndarray) -> None:
        self.store.write(name, values)

    def __contains__(self, name: str) -> bool:
        return name in self.store


def _partition(
    total: int,
    processes: int,
    kernel: Callable[..., Any],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    coordinate_input: bool,
    layout: DatasetLayout,
    arguments: Mapping[str, Any],
) -> list[WorkItem]:
    """Split ``total`` rows into at most ``processes`` chunks."""
    # if there are fewer chunks than the available number of processes
    # then only create the necessary number of tasks
    required_processes = min(total, processes)
    d, m = divmod(total, required_processes)
    split_indices = list(range(0, ((d + 1) * (m + 1)), (d + 1)))
    if d != 0:
        split_indices += list(range(split_indices[-1] + d, total, d))

    return [
        WorkItem(
            kernel=kernel,
            input_names=input_names,
            output_names=output_names,
            coordinate_input=coordinate_input,
            layout=layout,
            arguments=arguments,
            start=split_indices[i],
            end=split_indices[i + 1] if i < (required_processes - 1) else None,
        )
        for i in range(required_processes)
    ]
