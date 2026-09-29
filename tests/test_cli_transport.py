"""Tests for the CLI's shared-array transport (``climate_indices._cli_transport``)."""

from __future__ import annotations

import numpy as np
import pytest

from climate_indices import _cli_transport
from climate_indices.__main__ import DatasetLayout
from climate_indices._cli_transport import WorkItem


def _double(x, parameters):
    return np.asarray(x, dtype=float) * 2.0


def _run_doubling(transport: _cli_transport.Transport) -> np.ndarray:
    transport.write("values", np.arange(6.0))
    transport.allocate("out", (6,))
    transport.execute(
        kernel=_double,
        input_names=("values",),
        output_names=("out",),
        coordinate_input=False,
        layout=DatasetLayout.TIMESERIES,
        arguments={},
        worker=_cli_transport.run_along_axis,
    )
    return transport.read("out", (6,))


def test_pool_and_inline_executors_agree():
    """The two adapters of the transport's dispatch seam must agree.

    ``Transport(1)`` runs the in-process map; ``Transport(2)`` runs the real
    ``multiprocessing.Pool`` (through its initializer handoff), so this pins
    that the shared store reaches spawned workers.
    """
    inline = _run_doubling(_cli_transport.Transport(1))
    pooled = _run_doubling(_cli_transport.Transport(2))

    np.testing.assert_array_equal(inline, np.arange(6.0) * 2.0)
    np.testing.assert_array_equal(pooled, inline)


def test_execute_uses_the_supplied_executor():
    """``Transport.execute`` dispatches through the injected executor."""
    recorded: list[list[WorkItem]] = []

    class RecordingExecutor:
        def map(self, worker, items):
            assert worker is _cli_transport.run_along_axis
            recorded.append(list(items))

    transport = _cli_transport.Transport(1)
    transport.write("values", np.arange(6.0))
    transport.allocate("out", (6,))
    transport.execute(
        kernel=_double,
        input_names=("values",),
        output_names=("out",),
        coordinate_input=False,
        layout=DatasetLayout.TIMESERIES,
        arguments={},
        worker=_cli_transport.run_along_axis,
        executor=RecordingExecutor(),
    )

    assert len(recorded) == 1
    assert [item.start for item in recorded[0]] == [0]


@pytest.mark.parametrize(("total", "processes"), [(1, 1), (2, 5), (5, 5), (5, 2), (7, 3)])
def test_partition_spans_cover_every_row(total, processes):
    """Chunk spans are contiguous, cover ``[0, total)``, and number ``min(total, processes)``."""
    items = _cli_transport._partition(
        total,
        processes,
        kernel=_double,
        input_names=("values",),
        output_names=("out",),
        coordinate_input=False,
        layout=DatasetLayout.TIMESERIES,
        arguments={},
    )

    starts = [item.start for item in items]
    ends = [item.end if item.end is not None else total for item in items]

    assert len(items) == min(total, processes)
    assert starts[0] == 0
    assert ends[-1] == total
    assert all(end > start for start, end in zip(starts, ends, strict=True))
    assert all(ends[i] == starts[i + 1] for i in range(len(items) - 1))
