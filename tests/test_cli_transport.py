"""Tests for the CLI's shared-array transport (``climate_indices._cli_transport``)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from climate_indices import __main__ as cli_main
from climate_indices import _cli_transport, compute, indices
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


def test_copy_in_canonicalizes_time_major_input():
    """A time-major variable is transposed to the canonical time-last store."""
    values = np.arange(24, dtype=float).reshape(2, 3, 4)
    dataset = xr.Dataset({"prcp": (("time", "lat", "lon"), values)})
    transport = _cli_transport.Transport(1)

    shape = transport.copy_in(dataset, ["prcp"], None, DatasetLayout.GRID)

    assert shape == (3, 4, 2)
    np.testing.assert_array_equal(transport.read("prcp", shape), values.transpose(1, 2, 0))


def test_time_major_input_writes_the_same_values(monkeypatch, tmp_path):
    """A division input in either dimension order produces the same SPI output."""
    time = xr.date_range("1990-01-01", periods=36, freq="MS")
    coords = {"division": ["0101"], "time": time}
    values = np.random.default_rng(0).gamma(2.0, 30.0, size=36)
    latitude = (("division",), [35.0])
    datasets = {
        "last": xr.Dataset(
            {"prcp": (("division", "time"), values[None, :], {"units": "mm"}), "lat": latitude}, coords=coords
        ),
        "major": xr.Dataset(
            {"prcp": (("time", "division"), values[:, None], {"units": "mm"}), "lat": latitude}, coords=coords
        ),
    }

    outputs: dict[str, np.ndarray] = {}
    for name, dataset in datasets.items():
        monkeypatch.setattr(cli_main.xr, "open_mfdataset", lambda *_args, _dataset=dataset, **_kwargs: _dataset)
        base = str(tmp_path / name)
        request = cli_main._IndexRequest(
            index="spi",
            netcdf_precip="prcp.nc",
            var_name_precip="prcp",
            input_type=DatasetLayout.DIVISIONS,
            periodicity=compute.Periodicity.monthly,
            chunksizes="none",
            output_file_base=base,
            scale=3,
            distribution=indices.Distribution.gamma,
            calibration_start_year=1990,
            calibration_end_year=1992,
        )
        cli_main._compute_write_index(request, _cli_transport.Transport(1))
        with xr.open_dataset(f"{base}_spi_gamma_03.nc") as written:
            outputs[name] = written["spi_gamma_03"].values

    assert outputs["major"].shape == outputs["last"].shape
    np.testing.assert_allclose(outputs["major"], outputs["last"])


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
