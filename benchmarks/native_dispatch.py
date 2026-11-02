"""Report actual native dispatch and fallback gates, including worker-local state.

Run ``uv run benchmarks/native_dispatch.py`` after building the extension. This
is a diagnostic, not a timing harness. It reuses the test recorder and does not
change library eligibility, warning filters, or scheduler configuration.
"""

from __future__ import annotations

import importlib
import json
import multiprocessing
import os
import queue
import sys
import threading
import warnings
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from climate_indices import __main__ as cli  # noqa: E402
from climate_indices import _cli_transport, _native_backend, compute, indices, spi, standardized_index  # noqa: E402
from climate_indices.validation import DatasetLayout  # noqa: E402
from tests.conftest import NativeRecorder  # noqa: E402

DEFAULT_POLICY = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}


def fallback_reason(*arrays: np.ndarray) -> str | None:
    """Name the first shared gate, in the same order as ``takes``."""
    if not all(_native_backend.plain_float64(array) for array in arrays):
        return "layout: not plain aligned float64"
    reporting = [f"{name}={action}" for name, action in np.geterr().items() if action != "ignore"]
    if reporting:
        return "numpy-policy: " + ", ".join(reporting)
    if getattr(sys.flags, "context_aware_warnings", False):
        return "context-aware-warnings"
    if any(
        action == "error" and issubclass(RuntimeWarning, category) for action, _, category, _, _ in warnings.filters
    ):
        return "warning-filter: RuntimeWarning -> error"
    return None


def _emit(sink: Any, **event: Any) -> None:
    sink.put(
        {
            "pid": os.getpid(),
            "thread": threading.get_ident(),
            "numpy_policy": np.geterr(),
            "warning_filters": [
                [action, str(message), f"{category.__module__}.{category.__name__}", str(module), line]
                for action, message, category, module, line in warnings.filters
            ],
            **event,
        }
    )


@contextmanager
def recording(sink: Any) -> Iterator[None]:
    """Observe compute's shared gates and actual calls, without changing their decisions."""
    native = compute._native
    if native is None:
        raise RuntimeError("Build the extension with uv run maturin develop --release first")

    class ReportingRecorder(NativeRecorder):
        def __getattr__(self, name: str) -> Any:
            call = super().__getattr__(name)
            if isinstance(call, type):
                return call

            def recorded(*args: Any, **kwargs: Any) -> Any:
                result = call(*args, **kwargs)
                _emit(sink, kernel=name)
                return result

            return recorded

    original_takes = _native_backend.takes
    original_kernels = _native_backend.kernels
    original_sum = compute.sum_to_scale

    def takes(*arrays: np.ndarray) -> bool:
        allowed = original_takes(*arrays)
        reason = fallback_reason(*arrays)
        if allowed != (reason is None):
            raise RuntimeError("The diagnostic no longer matches the shared native gate")
        if reason is not None:
            _emit(sink, fallback=reason)
        return allowed

    def kernels(module: Any, *names: str) -> Any:
        result = original_kernels(module, *names)
        if result is None:
            missing = [name for name in names if module is None or not hasattr(module, name)]
            _emit(sink, fallback="kernel-missing: " + ", ".join(missing))
        return result

    def sum_to_scale(values: np.ndarray, scale: int) -> np.ndarray:
        # Mirror sum_to_scale's early return and mask normalization, so a reported
        # route-specific restriction is one the actual call can reach. The shared
        # gate's own reason is emitted by the patched ``takes``.
        if scale != 1:
            normalized = np.ma.filled(values.astype(float), np.nan) if np.ma.isMaskedArray(values) else values
            if fallback_reason(normalized) is None:
                if normalized.ndim == 1 and scale >= 8:
                    _emit(sink, fallback="trailing_sum: series Timescale >= 8")
                elif normalized.ndim > 2 and not normalized.flags.c_contiguous:
                    _emit(sink, fallback="trailing_sum: block is not C-contiguous")
        return original_sum(values, scale)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(compute, "_native", ReportingRecorder(native))
        patch.setattr(_native_backend, "takes", takes)
        patch.setattr(_native_backend, "kernels", kernels)
        patch.setattr(compute, "sum_to_scale", sum_to_scale)
        _emit(sink, observation="execution-state")
        yield


def _drain(sink: Any) -> list[dict[str, Any]]:
    events = []
    while True:
        try:
            events.append(sink.get_nowait())
        except queue.Empty:
            return events


def _initialize_probe(sink: Any, worker_policy: str | None) -> None:
    if worker_policy is not None:
        np.seterr(all=worker_policy)
    # A process owns this instrumentation until its pool exits.
    context = recording(sink)
    context.__enter__()
    global _worker_recording
    _worker_recording = context


_worker_recording: Any = None


def observe_dask(
    lazy: xr.DataArray, scheduler: str, worker_policy: str | None = None
) -> tuple[Any, list[dict[str, Any]]]:
    """Record kernels inside local Dask workers; leave their policy unchanged unless requested."""
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        sink = manager.Queue()
        if scheduler == "processes":
            with ProcessPoolExecutor(
                max_workers=2, mp_context=context, initializer=_initialize_probe, initargs=(sink, worker_policy)
            ) as pool:
                result = lazy.compute(scheduler=scheduler, pool=pool)
        else:
            # The switch is process-wide, but each thread owns its NumPy policy.
            kwargs = {} if worker_policy is None else {"initializer": np.seterr, "initargs": (worker_policy,)}
            with recording(sink), ThreadPoolExecutor(max_workers=2, **kwargs) as pool:
                result = lazy.compute(scheduler=scheduler, pool=pool)
        return result, _drain(sink)


def _cli_probe_worker(item: _cli_transport.WorkItem) -> None:
    with recording(item.arguments["report_queue"]):
        _cli_transport.run_along_axis(item)


def sample_grid() -> xr.DataArray:
    """A complete, nondegenerate float64 monthly grid with a 30-year Calibration Period."""
    return xr.DataArray(
        np.random.default_rng(1531).gamma(2.0, 15.0, size=(360, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={
            "time": xr.date_range("1981-01-01", periods=360, freq="MS"),
            "lat": [35.0, 36.0],
            "lon": [-100.0, -99.0],
        },
        attrs={"units": "mm"},
    )


def run_route(
    route: str, distribution: indices.Distribution, worker_policy: str | None = None
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Execute a representative public route, returning its output and dispatch evidence."""
    grid = sample_grid()
    arguments: dict[str, Any] = {
        "scale": 3,
        "distribution": distribution,
        "data_start_year": 1981,
        "calibration_year_initial": 1981,
        "calibration_year_final": 2010,
        "periodicity": compute.Periodicity.monthly,
    }
    call: Any = indices.spi if distribution == indices.Distribution.gamma else indices.standardized_index
    if route != "numpy":
        call = spi if distribution == indices.Distribution.gamma else standardized_index
    if route.startswith("dask-"):
        lazy = call(grid.chunk({"time": -1, "lat": 1, "lon": 1}), **arguments)
        result, events = observe_dask(lazy, route.removeprefix("dask-"), worker_policy)
        return result.values, events
    if route == "cli":
        if distribution != indices.Distribution.gamma:
            raise ValueError("The CLI SPI route supports gamma, not empirical")
        with multiprocessing.get_context("spawn").Manager() as manager:
            sink = manager.Queue()
            transport = _cli_transport.Transport(2)
            transport.write("precip", grid.values.transpose(1, 2, 0))
            transport.allocate("out", (2, 2, 360))
            transport.execute(
                kernel=cli._spi,
                input_names=("precip",),
                output_names=("out",),
                coordinate_input=False,
                layout=DatasetLayout.GRID,
                arguments={**arguments, "output_scale": "normal", "zero_handling": "classic", "report_queue": sink},
                worker=_cli_probe_worker,
            )
            return transport.read("out").transpose(2, 0, 1), _drain(sink)
    sink = queue.Queue()
    with recording(sink):
        values = grid.values[:, 0, 0].copy() if route == "numpy" else grid
        result = call(values, **arguments)
    return np.asarray(result), _drain(sink)


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep every distinct worker state and every observed call or fallback reason."""
    states = {
        json.dumps({key: event[key] for key in ("pid", "thread", "numpy_policy", "warning_filters")}, sort_keys=True)
        for event in events
    }
    return {
        "kernels": sorted({event["kernel"] for event in events if "kernel" in event}),
        "fallbacks": sorted({event["fallback"] for event in events if "fallback" in event}),
        "worker_states": [json.loads(state) for state in sorted(states)],
    }


# the routes whose own NumPy policy can reach Rust without a worker opt-in
_CALLER_POLICY_ROUTES = frozenset({"numpy", "xarray", "cli"})


def expected_native(route: str, caller_policy: str, worker_policy: str | None) -> bool:
    """Whether this route and policy should have run at least one native kernel."""
    if caller_policy == "default":
        return False
    if route in _CALLER_POLICY_ROUTES:
        return True
    return worker_policy is not None


def require_expected(
    route: str, caller_policy: str, worker_policy: str | None, summary: dict[str, Any]
) -> None:
    """Fail when a route expected to run natively reports no kernel, or the reverse."""
    expected = expected_native(route, caller_policy, worker_policy)
    if bool(summary["kernels"]) != expected:
        wanted = "native kernels" if expected else "Python"
        raise RuntimeError(f"{route} under the {caller_policy} caller policy: expected {wanted}, got {summary}")


def main() -> None:
    """Print one JSON record per route and policy, plus explicit Dask worker opt-ins."""
    importlib.import_module("climate_indices._native")
    for caller_policy in ("default", "ignore"):
        settings = DEFAULT_POLICY if caller_policy == "default" else dict.fromkeys(DEFAULT_POLICY, "ignore")
        with np.errstate(**settings):
            for route in ("numpy", "xarray", "dask-threads", "dask-processes", "cli"):
                distributions = (
                    (indices.Distribution.gamma,)
                    if route == "cli"
                    else (indices.Distribution.gamma, indices.Distribution.empirical)
                )
                for distribution in distributions:
                    _, events = run_route(route, distribution)
                    summary = summarize(events)
                    require_expected(route, caller_policy, None, summary)
                    print(
                        json.dumps(
                            {
                                "route": route,
                                "distribution": distribution.value,
                                "caller_policy": caller_policy,
                                **summary,
                            }
                        )
                    )
    with np.errstate(all="ignore"):
        for route in ("dask-threads", "dask-processes"):
            _, events = run_route(route, indices.Distribution.gamma, worker_policy="ignore")
            summary = summarize(events)
            require_expected(route, "ignore", "ignore", summary)
            print(
                json.dumps(
                    {
                        "route": route,
                        "distribution": "gamma",
                        "caller_policy": "ignore",
                        "worker_policy": "ignore",
                        **summary,
                    }
                )
            )


if __name__ == "__main__":
    main()
