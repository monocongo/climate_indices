"""Runnable check for the Rust-vs-Python benchmark harness (RUST-011)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests import conftest, parity_registry

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# the harness imports its sibling benchmark scripts by module name, as it does when run as a script;
# appended, not prepended, so a benchmarks/ module can never shadow a site-packages import
sys.path.append(str(ROOT / "benchmarks"))
harness = _load_script("rust_vs_python", "benchmarks/rust_vs_python.py")


@pytest.fixture(scope="module", autouse=True)
def native() -> ModuleType:
    """The built Rust extension; the harness has nothing to measure without it."""
    return conftest.import_native()


def test_report_covers_every_registry_entry() -> None:
    """The harness times the registry, so a kernel added to it cannot go unmeasured."""
    rendered = harness.render_entries(harness.time_entries(repeats=1))

    for entry in parity_registry.ENTRIES:
        assert f"`{entry.name}`" in rendered


def test_time_entry_reports_both_backends() -> None:
    """One entry is timed through the extension and on the pure-Python path."""
    entry = parity_registry.ENTRIES_BY_NAME["percentage_of_normal"]
    timings = harness.time_entry(entry, repeats=1)

    assert timings.name == entry.name
    assert timings.rust_seconds > 0.0
    assert timings.python_seconds > 0.0
    assert timings.ratio == pytest.approx(timings.python_seconds / timings.rust_seconds)


def test_backends_alternate_which_runs_first() -> None:
    """A fixed order lets drift favour one backend, so each repetition swaps which runs first."""
    order: list[str] = []
    rust_seconds, python_seconds = harness.measure_backends(
        lambda: order.append("python" if harness.compute_module()._native is None else "rust"),
        lambda: order.append("python" if harness.compute_module()._native is None else "rust"),
        repeats=3,
    )

    assert order == ["python", "rust", "rust", "python", "python", "rust", "rust", "python"]
    assert rust_seconds > 0.0
    assert python_seconds > 0.0


def test_python_backend_disables_every_dispatch_module() -> None:
    """The Python column is the pure-Python path, not Python orchestration over Rust kernels."""
    modules = {entry.dispatch for entry in parity_registry.ENTRIES}

    with harness.python_backend():
        assert all(module._native is None for module in modules)

    assert all(module._native is not None for module in modules)


def test_full_grid_run_reports_every_configuration(tmp_path: Path) -> None:
    """The --netcdf run times each entry eagerly and per thread count, on the grid's land cells only."""
    rng = np.random.default_rng(1)
    shape = (360, 3, 4)  # 1991-2020, the calibration period
    coords = {
        "time": pd.date_range("1991-01-01", periods=shape[0], freq="MS"),
        "lat": [30.0, 31.0, 32.0],
        "lon": [-100.0, -99.0, -98.0, -97.0],
    }
    precipitation = rng.gamma(2.0, 15.0, size=shape)
    precipitation[:, 0, :] = np.nan  # an ocean row the land mask must drop
    temperature = 15.0 + 10.0 * np.sin(np.arange(shape[0]) * np.pi / 6.0)[:, None, None] + rng.normal(size=shape)
    paths = {name: tmp_path / f"{name}.nc" for name in ("prcp", "tavg")}
    for name, values in (("prcp", precipitation), ("tavg", temperature)):
        xr.Dataset({name: (("time", "lat", "lon"), values)}, coords=coords).to_netcdf(paths[name], engine="h5netcdf")
    output = tmp_path / "report.txt"

    harness.main(
        ["--netcdf", str(paths["prcp"]), "--tavg", str(paths["tavg"]), "--repeat", "1", "--threads", "1,2"]
        + ["--output", str(output)]
    )
    report = output.read_text(encoding="utf-8")

    assert "8 land cells of 12" in report
    for name in harness.GRID_ENTRIES:
        assert sum(line.startswith(f"| `{name}` |") and line.endswith(" True |") for line in report.splitlines()) == 3


@pytest.mark.parametrize("start_month", range(2, 13))
def test_grid_rejects_non_january_start(tmp_path: Path, start_month: int) -> None:
    """Whole-year counts and calibration coverage do not guarantee January-aligned monthly data."""
    time = pd.date_range(pd.Timestamp(1990, start_month, 1), periods=31 * 12, freq="MS")
    path = tmp_path / "grid.nc"
    xr.Dataset(
        {"prcp": (("time", "lat", "lon"), np.full((time.size, 1, 1), 10.0))},
        coords={"time": time, "lat": [30.0], "lon": [-100.0]},
    ).to_netcdf(path, engine="h5netcdf")

    with pytest.raises(SystemExit, match="starting in January"):
        harness.load_grid(path, "prcp", path, "prcp")


def test_netcdf_needs_the_temperature_grid() -> None:
    """The PET-based entries compute from the grid's own temperature, so --tavg is required with --netcdf."""
    with pytest.raises(SystemExit) as error:
        harness.main(["--netcdf", "prcp.nc"])

    assert error.value.code == 2


@pytest.mark.parametrize("artifact", ["rust_vs_python.txt", "rust_vs_python_nclimgrid.txt"])
def test_readme_mirrors_every_committed_table_row(artifact: str) -> None:
    """Each data row of a committed run appears verbatim in benchmarks/README.md, so the two cannot drift."""
    readme = (ROOT / "benchmarks" / "README.md").read_text(encoding="utf-8")
    rows = (ROOT / "benchmarks" / "results" / artifact).read_text(encoding="utf-8").splitlines()

    missing = [row for row in rows if row.startswith("| ") and row not in readme]
    assert missing == []


def test_cold_call_runs_in_a_fresh_interpreter() -> None:
    """A cold measurement is a first call, in its own process, and is reported in seconds."""
    assert harness.measure_cold("percentage_of_normal") > 0.0


def test_render_entries_lists_every_measurement() -> None:
    """The rendered table carries one row per measurement, in the order given."""
    timings = [harness.Timings("first", "monthly", 0.001, 0.002), harness.Timings("second", "daily", 0.003, 0.001)]
    rendered = harness.render_entries(timings)

    assert "| `first` | monthly | 1.000 ms | 2.000 ms | 2.00 |" in rendered
    assert "| `second` | daily | 3.000 ms | 1.000 ms | 0.33 |" in rendered
    assert "Rust faster in 1 of 2 entries." in rendered


@pytest.mark.parametrize("cells", [harness.SWEEP_CELLS, (2, 4, 8)])
def test_fixed_and_per_cell_fits_a_known_line(cells: tuple[int, ...], monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom sweep counts remain coupled to fitting and rendering."""
    entry = parity_registry.ENTRIES_BY_NAME["spi_gamma_spatial_block"]

    def time_entry(entry: parity_registry.Entry, values: np.ndarray, repeats: int) -> object:
        count = values.shape[1]
        return harness.Timings(entry.name, "block", 0.1 + 0.002 * count, 0.3 + 0.004 * count)

    monkeypatch.setattr(harness, "time_entry", time_entry)
    sweep = harness.sweep_entry(entry, cells=cells, repeats=1)

    assert harness.fit_fixed_and_per_cell(sweep, "rust") == pytest.approx((0.1, 0.002))
    assert harness.fit_fixed_and_per_cell(sweep, "python") == pytest.approx((0.3, 0.004))
    rendered = harness.render_sweep(entry.name, sweep)
    for count in cells:
        assert f"| {count} |" in rendered


@pytest.mark.parametrize("cold", [False, True])
def test_warning_as_error_still_reaches_native(cold: bool, monkeypatch: pytest.MonkeyPatch, native: ModuleType) -> None:
    """Rust-labeled warm and cold calls must not silently time the Python fallback."""
    entry = parity_registry.ENTRIES_BY_NAME["percentage_of_normal"]
    recorder = conftest.NativeRecorder(native)
    monkeypatch.setattr(entry.dispatch, "_native", recorder)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        if cold:
            harness.cold_call_seconds(entry.name)
        else:
            harness.time_entry(entry, repeats=1)
        assert entry.kernels <= recorder.calls
        assert ("error", None, Warning, None, 0) in warnings.filters


def test_context_aware_warnings_rejects_rust_measurements(monkeypatch: pytest.MonkeyPatch) -> None:
    """The native guard forbids context-aware warnings even after filters are cleared."""
    monkeypatch.setattr(harness.sys, "flags", SimpleNamespace(context_aware_warnings=1))
    with pytest.raises(RuntimeError, match="context_aware_warnings"):
        harness.measure(lambda: None, repeats=1)


@pytest.mark.parametrize("repeats", [0, -1])
def test_nonpositive_repeats_rejected(repeats: int) -> None:
    """No warm-up or report is allowed when there are no timed samples."""
    with pytest.raises(ValueError, match="positive"):
        harness.measure(lambda: pytest.fail("warm-up ran"), repeats)
    with pytest.raises(SystemExit) as error:
        harness.main(["--repeat", str(repeats)])
    assert error.value.code == 2


def test_dask_thread_policy_and_process_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh task pool uses the requested policy; process rows have no local observation."""
    import dask
    from dask.base import compute

    def local_compute(*args: Any, scheduler: str, **kwargs: Any) -> tuple[Any, ...]:
        if scheduler == "processes":
            return ({"extension_imported": True, "float_error_policy": ["ignore", "warn"]},)
        return compute(*args, scheduler=scheduler, **kwargs)

    monkeypatch.setattr(dask, "compute", local_compute)
    with np.errstate(all="raise"):
        timings, probe = harness.dask_timings(side=2, repeats=1)
        assert set(np.geterr().values()) == {"raise"}
    reached = {(row.scheduler, row.error_policy): row.rust_kernels_reached for row in timings}
    assert reached == {
        ("threads", "default"): False,
        ("threads", "ignore"): True,
        ("processes", "default"): None,
        ("processes", "ignore"): None,
    }
    assert "| processes | ignore |" in harness.render_dask(timings, probe, side=2)
    assert "| n/a |" in harness.render_dask(timings, probe, side=2)


def test_thread_scaling_reaches_native_in_workers(monkeypatch: pytest.MonkeyPatch, native: ModuleType) -> None:
    """The Rust thread column must establish all-ignore in the task threads too."""
    recorder = conftest.NativeRecorder(native)
    monkeypatch.setattr(harness.compute_module(), "_native", recorder)
    with np.errstate(all="raise"), warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        harness.thread_scaling(workers=(1, 2), cells=4, years=35, repeats=1)
    assert {"gamma_parameters", "gamma_probabilities", "norm_ppf"} <= recorder.calls


@pytest.mark.parametrize("failure", [False, True])
def test_cold_output_and_errors(monkeypatch: pytest.MonkeyPatch, failure: bool) -> None:
    """Trailing logs do not hide JSON, and child failures expose captured stderr."""

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if failure:
            raise subprocess.CalledProcessError(1, ["python"], stderr="child failure details")
        return subprocess.CompletedProcess(["python"], 0, '{"seconds": 0.123}\ntrailing log\n', "")

    monkeypatch.setattr(harness.subprocess, "run", run)
    if failure:
        with pytest.raises(RuntimeError, match="child failure details"):
            harness.measure_cold("spi_gamma")
    else:
        assert harness.measure_cold("spi_gamma") == pytest.approx(0.123)
