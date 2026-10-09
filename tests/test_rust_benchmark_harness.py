"""Runnable check for the Rust-vs-Python benchmark harness (RUST-011)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

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


def test_python_backend_disables_every_dispatch_module() -> None:
    """The Python column is the pure-Python path, not Python orchestration over Rust kernels."""
    modules = {entry.dispatch for entry in parity_registry.ENTRIES}

    with harness.python_backend():
        assert all(module._native is None for module in modules)

    assert all(module._native is not None for module in modules)


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


def test_fixed_and_per_cell_fits_a_known_line() -> None:
    """The fit recovers a fixed cost and a per-cell cost from a synthetic sweep."""
    sweep = [
        harness.Timings(str(cells), "block", 0.1 + 0.002 * cells, 0.3 + 0.004 * cells) for cells in harness.SWEEP_CELLS
    ]

    assert harness.fit_fixed_and_per_cell(sweep, "rust") == pytest.approx((0.1, 0.002))
    assert harness.fit_fixed_and_per_cell(sweep, "python") == pytest.approx((0.3, 0.004))
