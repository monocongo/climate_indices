"""Runnable check for the AWS benchmark spread driver (benchmarks/aws/spread.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


spread = _load_script("aws_spread", "benchmarks/aws/spread.py")

# The harness this driver times lives in #1322 and is absent from a plain main
# checkout, so that one check skips rather than failing on a ref without it.
needs_harness = pytest.mark.skipif(
    not (ROOT / "benchmarks" / "rust_vs_python.py").exists(),
    reason="requires the RUST-011 harness (perf/1281-rust-benchmarks until #1322 merges)",
)


def test_stats_reports_the_median_extremes_and_a_relative_spread() -> None:
    """Median and extremes are exact; the IQR distinguishes a tight run from a noisy one."""
    result = spread.stats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    assert result["min"] == 1.0
    assert result["median"] == 5.0
    assert result["max"] == 9.0
    assert result["p25"] < result["median"] < result["p75"]
    assert result["iqr_pct"] == pytest.approx((result["p75"] - result["p25"]) / result["median"] * 100.0)

    tight = spread.stats([1.0] * 4 + [1.0001])
    noisy = spread.stats([1.0, 1.0, 2.0, 3.0, 10.0])
    assert tight["iqr_pct"] < noisy["iqr_pct"]


def test_separable_needs_disjoint_quartiles() -> None:
    """A ratio is only measurable when one backend's whole interquartile range clears the other's."""
    assert spread.separable({"p25": 0.9, "p75": 1.1}, {"p25": 9.0, "p75": 11.0})
    assert spread.separable({"p25": 9.0, "p75": 11.0}, {"p25": 0.9, "p75": 1.1})
    assert not spread.separable({"p25": 0.9, "p75": 1.2}, {"p25": 0.95, "p75": 1.25})


def test_load_harness_refuses_a_ref_without_one() -> None:
    """A missing harness reports which ref is needed instead of a bare FileNotFoundError."""
    if (ROOT / "benchmarks" / "rust_vs_python.py").exists():
        pytest.skip("harness is present on this ref")

    with pytest.raises(FileNotFoundError, match="perf/1281-rust-benchmarks"):
        spread.load_harness()


@needs_harness
def test_load_harness_finds_the_registry() -> None:
    """The harness is loaded by path, so an import sorter cannot move it above the path setup."""
    harness = spread.load_harness()

    assert harness.ENTRIES
    assert harness.SAMPLES
