"""Runnable check for the sampled per-cell driver (benchmarks/aws/percell.py)."""

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


percell = _load_script("aws_percell", "benchmarks/aws/percell.py")


def test_stats_pins_median_and_relative_spread() -> None:
    """Median and extremes are exact; the IQR distinguishes a tight sample from a noisy one."""
    result = percell.stats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    assert result["min"] == 1.0
    assert result["median"] == 5.0
    assert result["max"] == 9.0
    assert result["iqr_pct"] == pytest.approx((result["p75"] - result["p25"]) / result["median"] * 100.0)
    assert percell.stats([1.0] * 4 + [1.0001])["iqr_pct"] < percell.stats([1.0, 1.0, 2.0, 3.0, 10.0])["iqr_pct"]


def test_extrapolate_scales_per_cell_cost_linearly() -> None:
    """The full-grid figure is the per-cell cost times the cell count, in seconds."""
    assert percell.extrapolate(0.001, 1000) == pytest.approx(1.0)
    assert percell.extrapolate(1.39e-3, percell.CONUS_LAND_CELLS) == pytest.approx(653.0, rel=0.01)


def test_conus_cell_count_matches_the_prepared_grid() -> None:
    """The extrapolation target is the CONUS land-cell count the grid runs report."""
    assert percell.CONUS_LAND_CELLS == 469_758
    assert percell.CALIBRATION == (1991, 2020)
    assert percell.GRID_AWC_INCHES == 6.0
