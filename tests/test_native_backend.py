"""Smoke tests for the optional Rust extension, ``climate_indices._native``.

The extension is built by ``uv run maturin develop --release`` and is absent from
pure-Python installs, so every test that needs it skips when it cannot be imported.
"""

import re
import subprocess
import sys

import pytest

from climate_indices import compute


def test_native_extension_reports_crate_version() -> None:
    native = pytest.importorskip("climate_indices._native")
    assert re.fullmatch(r"\d+\.\d+\.\d+", native.__version__)


def test_compute_dispatches_to_the_built_extension() -> None:
    native = pytest.importorskip("climate_indices._native")
    assert compute._native is native


def test_python_implementation_runs_when_the_extension_is_absent() -> None:
    """Blocking the import leaves compute on its Python kernels, and SPI still runs.

    A subprocess keeps the blocked import from leaking into this interpreter.
    """
    code = """
import sys
sys.modules["climate_indices._native"] = None
import numpy as np
from climate_indices import compute, indices
assert compute._native is None
values = np.random.default_rng(0).gamma(2.0, 30.0, 360)
result = indices.spi(values, 3, indices.Distribution.gamma, 1990, 1990, 2019, compute.Periodicity.monthly)
assert np.isfinite(result[2:]).all() and np.isnan(result[:2]).all()
"""
    subprocess.run([sys.executable, "-c", code], check=True)
