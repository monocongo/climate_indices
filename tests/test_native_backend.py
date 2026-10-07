"""Smoke tests for the optional Rust extension, ``climate_indices._native``.

The extension is built by ``uv run maturin develop --release`` and is absent from
pure-Python installs, so every test that needs it skips when it cannot be imported.
"""

import re

import pytest


def test_native_extension_reports_crate_version() -> None:
    native = pytest.importorskip("climate_indices._native")
    assert re.fullmatch(r"\d+\.\d+\.\d+", native.__version__)
