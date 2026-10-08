"""Smoke tests for the optional Rust extension, ``climate_indices._native``.

The extension is built by ``uv run maturin develop --release`` and is absent from
pure-Python installs, so every test that needs it skips when it cannot be imported,
unless ``CLIMATE_INDICES_REQUIRE_NATIVE=1`` is set (CI's native legs), which makes a
missing extension a failure.
"""

import re
import subprocess
import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from climate_indices import compute, eto, indices, pm_eto
from tests import conftest


def test_native_extension_reports_crate_version() -> None:
    native = conftest.import_native()
    assert re.fullmatch(r"\d+\.\d+\.\d+", native.__version__)


def test_compute_dispatches_to_the_built_extension() -> None:
    native = conftest.import_native()
    assert compute._native is native


def test_pet_dispatches_to_the_built_extension() -> None:
    native = conftest.import_native()
    assert eto._native is native
    assert pm_eto._native is native


def test_pet_python_implementation_runs_when_the_extension_is_absent() -> None:
    """Blocking the import leaves the PET entry points on their Python kernels."""
    code = """
import sys
sys.modules["climate_indices._native"] = None
import numpy as np
from climate_indices import eto, pm_eto
assert eto._native is None and pm_eto._native is None
pet = eto.eto_thornthwaite(np.full(24, 15.0), 40.0, 2001)
assert np.isfinite(pet).all()
daily = np.full(366, 15.0)
hargreaves = eto.eto_hargreaves(daily - 3.0, daily + 5.0, daily, 35.0)
assert np.isfinite(hargreaves).all()
penman = pm_eto.penman_monteith_eto(12.3, 21.5, 50.8, 100.0, 2.78, 187, wind_speed_height_m=10.0)
assert np.isfinite(penman)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


_DAILY = np.random.default_rng(0).normal(15.0, 5.0, 366)


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(lambda: eto.eto_thornthwaite(_DAILY[:24], 40.0, 2001), id="thornthwaite"),
        pytest.param(lambda: eto.eto_hargreaves(_DAILY - 4.0, _DAILY + 4.0, _DAILY, 35.0), id="hargreaves"),
        pytest.param(
            lambda: pm_eto.pm_eto(*(np.full(3, value) for value in (13.3, 0.1, 16.9, 2.1, 2.0, 1.4, 0.1, 0.07))),
            id="pm_eto",
        ),
        pytest.param(
            lambda: pm_eto.penman_monteith_eto(_DAILY - 4.0, _DAILY + 4.0, 40.0, 100.0, 2.0, np.arange(1.0, 367.0)),
            id="fao56_eto",
        ),
    ],
)
def test_pet_uses_python_when_the_extension_predates_pet_kernels(
    monkeypatch: pytest.MonkeyPatch, run: Callable[[], Any]
) -> None:
    """An extension built before the PET kernels leaves the PET entry points on Python."""
    with np.errstate(all="ignore"):
        monkeypatch.setattr(eto, "_native", None)
        monkeypatch.setattr(pm_eto, "_native", None)
        expected = run()
        for module in (compute, eto, pm_eto):
            monkeypatch.setattr(module, "_native", SimpleNamespace())
        assert compute._native_float64(_DAILY)
        actual = run()
    np.testing.assert_array_equal(actual, expected)


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


def test_eddi_uses_python_when_the_extension_predates_eddi_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    values = np.random.default_rng(0).uniform(50.0, 150.0, 360)
    with np.errstate(all="ignore"):
        monkeypatch.setattr(compute, "_native", None)
        expected = indices.eddi(values, 3, 1990, 1990, 2019, compute.Periodicity.monthly)
        monkeypatch.setattr(compute, "_native", SimpleNamespace())
        assert compute._native_float64(values)
        actual = indices.eddi(values, 3, 1990, 1990, 2019, compute.Periodicity.monthly)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kernel", ["tukey_probabilities", "hastings_inverse_normal"])
def test_eddi_propagates_errors_from_present_kernels(monkeypatch: pytest.MonkeyPatch, kernel: str) -> None:
    def fail(*args: object) -> None:
        raise RuntimeError("native kernel failed")

    monkeypatch.setattr(compute, "_native", SimpleNamespace(**{kernel: fail}))
    values = np.ones((2, 1))
    args = (values, values, np.zeros(1)) if kernel == "tukey_probabilities" else (values,)
    native_kernel = getattr(compute, f"_native_{kernel}")
    with np.errstate(all="ignore"), pytest.raises(RuntimeError, match="native kernel failed"):
        native_kernel(*args)


def test_import_native_fails_instead_of_skipping_when_the_extension_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLIMATE_INDICES_REQUIRE_NATIVE=1 is what stops a broken native build from skipping the parity suite."""
    monkeypatch.setitem(sys.modules, "climate_indices._native", None)
    monkeypatch.setenv("CLIMATE_INDICES_REQUIRE_NATIVE", "1")
    with pytest.raises(ImportError):
        conftest.import_native()


def test_import_native_skips_when_the_extension_is_not_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "climate_indices._native", None)
    monkeypatch.delenv("CLIMATE_INDICES_REQUIRE_NATIVE", raising=False)
    with pytest.raises(pytest.skip.Exception):
        conftest.import_native()


@pytest.mark.skipif(sys.version_info < (3, 14), reason="context-aware warnings require Python 3.14")
def test_context_aware_warning_filters_stay_on_python():
    """Routing, not parity: with context-aware warnings, dispatch stays on Python.

    It lives here rather than in test_native_parity.py so that module has no skip other
    than a missing extension, which the native CI legs turn into a failure.
    """
    conftest.import_native()
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "context_aware_warnings=1",
            "-c",
            """
import sys
import warnings
import numpy as np
from climate_indices import compute, _native
assert sys.flags.context_aware_warnings
class NoNativeFit:
    def gamma_parameters(self, *args):
        raise AssertionError("native fit called")
compute._native = NoNativeFit()
with np.errstate(all="ignore"):
    assert not compute._native_float64(np.ones((30, 12)))
with warnings.catch_warnings(), np.errstate(divide="warn"):
    warnings.simplefilter("error", RuntimeWarning)
    try:
        compute.gamma_parameters(np.ones((30, 12)), 1895, 1895, 1924, compute.Periodicity.monthly)
    except RuntimeWarning as error:
        assert "divide by zero" in str(error)
    else:
        raise AssertionError("RuntimeWarning suppressed")
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
