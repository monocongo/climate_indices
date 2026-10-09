"""End-to-end parity of the Rust backend at the orchestration surfaces.

The registry suite (``tests/test_native_parity_registry.py``) proves parity kernel
by kernel. These tests prove the whole call path -- the xarray adapter, the
threaded and distributed Dask schedulers, the CLI, and ``fit_diagnostics`` --
produces the same result with the extension installed as it does without it, so
orchestration cannot mask a divergence the kernels do not have.

Skipped when the extension is not built, unless ``CLIMATE_INDICES_REQUIRE_NATIVE=1``
is set, as in CI's native legs, where a missing extension is a collection error.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import compute, fit_diagnostics, indices, spi
from climate_indices.__main__ import main
from tests import conftest

native = conftest.import_native()

# a synthetic station record: 36 years of monthly precipitation from 1981
_YEARS = 36
_START_YEAR = 1981
_CALIBRATION_START = 1983
_CALIBRATION_END = _START_YEAR + _YEARS - 1


def _precipitation(years: int = _YEARS, cells: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(20261010)
    values = rng.gamma(2.0, 40.0, size=(years * 12, cells) if cells else years * 12)
    values.reshape(-1)[::53] = 0.0
    return values


def _series(cells: int | None = None) -> xr.DataArray:
    values = _precipitation(cells=cells)
    time = pd.date_range(f"{_START_YEAR}-01-01", periods=values.shape[0], freq="MS")
    dims = ("time", "cell") if cells else ("time",)
    coords: dict[str, Any] = {"time": time}
    if cells:
        coords["cell"] = np.arange(values.shape[1])
    return xr.DataArray(values, dims=dims, coords=coords, name="precip", attrs={"units": "mm"})


def _run_both(monkeypatch: pytest.MonkeyPatch, run: Callable[[], Any]) -> tuple[Any, Any, set[str]]:
    """Run once with the extension behind a recorder, once with every dispatch module on Python."""
    recorder = conftest.NativeRecorder(conftest.import_native())
    with np.errstate(all="ignore"), monkeypatch.context() as patch:
        for module in (compute,):
            patch.setattr(module, "_native", recorder)
        rust = run()
        conftest.disable_native(patch)
        python = run()
    return rust, python, recorder.calls


def _spi(values: Any, **kwargs: Any) -> Any:
    return spi(
        values,
        6,
        indices.Distribution.gamma,
        _START_YEAR,
        _CALIBRATION_START,
        _CALIBRATION_END,
        compute.Periodicity.monthly,
        **kwargs,
    )


def test_xarray_adapter_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The xarray adapter returns the same DataArray on both backends."""
    rust, python, calls = _run_both(monkeypatch, lambda: _spi(_series()))
    assert calls == {"gamma_parameters", "gamma_probabilities", "norm_ppf"}
    conftest.assert_native_parity(rust, python)


def test_threaded_dask_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chunked, Dask-backed input computes to the same result on both backends."""
    chunked = _series().chunk({"time": -1})
    rust, python, calls = _run_both(monkeypatch, lambda: _spi(chunked).compute())
    assert calls == {"gamma_parameters", "gamma_probabilities", "norm_ppf"}
    conftest.assert_native_parity(rust, python)


def test_distributed_dask_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """A distributed scheduler with real workers computes the same result as the Python reference.

    The recorder cannot witness this one: a distributed client serializes the task
    graph before the workers run it, so the in-process dispatch patch never reaches
    a task. The native run is therefore the installed extension dispatching as
    usual, and it is compared against the pure-Python path computed in-process.
    """
    distributed = pytest.importorskip("distributed")
    assert compute._native is not None, "the distributed run has to dispatch to the extension"
    chunked = _series(cells=3).chunk({"time": -1, "cell": 1})
    with distributed.Client(n_workers=2, threads_per_worker=1, processes=False, dashboard_address=None):
        rust = _spi(chunked).compute()
    with np.errstate(all="ignore"), monkeypatch.context() as patch:
        conftest.disable_native(patch)
        python = _spi(chunked).compute()
    conftest.assert_native_parity(rust, python)


def test_fit_diagnostics_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public fitting diagnostics report the same fit on both backends."""
    rust, python, calls = _run_both(
        monkeypatch,
        lambda: fit_diagnostics(
            _series(),
            6,
            indices.Distribution.gamma,
            _START_YEAR,
            _CALIBRATION_START,
            _CALIBRATION_END,
            compute.Periodicity.monthly,
        ),
    )
    assert calls == {"gamma_parameters"}
    conftest.assert_native_parity(rust, python)


def test_cli_parity(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The CLI writes the same NetCDF values with the extension as without it."""
    time = pd.date_range(f"{_START_YEAR}-01-01", periods=_YEARS * 12, freq="MS")
    input_path = tmp_path / "precip.nc"
    xr.Dataset(
        {"precip": ("time", _precipitation(), {"units": "mm"})},
        coords={"time": time},
    ).to_netcdf(input_path)

    def arguments(output_base: str) -> list[str]:
        return [
            "--index",
            "spi",
            "--periodicity",
            "monthly",
            "--calibration_start_year",
            str(_CALIBRATION_START),
            "--calibration_end_year",
            str(_CALIBRATION_END),
            "--netcdf_precip",
            str(input_path),
            "--var_name_precip",
            "precip",
            "--output_file_base",
            output_base,
            "--scales",
            "6",
            "--multiprocessing",
            "single",
        ]

    with np.errstate(all="ignore"), monkeypatch.context() as patch:
        main(arguments(str(tmp_path / "native")))
        with xr.open_dataset(tmp_path / "native_spi_gamma_06.nc", mask_and_scale=False) as dataset:
            rust = dataset["spi_gamma_06"].values
        conftest.disable_native(patch)
        main(arguments(str(tmp_path / "python")))
        with xr.open_dataset(tmp_path / "python_spi_gamma_06.nc", mask_and_scale=False) as dataset:
            python = dataset["spi_gamma_06"].values

    assert np.isnan(rust).sum() < rust.size, "the CLI run produced nothing to compare"
    conftest.assert_native_parity(rust, python)
