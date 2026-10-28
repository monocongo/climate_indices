"""Icechunk I/O tests: round-trip fidelity, storage parity, and session semantics.

The module skips when the optional extra is absent (Python 3.10 or a base install);
run it with ``pytest -m icechunk``. The opt-in Cloudflare R2 round trip is also marked
``validation``, so the default selection excludes it and it needs credentials in the
environment (see ``docs/development-guide.md``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import dask.config
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import compute, spei, spi
from climate_indices.icechunk_io import open_icechunk, write_icechunk
from climate_indices.indices import Distribution
from tests import conftest

icechunk = pytest.importorskip("icechunk")

pytestmark = pytest.mark.icechunk

# Opt-in remote round trip against the shared Cloudflare R2 (S3-compatible) bucket.
# R2_TEST is a separate switch so exported credentials alone cannot reach the network.
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_TEST = os.environ.get("R2_TEST") == "1"
R2_BUCKET = os.environ.get("R2_BUCKET", "climate-indices")
R2_PREFIX = os.environ.get("R2_PREFIX", "examples/icechunk")


@pytest.fixture(scope="module")
def dataset() -> xr.Dataset:
    """A deterministic 30-year monthly precipitation and PET grid.

    Carries one missing precipitation value and one exact zero, the two inputs whose
    survival through write, commit, and reopen the round trip has to prove.
    """
    values = np.random.default_rng(7).gamma(2.0, 30.0, (360, 2, 2))
    values[0, 0, 0] = np.nan
    values[12, 1, 1] = 0.0
    pet = np.random.default_rng(11).gamma(2.0, 2.0, (360, 2, 2))
    return xr.Dataset(
        {
            "precip": (("time", "lat", "lon"), values, {"units": "mm"}),
            "pet": (("time", "lat", "lon"), pet, {"units": "mm"}),
        },
        coords={"time": pd.date_range("1990-01-01", periods=360, freq="MS"), "lat": [40.0, 41.0], "lon": [0.0, 1.0]},
        attrs={"source": "synthetic precipitation"},
    )


def test_round_trip(dataset: xr.Dataset, icechunk_repo: Any) -> None:
    """Coordinates, attributes, and missing values survive write, commit, and reopen."""
    snapshot = write_icechunk(dataset, icechunk_repo, message="Initial precipitation")
    assert snapshot is not None
    assert snapshot == icechunk_repo.lookup_branch("main")
    with xr.open_zarr(icechunk_repo.readonly_session(branch="main").store, consolidated=False) as direct:
        direct_values = direct.compute()
        xr.testing.assert_identical(direct_values, dataset)
        assert np.isnan(direct_values.precip.values[0, 0, 0])
    with open_icechunk(icechunk_repo) as reopened:
        xr.testing.assert_identical(reopened.compute(), dataset)


def test_import_without_icechunk() -> None:
    code = """
import sys
sys.modules['icechunk'] = None
import climate_indices
from climate_indices.icechunk_io import open_icechunk, write_icechunk
assert callable(climate_indices.spi)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("backend", ["icechunk", "netcdf", "zarr"])
@pytest.mark.parametrize("index", ["spi", "spei"])
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_storage_backend_parity(
    dataset: xr.Dataset,
    icechunk_repo: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    index: str,
    native: bool,
) -> None:
    """SPI and SPEI from Icechunk, NetCDF, and Zarr all match the Python/SciPy oracle."""
    if backend == "icechunk":
        write_icechunk(dataset, icechunk_repo, message="Index input")
        stored = open_icechunk(icechunk_repo)
    elif backend == "netcdf":
        dataset.to_netcdf(tmp_path / "precip.nc", engine="h5netcdf")
        stored = xr.open_dataset(tmp_path / "precip.nc")
    else:
        dataset.to_zarr(tmp_path / "precip.zarr", zarr_format=2, consolidated=True)
        stored = xr.open_zarr(tmp_path / "precip.zarr", consolidated=True)

    def calculate(source: xr.Dataset) -> xr.DataArray:
        chunks = {"time": -1, "lat": 1, "lon": 1}
        if index == "spi":
            return spi(
                source["precip"].chunk(chunks),
                scale=3,
                distribution=Distribution.gamma,
                calibration_year_initial=1990,
                calibration_year_final=2019,
            ).compute(scheduler="synchronous")
        return spei(
            source["precip"].chunk(chunks),
            source["pet"].chunk(chunks),
            scale=3,
            distribution=Distribution.gamma,
            calibration_year_initial=1990,
            calibration_year_final=2019,
        ).compute(scheduler="synchronous")

    conftest.disable_native(monkeypatch)
    with stored as source, np.errstate(all="ignore"):
        expected = calculate(dataset)
        recorder = conftest.NativeRecorder(conftest.import_native()) if native else None
        monkeypatch.setattr(compute, "_native", recorder)
        actual = calculate(source)
        # Storage must preserve values; the stricter native tolerance also applies.
        xr.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
        conftest.assert_native_parity(actual, expected)
        if recorder is not None:
            assert recorder.calls == {"gamma_parameters", "gamma_probabilities", "norm_ppf", "trailing_sum"}
    assert compute._native is recorder


def test_eager_write_uses_to_icechunk(dataset: xr.Dataset, icechunk_repo: Any) -> None:
    """An eager write commits through ``to_icechunk`` and never through ``Dataset.to_zarr``."""
    icechunk_xarray = pytest.importorskip("icechunk.xarray")
    with (
        patch.object(icechunk_xarray, "to_icechunk", wraps=icechunk_xarray.to_icechunk) as writer,
        patch.object(xr.Dataset, "to_zarr", side_effect=AssertionError("Icechunk must not use Dataset.to_zarr")),
    ):
        snapshot = write_icechunk(dataset, icechunk_repo, message="Eager write")
        writer.assert_called_once()
    assert snapshot == icechunk_repo.lookup_branch("main")
    with open_icechunk(icechunk_repo) as reopened:
        xr.testing.assert_identical(reopened.compute(), dataset)


@pytest.mark.parametrize("scheduler", ["threads", "distributed"])
def test_dask_write_uses_to_icechunk(
    dataset: xr.Dataset,
    icechunk_repo_filesystem: Any,
    request: pytest.FixtureRequest,
    scheduler: str,
) -> None:
    """A Dask-backed write commits through ``to_icechunk``, threaded or process-distributed."""
    icechunk_xarray = pytest.importorskip("icechunk.xarray")
    output = spi(
        dataset.precip.chunk({"time": -1, "lat": 1, "lon": 1}),
        scale=3,
        distribution=Distribution.gamma,
        calibration_year_initial=1990,
        calibration_year_final=2019,
    ).to_dataset(name="spi")
    expected = output.compute(scheduler="synchronous")
    with ExitStack() as stack:
        if scheduler == "distributed":
            request.getfixturevalue("dask_client")
        else:
            stack.enter_context(dask.config.set(scheduler="threads"))
        with (
            patch.object(icechunk_xarray, "to_icechunk", wraps=icechunk_xarray.to_icechunk) as writer,
            patch.object(xr.Dataset, "to_zarr", side_effect=AssertionError("Icechunk must not use Dataset.to_zarr")),
        ):
            write_icechunk(output.chunk({"time": 12}), icechunk_repo_filesystem, message="Dask SPI output")
            writer.assert_called_once()
        with open_icechunk(icechunk_repo_filesystem) as reopened:
            xr.testing.assert_identical(reopened.compute(), expected)


def test_malformed_write_preserves_snapshot(dataset: xr.Dataset, icechunk_repo: Any) -> None:
    """A write that a schema mismatch rejects leaves the committed snapshot in place."""
    snapshot = write_icechunk(dataset, icechunk_repo, message="Initial input")
    malformed = dataset.rename({"lon": "x"})
    with pytest.raises(ValueError, match="different dimension names"):
        write_icechunk(malformed, icechunk_repo, message="Malformed update", mode="a")
    assert icechunk_repo.lookup_branch("main") == snapshot
    with open_icechunk(icechunk_repo) as reopened:
        xr.testing.assert_identical(reopened.compute(), dataset)


def test_failed_write_preserves_snapshot(dataset: xr.Dataset, icechunk_repo: Any) -> None:
    """A write that fails before commit leaves the committed snapshot in place."""
    icechunk_xarray = pytest.importorskip("icechunk.xarray")
    snapshot = write_icechunk(dataset, icechunk_repo, message="Initial input")
    writer = icechunk_xarray.to_icechunk

    def fail_after_write(*args: Any, **kwargs: Any) -> None:
        writer(*args, **kwargs)
        raise RuntimeError("Write failed before commit")

    with patch.object(icechunk_xarray, "to_icechunk", side_effect=fail_after_write):
        with pytest.raises(RuntimeError, match="Write failed before commit"):
            write_icechunk(dataset * 2, icechunk_repo, message="Failed update", mode="w")
    assert icechunk_repo.lookup_branch("main") == snapshot
    with open_icechunk(icechunk_repo) as reopened:
        xr.testing.assert_identical(reopened.compute(), dataset)


def test_overwrite_rejected_by_default(dataset: xr.Dataset, icechunk_repo: Any) -> None:
    """A second write without replacement mode raises and leaves the snapshot unchanged."""
    snapshot = write_icechunk(dataset, icechunk_repo, message="Initial input")
    with pytest.raises(FileExistsError):
        write_icechunk(dataset * 2, icechunk_repo, message="Must not replace")
    assert icechunk_repo.lookup_branch("main") == snapshot
    with open_icechunk(icechunk_repo) as reopened:
        xr.testing.assert_identical(reopened.compute(), dataset)


def test_explicit_replacement_preserves_older_snapshot_and_readers(dataset: xr.Dataset, icechunk_repo: Any) -> None:
    """Explicit replacement adds a snapshot; the older snapshot and its readers stay intact."""
    snapshot = write_icechunk(dataset, icechunk_repo, message="Initial input")
    with open_icechunk(icechunk_repo) as original:
        replacement = dataset * 2
        write_icechunk(replacement, icechunk_repo, message="Replace input", mode="w")
        with open_icechunk(icechunk_repo) as reopened:
            xr.testing.assert_identical(reopened.compute(), replacement)
        xr.testing.assert_identical(original.compute(), dataset)
    with xr.open_zarr(icechunk_repo.readonly_session(snapshot_id=snapshot).store, consolidated=False) as previous:
        xr.testing.assert_identical(previous.compute(), dataset)


def test_session_isolation(dataset: xr.Dataset, icechunk_repo: Any) -> None:
    """An uncommitted session is invisible to other sessions and to the branch head."""
    snapshot = write_icechunk(dataset, icechunk_repo, message="Initial input")
    icechunk_repo.create_branch("feature", snapshot)
    writer = icechunk_repo.writable_session("feature")
    icechunk.xarray.to_icechunk(dataset * 2, writer, mode="w")
    assert icechunk_repo.lookup_branch("feature") == snapshot
    with (
        open_icechunk(icechunk_repo) as main,
        open_icechunk(icechunk_repo, branch="feature") as feature,
    ):
        xr.testing.assert_identical(main.compute(), dataset)
        xr.testing.assert_identical(feature.compute(), dataset)
    writer.commit("feature update")
    with open_icechunk(icechunk_repo, branch="feature") as committed:
        xr.testing.assert_identical(committed.compute(), dataset * 2)
    assert icechunk_repo.lookup_branch("main") == snapshot


def test_stale_session_commit_conflicts(dataset: xr.Dataset, icechunk_repo: Any) -> None:
    """A session that commits after the branch moved is rejected instead of clobbering it."""
    write_icechunk(dataset, icechunk_repo, message="Initial input")
    stale = icechunk_repo.writable_session("main")
    icechunk.xarray.to_icechunk(dataset + 1, stale, mode="w")
    snapshot = write_icechunk(dataset * 2, icechunk_repo, message="Concurrent writer", mode="w")
    with pytest.raises(icechunk.ConflictError):
        stale.commit("Stale writer")
    assert icechunk_repo.lookup_branch("main") == snapshot
    with open_icechunk(icechunk_repo) as reopened:
        xr.testing.assert_identical(reopened.compute(), dataset * 2)


@pytest.mark.validation
@pytest.mark.skipif(
    not (R2_TEST and R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY),
    reason="set R2_TEST=1 and the R2 credentials to run the remote round trip",
)
def test_r2_round_trip(dataset: xr.Dataset) -> None:
    """Write, commit, and reopen a snapshot in the shared Cloudflare R2 bucket.

    The run creates its own ``R2_PREFIX/pytest-<id>`` repository and leaves the snapshot
    for inspection, so no other object in the bucket is touched.
    """
    prefix = f"{R2_PREFIX}/pytest-{uuid4().hex}"
    repository = icechunk.Repository.create(
        icechunk.r2_storage(
            bucket=R2_BUCKET,
            prefix=prefix,
            account_id=R2_ACCOUNT_ID,
            access_key_id=R2_ACCESS_KEY_ID,
            secret_access_key=R2_SECRET_ACCESS_KEY,
        )
    )
    snapshot = write_icechunk(dataset, repository, message=f"R2 round trip {prefix}")
    assert snapshot == repository.lookup_branch("main")
    with open_icechunk(repository) as reopened:
        xr.testing.assert_identical(reopened.compute(), dataset)
