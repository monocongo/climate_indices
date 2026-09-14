"""Regression tests for E2E input generation publication."""

import hashlib
import importlib.util
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _prepare_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location("prepare_e2e_inputs_test", SCRIPTS_DIR / "prepare_e2e_inputs.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_downloads_use_private_temporary_files(tmp_path, monkeypatch):
    """Concurrent cold-cache downloads must not share an in-progress file."""
    module = _prepare_module(monkeypatch)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    payload = b"source bytes"
    checksum = hashlib.sha256(payload).hexdigest()
    barrier = threading.Barrier(2)

    def fake_urlopen(*args, **kwargs):
        barrier.wait(timeout=5)
        return io.BytesIO(payload)

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    with ThreadPoolExecutor(max_workers=2) as executor:
        downloads = list(executor.map(lambda _: module._cache_source(source_dir, "prcp", checksum), range(2)))

    assert all(path.read_bytes() == payload for path in downloads)
    assert not list(source_dir.glob("*.download"))


def test_failed_regeneration_preserves_current_generation(tmp_path, monkeypatch):
    """Only a fully validated generation may replace the current inputs."""
    pytest.importorskip("h5py")
    pytest.importorskip("zarr")
    module = _prepare_module(monkeypatch)
    times = pd.date_range("1980-01-01", "2016-12-01", freq="MS")
    payloads = {}
    for name, value in (("prcp", 50.0), ("pet", 10.0)):
        source_path = tmp_path / f"nclimgrid_lowres_{name}.nc"
        dataset = xr.Dataset(
            {name: (("time", "lat", "lon"), np.full((len(times), 1, 1), value))},
            coords={"time": times, "lat": [35.0], "lon": [-100.0]},
        )
        dataset[name].attrs["units"] = "mm"
        dataset.to_netcdf(source_path, engine="h5netcdf")
        payloads[source_path.name] = source_path.read_bytes()

    def fake_urlopen(url, **kwargs):
        return io.BytesIO(payloads[Path(url).name])

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        module,
        "SOURCES",
        {
            "prcp": hashlib.sha256(payloads["nclimgrid_lowres_prcp.nc"]).hexdigest(),
            "pet": hashlib.sha256(payloads["nclimgrid_lowres_pet.nc"]).hexdigest(),
        },
    )
    output_dir = tmp_path / "e2e"
    module.prepare_inputs(output_dir)
    current = output_dir / "current"
    published = current.resolve()
    manifest = json.loads((published / "manifest.json").read_text())
    assert current.is_symlink()

    def fail(*args, **kwargs):
        raise RuntimeError("simulated generation failure")

    monkeypatch.setattr(module, "clean_and_prepare_inputs", fail)
    with pytest.raises(RuntimeError, match="simulated generation failure"):
        module.prepare_inputs(output_dir)

    assert current.resolve() == published
    assert json.loads((current / "manifest.json").read_text()) == manifest
    assert all(
        (current / name).exists() for name in ("raw_precipitation.nc", "raw_pet.nc", "cache_prepared_input.zarr")
    )
