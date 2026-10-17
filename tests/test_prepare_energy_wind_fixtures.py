"""Offline checks for the wind fixture generator, without its optional providers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_energy_wind_fixtures.py"


@pytest.fixture
def generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_energy_wind_fixtures", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "result",
    [
        [[0.0, 1.0, 3.5, 12.0, np.nan]],
        [1.0, 1.0, 3.5, 12.0, np.nan],
        [0.0, 1.0, 3.5, 12.0, 1.0],
        [0.0, np.nan, 3.5, 12.0, np.nan],
        [0.0, np.inf, 3.5, 12.0, np.nan],
    ],
    ids=["shape", "calm", "missing-tail", "extra-missing", "infinity"],
)
def test_reject_invalid_oracle_output(generator: ModuleType, result: list[float] | list[list[float]]) -> None:
    speed = np.array([0.0, 1.0, 3.5, 12.0, np.nan])
    with pytest.raises(AssertionError):
        generator._validate_result(np.asarray(result), speed, False)


def test_identity_check_remains_exact(generator: ModuleType) -> None:
    speed = np.array([0.0, 1.0, 3.5, 12.0, np.nan])
    generator._validate_result(speed.copy(), speed, True)
    result = speed.copy()
    result[1] += 1e-14
    generator._validate_result(result, speed, False)
    with pytest.raises(AssertionError):
        generator._validate_result(result, speed, True)


def test_checks_run_with_optimized_python() -> None:
    code = f"""
import runpy
import numpy as np
validate = runpy.run_path({str(SCRIPT)!r})['_validate_result']
speed = np.array([0.0, 1.0, 3.5, 12.0, np.nan])
validate(speed.copy(), speed, True)
for result in (
    speed.reshape(1, -1),
    np.array([1.0, 1.0, 3.5, 12.0, np.nan]),
    np.array([0.0, 1.0, 3.5, 12.0, 1.0]),
    np.array([0.0, np.inf, 3.5, 12.0, np.nan]),
):
    try:
        validate(result, speed, False)
    except AssertionError:
        continue
    raise SystemExit('Invalid oracle output passed validation')
"""
    completed = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _write_pair(directory: Path, payload: bytes) -> None:
    directory.mkdir()
    (directory / "reference.json").write_bytes(payload)
    (directory / "provenance.json").write_text(
        json.dumps({"checksum_sha256": hashlib.sha256(payload).hexdigest()}), encoding="utf-8"
    )


@pytest.mark.parametrize("existing", [False, True])
def test_publish_installs_complete_pair(
    generator: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    target, staging = tmp_path / "fixtures", tmp_path / "staging"
    if existing:
        _write_pair(target, b"old")
    _write_pair(staging, b"new")
    monkeypatch.setattr(generator, "FIXTURE_DIR", target)
    generator._publish(staging)
    assert (target / "reference.json").read_bytes() == b"new"
    provenance = json.loads((target / "provenance.json").read_text())
    assert provenance["checksum_sha256"] == hashlib.sha256(b"new").hexdigest()
    assert not staging.exists()
    assert not target.with_name(".fixtures-backup").exists()


@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
def test_publish_restores_previous_pair(
    generator: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[BaseException]
) -> None:
    target, staging = tmp_path / "fixtures", tmp_path / "staging"
    _write_pair(target, b"old")
    _write_pair(staging, b"new")
    previous = {path.name: path.read_bytes() for path in target.iterdir()}
    monkeypatch.setattr(generator, "FIXTURE_DIR", target)
    replace = generator.os.replace

    def interrupted_replace(source: Path, destination: Path) -> None:
        if source == staging:
            raise error("interrupted publication")
        replace(source, destination)

    monkeypatch.setattr(generator.os, "replace", interrupted_replace)
    with pytest.raises(error, match="interrupted publication"):
        generator._publish(staging)
    assert {path.name: path.read_bytes() for path in target.iterdir()} == previous


def test_publish_recovers_backup_before_failed_retry(
    generator: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "fixtures"
    backup = tmp_path / ".fixtures-backup"
    _write_pair(backup, b"old")
    previous = {path.name: path.read_bytes() for path in backup.iterdir()}
    monkeypatch.setattr(generator, "FIXTURE_DIR", target)
    with pytest.raises(FileNotFoundError):
        generator._publish(tmp_path / "missing-staging")
    assert {path.name: path.read_bytes() for path in target.iterdir()} == previous
    assert not backup.exists()


def test_wind_source_urls_are_separate() -> None:
    path = SCRIPT.parents[1] / "tests" / "fixture" / "energy_wind" / "provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    assert provenance["url"] == "https://github.com/Ouranosinc/xclim/tree/v0.62.0"
    assert "https://github.com/wind-python/windpowerlib/tree/v0.2.2" in provenance["citation"]
