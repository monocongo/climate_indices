"""Regression coverage for the session-scoped Palmer sweep helpers in conftest."""

from pathlib import Path

import numpy as np
import pytest

from tests import conftest


def test_failing_division_does_not_abort_the_sweep(monkeypatch):
    """One failing division must leave the remaining divisions computed.

    The session-scoped sweep fixtures replaced per-division calls, so a
    division that raises has to be recorded against its own key instead of
    poisoning the whole fixture and every consumer (PR #935 review).
    """

    def fake_pdsi(precips, pet, awc, *args):
        if awc == "bad":
            raise ValueError("boom")
        return ("result",)

    monkeypatch.setattr("climate_indices.palmer.pdsi", fake_pdsi)

    # Failing division first, so a healthy second division proves the sweep continued.
    inputs = {
        "001": (np.zeros(2), np.zeros(2), "bad"),
        "002": (np.zeros(2), np.zeros(2), "good"),
    }

    results = conftest._palmer_sweep("pdsi", inputs)

    assert results["002"] == ("result",)
    with pytest.raises(RuntimeError, match="palmer.pdsi\\(\\) failed for division 001") as failure:
        results["001"]
    assert isinstance(failure.value.__cause__, ValueError)


def test_process_pool_sweep_matches_serial_and_isolates_failures(palmer_awcs):
    """A pooled sweep must equal the serial one and still isolate a failing division.

    The pool computes in child processes, so nothing here is monkeypatched: a real
    division proves bit-identical results, and an unusable AWC proves that a
    child-side exception is recorded against its own division, in input order.
    """
    root = Path(__file__).parent / "fixture" / "palmer" / "0101"
    good = (np.load(root / "precips.npy"), np.load(root / "pet.npy"), palmer_awcs["0101"])
    inputs = {"001": (good[0], good[1], "bad"), "002": good}

    pooled = conftest._palmer_sweep("pdsi", inputs, max_workers=2)
    serial = conftest._palmer_sweep("pdsi", {"002": good})

    assert list(pooled) == ["001", "002"]
    for pooled_array, serial_array in zip(pooled["002"][:4], serial["002"][:4], strict=True):
        np.testing.assert_array_equal(pooled_array, serial_array)
    with pytest.raises(RuntimeError, match="palmer.pdsi\\(\\) failed for division 001") as failure:
        pooled["001"]
    assert isinstance(failure.value.__cause__, Exception)


def test_sweep_worker_count_is_bounded(monkeypatch):
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setattr(conftest.os, "cpu_count", lambda: 64)
    assert conftest._sweep_workers() == 4
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    assert conftest._sweep_workers() is None


def test_division_input_load_failure_is_recorded_not_raised(tmp_path, monkeypatch):
    """A fixture load failure stays per-division instead of poisoning the session fixture."""
    directory = tmp_path / "001"
    directory.mkdir()  # no precips.npy/pet.npy
    monkeypatch.setattr(conftest, "_palmer_division_dirs", lambda: (str(directory),))

    inputs = conftest.palmer_division_inputs.__wrapped__({"001": 1.0})

    with pytest.raises(RuntimeError, match="failed to load Palmer fixture inputs for division 001") as failure:
        inputs["001"]
    assert isinstance(failure.value.__cause__, OSError)


def test_stored_input_failure_does_not_abort_the_sweep(monkeypatch):
    """A stored fixture-load failure is isolated by the sweep, like a calculation failure."""

    def fake_pdsi(precips, pet, awc, *args):
        return ("result",)

    monkeypatch.setattr("climate_indices.palmer.pdsi", fake_pdsi)

    inputs = conftest._PalmerSweep()
    inputs["001"] = RuntimeError("failed to load Palmer fixture inputs for division 001")
    inputs["002"] = (np.zeros(2), np.zeros(2), "good")

    results = conftest._palmer_sweep("pdsi", inputs)

    assert results["002"] == ("result",)
    with pytest.raises(RuntimeError, match="palmer.pdsi\\(\\) failed for division 001"):
        results["001"]
