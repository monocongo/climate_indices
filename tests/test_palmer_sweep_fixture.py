"""Regression coverage for the session-scoped Palmer sweep helpers in conftest."""

import numpy as np
import pytest

from tests import conftest


def test_failing_division_does_not_abort_the_sweep(tmp_path, monkeypatch):
    """One failing division must leave the remaining divisions computed.

    The session-scoped sweep fixtures replaced per-division calls, so a
    division that raises has to be recorded against its own key instead of
    poisoning the whole fixture and every consumer (PR #935 review).
    """
    for division in ("001", "002"):
        directory = tmp_path / division
        directory.mkdir()
        np.save(directory / "precips.npy", np.zeros(2))
        np.save(directory / "pet.npy", np.zeros(2))
    # Failing division first, so a healthy second division proves the sweep continued.
    dirs = tuple(str(tmp_path / division) for division in ("002", "001"))
    monkeypatch.setattr(conftest, "_palmer_division_dirs", lambda: dirs)

    def fake_pdsi(precips, pet, awc, *args):
        if awc == "bad":
            raise ValueError("boom")
        return ("result",)

    monkeypatch.setattr("climate_indices.palmer.pdsi", fake_pdsi)

    results = conftest._palmer_sweep("pdsi", {"001": "good", "002": "bad"})

    assert results["001"] == ("result",)
    with pytest.raises(RuntimeError, match="palmer.pdsi\\(\\) failed for division 002") as failure:
        results["002"]
    assert isinstance(failure.value.__cause__, ValueError)
