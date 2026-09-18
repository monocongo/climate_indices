"""Regression coverage for the session-scoped Palmer sweep helpers in conftest."""

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
