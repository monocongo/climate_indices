"""Offline regression check for the reliability tutorial notebook."""

from __future__ import annotations

import json
import socket
import warnings
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_reliability_notebook_executes_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the actual cells offline and re-check the documented fixture numbers."""

    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("The reliability notebook must not access the network")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.chdir(ROOT / "notebooks")
    notebook = json.loads((ROOT / "notebooks/reliability_demo.ipynb").read_text())
    namespace: dict[str, Any] = {"__name__": "__reliability_demo_test__"}
    with warnings.catch_warnings():
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), "reliability_demo.ipynb", "exec"), namespace)

    result = namespace["result"]
    assert result.saifi == 0.3
    assert result.saidi == 24.0
    assert result.caidi == 80.0
    assert result.asai == 1 - (24.0 / 60) / 24
    assert result.maifi_e == 0.3
    assert result.maifi == 0.75
    assert result.customer_interruptions == 300
    assert result.customer_minutes_interrupted == 24000.0
    assert result.outage_count == 2
    assert result.momentary_outage_count == 3
    assert namespace["grouped"]["duration"].tolist() == [3.0]
    assert namespace["year"].period_hours == 8784.0
