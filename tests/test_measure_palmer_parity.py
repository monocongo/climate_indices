"""Regression tests for the Palmer parity measurement and report comparison."""

import json
from pathlib import Path
from types import ModuleType

import pytest

from climate_indices import _native_backend
from climate_indices.drought import palmer
from scripts import measure_palmer_parity


@pytest.mark.parametrize("difference", ["none", "division", "index", "output", "digest", "empty", "missing digest"])
def test_compare_requires_matching_coverage_and_digests(tmp_path: Path, difference: str) -> None:
    stats = {"rust_sha256": "same"}
    entry = {"pdsi": {"pdsi": stats}, "scpdsi": {"pdsi": stats}}
    divisions = {"4002": entry, "4003": entry}
    other = {
        "none": divisions,
        "division": {"4002": entry},
        "index": {"4002": {"pdsi": entry["pdsi"]}, "4003": entry},
        "output": {"4002": {**entry, "pdsi": {**entry["pdsi"], "phdi": stats}}, "4003": entry},
        "digest": {"4002": {**entry, "pdsi": {"pdsi": {"rust_sha256": "different"}}}, "4003": entry},
        "empty": {},
        "missing digest": {"4002": {**entry, "pdsi": {"pdsi": {}}}, "4003": entry},
    }[difference]
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text(json.dumps({"divisions": divisions}), encoding="utf-8")
    second.write_text(json.dumps({"divisions": other}), encoding="utf-8")
    expected = 0 if difference == "none" else 2 if difference in {"empty", "missing digest"} else 1
    assert measure_palmer_parity.compare([first, second], "rust") == expected
    assert measure_palmer_parity.compare([second, first], "rust") == expected


def test_compare_rejects_two_empty_reports(tmp_path: Path) -> None:
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    for report in (first, second):
        report.write_text('{"divisions": {}}', encoding="utf-8")
    assert measure_palmer_parity.compare([first, second], "rust") == 2


def test_measure_rejects_output_outside_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert measure_palmer_parity.measure(Path("../parity.json"), workers=1) == 2
    assert "output path must stay inside" in capsys.readouterr().err
    assert not (tmp_path / "parity.json").exists()


def test_division_rejects_a_backend_that_never_runs_native_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_native_backend, "EXTENSION", ModuleType("stale_extension"))
    previous = palmer._native
    with pytest.raises(RuntimeError, match="4002 pdsi: Rust kernels not called:.*palmer_pdi"):
        measure_palmer_parity._division("4002")
    assert palmer._native is previous
