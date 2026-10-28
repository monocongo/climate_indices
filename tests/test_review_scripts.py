"""Regression tests for release-support scripts."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import sys
import weakref
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generate_llms_txt = _load_script(
    "generate_llms_txt",
    "scripts/generate_llms_txt.py",
)
try:
    prepare_noaa_eddi_fixtures: ModuleType | None = _load_script(
        "prepare_noaa_eddi_fixtures",
        "scripts/prepare_noaa_eddi_fixtures.py",
    )
except ModuleNotFoundError:
    # `requests` is declared in the script's own PEP 723 metadata, not the
    # project's `test` dependency group, so it is absent under `uv sync
    # --group test` (e.g. the minimum-dependencies CI job).
    prepare_noaa_eddi_fixtures = None

characterize_compound_bootstrap = _load_script(
    "characterize_compound_bootstrap",
    "scripts/characterize_compound_bootstrap.py",
)
measure_compound_xarray_memory = _load_script(
    "measure_compound_xarray_memory",
    "scripts/measure_compound_xarray_memory.py",
)


def test_characterization_output_accepts_only_json_below_design_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "docs" / "design"
    monkeypatch.setattr(characterize_compound_bootstrap, "OUTPUT_ROOT", root)
    monkeypatch.chdir(tmp_path)
    assert characterize_compound_bootstrap._output_path("docs/design/nested/report.json") == root / "nested/report.json"
    for path in (
        tmp_path / "outside.json",
        root / "../outside.json",
        root / "nested/../../outside.json",
        root / "report.txt",
    ):
        with pytest.raises(argparse.ArgumentTypeError, match="JSON file below"):
            characterize_compound_bootstrap._output_path(str(path))


@pytest.mark.parametrize("directory_link", [True, False])
def test_characterization_output_rejects_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory_link: bool
) -> None:
    root = tmp_path / "design"
    root.mkdir()
    outside = tmp_path / ("outside" if directory_link else "outside.json")
    link = root / ("link" if directory_link else "link.json")
    link.symlink_to(outside, target_is_directory=directory_link)
    monkeypatch.setattr(characterize_compound_bootstrap, "OUTPUT_ROOT", root)
    with pytest.raises(argparse.ArgumentTypeError, match="JSON file below"):
        characterize_compound_bootstrap._output_path(str(link / "report.json" if directory_link else link))


def test_compare_versions_local_ref_rejects_option_like_values() -> None:
    """--local-ref reaches git through uv, so option-like or URL-breaking refs must fail at parse time."""
    pytest.importorskip("matplotlib")  # the script's own PEP 723 dependency, not in the test group
    compare_versions = _load_script("compare_versions", "scripts/compare_versions.py")
    for ref in ("HEAD", "main", "v2.3.0", "origin/fix/issue-1", "HEAD~1", "abc123^"):
        assert compare_versions._git_ref(ref) == ref
    for ref in ("", "-x", "--upload-pack=touch /tmp/pwned", "main@evil", "a b"):
        with pytest.raises(argparse.ArgumentTypeError, match="invalid git ref"):
            compare_versions._git_ref(ref)


def test_characterization_writes_a_validated_json_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(characterize_compound_bootstrap, "OUTPUT_ROOT", tmp_path)
    output = tmp_path / "nested" / "report.json"
    report = {"passed": False, "results": []}
    characterize_compound_bootstrap._write_report(output, report)
    assert json.loads(output.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize("directory_link", [True, False])
def test_characterization_rechecks_output_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory_link: bool
) -> None:
    root = tmp_path / "design"
    root.mkdir()
    monkeypatch.setattr(characterize_compound_bootstrap, "OUTPUT_ROOT", root)
    output = characterize_compound_bootstrap._output_path(root / "nested" / "report.json")
    outside = tmp_path / ("outside" if directory_link else "outside.json")
    link = root / "nested" if directory_link else output
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=directory_link)
    with pytest.raises(argparse.ArgumentTypeError, match="JSON file below"):
        characterize_compound_bootstrap._write_report(output, {})
    assert not outside.exists()


def test_characterization_reduced_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    script = characterize_compound_bootstrap
    monkeypatch.setattr(script, "TRIALS", 2)
    monkeypatch.setattr(script, "YEARS", 60)
    monkeypatch.setattr(script, "RESAMPLES", 9)
    previous_disable = logging.root.manager.disable
    try:
        report = script.characterize(groups=("replication",))
    finally:
        logging.disable(previous_disable)
    assert len(report["results"]) == len(script.SCENARIOS) * len(script.STATISTICS) * len(script.LENGTHS)
    assert report["configuration"]["trials"] == 2
    assert {scenario["scenario"] for scenario in report["scenarios"]} == {name for name, _, _ in script.SCENARIOS}
    for result in report["results"]:
        assert len(result["truth"]) == len(result["covered"]) == 12
        assert all(0 <= count <= 2 for count in result["covered"])
        assert all(0 <= count <= 2 for count in result["rejected"])
        assert (result["null_gate_passed"] is None) == (result["rho"] != 0)
    assert report["passed"] == all(
        result["coverage_gate_passed"] and result["null_gate_passed"] is not False for result in report["results"]
    )
    # A subset run names the groups it requested and executed.
    assert report["complete_plan"] is False
    assert report["configuration"]["requested_groups"] == ["replication"]
    assert report["configuration"]["executed_groups"] == ["replication"]


def test_characterization_never_passes_without_measured_cases() -> None:
    script = characterize_compound_bootstrap
    previous_disable = logging.root.manager.disable
    try:
        report = script.characterize(groups=())
    finally:
        logging.disable(previous_disable)
    # An empty selection must not pass through `all([])`.
    assert report["results"] == []
    assert report["passed"] is False
    assert report["complete_plan"] is False
    assert report["configuration"]["requested_groups"] == []
    assert report["configuration"]["executed_groups"] == []


def test_characterization_plan_is_predeclared_and_streams_are_unique() -> None:
    script = characterize_compound_bootstrap
    scenarios = script._plan()
    assert [scenario.group for scenario in scenarios] == [
        *(["replication"] * 4),
        *(["record_length"] * 6),
        *(["persistence"] * 4),
        *(["lagged_conditioning"] * 6),
        *(["hazard_frequency"] * 9),
        *(["dependent_hazards"] * 3),
        *(["default_resamples"] * 4),
    ]
    # A repeated stream index would silently reuse another scenario's draws.
    assert len({scenario.stream for scenario in scenarios}) == len(scenarios)
    assert [(scenario.moving_average_months, scenario.rho) for scenario in scenarios[:4]] == [
        (width, rho) for _, width, rho in script.SCENARIOS
    ]
    assert {scenario.resamples for scenario in scenarios if scenario.group == "default_resamples"} == {1999}
    assert all(
        scenario.resamples == script.RESAMPLES for scenario in scenarios if scenario.group != "default_resamples"
    )
    assert {scenario.lag_window for scenario in scenarios if scenario.group == "lagged_conditioning"} == {
        (0, 1),
        (0, 3),
        (1, 3),
    }
    # Lagged conditioning is defined for the conditional statistic only.
    assert all(
        scenario.statistics == ("conditional_probability",) for scenario in scenarios if scenario.lag_window != (0, 0)
    )
    short = [scenario for scenario in scenarios if scenario.group == "record_length"]
    assert {scenario.years for scenario in short} == {30, 60, 100}
    # Every record-length scenario also requests the ten-block edge at six-year blocks.
    assert all(6 in scenario.lengths for scenario in short)


def test_lagged_truth_matches_the_closed_form_at_zero_lag() -> None:
    """The simulation reference agrees with quadrature, which validates it for nonzero lags."""
    script = characterize_compound_bootstrap
    scenario = script.Scenario(
        "check", "lagged_conditioning", 0, 1, 0.0, "seasonal", 200, 399, (0, 0), script.CONDITIONAL, (6,)
    )
    analytic = script._analytic_truths(scenario)["conditional_probability"]
    np.testing.assert_allclose(script._lagged_truth(scenario), analytic, rtol=0, atol=0.02)


def test_structural_guard_months_are_excluded_from_the_gates() -> None:
    script = characterize_compound_bootstrap
    scenario = script._plan()[0]
    tail = 0.01 / 864
    case = script.Case(
        covered=np.full(12, script.TRIALS, dtype=np.int64),
        rejected=np.zeros(12, dtype=np.int64),
        intervals=np.full(12, script.TRIALS, dtype=np.int64),
        p_values=np.full(12, script.TRIALS, dtype=np.int64),
        structural=np.zeros(12, dtype=np.int64),
        few_blocks=np.zeros(12, dtype=np.int64),
        length=6,
        decisions=np.zeros((script.TRIALS, 12), dtype=bool),
    )
    truth = np.full(12, 0.5)
    result = script._case_result(scenario, "joint_probability", 6, case, truth, tail)
    assert result["coverage_gate_passed"] is True
    assert result["null_gate_passed"] is True
    assert result["minimum_covered_records"] == [109] * 12
    assert result["maximum_null_rejections"] == [19] * 12
    case.covered[0] = 100
    assert script._case_result(scenario, "joint_probability", 6, case, truth, tail)["coverage_gate_passed"] is False
    # A month the library refused on every record is not inferable, so it leaves the gate.
    case.structural[0] = script.TRIALS
    result = script._case_result(scenario, "joint_probability", 6, case, truth, tail)
    assert result["coverage_gate_passed"] is True
    assert result["structural_exclusions"][0] == script.TRIALS
    assert result["inferable_records"][0] == 0
    # A partial exclusion judges the month on the records that carry an inference.
    case.covered[0] = script.TRIALS
    case.structural[1] = 64
    case.covered[1] = 61
    result = script._case_result(scenario, "joint_probability", 6, case, truth, tail)
    assert result["inferable_records"][1] == 64
    assert result["minimum_covered_records"][1] < 61
    assert result["coverage_gate_passed"] is True
    assert result["null_gate_passed"] is True
    # Each Wilson interval uses the same eligible denominator as its count.
    interval = result["coverage_wilson_95"][1]
    assert interval is not None
    assert interval[0] < 61 / 64 < interval[1]
    # A month with no inferable record has no interval to report.
    assert result["coverage_wilson_95"][0] is None
    assert result["rejection_wilson_95"][0] is None
    # A missing p-value fails the null gate unless the month carries a structural guard.
    case.p_values[2] = script.TRIALS - 1
    assert script._case_result(scenario, "joint_probability", 6, case, truth, tail)["null_gate_passed"] is False


def test_memory_budget_reproduces_the_design_figures() -> None:
    """The planning formula gives the ~1.97 GB daily and ~81 MB monthly figures that ADR-0022 quotes."""
    script = measure_compound_xarray_memory
    assert script._budget(36_524, 1_000) == 54 * 36_524 * 1_000
    assert round(script._budget(36_524, 1_000) / 1e9, 2) == 1.97
    assert script._budget(1_200, 1_000, resamples=1_999) == 54 * 1_200 * 1_000 + 8 * 1_999 * 1_000
    assert round(script._budget(1_200, 1_000, resamples=1_999) / 1e6) == 81


@pytest.mark.parametrize("prepared", [False, True])
def test_memory_measurement_keeps_only_counted_inputs_live(monkeypatch: pytest.MonkeyPatch, prepared: bool) -> None:
    """Release the float source payloads before measuring a prepared Boolean Dataset."""
    script = measure_compound_xarray_memory
    sources = script._sources
    payloads: list[weakref.ReferenceType[np.ndarray]] = []

    def tracked_sources(time: np.ndarray, cells: int, seed: int) -> dict[str, Any]:
        arrays: dict[str, Any] = sources(time, cells, seed)
        payloads[:] = [weakref.ref(value) for array in arrays.values() for value in array.data.dask.values()]
        return arrays

    def peak(stage: Any, input_bytes: int) -> int:
        gc.collect()
        assert payloads
        assert all((payload() is None) == prepared for payload in payloads)
        return input_bytes + 100

    monkeypatch.setattr(script, "_sources", tracked_sources)
    monkeypatch.setattr(script, "_peak", peak)
    time = np.arange("2000-01", "2001-01", dtype="datetime64[M]")
    script._measure(
        "test",
        (1, 2),
        len(time),
        time,
        lambda arrays: lambda: arrays,
        cadence="monthly",
        prepared=prepared,
    )


def test_memory_measurement_quick_report_is_complete() -> None:
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        report = measure_compound_xarray_memory.measure(quick=True)
    finally:
        logging.disable(previous_disable)
    assert report["quick"] is True
    assert [stage["stage"] for stage in report["stages"]] == [
        "harmonize",
        "joint_statistics",
        "conditional_probability",
        "bootstrap_dependence",
    ]
    for stage in report["stages"]:
        widths = len(stage["cells"])
        assert len(stage["peak_bytes"]) == len(stage["budget_bytes"]) == len(stage["input_bytes"]) == widths
        # the traced peak includes the stage's own working memory, so it exceeds the inputs alone
        assert all(peak > inputs for peak, inputs in zip(stage["peak_bytes"], stage["input_bytes"], strict=True))
        assert stage["bytes_per_cell"] > 0
        assert stage["fixed_bytes_per_task"] >= 0
        assert set(stage["max_cells_per_task"]) == {"256 MiB", "1 GiB"}
        assert stage["max_cells_per_task"]["1 GiB"] >= stage["max_cells_per_task"]["256 MiB"]
    assert json.loads(json.dumps(report)) == report


requires_requests = pytest.mark.skipif(
    prepare_noaa_eddi_fixtures is None,
    reason="requests not installed (outside prepare_noaa_eddi_fixtures.py's own dependency declaration)",
)


@requires_requests
def test_download_master_table_limits_requests_to_noaa(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixture downloader must not follow URLs outside NOAA PSL."""
    assert prepare_noaa_eddi_fixtures is not None
    post = Mock(return_value=Mock(text='<iframe src="/eddi/result">'))
    get = Mock(side_effect=[Mock(text='<a href="/eddi/master.table">'), Mock(text="table")])
    monkeypatch.setattr(prepare_noaa_eddi_fixtures.requests, "post", post)
    monkeypatch.setattr(prepare_noaa_eddi_fixtures.requests, "get", get)

    assert prepare_noaa_eddi_fixtures._download_master_table() == "table"

    calls = [post.call_args, *get.call_args_list]
    assert [call.args[0] for call in calls] == [
        "https://psl.noaa.gov/cgi-bin/eddi_int/eddi.ts.pl",
        "https://psl.noaa.gov/eddi/result",
        "https://psl.noaa.gov/eddi/master.table",
    ]
    assert all(call.kwargs["allow_redirects"] is False for call in calls)


@requires_requests
def test_download_master_table_rejects_external_iframe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixture downloader must reject an iframe outside NOAA PSL."""
    assert prepare_noaa_eddi_fixtures is not None
    post = Mock(return_value=Mock(text='<iframe src="https://example.com/result">'))
    get = Mock()
    monkeypatch.setattr(prepare_noaa_eddi_fixtures.requests, "post", post)
    monkeypatch.setattr(prepare_noaa_eddi_fixtures.requests, "get", get)

    with pytest.raises(ValueError, match="NOAA URL must use https://psl.noaa.gov"):
        prepare_noaa_eddi_fixtures._download_master_table()

    get.assert_not_called()


def test_read_missing_llms_source_names_file_and_source_lists() -> None:
    """Missing llms.txt source documents should report the bad path and lists."""
    missing_path = "docs/does-not-exist.md"

    with pytest.raises(FileNotFoundError) as exc_info:
        generate_llms_txt._read(missing_path)

    message = str(exc_info.value)
    assert missing_path in message
    assert "SUMMARY_FILES" in message
    assert "FULL_FILES" in message


@pytest.mark.parametrize(
    ("output", "sources"),
    [
        (Path("llms.txt"), generate_llms_txt.SUMMARY_FILES),
        (Path("llms-full.txt"), generate_llms_txt.FULL_FILES),
    ],
)
def test_llms_bundle_matches_configured_sources(output: Path, sources: list[str]) -> None:
    """Committed llms bundles should exactly match their configured sources."""
    assert (ROOT / output).read_text(encoding="utf-8") == generate_llms_txt._render(sources), (
        f"{output} is stale relative to {sources}. Regenerate with: uv run scripts/generate_llms_txt.py"
    )


def test_section_resolves_relative_links_from_repository_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bundled links should resolve from the repository root, not the source directory."""
    monkeypatch.setattr(generate_llms_txt, "ROOT", tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text(
        "[ADR](adr/0003.md) [root](../README.md) [web](https://example.com) [anchor](#chunking)\n"
        '[titled](adr/0004.md "ADR title")\n',
        encoding="utf-8",
    )

    section = generate_llms_txt._section("docs/guide.md")

    assert "[ADR](docs/adr/0003.md)" in section
    assert "[root](README.md)" in section
    assert "[web](https://example.com)" in section
    assert "[anchor](#chunking)" in section
    assert '[titled](docs/adr/0004.md "ADR title")' in section


def test_render_builds_header_and_ordered_sections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_render should emit the fixed preamble followed by sections in source order."""
    monkeypatch.setattr(generate_llms_txt, "ROOT", tmp_path)
    (tmp_path / "a.md").write_text("Content A\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("Content B\n", encoding="utf-8")

    result = generate_llms_txt._render(["a.md", "b.md"])
    lines = result.splitlines()

    assert lines[0] == "# climate_indices"
    assert lines[1] == ""
    assert lines[2] == ("Reference implementations of climate and drought indices for scientific monitoring workflows.")
    assert lines[3] == ""
    assert lines[4] == "Generated by `uv run scripts/generate_llms_txt.py` from committed repository documentation."
    assert "## a.md\n\nContent A\n" in result
    assert "## b.md\n\nContent B\n" in result
    assert result.index("## a.md") < result.index("## b.md")


def test_render_strips_trailing_blank_lines_and_whitespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_render should end with exactly one trailing newline and no trailing whitespace."""
    monkeypatch.setattr(generate_llms_txt, "ROOT", tmp_path)
    (tmp_path / "doc.md").write_text("Trailing space line.   \n\n\n", encoding="utf-8")

    result = generate_llms_txt._render(["doc.md"])

    assert "Trailing space line.   " not in result
    assert "Trailing space line." in result
    assert result.endswith("\n")
    assert not result.endswith("\n\n")


def test_render_preserves_internal_blank_lines_between_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_render must preserve blank lines inside a source file, not just strip trailing ones."""
    monkeypatch.setattr(generate_llms_txt, "ROOT", tmp_path)
    (tmp_path / "doc.md").write_text("Para one.\n\nPara two.\n", encoding="utf-8")

    result = generate_llms_txt._render(["doc.md"])

    assert "## doc.md\n\nPara one.\n\nPara two.\n" in result


def test_render_repeats_section_for_duplicate_source_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_render must not deduplicate sources; each entry renders its own section."""
    monkeypatch.setattr(generate_llms_txt, "ROOT", tmp_path)
    (tmp_path / "doc.md").write_text("Body.\n", encoding="utf-8")

    result = generate_llms_txt._render(["doc.md", "doc.md"])

    assert result.count("## doc.md") == 2


def test_render_with_no_sources_returns_only_preamble(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty source list should still render the fixed preamble with no sections."""
    monkeypatch.setattr(generate_llms_txt, "ROOT", tmp_path)

    result = generate_llms_txt._render([])

    assert "## " not in result
    assert result.startswith("# climate_indices")
    assert result.endswith("\n")
    assert not result.endswith("\n\n")


def test_render_propagates_missing_source_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_render should surface the same FileNotFoundError raised by _read."""
    monkeypatch.setattr(generate_llms_txt, "ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="missing.md"):
        generate_llms_txt._render(["missing.md"])


def test_write_persists_render_output_to_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_write should write exactly what _render produces for the same sources."""
    monkeypatch.setattr(generate_llms_txt, "ROOT", tmp_path)
    (tmp_path / "doc.md").write_text("Hello, world.\n", encoding="utf-8")

    generate_llms_txt._write("out.txt", ["doc.md"])

    written = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert written == generate_llms_txt._render(["doc.md"])
