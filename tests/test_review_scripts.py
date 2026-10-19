"""Regression tests for release-support scripts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

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
        report = script.characterize()
    finally:
        logging.disable(previous_disable)
    assert len(report["results"]) == len(script.SCENARIOS) * len(script.STATISTICS) * len(script.LENGTHS)
    assert report["configuration"]["trials"] == 2
    for result in report["results"]:
        assert len(result["truth"]) == len(result["covered"]) == 12
        assert all(0 <= count <= 2 for count in result["covered"])
        assert all(0 <= count <= 2 for count in result["rejected"])
        assert (result["null_gate_passed"] is None) == (result["rho"] != 0)
    assert report["passed"] == all(
        result["coverage_gate_passed"] and result["null_gate_passed"] is not False for result in report["results"]
    )


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
