"""Regression tests for release-support scripts."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

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


create_github_issues = _load_script(
    "create_github_issues",
    "scripts/create_github_issues.py",
)
generate_llms_txt = _load_script(
    "generate_llms_txt",
    "scripts/generate_llms_txt.py",
)


def test_get_issue_labels_raises_on_gh_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue label lookup should fail fast when gh cannot return labels."""

    def fail_run_gh(*_args: object, **_kwargs: object) -> str:
        raise subprocess.CalledProcessError(1, ["gh", "issue", "view"])

    monkeypatch.setattr(create_github_issues, "run_gh", fail_run_gh)

    with pytest.raises(RuntimeError, match="Failed to fetch labels for issue #42"):
        create_github_issues.get_issue_labels(42)


def test_get_issue_labels_raises_on_invalid_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue label lookup should reject malformed gh JSON payloads."""
    monkeypatch.setattr(create_github_issues, "run_gh", lambda *_args, **_kwargs: "{}")

    with pytest.raises(RuntimeError, match="Invalid label payload for issue #42"):
        create_github_issues.get_issue_labels(42)


def test_get_labels_rejects_unknown_story_status() -> None:
    """Story status typos should not silently drop managed status labels."""
    story = {"slug": "e1-example", "status": "inprogress", "labels": ["epic:test"]}

    with pytest.raises(ValueError, match="inprogress.*e1-example"):
        create_github_issues.get_labels(story)


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
        ("llms.txt", generate_llms_txt.SUMMARY_FILES),
        ("llms-full.txt", generate_llms_txt.FULL_FILES),
    ],
)
def test_llms_bundle_matches_configured_sources(output: str, sources: list[str]) -> None:
    """Committed llms bundles should exactly match their configured sources."""
    assert (ROOT / output).read_text(encoding="utf-8") == generate_llms_txt._render(sources)
