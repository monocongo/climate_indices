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
