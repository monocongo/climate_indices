"""Release integrity guardrails.

Catches packaging and release-sequencing regressions before they reach PyPI.

Two tiers of tests:

  Always-on (run in all CI environments):
    - Release workflow uses OIDC, not a stored token
    - Release workflow has a manual approval environment gate
    - Core public API symbols are importable after install

  Release-time only (run before pushing a release tag):
    uv run pytest -m release tests/test_release_integrity.py
    - pyproject.toml version is valid semver
    - CHANGELOG.md has no [Unreleased] block
    - CHANGELOG.md top entry matches pyproject.toml version
    - Installed package version matches pyproject.toml
"""

from __future__ import annotations

import re
from importlib.metadata import version as get_pkg_version
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def _read_pyproject_version() -> str:
    """Extract version from pyproject.toml without requiring tomllib."""
    content = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not find 'version = ...' in pyproject.toml")
    return match.group(1)


# ---------------------------------------------------------------------------
# Always-on guardrails
# ---------------------------------------------------------------------------


def test_release_workflow_uses_oidc_not_token() -> None:
    """release.yml must use OIDC trusted publishing, not a stored PyPI API token.

    Addresses: F2/F14 — prevents a token accidentally replacing the OIDC publisher
    and surfacing only at the manual-approval stage of a live release.
    """
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "PYPI_API_TOKEN" not in workflow, (
        "release.yml references PYPI_API_TOKEN — OIDC trusted publishing must be used instead; "
        "remove the token reference and verify the trusted publisher is registered on PyPI"
    )
    assert "id-token: write" in workflow, (
        "release.yml is missing 'id-token: write' permission — required for OIDC publishing"
    )


def test_release_workflow_has_environment_gate() -> None:
    """release.yml must gate the publish job behind 'environment: release'.

    Addresses: F14 — ensures the manual approval gate cannot be silently removed.
    """
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "environment: release" in workflow, (
        "release.yml missing 'environment: release' — the manual approval gate before PyPI publish has been removed"
    )


def test_release_workflow_requires_exact_semver_tags() -> None:
    """release.yml must trigger and publish only for exact vX.Y.Z tags."""
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "v*.*.*" in workflow, "release.yml should only trigger on v*.*.* release tag candidates"
    assert r"^refs/tags/v[0-9]+\.[0-9]+\.[0-9]+$" in workflow, "release.yml missing explicit exact SemVer tag guard"


def test_release_workflow_creates_github_release() -> None:
    """release.yml must create the GitHub Release after PyPI publish."""
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "gh release create" in workflow, "release.yml must create a GitHub Release for the published tag"


def test_core_public_api_importable() -> None:
    """Core public API symbols must be importable from the installed package.

    Addresses: F9 (Murat) — catches a broken __init__.py or missing transitive
    dependency that lets the package install but fail on first use.
    """
    from climate_indices import pet_hargreaves, pet_thornthwaite, spei, spi

    assert callable(spi)
    assert callable(spei)
    assert callable(pet_thornthwaite)
    assert callable(pet_hargreaves)


def test_v240_public_api_importable() -> None:
    """v2.4.0-specific public API additions must be importable.

    eddi() was exposed in the public API in v2.4.0. If it is missing, the v2.4.0
    __init__.py export was not applied correctly.
    """
    from climate_indices import eddi

    assert callable(eddi)


# ---------------------------------------------------------------------------
# Release-time guardrails
# Run before pushing a release tag:
#   uv run pytest -m release tests/test_release_integrity.py
# ---------------------------------------------------------------------------


@pytest.mark.release
def test_pyproject_version_is_semver() -> None:
    """pyproject.toml version must be a valid semver string (X.Y.Z).

    Addresses: F1 — catches placeholder values like 'unreleased' or '0.0.0-dev'.
    """
    version = _read_pyproject_version()
    assert re.match(r"^\d+\.\d+\.\d+$", version), (
        f"pyproject.toml version '{version}' is not valid semver — expected X.Y.Z"
    )


@pytest.mark.release
def test_changelog_has_no_unreleased_block() -> None:
    """CHANGELOG.md must not contain an [Unreleased] block at release time.

    Addresses: F8 (AC-0 guardrail) — the single most common release mistake in this
    repo; catches the exact scenario that triggered the v2.3.0 sequencing issue.
    """
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert "## [Unreleased]" not in changelog, (
        "CHANGELOG.md contains '## [Unreleased]' — change the header to '## [X.Y.Z] - YYYY-MM-DD' before releasing"
    )


@pytest.mark.release
def test_changelog_top_entry_matches_pyproject_version() -> None:
    """CHANGELOG.md top release block must match pyproject.toml version.

    Addresses: F8 — catches version bump in one file without the other.
    """
    pyproject_version = _read_pyproject_version()
    changelog = (ROOT / "CHANGELOG.md").read_text()
    match = re.search(r"^## \[(\d+\.\d+\.\d+)\]", changelog, re.MULTILINE)
    assert match is not None, "No versioned release block (## [X.Y.Z]) found in CHANGELOG.md"
    changelog_version = match.group(1)
    assert changelog_version == pyproject_version, (
        f"CHANGELOG.md top entry [{changelog_version}] does not match "
        f"pyproject.toml version [{pyproject_version}] — update one to match the other"
    )


@pytest.mark.release
def test_installed_package_version_matches_pyproject() -> None:
    """Installed climate-indices package version must match pyproject.toml.

    Addresses: F5 — catches the case where pyproject.toml was edited but the package
    was not reinstalled, or the installed package is from a stale editable install.
    Run after `uv pip install -e .` to ensure the check reflects the current state.
    """
    pyproject_version = _read_pyproject_version()
    installed_version = get_pkg_version("climate-indices")
    assert installed_version == pyproject_version, (
        f"Installed climate-indices=={installed_version} does not match "
        f"pyproject.toml version {pyproject_version} — run 'uv pip install -e .' to sync"
    )
