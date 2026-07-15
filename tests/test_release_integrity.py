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
import sys
from importlib.metadata import version as get_pkg_version
from pathlib import Path
from typing import Any

import pytest
from packaging.specifiers import SpecifierSet

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).parent.parent
PYTHON_CLASSIFIER_PREFIX = "Programming Language :: Python :: "


def _read_toml(relative_path: Path) -> dict[str, Any]:
    """Read a TOML file relative to the repository root."""
    with (ROOT / relative_path).open("rb") as file:
        return tomllib.load(file)


def _declared_python_versions() -> list[str]:
    """Return supported Python minors from package classifiers."""
    classifiers = _read_toml(Path("pyproject.toml"))["project"]["classifiers"]
    return [
        classifier.removeprefix(PYTHON_CLASSIFIER_PREFIX)
        for classifier in classifiers
        if re.fullmatch(rf"{re.escape(PYTHON_CLASSIFIER_PREFIX)}\d+\.\d+", classifier)
    ]


def _expected_python_constraint() -> SpecifierSet:
    """Derive the exact supported range from the first and last classifiers."""
    versions = _declared_python_versions()
    minimum_major, minimum_minor = map(int, versions[0].split("."))
    maximum_major, maximum_minor = map(int, versions[-1].split("."))
    assert minimum_major == maximum_major, "Python classifiers must remain within one major version"
    return SpecifierSet(f">={minimum_major}.{minimum_minor},<{maximum_major}.{maximum_minor + 1}")


def _workflow_python_matrix(relative_path: Path) -> list[str]:
    """Extract the single inline Python test matrix from a workflow."""
    workflow = (ROOT / relative_path).read_text()
    matches = re.findall(r"^\s+python-version:\s*\[([^]]+)]", workflow, re.MULTILINE)
    assert len(matches) == 1, f"Expected one Python matrix in {relative_path}, found {len(matches)}"
    return re.findall(r"['\"](\d+\.\d+)['\"]", matches[0])


def _expected_badge_url() -> str:
    """Derive the static Shields badge URL from package classifiers."""
    versions = _declared_python_versions()
    return f"https://img.shields.io/badge/Python-{versions[0]}--{versions[-1]}-blue?logo=python"


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


def test_python_classifiers_are_contiguous_and_match_requires_python() -> None:
    """Classifiers are the source of truth for a contiguous supported range."""
    versions = _declared_python_versions()
    assert versions, "pyproject.toml must declare at least one Python minor classifier"

    parsed = [tuple(map(int, version.split("."))) for version in versions]
    major_versions = {major for major, _minor in parsed}
    assert len(major_versions) == 1, "Python classifiers must remain within one major version"
    major = parsed[0][0]
    expected_versions = [f"{major}.{minor}" for minor in range(parsed[0][1], parsed[-1][1] + 1)]
    assert versions == expected_versions, "Python classifiers must be ordered and contiguous"

    requires_python = _read_toml(Path("pyproject.toml"))["project"]["requires-python"]
    assert SpecifierSet(requires_python) == _expected_python_constraint(), (
        "project.requires-python must exactly span the classified Python minors"
    )


def test_uv_lock_matches_declared_python_constraint() -> None:
    """The lockfile must use the same normalized Python constraint as metadata."""
    locked_constraint = _read_toml(Path("uv.lock"))["requires-python"]
    assert SpecifierSet(locked_constraint) == _expected_python_constraint()


def test_declared_python_versions_ignores_non_minor_classifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare-major or 'X :: Only' classifiers must not be mistaken for supported minors."""
    monkeypatch.setattr(
        sys.modules[__name__],
        "_read_toml",
        lambda relative_path: {  # noqa: ARG005 - signature must match _read_toml
            "project": {
                "classifiers": [
                    "Programming Language :: Python :: 3",
                    "Programming Language :: Python :: 3 :: Only",
                    "Programming Language :: Python :: 3.10",
                    "Programming Language :: Python :: 3.11",
                ]
            }
        },
    )

    assert _declared_python_versions() == ["3.10", "3.11"]


def test_expected_python_constraint_excludes_adjacent_minors() -> None:
    """The derived constraint must admit only the declared minor range, not its neighbors."""
    versions = _declared_python_versions()
    minimum_major, minimum_minor = map(int, versions[0].split("."))
    maximum_major, maximum_minor = map(int, versions[-1].split("."))
    constraint = _expected_python_constraint()

    assert f"{minimum_major}.{minimum_minor}.0" in constraint
    assert f"{maximum_major}.{maximum_minor}.99" in constraint
    assert f"{minimum_major}.{minimum_minor - 1}.9" not in constraint
    assert f"{maximum_major}.{maximum_minor + 1}.0" not in constraint


def test_workflow_python_matrix_extracts_quoted_versions_in_order(tmp_path: Path) -> None:
    """A single inline matrix must yield versions in file order, regardless of quote style."""
    workflow = tmp_path / "workflow.yml"
    workflow.write_text(
        'jobs:\n  a:\n    strategy:\n      matrix:\n        python-version: ["3.11", \'3.10\', "3.12"]\n',
        encoding="utf-8",
    )

    assert _workflow_python_matrix(workflow) == ["3.11", "3.10", "3.12"]


def test_workflow_python_matrix_rejects_missing_matrix(tmp_path: Path) -> None:
    """A workflow with no inline python-version matrix must fail fast, not silently pass."""
    workflow = tmp_path / "workflow.yml"
    workflow.write_text("jobs:\n  a:\n    steps: []\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="Expected one Python matrix"):
        _workflow_python_matrix(workflow)


def test_workflow_python_matrix_rejects_multiple_matrices(tmp_path: Path) -> None:
    """A workflow with more than one inline python-version matrix must fail fast."""
    workflow = tmp_path / "workflow.yml"
    workflow.write_text(
        "jobs:\n"
        "  a:\n"
        "    strategy:\n"
        "      matrix:\n"
        "        python-version: ['3.10', '3.11']\n"
        "  b:\n"
        "    strategy:\n"
        "      matrix:\n"
        "        python-version: ['3.12']\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="Expected one Python matrix"):
        _workflow_python_matrix(workflow)


@pytest.mark.parametrize(
    "workflow",
    [
        Path(".github/workflows/unit-tests-workflow.yml"),
        Path(".github/workflows/release.yml"),
    ],
)
def test_workflow_python_matrix_matches_classifiers(workflow: Path) -> None:
    """Unit-test and release matrices must cover exactly the supported minors."""
    assert _workflow_python_matrix(workflow) == _declared_python_versions()


def test_macos_covers_minimum_and_maximum_python() -> None:
    """The unit-test workflow must exercise both support boundaries on macOS."""
    workflow = (ROOT / ".github" / "workflows" / "unit-tests-workflow.yml").read_text()
    macos_versions = re.findall(
        r"- python-version:\s*['\"](\d+\.\d+)['\"]\s+os:\s*macos-latest",
        workflow,
    )
    versions = _declared_python_versions()
    assert macos_versions == [versions[0], versions[-1]]


def test_docker_uses_latest_supported_python() -> None:
    """Every Docker build stage must use the latest classified Python minor."""
    dockerfile = (ROOT / "Dockerfile").read_text()
    base_versions = re.findall(r"^FROM python:(\d+\.\d+)-slim", dockerfile, re.MULTILINE)
    assert base_versions, "Dockerfile must use an official python:<minor>-slim base image"
    assert set(base_versions) == {_declared_python_versions()[-1]}


def test_front_page_python_support_matches_classifiers() -> None:
    """README rows, latest marker, and front-page badges must match metadata."""
    versions = _declared_python_versions()
    badge_url = _expected_badge_url()
    readme = (ROOT / "README.md").read_text()
    docs_index = (ROOT / "docs" / "index.rst").read_text()
    release_process = (ROOT / "docs" / "release-process.md").read_text()

    support_rows = re.findall(r"^\| (\d+\.\d+) \| Supported \|([^|]*)\|$", readme, re.MULTILINE)
    assert [version for version, _notes in support_rows] == versions
    assert support_rows[0][1].strip() == "Minimum supported version"
    assert support_rows[-1][1].strip() == "Latest supported version"

    badge_label = f"Python | {versions[0]}-{versions[-1]}"
    assert f"[![{badge_label}]({badge_url})](#supported-python-versions)" in readme
    assert f".. |Python| image:: {badge_url}" in docs_index
    assert badge_url in release_process


def test_release_process_documents_pypi_metadata_verification() -> None:
    """Post-release checklist must direct maintainers to verify PyPI Requires-Python
    and classifiers, and to cross-check the badge against them.
    """
    release_process = (ROOT / "docs" / "release-process.md").read_text()
    assert "Requires-Python" in release_process, (
        "docs/release-process.md must instruct maintainers to verify the live PyPI Requires-Python metadata"
    )
    assert "every supported Python minor appears in the classifiers" in release_process
    assert _expected_badge_url() in release_process


def test_llms_bundles_reference_main_branch_not_master() -> None:
    """Generated llms bundles must not reference the renamed default branch."""
    for relative_path in (Path("llms.txt"), Path("llms-full.txt")):
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "master" not in content, f"{relative_path} still references the old 'master' branch name"


def test_llms_bundles_contain_python_support_badge() -> None:
    """Generated llms bundles must mirror the front-page Python support badge."""
    badge_url = _expected_badge_url()
    for relative_path in (Path("llms.txt"), Path("llms-full.txt")):
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        assert badge_url in content, f"{relative_path} is missing the Python support badge URL"


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
