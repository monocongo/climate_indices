"""Release integrity guardrails.

Catches packaging and release-sequencing regressions before they reach PyPI.

Two tiers of tests:

  Always-on (run in all CI environments):
    - Release workflow uses OIDC, not a stored token
    - Release workflow has a manual approval environment gate
    - Release tags must point to commits on main
    - Built wheels are smoke-tested outside the source checkout
    - Core public API symbols are importable after install

  Release-time only (run before pushing a release tag):
    uv run pytest -m release tests/test_release_integrity.py
    - pyproject.toml version is valid semver
    - CHANGELOG.md has no [Unreleased] block
    - CHANGELOG.md top entry matches pyproject.toml version
    - Installed package version matches pyproject.toml
"""

from __future__ import annotations

import fnmatch
import re
import shlex
import subprocess
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
    workflow = (ROOT / relative_path).read_text(encoding="utf-8")
    matches = re.findall(r"^\s+python-version:\s*\[([^]]+)]", workflow, re.MULTILINE)
    assert len(matches) == 1, f"Expected one Python matrix in {relative_path}, found {len(matches)}"
    return re.findall(r"['\"](\d+\.\d+)['\"]", matches[0])


UNIT_TESTS_WORKFLOW = Path(".github/workflows/unit-tests-workflow.yml")


def _workflow_job(workflow: str, job: str) -> str:
    """Return one top-level job's text, without the comment block that precedes the next job."""
    match = re.search(rf"^  {re.escape(job)}:\n(.*?)(?=^  [\w-]+:\n|\Z)", workflow, re.MULTILINE | re.DOTALL)
    assert match, f"Expected a top-level job named {job!r}"
    return re.split(r"^  #", match.group(1), maxsplit=1, flags=re.MULTILINE)[0]


def _job_legs(job: str) -> set[tuple[str, str]]:
    """Expand a job's inline python-version x os matrix, plus `include` entries, to (python, os) legs."""
    pythons = re.findall(r"^\s+python-version:\s*\[([^]]+)]", job, re.MULTILINE)
    systems = re.findall(r"^\s+os:\s*\[([^]]+)]", job, re.MULTILINE)
    assert len(pythons) == 1, "Expected one inline python-version list"
    assert len(systems) == 1, "Expected one inline os list"
    legs = {
        (python, system)
        for python in re.findall(r"\d+\.\d+", pythons[0])
        for system in re.findall(r"[\w.-]+", systems[0])
    }
    return legs | set(re.findall(r"- python-version:\s*['\"](\d+\.\d+)['\"]\s+os:\s*([\w.-]+)", job))


def _expected_badge_url() -> str:
    """Derive the static Shields badge URL from package classifiers."""
    versions = _declared_python_versions()
    return f"https://img.shields.io/badge/Python-{versions[0]}--{versions[-1]}-blue?logo=python"


def _read_pyproject_version() -> str:
    """Extract version from pyproject.toml without requiring tomllib."""
    content = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
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


def test_job_helpers_expand_legs_and_stop_at_the_next_jobs_comment() -> None:
    """A job's text must end before the comment block that introduces the next job."""
    workflow = (
        "jobs:\n"
        "  a:\n"
        "    strategy:\n"
        "      matrix:\n"
        "        python-version: ['3.10', '3.11']\n"
        "        os: [ubuntu-latest]\n"
        "        include:\n"
        "          - python-version: '3.11'\n"
        "            os: macos-latest\n"
        "\n"
        "  # about b\n"
        "  b-c:\n"
        "    steps: []\n"
    )

    job = _workflow_job(workflow, "a")

    assert "about b" not in job
    assert _job_legs(job) == {("3.10", "ubuntu-latest"), ("3.11", "ubuntu-latest"), ("3.11", "macos-latest")}
    assert _workflow_job(workflow, "b-c").strip() == "steps: []"


def test_release_workflow_python_matrix_matches_classifiers() -> None:
    """The release matrix must cover exactly the supported minors."""
    assert _workflow_python_matrix(Path(".github/workflows/release.yml")) == _declared_python_versions()


def _unit_test_legs() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """The (python, os) legs of the pull-request `test` job and the `test-full` remainder."""
    workflow = (ROOT / UNIT_TESTS_WORKFLOW).read_text(encoding="utf-8")
    return _job_legs(_workflow_job(workflow, "test")), _job_legs(_workflow_job(workflow, "test-full"))


def test_unit_test_jobs_cover_every_supported_python_without_overlap() -> None:
    """`test` and `test-full` together cover exactly the supported minors, and no leg runs in both."""
    pull_request_legs, full_legs = _unit_test_legs()

    assert {python for python, _ in pull_request_legs | full_legs} == set(_declared_python_versions())
    assert {(python, "ubuntu-latest") for python in _declared_python_versions()} <= pull_request_legs | full_legs
    assert {(python, "ubuntu-latest") for python in _declared_python_versions()} <= pull_request_legs
    assert not pull_request_legs & full_legs, "a leg in both jobs runs twice on pushes to main"


def test_pull_request_legs_cover_both_boundaries_on_linux_and_the_newest_on_macos() -> None:
    """The legs every pull request runs must still hit both support boundaries."""
    pull_request_legs, _ = _unit_test_legs()
    versions = _declared_python_versions()

    assert (versions[0], "ubuntu-latest") in pull_request_legs
    assert (versions[-1], "ubuntu-latest") in pull_request_legs
    assert (versions[-1], "macos-latest") in pull_request_legs


def test_macos_covers_minimum_and_maximum_python() -> None:
    """Across both unit-test jobs, macOS must exercise both support boundaries."""
    pull_request_legs, full_legs = _unit_test_legs()
    versions = _declared_python_versions()

    macos_versions = {python for python, system in pull_request_legs | full_legs if system == "macos-latest"}
    assert macos_versions == {versions[0], versions[-1]}


def test_full_matrix_runs_beyond_pull_requests_and_the_pull_request_job_is_ungated() -> None:
    """`test-full` must run on main pushes and the schedule, `test` on every event.

    Gating `test-full` with `!= 'pull_request'` also runs it in a merge queue, which is
    where a merge is checked before it lands. Losing the schedule or the main trigger,
    or gating `test`, would silently shrink what a leg means.
    """
    workflow = (ROOT / UNIT_TESTS_WORKFLOW).read_text(encoding="utf-8")
    pull_request_job = _workflow_job(workflow, "test")
    full_job = _workflow_job(workflow, "test-full")

    assert re.search(r"^    if: github\.event_name != 'pull_request'$", full_job, re.MULTILINE)
    assert not re.search(r"^    if:", pull_request_job, re.MULTILINE)
    assert re.search(r"^  push:\n    branches: \[main]$", workflow, re.MULTILINE)
    assert re.search(r"^  merge_group:", workflow, re.MULTILINE)
    assert re.search(r"^  workflow_dispatch:", workflow, re.MULTILINE)
    assert re.search(r"^  schedule:\n    - cron: ", workflow, re.MULTILINE)


def test_test_full_job_runs_the_same_core_command_as_the_pull_request_job() -> None:
    """A leg must mean the same thing in `test` and `test-full`: locked dev install, same core command.

    Source builds stay allowed in both jobs: the 3.10 legs need the sdist-only asciitree
    from the zarr 2.x stack, so neither job can pass `--no-build`.
    """
    workflow = (ROOT / UNIT_TESTS_WORKFLOW).read_text(encoding="utf-8")
    pull_request_job = _workflow_job(workflow, "test")
    full_job = _workflow_job(workflow, "test-full")

    assert "- name: Install dependencies\n        run: uv sync --locked --dev" in pull_request_job
    assert "- name: Install dependencies\n        run: uv sync --locked --dev" in full_job
    assert (
        pull_request_job.split("- name: Run core tests")[1].strip()
        == full_job.split("- name: Run core tests")[1].strip()
    )


def test_docker_uses_latest_supported_python() -> None:
    """Every Docker build stage must use the latest classified Python minor."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    base_versions = re.findall(r"^FROM python:(\d+\.\d+)-slim", dockerfile, re.MULTILINE)
    assert base_versions, "Dockerfile must use an official python:<minor>-slim base image"
    assert set(base_versions) == {_declared_python_versions()[-1]}


def test_front_page_python_support_matches_classifiers() -> None:
    """README rows, latest marker, and front-page badges must match metadata."""
    versions = _declared_python_versions()
    badge_url = _expected_badge_url()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    release_process = (ROOT / "docs" / "release-process.md").read_text(encoding="utf-8")

    support_rows = re.findall(r"^\| (\d+\.\d+) \| Supported \|([^|]*)\|$", readme, re.MULTILINE)
    assert [version for version, _notes in support_rows] == versions
    assert support_rows[0][1].strip() == "Minimum supported version"
    assert support_rows[-1][1].strip() == "Latest supported version"

    badge_label = f"Python | {versions[0]}-{versions[-1]}"
    assert f"[![{badge_label}]({badge_url})](#supported-python-versions)" in readme
    assert f"![Python | {versions[0]}-{versions[-1]}]({badge_url})" in docs_index
    assert badge_url in release_process


def test_wildfire_applications_cross_links_use_myst_roles() -> None:
    """Cross-page links must be MyST `{doc}` roles, not literal RST `:doc:` text."""
    page = (ROOT / "docs" / "wildfire_applications.md").read_text(encoding="utf-8")
    assert "{doc}`xarray_migration`" in page
    assert "{doc}`index`" in page
    assert ":doc:" not in page, (
        "docs/wildfire_applications.md must use MyST {doc} roles; an RST :doc: role renders as literal text"
    )


def test_every_published_docs_page_has_one_visible_section() -> None:
    """Published pages must be reachable from the four visible section toctrees.

    Sphinx's warnings-as-errors build only catches a page that is in no toctree at
    all; a page parked in a hidden toctree, or one misassigned to a section, still
    builds green. This keeps the four-section navigation and the exclusion list in
    `docs/conf.py` honest without parsing built HTML.
    """
    docs = ROOT / "docs"
    conf = (docs / "conf.py").read_text(encoding="utf-8")
    exclude_block = re.search(r"^exclude_patterns = \[(.*?)^\]", conf, re.MULTILINE | re.DOTALL)
    assert exclude_block is not None, "docs/conf.py must define exclude_patterns"
    excluded = re.findall(r'"([^"]+)"', exclude_block.group(1))

    def is_excluded(relative: str) -> bool:
        return any(
            fnmatch.fnmatch(relative, pattern) or any(fnmatch.fnmatch(part, pattern) for part in Path(relative).parts)
            for pattern in excluded
        )

    sources = {
        path.relative_to(docs).with_suffix("").as_posix()
        for path in docs.rglob("*.md")
        if not is_excluded(path.relative_to(docs).as_posix())
    }

    toctree = re.compile(r"^```\{toctree\}(.*?)^```", re.MULTILINE | re.DOTALL)
    visible: set[str] = set()
    hidden: set[str] = set()
    for page in sorted(sources):
        text = (docs / f"{page}.md").read_text(encoding="utf-8")
        for block in toctree.findall(text):
            entries = {
                (Path(page).parent / line.strip()).as_posix()
                for line in block.splitlines()
                if line.strip() and not line.strip().startswith(":")
            }
            (hidden if ":hidden:" in block else visible).update(entries)

    orphans = {
        page
        for page in sources
        if re.search(r"^orphan:\s*true\s*$", (docs / f"{page}.md").read_text(encoding="utf-8"), re.MULTILINE)
    }

    required = sources - orphans - hidden - {"index"}
    assert required == visible, "every published page must appear in exactly one visible section toctree"
    assert not visible & hidden, "a page cannot be both a visible section member and hidden"
    assert not hidden, "no published page may sit in a hidden toctree; add a staged page to its section toctree instead"
    assert orphans == {"pypi_release"}, "pypi_release is the only documented orphan"

    homepage = toctree.search((docs / "index.md").read_text(encoding="utf-8"))
    assert homepage is not None, "docs/index.md must route into the four sections"
    assert {
        line.strip() for line in homepage.group(1).splitlines() if line.strip() and not line.strip().startswith(":")
    } == {"tutorials", "how-to", "reference", "explanation"}


def test_release_process_documents_pypi_metadata_verification() -> None:
    """Post-release checklist must direct maintainers to verify PyPI Requires-Python
    and classifiers, and to cross-check the badge against them.
    """
    release_process = (ROOT / "docs" / "release-process.md").read_text(encoding="utf-8")
    assert "Requires-Python" in release_process, (
        "docs/release-process.md must instruct maintainers to verify the live PyPI Requires-Python metadata"
    )
    assert "every supported Python minor appears in the classifiers" in release_process
    assert _expected_badge_url() in release_process


def test_release_process_documents_the_rehearsal_boundary() -> None:
    """The rehearsal runbook must state the failure it expects and what it cannot prove.

    `create-release` is skipped when `publish` fails, so a green rehearsal must not
    read as a proven release path.
    """
    release_process = (ROOT / "docs" / "release-process.md").read_text(encoding="utf-8")
    section = re.search(
        r"^## Release rehearsal.*?$(.*?)(?=^## |\Z)",
        release_process,
        re.MULTILINE | re.DOTALL,
    )
    assert section is not None, "docs/release-process.md must keep the '## Release rehearsal' section"
    boundary = " ".join(section.group(1).split())
    assert "`publish` fails; `create-release` is skipped" in boundary
    assert "not a trusted publisher" in boundary
    assert "only the real repository can validate these" in boundary
    assert "Rehearsals run in a **rehearsal repository**" in boundary, (
        "the runbook must define the rehearsal target once so the release ticket and this page agree"
    )
    assert "A fork under a different owner is equivalent" in boundary


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
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "PYPI_API_TOKEN" not in workflow, (
        "release.yml references PYPI_API_TOKEN — OIDC trusted publishing must be used instead; "
        "remove the token reference and verify the trusted publisher is registered on PyPI"
    )
    assert workflow.count("id-token: write") == 1, (
        "release.yml must grant 'id-token: write' only to its publish-only job"
    )
    publish_job = workflow.split("\n  publish:", maxsplit=1)[1].split("\n  create-release:", maxsplit=1)[0]
    assert "id-token: write" in publish_job
    assert "pypa/gh-action-pypi-publish" in publish_job
    assert "actions/checkout" not in publish_job
    assert "python -m build" not in publish_job


def test_release_workflow_has_environment_gate() -> None:
    """release.yml must gate the publish job behind 'environment: release'.

    Addresses: F14 — ensures the manual approval gate cannot be silently removed.
    """
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "environment: release" in workflow, (
        "release.yml missing 'environment: release' — the manual approval gate before PyPI publish has been removed"
    )


def test_release_workflow_requires_exact_semver_tags() -> None:
    """release.yml must trigger and publish only for exact vX.Y.Z tags."""
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "v*.*.*" in workflow, "release.yml should only trigger on v*.*.* release tag candidates"
    assert r"^refs/tags/v[0-9]+\.[0-9]+\.[0-9]+$" in workflow, "release.yml missing explicit exact SemVer tag guard"


def test_release_workflow_requires_tag_commit_on_main() -> None:
    """A release tag must point to a commit reachable from origin/main."""
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "fetch-depth: 0" in workflow
    assert 'git merge-base --is-ancestor "${GITHUB_SHA}" "origin/main"' in workflow


def test_release_workflow_creates_github_release() -> None:
    """release.yml must create the GitHub Release after PyPI publish."""
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    create_release_job = workflow.split("\n  create-release:", maxsplit=1)[1]
    create_release_job = re.split(r"^  (?=\S)", create_release_job, maxsplit=1, flags=re.MULTILINE)[0]
    release_command = re.search(r"^\s+run: (gh release create .+)$", create_release_job, re.MULTILINE)

    assert create_release_job.startswith("\n    needs: publish\n")
    assert release_command is not None, "release.yml must create a GitHub Release for the published tag"
    lexer = shlex.shlex(release_command.group(1), posix=True, punctuation_chars=";&")
    lexer.whitespace_split = True
    first_command = []
    for token in lexer:
        if token in {";", "&", "&&"}:
            break
        first_command.append(token)

    assert first_command[:3] == ["gh", "release", "create"]
    assert "--repo" in first_command
    repo_index = first_command.index("--repo")
    assert first_command[repo_index + 1 : repo_index + 2] == ["${GITHUB_REPOSITORY}"]


def test_release_workflow_smoke_tests_built_wheel() -> None:
    """The built wheel must install and expose the public API outside the checkout."""
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    wheel_check = workflow.split("\n  wheel-check:", maxsplit=1)[1].split("\n  publish:", maxsplit=1)[0]

    assert "Test wheel installation" in wheel_check
    assert 'python -m venv "${RUNNER_TEMP}/wheel-check"' in wheel_check
    assert 'cd "${RUNNER_TEMP}"' in wheel_check
    assert (
        "from climate_indices import eddi, fire, pci, percentage_of_normal, pet_hargreaves, pet_thornthwaite, spei, spi"
        in wheel_check
    )
    assert (
        "assert all(map(callable, (fire.kbdi, fire.cffwis, fire.fosberg_ffwi, fire.hot_dry_windy, fire.haines_index)))"
        in wheel_check
    )


def test_release_workflow_installs_wheel_on_boundary_pythons() -> None:
    """The wheel must install and expose console scripts on the oldest and newest supported
    Pythons, and that check must gate publishing.
    """
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    wheel_check = workflow.split("\n  wheel-check:", maxsplit=1)[1].split("\n  publish:", maxsplit=1)[0]
    versions = _declared_python_versions()

    assert wheel_check.count("- python-version:") == 2, "wheel checks must stay at the boundary versions"
    assert f"- python-version: '{versions[0]}'" in wheel_check
    assert f"- python-version: '{versions[-1]}'" in wheel_check
    declared_scripts = " ".join(_read_toml(Path("pyproject.toml"))["project"]["scripts"])
    assert f"for entry_point in {declared_scripts}; do" in wheel_check
    assert "needs: [build, wheel-check]" in workflow, "publishing must wait for the wheel installation check"


def test_minimum_dependency_job_preserves_resolved_environment() -> None:
    """Minimum-dependency tests must not resynchronize to the normal lock."""
    workflow = (ROOT / ".github" / "workflows" / "unit-tests-workflow.yml").read_text(encoding="utf-8")
    minimum_job = workflow.split("\n  test-minimum-deps:", maxsplit=1)[1].split("\n  notebooks:", maxsplit=1)[0]

    assert "uv sync --no-dev --group test --resolution lowest-direct" in minimum_job
    assert "--locked" not in minimum_job
    # Core tests only: validation and repo/meta checks are single-owner jobs.
    assert minimum_job.count("uv run --no-sync --no-build pytest") == 1


@pytest.mark.parametrize(
    "workflow_path",
    [
        Path(".github/workflows/unit-tests-workflow.yml"),
        Path(".github/workflows/benchmarks.yml"),
        Path(".github/workflows/release.yml"),
    ],
)
def test_ci_commands_use_fresh_lock_and_prepared_environment(workflow_path: Path) -> None:
    """Ordinary CI commands must check lock freshness and avoid implicit resyncs."""
    workflow = (ROOT / workflow_path).read_text(encoding="utf-8")
    commands = [line.strip() for line in workflow.splitlines()]
    ordinary_syncs = [
        command
        for command in commands
        if command.startswith(("run: uv sync", "uv sync")) and "--resolution lowest-direct" not in command
    ]
    exports = [command for command in commands if command.startswith(("run: uv export", "uv export"))]
    runs = [command for command in commands if "uv run " in command]

    assert ordinary_syncs and all("--locked" in command for command in ordinary_syncs)
    assert all("--locked" in command for command in exports)
    assert "--frozen" not in workflow
    assert runs
    assert all("uv run --no-sync --no-build " in command for command in runs)


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


def test_v300_public_api_importable() -> None:
    """The 3.0.0 fire namespace must be a public package export with a stable surface.

    The fire subsystem is the headline 3.0.0 addition. If the public __init__ stops
    exporting it, the package still installs but the namespace is absent from the
    advertised API; the export list is pinned so the 3.0.0 public surface cannot
    shrink unnoticed.
    """
    import climate_indices
    from climate_indices import fire

    assert "fire" in climate_indices.__all__
    # A fresh interpreter proves the eager __init__ export: in this process
    # `from climate_indices import fire` would import the submodule as a fallback
    # and mask a removed top-level import.
    subprocess.run([sys.executable, "-c", "import climate_indices; assert climate_indices.fire"], check=True)
    assert set(fire.__all__) == {
        "CFFWISResult",
        "CFFWISState",
        "DCResult",
        "DCState",
        "DMCResult",
        "DMCState",
        "FFMCResult",
        "FFMCState",
        "KBDIResult",
        "KBDIState",
        "buildup_index",
        "cffwis",
        "cffwis_fwi",
        "daily_severity_rating",
        "drought_code",
        "duff_moisture_code",
        "ffmc",
        "fosberg_ffwi",
        "haines_index",
        "haines_index_from_profile",
        "hot_dry_windy",
        "initial_spread_index",
        "kbdi",
        "overwinter_drought_code",
    }
    for name in fire.__all__:
        assert callable(getattr(fire, name)), f"fire.{name} is not callable"


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
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]" not in changelog, (
        "CHANGELOG.md contains '## [Unreleased]' — change the header to '## [X.Y.Z] - YYYY-MM-DD' before releasing"
    )


@pytest.mark.release
def test_changelog_top_entry_matches_pyproject_version() -> None:
    """CHANGELOG.md top release block must match pyproject.toml version.

    Addresses: F8 — catches version bump in one file without the other.
    """
    pyproject_version = _read_pyproject_version()
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
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
