"""Release integrity guardrails.

Catches packaging and release-sequencing regressions before they reach PyPI or
crates.io.

Two tiers of tests:

  Always-on (run in all CI environments):
    - Release workflow uses OIDC, not a stored token
    - Release workflow has a manual approval environment gate
    - Release tags must point to commits on main
    - Built wheels are smoke-tested outside the source checkout
    - Core public API symbols are importable after install
    - The crate publishes only after its gates, from its own tag namespace
    - The reviewed required-status-check names match the workflow's jobs and matrices

  Release-time only (run before pushing a release tag):
    uv run pytest -m release tests/test_release_integrity.py
    - pyproject.toml version is valid semver
    - CHANGELOG.md has no [Unreleased] block
    - CHANGELOG.md top entry matches pyproject.toml version
    - Installed package version matches pyproject.toml
"""

from __future__ import annotations

import contextlib
import fnmatch
import importlib.util
import os
import re
import shlex
import subprocess
import sys
import tomllib
from datetime import date
from importlib.metadata import version as get_pkg_version
from pathlib import Path
from typing import Any

import pytest
from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).parent.parent
PYTHON_CLASSIFIER_PREFIX = "Programming Language :: Python :: "

_sync_spec = importlib.util.spec_from_file_location("sync_required_checks", ROOT / "scripts/sync_required_checks.py")
assert _sync_spec is not None
assert _sync_spec.loader is not None
sync_required_checks = importlib.util.module_from_spec(_sync_spec)
_sync_spec.loader.exec_module(sync_required_checks)


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
REQUIRED_CHECKS_FILE = Path(".github/required-status-checks.txt")
CRATE_RELEASE_WORKFLOW = Path(".github/workflows/crate-release.yml")
CRATE_CHANGELOG = Path("crates/climate-indices/CHANGELOG.md")
CRATE_MANIFEST = Path("crates/climate-indices/Cargo.toml")

# The 3.1.0 Stable DataArray surface and public `xarray_adapter` decorator
# (docs/xarray_compatibility.md, ADR-0012). The Beta routes stay out of these sets,
# so moving a name between the tiers needs an ADR change and an edit here.
STABLE_DATAARRAY_API_NAMES = frozenset(
    {
        "eddi",
        "edi",
        "fit_diagnostics",
        "pci",
        "pdsi",
        "percentage_of_normal",
        "pet_hargreaves",
        "pet_penman_monteith",
        "pet_thornthwaite",
        "spei",
        "spi",
        "standardized_index",
        "xarray_adapter",
    }
)
STABLE_DATAARRAY_MODULE_API_NAMES: dict[str, frozenset[str]] = {
    "energy": frozenset(
        {
            "cooling_degree_days",
            "energy_degree_days",
            "heating_degree_days",
            "hub_height_wind_speed",
            "srepi",
            "srli",
            "standardized_demand_index",
            "wind_shear_exponent",
        }
    ),
    "fire": frozenset({"cffwis", "haines_index", "hot_dry_windy", "kbdi"}),
    "flood": frozenset({"antecedent_precipitation_index", "edi", "effective_precipitation", "flood_index"}),
}
BETA_DATAARRAY_API_NAMES: dict[str, frozenset[str]] = {
    "compound": frozenset(
        {
            "bootstrap_dependence",
            "compound_events_xarray",
            "conditional_probability",
            "harmonize",
            "joint_statistics",
        }
    ),
    "energy": frozenset({"energy_drought_events"}),
}
# Beta routes exported at the package root; SPAI ships Beta in 3.2.0 (#1480)
BETA_ROOT_API_NAMES = frozenset({"spai"})


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


def _declared_required_checks() -> list[str]:
    """The reviewed required status-check contexts, in file order, with comments dropped."""
    text = (ROOT / REQUIRED_CHECKS_FILE).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def test_required_check_names_match_the_workflow_contexts() -> None:
    """A matrix change renames contexts, so the reviewed list must name what the jobs report.

    GitHub reports a stale required context as "Expected - Waiting for status to be
    reported", which blocks every pull request until branch protection is updated.
    """
    declared = _declared_required_checks()

    assert len(declared) == len(set(declared)), f"{REQUIRED_CHECKS_FILE} repeats a context"
    assert sorted(declared) == sorted(sync_required_checks.workflow_contexts()), (
        f"{REQUIRED_CHECKS_FILE} and {UNIT_TESTS_WORKFLOW} disagree; "
        "run scripts/sync_required_checks.py to see the difference"
    )


def test_required_checks_exclude_jobs_that_skip_on_pull_requests() -> None:
    """A job gated off pull requests never reports there, so requiring it blocks every PR."""
    assert "test-full" not in sync_required_checks.REQUIRED_JOBS
    assert sync_required_checks.workflow_contexts()


def test_required_checks_still_require_the_rust_and_native_jobs() -> None:
    """#1292's purpose: a broken Rust build, lint, or wheel must not merge."""
    required_jobs = {check.split(" (", 1)[0] for check in _declared_required_checks()}

    assert {"lint", "test", "test-minimum-deps", "notebooks", "security-audit"} <= required_jobs
    assert {"rust", "test-native", "native-wheel"} <= required_jobs
    assert {"rust-standalone", "rust-crate", "rust-msrv", "rust-semver", "rust-deny"} <= required_jobs


def test_required_check_drift_runs_in_a_required_job() -> None:
    """The drift test must block merges even when the optional meta job fails."""
    workflow = (ROOT / UNIT_TESTS_WORKFLOW).read_text(encoding="utf-8")
    assert "lint" in sync_required_checks.REQUIRED_JOBS
    assert re.search(
        r"^[ \t]+run:[ \t]+uv run --no-sync --no-build pytest tests/test_release_integrity\.py -k required_check[ \t]*$",
        _workflow_job(workflow, "lint"),
        re.MULTILINE,
    )


def test_required_check_drift_rejects_a_commented_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A commented command does not run the required-check tests."""
    workflow = tmp_path / UNIT_TESTS_WORKFLOW
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "jobs:\n"
        "  lint:\n"
        "    steps:\n"
        "      - name: Check required status contexts\n"
        "        # run: uv run --no-sync --no-build pytest tests/test_release_integrity.py -k required_check\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)

    with pytest.raises(AssertionError):
        test_required_check_drift_runs_in_a_required_job()


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


def _tag_date(tag: str, repo: Path | None = None) -> date | None:
    """Return an annotated tag's own tagger date, or None when the tag carries none.

    The date is recorded in the tag object, so it stays fixed across workflow
    reruns. Lightweight tags (what the release rehearsal pushes) carry no tagger
    date, and a missing tag ref is an error rather than a silently skipped check.
    """
    result = subprocess.run(
        ["git", "for-each-ref", f"refs/tags/{tag}", "--format=%(objecttype) %(taggerdate:short)"],
        cwd=repo or ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    object_type, _, tagger_date = result.stdout.strip().partition(" ")
    assert object_type, f"tag ref refs/tags/{tag} is missing from the checkout; fetch tags before this check"
    if object_type != "tag" or not tagger_date:
        return None
    return date.fromisoformat(tagger_date)


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


def test_python_support_starts_at_311() -> None:
    """Metadata and analysis tools must reject the retired Python 3.10 baseline."""
    pyproject = _read_toml(Path("pyproject.toml"))
    constraint = SpecifierSet(pyproject["project"]["requires-python"])

    assert "3.10.99" not in constraint
    assert "3.11.0" in constraint
    assert _declared_python_versions()[0] == "3.11"
    assert pyproject["tool"]["ruff"]["target-version"] == "py311"
    assert pyproject["tool"]["mypy"]["python_version"] == "3.11"


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

    Source builds stay allowed in both jobs because newer interpreters can lack dependency wheels.
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


def _native_jobs() -> dict[str, str]:
    """The Rust-backend jobs of the unit-test workflow, keyed by job name."""
    workflow = (ROOT / UNIT_TESTS_WORKFLOW).read_text(encoding="utf-8")
    return {name: _workflow_job(workflow, name) for name in ("rust", "test-native", "native-wheel")}


def test_rust_toolchain_is_pinned_with_the_cargo_gate_components() -> None:
    """A floating compiler lets a new stable clippy lint fail `-D warnings` on an unrelated change."""
    toolchain = _read_toml(Path("rust-toolchain.toml"))["toolchain"]

    assert re.fullmatch(r"\d+\.\d+\.\d+", toolchain["channel"]), "pin an exact release, not 'stable'"
    assert {"rustfmt", "clippy"} <= set(toolchain["components"])


def test_rust_job_runs_the_cargo_gate_against_the_workspace_interpreter() -> None:
    """Format, lint (warnings are errors), and test the whole workspace; `climate-py` links libpython."""
    rust_job = _native_jobs()["rust"]

    assert "run: rustup toolchain install --no-self-update" in rust_job
    assert "run: cargo fmt --all -- --check" in rust_job
    assert "run: cargo clippy --workspace --all-targets -- -D warnings" in rust_job
    assert "run: cargo test --workspace" in rust_job
    assert "PYO3_PYTHON: ${{ github.workspace }}/.venv/bin/python" in rust_job


def test_standalone_rust_job_checks_the_crate_without_python() -> None:
    """A Rust-only container must prove that Cargo needs no Python build tools.

    Image and Cargo fetches run against external registries, so the job also needs a
    bounded failure window, and interpreter aliases (for example `python3.12`)
    must fail the absence check rather than slip past it.
    """
    workflow = (ROOT / UNIT_TESTS_WORKFLOW).read_text(encoding="utf-8")
    job = _workflow_job(workflow, "rust-standalone")
    assert "if:" not in job
    assert "needs:" not in job
    assert "persist-credentials: false" in job
    assert "continue-on-error" not in job
    assert "rust-toolchain.toml" in job
    assert "timeout-minutes: 30" in job
    assert """toolchain=$(awk -F '"' '/^channel = / { print $2 }' rust-toolchain.toml)""" in job
    assert '"rust:${toolchain}-slim"' in job
    assert '--volume "$PWD:/source:ro"' in job
    assert "CARGO_TARGET_DIR=/tmp/target" in job
    assert "for name in python python2 python3 python2.* python3.* uv maturin; do" in job
    assert '[ -x "$tool" ] || continue' in job
    assert "exit 1" in job
    assert "cargo test --locked -p climate-indices" in job
    assert "setup-python" not in job
    assert "setup-uv" not in job


def test_crate_license_matches_original_bytes() -> None:
    """Match the pre-split root license from commit 150c7366."""
    original = (ROOT / "tests/fixture/original_combined_license.txt").read_bytes()
    assert (ROOT / "crates/climate-indices/LICENSE").read_bytes() == original


def test_primary_license_is_separate_and_all_original_notices_are_preserved() -> None:
    """The crate retains the original combined license as a standalone distribution."""
    combined = (ROOT / "crates/climate-indices/LICENSE").read_text(encoding="utf-8")
    primary, historical = combined.split("Previous Public Domain License", maxsplit=1)
    historical, third_party = historical.split("Third-party notices for", maxsplit=1)

    def legal_text(text: str) -> str:
        # Ignore only decorative separators, blank lines, and trailing whitespace.
        return "\n".join(line.rstrip() for line in text.splitlines() if line.strip() and not line.startswith("===="))

    primary_license = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert legal_text(primary_license) == legal_text(primary)
    assert "Copyright 2017, James Adams" in primary_license
    assert "NOAA" not in primary_license
    assert "Third-party" not in primary_license
    history = (ROOT / "docs/licensing.md").read_text(encoding="utf-8")
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert legal_text("Previous Public Domain License" + historical) == legal_text(
        history.split("```text\n", maxsplit=1)[1].split("```", maxsplit=1)[0]
    )
    assert legal_text("Third-party notices for" + third_party) == legal_text(
        notices.split("```text\n", maxsplit=1)[1].split("```", maxsplit=1)[0]
    )


def test_rust_job_proves_the_crates_package_correctly() -> None:
    """`climate-indices` must package with its README and license, and `climate-py` must not.

    `cargo publish --dry-run` resolves the packaged file list and builds it, which fails
    if a file the crate includes at compile time (`README.md`, `NUMERICS.md`) is missing
    from the package. `climate-py` inherits `publish = false`, so its metadata entry is
    an empty publish list; that is what makes it unpublishable by construction.
    """
    job = _native_jobs()["rust"]

    assert "cargo package --list -p climate-indices" in job
    for required in ("Cargo.toml", "README.md", "LICENSE", "NUMERICS.md", "src/lib.rs"):
        assert 'grep -qx "$required" package-files.txt' in job, required
    # The gate resolves against the crates.io registry, so it needs a finite bound.
    assert "timeout-minutes: 30" in job
    assert "cargo publish --dry-run -p climate-indices" in job
    assert 'select(.name == "climate-py") | .publish == []' in job


def _crate_quality_jobs() -> dict[str, str]:
    """The published-crate quality gates of the unit-test workflow, keyed by job name."""
    workflow = (ROOT / UNIT_TESTS_WORKFLOW).read_text(encoding="utf-8")
    return {name: _workflow_job(workflow, name) for name in ("rust-crate", "rust-msrv", "rust-semver", "rust-deny")}


def test_crate_quality_gates_are_separate_required_checks() -> None:
    """Each gate is its own job, so each can be a required check, and none may be optional."""
    for name, job in _crate_quality_jobs().items():
        assert job.strip(), f"{name} is missing from the workflow"
        assert not re.search(r"^    if:", job, re.MULTILINE), f"{name} must run on every event"
        assert "continue-on-error" not in job, f"{name} must block the pull request"
        assert "persist-credentials: false" in job, f"{name} never pushes, so it takes no token"
        # The gates resolve against the crates.io registry, so they need a finite bound.
        assert "timeout-minutes: 30" in job, f"{name} needs a bounded failure window"


def test_crate_gate_checks_the_packaged_artifact() -> None:
    """Lint, test, and document the unpacked `.crate`, not the checkout.

    A test fixture or README the crate needs but does not ship only fails here: the
    unpacked directory sits outside the checkout, so the workspace and its files cannot
    mask a packaging omission.
    """
    job = _crate_quality_jobs()["rust-crate"]

    assert "run: rustup toolchain install --no-self-update" in job
    assert "cargo package -p climate-indices --locked" in job
    assert "tar -xzf target/package/climate-indices-*.crate" in job
    assert '"$RUNNER_TEMP/published"' in job
    assert "cp rust-toolchain.toml" in job, "the unpacked crate must use the pinned compiler"
    assert "working-directory: ${{ runner.temp }}/published" in job
    assert "run: cargo clippy --all-targets --all-features --locked -- -D warnings" in job
    assert "run: cargo test --all-features --locked" in job
    assert 'RUSTDOCFLAGS: "-D warnings"' in job
    assert "run: cargo doc --all-features --no-deps --locked" in job
    assert "setup-python" not in job, "the crate gate proves Rust needs no Python"


def test_crate_gate_builds_on_the_declared_minimum_rust_version() -> None:
    """The MSRV job reads `rust-version` from the crate, so the gate cannot drift from the promise."""
    job = _crate_quality_jobs()["rust-msrv"]
    crate_manifest = _read_toml(Path("crates/climate-indices/Cargo.toml"))["package"]

    assert re.fullmatch(r"\d+\.\d+", crate_manifest["rust-version"]), "an exact MSRV, not a range"
    assert 'select(.name == "climate-indices") | .rust_version' in job
    assert 'rustup toolchain install "$msrv" --profile minimal --no-self-update' in job
    assert 'cargo "+$msrv" check -p climate-indices --lib --all-features --locked' in job


def test_crate_gate_checks_the_public_api_against_the_published_version() -> None:
    """cargo-semver-checks compares the crate with the latest crates.io release."""
    job = _crate_quality_jobs()["rust-semver"]

    assert "uses: taiki-e/install-action@" in job, "install the pinned tool, do not build it per run"
    assert "tool: cargo-semver-checks@" in job, "an unpinned linter can fail an unrelated change"
    assert "run: cargo semver-checks check-release -p climate-indices" in job


def test_crate_gate_enforces_the_dependency_policy() -> None:
    """`cargo deny` reads deny.toml, which must allow only redistributable licenses.

    The policy deliberately covers the whole workspace graph, not just the published
    crate: `climate-py` is not publishable, but the dependencies it compiles into the
    wheel are redistributed, so excluding it would hide pyo3 and numpy from the check.
    """
    job = _crate_quality_jobs()["rust-deny"]
    policy = _read_toml(Path("deny.toml"))

    assert "tool: cargo-deny@" in job, "an unpinned linter can fail an unrelated change"
    assert "run: cargo deny --locked check -W unmaintained -W unsound" in job
    # The bindings compile into the shipped wheel, so their licenses must be allowed too.
    assert {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"} <= set(policy["licenses"]["allow"])
    assert policy["graph"].get("exclude-dev") is True
    assert "exclude" not in policy["graph"], "the shipped bindings tree stays in scope"
    assert policy["bans"]["multiple-versions"] == "deny"
    assert policy["bans"]["wildcards"] == "deny"
    assert policy["sources"]["unknown-registry"] == "deny"
    assert policy["advisories"]["yanked"] == "deny"


def test_pyo3_seam_accepts_any_0x_crate_version_without_a_wildcard() -> None:
    """A path-only dependency is a wildcard, which `bans.wildcards = "deny"` rejects.

    The range admits stable 0.x releases from 0.1.0 (docs/adr/0021 decisions 2 and 3).
    A rehearsal pre-release, such as 0.2.0-rc.1, needs a bindings requirement change. A caret requirement such as `"0"` matches
    no pre-release, and `cargo package` then cannot resolve the workspace.
    """
    dependency = _read_toml(Path("crates/climate-py/Cargo.toml"))["dependencies"]["climate-indices"]

    assert dependency["path"] == "../climate-indices"
    assert dependency.get("version") == ">=0.1.0, <1.0.0", "an explicit 0.x range, not a wildcard"


def _crate_release_workflow() -> str:
    """The crates.io release workflow text (CRATE-10)."""
    return (ROOT / CRATE_RELEASE_WORKFLOW).read_text(encoding="utf-8")


def _crate_release_job_names() -> list[str]:
    """Every job name the crate release workflow declares, derived from its own text.

    Deriving the names keeps a newly added job inside the checks below instead of
    silently outside them; comparing that set against an explicit one is what makes
    adding a job a deliberate act.
    """
    workflow = _crate_release_workflow()
    candidates = re.findall(r"^  ([\w-]+):$", workflow, re.MULTILINE)
    return [name for name in candidates if "runs-on:" in _workflow_job(workflow, name)]


def _crate_release_jobs() -> dict[str, str]:
    """Every crate release job, keyed by job name."""
    workflow = _crate_release_workflow()
    return {name: _workflow_job(workflow, name) for name in _crate_release_job_names()}


def _uncommented(text: str) -> str:
    """Drop comment lines, so a commented-out guard cannot satisfy a check."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _job_needs(job: str) -> set[str]:
    """A job's `needs:` as a set, whether it is a scalar, a flow list, or a block list."""
    flow = re.search(r"^    needs: \[(.*?)\]$", job, re.MULTILINE)
    if flow:
        return {name.strip() for name in flow.group(1).split(",") if name.strip()}
    block = re.search(r"^    needs:\n((?:      - \S+\n)+)", job, re.MULTILINE)
    if block:
        return {line.strip().removeprefix("- ") for line in block.group(1).splitlines()}
    scalar = re.search(r"^    needs: (\S+)$", job, re.MULTILINE)
    return {scalar.group(1)} if scalar else set()


def _job_steps(job: str) -> list[str]:
    """A job's steps, each kept whole so a check binds to the step that carries it."""
    return re.split(r"^      - ", job, flags=re.MULTILINE)[1:]


def _job_scripts(job: str) -> str:
    """Only the `run:` scripts a job executes.

    A command left in a step name, an `env:` value, or a comment must not satisfy a
    check that claims the job runs it.
    """
    finder = re.compile(r"^ {8}run: (.*)((?:\n {10}.*)*)", re.MULTILINE)
    return "\n".join("".join(match.groups()) for match in finder.finditer(job))


def _tag_patterns(workflow: str) -> list[str]:
    """A workflow's push tag patterns, so an extra pattern cannot slip past unnoticed."""
    block = re.search(r"^on:\n  push:\n    tags:\n((?:      - .+\n)+)", workflow, re.MULTILINE)
    assert block, "the workflow must trigger on push tags"
    return [line.strip().removeprefix("- ").strip().strip("'\"") for line in block.group(1).splitlines()]


def test_crate_release_workflow_starts_only_on_crate_tags() -> None:
    """The two release namespaces cannot start each other (docs/adr/0021 decision 5).

    The Python release workflow triggers on `v*.*.*`, a glob that requires a leading
    `v`, so a `climate-indices-` prefix never matches it. A crate tag therefore
    publishes no PyPI release, and a Python tag publishes no crate.
    """
    assert _tag_patterns(_crate_release_workflow()) == ["climate-indices-v*"]
    python_release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert _tag_patterns(python_release) == ["v*.*.*"]
    assert not fnmatch.fnmatch("climate-indices-v0.1.0", "v*.*.*"), "the tag namespaces must not overlap"
    assert fnmatch.fnmatch("v0.1.0", "v*.*.*")


def test_python_release_tag_guard_rejects_the_crate_namespace() -> None:
    """The Python release workflow refuses a crate tag, so the two lanes cannot start each other.

    `test_crate_release_workflow_starts_only_on_crate_tags` compares the trigger globs;
    this executes the guard behind the Python glob, so the refusal rests on the workflow
    itself and not only on the trigger pattern.
    """
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    step = next(
        step
        for step in _job_steps(_workflow_job(workflow, "validate-release-tag"))
        if step.startswith("name: Validate SemVer tag\n")
    )
    script = _job_scripts(step).removeprefix("|\n")

    for tag, accepted in (("v3.1.0", True), ("climate-indices-v0.1.0", False)):
        result = subprocess.run(
            ["bash", "-c", script],
            env={**os.environ, "GITHUB_REF": f"refs/tags/{tag}"},
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == (0 if accepted else 1), result.stdout + result.stderr
        if not accepted:
            assert "::error::Release tags must match" in result.stdout


def test_crate_release_uses_oidc_not_a_stored_token() -> None:
    """The crate publish job exchanges the workflow's OIDC identity for a short-lived token.

    A stored crates.io token is the failure this prevents: it is long-lived, usable
    from anywhere, and must be revoked by hand after an exposure. Only the
    publish-only job may take the OIDC identity, and every other job is limited to a
    read-only token.
    """
    workflow = _crate_release_workflow()
    jobs = _crate_release_jobs()

    assert "secrets." not in workflow, "the crate release workflow needs no repository secret"
    assert workflow.count("id-token: write") == 1, "only the publish-only job may take an OIDC token"
    assert "id-token: write" in jobs["publish-crate"]
    assert "environment: release" in jobs["publish-crate"], "the manual approval gate was removed"
    assert "CARGO_REGISTRY_TOKEN: ${{ steps.auth.outputs.token }}" in jobs["publish-crate"]
    for name, job in jobs.items():
        if name == "publish-crate":
            continue
        # `write-all` would grant the OIDC identity too, and counting the literal
        # `id-token: write` would not see it.
        assert re.search(r"^    permissions:\n      contents: read$", job, re.MULTILINE), f"{name} is not read-only"
        assert "id-token" not in job, f"{name} must not reach the OIDC identity"
        assert "write-all" not in job, f"{name} must not take a write token"


def test_crate_release_pins_every_action_and_checkout() -> None:
    """Every action is pinned to a commit, and every checkout to the event SHA.

    An unpinned action is repository-controlled code running with the publish job's
    identity. A checkout of the tag name rather than the event SHA would let a tag
    moved mid-run build commits the tag validation never approved.
    """
    workflow = _crate_release_workflow()

    assert "ref: ${{ github.ref }}" not in workflow
    for use in re.findall(r"^\s+uses: (.*)$", workflow, re.MULTILINE):
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}\s{2}# v\S+", use.strip()), f"not commit-pinned: {use}"
    for name, job in _crate_release_jobs().items():
        checkouts = [step for step in _job_steps(job) if "uses: actions/checkout@" in step]
        assert checkouts, f"{name} has no checkout step"
        for step in checkouts:
            assert "ref: ${{ github.sha }}" in step, f"{name} checks out a mutable ref"


def test_crate_release_gates_match_the_required_checks() -> None:
    """The release gates run the same commands as the CRATE-09 required checks.

    They repeat in this workflow instead of trusting a run of uncertain provenance:
    a tag may point at a commit older than the tip of `main`. A gate that drifted
    from the check the pull request passed would let a release succeed where the
    check would have failed.
    """
    jobs = _crate_release_jobs()
    gate_names = ("crate-gate", "crate-msrv", "crate-semver", "crate-deny")
    required = "\n".join(_crate_quality_jobs().values())

    assert set(jobs) == {"validate-crate-tag", *gate_names, "publish-crate"}, "a new job needs covering here"

    gate_commands = {
        "crate-gate": (
            "cargo package -p climate-indices --locked",
            "cargo clippy --all-targets --all-features --locked -- -D warnings",
            "cargo test --all-features --locked",
            "cargo doc --all-features --no-deps --locked",
        ),
        "crate-msrv": ('cargo "+$msrv" check -p climate-indices --lib --all-features --locked',),
        "crate-semver": ("cargo semver-checks check-release -p climate-indices",),
        "crate-deny": ("cargo deny --locked check -W unmaintained -W unsound",),
    }
    commands = [command for gate in gate_commands.values() for command in gate]

    for command in commands:
        assert command in required, f"a required check no longer runs: {command}"
    for name, gate in gate_commands.items():
        scripts = _job_scripts(jobs[name])
        for command in gate:
            assert command in scripts, f"{name} no longer runs: {command}"
    assert any(
        'RUSTDOCFLAGS: "-D warnings"' in step
        for step in _job_steps(jobs["crate-gate"])
        if "cargo doc --all-features --no-deps --locked" in step
    ), "documenting the packaged crate must deny rustdoc warnings"
    assert 'RUSTDOCFLAGS: "-D warnings"' in required, "the required check no longer denies rustdoc warnings"

    for name, job in jobs.items():
        assert job.strip(), f"{name} is missing from the workflow"
        assert not re.search(r"^ +if:", job, re.MULTILINE), f"{name} must run on every crate tag"
        assert "continue-on-error" not in job, f"{name} must be able to block the release"
        assert "timeout-minutes: 30" in job, f"{name} needs a bounded failure window"

    for name in gate_names:
        assert _job_needs(jobs[name]) == {"validate-crate-tag"}, f"{name} no longer waits for the tag validation"
    assert _job_needs(jobs["publish-crate"]) == set(gate_names)
    assert "run: cargo publish -p climate-indices --locked" in jobs["publish-crate"]


@pytest.mark.parametrize(
    ("tag", "accepted"),
    [
        ("climate-indices-v0.1.0", True),
        ("climate-indices-v1.2.3", True),
        ("climate-indices-v0.1.0-rc.1", True),
        ("climate-indices-v0.2.0-rc.12", True),
        ("climate-indices-v0.1.0-beta.1", False),
        ("climate-indices-v0.1.0-rc1", False),
        ("climate-indices-v0.1.0-rc.", False),
        ("climate-indices-v0.1.0-rc.x", False),
        ("climate-indices-v0.1.0-rc.1.extra", False),
        ("climate-indices-v0.1.0-rc.1+build", False),
        ("climate-indices-v0.1", False),
        ("v0.1.0", False),
    ],
)
def test_crate_release_tag_guard_accepts_only_stable_or_numeric_rc_tags(tag: str, accepted: bool) -> None:
    """Execute the workflow's Bash guard against supported and unsupported tags."""
    step = next(
        step
        for step in _job_steps(_crate_release_jobs()["validate-crate-tag"])
        if step.startswith("name: Validate crate tag format\n")
    )
    script = _job_scripts(step).removeprefix("|\n")
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "GITHUB_REF": f"refs/tags/{tag}"},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == (0 if accepted else 1), result.stdout + result.stderr
    if not accepted:
        assert "::error::Crate tags must match" in result.stdout


def test_crate_release_validates_the_tag_version_and_changelog() -> None:
    """A tag, the manifest version, and the changelog heading must agree before publish.

    A crates.io version cannot be replaced, so a mismatch is caught while it is still
    free. The version is read from the manifest the publish job reads, so the guard
    cannot drift from it. The tag shape accepts only stable versions or numeric
    release candidates for the rehearsal lane (docs/release-process.md).
    """
    job = _uncommented(_crate_release_jobs()["validate-crate-tag"])
    manifest = _read_toml(Path("crates/climate-indices/Cargo.toml"))["package"]

    assert (
        'if [[ ! "${GITHUB_REF}" =~ '
        "^refs/tags/climate-indices-v[0-9]+\\.[0-9]+\\.[0-9]+(-rc\\.[0-9]+)?$ ]]; then" in job
    )
    assert "fetch-depth: 0" in job
    assert 'git merge-base --is-ancestor "${GITHUB_SHA}" "origin/main"' in job
    assert 'TAG_VERSION="${GITHUB_REF#refs/tags/climate-indices-v}"' in job
    assert 'select(.name == "climate-indices") | .version' in job
    assert "if grep -qE '^## Unreleased[[:space:]]*$' \"$changelog\"; then" in job, (
        "an '## Unreleased' heading would be published as a released version"
    )
    assert 'if ! grep -qE "^## ${CRATE_VERSION//./\\\\.} - [0-9]{4}-[0-9]{2}-[0-9]{2}$" "$changelog"; then' in job
    # The independent 0.x track, the pre-release shape of the rehearsal lane, and
    # the publishability the guard reads, must stay put.
    assert re.fullmatch(r"0\.\d+\.\d+(-rc\.\d+)?", manifest["version"])
    assert manifest["publish"] is True


def test_crate_changelog_guard_matches_the_manifest_version() -> None:
    """The crate changelog follows the same retitle-before-tag rule the release guard enforces.

    `cargo publish` spends a version permanently, so the tag validation refuses a tag
    whose manifest version has no dated changelog entry or whose changelog still carries
    an `## Unreleased` section. This applies the same rule to the tree, so a mismatch
    fails on a pull request instead of on the tag.
    """
    job = _uncommented(_crate_release_jobs()["validate-crate-tag"])
    changelog_ref = re.search(r"^[ \t]*changelog=(\S+)$", job, re.MULTILINE)
    assert changelog_ref, "the tag validation no longer names the changelog it reads"
    assert changelog_ref.group(1) == str(CRATE_CHANGELOG), "the guard must read the crate changelog"

    changelog = (ROOT / CRATE_CHANGELOG).read_text(encoding="utf-8")
    version = _read_toml(CRATE_MANIFEST)["package"]["version"]

    assert not re.search(r"^## Unreleased[ \t]*$", changelog, re.MULTILINE), (
        "an '## Unreleased' heading would be published as a released version"
    )
    assert re.search(rf"^## {re.escape(version)} - \d{{4}}-\d{{2}}-\d{{2}}$", changelog, re.MULTILINE), (
        f"{CRATE_CHANGELOG} has no dated '## {version} - YYYY-MM-DD' entry"
    )


def test_release_process_documents_the_crate_release_path() -> None:
    """The crate runbook must state its setup, its steps, its rehearsal, and its failure paths.

    A maintainer follows this page to publish a version that cannot be replaced, so
    each documented step and each failure path has to survive edits to the page.
    """
    release_process = (ROOT / "docs" / "release-process.md").read_text(encoding="utf-8")
    section = re.search(r"^## Rust crate release$.*?(?=^## |\Z)", release_process, re.MULTILINE | re.DOTALL)
    assert section is not None, "docs/release-process.md must keep the '## Rust crate release' section"
    text = " ".join(section.group(0).split())

    for expected in (
        ".github/workflows/crate-release.yml",
        "crates-io-auth-action",
        "climate-indices-vX.Y.Z",
        "### Crate one-time setup",
        "### Crate release steps",
        "### Crate rehearsal",
        "### If a crate publication fails",
        "### Crate post-release checks",
        # the one-time setup that gates the first release
        "trusted publisher",
        "refs/tags/climate-indices-v*",
        # the steps a release cannot skip
        "cargo update --workspace",
        "cargo publish --dry-run --locked -p climate-indices",
        "User-Agent",
        # the failure paths
        "crates.io rejects the OIDC exchange",
        "reports that the version is already uploaded",
        "A gate job fails",
        "The publish state is unknown",
        "cargo yank --version X.Y.Z climate-indices",
        # the rehearsal lane: the pre-release tag, the version a user has to name,
        # and the bindings range that makes a pre-release resolvable
        "climate-indices-vX.Y.Z-rc.N",
        "cargo add climate-indices@X.Y.Z",
        "Cargo skips a pre-release",
        'climate-indices = ">=0.1.0, <1.0.0"',
        "candidate versions found which didn't match: X.Y.Z-rc.N",
    ):
        assert expected in text, expected

    assert "climate_indices-rehearsal" not in text, "the crate lane publishes from this repository"


def test_native_jobs_run_on_every_event_and_pure_python_legs_stay_rust_free() -> None:
    """The Rust gates are ungated and blocking, and the pure-Python legs never build Rust.

    `continue-on-error` would let a red Rust job pass; every native job also installs the
    pinned toolchain itself rather than relying on whatever the runner ships.
    """
    workflow = (ROOT / UNIT_TESTS_WORKFLOW).read_text(encoding="utf-8")

    for name, job in _native_jobs().items():
        assert not re.search(r"^    if:", job, re.MULTILINE), f"{name} must run on every event"
        assert "continue-on-error" not in job, f"{name} must block the pull request"
        assert "run: rustup toolchain install --no-self-update" in job, f"{name} must install the pinned toolchain"
    for name in ("test", "test-full"):
        job = _workflow_job(workflow, name)
        assert "rustup" not in job
        assert "maturin" not in job
        assert "CLIMATE_INDICES_REQUIRE_NATIVE" not in job


def test_native_pytest_legs_are_the_pull_request_boundary_legs_of_the_core_job() -> None:
    """`test-native` reuses boundary legs of `test` rather than adding a new Cartesian product."""
    native_legs = _job_legs(_native_jobs()["test-native"])
    pull_request_legs, _ = _unit_test_legs()
    versions = _declared_python_versions()

    assert native_legs == {
        (versions[0], "ubuntu-latest"),
        (versions[-1], "ubuntu-latest"),
        (versions[-1], "macos-latest"),
    }
    assert native_legs <= pull_request_legs


def test_native_pytest_leg_builds_the_extension_and_cannot_skip_the_parity_suite() -> None:
    """The extension is built before pytest, and a missing build fails instead of skipping.

    `CLIMATE_INDICES_REQUIRE_NATIVE` makes `tests/conftest.py::import_native` raise rather than
    skip, so a broken build cannot pass by silently dropping `tests/test_native_parity.py`.
    `tests/test_native_backend.py` tests that behavior; this pins that the leg sets the variable.
    The core suite runs exactly as the `test` leg runs it; the consolidated parity suite and the
    recorded parity maxima are steps only the native leg can run.
    """
    workflow = (ROOT / UNIT_TESTS_WORKFLOW).read_text(encoding="utf-8")
    native_job = _native_jobs()["test-native"]
    install = "run: uv sync --locked --dev"
    build = "run: uv run --no-sync --no-build maturin develop --release"
    check = 'run: uv run --no-sync --no-build python -c "import climate_indices._native"'

    assert "CLIMATE_INDICES_REQUIRE_NATIVE: '1'" in native_job
    assert native_job.index(install) < native_job.index(build) < native_job.index(check)
    assert native_job.index(check) < native_job.index("- name: Run core tests")

    def step(job: str, name: str) -> str:
        # a step runs until the blank line that separates it from the next
        return job.split(f"- name: {name}\n")[1].split("\n\n")[0].strip()

    assert step(native_job, "Run core tests") == step(_workflow_job(workflow, "test"), "Run core tests")
    suite = step(native_job, "Run the consolidated parity suite")
    assert "tests/test_native_parity_registry.py tests/test_native_e2e_parity.py" in suite
    maxima = step(native_job, "Record cross-backend parity maxima")
    assert 'scripts/native_parity_maxima.py >> "$GITHUB_STEP_SUMMARY"' in maxima


def test_native_wheel_job_checks_built_wheels_outside_the_checkout() -> None:
    """Development wheels are built at both boundaries on Linux and macOS, plus a Windows smoke build.

    Each wheel is installed into a fresh venv and imported from outside the checkout, and SPI must
    reach the bundled kernels. This job validates the developer build; the published wheel matrix,
    its glibc floor, and the abi3 policy live in `release.yml` (ADR-0018).
    """
    wheel_job = _native_jobs()["native-wheel"]
    versions = _declared_python_versions()

    assert _job_legs(wheel_job) == {
        (python, system) for python in (versions[0], versions[-1]) for system in ("ubuntu-latest", "macos-latest")
    } | {(versions[-1], "windows-latest")}
    assert (
        "run: uv run --no-sync --no-build maturin build --release --interpreter python --out dist-native" in wheel_job
    )
    assert "*manylinux*" in wheel_job
    assert wheel_job.index("python -m venv") < wheel_job.index('cd "${RUNNER_TEMP}"') < wheel_job.index("<<'PY'")
    assert "assert not Path(module.__file__).resolve().is_relative_to(workspace), module.__file__" in wheel_job
    assert (
        'assert recorder.calls == {"gamma_parameters", "gamma_probabilities", '
        '"gamma_ks_statistics", "norm_ppf", "trailing_sum"}' in wheel_job
    )
    assert 'xr.DataArray(values, dims="time"' in wheel_job, "the smoke test must exercise the DataArray API"
    assert "climate_indices.spi(dataarray, 3, indices.Distribution.gamma)" in wheel_job


def test_every_built_artifact_embeds_and_reports_its_source_commit() -> None:
    """Hatchling embeds the commit itself; the maturin wheels need hatch_build.py run first (#1470).

    Each job that installs a built wheel outside the checkout asserts the commit it reports.
    """
    pyproject = _read_toml(Path("pyproject.toml"))
    assert "custom" in pyproject["tool"]["hatch"]["build"]["hooks"]
    assert (ROOT / "hatch_build.py").is_file()
    build_info = "src/climate_indices/_build_info.json"
    assert {"path": build_info, "format": "wheel"} in pyproject["tool"]["maturin"]["include"]
    assert build_info in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    build_wheels = _workflow_job(release, "build-wheels")
    assert build_wheels.index("run: python hatch_build.py") < build_wheels.index("uses: PyO3/maturin-action@")
    native_wheel = _native_jobs()["native-wheel"]
    assert native_wheel.index("python hatch_build.py") < native_wheel.index("maturin build")
    reported = 'assert identity["git_commit"] == os.environ["GITHUB_SHA"], identity'
    for job in (_workflow_job(release, "wheel-check"), _workflow_job(release, "no-rust-install"), native_wheel):
        assert reported in job
        assert 'identity["git_commit_source"] == "build-metadata"' in job


def test_every_git_lfs_tracked_file_is_stored_as_an_lfs_pointer() -> None:
    """A file marked `filter=lfs` but committed as a plain blob reports a dirty tree.

    `hatch_build.py` reads `git status --porcelain --untracked-files=no`, and the release
    workflow requires the embedded `git_dirty` to be false. With Git LFS configured, the
    clean filter makes `git status` report such a file as modified on a fresh checkout,
    which failed the 3.1.1 release rehearsal (#1524).
    """
    # an empty file is its own LFS pointer (git-lfs spec), so the empty blob is valid too
    pointer = re.compile(rb"(?:version https://git-lfs\.github\.com/spec/v1\noid sha256:[0-9a-f]{64}\nsize [0-9]+\n)?")
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    attributes = subprocess.run(
        ["git", "check-attr", "--cached", "-z", "--stdin", "filter"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        input=listed,
    ).stdout.split(b"\0")
    tracked_as_lfs = [
        os.fsdecode(path)
        for path, attribute, value in zip(attributes[0::3], attributes[1::3], attributes[2::3], strict=False)
        if attribute == b"filter" and value == b"lfs"
    ]
    assert tracked_as_lfs, "no tracked file is marked filter=lfs, so the check below proves nothing"
    invalid_pointers = [
        path
        for path in tracked_as_lfs
        if not pointer.fullmatch(
            subprocess.run(["git", "cat-file", "blob", f":{path}"], cwd=ROOT, check=True, capture_output=True).stdout
        )
    ]
    assert not invalid_pointers, (
        f"{invalid_pointers} are marked filter=lfs but their staged blobs are not valid LFS pointers; "
        "store them as LFS objects or clear the filter attribute for those paths"
    )


@pytest.mark.parametrize("change", ["new", "renamed", "replaced"])
@pytest.mark.parametrize(
    ("blob", "valid"),
    [
        (b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 123\n", True),
        (b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 0\n", True),
        (b"", True),
        (b"plain data\n", False),
        (b"version https://git-lfsXgithubXcom/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 1\n", False),
        (b"version https://git-lfs.github.com/spec/v1\n", False),
        (b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n", False),
        (b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"g" * 64 + b"\nsize 123\n", False),
        (b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\n", False),
        (b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize -1\n", False),
        (b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize NaN\n", False),
        (b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 1\nextra\n", False),
    ],
)
def test_lfs_integrity_checks_the_staged_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str, blob: bytes, valid: bool
) -> None:
    """Check staged additions, renames, and replacements without using HEAD or worktree bytes."""

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "filter.lfs.clean=", "-c", "filter.lfs.process=", "-c", "filter.lfs.required=false", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "--quiet")
    attributes = tmp_path / ".gitattributes"
    attributes.write_text("*.bin -filter\n", encoding="utf-8")
    if change != "new":
        (tmp_path / ("old.bin" if change == "renamed" else "fixture.bin")).write_bytes(b"old plain data\n")
    git("add", ".")
    git(
        "-c",
        "user.name=release",
        "-c",
        "user.email=release@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        "baseline",
    )
    if change == "renamed":
        git("mv", "old.bin", "fixture.bin")
    attributes.write_text("*.bin filter=lfs\n", encoding="utf-8")
    fixture = tmp_path / "fixture.bin"
    fixture.write_bytes(blob)
    git("add", ".gitattributes", "fixture.bin")
    attributes.write_text("*.bin -filter\n", encoding="utf-8")
    fixture.write_bytes(b"unstaged data\n")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)

    if valid:
        test_every_git_lfs_tracked_file_is_stored_as_an_lfs_pointer()
    else:
        with pytest.raises(AssertionError, match="not valid LFS pointers"):
            test_every_git_lfs_tracked_file_is_stored_as_an_lfs_pointer()


def test_climate_py_builds_abi3_wheels() -> None:
    """abi3 lets one wheel per platform cover every supported Python.

    Dropping the feature silently reintroduces an interpreter axis in the release wheel matrix,
    so it is pinned here rather than only in the workflow.
    """
    manifest = _read_toml(Path("crates/climate-py/Cargo.toml"))
    pyo3 = manifest["dependencies"]["pyo3"]

    minimum = _declared_python_versions()[0].replace(".", "")
    assert {feature for feature in pyo3["features"] if feature.startswith("abi3-py")} == {f"abi3-py{minimum}"}


def test_docker_uses_latest_supported_python() -> None:
    """Every Docker build stage must use the latest classified Python minor."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    base_versions = re.findall(r"^FROM python:(\d+\.\d+)-slim", dockerfile, re.MULTILINE)
    assert base_versions, "Dockerfile must use an official python:<minor>-slim base image"
    assert set(base_versions) == {_declared_python_versions()[-1]}


def test_front_page_python_support_matches_classifiers() -> None:
    """README support range and front-page badges must match metadata."""
    versions = _declared_python_versions()
    badge_url = _expected_badge_url()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    release_process = (ROOT / "docs" / "release-process.md").read_text(encoding="utf-8")

    assert f"Supports Python **{versions[0]}–{versions[-1]}**." in readme

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
    assert "ref: ${{ github.sha }}" in workflow
    assert "ref: ${{ github.ref }}" not in workflow, (
        "checking out the mutable tag name lets a moved tag build commits the validation never approved"
    )


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


def test_release_workflow_builds_an_abi3_wheel_for_every_published_platform() -> None:
    """`_native` ships as one abi3 wheel per platform, built with maturin at the pure wheel's version.

    The Linux wheels take their glibc floor from the manylinux container; macOS and Windows build
    on their own runner. musllinux is deliberately not published: musl users get the pure wheel
    (ADR-0018). An abi3 wheel has no interpreter axis, so this matrix has no Python matrix.
    """
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    build_wheels = _workflow_job(workflow, "build-wheels")

    targets = (
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "x86_64-pc-windows-msvc",
    )
    for target in targets:
        assert f"target: {target}" in build_wheels, target
    assert build_wheels.count("manylinux: '2_28'") == 2, "both Linux wheels share one glibc floor"
    assert "musllinux" not in build_wheels
    assert not re.search(r"^\s+python-version:\s*\[", build_wheels, re.MULTILINE), "abi3 needs no Python axis"
    assert "uses: PyO3/maturin-action@" in build_wheels
    assert "args: --release --locked --out dist" in build_wheels
    assert "*-abi3-*" in build_wheels, "a non-abi3 tag must fail the build, not publish"
    assert "name: dist-native-${{ matrix.target }}" in build_wheels


def test_release_workflow_wheel_check_installs_every_published_wheel() -> None:
    """Every platform wheel is installed outside the checkout, and pip prefers it over the pure wheel.

    The install resolves `climate_indices` by name with only the built artifacts visible, so the
    assertion that `py3-none-any` was not chosen covers the fallback rule ADR-0018 depends on.
    SPI must then reach the bundled gamma kernels, which proves the extension dispatches.
    """
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    wheel_check = _workflow_job(workflow, "wheel-check")

    assert "pattern: dist*" in wheel_check
    assert "merge-multiple: true" in wheel_check
    for pattern in (
        "*manylinux*x86_64.whl",
        "*manylinux*aarch64.whl",
        "*macosx*arm64.whl",
        "*macosx*x86_64.whl",
        "*win_amd64.whl",
    ):
        assert f"wheel: '{pattern}'" in wheel_check, pattern
    assert wheel_check.count("- os: ") == 6, "one leg per platform wheel, with the oldest Python on Linux"
    assert 'python -m venv "${RUNNER_TEMP}/wheel-check"' in wheel_check
    assert 'cd "${RUNNER_TEMP}"' in wheel_check
    assert "--dry-run --no-index --no-deps" in wheel_check
    assert '--report "${RUNNER_TEMP}/resolution.json"' in wheel_check
    assert 'assert "py3-none-any" not in resolved, resolved' in wheel_check
    assert "assert not Path(module.__file__).resolve().is_relative_to(workspace), module.__file__" in wheel_check
    assert (
        'assert recorder.calls == {"gamma_parameters", "gamma_probabilities", '
        '"gamma_ks_statistics", "norm_ppf", "trailing_sum"}' in wheel_check
    )
    declared_scripts = " ".join(_read_toml(Path("pyproject.toml"))["project"]["scripts"])
    assert f"for entry_point in {declared_scripts}; do" in wheel_check
    assert 'xr.DataArray(values, dims="time"' in wheel_check, "the smoke test must exercise the DataArray API"
    assert "climate_indices.spi(dataarray, 3, indices.Distribution.gamma)" in wheel_check


def test_release_workflow_installs_wheel_on_boundary_pythons() -> None:
    """The abi3 wheel installs on the oldest and newest supported Pythons, and that gates publishing.

    The boundary versions run in `wheel-check` (the platform wheel) and in `no-rust-install` (the
    pure wheel and the sdist); everything else installs on the newest version only.
    """
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    wheel_check = _workflow_job(workflow, "wheel-check")
    no_rust = _workflow_job(workflow, "no-rust-install")
    versions = _declared_python_versions()

    assert re.findall(r"- os: (\S+)\n\s+python-version: '(\d+\.\d+)'", wheel_check) == [
        ("ubuntu-latest", versions[0]),
        ("ubuntu-latest", versions[-1]),
        ("ubuntu-24.04-arm", versions[-1]),
        ("macos-latest", versions[-1]),
        ("macos-15-intel", versions[-1]),
        ("windows-latest", versions[-1]),
    ], "one leg per platform wheel, and both boundaries on Linux"
    assert re.findall(r"- python-version: '(\d+\.\d+)'", no_rust) == [versions[0], versions[-1]]
    assert "needs: [build, build-wheels, wheel-check, no-rust-install]" in workflow, (
        "publishing must wait for every installation check"
    )


def test_release_workflow_tests_the_no_rust_install_path() -> None:
    """The supported install path without a Rust toolchain is the pure wheel and the sdist.

    The job removes cargo and rustc from PATH before installing, so an sdist build that started
    depending on the Rust toolchain would fail here instead of on a user's machine.
    """
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    no_rust = _workflow_job(workflow, "no-rust-install")

    assert "needs: build" in no_rust
    assert "uv sync --locked --dev --no-install-project" in no_rust, "install the artifact, not the checkout"
    assert "grep -vE 'cargo|rustup'" in no_rust
    assert "if command -v cargo" in no_rust
    assert "if command -v rustc" in no_rust
    assert "-name '*py3-none-any.whl'" in no_rust
    assert 'importlib.util.find_spec("climate_indices._native") is None' in no_rust
    assert 'wheel.name.endswith("-py3-none-any.whl")' in no_rust
    assert 'name.endswith((".so", ".pyd", ".dylib"))' in no_rust
    assert "uv pip install --python .venv/bin/python --no-deps --force-reinstall" in no_rust, (
        "the suite must run against the wheel the sdist builds, not the shipped wheel"
    )
    assert "is_relative_to(prefix)" in no_rust, "the installed-environment check must use sys.prefix"
    assert "uv run --no-sync --no-build pytest -n auto" in no_rust
    order = (
        no_rust.index("uv sync --locked --dev --no-install-project"),
        no_rust.index("- name: Remove the Rust toolchain from PATH"),
        no_rust.index("- name: Install the pure-Python wheel"),
        no_rust.index("- name: Build the sdist into the pure-Python wheel"),
        no_rust.index("- name: Install the wheel rebuilt from the sdist"),
        no_rust.index("- name: Assert the installed package is the pure-Python implementation"),
        no_rust.index("- name: Run core tests against the installed package"),
    )
    assert order == tuple(sorted(order)), "the toolchain must be gone before anything is installed"
    assert 'map(Requirement, wheel_metadata.get_all("Requires-Dist") or [])' in no_rust
    assert 'if requirement.name == "icechunk"' in no_rust, "the check must match the distribution, not a substring"
    assert 'assert "icechunk" in selecting' in no_rust, "the named extra, not another one, must select Icechunk"
    assert 'assert "" not in selecting' in no_rust, "the pure wheel must prove Icechunk is not a core requirement"
    assert "import climate_indices.icechunk_io" in no_rust
    assert 'assert "icechunk" not in sys.modules' in no_rust


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


def test_v310_stable_dataarray_api_surface_is_pinned() -> None:
    """The 3.1.0 promotion fixes which DataArray routes are Stable and which stay Beta.

    ADR-0012 promotes the DataArray index APIs for 3.1.0; ADR-0022 keeps the compound
    Dataset routes and the energy-drought event statistics Beta. A name that silently
    moved between the tiers would change the compatibility promise, so both sets are
    pinned, and `compound` stays an explicit-import package rather than a root export.
    Export alone cannot tell the tiers apart, so each Beta route must also keep its
    `Yes (Beta)` row in the compatibility matrix; the compound routes'
    `BetaFeatureWarning` is pinned in tests/test_compound_xarray.py and
    tests/test_compound_catalog.py. SPAI is Beta at the package root in 3.2.0 (#1480).
    """
    import climate_indices
    from climate_indices import compound

    for name in sorted(STABLE_DATAARRAY_API_NAMES):
        assert name in climate_indices.__all__, name
        assert callable(getattr(climate_indices, name)), name

    for module_name, names in sorted(STABLE_DATAARRAY_MODULE_API_NAMES.items()):
        assert module_name in climate_indices.__all__, module_name
        module = getattr(climate_indices, module_name)
        for name in sorted(names):
            assert name in module.__all__, f"{module_name}.{name}"
            assert callable(getattr(module, name)), f"{module_name}.{name}"

    assert "compound" not in climate_indices.__all__, "compound stays an explicit import (ADR-0022)"
    matrix = (ROOT / "docs" / "xarray_compatibility.md").read_text(encoding="utf-8")
    # (Supported, Coverage) per feature row; only the Coverage cell may contain a pipe
    rows = [line.split("|", 3)[2:] for line in matrix.splitlines() if line.startswith("| `DataArray` inputs for ")]
    for module_name, names in sorted(BETA_DATAARRAY_API_NAMES.items()):
        module = compound if module_name == "compound" else getattr(climate_indices, module_name)
        for name in sorted(names):
            assert name in module.__all__, f"{module_name}.{name}"
            tiers = {tier.strip() for tier, coverage in rows if f"`{module_name}.{name}()`" in coverage}
            assert tiers == {"Yes (Beta)"}, f"{module_name}.{name} must stay Beta in the compatibility matrix: {tiers}"

    for name in sorted(BETA_ROOT_API_NAMES):
        assert name in climate_indices.__all__, name
        assert name not in STABLE_DATAARRAY_API_NAMES, name
        tiers = {tier.strip() for tier, coverage in rows if f"`climate_indices.{name}()`" in coverage}
        assert tiers == {"Yes (Beta)"}, f"{name} must stay Beta in the compatibility matrix: {tiers}"


def test_icechunk_is_an_optional_extra_and_imports_lazily() -> None:
    """A core install must not need Icechunk: the extra carries it and the code imports it lazily."""
    project = _read_toml(Path("pyproject.toml"))["project"]

    assert not [dependency for dependency in project["dependencies"] if "icechunk" in dependency], (
        "icechunk must stay out of the core dependencies"
    )
    assert project["optional-dependencies"]["icechunk"] == ["icechunk>=1.1.21,<2"]
    module = (ROOT / "src" / "climate_indices" / "icechunk_io.py").read_text(encoding="utf-8")
    assert not re.search(r"^(?:import|from)\s+icechunk", _uncommented(module), re.MULTILINE), (
        "icechunk_io must import Icechunk lazily, so a core install still imports it"
    )
    # A fresh interpreter proves the import chain, not this session's module cache.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, climate_indices, climate_indices.icechunk_io; "
            "assert 'icechunk' not in sys.modules, 'the core import path imported Icechunk'",
        ],
        check=True,
    )


def test_tag_date_reads_the_annotated_tag_and_ignores_lightweight_tags(tmp_path: Path) -> None:
    """The tag-date guard compares against the tag's own date; a missing ref is an error."""
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "release",
        "GIT_AUTHOR_EMAIL": "release@example.invalid",
        "GIT_COMMITTER_NAME": "release",
        "GIT_COMMITTER_EMAIL": "release@example.invalid",
        "GIT_AUTHOR_DATE": "2026-10-12T12:00:00-04:00",
        "GIT_COMMITTER_DATE": "2026-10-12T12:00:00-04:00",
    }

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env)

    git("init", "--quiet")
    (repo / "release-record.md").write_text("record\n", encoding="utf-8")
    git("add", "release-record.md")
    git("-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "record")
    git("-c", "tag.gpgsign=false", "tag", "--annotate", "v1.2.3", "-m", "release")
    git("tag", "v1.2.4")

    assert _tag_date("v1.2.3", repo) == date(2026, 10, 12)
    assert _tag_date("v1.2.4", repo) is None
    with pytest.raises(AssertionError, match="missing from the checkout"):
        _tag_date("v9.9.9", repo)


def test_release_notes_must_carry_the_tag_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The versioned release-notes document must state the date the tag build verifies."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "pyproject.toml").write_text('version = "3.0.0"\n', encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("## [3.0.0] - 2026-10-12\n", encoding="utf-8")
    release_notes = tmp_path / "docs" / "release-notes-3.0.0.md"

    def stubbed_tag_date(tag: str, repo: Path | None = None) -> date | None:
        return date(2026, 10, 12)

    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "_tag_date", stubbed_tag_date)
    monkeypatch.setenv("GITHUB_REF", "refs/tags/v3.0.0")

    release_notes.write_text("# notes\n\nReleased 2026-10-12.\n", encoding="utf-8")
    test_changelog_top_entry_matches_pyproject_version()

    release_notes.write_text("# notes\n\nReleased 2026-10-05.\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="release date"):
        test_changelog_top_entry_matches_pyproject_version()


@pytest.mark.parametrize(
    ("github_ref", "release_date", "tag_date", "accepted"),
    [
        ("", "unreleased", None, True),
        ("refs/heads/main", "unreleased", None, True),
        ("refs/tags/v3.0.0", "2026-10-10", "2026-10-10", True),
        # A lightweight tag carries no date, so only calendar validity is checked.
        ("refs/tags/v3.0.0", "2026-10-10", None, True),
        ("refs/tags/v3.0.0", "2026-10-10", "2026-10-11", False),
        ("refs/tags/v3.0.0", "unreleased", None, False),
        ("refs/tags/v3.0.0", "", None, False),
        ("refs/tags/v3.0.0", "Oct 10, 2026", None, False),
        ("refs/tags/v3.0.0", "2026-02-30", None, False),
        ("refs/tags/v3.0.0", "2026-13-01", None, False),
    ],
)
def test_changelog_date_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    github_ref: str,
    release_date: str,
    tag_date: str | None,
    accepted: bool,
) -> None:
    """Tag builds require a top entry dated on the tag's own day; development can stay unreleased."""
    (tmp_path / "pyproject.toml").write_text('version = "3.0.0"\n', encoding="utf-8")
    suffix = f" - {release_date}" if release_date else ""
    (tmp_path / "CHANGELOG.md").write_text(f"## [3.0.0]{suffix}\n\n## [2.4.0] - 2026-04-05\n", encoding="utf-8")

    def stubbed_tag_date(tag: str, repo: Path | None = None) -> date | None:  # noqa: ARG001 - matches _tag_date
        return date.fromisoformat(tag_date) if tag_date else None

    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "_tag_date", stubbed_tag_date)
    monkeypatch.setenv("GITHUB_REF", github_ref)
    if accepted:
        test_changelog_top_entry_matches_pyproject_version()
    else:
        with pytest.raises(AssertionError, match="release date"):
            test_changelog_top_entry_matches_pyproject_version()


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
    """CHANGELOG.md top release block must match the version and be dated on tag builds.

    Addresses: F8 — catches version bump in one file without the other; on a tag
    build the date must also be the annotated tag's own tagger date.
    """
    pyproject_version = _read_pyproject_version()
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(r"^## \[(\d+\.\d+\.\d+)\](.*)$", changelog, re.MULTILINE)
    assert match is not None, "No versioned release block (## [X.Y.Z]) found in CHANGELOG.md"
    changelog_version = match.group(1)
    assert changelog_version == pyproject_version, (
        f"CHANGELOG.md top entry [{changelog_version}] does not match "
        f"pyproject.toml version [{pyproject_version}] — update one to match the other"
    )
    if os.environ.get("GITHUB_REF", "").startswith("refs/tags/"):
        tag = os.environ["GITHUB_REF"].removeprefix("refs/tags/")
        date_match = re.fullmatch(r" - (\d{4}-\d{2}-\d{2})", match.group(2))
        release_date = None
        if date_match:
            with contextlib.suppress(ValueError):
                release_date = date.fromisoformat(date_match.group(1))
        assert release_date is not None, (
            "CHANGELOG.md top entry must have a valid calendar release date (YYYY-MM-DD) before tagging"
        )
        tag_date = _tag_date(tag)
        if tag_date is not None:
            assert release_date == tag_date, (
                f"CHANGELOG.md top entry release date {release_date} does not match the annotated tag "
                f"{tag} release date {tag_date} — re-date the release record, freeze the candidate again, "
                "and rehearse it"
            )
        release_notes = ROOT / f"docs/release-notes-{pyproject_version}.md"
        if release_notes.exists():
            opening = "\n".join(release_notes.read_text(encoding="utf-8").splitlines()[:6])
            assert release_date.isoformat() in opening, (
                f"{release_notes.relative_to(ROOT)} must state the release date {release_date.isoformat()} "
                "in its opening paragraph"
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
