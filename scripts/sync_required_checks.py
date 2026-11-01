"""Synchronize the required status checks on the default branch with the workflow.

`main` branch protection requires a fixed list of status-check contexts. GitHub
names a matrix job's context `<job> (<python>, <os>)` and a plain job's context
`<job>`, so a change to a job's matrix silently renames the contexts. GitHub then
reports the old name as "Expected - Waiting for status to be reported", which
blocks every pull request until someone updates the repository setting by hand.

`.github/required-status-checks.txt` is the reviewed list. This script derives the
context names from the workflow, checks them against the list and the live
setting, and with ``--apply`` writes the list to the setting and reads it back.

    uv run python scripts/sync_required_checks.py            # report only
    uv run python scripts/sync_required_checks.py --apply    # update the setting

The token must be able to write branch protection, which means admin on the
repository. ``gh`` infers the repository from the checkout; pass ``--repo`` to
override it.

The parser rejects custom job names and unsupported matrices instead of guessing
context names. Apply requires both local files to match the protected branch and
checks for concurrent protection changes before writing. GitHub does not offer
an atomic compare-and-update operation here; do not run concurrent updates.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess  # nosec B404 - Only the fixed gh executable runs, without a shell.
import sys
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = Path(".github/workflows/unit-tests-workflow.yml")
CHECKS_FILE = Path(".github/required-status-checks.txt")

# Keep the required jobs independent of the list that the script validates.
REQUIRED_JOBS = frozenset(
    {
        "lint",
        "test",
        "test-minimum-deps",
        "notebooks",
        "security-audit",
        "test-native",
        "rust",
        "native-wheel",
        "rust-standalone",
        "rust-crate",
        "rust-msrv",
        "rust-semver",
        "rust-deny",
    }
)

# The GitHub Actions app, which owns every check in this workflow.
ACTIONS_APP_ID = 15368


def declared_checks(path: Path | None = None) -> list[str]:
    """Read the reviewed required contexts in file order.

    Args:
        path: An alternate checks file. Defaults to the repository checks file.

    Returns:
        The context names, without comments or blank lines.
    """
    text = (path or (REPO_ROOT / CHECKS_FILE)).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _job_text(workflow: str, job: str) -> str:
    """One top-level job's text, without the comment block that precedes the next job."""
    match = re.search(rf"^  {re.escape(job)}:\n(.*?)(?=^  [\w-]+:\n|\Z)", workflow, re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError(f"no top-level job named {job!r}")
    return re.split(r"^  #", match.group(1), maxsplit=1, flags=re.MULTILINE)[0]


def _job_contexts(job: str, body: str) -> set[str]:
    """The contexts one job reports: one per matrix leg, or the bare job name."""
    if re.search(r"^    name:", body, re.MULTILINE):
        raise ValueError(f"job {job!r}: explicit job names are not supported")
    condition = re.search(r"^    if:\s*([^\n]+)", body, re.MULTILINE)
    always = condition is not None and condition[1].strip() in {"always()", "${{ always() }}"}
    if condition and not always:
        raise ValueError(f"job {job!r}: condition may skip pull requests")
    if re.search(r"^    needs:", body, re.MULTILINE) and not always:
        raise ValueError(f"job {job!r}: dependencies may skip pull requests; use an unconditional job")
    if not re.search(r"^\s+matrix:", body, re.MULTILINE):
        return {job}

    # Fail closed: only the repository's ordered two-axis matrix format is supported.
    matrix = re.search(r"^      matrix:(.*?)(?=^ {0,6}\S|\Z)", body, re.MULTILINE | re.DOTALL)
    shape = re.fullmatch(
        r"\n        python-version: \[([^]\n]+)]\n        os: \[([^]\n]+)]\n"
        r"(?:        include:\n((?:          - python-version: ['\"]\d+\.\d+['\"]\n"
        r"            os: [\w.-]+\n)+))?\s*",
        matrix[1] if matrix else "",
    )
    if (
        not shape
        or not re.fullmatch(r"['\"]\d+\.\d+['\"](?:,\s*['\"]\d+\.\d+['\"])*", shape[1])
        or not re.fullmatch(r"[\w.-]+(?:,\s*[\w.-]+)*", shape[2])
    ):
        raise ValueError(f"job {job!r}: parser does not support this matrix shape")
    legs = {
        (python, system) for python in re.findall(r"\d+\.\d+", shape[1]) for system in re.findall(r"[\w.-]+", shape[2])
    }
    legs |= set(re.findall(r"- python-version: ['\"](\d+\.\d+)['\"]\n            os: ([\w.-]+)", shape[3] or ""))
    return {f"{job} ({python}, {system})" for python, system in legs}


def workflow_contexts(path: Path | None = None) -> set[str]:
    """Derive contexts for the independently defined required jobs.

    Args:
        path: An alternate workflow. Defaults to the repository unit-test workflow.

    Returns:
        The required context names.

    Raises:
        ValueError: A required job has an unsupported name, condition, dependency, or matrix.
    """
    text = (path or (REPO_ROOT / WORKFLOW)).read_text(encoding="utf-8")
    contexts: set[str] = set()
    for job in sorted(REQUIRED_JOBS):
        contexts |= _job_contexts(job, _job_text(text, job))
    return contexts


def _gh_api(
    path: str, repo: str | None, *, method: str | None = None, input_data: dict[str, object] | None = None
) -> str:
    """Call `gh api` and return stdout. Without ``repo`` gh fills the {owner}/{repo} placeholders itself."""
    if repo is not None and not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("repo must be owner/name")
    if method not in {None, "PATCH"} or not path.startswith("repos/{owner}/{repo}/"):
        raise ValueError("unsupported API request")
    endpoint = path if repo is None else path.replace("{owner}/{repo}", repo, 1)
    command = ["gh", "api"]
    if method:
        command += ["-X", method]
    if input_data is not None:
        command += ["--input", "-"]
    command += ["--", endpoint]
    # Arguments use a fixed API route and separate options; no shell interprets them.
    result = subprocess.run(  # nosec B603 B607
        command,
        input=json.dumps(input_data) if input_data is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"{' '.join(command)} failed:\n{result.stderr.strip()}")
    return result.stdout


def live_checks(repo: str | None, branch: str) -> tuple[dict[str, object], list[str]]:
    """Read the live required status checks.

    Args:
        repo: The owner/name, or None to use the checkout's repository.
        branch: The protected branch name.

    Returns:
        The API payload and its context names.
    """
    path = f"repos/{{owner}}/{{repo}}/branches/{quote(branch, safe='')}/protection/required_status_checks"
    try:
        payload = json.loads(_gh_api(path, repo))
    except json.JSONDecodeError as error:
        raise SystemExit(f"unexpected response for {path}: {error}") from error
    return payload, list(payload["contexts"])


def main(argv: list[str] | None = None) -> int:
    """Compare or apply the reviewed checks without changing other protection settings.

    Args:
        argv: Command arguments, or None to use sys.argv.

    Returns:
        Zero on success, or one if validation or verification fails.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write the reviewed list to branch protection")
    parser.add_argument("--branch", default="main", help="protected branch to update (default: main)")
    parser.add_argument("--repo", default=None, help="owner/name; gh infers it from the checkout otherwise")
    options = parser.parse_args(argv)

    declared = declared_checks()
    try:
        derived = workflow_contexts()
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    if sorted(declared) != sorted(derived):
        print("The reviewed list and the workflow disagree; update the list first.", file=sys.stderr)
        print(f"  only in {CHECKS_FILE}: {sorted(set(declared) - derived)}", file=sys.stderr)
        print(f"  only in the workflow: {sorted(derived - set(declared))}", file=sys.stderr)
        return 1

    payload, live = live_checks(options.repo, options.branch)
    strict = payload.get("strict")
    print(f"{len(declared)} required contexts; live: {len(live)}; strict: {strict}")
    added, removed = sorted(set(declared) - set(live)), sorted(set(live) - set(declared))
    print(f"  to add: {added or 'none'}")
    print(f"  to remove: {removed or 'none'}")

    if not options.apply:
        print("\nReport only. Re-run with --apply to update the setting.")
        return 0

    # Apply only files already present on the protected branch, not an older checkout.
    for path in (CHECKS_FILE, WORKFLOW):
        endpoint = f"repos/{{owner}}/{{repo}}/contents/{path.as_posix()}?ref={quote(options.branch, safe='')}"
        try:
            remote = json.loads(_gh_api(endpoint, options.repo))
            matches = base64.b64decode(remote["content"]) == (REPO_ROOT / path).read_bytes()
        except (ValueError, KeyError, TypeError, OSError) as error:
            print(f"Cannot verify {path}: {error}", file=sys.stderr)
            return 1
        if not matches:
            print(f"{path} differs from the protected branch; refusing to apply.", file=sys.stderr)
            return 1
    if live_checks(options.repo, options.branch)[0] != payload:
        print("Branch protection changed during this run; refusing to apply.", file=sys.stderr)
        return 1

    body = {"strict": bool(strict), "checks": [{"context": check, "app_id": ACTIONS_APP_ID} for check in declared]}
    endpoint = f"repos/{{owner}}/{{repo}}/branches/{quote(options.branch, safe='')}/protection/required_status_checks"
    _gh_api(endpoint, options.repo, method="PATCH", input_data=body)
    _, after = live_checks(options.repo, options.branch)
    if sorted(after) != sorted(declared):
        print(f"read-back mismatch: live is {sorted(set(after) ^ set(declared))}", file=sys.stderr)
        return 1
    print(f"\nApplied and verified: {len(after)} required contexts on {options.branch}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
