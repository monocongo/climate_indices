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
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = Path(".github/workflows/unit-tests-workflow.yml")
CHECKS_FILE = Path(".github/required-status-checks.txt")

# The GitHub Actions app, which owns every check in this workflow.
ACTIONS_APP_ID = 15368


def declared_checks(path: Path | None = None) -> list[str]:
    """The reviewed required contexts, in file order, with comments and blanks dropped."""
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
    pythons = re.findall(r"^\s+python-version:\s*\[([^]]+)]", body, re.MULTILINE)
    systems = re.findall(r"^\s+os:\s*\[([^]]+)]", body, re.MULTILINE)
    if not pythons and not systems:
        return {job}
    if len(pythons) != 1 or len(systems) != 1:
        raise ValueError(f"job {job!r} has {len(pythons)} python-version lists and {len(systems)} os lists")
    legs = {
        (python, system)
        for python in re.findall(r"\d+\.\d+", pythons[0])
        for system in re.findall(r"[\w.-]+", systems[0])
    }
    # an `include:` entry adds a leg the cross product does not produce
    legs |= set(re.findall(r"- python-version:\s*['\"](\d+\.\d+)['\"]\s+os:\s*([\w.-]+)", body))
    return {f"{job} ({python}, {system})" for python, system in legs}


def workflow_contexts(path: Path | None = None) -> set[str]:
    """Every context the workflow reports for the jobs the reviewed list names."""
    text = (path or (REPO_ROOT / WORKFLOW)).read_text(encoding="utf-8")
    contexts: set[str] = set()
    for check in declared_checks():
        job = check.split(" (", 1)[0]
        contexts |= _job_contexts(job, _job_text(text, job))
    return contexts


def _gh_api(
    path: str, repo: str | None, *, method: str | None = None, input_data: dict[str, object] | None = None
) -> str:
    """Call `gh api` and return stdout. Without ``repo`` gh fills the {owner}/{repo} placeholders itself."""
    endpoint = path if repo is None else path.replace("{owner}/{repo}", repo, 1)
    command = ["gh", "api"]
    if method:
        command += ["-X", method]
    if input_data is not None:
        command += ["--input", "-"]
    command.append(endpoint)
    result = subprocess.run(
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
    """The live required_status_checks object, and its context names."""
    path = f"repos/{{owner}}/{{repo}}/branches/{branch}/protection/required_status_checks"
    try:
        payload = json.loads(_gh_api(path, repo))
    except json.JSONDecodeError as error:
        raise SystemExit(f"unexpected response for {path}: {error}") from error
    return payload, list(payload["contexts"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write the reviewed list to branch protection")
    parser.add_argument("--branch", default="main", help="protected branch to update (default: main)")
    parser.add_argument("--repo", default=None, help="owner/name; gh infers it from the checkout otherwise")
    options = parser.parse_args(argv)

    declared = declared_checks()
    derived = workflow_contexts()

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

    if "test-full" in {check.split(" (", 1)[0] for check in declared}:
        print("test-full skips on pull requests and would block every one; refusing to apply.", file=sys.stderr)
        return 1

    body = {"strict": bool(strict), "checks": [{"context": check, "app_id": ACTIONS_APP_ID} for check in declared]}
    endpoint = f"repos/{{owner}}/{{repo}}/branches/{options.branch}/protection/required_status_checks"
    _gh_api(endpoint, options.repo, method="PATCH", input_data=body)
    _, after = live_checks(options.repo, options.branch)
    if sorted(after) != sorted(declared):
        print(f"read-back mismatch: live is {sorted(set(after) ^ set(declared))}", file=sys.stderr)
        return 1
    print(f"\nApplied and verified: {len(after)} required contexts on {options.branch}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
