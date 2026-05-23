#!/usr/bin/env python3
# /// pyproject
# [context]
# dependencies = [
#   "pyyaml",
# ]
# ///
"""Create or update GitHub issues from sprint-status.yaml.

Reads story definitions from ``_bmad-output/sprint-status.yaml`` and creates
(or updates) corresponding GitHub issues via the ``gh`` CLI. The script is
**idempotent**: re-running it will update existing issues rather than creating
duplicates, using the story slug embedded in the issue title as a match key.

Usage:
    uv run scripts/create_github_issues.py --yaml PATH [--dry-run]

Options:
    --dry-run   Print the gh commands that would be executed without running them.
    --yaml      Path to the generated sprint-status.yaml.

Expected YAML schema
--------------------

.. code-block:: yaml

    release: "v2.5"
    milestone: "v2.5"
    epics:
      - id: 1
        title: "Index Validation: EDDI & Palmer"
        stories:
          - slug: "e1-eddi-literature"
            title: "EDDI Literature Extraction & Algorithm Spec"
            description: |
              Extract EDDI algorithm from Hobbins et al. (2016) ...
            labels:
              - "epic:validation"
            status: "pending"          # pending | in-progress | done | skipped
            blocked_by: []             # list of slugs this story depends on
          - slug: "e1-palmer-literature"
            ...

Notes:
    - ``gh`` CLI must be installed and authenticated.
    - Issues are matched by title prefix ``[{slug}]``.
    - The script does NOT close issues — use ``gh issue close`` manually or
      let PR auto-close keywords handle it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

STATUS_LABELS = {
    "pending": None,
    "in-progress": "status:in-progress",
    "in-review": "status:in-review",
    "done": None,
    "skipped": None,
    "blocked": "status:blocked",
}

MANAGED_STATUS_LABELS = {label for label in STATUS_LABELS.values() if label}


@dataclass(frozen=True)
class IssuePayload:
    """GitHub issue fields derived from one story."""

    title: str
    body: str
    labels: list[str]
    milestone: str | None


def run_gh(args: list[str], *, dry_run: bool = False) -> str:
    """Run a gh CLI command, returning stdout.

    Args:
        args: Arguments to pass to ``gh``.
        dry_run: If True, print the command instead of executing it.

    Returns:
        The command's stdout, or an empty string in dry-run mode.
    """
    cmd = ["gh", *args]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return ""
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def find_existing_issue(slug: str, *, dry_run: bool = False) -> int | None:
    """Search for an open issue whose title starts with ``[slug]``.

    Args:
        slug: The story slug to search for.
        dry_run: If True, skip the search and return None.

    Returns:
        The issue number if found, otherwise None.
    """
    if dry_run:
        return None
    try:
        out = run_gh(
            [
                "issue",
                "list",
                "--state",
                "open",
                "--search",
                f"[{slug}] in:title",
                "--json",
                "number,title",
                "--limit",
                "50",
            ]
        )
        if not out:
            return None
        issues = json.loads(out)
        for issue in issues:
            if issue["title"].startswith(f"[{slug}]"):
                return issue["number"]
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        pass
    return None


def get_issue_labels(issue_number: int, *, dry_run: bool = False) -> set[str]:
    """Return the current labels on an issue."""
    if dry_run:
        return set()
    try:
        out = run_gh(["issue", "view", str(issue_number), "--json", "labels"])
        labels = json.loads(out).get("labels", [])
        return {label["name"] for label in labels}
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        return set()


def build_issue_body(story: dict, epic: dict, release: str) -> str:
    """Build the issue body markdown from story and epic metadata.

    Args:
        story: Story dictionary from the YAML.
        epic: Parent epic dictionary.
        release: Release version string.

    Returns:
        Markdown string for the issue body.
    """
    lines = [
        f"**Release:** {release}",
        f"**Epic {epic['id']}:** {epic['title']}",
        "",
    ]

    desc = story.get("description", "").strip()
    if desc:
        lines.append("## Description")
        lines.append("")
        lines.append(desc)
        lines.append("")

    blocked_by = story.get("blocked_by", [])
    if blocked_by:
        lines.append("## Dependencies")
        lines.append("")
        for dep in blocked_by:
            lines.append(f"- Blocked by: `{dep}`")
        lines.append("")

    lines.append("---")
    lines.append("_Generated from `sprint-status.yaml` by `scripts/create_github_issues.py`_")
    return "\n".join(lines)


def get_labels(story: dict) -> list[str]:
    """Build the managed label list for a story."""
    labels = list(story.get("labels", []))
    status_label = STATUS_LABELS.get(story.get("status", "pending"))
    if status_label:
        labels.append(status_label)
    return list(dict.fromkeys(labels))


def build_issue_payload(
    story: dict,
    epic: dict,
    release: str,
    labels: list[str],
    milestone: str | None,
) -> IssuePayload:
    """Build the issue fields for one story."""
    return IssuePayload(
        title=f"[{story['slug']}] {story['title']}",
        body=build_issue_body(story, epic, release),
        labels=labels,
        milestone=milestone,
    )


def apply_issue(
    existing: int | None,
    payload: IssuePayload,
    *,
    dry_run: bool = False,
) -> tuple[str, str]:
    """Create or update an issue and return an action plus issue reference."""
    if existing:
        edit_args = ["issue", "edit", str(existing), "--title", payload.title, "--body", payload.body]
        current_labels = get_issue_labels(existing, dry_run=dry_run)
        stale_status_labels = sorted((current_labels & MANAGED_STATUS_LABELS) - set(payload.labels))
        if stale_status_labels:
            edit_args.extend(["--remove-label", ",".join(stale_status_labels)])
        if payload.labels:
            edit_args.extend(["--add-label", ",".join(payload.labels)])
        if payload.milestone:
            edit_args.extend(["--milestone", payload.milestone])
        run_gh(edit_args, dry_run=dry_run)
        return "updated", f"#{existing}"

    create_args = ["issue", "create", "--title", payload.title, "--body", payload.body]
    if payload.labels:
        create_args.extend(["--label", ",".join(payload.labels)])
    if payload.milestone:
        create_args.extend(["--milestone", payload.milestone])
    out = run_gh(create_args, dry_run=dry_run)
    return "created", out or "[dry-run]"


def process_stories(config: dict, *, dry_run: bool = False) -> None:
    """Create or update GitHub issues for all stories in the config.

    Args:
        config: Parsed sprint-status.yaml content.
        dry_run: If True, print commands without executing.
    """
    release = config.get("release", "v2.5")
    milestone = config.get("milestone")

    created = 0
    updated = 0
    skipped = 0

    for epic in config.get("epics", []):
        for story in epic.get("stories", []):
            slug = story["slug"]
            payload = build_issue_payload(story, epic, release, get_labels(story), milestone)
            existing = find_existing_issue(slug, dry_run=dry_run)
            action, issue_ref = apply_issue(existing, payload, dry_run=dry_run)
            if action == "updated":
                print(f"  updated {issue_ref}: {payload.title}")
                updated += 1
            else:
                print(f"  created {issue_ref}: {payload.title}")
                created += 1

    print(f"\nDone: {created} created, {updated} updated, {skipped} skipped")


def main() -> None:
    """Entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Create/update GitHub issues from sprint-status.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print gh commands without executing them",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        required=True,
        help="Path to generated sprint-status.yaml.",
    )
    args = parser.parse_args()

    if not args.yaml.exists():
        print(f"Error: {args.yaml} not found", file=sys.stderr)
        print("Run BMAD sprint planning first to generate this file.", file=sys.stderr)
        sys.exit(1)

    with args.yaml.open() as f:
        config = yaml.safe_load(f)

    if not config or "epics" not in config:
        print(f"Error: {args.yaml} has no 'epics' key", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {args.yaml} for release {config.get('release', '?')}...")
    process_stories(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
