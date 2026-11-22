"""Consistency checks between the comparator lock file and the Linux smoke harness."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PAPERS = ROOT / "docs" / "research" / "reference-papers"
SMOKE_DOCKERFILE = REFERENCE_PAPERS / "containers" / "linux-smoke" / "Dockerfile"
COMPARATORS_LOCK = REFERENCE_PAPERS / "comparators.lock.yaml"

# `fetch URL SHA256 FILE` and `cran FILE SHA256`. Each download is named after the lock file's URL
# basename, so the name binds one Dockerfile checksum to one locked artifact.
FETCH = re.compile(r"\bfetch\s+(\S+)\s+([0-9a-f]{64})\s+(\S+)")
CRAN = re.compile(r"\bcran\s+(\S+)\s+([0-9a-f]{64})")


def _locked_artifacts() -> dict[str, str]:
    """Map URL basename to sha256 for every lock entry that records both."""
    locked: dict[str, str] = {}
    url = ""
    for line in COMPARATORS_LOCK.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- name:"):
            url = ""
        elif stripped.startswith("url:"):
            url = stripped.removeprefix("url:").strip()
        elif stripped.startswith("sha256:") and url:
            locked[url.rsplit("/", 1)[-1]] = stripped.removeprefix("sha256:").strip()
            url = ""
    return locked


def _dockerfile_text() -> str:
    """Read the Dockerfile with its continuations joined, as the Dockerfile parser does."""
    return SMOKE_DOCKERFILE.read_text(encoding="utf-8").replace("\\\n", " ")


def _dockerfile_artifacts() -> dict[str, str]:
    """Map each downloaded artifact name to the sha256 the Dockerfile verifies it with."""
    text = _dockerfile_text()
    pinned = {name.rstrip(";"): digest for _, digest, name in FETCH.findall(text)}
    pinned.update(CRAN.findall(text))
    return pinned


def test_linux_smoke_checksums_are_bound_to_the_lock_file() -> None:
    """Each download carries its own lock sha256, and every locked tarball is downloaded."""
    locked = _locked_artifacts()
    pinned = _dockerfile_artifacts()

    helpers = [m.group(1) for m in re.finditer(r"(?:fetch|cran)\(\) \{([^}]*)\}", _dockerfile_text())]
    assert len(helpers) == 2, f"expected the fetch() and cran() helpers, found {len(helpers)}"
    assert all("fetch.sh" in body for body in helpers), "a download helper bypasses fetch.sh"

    assert len(locked) == 9, f"expected nine locked PyPI and CRAN artifacts, found {sorted(locked)}"
    assert pinned, "the Linux smoke Dockerfile no longer checks any artifact sha256"
    unpinned = sorted(locked.keys() - pinned.keys())
    unlocked = sorted(pinned.keys() - locked.keys())
    mismatched = sorted(k for k in pinned.keys() & locked.keys() if pinned[k] != locked[k])
    assert pinned == locked, (
        "Dockerfile and lock disagree on the verified artifacts: "
        f"not downloaded={unpinned}, not in the lock={unlocked}, hash mismatch={mismatched}"
    )
