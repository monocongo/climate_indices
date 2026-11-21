"""Consistency checks between the comparator lock file and the Linux smoke harness."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PAPERS = ROOT / "docs" / "research" / "reference-papers"
SMOKE_DOCKERFILE = REFERENCE_PAPERS / "containers" / "linux-smoke" / "Dockerfile"
COMPARATORS_LOCK = REFERENCE_PAPERS / "comparators.lock.yaml"

SHA256 = re.compile(r"\b[0-9a-f]{64}\b")


def test_linux_smoke_checksums_come_from_the_lock_file() -> None:
    """The Dockerfile hard-codes the checksums, so a lock update must reach it."""
    dockerfile = SMOKE_DOCKERFILE.read_text(encoding="utf-8")
    lock = COMPARATORS_LOCK.read_text(encoding="utf-8")
    pinned = set(SHA256.findall(dockerfile))
    recorded = set(SHA256.findall(lock))

    assert pinned, "the Linux smoke Dockerfile no longer checks any artifact sha256"
    assert pinned <= recorded, (
        "sha256 values in the Linux smoke Dockerfile are missing from comparators.lock.yaml: "
        f"{sorted(pinned - recorded)}"
    )
