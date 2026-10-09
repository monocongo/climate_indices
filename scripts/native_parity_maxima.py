"""Measure cross-backend parity per registered kernel and print the tolerance table.

The table is the evidence for the parity section of ``docs/architecture.md``: one
row per entry in ``tests/parity_registry.py``, with the largest absolute and
relative deviation measured on this platform. CI's native leg appends the table to
its job summary, so the same measurement is recorded on Linux x86-64 as well as on
a developer's macOS arm64 machine.

Usage: ``uv run python scripts/native_parity_maxima.py`` with the extension built
(``uv run maturin develop --release``).
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import parity_registry  # noqa: E402 - the repository root has to be importable first


def measured_platform() -> str:
    """The platform the maxima below were measured on."""
    machine = platform.machine()
    system = platform.system()
    return f"{system} {machine} ({platform.python_implementation()} {platform.python_version()})"


def main() -> int:
    monkeypatch = pytest.MonkeyPatch()
    try:
        runs = parity_registry.measure(monkeypatch)
    finally:
        monkeypatch.undo()

    print(f"Parity maxima measured on {measured_platform()}.")
    print()
    print("| Entry | Kernels | Max absolute | Max relative |")
    print("| --- | --- | --- | --- |")
    for run in sorted(runs, key=lambda item: item.entry.name):
        kernels = ", ".join(f"`{name}`" for name in sorted(run.calls))
        print(f"| `{run.entry.name}` | {kernels} | {run.max_absolute:.3e} | {run.max_relative:.3e} |")
    print()
    worst = max(runs, key=lambda item: item.max_absolute)
    print(
        f"The largest absolute deviation is {worst.max_absolute:.3e} ({worst.entry.name}); "
        f"every entry is compared at rtol = atol = {parity_registry.RTOL:.0e} with matching NaN positions."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
