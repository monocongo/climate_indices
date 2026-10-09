"""The consolidated cross-backend parity suite, driven by ``tests/parity_registry.py``.

Every ported kernel is registered once, with the Python entry point that reaches
it, the extension module its dispatch lives behind, and the input family its
values come from. These tests then cover, for the whole registry at once:

- value parity at ``rtol = atol = 1e-10`` with matching NaN positions, on a fixed
  sample per family;
- the same parity on Hypothesis draws that vary series length, NaN patterns,
  zero runs, extreme magnitudes, and the number of spatial cells;
- the documented dispatch decision for the input kinds that stay on one path or
  the other;
- that the registry covers every kernel the extension exposes, and that
  ``docs/architecture.md`` publishes a tolerance row for every entry.

The per-family modules (``test_native_parity.py`` and its peers) keep the
targeted cases: fixtures against published references, error and warning
parity, and the boundary behavior of individual kernels.

Skipped when the extension is not built, unless ``CLIMATE_INDICES_REQUIRE_NATIVE=1``
is set, as in CI's native legs, where a missing extension is a collection error.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings

from tests import conftest, parity_registry
from tests.parity_registry import ENTRIES, ROUTING, Entry, RoutingCase

native = conftest.import_native()

_ARCHITECTURE = Path(__file__).resolve().parents[1] / "docs" / "architecture.md"

_ENTRY_IDS = [entry.name for entry in ENTRIES]
_ROUTING_IDS = [case.name for case in ROUTING]


@pytest.mark.parametrize("entry", ENTRIES, ids=_ENTRY_IDS)
def test_entry_parity(monkeypatch: pytest.MonkeyPatch, entry: Entry) -> None:
    """Each registered entry reaches its kernels and matches Python on the family sample."""
    parity_registry.run_parity(monkeypatch, entry)


def test_spatial_block_entries_return_a_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Spatial Block entry keeps its cells, so it runs the spatial path rather than a flattened series."""
    block = parity_registry.SAMPLES[parity_registry.BLOCK]
    for entry in (entry for entry in ENTRIES if entry.family == parity_registry.BLOCK):
        assert parity_registry.run_parity(monkeypatch, entry).rust.shape == block.shape, entry.name


@pytest.mark.parametrize("entry", ENTRIES, ids=_ENTRY_IDS)
def test_entry_parity_under_property_draws(monkeypatch: pytest.MonkeyPatch, entry: Entry) -> None:
    """Each entry holds parity across drawn lengths, NaN patterns, zeros, and extremes."""

    @hypothesis_settings(max_examples=15, deadline=None)
    @given(values=parity_registry.STRATEGIES[entry.family])
    def check(values) -> None:
        parity_registry.run_parity(monkeypatch, entry, values)

    check()


@pytest.mark.parametrize("case", ROUTING, ids=_ROUTING_IDS)
def test_routing_is_documented(monkeypatch: pytest.MonkeyPatch, case: RoutingCase) -> None:
    """A documented dispatch decision holds: the named kernels run, the others never do."""
    calls = parity_registry.routing_calls(monkeypatch, case)
    assert case.expected <= calls, f"{case.name}: {sorted(case.expected - calls)} did not run"
    assert not (case.forbidden & calls), f"{case.name}: {sorted(case.forbidden & calls)} ran anyway"


def test_registry_covers_every_kernel() -> None:
    """The registry is the whole kernel surface: every function the extension exposes has an entry."""
    exposed = frozenset(name for name in dir(native) if not name.startswith("_")) - parity_registry.NON_KERNELS
    assert exposed == parity_registry.REGISTERED_KERNELS


def test_tolerance_table_publishes_every_entry() -> None:
    """The published tolerance table names every registered entry, so the registry cannot drift from it."""
    table = _ARCHITECTURE.read_text(encoding="utf-8").split("### Parity tolerance")[-1].split("\n## ")[0]
    rows = {match.group(1) for match in re.finditer(r"^\| `?([a-z0-9_]+)`? \|", table, re.MULTILINE)}
    assert set(_ENTRY_IDS) <= rows, f"missing tolerance rows: {sorted(set(_ENTRY_IDS) - rows)}"


def test_routing_table_publishes_every_case() -> None:
    """The published dispatch table names every asserted routing case."""
    table = _ARCHITECTURE.read_text(encoding="utf-8").split("### Dispatch routing")[-1].split("\n### ")[0]
    rows = {match.group(1) for match in re.finditer(r"^\| `?([a-z0-9_]+)`? \|", table, re.MULTILINE)}
    assert set(_ROUTING_IDS) <= rows, f"missing routing rows: {sorted(set(_ROUTING_IDS) - rows)}"
