"""Find a minimal Z-Index sequence that reproduces each divergence class (REFIMPL-29, #1631).

For each attribution label in ``palmer-recursion-divergence-results.json`` a window of a real
division's Z series is shrunk by greedy deletion until no single month can be removed without
losing the class (locally minimal, not proven shortest), then its values are rounded for
readability. A series *reproduces* a label when the pdi.f and Wells recursions disagree on it,
and every baseline episode, labelled over its own span exactly as the study labels episodes,
carries that label with no variant disagreement outside the baseline episodes.
Series are left-padded with zeros to whole years, which both recursions treat as no-ops.

Run: ``uv run python scripts/find_palmer_minimal_sequences.py``; writes
``palmer-recursion-divergence-sequences.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import characterize_palmer_recursion_divergence as c  # noqa: E402

RESULTS = c.ROOT / "docs" / "research" / "reference-papers" / "palmer-recursion-divergence-results.json"
OUT = c.ROOT / "docs" / "research" / "reference-papers" / "palmer-recursion-divergence-sequences.json"

_VARIANTS: dict[tuple[str, ...], object] = {}


def _variant(mechs: tuple[str, ...]):  # noqa: ANN202
    if mechs not in _VARIANTS:
        _VARIANTS[mechs] = c.build_pdi_variant(mechs)
    return _VARIANTS[mechs]


def pad(active: list[float]) -> np.ndarray:
    n = -(-len(active) // 12) * 12
    return np.concatenate([np.zeros(n - len(active)), np.asarray(active, dtype=float)])


def run_variant(z: np.ndarray, mechs: tuple[str, ...]) -> np.ndarray:
    pdi_f, _ = c.default_factors()
    module = _variant(mechs)
    factors = module.PdiDurationFactors(pdi_f.wetm, pdi_f.wetb, pdi_f.drym, pdi_f.dryb)
    return module.calculate(z.reshape(-1, 12, 1), factors).pdsi.reshape(-1)


def label_of(active: list[float]) -> str | None:
    """The series' single label, or None if it has no disagreement.

    Each baseline episode is labelled over its own span, as ``attribute()`` does in the study. A
    series whose episodes carry different labels, or whose harmonized variants disagree with Wells
    outside a baseline episode, is ``"mixed"``.
    """
    z = pad(active)
    _, wells_f = c.default_factors()
    wells = c._palmer_wells.calculate(z, factors=wells_f).pdsi

    def mask(mechs: tuple[str, ...]) -> np.ndarray:
        return np.abs(run_variant(z, mechs) - wells) > c.DISAGREE

    base = mask(())
    if not base.any():
        return None
    spans = c.episodes(base)
    full = mask(c.MECH_NAMES)
    loo = {m: mask(tuple(x for x in c.MECH_NAMES if x != m)) for m in c.MECH_NAMES}
    inside = np.zeros_like(base)
    for s, e in spans:
        inside[s:e] = True
    if (full & ~inside).any() or any((v & ~inside).any() for v in loo.values()):
        return "mixed"
    singles = {m: mask((m,)) for m in c.MECH_NAMES}
    labels = {c.episode_label(full, loo, singles, s, e) for s, e in spans}
    return labels.pop() if len(labels) == 1 else "mixed"


def shrink(active: list[float], target: str) -> list[float]:
    """Chunked greedy deletion, then single-month deletion to a fixed point."""
    chunk = max(len(active) // 2, 1)
    while True:
        changed = False
        i = 0
        while i < len(active):
            trial = active[:i] + active[i + chunk :]
            if trial and label_of(trial) == target:
                active, changed = trial, True
            else:
                i += chunk
        if chunk == 1 and not changed:
            return active
        if chunk > 1:
            chunk = max(chunk // 2, 1)


def simplify(active: list[float], target: str) -> list[float]:
    """Round each value as coarsely as the label allows (1 decimal, then 2)."""
    for digits in (0, 1, 2):
        for i in range(len(active)):
            trial = list(active)
            trial[i] = round(trial[i], digits)
            if trial[i] != active[i] and label_of(trial) == target:
                active = trial
    return active


def seed_window(division: str, start: int, stop: int, target: str) -> list[float] | None:
    """The smallest left extension of a real episode that reproduces ``target`` on its own."""
    z = c.load_z(division)
    for tail in (6, 12):
        end = len(z) if target == "F" else min(stop + tail, len(z))
        for lead in range(6, start + 1, 6):
            active = [float(v) for v in z[start - lead : end]]
            if label_of(active) == target:
                return active
    return None


def main() -> None:
    results_in = json.loads(RESULTS.read_text())
    examples = results_in["example_episodes_by_label"]
    _, wells_f = c.default_factors()
    results: dict[str, dict] = {}
    for target, episodes_ in examples.items():
        if target == "residual" or target.startswith("overdetermined"):
            continue
        found = None
        for _, division, s, e in episodes_:
            seed = seed_window(division, s, e, target)
            if seed is not None:
                found = (division, s, e, seed)
                break
        if found is None:
            results[target] = {"status": "no example episode reproduces the label on its own"}
            print(target, "no seed", flush=True)
            continue
        division, s, e, seed = found
        active = simplify(shrink(seed, target), target)
        z = pad(active)
        wells = c._palmer_wells.calculate(z, factors=wells_f).pdsi
        pdi = run_variant(z, ())
        first = int(np.flatnonzero(np.abs(pdi - wells) > c.DISAGREE)[0])
        results[target] = {
            "seed": {"division": division, "episode_start": s, "episode_stop": e, "seed_months": len(seed)},
            "active_months": len(active),
            "z": active,
            "padded_months": int(z.size),
            "first_differing_index": first,
            "pdi_pdsi": [float(v) for v in pdi],
            "wells_pdsi": [float(v) for v in wells],
        }
        print(target, len(seed), "->", len(active), flush=True)
    OUT.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
