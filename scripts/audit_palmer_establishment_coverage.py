"""Count matched spell-establishment decisions and replay the E intervention in #1631."""

import json
from collections import Counter, defaultdict
from hashlib import sha256
from typing import Any

import characterize_palmer_recursion_divergence as study
import numpy as np

from climate_indices.drought import _palmer_wells
from climate_indices.drought._palmer_duration import DurationFactors


def matched_months(
    z: np.ndarray,
    pdi_trace: list[dict[str, Any]],
    wells: _palmer_wells.WellsResult,
    factors: DurationFactors,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return eligible month indices, signed dominant candidates, and standard X3."""
    pdi_states = np.array([row["post"][:3] for row in pdi_trace])
    wells_states = np.column_stack((wells.x1, wells.x2, wells.x3))
    pdi_previous = np.vstack((np.zeros(3), pdi_states[:-1]))
    wells_previous = np.vstack((np.zeros(3), wells_states[:-1]))
    no_spell = (pdi_previous[:, 2] == 0.0) & (wells_previous[:, 2] == 0.0)  # NOSONAR
    same_candidates = (np.abs(pdi_previous[:, :2] - wells_previous[:, :2]) <= study.DISAGREE).all(axis=1)

    wet = np.maximum(0.0, factors.wetc * wells_previous[:, 0] + z / factors.wet_denominator)
    dry = np.minimum(0.0, factors.dryc * wells_previous[:, 1] + z / factors.dry_denominator)
    dominant = np.where(wet > -dry, wet, dry)
    smaller = np.minimum(wet, -dry)
    magnitude = np.abs(dominant)
    separated = (smaller < 0.5 - study.DISAGREE) & (magnitude - smaller > study.DISAGREE)
    off_boundary = (np.abs(magnitude - 0.5) > study.DISAGREE) & (np.abs(magnitude - 1.0) > study.DISAGREE)
    return np.flatnonzero(no_spell & same_candidates & separated & off_boundary), dominant, pdi_states[:, 2]


def stable_examples(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Take two hash-ranked cases with distinct state codes."""
    ranked = sorted(cases, key=lambda row: sha256(f"{row['division']}:{row['month']}".encode()).digest())
    states: dict[str, dict[str, Any]] = {}
    for row in ranked:
        states.setdefault(row["division"][:2], row)
    return list(states.values())[:2]


def main() -> None:
    study.verify_pinned_sources()
    pdi_factors, wells_factors = study.default_factors()
    variant = study.build_pdi_variant(("E",))
    variant_factors = variant.PdiDurationFactors(pdi_factors.wetm, pdi_factors.wetb, pdi_factors.drym, pdi_factors.dryb)
    output_counts: Counter[str] = Counter()
    onset_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    split_cases: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for division in study.divisions():
        z = study.load_z(division)
        standard, trace = study.trace_pdi(z, pdi_factors)
        wells = _palmer_wells.calculate(z, factors=wells_factors)
        threshold_only = variant.calculate(z.reshape(-1, 12, 1), variant_factors).pdsi.reshape(-1)
        standard_pdsi = standard.pdsi.reshape(-1)
        base = np.abs(standard_pdsi - wells.pdsi) > study.DISAGREE
        changed = np.abs(threshold_only - wells.pdsi) > study.DISAGREE
        episodes = study.episodes(base)
        output_counts.update(
            divisions=1,
            months=z.size,
            baseline_months=int(base.sum()),
            remaining_months=int(changed.sum()),
            cleared_months=int((base & ~changed).sum()),
            new_months=int((changed & ~base).sum()),
            baseline_episodes=len(episodes),
            cleared_episodes=sum(1 for start, stop in episodes if not changed[start:stop].any()),
            partly_cleared_episodes=sum(
                1 for start, stop in episodes if changed[start:stop].any() and not changed[start:stop].all()
            ),
        )

        indices, candidates, pdi_x3 = matched_months(z, trace, wells, wells_factors)
        for index in indices:
            candidate = float(candidates[index])
            magnitude = abs(candidate)
            pdi_expected = candidate if magnitude >= 1.0 else 0.0
            wells_expected = candidate if magnitude >= 0.5 else 0.0
            matches = abs(pdi_x3[index] - pdi_expected) <= study.DISAGREE and (
                abs(wells.x3[index] - wells_expected) <= study.DISAGREE
            )
            year_offset, month_offset = divmod(int(index), 12)
            year = study.START_YEAR + year_offset
            era = "1895–1936" if year < 1937 else "1937–1979" if year < 1980 else "1980–2022"
            sign = "wet" if candidate > 0.0 else "dry"
            band = "between" if 0.5 < magnitude < 1.0 else "below" if magnitude < 0.5 else "above"
            counts = onset_counts[era, sign, band]
            counts.update(months=1, matches=int(matches), pdsi_disagreements=int(base[index]))
            if band == "between":
                split_cases[era, sign].append(
                    {
                        "division": division,
                        "month": f"{year}-{month_offset + 1:02d}",
                        "z": float(z[index]),
                        "candidate": candidate,
                        "standard_x3": float(pdi_x3[index]),
                        "wells_x3": float(wells.x3[index]),
                        "standard_pdsi": float(standard_pdsi[index]),
                        "wells_pdsi": float(wells.pdsi[index]),
                    }
                )

    baseline_path = study.ROOT / "docs/research/reference-papers/palmer-recursion-divergence-results.json"
    reference = json.loads(baseline_path.read_text())["attribution"]
    expected = {
        "baseline_months": reference["remaining_disagreeing_months"]["none"],
        "remaining_months": reference["remaining_disagreeing_months"]["only E"],
        "baseline_episodes": sum(reference["episodes_by_label"].values()),
    }
    if any(output_counts[name] != count for name, count in expected.items()):
        raise RuntimeError("The replay differs from the committed #1631 totals.")
    print(
        json.dumps(
            {
                "output_counts": output_counts,
                "onset_counts": [
                    {"era": era, "sign": sign, "band": band, **counts}
                    for (era, sign, band), counts in sorted(onset_counts.items())
                ],
                "between_thresholds_divisions": len(
                    {row["division"] for cases in split_cases.values() for row in cases}
                ),
                "examples": [
                    {"era": era, "sign": sign, "cases": stable_examples(cases)}
                    for (era, sign), cases in sorted(split_cases.items())
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
