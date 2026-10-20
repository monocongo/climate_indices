"""Reproduce the compound-bootstrap synthetic characterization for issue #1398.

Predeclared configuration, gates, scenario groups, and decision rules are recorded in
`docs/design/compound-bootstrap-calibration.md`, which was committed before this run.
The plan extends the #1371 design (PR #1397): the four original scenarios are
replicated with identical seeds and gates, and further groups measure record length,
persistence between 6 and 36 months, lagged conditioning, hazard frequency tails,
dependent hazards at short persistence, and the default 1,999 resamples.

Gates. 128 independent records per scenario. A coverage case passes when at least 109
records cover the truth; an independent-null case passes when at most 19 of 128 records
reject at nominal 0.05. Both bands are the exact binomial with the per-comparison tail
0.01 / 864 that #1371 predeclared, so no case is judged by a wider band. Every missing
interval counts as uncovered and every missing p-value fails the null gate, except that
a structural guard (`insufficient_years`, `insufficient_blocks`, or
`sparse_occurrences`) removes that record from the band instead. A guard can fire on some
records and not others, so each month applies the same exact-binomial tail to the records
that carry an inference and reports the excluded count. With nothing excluded the band is
109 and 19 of 128, which is the predeclared case. Wilson 95% intervals accompany every
measured proportion. Correlated scenarios test interval coverage only; their rejection
rate describes power, not null calibration. Do not select a block length by its p-value.

Truths. Analytic bivariate-normal quadrature supplies each month's truth for lag window
(0, 0). Lagged conditioning has no closed form under the moving-average model, so its
truth is a 40,000-year direct NumPy simulation, separate from the library code.

Reproduction is seeded: a scenario's resample streams derive from the seed, its
predeclared stream index, and the trial number. Runtime is serial, about three minutes
for `--group replication` and about twenty minutes for every group.

Run: uv run scripts/characterize_compound_bootstrap.py --output docs/design/<report>.json
The output must be a JSON file below this repository's docs/design directory. Repeat
`--group` to run a subset of the plan.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy
from scipy.integrate import quad
from scipy.stats import binom, binomtest, norm

from climate_indices import compound
from climate_indices.compound._bootstrap import Statistic
from climate_indices.exceptions import ClimateIndicesWarning

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "docs" / "design"
TRIALS = 128
YEARS = 200
RESAMPLES = 399
SEED = 1371
LENGTHS: tuple[int | None, ...] = (3, 6, 12)
STATISTICS: tuple[Statistic, ...] = ("joint_probability", "likelihood_multiplication_factor", "conditional_probability")
CONDITIONAL: tuple[Statistic, ...] = ("conditional_probability",)
SCENARIOS = (
    ("seasonal_independent", 1, 0.0),
    ("overlapping_6_month_independent", 6, 0.0),
    ("persistent_36_month_independent", 36, 0.0),
    ("persistent_36_month_correlated", 36, 0.5),
)
PROFILES = {"seasonal": (0.45, 0.1), "low": (0.25, 0.05), "high": (0.75, 0.05), "wide": (0.45, 0.15)}
MONTE_CARLO_YEARS = 40_000
_TIE_ALPHA = 0.05
_FAMILY_WISE_ALPHA = 0.01
_STRUCTURAL_GUARDS = ("insufficient_years", "insufficient_blocks", "sparse_occurrences")


@dataclass(frozen=True)
class Scenario:
    """One predeclared measurement: source model, record, and inference choices."""

    name: str
    group: str
    stream: int
    moving_average_months: int
    rho: float
    profile: str
    years: int
    resamples: int
    lag_window: tuple[int, int]
    statistics: tuple[Statistic, ...]
    lengths: tuple[int | None, ...]


def _plan() -> tuple[Scenario, ...]:
    """Return the predeclared plan, replication first and then the extension groups."""

    def extension(
        name: str,
        group: str,
        stream: int,
        width: int,
        rho: float = 0.0,
        profile: str = "seasonal",
        years: int = YEARS,
        resamples: int = RESAMPLES,
        lag_window: tuple[int, int] = (0, 0),
        statistics: tuple[Statistic, ...] = STATISTICS,
        lengths: tuple[int | None, ...] = (6,),
    ) -> Scenario:
        return Scenario(
            name,
            group,
            stream,
            width,
            rho,
            profile,
            years,
            resamples,
            lag_window,
            statistics,
            lengths,
        )

    replication = tuple(
        Scenario(name, "replication", stream, width, rho, "seasonal", YEARS, RESAMPLES, (0, 0), STATISTICS, LENGTHS)
        for stream, (name, width, rho) in enumerate(SCENARIOS)
    )
    return replication + (
        extension("record_length_30_years_independent", "record_length", 4, 1, years=30, lengths=(None, 6, 3)),
        extension("record_length_60_years_independent", "record_length", 5, 1, years=60, lengths=(None, 6)),
        extension("record_length_100_years_independent", "record_length", 6, 1, years=100, lengths=(None, 6)),
        extension("record_length_30_years_6_month_overlap", "record_length", 7, 6, years=30, lengths=(None, 6, 3)),
        extension("record_length_60_years_6_month_overlap", "record_length", 8, 6, years=60, lengths=(None, 6)),
        extension("record_length_100_years_6_month_overlap", "record_length", 9, 6, years=100, lengths=(None, 6)),
        extension("persistence_12_month_independent", "persistence", 10, 12),
        extension("persistence_18_month_independent", "persistence", 11, 18),
        extension("persistence_24_month_independent", "persistence", 12, 24),
        extension("persistence_30_month_independent", "persistence", 13, 30),
        extension(
            "lag_0_1_independent",
            "lagged_conditioning",
            14,
            1,
            lag_window=(0, 1),
            statistics=CONDITIONAL,
            lengths=LENGTHS,
        ),
        extension(
            "lag_0_3_independent",
            "lagged_conditioning",
            15,
            1,
            lag_window=(0, 3),
            statistics=CONDITIONAL,
            lengths=LENGTHS,
        ),
        extension(
            "lag_1_3_independent",
            "lagged_conditioning",
            16,
            1,
            lag_window=(1, 3),
            statistics=CONDITIONAL,
            lengths=LENGTHS,
        ),
        extension(
            "lag_0_1_6_month_overlap",
            "lagged_conditioning",
            17,
            6,
            lag_window=(0, 1),
            statistics=CONDITIONAL,
            lengths=LENGTHS,
        ),
        extension(
            "lag_0_3_6_month_overlap",
            "lagged_conditioning",
            18,
            6,
            lag_window=(0, 3),
            statistics=CONDITIONAL,
            lengths=LENGTHS,
        ),
        extension(
            "lag_1_3_6_month_overlap",
            "lagged_conditioning",
            19,
            6,
            lag_window=(1, 3),
            statistics=CONDITIONAL,
            lengths=LENGTHS,
        ),
        extension("frequency_low_independent", "hazard_frequency", 20, 1, profile="low"),
        extension("frequency_high_independent", "hazard_frequency", 21, 1, profile="high"),
        extension("frequency_wide_independent", "hazard_frequency", 22, 1, profile="wide"),
        extension("frequency_low_6_month_overlap", "hazard_frequency", 23, 6, profile="low"),
        extension("frequency_high_6_month_overlap", "hazard_frequency", 24, 6, profile="high"),
        extension("frequency_wide_6_month_overlap", "hazard_frequency", 25, 6, profile="wide"),
        extension("frequency_low_36_month_persistence", "hazard_frequency", 26, 36, profile="low"),
        extension("frequency_high_36_month_persistence", "hazard_frequency", 27, 36, profile="high"),
        extension(
            "frequency_low_30_years_independent", "hazard_frequency", 28, 1, profile="low", years=30, lengths=(None, 6)
        ),
        extension("correlated_hazards_1_month", "dependent_hazards", 29, 1, rho=0.5),
        extension("correlated_hazards_6_month", "dependent_hazards", 30, 6, rho=0.5),
        extension("correlated_hazards_12_month", "dependent_hazards", 31, 12, rho=0.5),
        *(
            extension(
                f"default_resamples_{name}",
                "default_resamples",
                32 + stream,
                width,
                rho,
                resamples=1999,
                lengths=LENGTHS,
            )
            for stream, (name, width, rho) in enumerate(SCENARIOS)
        ),
    )


@dataclass(frozen=True)
class Case:
    """Per-month tallies of one statistic and block-length case."""

    covered: npt.NDArray[np.int64]
    rejected: npt.NDArray[np.int64]
    intervals: npt.NDArray[np.int64]
    p_values: npt.NDArray[np.int64]
    structural: npt.NDArray[np.int64]
    few_blocks: npt.NDArray[np.int64]
    length: int
    decisions: npt.NDArray[np.bool_]


def _output_path(value: str | Path) -> Path:
    """Reject output paths that escape the report directory, including symlinks."""
    output = Path(value).resolve()
    if not output.is_relative_to(OUTPUT_ROOT) or output.suffix != ".json":
        raise argparse.ArgumentTypeError(f"Output must be a JSON file below {OUTPUT_ROOT}.")
    return output


def _write_report(value: Path, report: dict[str, Any]) -> None:
    # Recheck the path after characterization because directory symlinks can change during the calculation.
    output = _output_path(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _probabilities(profile: str) -> npt.NDArray[np.float64]:
    """Return the twelve monthly exceedance probabilities of a predeclared profile."""
    center, amplitude = PROFILES[profile]
    return center + amplitude * np.cos(2 * np.pi * np.arange(12) / 12)


def _time_axis(years: int) -> npt.NDArray[np.datetime64]:
    """Return the monthly coordinate; every scenario starts in January 1800."""
    return (np.datetime64("1800-01", "M") + np.arange(years * 12)).astype("datetime64[D]")


def _analytic_truths(scenario: Scenario) -> dict[Statistic, npt.NDArray[np.float64]]:
    """Return the bivariate-normal quadrature truth for lag window (0, 0)."""
    probabilities = _probabilities(scenario.profile)
    thresholds = norm.ppf(probabilities)
    rho = scenario.rho
    joint = np.array(
        [
            quad(
                lambda x, z=z, rho=rho: norm.pdf(x) * norm.cdf((z - rho * x) / math.sqrt(1 - rho * rho)),
                -np.inf,
                z,
                epsabs=1e-12,
                epsrel=1e-12,
            )[0]
            for z in thresholds
        ]
    )
    candidates: dict[Statistic, npt.NDArray[np.float64]] = {
        "joint_probability": joint,
        "likelihood_multiplication_factor": joint / probabilities**2,
        "conditional_probability": joint / probabilities,
    }
    return {statistic: candidates[statistic] for statistic in scenario.statistics}


def _lagged_truth(scenario: Scenario) -> npt.NDArray[np.float64]:
    """Simulate the anchor-window conditional truth without any library code path."""
    if scenario.statistics != CONDITIONAL:
        raise ValueError(f"{scenario.name} measures a lag window, so only the conditional statistic is defined.")
    width = scenario.moving_average_months
    low, high = scenario.lag_window
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 999_999, scenario.stream]))
    noise = rng.standard_normal((2, MONTE_CARLO_YEARS * 12 + width - 1))
    noise[1] = scenario.rho * noise[0] + math.sqrt(1 - scenario.rho**2) * noise[1]
    thresholds = norm.ppf(_probabilities(scenario.profile))
    occurrence = np.stack([np.convolve(row, np.ones(width) / math.sqrt(width), mode="valid") for row in noise])
    occurrence = occurrence < np.tile(thresholds, MONTE_CARLO_YEARS)
    anchors = occurrence.shape[1] - high
    # One count per anchor window, then the low offset selects the first window month.
    windows = np.convolve(occurrence[1].astype(np.int64), np.ones(high - low + 1, dtype=np.int64), mode="valid")[low:]
    given = occurrence[0][:anchors]
    calendar = np.arange(anchors) % 12
    truth = np.full(12, np.nan)
    for month in range(12):
        selected = (calendar == month) & given
        if np.any(selected):
            truth[month] = np.count_nonzero(windows[selected]) / np.count_nonzero(selected)
    return truth


def _truths(scenario: Scenario) -> dict[Statistic, npt.NDArray[np.float64]]:
    """Return the predeclared per-month truth for each statistic of the scenario."""
    if scenario.lag_window == (0, 0):
        return _analytic_truths(scenario)
    return {"conditional_probability": _lagged_truth(scenario)}


def _diagnostics(rows: npt.NDArray[np.bool_], years: int) -> tuple[float, float]:
    """Return the mean within-month lag-1 autocorrelation and the exceedance fraction."""
    series = rows[0].reshape(years, 12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        correlations = [
            float(np.corrcoef(series[:-1, month].astype(np.float64), series[1:, month].astype(np.float64))[0, 1])
            for month in range(12)
        ]
    return float(np.nanmean(correlations)), float(series.mean())


def _scenario_counts(
    scenario: Scenario,
    truths: dict[Statistic, npt.NDArray[np.float64]],
) -> tuple[dict[tuple[Statistic, int | None], Case], dict[str, float]]:
    """Accumulate every predeclared case of one scenario over the independent records."""
    time = _time_axis(scenario.years)
    thresholds = norm.ppf(_probabilities(scenario.profile))
    keys: tuple[tuple[Statistic, int | None], ...] = tuple(
        (statistic, length) for statistic in scenario.statistics for length in scenario.lengths
    )
    covered: dict[tuple[Statistic, int | None], npt.NDArray[np.int64]] = {
        key: np.zeros(12, dtype=np.int64) for key in keys
    }
    rejected = {key: np.zeros(12, dtype=np.int64) for key in keys}
    intervals = {key: np.zeros(12, dtype=np.int64) for key in keys}
    p_values = {key: np.zeros(12, dtype=np.int64) for key in keys}
    structural = {key: np.zeros(12, dtype=np.int64) for key in keys}
    few_blocks = {key: np.zeros(12, dtype=np.int64) for key in keys}
    lengths_used: dict[tuple[Statistic, int | None], int] = dict.fromkeys(keys, 0)
    decisions: dict[tuple[Statistic, int | None], npt.NDArray[np.bool_]] = {
        key: np.zeros((TRIALS, 12), dtype=bool) for key in keys
    }
    correlations, fractions = 0.0, 0.0
    for trial in range(TRIALS):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, scenario.stream, trial]))
        noise = rng.standard_normal((2, scenario.years * 12 + scenario.moving_average_months - 1))
        noise[1] = scenario.rho * noise[0] + math.sqrt(1 - scenario.rho * scenario.rho) * noise[1]
        values = np.stack(
            [
                np.convolve(
                    row,
                    np.ones(scenario.moving_average_months) / math.sqrt(scenario.moving_average_months),
                    mode="valid",
                )
                for row in noise
            ]
        )
        rows = values < np.tile(thresholds, scenario.years)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            correlation, fraction = _diagnostics(rows, scenario.years)
        correlations += correlation
        fractions += fraction
        hazards = compound.AlignedHazards(("A", "B"), time, "monthly", rows, np.ones_like(rows), {})
        for statistic, length in keys:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ClimateIndicesWarning)
                result = compound.bootstrap_dependence(
                    hazards,
                    statistic=statistic,
                    given="A" if statistic == "conditional_probability" else None,
                    target="B" if statistic == "conditional_probability" else None,
                    lag_window=scenario.lag_window,
                    block_length_years=length,
                    n_resamples=scenario.resamples,
                    seed=trial + SEED,
                )
            truth = truths[statistic]
            bits = dict(result.metadata["quality_flags"])
            excluded = np.zeros(12, dtype=bool)
            for name in _STRUCTURAL_GUARDS:
                excluded |= (result.quality_flags & bits[name]) != 0
            lengths_used[(statistic, length)] = int(result.block_length_years)
            structural[(statistic, length)] += excluded
            few_blocks[(statistic, length)] += (result.quality_flags & bits["few_blocks"]) != 0
            intervals[(statistic, length)] += np.isfinite(result.confidence_low)
            p_values[(statistic, length)] += np.isfinite(result.p_value)
            covered[(statistic, length)] += (result.confidence_low <= truth) & (truth <= result.confidence_high)
            decisions[(statistic, length)][trial] = result.p_value <= _TIE_ALPHA
            rejected[(statistic, length)] += decisions[(statistic, length)][trial]
    counts = {
        key: Case(
            covered=covered[key],
            rejected=rejected[key],
            intervals=intervals[key],
            p_values=p_values[key],
            structural=structural[key],
            few_blocks=few_blocks[key],
            length=lengths_used[key],
            decisions=decisions[key],
        )
        for key in keys
    }
    return counts, {
        "mean_lag1_within_month_autocorrelation": correlations / TRIALS,
        "mean_occurrence_fraction": fractions / TRIALS,
    }


def _band(tail: float, records: int, probability: float) -> int:
    """Return the lower quantile at `probability` or the upper quantile at `1 - probability`."""
    if records <= 0:
        return 0
    if probability >= 0.5:
        return int(binom.ppf(tail, records, probability))
    return int(binom.ppf(1 - tail, records, probability))


def _case_result(
    scenario: Scenario,
    statistic: Statistic,
    length: int | None,
    case: Case,
    truth: npt.NDArray[np.float64],
    tail: float,
) -> dict[str, Any]:
    """Apply the predeclared gates to one case and report every measured count.

    A structural guard can fire on some records and not others, so the band applies to the
    records that carry an inference. With nothing excluded this is the predeclared 109 and 19
    of 128.
    """
    covered = case.covered
    rejected = case.rejected
    eligible = TRIALS - case.structural
    inferable = eligible > 0
    minimum_covered = np.array([_band(tail, int(n), 0.95) for n in eligible])
    maximum_rejected = np.array([_band(tail, int(n), 0.05) for n in eligible])
    coverage_failed = np.flatnonzero((covered < minimum_covered) & inferable) + 1
    null_failed = np.flatnonzero(inferable & ((rejected > maximum_rejected) | (case.p_values < eligible))) + 1
    return {
        "group": scenario.group,
        "scenario": scenario.name,
        "stream": scenario.stream,
        "profile": scenario.profile,
        "moving_average_months": scenario.moving_average_months,
        "rho": scenario.rho,
        "years": scenario.years,
        "resamples": scenario.resamples,
        "lag_window": list(scenario.lag_window),
        "statistic": statistic,
        "block_length_override": length,
        "block_length_years": case.length,
        "truth": truth.tolist(),
        "covered": covered.tolist(),
        "rejected": rejected.tolist(),
        "finite_intervals": case.intervals.tolist(),
        "finite_p_values": case.p_values.tolist(),
        "structural_exclusions": case.structural.tolist(),
        "inferable_records": eligible.tolist(),
        "minimum_covered_records": minimum_covered.tolist(),
        "maximum_null_rejections": maximum_rejected.tolist(),
        "few_blocks_months": case.few_blocks.tolist(),
        "coverage_failed_months": coverage_failed.tolist(),
        "null_failed_months": null_failed.tolist(),
        "coverage_wilson_95": [
            [float(bound) for bound in binomtest(int(n), TRIALS).proportion_ci(method="wilson")] for n in covered
        ],
        "rejection_wilson_95": [
            [float(bound) for bound in binomtest(int(n), TRIALS).proportion_ci(method="wilson")] for n in rejected
        ],
        "coverage_gate_passed": bool(coverage_failed.size == 0 and np.any(inferable)),
        "null_gate_passed": None if scenario.rho != 0 else bool(null_failed.size == 0 and np.any(inferable)),
    }


def characterize(groups: Sequence[str] | None = None) -> dict[str, Any]:
    """Run the predeclared plan and return the report; `groups` selects a subset."""
    logging.disable(logging.CRITICAL)
    plan = _plan()
    unknown = sorted(set(groups or ()) - {scenario.group for scenario in plan})
    if unknown:
        raise ValueError(f"Unknown characterization group(s): {', '.join(unknown)}.")
    selected = tuple(scenario for scenario in plan if groups is None or scenario.group in groups)
    comparisons = len(SCENARIOS) * len(STATISTICS) * len(LENGTHS) * 12 * 2
    tail = _FAMILY_WISE_ALPHA / comparisons
    results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for scenario in selected:
        truths = _truths(scenario)
        counts, diagnostics = _scenario_counts(scenario, truths)
        for (statistic, length), case in counts.items():
            result = _case_result(scenario, statistic, length, case, truths[statistic], tail)
            default = counts.get((statistic, 6))
            result["decisions_differ_from_default"] = (
                np.count_nonzero(case.decisions != default.decisions, axis=0).tolist() if default is not None else None
            )
            results.append(result)
        summaries.append(
            {
                "scenario": scenario.name,
                "group": scenario.group,
                "stream": scenario.stream,
                "profile": scenario.profile,
                "moving_average_months": scenario.moving_average_months,
                "rho": scenario.rho,
                "years": scenario.years,
                "resamples": scenario.resamples,
                "lag_window": list(scenario.lag_window),
                **diagnostics,
            }
        )
        print(f"Completed {scenario.name}.", flush=True)
    passed = all(result["coverage_gate_passed"] and result["null_gate_passed"] is not False for result in results)
    return {
        "schema_version": 2,
        "classification": "synthetic characterization only; no external validation",
        "configuration": {
            "trials": TRIALS,
            "years": YEARS,
            "resamples": RESAMPLES,
            "seed": SEED,
            "groupby": "month",
            "lengths": LENGTHS,
            "default_length": "ceil(years ** (1 / 3)) when no override is given",
            "marginal_probabilities": _probabilities("seasonal").tolist(),
            "profiles": {name: list(values) for name, values in PROFILES.items()},
            "lag_windows": sorted({scenario.lag_window for scenario in plan if scenario.lag_window != (0, 0)}),
            "monte_carlo_years": MONTE_CARLO_YEARS,
            "structural_guards": list(_STRUCTURAL_GUARDS),
            "familywise_alpha": _FAMILY_WISE_ALPHA,
            "binomial_comparisons": comparisons,
            "minimum_covered": int(binom.ppf(tail, TRIALS, 0.95)),
            "maximum_null_rejections": int(binom.ppf(1 - tail, TRIALS, 0.05)),
            "band_rule": (
                "exact binomial at familywise_alpha / binomial_comparisons, applied to the records where the "
                "structural guards allowed inference"
            ),
            "null_alpha": _TIE_ALPHA,
            "confidence_level": 0.95,
            "groups": sorted({scenario.group for scenario in plan}),
        },
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "passed": passed,
        "scenarios": summaries,
        "results": results,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=_output_path, required=True)
    parser.add_argument(
        "--group",
        action="append",
        choices=sorted({scenario.group for scenario in _plan()}),
        help="Repeat to select a group; the default runs every group.",
    )
    args = parser.parse_args(argv)
    report = characterize(args.group)
    _write_report(args.output, report)
    failed = [
        result
        for result in report["results"]
        if not result["coverage_gate_passed"] or result["null_gate_passed"] is False
    ]
    print(
        f"Characterization passed: {report['passed']}. Cases: {len(report['results'])}, failed: {len(failed)}. "
        f"Report: {args.output}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
