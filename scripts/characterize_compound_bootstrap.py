"""Reproduce CMP-07 synthetic characterization, not external scientific validation.

Predeclared configuration: 128 independent records per scenario, 200 years,
399 resamples, month groups, and block lengths 3, 6 (the default), and 12 years.
Gaussian moving averages preserve overlapping 6- and 36-month source windows.
Seasonal marginal probabilities range from .35 to .55. Same-time correlation is
zero under the null and .5 in the associated scenario. Analytic bivariate normal
quadrature supplies each month's truth, independently of the counting code.

Acceptance uses exact binomial bands with a familywise .01 error budget across
all scenario/statistic/length/month coverage checks and independent-null checks.
Coverage must exceed the lower count quantile under nominal .95 coverage. Null
rejections must not exceed the upper count quantile under nominal .05 rejection.
Conservative null tests are allowed. Every missing interval counts as uncovered;
every missing p-value is reported and fails the null gate. Correlated-scenario
rejection rates describe power, not null calibration. Report Wilson 95% intervals
for every measured proportion. Do not select a block length by its p-value.

Run: uv run scripts/characterize_compound_bootstrap.py --output <JSON path>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import warnings
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.stats import binom, binomtest, norm

from climate_indices import compound
from climate_indices.exceptions import ClimateIndicesWarning

TRIALS = 128
YEARS = 200
RESAMPLES = 399
SEED = 1371
LENGTHS = (3, 6, 12)
STATISTICS = ("joint_probability", "likelihood_multiplication_factor", "conditional_probability")
SCENARIOS = (
    ("seasonal_independent", 1, 0.0),
    ("overlapping_6_month_independent", 6, 0.0),
    ("persistent_36_month_independent", 36, 0.0),
    ("persistent_36_month_correlated", 36, 0.5),
)


def characterize() -> dict:
    logging.disable(logging.CRITICAL)
    probabilities = 0.45 + 0.1 * np.cos(2 * np.pi * np.arange(12) / 12)
    thresholds = norm.ppf(probabilities)
    comparisons = len(SCENARIOS) * len(STATISTICS) * len(LENGTHS) * 12 * 2
    tail = 0.01 / comparisons
    minimum_covered = int(binom.ppf(tail, TRIALS, 0.95))
    maximum_rejected = int(binom.ppf(1 - tail, TRIALS, 0.05))
    time = (np.datetime64("1800-01", "M") + np.arange(YEARS * 12)).astype("datetime64[D]")
    results = []
    passed = True
    for scenario_index, (name, width, rho) in enumerate(SCENARIOS):
        truth_joint = np.array(
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
        truths = (truth_joint, truth_joint / probabilities**2, truth_joint / probabilities)
        covered = np.zeros((3, 3, 12), dtype=int)
        rejected = np.zeros_like(covered)
        intervals = np.zeros_like(covered)
        p_values = np.zeros_like(covered)
        decisions = np.zeros((TRIALS, 3, 3, 12), dtype=bool)
        for trial in range(TRIALS):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, scenario_index, trial]))
            noise = rng.standard_normal((2, YEARS * 12 + width - 1))
            noise[1] = rho * noise[0] + math.sqrt(1 - rho * rho) * noise[1]
            values = np.stack([np.convolve(row, np.ones(width) / math.sqrt(width), mode="valid") for row in noise])
            rows = values < np.tile(thresholds, YEARS)
            hazards = compound.AlignedHazards(("A", "B"), time, "monthly", rows, np.ones_like(rows), {})
            for si, statistic in enumerate(STATISTICS):
                for li, length in enumerate(LENGTHS):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ClimateIndicesWarning)
                        result = compound.bootstrap_dependence(
                            hazards,
                            statistic=statistic,
                            given="A" if si == 2 else None,
                            target="B" if si == 2 else None,
                            block_length_years=length,
                            n_resamples=RESAMPLES,
                            seed=trial + SEED,
                        )
                    intervals[si, li] += np.isfinite(result.confidence_low)
                    p_values[si, li] += np.isfinite(result.p_value)
                    covered[si, li] += (result.confidence_low <= truths[si]) & (truths[si] <= result.confidence_high)
                    decisions[trial, si, li] = result.p_value <= 0.05
                    rejected[si, li] += decisions[trial, si, li]
        for si, statistic in enumerate(STATISTICS):
            for li, length in enumerate(LENGTHS):
                coverage_ok = bool(np.all(covered[si, li] >= minimum_covered))
                null_ok = (
                    bool(np.all(rejected[si, li] <= maximum_rejected) and np.all(p_values[si, li] == TRIALS))
                    if rho == 0
                    else None
                )
                passed = passed and coverage_ok and null_ok is not False
                results.append(
                    {
                        "scenario": name,
                        "moving_average_months": width,
                        "rho": rho,
                        "statistic": statistic,
                        "block_length_years": length,
                        "truth": truths[si].tolist(),
                        "covered": covered[si, li].tolist(),
                        "finite_intervals": intervals[si, li].tolist(),
                        "finite_p_values": p_values[si, li].tolist(),
                        "rejected": rejected[si, li].tolist(),
                        "coverage_wilson_95": [
                            [float(bound) for bound in binomtest(int(n), TRIALS).proportion_ci(method="wilson")]
                            for n in covered[si, li]
                        ],
                        "rejection_wilson_95": [
                            [float(bound) for bound in binomtest(int(n), TRIALS).proportion_ci(method="wilson")]
                            for n in rejected[si, li]
                        ],
                        "decisions_differ_from_default": np.count_nonzero(
                            decisions[:, si, li] != decisions[:, si, 1], axis=0
                        ).tolist(),
                        "coverage_gate_passed": coverage_ok,
                        "null_gate_passed": null_ok,
                    }
                )
        print(f"Completed {name}.", flush=True)
    return {
        "schema_version": 1,
        "classification": "synthetic characterization only; no external validation",
        "configuration": {
            "trials": TRIALS,
            "years": YEARS,
            "resamples": RESAMPLES,
            "seed": SEED,
            "groupby": "month",
            "lengths": LENGTHS,
            "default_length": 6,
            "marginal_probabilities": probabilities.tolist(),
            "familywise_alpha": 0.01,
            "binomial_comparisons": comparisons,
            "minimum_covered": minimum_covered,
            "maximum_null_rejections": maximum_rejected,
            "null_alpha": 0.05,
            "confidence_level": 0.95,
        },
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "passed": passed,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = characterize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Characterization passed: {report['passed']}. Report: {args.output}")
    raise SystemExit(0 if report["passed"] else 1)
