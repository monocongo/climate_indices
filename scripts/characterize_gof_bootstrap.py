"""Reproduce the synthetic characterization of the goodness-of-fit bootstrap (#1586, ADR-0028 section 5).

This script draws records whose every calendar step and cell is an independent sample from one candidate
family. It runs ``compare_distributions`` with that family as the only candidate and counts how often
the Kolmogorov-Smirnov and Anderson-Darling bootstrap p-values reject at ``alpha = 0.05``.

For a series with ``B_ok`` successful refits, the fraction of p-value grid points at or below alpha is
``floor(alpha * (B_ok + 1)) / (B_ok + 1)``. The report averages this grid fraction over series with finite
p-values. Estimation, ties, and the filtering of failed refits can affect rank probabilities, so size
is measured from observed rejection frequencies. The 99.9% exact-binomial count band is a reference
for independent tests with rejection probability alpha. The report also gives Wilson intervals for
the measured rates. Series without a p-value leave the rejection denominator.

The cases are the four candidate families on a continuous record, gamma on a record rounded to 10 and
bootstrapped with ``resolution=10``, and two controls: the naive K-S p-value for gamma and the same
rounded record bootstrapped without a resolution. One further case probes a harder regime: a zero-heavy gamma of shape
0.8 and scale 3 with 15% exact zeros, rounded to a unit of 1, where 35% of the record is zero and the unit is
comparable to the mean.

Reproduction uses a fixed seed. The 399 draws reduce the gap between the nominal level and the
attainable p-value threshold when refits fail. Runtime depends on the number of draws.

Run: uv run scripts/characterize_gof_bootstrap.py --output docs/design/gof-bootstrap-characterization.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import scipy.stats
from scipy.stats import binom, binomtest

from climate_indices import __version__, compute, suitability
from climate_indices.vocabulary import Distribution

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "docs" / "design"
YEARS = 40
CELLS = (6, 10)
REPEATS = 5
N_BOOTSTRAP = 399
ALPHA = 0.05
CONFIDENCE = 0.999
SEED = 1586
ROUNDING = 10.0  # the unit of the rounded record, coarse enough that a sample holds many ties


def _draw(
    distribution: Distribution, rng: np.random.Generator, zero_heavy: bool = False
) -> tuple[np.ndarray, suitability.IndexRoute, dict[str, Any]]:
    """One record whose every calendar step and cell is iid from ``distribution``, with its route and extra arguments."""
    shape = (YEARS * 12, *CELLS)
    if distribution is Distribution.gamma and zero_heavy:
        values = rng.gamma(0.8, 3.0, size=shape)
        values[rng.random(shape) < 0.15] = 0.0
        return values, "spi", {}
    if distribution is Distribution.gamma:
        return rng.gamma(2.0, 30.0, size=shape), "spi", {}
    if distribution is Distribution.pearson:
        draw = scipy.stats.pearson3.rvs(1.0, loc=100.0, scale=30.0, size=shape, random_state=rng)
        return np.asarray(draw), "spi", {}
    if distribution is Distribution.loglogistic:
        # Fisk(c=1/0.3, scale=60) is Hosking's GLO with loc=60, scale=18, k=-0.3 (CDFs agree to 2e-16)
        draw = scipy.stats.fisk.rvs(c=1 / 0.3, scale=60.0, size=shape, random_state=rng)
    else:
        draw = scipy.stats.genextreme.rvs(-0.1, loc=100.0, scale=30.0, size=shape, random_state=rng)
    return np.asarray(draw), "spei", {"pet_mm": np.zeros(shape)}  # zero PET leaves the water balance equal to the draw


def _fit_record(
    distribution: Distribution,
    repeat: int,
    *,
    rounded: bool,
    resolution: float | None,
    naive: bool,
    zero_heavy: bool,
    unit: float,
) -> suitability.GoodnessOfFit:
    values, index, extra = _draw(distribution, np.random.default_rng([SEED, repeat]), zero_heavy)
    if rounded:
        values = np.round(values / unit) * unit
    gof = suitability.compare_distributions(
        values,
        1,
        index=index,
        data_start_year=1950,
        calibration_year_initial=1950,
        calibration_year_final=1950 + YEARS - 1,
        periodicity=compute.Periodicity.monthly,
        candidates=(distribution,),
        n_bootstrap=0 if naive else N_BOOTSTRAP,
        resolution=resolution,
        seed=SEED + repeat,
        spatial_time_major=True,
        **extra,
    ).goodness_of_fit
    if gof is None:
        raise RuntimeError("compare_distributions returned no goodness of fit")
    return gof


def _run(
    distribution: Distribution,
    *,
    rounded: bool,
    resolution: float | None,
    naive: bool = False,
    zero_heavy: bool = False,
    unit: float = ROUNDING,
) -> dict[str, Any]:
    rejected = np.zeros(2, dtype=np.int64)
    series = 0
    failed_shares: list[float] = []
    grid_fractions: list[float] = []
    for repeat in range(REPEATS):
        gof = _fit_record(
            distribution,
            repeat,
            rounded=rounded,
            resolution=resolution,
            naive=naive,
            zero_heavy=zero_heavy,
            unit=unit,
        )
        if naive:
            p_values = (gof.ks_p_value_naive,)
        else:
            if (
                gof.ks_p_value_bootstrap is None
                or gof.ad_p_value_bootstrap is None
                or gof.bootstrap_failed_share is None
            ):
                raise RuntimeError("the bootstrap p-values are missing from the goodness of fit")
            p_values = (gof.ks_p_value_bootstrap, gof.ad_p_value_bootstrap)
            shares = gof.bootstrap_failed_share[np.isfinite(gof.bootstrap_failed_share)]
            failed_shares += [float(share) for share in shares]
            share_with_p = gof.bootstrap_failed_share[np.isfinite(p_values[0])]
            grid_size = np.rint(N_BOOTSTRAP * (1.0 - share_with_p)) + 1
            grid_fractions.extend(np.floor(ALPHA * grid_size) / grid_size)
        series += int(np.isfinite(p_values[0]).sum())
        for position, p in enumerate(p_values):
            rejected[position] += int((p <= ALPHA).sum())
    names = ["ks"] if naive else ["ks", "ad"]
    low, high = (float(bound) for bound in binom.interval(CONFIDENCE, series, ALPHA))
    return {
        "rounding_unit": unit if rounded else None,
        "resolution": resolution,
        "series": series,
        "statistics": {
            name: {
                "rejected": int(rejected[position]),
                "rate": float(rejected[position] / series),
                "wilson_95": [
                    float(b) for b in binomtest(int(rejected[position]), series).proportion_ci(method="wilson")
                ],
                "within_nominal_band": bool(low <= rejected[position] <= high),
            }
            for position, name in enumerate(names)
        },
        "mean_grid_fraction_at_or_below_alpha": None if naive else float(np.mean(grid_fractions)),
        "nominal_band": [low, high],
        "bootstrap_failed_share": None
        if naive
        else {"mean": float(np.mean(failed_shares)), "max": float(np.max(failed_shares))},
    }


CASES: dict[str, Callable[[], dict[str, Any]]] = {
    "gamma": lambda: _run(Distribution.gamma, rounded=False, resolution=None),
    "pearson": lambda: _run(Distribution.pearson, rounded=False, resolution=None),
    "loglogistic": lambda: _run(Distribution.loglogistic, rounded=False, resolution=None),
    "gev": lambda: _run(Distribution.gev, rounded=False, resolution=None),
    "gamma_rounded_with_resolution": lambda: _run(Distribution.gamma, rounded=True, resolution=ROUNDING),
    "control_gamma_rounded_without_resolution": lambda: _run(Distribution.gamma, rounded=True, resolution=None),
    "control_gamma_naive_k_s": lambda: _run(Distribution.gamma, rounded=False, resolution=None, naive=True),
    "gamma_zero_heavy_rounded_to_the_mean_with_resolution": lambda: _run(
        Distribution.gamma, rounded=True, resolution=1.0, zero_heavy=True, unit=1.0
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--output", type=Path, required=True, help="JSON file below docs/design")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if OUTPUT_ROOT not in output.parents or output.suffix != ".json":
        parser.error(f"--output must be a .json file below {OUTPUT_ROOT}")
    warnings.simplefilter("ignore")
    results: dict[str, Any] = {}
    for name, case in CASES.items():
        started = time.perf_counter()
        results[name] = case() | {"seconds": round(time.perf_counter() - started, 1)}
        print(name, {k: v["rejected"] for k, v in results[name]["statistics"].items()}, "of", results[name]["series"])
    report = {
        "configuration": {
            "years": YEARS,
            "cells": list(CELLS),
            "calendar_steps": 12,
            "repeats": REPEATS,
            "n_bootstrap": N_BOOTSTRAP,
            "alpha": ALPHA,
            "default_rounding_unit": ROUNDING,
            "band_confidence": CONFIDENCE,
            "seed": SEED,
        },
        "environment": {
            "climate_indices": __version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "python": platform.python_version(),
        },
        "cases": results,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
