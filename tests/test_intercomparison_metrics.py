"""Offline tests of the protocol measures (REFIMPL-16, issue 1618).

Every row of the Measure reference in docs/research/reference-papers/protocol.md is checked
against values worked out by hand on small committed inputs, plus properties (symmetry, identity,
order invariance) and tripwires against the two classifiers the measures must agree with:
``climate_indices.suitability.drought_category`` and the Earth Engine pilot's ``usdm_bin``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import statistics
import sys
import warnings
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.special import ndtri
from scipy.stats import kurtosis, skew

from climate_indices.suitability import drought_category
from scripts import earthengine_climate_engine_pilot as pilot

ROOT = Path(__file__).parent.parent
HARNESS = ROOT / "intercomparison"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


metrics = _load("intercomparison_metrics", HARNESS / "metrics.py")
nan, inf = math.nan, math.inf

# The committed synthetic pair. Steps 4 and 5 are finite in b only, so 6 of 8 steps pair up and
# the differences on those are -0.1, -0.1, -0.1, 0, 0 and -0.5.
A = np.array([0.0, -0.5, -1.3, 2.1, nan, inf, 0.5, -2.5])
B = np.array([0.1, -0.4, -1.2, 2.1, 0.3, 1.0, 0.5, -2.0])
REL = 1e-12  # the hand values are exact up to a few binary64 roundings


def approx(value: float) -> Any:
    return pytest.approx(value, rel=REL, abs=1e-15)


# Classification and conventions


@pytest.mark.parametrize(
    ("value", "edges", "rule", "code"),
    [
        (-0.5, metrics.SIX_CLASS_EDGES, "upper_inclusive", 4),  # exactly -0.5 is D0
        (-0.5, metrics.SIX_CLASS_EDGES, "lower_inclusive", 5),  # the pilot's rule: near normal
        (-2.0, metrics.SIX_CLASS_EDGES, "upper_inclusive", 0),  # D4 includes its upper edge
        (-2.0, metrics.SIX_CLASS_EDGES, "lower_inclusive", 1),
        (0.5, metrics.ELEVEN_CLASS_EDGES, "upper_inclusive", 5),  # middle class (-0.5, 0.5]
        (0.5, metrics.ELEVEN_CLASS_EDGES, "lower_inclusive", 6),
        (3.0, metrics.ELEVEN_CLASS_EDGES, "upper_inclusive", 10),
    ],
)
def test_classify_puts_each_edge_on_the_side_its_rule_names(
    value: float, edges: tuple[float, ...], rule: str, code: int
) -> None:
    assert metrics.classify([value], edges, rule).tolist() == [code]


def test_classify_casts_the_edges_to_the_series_width() -> None:
    # float32 -1.3 is -1.29999995 in float64: on the edge at its own width, above it once widened
    x32 = np.array([-1.3], dtype=np.float32)
    assert metrics.classify(x32, metrics.SIX_CLASS_EDGES).tolist() == [2]
    assert metrics.classify(x32.astype(np.float64), metrics.SIX_CLASS_EDGES).tolist() == [3]


def test_classify_marks_values_that_are_not_finite_and_rejects_an_unknown_rule() -> None:
    assert metrics.classify([nan, inf, -inf, 0.0], metrics.SIX_CLASS_EDGES).tolist() == [-1, -1, -1, 5]
    with pytest.raises(ValueError, match="boundary rule"):
        metrics.classify([0.0], metrics.SIX_CLASS_EDGES, "nearest")


def test_six_class_codes_are_the_reverse_of_drought_category_codes() -> None:
    values = np.unique(np.r_[np.linspace(-3.0, 1.0, 401), metrics.SIX_CLASS_EDGES])
    for x in (values, values.astype(np.float32)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the suitability module warns that it is Beta
            library = np.asarray(drought_category(x, output_scale="normal"))
        np.testing.assert_array_equal(library, 5 - metrics.classify(x, metrics.SIX_CLASS_EDGES))


def test_eleven_class_edges_are_the_pilot_edges() -> None:
    np.testing.assert_array_equal(metrics.ELEVEN_CLASS_EDGES, pilot.USDM_EDGES[1:-1])


def test_series_casts_integers_keeps_float32_and_fills_masked_values() -> None:
    assert metrics.series([1, 0]).dtype == np.float64
    assert metrics.series(np.array([True])).dtype == np.float64
    masked = metrics.series(np.ma.array([1.0, 2.0], mask=[False, True], dtype=np.float32))
    assert masked.dtype == np.float32
    assert np.isnan(masked[1])
    with pytest.raises(TypeError, match="object"):
        metrics.series(np.array([1.0, None], dtype=object))
    with pytest.raises(ValueError, match="1-D"):
        metrics.series(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="length mismatch"):
        metrics.pair_agreement([1.0, 2.0], [1.0])


def test_tolerance_comparison_allows_representation_error_only() -> None:
    assert abs(1.255 - 1.26) > 0.005  # 0.005000000000000115 in binary64
    assert metrics.within_tolerance(abs(1.255 - 1.26), 0.005)
    assert not metrics.within_tolerance(0.00501, 0.005)


# C1, C2


def test_printed_value_pass() -> None:
    assert metrics.printed_value_pass(1.255, 1.26, 2)
    assert not metrics.printed_value_pass(1.2549, 1.26, 2)
    assert metrics.printed_value_pass(1.30, 1.26, 2, tolerance=0.05)
    assert not metrics.printed_value_pass(nan, 1.0, 1)
    assert not metrics.printed_value_pass(inf, 1.0, 1)
    with pytest.raises(ValueError, match="negative"):
        metrics.printed_value_pass(1.0, 1.0, -1)


def test_error_by_band_on_hand_values() -> None:
    # 40 errors of +0.1 at |z*| = 0.5, 40 of -0.2 at 1.5, 10 of +0.3 at 2.5 (too few for a band),
    # two truths that are not finite, one index that is not, and three excluded zeros
    truth = np.r_[np.full(40, 0.5), np.full(40, -1.5), np.full(10, 2.5), [inf, nan, 0.5], np.full(3, 0.2)]
    index = np.r_[np.full(40, 0.6), np.full(40, -1.7), np.full(10, 2.8), [1.0, 1.0, nan], np.full(3, 9.0)]
    exclude = np.r_[np.zeros(93, bool), np.ones(3, bool)]
    result = metrics.error_by_band(index, truth, exclude=exclude)
    assert result["overall"] == {
        "bias": approx(-1 / 90),
        "rmse": approx(math.sqrt(2.9 / 90)),
        "max_abs_error": approx(0.3),
        "n": 90,
    }
    low, middle, high = result["bands"]
    assert (low["lower"], low["upper"], low["n"]) == (0.0, 1.0, 40)
    assert low["bias"] == approx(0.1)
    assert low["rmse"] == approx(0.1)
    assert middle["bias"] == approx(-0.2)
    assert middle["max_abs_error"] == approx(0.2)
    assert high["n"] == 10
    assert math.isnan(high["bias"])
    assert high["upper"] == inf
    assert (result["n_truth_not_finite"], result["n_excluded"], result["n_index_not_finite"]) == (2, 3, 1)


def test_replicate_summary() -> None:
    assert metrics.replicate_summary([1.0, 2.0, 3.0, 4.0, nan]) == {
        "median": 2.5,
        "p2_5": approx(1.075),
        "p97_5": approx(3.925),
        "n_non_empty": 4,
    }
    assert math.isnan(metrics.replicate_summary([nan])["median"])


# C3 and the difference classes


def test_pair_agreement_on_the_committed_pair() -> None:
    finite = np.isfinite(A) & np.isfinite(B)
    result = metrics.pair_agreement(A, B)
    assert result == {
        "bias": approx(-0.8 / 6),
        "mae": approx(0.8 / 6),
        "rmse": approx(math.sqrt(0.28 / 6)),
        "r": approx(statistics.correlation(A[finite].tolist(), B[finite].tolist())),
        "max_abs": approx(0.5),
        "n": 6,
    }


def test_pair_agreement_leaves_r_undefined_without_variance_or_pairs() -> None:
    assert math.isnan(metrics.pair_agreement([1.0, 1.0, 1.0], [0.0, 1.0, 2.0])["r"])
    assert math.isnan(metrics.pair_agreement([1.0], [2.0])["r"])
    assert metrics.pair_agreement([nan], [1.0])["n"] == 0


def test_mask_mismatch_counts_steps_finite_in_one_series() -> None:
    assert metrics.mask_mismatch(A, B) == 0.25
    assert metrics.mask_mismatch([nan, 1.0], [nan, 1.0]) == 0.0
    assert math.isnan(metrics.mask_mismatch([], []))


@pytest.mark.parametrize(
    ("a", "b", "edges", "expected"),
    [
        ([0.3, -1.0], [0.3 + 5e-11, -1.0], metrics.SIX_CLASS_EDGES, "numerically identical"),
        ([0.3, -1.0], [0.3 + 5e-7, -1.0], metrics.SIX_CLASS_EDGES, "equivalent"),
        ([1.255], [1.26], metrics.SIX_CLASS_EDGES, "below reporting precision"),
        ([-2.5, 0.0], [-2.6, 0.2], metrics.SIX_CLASS_EDGES, "category-preserving"),
        (A, B, metrics.SIX_CLASS_EDGES, "category-changing"),
        (A, B, None, "above reporting precision"),
        ([nan, 1.0], [1.0, nan], metrics.SIX_CLASS_EDGES, "no overlap"),
    ],
)
def test_difference_class(a: Any, b: Any, edges: Any, expected: str) -> None:
    assert metrics.difference_class(a, b, edges=edges)["class"] == expected


def test_difference_class_compares_categories_at_each_series_width() -> None:
    # widened, float32 -1.3 is 4.8e-8 above -1.3, but both sit on the edge at their own width
    result = metrics.difference_class(np.array([-1.3], np.float32), np.array([-1.3]))
    assert result["class"] == "equivalent"
    assert result["d"] == approx(abs(float(np.float32(-1.3)) + 1.3))


def test_category_measures_on_the_committed_pair() -> None:
    # six-class codes, upper edges included: a = 5 4 2 5 5 0, b = 5 5 3 5 5 0
    assert metrics.class_change_share(A, B) == {"share": approx(2 / 6), "share_touching_drought": 1.0, "n": 6}
    assert metrics.category_agreement(A, B) == {"same_class": approx(4 / 6), "within_one": 1.0, "n": 6}
    expected = np.zeros((6, 6), int)
    expected[5, 5], expected[5, 4], expected[3, 2], expected[0, 0] = 3, 1, 1, 1  # rows b, columns a
    np.testing.assert_array_equal(metrics.confusion_matrix(A, B), expected)
    # lower edges included: -0.5 and -0.4 agree, -1.3 and -1.2 agree, -2.5 and -2.0 now differ
    lower = metrics.class_change_share(A, B, rule="lower_inclusive")
    assert lower["share"] == approx(1 / 6)
    assert metrics.confusion_matrix(A, B, scale="eleven").shape == (11, 11)


def test_class_change_share_leaves_shares_undefined_without_a_denominator() -> None:
    wet = metrics.class_change_share([0.6, 1.0], [0.9, 1.0], scale="eleven")
    assert wet["share"] == 0.5
    assert wet["share_touching_drought"] == 0.0
    same = metrics.class_change_share([0.0], [0.1])
    assert same["share"] == 0.0
    assert math.isnan(same["share_touching_drought"])
    assert math.isnan(metrics.category_agreement([nan], [0.0])["same_class"])


def test_category_measures_accept_pre_registered_edges() -> None:
    # a PDSI-like study fixes its own edges: classes (-inf, -3], (-3, -2], (-2, 2], (2, inf)
    edges = (-3.0, -2.0, 2.0)
    a, b = [-3.5, -2.5, 0.0, 2.5], [-2.9, -2.5, 1.0, 3.0]
    assert metrics.class_change_share(a, b, scale=edges, drought_codes=2) == {
        "share": 0.25,
        "share_touching_drought": 1.0,
        "n": 4,
    }
    assert metrics.category_agreement(a, b, scale=edges)["same_class"] == 0.75
    assert metrics.confusion_matrix(a, b, scale=edges).shape == (4, 4)
    assert metrics.rung_categories([(a, b)], scale=edges)["classes"] == ["category-changing"]
    with pytest.raises(ValueError, match="drought_codes"):
        metrics.class_change_share(a, b, scale=edges)
    with pytest.raises(ValueError, match="strictly increasing"):
        metrics.category_agreement(a, b, scale=(1.0, 0.0))
    with pytest.raises(ValueError, match="unknown scale"):
        metrics.category_agreement(a, b, scale="seven")


# C4


def test_period_moments_on_hand_values() -> None:
    x = [-2.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 5.0, nan]
    # mean 0.5; sums of squared, cubed and fourth-power deviations 34.5, 72 and 464.625 over n = 10
    result = metrics.period_moments(x)
    assert result["mean"] == approx(0.5)
    assert result["sd"] == approx(math.sqrt(34.5 / 9))
    assert result["skewness"] == approx(7.2 / 3.45**1.5)
    assert result["excess_kurtosis"] == approx(46.4625 / 3.45**2 - 3)
    assert result["skewness"] == approx(float(skew(x[:-1])))
    assert result["excess_kurtosis"] == approx(float(kurtosis(x[:-1])))
    assert (result["n"], result["too_short"]) == (10, False)


def test_period_moments_of_constant_and_short_periods() -> None:
    constant = metrics.period_moments(np.full(12, 0.7))
    assert constant["sd"] == 0.0
    assert math.isnan(constant["skewness"])
    assert math.isnan(constant["excess_kurtosis"])
    short = metrics.period_moments(np.arange(9.0))
    assert short["too_short"]
    assert math.isnan(short["mean"])


def test_quantile_residual() -> None:
    q = ndtri((np.arange(1, 11) - 0.5) / 10)
    assert metrics.quantile_residual(q) == approx(0.0)
    shifted = q.copy()
    shifted[-1] += 0.3
    assert metrics.quantile_residual(np.r_[shifted, nan]) == approx(0.3)
    assert math.isnan(metrics.quantile_residual(q[:9]))


def test_null_band_reproduces_its_stream_with_independent_code() -> None:
    n, seed, draws = 30, 7, 2000
    band = metrics.null_band(n, 0.0, seed=seed, draws=draws)
    sample = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 1, n]))).standard_normal((draws, n))
    independent = {
        "mean": sample.mean(axis=1),
        "sd": sample.std(axis=1, ddof=1),
        "skewness": skew(sample, axis=1),
        "excess_kurtosis": kurtosis(sample, axis=1),
        "residual": np.abs(np.sort(sample, axis=1) - ndtri((np.arange(1, n + 1) - 0.5) / n)).max(axis=1),
    }
    assert band["available"]
    for name, values in independent.items():
        limits = band["statistics"][name]
        np.testing.assert_allclose([limits["lower"], limits["upper"]], np.percentile(values, [2.5, 97.5]), rtol=1e-12)
        assert limits["n_finite"] == draws


def test_null_band_of_the_mean_is_near_its_normal_limits() -> None:
    limits = metrics.null_band(30, 0.0, seed=1)["statistics"]["mean"]
    half_width = 1.959964 / math.sqrt(30)
    assert limits["lower"] == pytest.approx(-half_width, abs=0.02)  # Monte Carlo error at 10,000 draws
    assert limits["upper"] == pytest.approx(half_width, abs=0.02)


def test_null_samples_with_a_zero_mass_follow_the_protocol_recipe() -> None:
    n, p0, zero, seed, draws = 30, 0.2, -0.9, 7, 50
    samples = metrics.null_samples(n, p0, zero, seed=seed, draws=draws)
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 1, n])))
    k = generator.binomial(n, p0, size=draws)
    u = 1 - generator.random(size=(draws, n))
    for row in range(draws):
        expected = np.r_[np.full(k[row], zero), ndtri(p0 + (1 - p0) * u[row, : n - k[row]])]
        np.testing.assert_array_equal(samples[row], expected)
    assert metrics.null_band(n, p0, seed=seed, zero_score=zero, zero_handling="center")["available"]


def test_null_band_is_unavailable_without_its_inputs() -> None:
    assert not metrics.null_band(30, None, seed=1)["available"]
    assert not metrics.null_band(30, nan, seed=1)["available"]
    assert not metrics.null_band(30, 0.2, seed=1)["available"]
    assert not metrics.null_band(30, 0.2, seed=1, zero_score=nan)["available"]
    with pytest.raises(ValueError, match="p0"):
        metrics.null_band(30, 1.5, seed=1, zero_score=0.0)


def test_null_band_of_an_all_zero_period_is_the_zero_score() -> None:
    band = metrics.null_band(30, 1.0, seed=1, zero_score=-3.09, draws=100)["statistics"]
    assert (band["mean"]["lower"], band["mean"]["upper"]) == (-3.09, -3.09)
    assert (band["sd"]["lower"], band["sd"]["upper"]) == (0.0, 0.0)
    assert band["skewness"]["n_finite"] == 0  # every draw is constant
    assert math.isnan(band["skewness"]["upper"])


def test_outside_band_share() -> None:
    band = {
        "available": True,
        "statistics": {
            "mean": {"lower": -0.3, "upper": 0.3, "n_finite": 10},
            "residual": {"lower": 0.1, "upper": 0.5, "n_finite": 10},
        },
    }
    unavailable = {"available": False}
    assert metrics.outside_band_share([0.0, 0.4, -0.5, nan, 0.0], [band, band, band, band, None], "mean") == {
        "share": approx(2 / 3),
        "n_scored": 3,
    }
    # a residual below its lower limit is not outside
    residual = metrics.outside_band_share([0.05, 0.6, 0.2, 0.9], [band, band, band, unavailable], "residual")
    assert residual == {"share": approx(1 / 3), "n_scored": 3}
    assert math.isnan(metrics.outside_band_share([0.0], [None], "mean")["share"])
    with pytest.raises(ValueError, match="statistic"):
        metrics.outside_band_share([0.0], [band], "median")


def test_output_bound_share() -> None:
    assert metrics.output_bound_share([-3.09, 0.0, 3.09, 4.0, nan], 3.09) == 0.75
    assert metrics.output_bound_share([0.0], None) == "not applicable"
    # a float32 output clipped at the library's own bound counts as at the bound
    clipped = np.clip(np.array([-9.0, 9.0, 0.0, 3.0], np.float32), -3.09, 3.09)
    assert metrics.output_bound_share(clipped, 3.09) == 0.5
    assert math.isnan(metrics.output_bound_share([nan], 3.09))


# C5


@pytest.mark.parametrize(
    ("output", "expected", "warned", "outcome"),
    [
        ([1.0, 2.0], [1.0, 2.0005], False, "correct"),
        ([nan], [nan], False, "correct"),
        ([nan], [1.0], True, "NaN with warning"),
        ([nan], [1.0], False, "NaN silent"),
        ([1.5], [1.0], True, "wrong with warning"),
        ([1.5], [1.0], False, "silently wrong"),
        ([inf], [1.0], False, "silently wrong"),
        ([1.0], [nan], True, "wrong with warning"),
        ([1.0, nan, 3.0], [1.0, 2.0, 2.0], False, "silently wrong"),  # mixed: the most severe wins
    ],
)
def test_robustness_outcome(output: list[float], expected: list[float], warned: bool, outcome: str) -> None:
    result = metrics.robustness_outcome(
        output, expected=expected, affected_mask=np.ones(len(output), bool), tolerance=0.001, warned=warned
    )
    assert result["outcome"] == outcome


def test_robustness_outcome_mixed_case_lists_each_value_and_respects_the_mask() -> None:
    mixed = metrics.robustness_outcome(
        [1.0, nan, 3.0, 99.0],
        expected=[1.0, 2.0, 2.0, 0.0],
        affected_mask=[True, True, True, False],
        tolerance=0.0,
        warned=False,
        warning_required=True,
    )
    assert mixed["values"] == ["correct", "NaN silent", "silently wrong"]
    assert mixed["missing_required_warning"]
    correct = metrics.robustness_outcome(
        [99.0, 1.0], expected=1.0, affected_mask=[False, True], tolerance=0.0, warned=False, warning_required=True
    )
    assert correct["outcome"] == "correct"
    assert correct["missing_required_warning"]  # reported beside the outcome, never instead of it
    with pytest.raises(ValueError, match="affected"):
        metrics.robustness_outcome([1.0], expected=1.0, affected_mask=[False], tolerance=0.0, warned=False)


def test_robustness_outcome_of_an_exception() -> None:
    kwargs: dict[str, Any] = {"expected": 1.0, "affected_mask": [True], "tolerance": 0.0, "warned": False}
    result = metrics.robustness_outcome(None, exception=ValueError("too few values"), error_clear=True, **kwargs)
    assert (result["outcome"], result["error"]) == ("error", "clear")
    assert metrics.robustness_outcome(None, exception=RuntimeError(), error_clear=False, **kwargs)["error"] == "unclear"
    with pytest.raises(ValueError, match="error_clear"):
        metrics.robustness_outcome(None, exception=RuntimeError(), **kwargs)


# Units and sample size


def test_across_unit_summary() -> None:
    assert metrics.across_unit_summary([1.0, 2.0, 3.0, nan]) == {
        "median": 2.0,
        "p5": approx(1.1),
        "p95": approx(2.9),
        "n_units": 4,
        "n_units_defined": 3,
    }


def test_median_interval_reproduces_its_stream_with_independent_code() -> None:
    values, seed, row, draws = [3.0, 1.0, nan, 2.0, 5.0, 4.0], 11, 4, 500
    defined = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    index = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 2, row]))).integers(0, 5, size=(draws, 5))
    expected = np.percentile(np.median(defined[index], axis=1), [2.5, 97.5])
    np.testing.assert_allclose(metrics.median_interval(values, seed=seed, row=row, draws=draws), expected, rtol=1e-12)
    assert metrics.median_interval([2.0, 2.0, 2.0], seed=seed, row=row) == (2.0, 2.0)
    assert all(math.isnan(v) for v in metrics.median_interval([1.0, nan], seed=seed, row=row))


def test_block_interval_reproduces_its_stream_with_independent_code() -> None:
    a, b, length, seed, unit, draws = np.arange(7.0), np.zeros(7), 3, 5, 2, 200
    starts = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 3, unit]))).integers(
        0, 5, size=(draws, 3)
    )
    index = (starts[:, :, None] + np.arange(3)).reshape(draws, 9)[:, :7]
    expected = np.percentile(a[index].mean(axis=1), [2.5, 97.5])

    def bias(x: np.ndarray, y: np.ndarray) -> float:
        return float(metrics.pair_agreement(x, y)["bias"])

    result = metrics.block_interval(a, b, bias, block_length=length, seed=seed, unit_index=unit, draws=draws)
    np.testing.assert_allclose([result["lower"], result["upper"]], expected, rtol=1e-12)
    assert (result["skipped"], result["n_finite"]) == (False, draws)
    same = metrics.block_interval(a, a, bias, block_length=length, seed=seed, unit_index=unit, draws=draws)
    assert (same["lower"], same["upper"]) == (0.0, 0.0)
    skipped = metrics.block_interval(a[:3], b[:3], bias, block_length=3, seed=seed, unit_index=unit)
    assert skipped["skipped"]
    assert math.isnan(skipped["lower"])


def test_block_interval_counts_resamples_whose_statistic_is_undefined() -> None:
    a = np.r_[nan, nan, nan, np.arange(5.0)]

    def r(x: np.ndarray, y: np.ndarray) -> float:
        return float(metrics.pair_agreement(x, y)["r"])

    result = metrics.block_interval(a, a, r, block_length=3, seed=1, unit_index=0, draws=300)
    assert 0 < result["n_finite"] < 300  # resamples of only NaN or constant blocks have no r


# Comparison ladder


def test_rung_effects_use_the_steps_finite_at_every_rung() -> None:
    b = np.zeros(4)
    rungs = [([1.0, 1.0, nan, 1.0], b), ([0.5, 0.5, 0.5, nan], b), ([0.0, 0.0, 0.0, 0.0], b)]

    def mae(x: np.ndarray, y: np.ndarray) -> float:
        return float(metrics.pair_agreement(x, y)["mae"])

    result = metrics.rung_effects(rungs, mae)
    assert result == {"values": [1.0, 0.5, 0.0], "effects": [-0.5, -0.5], "n_common": 2}
    with pytest.raises(ValueError, match="time axis"):
        metrics.rung_effects([([0.0], [0.0]), ([0.0, 1.0], [0.0, 1.0])], mae)


def test_rung_categories_and_transitions() -> None:
    result = metrics.rung_categories([([-1.0, 0.0], [0.0, 0.0]), ([1e-4, 0.0], [0.0, 0.0])])
    assert result["classes"] == ["category-changing", "below reporting precision"]
    assert result["mask_mismatch"] == [0.0, 0.0]
    delta = result["confusion_deltas"][0]
    assert (delta[5, 3], delta[5, 5], int(np.abs(delta).sum())) == (-1, 1, 2)
    counts = metrics.transition_counts([result["classes"], ["category-changing", "category-changing"]])
    assert counts == [
        Counter({("category-changing", "below reporting precision"): 1, ("category-changing", "category-changing"): 1})
    ]
    with pytest.raises(ValueError, match="same rungs"):
        metrics.transition_counts([["a", "b"], ["a"]])


# C7


def test_performance_summary() -> None:
    result = metrics.performance_summary(
        [2.0, 1.0, 4.0], [1024, 2048, 1536], maxrss_unit="KiB", n_series=10, cell_months=120
    )
    assert result["wall_s"] == {"median": 2.0, "min": 1.0, "max": 4.0}
    assert result["peak_rss_bytes"] == {"median": 1536 * 1024, "min": 1024 * 1024, "max": 2048 * 1024}
    assert result["series_per_s"] == {"median": 5.0, "min": 2.5, "max": 10.0}
    assert result["cell_months_per_s"]["median"] == 60.0
    on_macos = metrics.performance_summary([1.0, 1.0, 1.0], [100, 200, 300], maxrss_unit="bytes", n_series=1)
    assert on_macos["peak_rss_bytes"]["median"] == 200
    assert "cell_months_per_s" not in on_macos
    with pytest.raises(ValueError, match="3 timed runs"):
        metrics.performance_summary([1.0, 1.0], [1, 1], maxrss_unit="bytes", n_series=1)
    with pytest.raises(ValueError, match="ru_maxrss"):
        metrics.performance_summary([1.0, 1.0, 1.0], [1], maxrss_unit="bytes", n_series=1)


# Properties


@st.composite
def pairs(draw: Any) -> tuple[np.ndarray, np.ndarray]:
    n = draw(st.integers(1, 30))
    values = st.one_of(st.floats(-4, 4), st.sampled_from([nan, -0.5, 0.5, -1.3, -2.0]))
    return draw(arrays(np.float64, n, elements=values)), draw(arrays(np.float64, n, elements=values))


@settings(max_examples=60, deadline=None)
@given(pairs())
def test_pair_measures_are_symmetric(pair: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = pair
    forward, backward = metrics.pair_agreement(a, b), metrics.pair_agreement(b, a)
    for name in ("mae", "rmse", "max_abs", "r"):
        assert forward[name] == pytest.approx(backward[name], nan_ok=True)
    assert forward["bias"] == pytest.approx(-backward["bias"], nan_ok=True)
    assert metrics.difference_class(a, b) == metrics.difference_class(b, a)
    assert metrics.category_agreement(a, b) == metrics.category_agreement(b, a)
    assert metrics.mask_mismatch(a, b) == metrics.mask_mismatch(b, a)
    np.testing.assert_array_equal(metrics.confusion_matrix(a, b), metrics.confusion_matrix(b, a).T)


@settings(max_examples=60, deadline=None)
@given(pairs(), st.randoms(use_true_random=False))
def test_pair_measures_ignore_time_order(pair: tuple[np.ndarray, np.ndarray], random: Any) -> None:
    a, b = pair
    order = list(range(a.size))
    random.shuffle(order)
    shuffled = metrics.pair_agreement(a[order], b[order])
    for name, value in metrics.pair_agreement(a, b).items():
        assert shuffled[name] == pytest.approx(value, rel=1e-9, abs=1e-12, nan_ok=True)
    assert metrics.difference_class(a[order], b[order]) == metrics.difference_class(a, b)
    np.testing.assert_array_equal(metrics.confusion_matrix(a[order], b[order]), metrics.confusion_matrix(a, b))


@settings(max_examples=60, deadline=None)
@given(pairs())
def test_a_series_agrees_with_itself(pair: tuple[np.ndarray, np.ndarray]) -> None:
    a = pair[0]
    if not np.isfinite(a).any():
        assert metrics.difference_class(a, a)["class"] == "no overlap"
        return
    agreement = metrics.pair_agreement(a, a)
    assert (agreement["bias"], agreement["max_abs"]) == (0.0, 0.0)
    assert metrics.difference_class(a, a)["class"] == "numerically identical"
    assert metrics.category_agreement(a, a)["same_class"] == 1.0
    assert metrics.mask_mismatch(a, a) == 0.0


# Independence labels and the results document


def test_independence_class_reads_lineage_in_either_order() -> None:
    lineage = json.loads((ROOT / "docs/research/reference-papers/lineage.json").read_text(encoding="utf-8"))
    label = metrics.independence_class(lineage, "SPI", "climate_indices", "xclim")
    assert label in lineage["classes"]
    assert metrics.independence_class(lineage, "SPI", "xclim", "climate_indices") == label
    with pytest.raises(KeyError):
        metrics.independence_class(lineage, "SPI", "climate_indices", "not_a_package")


def _document(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_version": metrics.PROTOCOL_VERSION,
        "study": "synthetic",
        "created": "2026-11-27",
        "harness": {"git_commit": "0" * 40, "git_dirty": False},
        "preregistration": None,
        "manifest_sha256": "a" * 64,
        "records": records,
    }


def test_metrics_document_round_trips_with_nan_as_null(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    record = {
        "row": 4,
        "measure": "pair agreement",
        "implementations": ["climate_indices", "xclim"],
        "independence": "validated against",
        "rung": "L0",
        "unit": "0101",
        "options": {},
        "values": metrics.pair_agreement([1.0], [np.float32(2.0)]),  # r is NaN
    }
    confusion = metrics.record(
        8, metrics.confusion_matrix(A, B), implementations=["climate_indices", "xclim"], unit="0101"
    )
    path = tmp_path / "metrics.json"
    metrics.write_metrics(path, _document([record, confusion]))
    stored = metrics.read_metrics(path)
    assert stored["records"][0]["values"]["r"] is None
    assert stored["records"][0]["values"]["bias"] == -1.0
    assert stored["records"][1]["values"]["value"][5][5] == 3


def test_metrics_document_rejects_a_bad_header_or_record(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    record = {"row": 9, "measure": "mask mismatch", "implementations": ["a", "b"], "unit": "x", "values": {}}
    bad_header = _document([record])
    del bad_header["harness"]
    bad_measure = _document([{**record, "measure": "rank correlation"}])  # not adopted in protocol 1.0.0
    for document in (bad_header, bad_measure):
        with pytest.raises(jsonschema.ValidationError):
            metrics.write_metrics(tmp_path / "metrics.json", document)
    assert not (tmp_path / "metrics.json").exists()


def test_every_measure_output_is_a_valid_record(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    a, b = A[np.isfinite(A)], B[np.isfinite(A)]
    band = metrics.null_band(12, 0.0, seed=1, draws=200)
    outputs = {
        1: metrics.printed_value_pass(1.255, 1.26, 2),
        2: metrics.error_by_band(a, b),
        3: metrics.replicate_summary([1.0, nan]),
        4: metrics.pair_agreement(a, b),
        5: metrics.difference_class(a, b),
        6: metrics.class_change_share(a, b),
        7: metrics.category_agreement(a, b),
        8: metrics.confusion_matrix(a, b),
        9: metrics.mask_mismatch(A, B),
        10: metrics.period_moments(np.arange(12.0)),
        11: metrics.quantile_residual(np.arange(12.0)),
        12: band,
        13: metrics.outside_band_share([0.0], [band], "mean"),
        14: metrics.output_bound_share(a, None),
        15: metrics.robustness_outcome([1.0], expected=1.0, affected_mask=[True], tolerance=0.0, warned=False),
        16: metrics.across_unit_summary([1.0, 2.0]),
        17: metrics.median_interval([1.0, 2.0, 3.0], seed=1, row=4, draws=50),
        18: metrics.block_interval(a, b, lambda x, y: 0.0, block_length=2, seed=1, unit_index=0, draws=20),
        19: metrics.rung_categories([(a, b), (a, a)]),
        20: metrics.performance_summary([1.0, 2.0, 3.0], [1, 2, 3], maxrss_unit="KiB", n_series=1),
    }
    records = [metrics.record(row, value, implementations=["a", "b"], unit="u") for row, value in outputs.items()]
    path = tmp_path / "metrics.json"
    metrics.write_metrics(path, _document(records))
    stored = metrics.read_metrics(path)["records"]
    assert [entry["measure"] for entry in stored] == list(metrics.MEASURES)
    assert stored[0]["values"] == {"value": True}
    assert stored[13]["values"] == {"value": "not applicable"}


def test_record_rejects_a_measure_that_is_not_its_row(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    mislabelled = {**metrics.record(4, {"bias": 0.0}, implementations=["a"], unit="u"), "measure": "confusion matrix"}
    unwrapped = {**metrics.record(9, 0.25, implementations=["a"], unit="u"), "values": {"share": 0.25}}
    for entry in (mislabelled, unwrapped):
        with pytest.raises(jsonschema.ValidationError):
            metrics.write_metrics(tmp_path / "metrics.json", _document([entry]))
    with pytest.raises(ValueError, match="row"):
        metrics.record(21, {}, implementations=["a"], unit="u")


def test_schema_names_every_measure_of_the_protocol_in_row_order() -> None:
    protocol = (ROOT / "docs/research/reference-papers/protocol.md").read_text(encoding="utf-8")
    table = protocol.split("## Measure reference", 1)[1].split("\n## ", 1)[0]
    rows = [(int(n), name) for n, name in re.findall(r"^\| (\d+) \| ([^|]+?) \|", table, flags=re.MULTILINE)]
    assert [n for n, _ in rows] == list(range(1, 21))
    schema = json.loads(metrics.SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["$defs"]["record"]["properties"]["measure"]["enum"] == [name for _, name in rows]
    assert list(metrics.MEASURES) == [name for _, name in rows]
    assert metrics.PROTOCOL_VERSION in protocol.split("\n", 5)[3]  # the "Version:" line


# The Earth Engine pilot through the library


def _pilot_like_arrays(seed: int) -> tuple[np.ndarray, np.ndarray]:
    # rounding to one decimal puts many values exactly on the class edges
    generator = np.random.default_rng(seed)
    ours = np.round(generator.normal(size=(24, 4, 3)), 1)
    theirs = np.round(ours + generator.normal(scale=0.3, size=ours.shape), 1).astype(np.float32)
    ours[0, 0, 0] = theirs[1, 1, 1] = nan
    return ours, theirs


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_library_reproduces_the_pilot_comparison_metrics(seed: int) -> None:
    ours, theirs = _pilot_like_arrays(seed)
    expected = pilot.comparison_metrics(ours, theirs)
    got = pilot.library_metrics(ours, theirs)
    # the protocol's figures; the pilot also reports means and a normalized RMSE, which are not measures
    figures = (
        "n",
        "bias",
        "mae",
        "rmse",
        "correlation",
        "max_abs_difference",
        "category_agreement",
        "category_within_one",
    )
    assert set(got) == set(figures)
    for name, value in got.items():
        assert value == pytest.approx(expected[name], rel=REL, abs=1e-15), name


def _pilot_inputs(seed: int) -> tuple[dict[str, xr.DataArray], xr.Dataset]:
    ours, theirs = _pilot_like_arrays(seed)
    coords = {"time": xr.date_range("1990-01-01", periods=ours.shape[0], freq="MS"), "y": range(4), "x": range(3)}
    dims = ("time", "y", "x")
    return {"spi_3": xr.DataArray(ours, coords, dims)}, xr.Dataset({"spi90d": xr.DataArray(theirs, coords, dims)})


def test_check_metrics_library_passes_and_reports_drift(capsys: pytest.CaptureFixture[str]) -> None:
    ours, reference = _pilot_inputs(0)
    assert pilot.check_metrics_library(ours, reference, {"spi_3": {"bias": 0.0}}) == 0
    output = capsys.readouterr().out
    assert "spi_3 bias" in output
    assert "committed 0.0" in output
    assert "MISMATCH" not in output


def test_check_metrics_library_fails_when_the_library_disagrees(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ours, reference = _pilot_inputs(0)
    real = pilot.library_metrics
    monkeypatch.setattr(pilot, "library_metrics", lambda a, b: {**real(a, b), "rmse": 9.0})
    assert pilot.check_metrics_library(ours, reference, {}) == 1
    assert "rmse: library 9.0" in capsys.readouterr().out


def test_check_metrics_library_refuses_the_tukey_followup() -> None:
    with pytest.raises(SystemExit):
        pilot.main(["--check-metrics-library", "--spi-spei-position", "tukey"])
