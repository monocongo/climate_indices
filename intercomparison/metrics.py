"""Measures of the evaluation protocol (REFIMPL-16, issue 1618).

One function per row of the Measure reference in
``docs/research/reference-papers/protocol.md`` (version ``PROTOCOL_VERSION``); each function's
docstring names its row. The protocol is the specification: where a docstring and the protocol
differ, the protocol governs.

Conventions applied everywhere (protocol, Conventions):

* A series is a 1-D array along time. Integer and boolean input is cast to ``float64``; object
  arrays are rejected; a pair of different lengths raises. A float32 series stays float32 for
  classification (the class edges are cast to its width) and is widened to ``float64`` for
  every difference.
* A time step enters a pair measure only when both values are finite. NaN, masked and infinite
  values are not finite.
* An undefined measure is NaN. ``write_metrics`` stores NaN as JSON ``null``.
* Random draws come from ``Generator(PCG64(SeedSequence([seed, stream, key])))``.
* Percentiles use linear interpolation.

Category codes ascend toward wet on both scales, so codes 0 to 4 are the drought classes
(D4 to D0 on the six-class scale) and ``5 - code`` is the code of
``climate_indices.suitability.drought_category``.
"""

from __future__ import annotations

import itertools
import math
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.special import ndtri

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract  # noqa: E402

PROTOCOL_VERSION = "1.0.0"
SCHEMA_VERSION = 1
SCHEMA_PATH = Path(__file__).with_name("metrics.schema.json")

SIX_CLASS_EDGES = (-2.0, -1.6, -1.3, -0.8, -0.5)
ELEVEN_CLASS_EDGES = (-2.0, -1.6, -1.3, -0.8, -0.5, 0.5, 0.8, 1.3, 1.6, 2.0)
SCALES = {"six": SIX_CLASS_EDGES, "eleven": ELEVEN_CLASS_EDGES}
DROUGHT_CODES = 5  # codes below this are drought classes on both scales
Scale = Literal["six", "eleven"]
ScaleOrEdges = Scale | Sequence[float]  # a named scale, or the class edges a study pre-registered
Rule = Literal["upper_inclusive", "lower_inclusive"]
_SIDE: dict[str, Literal["left", "right"]] = {"upper_inclusive": "left", "lower_inclusive": "right"}

MEASURES = (  # the Measure reference, in row order; row n is MEASURES[n - 1]
    "printed-value pass",
    "error, by band",
    "replicate summary",
    "pair agreement",
    "difference class",
    "class-change share",
    "same-class and within-one share",
    "confusion matrix",
    "mask mismatch",
    "calendar-period moments",
    "quantile residual",
    "null band",
    "outside-band share",
    "output-bound share",
    "robustness outcome",
    "across-unit summary",
    "bootstrap interval of a median",
    "block bootstrap interval",
    "rung effect",
    "performance",
)

DIFFERENCE_CLASSES = (
    "numerically identical",
    "equivalent",
    "below reporting precision",
    "category-preserving",
    "category-changing",
)
ABOVE_REPORTING_PRECISION = "above reporting precision"
NO_OVERLAP = "no overlap"
NOT_APPLICABLE = "not applicable"
_MAGNITUDE_CLASSES = ((1e-10, DIFFERENCE_CLASSES[0]), (1e-6, DIFFERENCE_CLASSES[1]), (0.005, DIFFERENCE_CLASSES[2]))

OUTCOMES = (  # most severe first
    "error",
    "silently wrong",
    "wrong with warning",
    "NaN silent",
    "NaN with warning",
    "correct",
)

STREAM_NULL_BAND = 1
STREAM_UNIT_BOOTSTRAP = 2
STREAM_BLOCK_BOOTSTRAP = 3
STREAM_SYNTHETIC = 4
N_DRAWS = 10_000
_CHUNK_ELEMENTS = 1 << 22  # bootstrap indices drawn per chunk: about 100 MB of work arrays at a time
MIN_PERIOD_VALUES = 10
MIN_BAND_OBSERVATIONS = 30
_LAST_UNIFORM = math.nextafter(1.0, 0.0)
_MOMENTS = ("mean", "sd", "skewness", "excess_kurtosis")
BAND_STATISTICS = (*_MOMENTS, "residual")


# Shared helpers


def series(values: Any) -> np.ndarray:
    """Return a 1-D float series: masked values become NaN, int and bool become float64."""
    if np.ma.isMaskedArray(values):
        x = np.ma.filled(values if np.issubdtype(values.dtype, np.floating) else values.astype(np.float64), np.nan)
    else:
        x = np.asarray(values)
    if x.dtype == object:
        raise TypeError("object arrays are not accepted; pass numbers")
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D series, got shape {x.shape}")
    return x if np.issubdtype(x.dtype, np.floating) else x.astype(np.float64)


def _pair(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xa, xb = series(a), series(b)
    if xa.shape != xb.shape:
        raise ValueError(f"length mismatch: {xa.shape[0]} versus {xb.shape[0]}")
    return xa, xb, np.isfinite(xa) & np.isfinite(xb)


def _differences(xa: np.ndarray, xb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    d: np.ndarray = xa[mask].astype(np.float64) - xb[mask].astype(np.float64)
    return d


def within_tolerance(error: float, tolerance: float) -> bool:
    """Compare a nonnegative error with its tolerance, allowing binary64 representation error."""
    return bool(error <= tolerance * (1 + 1e-9))


def rng(seed: int, stream: int, key: int) -> np.random.Generator:
    """The protocol's random stream ``(stream, key)`` for a pre-registered ``seed``."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream, key])))


def _percentiles(values: np.ndarray, q: Sequence[float]) -> list[float]:
    if values.size == 0:
        return [math.nan] * len(q)
    return [float(v) for v in np.percentile(values, q, method="linear")]


def classify(values: Any, edges: Sequence[float], rule: Rule = "upper_inclusive") -> np.ndarray:
    """Class codes of a series; -1 marks a value that is not finite.

    The edges are cast to the series' own width, as ``suitability.drought_category`` does, so a
    float32 -1.3 sits on the edge. ``upper_inclusive`` (the protocol's rule) puts exactly -0.5 in
    D0; ``lower_inclusive`` reproduces ``np.digitize(..., right=False)`` of the Earth Engine pilot.
    """
    if rule not in _SIDE:
        raise ValueError(f"unknown boundary rule {rule!r}; use one of {tuple(_SIDE)}")
    x = series(values)
    codes: np.ndarray = np.searchsorted(np.asarray(edges, dtype=x.dtype), x, side=_SIDE[rule]).astype(np.int64)
    codes[~np.isfinite(x)] = -1
    return codes


def class_edges(scale: ScaleOrEdges) -> tuple[float, ...]:
    """The edges of a named scale, or a study's own edges checked to be strictly increasing.

    A study of an index that is not normal-scale (PDSI, for example) passes the class edges it
    fixed in its pre-registration (protocol, Conventions and Difference classes).
    """
    if isinstance(scale, str):
        if scale not in SCALES:
            raise ValueError(f"unknown scale {scale!r}; use one of {tuple(SCALES)} or a sequence of class edges")
        return SCALES[scale]
    edges = tuple(float(edge) for edge in scale)
    if not edges or not all(math.isfinite(e) for e in edges) or any(lo >= hi for lo, hi in itertools.pairwise(edges)):
        raise ValueError(f"class edges must be finite and strictly increasing, got {edges}")
    return edges


def _pair_codes(a: Any, b: Any, scale: ScaleOrEdges, rule: Rule) -> tuple[np.ndarray, np.ndarray]:
    xa, xb, mask = _pair(a, b)
    edges = class_edges(scale)
    return classify(xa, edges, rule)[mask], classify(xb, edges, rule)[mask]


# C1, C2


def printed_value_pass(computed: float, printed: float, k: int, tolerance: float | None = None) -> bool:
    """Row 1: does a computed value reproduce a value printed with ``k`` decimals?"""
    if k < 0:
        raise ValueError("k counts printed decimals and cannot be negative")
    if not math.isfinite(computed):
        return False
    return within_tolerance(abs(computed - printed), 0.5 * 10.0**-k if tolerance is None else tolerance)


def _error_stats(e: np.ndarray) -> dict[str, float]:
    if e.size == 0:
        return {"bias": math.nan, "rmse": math.nan, "max_abs_error": math.nan, "n": 0}
    return {
        "bias": float(np.mean(e)),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "max_abs_error": float(np.max(np.abs(e))),
        "n": int(e.size),
    }


def error_by_band(
    index: Any,
    truth: Any,
    *,
    exclude: Any = None,
    band_edges: Sequence[float] = (0.0, 1.0, 2.0),
    min_count: int = MIN_BAND_OBSERVATIONS,
) -> dict[str, Any]:
    """Row 2: signed error ``index - truth`` overall and within bands of ``|truth|``.

    ``exclude`` marks observations left out of C2 by rule (observations equal to zero). A truth
    value that is not finite is excluded and counted. A band with fewer than ``min_count``
    observations is empty: its statistics are NaN.
    """
    z, z_true, _ = _pair(index, truth)
    excluded = np.zeros(z.shape, bool) if exclude is None else np.asarray(exclude, bool)
    if excluded.shape != z.shape:
        raise ValueError("exclude must match the series length")
    truth_finite = np.isfinite(z_true)
    scored = truth_finite & ~excluded & np.isfinite(z)
    e = _differences(z, z_true, scored)
    magnitude = np.abs(z_true[scored].astype(np.float64))
    uppers = (*band_edges[1:], math.inf)
    bands = []
    for lower, upper in zip(band_edges, uppers, strict=True):
        stats = _error_stats(e[(magnitude >= lower) & (magnitude < upper)])
        if stats["n"] < min_count:
            stats.update(bias=math.nan, rmse=math.nan, max_abs_error=math.nan)
        bands.append({"lower": lower, "upper": upper, **stats})
    return {
        "overall": _error_stats(e),
        "bands": bands,
        "n_truth_not_finite": int(np.count_nonzero(~truth_finite)),
        "n_excluded": int(np.count_nonzero(excluded & truth_finite)),
        "n_index_not_finite": int(np.count_nonzero(truth_finite & ~excluded & ~np.isfinite(z))),
    }


def replicate_summary(values: Any) -> dict[str, float]:
    """Row 3: median and 2.5th and 97.5th percentiles over the non-empty replicates."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    median, low, high = _percentiles(v, (50, 2.5, 97.5))
    return {"median": median, "p2_5": low, "p97_5": high, "n_non_empty": int(v.size)}


# C3 and the difference classes


def pair_agreement(a: Any, b: Any) -> dict[str, float]:
    """Row 4: bias ``mean(a - b)``, MAE, RMSE, Pearson ``r`` and ``max|a - b|`` on finite pairs."""
    xa, xb, mask = _pair(a, b)
    d = _differences(xa, xb, mask)
    if d.size == 0:
        return {"bias": math.nan, "mae": math.nan, "rmse": math.nan, "r": math.nan, "max_abs": math.nan, "n": 0}
    pa, pb = xa[mask].astype(np.float64), xb[mask].astype(np.float64)
    # r is undefined for fewer than 2 pairs or a series with zero variance
    defined = d.size >= 2 and np.ptp(pa) > 0 and np.ptp(pb) > 0
    return {
        "bias": float(np.mean(d)),
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d**2))),
        "r": float(np.corrcoef(pa, pb)[0, 1]) if defined else math.nan,
        "max_abs": float(np.max(np.abs(d))),
        "n": int(d.size),
    }


def difference_class(
    a: Any, b: Any, *, edges: Sequence[float] | None = SIX_CLASS_EDGES, rule: Rule = "upper_inclusive"
) -> dict[str, Any]:
    """Row 5: the pair's difference class and ``d = max|a - b|`` over finite pairs.

    ``edges=None`` is for an index with no class edges: a pair above 0.005 is then
    *above reporting precision*. A study of a non-normal index passes the edges it pre-registered.
    """
    xa, xb, mask = _pair(a, b)
    if not mask.any():
        return {"class": NO_OVERLAP, "d": math.nan}
    d = float(np.max(np.abs(_differences(xa, xb, mask))))
    for threshold, name in _MAGNITUDE_CLASSES:
        if within_tolerance(d, threshold):
            return {"class": name, "d": d}
    if edges is None:
        return {"class": ABOVE_REPORTING_PRECISION, "d": d}
    same = np.array_equal(classify(xa, edges, rule)[mask], classify(xb, edges, rule)[mask])
    return {"class": DIFFERENCE_CLASSES[3] if same else DIFFERENCE_CLASSES[4], "d": d}


def class_change_share(
    a: Any, b: Any, *, scale: ScaleOrEdges = "six", rule: Rule = "upper_inclusive", drought_codes: int | None = None
) -> dict[str, float]:
    """Row 6: share of finite pairs that change class, and the share of those changes that start
    or end in a drought class.

    Codes below ``drought_codes`` are the drought classes: 5 on both named scales. A study that
    passes its own class edges also passes how many of its lowest classes are drought classes.
    """
    if drought_codes is None:
        if not isinstance(scale, str):
            raise ValueError("pass drought_codes with pre-registered class edges")
        drought_codes = DROUGHT_CODES
    ca, cb = _pair_codes(a, b, scale, rule)
    changed = ca != cb
    n_changed = int(np.count_nonzero(changed))
    touching = changed & ((ca < drought_codes) | (cb < drought_codes))
    return {
        "share": n_changed / ca.size if ca.size else math.nan,
        "share_touching_drought": np.count_nonzero(touching) / n_changed if n_changed else math.nan,
        "n": int(ca.size),
    }


def category_agreement(
    a: Any, b: Any, *, scale: ScaleOrEdges = "six", rule: Rule = "upper_inclusive"
) -> dict[str, float]:
    """Row 7: same-class share and within-one-class share of the finite pairs."""
    ca, cb = _pair_codes(a, b, scale, rule)
    if ca.size == 0:
        return {"same_class": math.nan, "within_one": math.nan, "n": 0}
    return {
        "same_class": float(np.mean(ca == cb)),
        "within_one": float(np.mean(np.abs(ca - cb) <= 1)),
        "n": int(ca.size),
    }


def confusion_matrix(a: Any, b: Any, *, scale: ScaleOrEdges = "six", rule: Rule = "upper_inclusive") -> np.ndarray:
    """Row 8: class counts with rows the product ``b`` and columns the package ``a``."""
    ca, cb = _pair_codes(a, b, scale, rule)
    k = len(class_edges(scale)) + 1
    return np.bincount(cb * k + ca, minlength=k * k).reshape(k, k)


def mask_mismatch(a: Any, b: Any) -> float:
    """Row 9: share of time steps finite in exactly one series."""
    xa, xb, _ = _pair(a, b)
    if xa.size == 0:
        return math.nan
    return float(np.mean(np.isfinite(xa) != np.isfinite(xb)))


# C4


def _finite_float64(values: Any) -> np.ndarray:
    x = series(values).astype(np.float64)
    return x[np.isfinite(x)]


def _moments(x: np.ndarray) -> dict[str, np.ndarray]:
    """Moments along the last axis; skewness and kurtosis are NaN where the values are constant."""
    n = x.shape[-1]
    constant = np.ptp(x, axis=-1) == 0
    # a constant period has m2 = 0 exactly (the protocol's case), though its float mean can miss x
    mean = np.where(constant, x[..., 0], x.mean(axis=-1))
    dev = x - mean[..., None]
    m2 = np.mean(dev**2, axis=-1)
    safe = np.where(constant, 1.0, m2)
    return {
        "mean": mean,
        "sd": np.sqrt(np.sum(dev**2, axis=-1) / (n - 1)) if n > 1 else np.full(mean.shape, math.nan),
        "skewness": np.where(constant, math.nan, np.mean(dev**3, axis=-1) / safe**1.5),
        "excess_kurtosis": np.where(constant, math.nan, np.mean(dev**4, axis=-1) / safe**2 - 3),
    }


def _residual(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1]
    quantiles = ndtri((np.arange(1, n + 1) - 0.5) / n)
    residual: np.ndarray = np.max(np.abs(np.sort(x, axis=-1) - quantiles), axis=-1)
    return residual


def period_moments(values: Any, *, min_count: int = MIN_PERIOD_VALUES) -> dict[str, Any]:
    """Row 10: mean, standard deviation (``ddof=1``), skewness ``m3 / m2**1.5`` and excess
    kurtosis ``m4 / m2**2 - 3`` of one calendar period's finite values.

    A period with fewer than ``min_count`` finite values is too short and every moment is NaN.
    """
    x = _finite_float64(values)
    if x.size < min_count:
        return {**dict.fromkeys(_MOMENTS, math.nan), "n": int(x.size), "too_short": True}
    return {**{k: float(v) for k, v in _moments(x).items()}, "n": int(x.size), "too_short": False}


def quantile_residual(values: Any, *, min_count: int = MIN_PERIOD_VALUES) -> float:
    """Row 11: ``max_i |s_(i) - Phi^-1((i - 0.5) / n)|`` over the sorted finite values; NaN when
    the period is too short."""
    x = _finite_float64(values)
    return math.nan if x.size < min_count else float(_residual(x))


def null_samples(n: int, p0: float, zero_score: float | None, *, seed: int, draws: int = N_DRAWS) -> np.ndarray:
    """The ``(draws, n)`` null sample of row 12, drawn from stream 1 with key ``n``."""
    generator = rng(seed, STREAM_NULL_BAND, n)
    if p0 == 0:
        return generator.standard_normal(size=(draws, n))
    if zero_score is None:
        raise ValueError("a zero mass needs a zero score")
    k = generator.binomial(n, p0, size=draws)
    u = 1 - generator.random(size=(draws, n))
    u[u == 1] = _LAST_UNIFORM
    scores = ndtri(np.minimum(p0 + (1 - p0) * u, _LAST_UNIFORM))
    # the first k values of a row are the zero score, the rest use the row's first n - k uniforms
    column = np.arange(n)
    shifted = np.take_along_axis(scores, np.clip(column - k[:, None], 0, n - 1), axis=1)
    return np.where(column < k[:, None], zero_score, shifted)


def null_band(
    n: int,
    p0: float | None,
    *,
    seed: int,
    zero_handling: str | None = None,
    zero_score: float | None = None,
    draws: int = N_DRAWS,
) -> dict[str, Any]:
    """Row 12: 2.5th and 97.5th percentile limits of each C4 statistic under the null.

    No band is available when ``p0`` is unknown (``None`` or NaN), or when ``p0 > 0`` and the
    zero-handling rule gives no single finite zero score. ``p0 = 1`` (an all-zero period) has a
    band: every draw is the zero score.
    """
    key = {"n": n, "p0": p0, "zero_handling": zero_handling, "zero_score": zero_score}
    if p0 is None or math.isnan(p0):
        return {**key, "available": False, "reason": "p0 is unknown"}
    if not 0 <= p0 <= 1:
        raise ValueError(f"p0 must be in [0, 1], got {p0}")
    if p0 > 0 and (zero_score is None or not math.isfinite(zero_score)):
        return {**key, "available": False, "reason": "no single finite zero score"}
    samples = null_samples(n, p0, zero_score, seed=seed, draws=draws)
    stats = {**_moments(samples), "residual": _residual(samples)}
    limits = {}
    for name, values in stats.items():
        finite = values[np.isfinite(values)]
        lower, upper = _percentiles(finite, (2.5, 97.5))
        limits[name] = {"lower": lower, "upper": upper, "n_finite": int(finite.size)}
    return {**key, "available": True, "statistics": limits}


def outside_band_share(
    values: Sequence[float], bands: Sequence[Mapping[str, Any] | None], statistic: str
) -> dict[str, Any]:
    """Row 13: share of scored periods whose ``statistic`` lies outside its matching band.

    ``values[i]`` is period ``i``'s statistic and ``bands[i]`` its band from ``null_band`` (or
    ``None``). A period counts only when its value is finite and its band has finite limits for
    the statistic. The residual is outside only above its upper limit.
    """
    if statistic not in BAND_STATISTICS:
        raise ValueError(f"unknown statistic {statistic!r}")
    if len(values) != len(bands):
        raise ValueError("one band per period")
    scored = outside = 0
    for value, band in zip(values, bands, strict=True):
        limits = band["statistics"].get(statistic) if band and band.get("available") else None
        if not math.isfinite(value) or limits is None or not math.isfinite(limits["upper"]):
            continue
        scored += 1
        outside += value > limits["upper"] or (statistic != "residual" and value < limits["lower"])
    return {"share": outside / scored if scored else math.nan, "n_scored": scored}


def output_bound_share(values: Any, output_bound: float | None) -> float | str:
    """Row 14: share of finite values with ``|z| >= output_bound``; *not applicable* without one.

    The bound is cast to the series' width, as the class edges are: a float32 output clipped at
    3.09 holds float32(3.09), which is below 3.09 in float64.
    """
    if output_bound is None:
        return NOT_APPLICABLE
    x = series(values)
    x = x[np.isfinite(x)]
    return float(np.mean(np.abs(x) >= x.dtype.type(output_bound))) if x.size else math.nan


# C5


def _value_outcome(out: float, expected: float, tolerance: float, warned: bool) -> str:
    if math.isnan(expected) and math.isnan(out):
        return OUTCOMES[5]
    if math.isnan(out):
        return OUTCOMES[4] if warned else OUTCOMES[3]
    if not math.isnan(expected) and math.isfinite(out) and within_tolerance(abs(out - expected), tolerance):
        return OUTCOMES[5]
    return OUTCOMES[2] if warned else OUTCOMES[1]


def robustness_outcome(
    output: Any,
    *,
    expected: Any,
    affected_mask: Any,
    tolerance: float,
    warned: bool,
    warning_required: bool = False,
    exception: BaseException | None = None,
    error_clear: bool | None = None,
) -> dict[str, Any]:
    """Row 15: the C5 outcome of one case, judged on the values at ``affected_mask``.

    ``warned`` says whether a warning or a WARNING log record was captured. For an exception the
    caller judges whether its message names the cause (``error_clear``). A mixed case takes the
    most severe value outcome. A missing required warning is reported separately and never
    changes the outcome.
    """
    missing_warning = warning_required and not warned
    if exception is not None:
        if error_clear is None:
            raise ValueError("an error outcome needs error_clear")
        return {
            "outcome": OUTCOMES[0],
            "error": "clear" if error_clear else "unclear",
            "values": [],
            "missing_required_warning": missing_warning,
        }
    out = series(output).astype(np.float64)
    mask = np.asarray(affected_mask, bool)
    if not mask.any():
        raise ValueError("a case needs at least one affected value")
    wanted = np.broadcast_to(np.asarray(expected, np.float64), out.shape)[mask]
    per_value = [_value_outcome(o, e, tolerance, warned) for o, e in zip(out[mask], wanted, strict=True)]
    return {
        "outcome": min(per_value, key=OUTCOMES.index),
        "error": None,
        "values": per_value,
        "missing_required_warning": missing_warning,
    }


# Units and sample size


def across_unit_summary(values: Any) -> dict[str, float]:
    """Row 16: median, 5th and 95th percentiles over the defined units."""
    v = np.asarray(values, dtype=np.float64)
    defined = v[np.isfinite(v)]
    median, low, high = _percentiles(defined, (50, 5, 95))
    return {"median": median, "p5": low, "p95": high, "n_units": int(v.size), "n_units_defined": int(defined.size)}


def median_interval(values: Any, *, seed: int, row: int, draws: int = N_DRAWS) -> tuple[float, float]:
    """Row 17: 95% percentile bootstrap interval of the across-unit median.

    ``row`` is the Measure reference row of the statistic (for example 4 for RMSE): it keys
    stream 2, so statistics of the same row and sample size reuse the same index draws.
    """
    v = np.asarray(values, dtype=np.float64)
    defined = np.sort(v[np.isfinite(v)])
    if defined.size < 2:
        return (math.nan, math.nan)
    m, generator = defined.size, rng(seed, STREAM_UNIT_BOOTSTRAP, row)
    # drawn in blocks of rows, which gives the same stream as one (draws, m) draw but bounds memory
    # at grid scale (10,000 x 470,000 indices would be 38 GB)
    rows = max(1, _CHUNK_ELEMENTS // m)
    medians = [
        np.median(defined[generator.integers(0, m, size=(min(rows, draws - start), m))], axis=1)
        for start in range(0, draws, rows)
    ]
    low, high = _percentiles(np.concatenate(medians), (2.5, 97.5))
    return (low, high)


def block_interval(
    a: Any,
    b: Any,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    block_length: int,
    seed: int,
    unit_index: int,
    draws: int = N_DRAWS,
) -> dict[str, Any]:
    """Row 18: moving-block bootstrap interval of a pair statistic within one unit.

    ``block_length`` is the number of steps in 12 months, or the accumulation scale if larger.
    Stream 3 is keyed by the unit's zero-based index, so every statistic of a unit reuses the same
    block starts. Both series are resampled with the same indices and ``statistic`` drops the
    pairs that are not finite. Resamples whose statistic is not finite are left out and counted.
    """
    xa, xb, _ = _pair(a, b)
    n, length = xa.size, block_length
    if length < 1:
        raise ValueError("block_length must be at least 1")
    if n <= length:
        return {"lower": math.nan, "upper": math.nan, "skipped": True, "n_finite": 0}
    starts = rng(seed, STREAM_BLOCK_BOOTSTRAP, unit_index).integers(0, n - length + 1, size=(draws, -(-n // length)))
    offsets = np.arange(length)
    # ponytail: one Python call per resample (about 0.5 s per statistic at 10,000 draws); accept a
    # vectorized statistic if a study needs many units. Each row's indices are built as it is used,
    # so memory does not grow with draws x n.
    values = np.array(
        [statistic(xa[i], xb[i]) for i in ((row[:, None] + offsets).ravel()[:n] for row in starts)],
        dtype=np.float64,
    )
    finite = values[np.isfinite(values)]
    low, high = _percentiles(finite, (2.5, 97.5))
    return {"lower": low, "upper": high, "skipped": False, "n_finite": int(finite.size)}


# Comparison ladder


def _rung_pairs(rungs: Sequence[tuple[Any, Any]]) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray]:
    pairs = [_pair(a, b) for a, b in rungs]
    if len({xa.size for xa, _, _ in pairs}) > 1:
        raise ValueError("every rung must have the same time axis")
    common: np.ndarray = np.logical_and.reduce([mask for _, _, mask in pairs])
    return pairs, common


def rung_effects(
    rungs: Sequence[tuple[Any, Any]], measure: Callable[[np.ndarray, np.ndarray], float]
) -> dict[str, Any]:
    """Row 19, scalar measures: ``m(L_k) - m(L_{k-1})`` for one unit.

    ``rungs`` holds the unit's pair at each rung, lowest first. Every rung is measured on the
    time steps finite in both series at every rung.
    """
    pairs, common = _rung_pairs(rungs)
    values = [float(measure(xa[common], xb[common])) for xa, xb, _ in pairs]
    return {
        "values": values,
        "effects": [hi - lo for lo, hi in itertools.pairwise(values)],
        "n_common": int(common.sum()),
    }


def rung_categories(
    rungs: Sequence[tuple[Any, Any]], *, scale: ScaleOrEdges = "six", rule: Rule = "upper_inclusive"
) -> dict[str, Any]:
    """Row 19, categorical measures for one unit: the difference class and the confusion matrix
    at each rung, the cell-wise change of the confusion matrix between adjacent rungs, and the
    mask mismatch at each rung.

    The class and the confusion matrix use the time steps finite in both series at every rung,
    as the scalar effects do, so a confusion delta counts class changes rather than steps that
    pair up at one rung only. The mask mismatch uses each rung's own masks, because common finite
    pairs would hide a mask change (protocol, Comparison ladder). The difference class uses the
    six-class scale for a named scale (protocol, Difference classes) and the study's own edges
    when it passes them.
    """
    pairs, common = _rung_pairs(rungs)
    edges = SCALES["six"] if isinstance(scale, str) else class_edges(scale)
    classes = [difference_class(xa[common], xb[common], edges=edges, rule=rule)["class"] for xa, xb, _ in pairs]
    confusion = [confusion_matrix(xa[common], xb[common], scale=scale, rule=rule) for xa, xb, _ in pairs]
    return {
        "classes": classes,
        "mask_mismatch": [mask_mismatch(xa, xb) for xa, xb, _ in pairs],
        "confusion": confusion,
        "confusion_deltas": [hi - lo for lo, hi in itertools.pairwise(confusion)],
    }


def transition_counts(classes_by_unit: Sequence[Sequence[str]]) -> list[Counter[tuple[str, str]]]:
    """Row 19: for each pair of adjacent rungs, how many units move from one class to another."""
    steps = {len(classes) for classes in classes_by_unit}
    if len(steps) > 1:
        raise ValueError("every unit must have the same rungs")
    n = steps.pop() if steps else 0
    return [Counter((unit[k - 1], unit[k]) for unit in classes_by_unit) for k in range(1, n)]


# C7


def performance_summary(
    wall_s: Sequence[float],
    maxrss: Sequence[int],
    *,
    maxrss_unit: Literal["KiB", "bytes"],
    n_series: int,
    cell_months: int | None = None,
) -> dict[str, Any]:
    """Row 20: median, minimum and maximum of wall time, peak memory and throughput.

    ``maxrss`` is each repetition's raw ``ru_maxrss``: KiB on Linux, bytes on macOS.
    """
    seconds = np.asarray(wall_s, dtype=np.float64)
    if seconds.size < 3:
        raise ValueError("C7 needs at least 3 timed runs")
    if len(maxrss) != seconds.size:
        raise ValueError("one ru_maxrss per timed run")
    scale = {"KiB": 1024, "bytes": 1}[maxrss_unit]
    peak = np.asarray(maxrss, dtype=np.int64) * scale

    def spread(v: np.ndarray) -> dict[str, float]:
        return {"median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v))}

    summary = {
        "wall_s": spread(seconds),
        "peak_rss_bytes": spread(peak),
        "series_per_s": spread(n_series / seconds),
        "n_runs": int(seconds.size),
    }
    if cell_months is not None:
        summary["cell_months_per_s"] = spread(cell_months / seconds)
    return summary


# Independence labels and the results document


def independence_class(lineage: Mapping[str, Any], index: str, a: str, b: str) -> str:
    """The ``lineage.json`` class of a pair (the strongest over all configurations)."""
    for pair in lineage["indices"][index]["pairs"]:
        if {pair["a"], pair["b"]} == {a, b}:
            return str(pair["class"])
    raise KeyError(f"lineage.json has no {index} pair {a} / {b}")


def jsonable(value: Any) -> Any:
    """Convert numpy values, tuples and counters for JSON; NaN becomes ``None``."""
    if isinstance(value, Mapping):
        return {" -> ".join(k) if isinstance(k, tuple) else str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray | list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def record(row: int, values: Any, *, implementations: Sequence[str], unit: str | None, **fields: Any) -> dict[str, Any]:
    """One record of a metrics document for the output ``values`` of Measure reference ``row``.

    A mapping is stored as it is; a single value (a share, a matrix, an interval) is stored as
    ``{"value": values}``. ``fields`` are the optional record fields of ``metrics.schema.json``:
    ``independence``, ``rung``, ``pooled``, ``period_label`` and ``options``.
    """
    if not 1 <= row <= len(MEASURES):
        raise ValueError(f"row must be 1 to {len(MEASURES)}, got {row}")
    stored = dict(values) if isinstance(values, Mapping) else {"value": values}
    return {
        "row": row,
        "measure": MEASURES[row - 1],
        "implementations": list(implementations),
        "unit": unit,
        **fields,
        "values": stored,
    }


def _validate(document: Mapping[str, Any]) -> None:
    import jsonschema  # type: ignore[import-untyped]

    jsonschema.validate(document, contract.read_json(SCHEMA_PATH))
    for i, entry in enumerate(document["records"]):
        if entry["measure"] != MEASURES[entry["row"] - 1]:
            raise jsonschema.ValidationError(
                f"records[{i}]: row {entry['row']} is {MEASURES[entry['row'] - 1]!r}, not {entry['measure']!r}"
            )


def write_metrics(path: Path, document: Mapping[str, Any]) -> None:
    """Validate a metrics document against ``metrics.schema.json`` and write it."""
    converted = jsonable(document)
    _validate(converted)
    contract.write_json(path, converted)


def read_metrics(path: Path) -> dict[str, Any]:
    """Read and validate a metrics document."""
    document = contract.read_json(path)
    _validate(document)
    return document
