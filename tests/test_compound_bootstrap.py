"""Specification-level monthly bootstrap tests; no external reference is claimed."""

from __future__ import annotations

import dataclasses
import itertools
import json
import warnings
from typing import Any
from unittest import mock

import numpy as np
import pytest

from climate_indices import compound
from climate_indices.compound import _bootstrap as implementation
from climate_indices.exceptions import ClimateIndicesWarning, InvalidArgumentError


def prepared(years: int = 60, *, cells: int | None = None) -> compound.AlignedHazards:
    """Use a balanced four-row contingency table in every calendar month."""
    y = np.arange(years)
    rows = np.repeat(np.stack((y % 4 < 2, y % 2 == 0)), 12, axis=1)
    if cells is not None:
        rows = np.repeat(rows[..., None], cells, axis=2)
    time = np.arange("1900-01", np.datetime64("1900-01", "M") + years * 12, dtype="datetime64[M]")
    return compound.AlignedHazards(("A", "B"), time.astype("datetime64[D]"), "monthly", rows, np.ones_like(rows), {})


def table(years: int, n_a: int, n_b: int, n_joint: int) -> compound.AlignedHazards:
    """Repeat one ``years``-long contingency table in every month: ``n_a`` and ``n_b`` marginal and ``n_joint`` joint years."""
    rows = np.zeros((2, years), dtype=bool)
    rows[0, :n_a] = True
    rows[1, n_a - n_joint : n_a - n_joint + n_b] = True
    return dataclasses.replace(prepared(years), exceedance=np.repeat(rows, 12, axis=1))


def run(hazards: compound.AlignedHazards | None = None, **options: Any) -> compound.BootstrapResult:
    return compound.bootstrap_dependence(
        prepared() if hazards is None else hazards,
        **{"statistic": "joint_probability", "seed": 17, "n_resamples": 39, "block_length_years": 4, **options},
    )


@pytest.mark.parametrize(
    "statistic,truth",
    [("joint_probability", 0.25), ("likelihood_multiplication_factor", 1), ("conditional_probability", 0.5)],
)
@pytest.mark.parametrize("groupby", ["month", "pooled"])
def test_counts_and_point_estimates(statistic: str, truth: float, groupby: str) -> None:
    options: dict[str, Any] = {"given": "A", "target": "B"} if statistic == "conditional_probability" else {}
    result = run(statistic=statistic, groupby=groupby, **options)
    np.testing.assert_array_equal(result.estimate, np.full((12,) if groupby == "month" else (), truth))
    np.testing.assert_array_equal(result.estimate, result.full_record_estimate)
    assert result.estimate.dtype == np.float64
    assert result.n_years == 60
    assert result.inference_start == np.datetime64("1900-01-01")
    assert result.inference_end == np.datetime64("1959-12-01")
    assert np.all(result.n_interval_resamples == 39)
    assert np.all(result.n_null_resamples == 39)
    assert np.all((result.p_value > 0) & (result.p_value <= 1))
    assert np.all(result.confidence_low <= result.confidence_high)
    assert np.all(result.quality_flags == 0)


def test_non_circular_blocks_and_exact_tiny_null_enumeration() -> None:
    """Enumerate all four moving-block draws and all 16 independent pairings."""
    possibilities = []
    for starts in itertools.product(range(2), repeat=2):
        rng = mock.Mock()
        rng.integers.return_value = np.array(starts)
        indices = implementation._year_indices(rng, 3, 2)
        expected = np.array([starts[0], starts[0] + 1, starts[1]])
        np.testing.assert_array_equal(indices, expected)
        rng.integers.assert_called_once_with(0, 2, size=2)
        possibilities.append(expected)
    rows = np.repeat(np.array([[0, 1, 1], [1, 0, 1]], dtype=bool)[..., None], 12, axis=2)
    schedule, expected_paired, expected_null = [], [], []
    for b, (left, right) in enumerate(itertools.product(possibilities, repeat=2)):
        paired = possibilities[b % 4]
        schedule.extend((paired, left, right))
        expected_paired.append(sum(rows[0, i, 0] and rows[1, i, 0] for i in paired) / 3)
        a, r = rows[0, left, 0], rows[1, right, 0]
        expected_null.append(sum(x and y for x, y in zip(a, r, strict=True)) / 3 - sum(a) / 3 * sum(r) / 3)
    with mock.patch.object(implementation, "_year_indices", side_effect=schedule):
        paired, null = implementation._resample(
            rows,
            statistic="joint_probability",
            groupby="pooled",
            length=2,
            n_resamples=16,
            seed=0,
            support="tiny enumeration",
        )
    np.testing.assert_array_equal(paired, expected_paired)
    np.testing.assert_allclose(null, expected_null, rtol=0, atol=1e-16)


def test_inclusive_ties_never_produce_zero_p_values() -> None:
    hazards = prepared()
    result = run(dataclasses.replace(hazards, exceedance=np.ones_like(hazards.exceedance)))
    np.testing.assert_array_equal(result.p_value, np.ones(12))
    np.testing.assert_array_equal(result.confidence_low, np.ones(12))
    np.testing.assert_array_equal(result.confidence_high, np.ones(12))


@pytest.mark.parametrize(
    ("alternative", "n_a", "n_b", "n_joint"),
    [("greater", 10, 18, 6), ("less", 12, 25, 10)],
    ids=["float-excess-above-zero", "float-excess-below-zero"],
)
def test_inclusive_ties_survive_float_rounding(alternative: str, n_a: int, n_b: int, n_joint: int) -> None:
    """Each month holds 30 years with ``n_joint * 30 == n_a * n_b``: independence, a mathematical zero."""
    hazards = table(30, n_a, n_b, n_joint)
    # The observed float excess is +/-2.8e-17 rather than 0; a null of exact zeros must still count as tied.
    observed = implementation._metrics(
        hazards.exceedance.reshape(2, 30, 12), np.ones((30, 12), dtype=bool), "joint_probability", "month"
    )[1]
    assert np.all(observed != 0)
    zeros = np.zeros((39, 12))
    with (
        mock.patch.object(implementation, "_resample", return_value=(zeros, zeros)),
        pytest.warns(ClimateIndicesWarning),
    ):
        result = run(hazards, alternative=alternative)
    np.testing.assert_array_equal(result.p_value, np.ones(12))


@pytest.mark.parametrize(
    "statistic", ["joint_probability", "likelihood_multiplication_factor", "conditional_probability"]
)
def test_zero_joint_suppression_has_no_false_zero_width_interval(statistic: str) -> None:
    hazards = prepared()
    rows = hazards.exceedance.copy()
    rows[1] = ~rows[0]
    options = {"given": "A", "target": "B"} if statistic == "conditional_probability" else {}
    with pytest.warns(ClimateIndicesWarning):
        result = run(dataclasses.replace(hazards, exceedance=rows), statistic=statistic, alternative="less", **options)
    assert np.all(result.estimate == 0)
    assert np.isnan(result.confidence_low).all()
    assert np.isnan(result.confidence_high).all()
    assert np.all(result.n_interval_resamples == 0)
    assert np.all(result.n_null_resamples == 39)
    assert np.all(result.p_value == 1 / 40)
    assert np.all(result.quality_flags & implementation._FLAGS["insufficient_successes"])
    assert not np.any(result.quality_flags & implementation._FLAGS["insufficient_expected_count"])


def test_missing_years_partial_boundaries_and_earliest_tie() -> None:
    hazards = prepared(63)
    validity = hazards.valid.copy()
    validity[:, 31 * 12 + 4] = False
    with pytest.warns(ClimateIndicesWarning):
        result = run(dataclasses.replace(hazards, valid=validity))
    assert result.n_years == 31
    assert result.inference_end == np.datetime64("1930-12-01")
    assert np.all(result.estimate == pytest.approx(8 / 31, rel=0, abs=1e-12))
    assert result.full_record_estimate[0] == pytest.approx(16 / 63, rel=0, abs=1e-12)
    # A missing coordinate must split the years too, not join distant months.
    keep = np.arange(len(hazards.time)) != 31 * 12 + 4
    with pytest.warns(ClimateIndicesWarning):
        gapped = run(
            dataclasses.replace(
                hazards, time=hazards.time[keep], exceedance=hazards.exceedance[:, keep], valid=hazards.valid[:, keep]
            )
        )
    np.testing.assert_array_equal(gapped.estimate, result.estimate)
    # Partial first/last years do not join an inference segment.
    trimmed = dataclasses.replace(
        hazards, time=hazards.time[2:-2], exceedance=hazards.exceedance[:, 2:-2], valid=hazards.valid[:, 2:-2]
    )
    partial = run(trimmed)
    assert partial.n_years == 61
    assert partial.inference_start == np.datetime64("1901-01-01")
    assert partial.inference_end == np.datetime64("1961-12-01")


def test_forward_responses_precede_sampling_and_ignore_unselected_source() -> None:
    hazards = prepared(61)
    # Every December anchor responds to the next January; the last December is incomplete.
    rows = np.zeros_like(hazards.exceedance)
    rows[0, 11::12] = True
    rows[1, ::12] = True
    third = np.zeros((1, rows.shape[1]), dtype=bool)
    validity = np.concatenate((hazards.valid, third), axis=0)
    hazards = dataclasses.replace(
        hazards, names=("A", "B", "unused"), exceedance=np.concatenate((rows, third)), valid=validity
    )
    with pytest.warns(ClimateIndicesWarning):
        result = run(hazards, statistic="conditional_probability", given="A", target="B", lag_window=(1, 1))
    assert result.n_years == 60
    assert result.estimate[11] == 1
    assert result.p_value[11] == 1
    assert np.isnan(result.estimate[:11]).all()
    assert result.inference_end == np.datetime64("1959-12-01")
    # Check the derived rows passed into the sampler, not reconstructed joins.
    with mock.patch.object(implementation, "_resample", wraps=implementation._resample) as sampler:
        with pytest.warns(ClimateIndicesWarning):
            run(hazards, statistic="conditional_probability", given="A", target="B", lag_window=(1, 1))
    supplied = sampler.call_args.args[0]
    assert supplied.shape == (2, 60, 12)
    np.testing.assert_array_equal(supplied[0], supplied[1])


def test_cell_tile_and_global_rng_invariance() -> None:
    hazards = prepared(cells=3)
    validity = hazards.valid.copy()
    validity[:, :12, 1] = False
    validity[:, :, 2] = False
    hazards = dataclasses.replace(hazards, valid=validity)
    before = np.random.get_state()
    with pytest.warns(ClimateIndicesWarning):
        together = run(hazards)
        repeated = run(hazards)
    np.testing.assert_array_equal(together.p_value, repeated.p_value)
    for a, b in zip(before, np.random.get_state(), strict=True):
        np.testing.assert_array_equal(a, b)
    for cell in range(3):
        one = dataclasses.replace(hazards, exceedance=hazards.exceedance[..., cell], valid=validity[..., cell])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ClimateIndicesWarning)
            separate = run(one)
        for name in ("estimate", "confidence_low", "confidence_high", "p_value", "quality_flags"):
            np.testing.assert_array_equal(getattr(together, name)[..., cell], getattr(separate, name))
    assert np.isnat(together.inference_start[2])
    assert together.n_years[2] == 0
    assert np.isnan(together.p_value[:, 2]).all()


@pytest.mark.parametrize("missing", [5, 6])
def test_exact_valid_fraction_boundary_and_linear_quantiles(missing: int) -> None:
    values = np.arange(100, dtype=float) / 100
    values[-missing:] = np.nan
    null = values / 10
    with mock.patch.object(implementation, "_resample", return_value=(values, null)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ClimateIndicesWarning)
            result = run(groupby="pooled", n_resamples=100)
    assert result.n_interval_resamples == 100 - missing
    assert result.n_null_resamples == 100 - missing
    if missing == 5:
        np.testing.assert_allclose(
            [result.confidence_low, result.confidence_high],
            np.quantile(values[:-missing], [0.025, 0.975], method="linear"),
            rtol=0,
            atol=1e-12,
        )
        assert result.p_value == (1 + 95) / 96  # observed excess is exactly zero; ties count
    else:
        assert np.isnan(result.confidence_low)
        assert np.isnan(result.p_value)
        assert result.quality_flags & implementation._FLAGS["invalid_interval_resamples"]
        assert result.quality_flags & implementation._FLAGS["invalid_null_resamples"]


@pytest.mark.parametrize(
    "years,length,flag", [(29, 2, "insufficient_years"), (30, 7, "insufficient_blocks"), (60, 7, "few_blocks")]
)
def test_year_and_nonoverlapping_block_guards(years: int, length: int, flag: str) -> None:
    with pytest.warns(ClimateIndicesWarning):
        result = run(prepared(years), block_length_years=length)
    assert np.all(result.quality_flags & implementation._FLAGS[flag])
    if flag != "few_blocks":
        assert np.isnan(result.p_value).all()
    else:
        assert np.isfinite(result.p_value).all()


def test_default_block_length_and_three_hazard_mutual_null() -> None:
    hazards = prepared(125)
    third = hazards.exceedance[0:1].copy()
    hazards = dataclasses.replace(
        hazards,
        names=("A", "B", "C"),
        exceedance=np.concatenate((hazards.exceedance, third)),
        valid=np.ones((3, 1500), dtype=bool),
    )
    result = run(hazards, block_length_years=None)
    assert result.block_length_years == 5
    assert np.all(result.estimate == 32 / 125)
    assert result.metadata["source_names"] == ("A", "B", "C")


@pytest.mark.parametrize(
    ("years", "length", "flag"),
    [(30, 2, "insufficient_years"), (30, 6, "insufficient_blocks"), (60, 6, "few_blocks")],
)
def test_year_and_block_thresholds_include_their_boundary(years: int, length: int, flag: str) -> None:
    """30 years, 5 blocks, and 10 blocks are the smallest values that do not raise their flag."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ClimateIndicesWarning)
        result = run(prepared(years), block_length_years=length)
    assert not np.any(result.quality_flags & implementation._FLAGS[flag])
    assert np.isfinite(result.p_value).all()


@pytest.mark.parametrize(
    ("counts", "interval", "null", "flag"),
    [
        ((60, 30, 30, 5), True, True, None),
        ((60, 30, 30, 4), False, True, "insufficient_successes"),
        ((60, 10, 30, 5), True, True, None),
        ((60, 10, 29, 5), True, False, "insufficient_expected_count"),
    ],
    ids=["five-successes", "four-successes", "expected-five", "expected-below-five"],
)
def test_success_and_expected_count_thresholds_include_their_boundary(
    counts: tuple[int, int, int, int], interval: bool, null: bool, flag: str | None
) -> None:
    """Five observed successes pass the interval guard; 10 * 30 / 60 = 5 expected passes the null guard."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ClimateIndicesWarning)
        result = run(table(*counts))
    assert bool(np.isfinite(result.confidence_low).all()) is interval
    assert bool(np.isfinite(result.p_value).all()) is null
    if flag is None:
        assert np.all(result.quality_flags == 0)
    else:
        assert np.all(result.quality_flags & implementation._FLAGS[flag])


def test_seed_selects_the_resample_stream() -> None:
    hazards = table(60, 25, 30, 12)  # not periodic, so different block draws give different statistics
    first, second = run(hazards, seed=1), run(hazards, seed=2)
    np.testing.assert_array_equal(first.p_value, run(hazards, seed=1).p_value)
    assert not np.array_equal(first.p_value, second.p_value)
    assert not np.array_equal(first.confidence_low, second.confidence_low)


def test_confidence_level_sets_the_quantile_levels() -> None:
    values = np.arange(100, dtype=float) / 100
    for level in (0.95, 0.5):
        with (
            mock.patch.object(implementation, "_resample", return_value=(values, values / 10)),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", ClimateIndicesWarning)
            result = run(groupby="pooled", n_resamples=100, confidence_level=level)
        np.testing.assert_allclose(
            [result.confidence_low, result.confidence_high],
            np.quantile(values, [(1 - level) / 2, (1 + level) / 2], method="linear"),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    ("statistic", "pattern"),
    [
        ("conditional_probability", [[1, 1, 0], [0, 1, 1]]),
        ("joint_probability", [[1, 1, 0], [0, 1, 1], [1, 0, 1]]),
    ],
    ids=["conditional", "three-hazard"],
)
def test_exact_tiny_enumeration_for_conditional_and_three_hazard_nulls(
    statistic: implementation.Statistic, pattern: list[list[int]]
) -> None:
    """Enumerate every paired draw and every independent per-channel draw of three years in blocks of two."""
    blocks = [np.array([s, s + 1, t]) for s, t in itertools.product(range(2), repeat=2)]
    rows = np.repeat(np.array(pattern, dtype=bool)[..., None], 12, axis=2)

    def point(x: np.ndarray) -> float:
        return x.all(axis=0).sum() / x[0].sum() if statistic == "conditional_probability" else x.all(axis=0).mean()

    def excess(x: np.ndarray) -> float:
        if statistic == "conditional_probability":
            return point(x) - x[1].mean()
        return x.all(axis=0).mean() - np.prod(x.mean(axis=1))

    draws = list(itertools.product(blocks, repeat=len(rows)))
    schedule, expected_paired, expected_null = [], [], []
    for b, channels in enumerate(draws):
        paired = blocks[b % 4]
        schedule.extend((paired, *channels))
        expected_paired.append(point(rows[:, paired, 0]))
        expected_null.append(excess(np.stack([rows[i, ch, 0] for i, ch in enumerate(channels)])))
    with mock.patch.object(implementation, "_year_indices", side_effect=schedule):
        paired, null = implementation._resample(
            rows,
            statistic=statistic,
            groupby="pooled",
            length=2,
            n_resamples=len(draws),
            seed=0,
            support="tiny enumeration",
        )
    np.testing.assert_allclose(paired, expected_paired, rtol=0, atol=1e-15)
    np.testing.assert_allclose(null, expected_null, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    "option,value",
    [
        ("seed", True),
        ("seed", -1),
        ("seed", 1.2),
        ("n_resamples", 0),
        ("n_resamples", True),
        ("block_length_years", 0),
        ("block_length_years", 1.5),
        ("confidence_level", float("nan")),
        ("confidence_level", True),
        ("confidence_level", 1),
        ("confidence_level", "0.95"),
        ("alternative", "two-sided"),
        ("groupby", "season"),
        ("statistic", "LMF"),
        ("lag_window", (-1, 0)),
        ("given", "A"),
    ],
)
def test_option_validation(option: str, value: Any) -> None:
    with pytest.raises(InvalidArgumentError):
        run(**{option: value})


def test_daily_manual_input_and_conditional_role_validation() -> None:
    hazards = prepared()
    with pytest.raises(InvalidArgumentError, match="monthly"):
        run(dataclasses.replace(hazards, cadence="daily"))
    with pytest.raises(InvalidArgumentError, match="AlignedHazards"):
        run("not prepared")  # type: ignore[arg-type]
    with pytest.raises(InvalidArgumentError):
        run(statistic="conditional_probability", given="A", target="A")
    with pytest.raises(InvalidArgumentError):
        run(statistic="conditional_probability", given="missing", target="B")


def test_expected_count_guard_is_exact_at_its_boundary() -> None:
    """11 * 15 / 33 is exactly 5, but ``33 * (11/33) * (15/33)`` rounds to 4.999999999999999."""
    result = run(table(33, 11, 15, 5), block_length_years=3)
    assert np.all(np.isfinite(result.p_value))
    assert np.all(result.quality_flags == 0)


def test_sparse_and_expected_count_guards_remain_distinct() -> None:
    hazards = prepared()
    rows = np.zeros_like(hazards.exceedance)
    # Ten occurrences per month pass the marginal guard, but 60*(10/60)^2 < 5.
    rows[:, : 10 * 12] = True
    with pytest.warns(ClimateIndicesWarning):
        result = run(dataclasses.replace(hazards, exceedance=rows))
    assert np.isfinite(result.confidence_low).all()
    assert np.isnan(result.p_value).all()
    assert np.all(result.quality_flags & implementation._FLAGS["insufficient_expected_count"])
    assert not np.any(result.quality_flags & implementation._FLAGS["sparse_occurrences"])
    rows[:, 9 * 12 :] = False
    with pytest.warns(ClimateIndicesWarning):
        sparse = run(dataclasses.replace(hazards, exceedance=rows))
    assert np.isnan(sparse.confidence_low).all()
    assert np.all(sparse.quality_flags & implementation._FLAGS["sparse_occurrences"])


def test_sparse_marginal_blocks_the_null_when_the_expected_count_is_met() -> None:
    """9 * 40 / 60 = 6 expected successes, yet one channel has fewer than ten occurrences."""
    with pytest.warns(ClimateIndicesWarning):
        result = run(table(60, 9, 40, 5))
    assert np.isnan(result.p_value).all()
    assert np.all(result.quality_flags & implementation._FLAGS["sparse_occurrences"])
    assert not np.any(result.quality_flags & implementation._FLAGS["insufficient_expected_count"])


def test_undefined_point_estimate_is_flagged() -> None:
    hazards = table(60, 30, 30, 5)
    rows = hazards.exceedance.copy()
    rows[1] = False
    with pytest.warns(ClimateIndicesWarning):
        result = run(dataclasses.replace(hazards, exceedance=rows), statistic="likelihood_multiplication_factor")
    assert np.isnan(result.estimate).all()
    assert np.all(result.quality_flags & implementation._FLAGS["undefined_denominator"])


def test_quality_warning_points_at_the_caller() -> None:
    with pytest.warns(ClimateIndicesWarning) as caught:
        run(prepared(30))
    assert caught[0].filename == __file__


@pytest.mark.parametrize("statistic", ["likelihood_multiplication_factor", "conditional_probability"])
def test_undefined_resampled_denominators_remain_nan(statistic: implementation.Statistic) -> None:
    paired, null = implementation._resample(
        np.zeros((2, 3, 12), dtype=bool),
        statistic=statistic,
        groupby="month",
        length=1,
        n_resamples=3,
        seed=0,
        support="undefined",
    )
    assert np.isnan(paired).all()
    assert np.isnan(null).all()


def test_month_phase_and_unknown_future_response() -> None:
    hazards = prepared()
    rows = np.zeros_like(hazards.exceedance)
    rows[:, ::12] = True
    with pytest.warns(ClimateIndicesWarning):
        phase = run(dataclasses.replace(hazards, exceedance=rows))
    assert phase.confidence_low[0] == phase.confidence_high[0] == 1
    assert np.isnan(phase.confidence_low[1:]).all()
    validity = hazards.valid.copy()
    validity[1, 30 * 12] = False
    with pytest.warns(ClimateIndicesWarning):
        future = run(
            dataclasses.replace(hazards, valid=validity),
            statistic="conditional_probability",
            given="A",
            target="B",
            lag_window=(1, 1),
        )
    # The missing January invalidates the preceding December anchor year too.
    assert future.n_years == 29
    assert future.inference_start == np.datetime64("1900-01-01")
    assert future.inference_end == np.datetime64("1928-12-01")


def test_immutable_manifest_arrays_and_lifecycle() -> None:
    hazards = prepared()
    original = hazards.exceedance.copy()
    log = mock.Mock()
    with mock.patch.object(implementation._logger, "bind", return_value=log):
        result = run(hazards)
    assert log.info.call_args_list[0].args == ("calculation_started",)
    assert log.info.call_args_list[-1].args == ("calculation_completed",)
    np.testing.assert_array_equal(hazards.exceedance, original)
    for field in dataclasses.fields(result):
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray):
            with pytest.raises(ValueError):
                value.setflags(write=True)
    with pytest.raises(TypeError):
        result.metadata["seed"] = 1  # type: ignore[index]
    manifest = json.loads(json.dumps(dict(result.metadata)))
    assert manifest["p_value_scope"].startswith("local, unadjusted")
    assert manifest["numpy_version"] == np.__version__
    with (
        mock.patch.object(implementation._logger, "bind", return_value=log),
        mock.patch.object(implementation, "log_calculation_failure") as failure,
    ):
        with pytest.raises(InvalidArgumentError):
            run(seed=-1)
        failure.assert_called_once()
