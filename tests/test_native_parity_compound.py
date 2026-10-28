"""Exact compound bootstrap parity, including the Python random streams."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from climate_indices.compound import _bootstrap as implementation
from climate_indices.compound._statistics import GroupBy
from tests import conftest
from tests.test_compound_bootstrap import prepared, run

STATISTICS = ("joint_probability", "likelihood_multiplication_factor", "conditional_probability")


def test_native_resample_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("native metric failure")

    monkeypatch.setattr(implementation, "_native", SimpleNamespace(compound_resample_metrics=fail), raising=False)
    with np.errstate(all="ignore"), pytest.raises(RuntimeError, match="native metric failure"):
        implementation._resample(
            np.ones((2, 3, 12), dtype=bool),
            statistic="joint_probability",
            groupby="month",
            length=1,
            n_resamples=2,
            seed=0,
            support="failure",
        )


@pytest.mark.parametrize("statistic", STATISTICS)
@pytest.mark.parametrize("groupby", ["month", "pooled"])
@pytest.mark.parametrize("length", [1, 4, 60])
@pytest.mark.parametrize("sources", [2, 3])
def test_resample_streams_are_exact(
    monkeypatch: pytest.MonkeyPatch,
    statistic: implementation.Statistic,
    groupby: GroupBy,
    length: int,
    sources: int,
) -> None:
    native = conftest.import_native()
    rows = np.random.default_rng(124).random((sources, 60, 12)) < 0.55
    if statistic == "conditional_probability":
        rows = rows[:2]
    # More than one native batch must preserve every per-resample, per-channel draw.
    options: dict[str, Any] = {
        "statistic": statistic,
        "groupby": groupby,
        "length": length,
        "n_resamples": 259,
        "seed": 719,
        "support": "stream",
    }
    recorder = conftest.NativeRecorder(native)
    monkeypatch.setattr(implementation, "_native", recorder)
    with np.errstate(all="ignore"):
        actual = implementation._resample(rows, **options)
        monkeypatch.setattr(implementation, "_native", None)
        expected = implementation._resample(rows, **options)
    assert recorder.calls == {"compound_resample_metrics"}
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("backend", [None, SimpleNamespace()])
def test_missing_or_stale_extension_keeps_python(
    monkeypatch: pytest.MonkeyPatch,
    backend: Any,
) -> None:
    monkeypatch.setattr(implementation, "_native", backend)
    paired, null = implementation._resample(
        np.ones((2, 3, 12), dtype=bool),
        statistic="joint_probability",
        groupby="month",
        length=1,
        n_resamples=2,
        seed=0,
        support="fallback",
    )
    np.testing.assert_array_equal(paired, np.ones((2, 12)))
    np.testing.assert_array_equal(null, np.zeros((2, 12)))


def test_reporting_policy_keeps_python(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        pytest.fail("native bypassed the reporting policy")

    monkeypatch.setattr(implementation, "_native", SimpleNamespace(compound_resample_metrics=fail))
    with np.errstate(all="raise"):
        implementation._resample(
            np.ones((2, 3, 12), dtype=bool),
            statistic="joint_probability",
            groupby="month",
            length=1,
            n_resamples=2,
            seed=0,
            support="policy",
        )


@pytest.mark.parametrize("statistic", STATISTICS)
@pytest.mark.parametrize("groupby", ["month", "pooled"])
def test_full_bootstrap_result_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    statistic: implementation.Statistic,
    groupby: GroupBy,
) -> None:
    options = {"given": "A", "target": "B", "lag_window": (0, 2)} if statistic == "conditional_probability" else {}
    native, python, calls = conftest.rust_and_python(
        monkeypatch, implementation, lambda: run(prepared(cells=4), statistic=statistic, groupby=groupby, **options)
    )
    assert calls == {"compound_resample_metrics"}
    for field in dataclasses.fields(native):
        left, right = getattr(native, field.name), getattr(python, field.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right


@pytest.mark.parametrize("case", ["paired_index", "null_index", "sources", "months", "statistic", "groupby", "dtype"])
def test_binding_rejects_invalid_inputs(case: str) -> None:
    native = conftest.import_native()
    rows = np.ones((2, 3, 12), dtype=bool)
    paired = np.zeros((1, 3), dtype=np.int64)
    null = np.zeros((1, 2, 3), dtype=np.int64)
    statistic, groupby = "joint_probability", "month"
    if case == "paired_index":
        paired[0, 0] = -1
    elif case == "null_index":
        null[0, 1, 2] = 3
    elif case == "sources":
        null = null[:, :1]
    elif case == "months":
        rows = rows[:, :, :11]
    elif case == "statistic":
        statistic = "unknown"
    elif case == "groupby":
        groupby = "unknown"
    elif case == "dtype":
        rows = rows.astype(np.int64)
    with pytest.raises(TypeError if case == "dtype" else ValueError):
        native.compound_resample_metrics(rows, paired, null, statistic, groupby)


@pytest.mark.parametrize("statistic", STATISTICS)
def test_zero_marginals_have_matching_nan_positions(
    monkeypatch: pytest.MonkeyPatch,
    statistic: implementation.Statistic,
) -> None:
    native = conftest.import_native()
    rows = np.zeros((2, 3, 12), dtype=bool)
    options: dict[str, Any] = {
        "statistic": statistic,
        "groupby": "month",
        "length": 1,
        "n_resamples": 3,
        "seed": 0,
        "support": "zero",
    }
    with np.errstate(all="ignore"):
        monkeypatch.setattr(implementation, "_native", native)
        actual = implementation._resample(rows, **options)
        monkeypatch.setattr(implementation, "_native", None)
        expected = implementation._resample(rows, **options)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left, right)
