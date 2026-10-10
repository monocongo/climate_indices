"""Batched Pearson goodness-of-fit statistics and unchanged warning decisions."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.stats

from climate_indices import compute
from tests import conftest

native = conftest.import_native()


@pytest.mark.parametrize("rows", [0, 1, 35, 1000])
def test_statistics_match_scipy_with_missing_zero_and_invalid_parameters(monkeypatch, rows):
    rng = np.random.default_rng(1281)
    values = rng.normal(60.0, 30.0, (rows, 12))
    values.reshape(-1)[::13] = np.nan
    values.reshape(-1)[::17] = 0.0
    if rows:
        values[0, 0] = np.inf
        values[0, 1] = -np.inf
    locs = np.full(12, 60.0)
    scales = np.full(12, 30.0)
    skews = np.array([-2.0, 2.0, 0.0, 1.5e-5, 1.6e-5, -1.6e-5, 1e-10, np.nan, 1.0, 0.0, 0.0, 0.0])
    scales[8:10] = [0.0, np.inf]
    locs[10:] = [np.inf, np.nan]
    with np.errstate(all="ignore"):
        rust = native.pearson_ks_statistics(values, skews, locs, scales)
        monkeypatch.setattr(compute, "_native", None)
        python = compute._pearson_ks_statistics(values, locs, scales, skews)
    np.testing.assert_array_equal(np.isnan(rust), np.isnan(python))
    np.testing.assert_allclose(rust, python, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("poor", [False, True])
def test_warning_text_categories_and_metadata_match(monkeypatch, poor):
    values = np.random.default_rng(1281).normal(60.0, 30.0, (35, 12))
    if poor:
        values[:] = 1000.0
    locs, scales, skews = np.full(12, 60.0), np.full(12, 30.0), np.zeros(12)
    outcomes = []
    for backend in (native, None):
        monkeypatch.setattr(compute, "_native", backend)
        with warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
            warnings.simplefilter("always")
            compute._check_goodness_of_fit_pearson(values, np.zeros(12), locs, scales, skews)
        outcomes.append([(item.category, str(item.message), vars(item.message)) for item in caught])
    assert outcomes[0] == outcomes[1]
    if poor:
        assert outcomes[0]


def test_near_threshold_keeps_oracle_cdf(monkeypatch):
    values = np.linspace(1.0, 100.0, 35).reshape(-1, 1)
    critical = compute._ks_critical_value(35)
    monkeypatch.setattr(
        compute, "_native", SimpleNamespace(pearson_ks_statistics=lambda *args: np.array([critical - 1e-12]))
    )
    original = scipy.stats.pearson3.cdf
    calls = []

    def cdf(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(scipy.stats.pearson3, "cdf", cdf)
    with np.errstate(all="ignore"):
        compute._check_goodness_of_fit_pearson(values, np.zeros(1), np.full(1, 50.0), np.full(1, 30.0), np.zeros(1))
    assert len(calls) == 1


def test_native_failure_is_not_swallowed_by_gof_exception_handler(monkeypatch):
    def fail(*args):
        raise RuntimeError("native failure")

    monkeypatch.setattr(compute, "_native", SimpleNamespace(pearson_ks_statistics=fail))
    with np.errstate(all="ignore"), pytest.raises(RuntimeError, match="native failure"):
        compute._check_goodness_of_fit_pearson(np.ones((35, 1)), np.zeros(1), np.ones(1), np.ones(1), np.zeros(1))
