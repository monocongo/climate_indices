"""Tests for the goodness-of-fit evidence of ``suitability.compare_distributions`` (#1586, ADR-0028 section 5)."""

from __future__ import annotations

import json
import sys
import threading
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.stats
import xarray as xr

from climate_indices import compute, suitability
from climate_indices.exceptions import GoodnessOfFitWarning, InvalidArgumentError
from climate_indices.vocabulary import Distribution
from tests import conftest

MONTHLY = compute.Periodicity.monthly
START, YEARS = 1950, 40
END = START + YEARS - 1
GAMMA = (Distribution.gamma,)  # Pearson Type III fails to fit some steps of any record, by design

pytestmark = [
    pytest.mark.filterwarnings("ignore::climate_indices.exceptions.BetaFeatureWarning"),
    # a fit of a masked cell takes the mean of an empty slice, the documented all-missing outcome
    pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning"),
]


def _compare(values: Any, scale: int = 1, *, index: str = "spi", **kwargs: Any) -> Any:
    kwargs.setdefault("data_start_year", START)
    kwargs.setdefault("calibration_year_initial", START)
    kwargs.setdefault("calibration_year_final", END)
    kwargs.setdefault("periodicity", MONTHLY)
    return suitability.compare_distributions(values, scale, index=index, **kwargs)


def _gamma(seed: int, *cells: int, years: int = YEARS) -> np.ndarray:
    """Stationary gamma monthly precipitation, ``(years * 12, *cells)``: every calendar step is iid gamma."""
    return np.random.default_rng(seed).gamma(2.0, 30.0, size=(years * 12, *cells))


def _grid(seed: int = 0, *, years: int = YEARS) -> xr.DataArray:
    times = pd.date_range(f"{START}-01-01", periods=years * 12, freq="MS")
    return xr.DataArray(_gamma(seed, 3, 2, years=years), dims=("time", "lat", "lon"), coords={"time": times})


def _gof(values: Any, scale: int = 1, **kwargs: Any) -> suitability.GoodnessOfFit:
    result = _compare(values, scale, **kwargs)
    assert result.goodness_of_fit is not None
    return result.goodness_of_fit


# ---- the building blocks, against independent implementations ---------------------------------------------


def _anderson_darling(u: np.ndarray) -> float:
    """A^2 by the textbook double sum, one term at a time."""
    ordered = np.sort(u)
    n = len(ordered)
    total = sum((2 * i - 1) * (np.log(ordered[i - 1]) + np.log(1.0 - ordered[n - i])) for i in range(1, n + 1))
    return float(-n - total / n)


class TestStatistics:
    def test_k_s_and_anderson_darling_match_independent_formulas(self):
        rng = np.random.default_rng(1)
        pit = rng.uniform(size=(30, 4))
        pit[25:, 1] = np.nan  # a column with fewer finite values
        pit[:, 3] = np.nan  # a column with none
        ks, ad, count = suitability._pit_statistics(pit)
        for column in range(3):
            finite = pit[:, column][np.isfinite(pit[:, column])]
            assert count[column] == finite.size
            assert ks[column] == pytest.approx(scipy.stats.kstest(finite, "uniform").statistic, rel=1e-12)
            assert ad[column] == pytest.approx(_anderson_darling(finite), rel=1e-12)
        assert count[3] == 0
        assert np.isnan(ks[3])
        assert np.isnan(ad[3])

    def test_a_value_on_the_edge_of_the_support_scores_a_finite_statistic(self):
        ks, ad, _ = suitability._pit_statistics(np.array([[0.0], [0.4], [1.0]]))
        assert np.isfinite(ks[0])
        assert np.isfinite(ad[0])

    @pytest.mark.parametrize("size", [1, 2, 5, 6, 7, 12, 30, 41, 97])
    def test_naive_p_value_is_the_exact_k_s_p_value(self, size):
        rng = np.random.default_rng(size)
        samples = rng.uniform(size=(size, 40))
        samples[:, ::3] = samples[:, ::3] ** 1.7  # some that are not uniform
        statistics = np.array([scipy.stats.kstest(samples[:, c], "uniform").statistic for c in range(40)])
        expected = np.array([scipy.stats.kstest(samples[:, c], "uniform").pvalue for c in range(40)])
        got = suitability._ks_naive_p_value(statistics, np.full(40, size))
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_naive_p_value_is_nan_without_a_sample(self):
        got = suitability._ks_naive_p_value(np.array([0.3, np.nan, 0.2]), np.array([0, 10, 10]))
        assert np.isnan(got[0])
        assert np.isnan(got[1])
        assert np.isfinite(got[2])

    @pytest.mark.parametrize(
        ("distribution", "p"),
        [
            (Distribution.gamma, {"alpha": 2.5, "beta": 20.0}),
            (Distribution.pearson, {"loc": 50.0, "scale": 20.0, "skew": 1.2}),
            (Distribution.pearson, {"loc": 50.0, "scale": 20.0, "skew": -0.8}),
            (Distribution.loglogistic, {"loc": 100.0, "scale": 30.0, "shape": -0.3}),
            (Distribution.loglogistic, {"loc": 100.0, "scale": 30.0, "shape": 0.25}),
            (Distribution.loglogistic, {"loc": 100.0, "scale": 30.0, "shape": 0.0}),
            (Distribution.gev, {"loc": 100.0, "scale": 30.0, "shape": -0.2}),
            (Distribution.gev, {"loc": 100.0, "scale": 30.0, "shape": 0.2}),
        ],
    )
    def test_quantile_inverts_the_cdf_and_the_cdf_integrates_the_density(self, distribution, p):
        params = {key: np.asarray(value) for key, value in p.items()}
        u = np.linspace(0.02, 0.98, 49)
        x = suitability._continuous_ppf(distribution, u, params)
        np.testing.assert_allclose(suitability._continuous_cdf(distribution, x, params), u, atol=1e-9)
        # the CDF's slope is the density the likelihood uses, so the PIT is the likelihood's own CDF
        step = 1e-5 * np.abs(x).clip(1.0)
        slope = (
            suitability._continuous_cdf(distribution, x + step, params)
            - suitability._continuous_cdf(distribution, x - step, params)
        ) / (2 * step)
        density = np.exp(suitability._log_density(distribution, x, params))
        np.testing.assert_allclose(slope, density, rtol=1e-5)

    def test_log_logistic_cdf_is_zero_or_one_beyond_its_support(self):
        # shape > 0 has a finite upper end, shape < 0 a finite lower end
        upper = suitability._glo_cdf(np.array([1e6]), np.array(0.0), np.array(1.0), np.array(0.5))
        lower = suitability._glo_cdf(np.array([-1e6]), np.array(0.0), np.array(1.0), np.array(-0.5))
        assert upper[0] == 1.0
        assert lower[0] == 0.0


# ---- the result on the NumPy route -----------------------------------------------------------------------


class TestResult:
    def test_default_call_computes_the_statistics_and_the_naive_p_value_only(self):
        result = _compare(_gamma(0))
        gof = result.goodness_of_fit
        assert gof.ks_statistic.shape == gof.ks_p_value_naive.shape == gof.ad_statistic.shape == (2, 12)
        scored = ~result.fit_failed  # a candidate that failed to fit has no statistic
        assert scored[1].all()  # gamma fits every step of this record
        for name in ("ks_statistic", "ad_statistic", "ks_p_value_naive"):
            assert (np.isfinite(getattr(gof, name)) == scored).all(), name
        assert ((gof.ks_p_value_naive[scored] > 0) & (gof.ks_p_value_naive[scored] <= 1)).all()
        assert gof.n_bootstrap == 0
        for name in ("ks_p_value_bootstrap", "ad_p_value_bootstrap", "bootstrap_failed_share"):
            assert getattr(gof, name) is None
        for name in ("ks_p_value_bh", "ad_p_value_bh", "rejection_share"):
            assert getattr(gof, name) is None  # a single series has no cell axis

    def test_an_empty_gof_skips_goodness_of_fit(self):
        assert _compare(_gamma(0), gof=()).goodness_of_fit is None

    def test_a_statistic_left_out_is_nan(self):
        only_ad = _gof(_gamma(0), gof=("ad",), candidates=GAMMA)
        assert np.isnan(only_ad.ks_statistic).all()
        assert np.isnan(only_ad.ks_p_value_naive).all()
        assert np.isfinite(only_ad.ad_statistic).all()
        only_ks = _gof(_gamma(0), gof=["ks"], n_bootstrap=9, seed=1, candidates=GAMMA)
        assert np.isnan(only_ks.ad_statistic).all()
        assert np.isnan(only_ks.ad_p_value_bootstrap).all()
        assert np.isfinite(only_ks.ks_p_value_bootstrap).all()

    def test_the_statistic_is_the_one_of_the_fitted_candidate_on_its_positive_values(self):
        values = _gamma(2)
        values[::7] = 0.0  # exact zeros are the zero mass, not part of the continuous distribution
        result = _compare(values, candidates=(Distribution.gamma,))
        alpha, beta = (result.parameters[Distribution.gamma][name] for name in ("alpha", "beta"))
        scaled = compute.prepare_scaled(values, 1, MONTHLY)
        for step in range(12):
            positive = scaled[:, step][scaled[:, step] > 0]
            expected = scipy.stats.kstest(positive, scipy.stats.gamma(a=alpha[step], scale=beta[step]).cdf)
            assert result.goodness_of_fit.ks_statistic[0, step] == pytest.approx(expected.statistic, rel=1e-10)
            assert result.goodness_of_fit.ks_p_value_naive[0, step] == pytest.approx(expected.pvalue, abs=1e-5)

    def test_log_logistic_keeps_an_exact_zero_as_an_ordinary_value(self):
        # a family with no zero mass scores a zero like any value, so its statistics cover it too
        rng = np.random.default_rng(5)
        sample = scipy.stats.fisk.rvs(c=1 / 0.3, scale=60.0, size=(YEARS, 12), random_state=rng)
        sample[:3] = 0.0  # a prepared SPEI sample (water balance plus the 1000 offset) of exactly zero
        result = _compare(
            np.zeros(YEARS * 12),
            index="spei",
            pet_mm=(1000.0 - sample).reshape(-1),
            candidates=(Distribution.loglogistic,),
        )
        assert not result.fit_failed.any()
        params = result.parameters[Distribution.loglogistic]
        for step in range(12):
            loc, scale, k = (params[name][step] for name in ("loc", "scale", "shape"))
            z = (sample[:, step] - loc) / scale  # Hosking's CDF, written out independently of the module
            u = 1 / (1 + np.exp(np.log1p(-k * z) / k)) if abs(k) > 1e-6 else 1 / (1 + np.exp(-z))
            expected = scipy.stats.kstest(u, "uniform")
            assert result.goodness_of_fit.ks_statistic[0, step] == pytest.approx(expected.statistic, rel=1e-9)
            assert result.goodness_of_fit.ks_p_value_naive[0, step] == pytest.approx(expected.pvalue, abs=1e-5)

    def test_repeated_positive_values_are_kept_with_their_frequency(self):
        values = np.round(_gamma(3), -1) + 10.0  # a record rounded to tens: many ties, none of them zero
        result = _compare(values, candidates=(Distribution.gamma,))
        alpha, beta = (result.parameters[Distribution.gamma][name] for name in ("alpha", "beta"))
        scaled = compute.prepare_scaled(values, 1, MONTHLY)
        assert np.unique(scaled[:, 0]).size < scaled.shape[0]
        for step in range(12):
            expected = scipy.stats.kstest(scaled[:, step], scipy.stats.gamma(a=alpha[step], scale=beta[step]).cdf)
            assert result.goodness_of_fit.ks_statistic[0, step] == pytest.approx(expected.statistic, rel=1e-10)

    def test_a_candidate_without_a_usable_fit_has_no_goodness_of_fit(self):
        values = _gamma(4, 2, 2)
        values[:, 0, 0] = np.nan
        result = _compare(values, spatial_time_major=True, n_bootstrap=5, seed=1, candidates=GAMMA)
        gof = result.goodness_of_fit
        for name in (
            "ks_statistic",
            "ks_p_value_naive",
            "ad_statistic",
            "ks_p_value_bootstrap",
            "ad_p_value_bootstrap",
        ):
            assert np.isnan(getattr(gof, name)[:, :, 0, 0]).all()
        assert np.isnan(gof.bootstrap_failed_share[:, :, 0, 0]).all()
        assert np.isfinite(gof.ks_statistic[:, :, 1, 1]).all()

    def test_a_candidate_that_failed_to_fit_has_no_goodness_of_fit(self, monkeypatch):
        def failing_resolve(*args, **kwargs):
            raise suitability.DistributionFittingError("no fit")

        monkeypatch.setattr(compute.FittedDistribution, "resolve", failing_resolve)
        gof = _gof(_gamma(0), n_bootstrap=3, seed=1, candidates=GAMMA)
        assert np.isnan(gof.ks_statistic).all()
        assert np.isnan(gof.ks_p_value_bootstrap).all()


class TestBootstrap:
    def test_p_values_follow_the_floor_formula_and_report_the_failed_share(self):
        gof = _gof(_gamma(5, 2, 3), n_bootstrap=49, seed=3, spatial_time_major=True, candidates=GAMMA)
        for p in (gof.ks_p_value_bootstrap, gof.ad_p_value_bootstrap):
            assert p.shape == gof.ks_statistic.shape
            assert np.nanmin(p) >= 1 / 50
            assert np.nanmax(p) <= 1
            # with no failed refit every p-value is a multiple of 1 / (B + 1)
            np.testing.assert_allclose(p * 50, np.round(p * 50), atol=1e-9)
        assert (gof.bootstrap_failed_share == 0).all()
        assert gof.n_bootstrap == 49

    def test_a_failed_refit_leaves_the_denominator_and_is_reported(self):
        # about 6 positive values of 40 per step: a refit loses its fit once it draws fewer than the 4 Pearson needs
        values = _gamma(6, 2, 2)
        values[np.random.default_rng(7).random(values.shape) < 0.84] = 0.0
        gof = _gof(values, candidates=(Distribution.pearson,), n_bootstrap=60, seed=2, spatial_time_major=True)
        share = gof.bootstrap_failed_share
        assert np.nanmax(share) > 0
        assert ((share > 0) & (share < 1)).any()  # a column that lost some refits and kept others
        succeeded = np.round(60 * (1 - np.nan_to_num(share, nan=1.0))).astype(int)  # no fit, no refit
        p = gof.ks_p_value_bootstrap
        defined = succeeded > 0
        np.testing.assert_allclose((p * (succeeded + 1))[defined], np.round((p * (succeeded + 1))[defined]), atol=1e-9)
        assert (p[defined] >= 1 / (succeeded[defined] + 1) - 1e-12).all()
        assert np.isnan(p[~defined]).all()

    def test_every_refit_failing_gives_a_nan_p_value_and_a_full_failed_share(self, monkeypatch):
        def never_fits(distribution, draws, period, periodicity):
            nan = np.full(draws.shape[1:], np.nan)
            return {"ks": nan, "ad": nan}, np.zeros(draws.shape[1:], dtype=bool)

        monkeypatch.setattr(suitability, "_refit_statistics", never_fits)
        gof = _gof(_gamma(0), n_bootstrap=8, seed=1, candidates=GAMMA)
        assert np.isnan(gof.ks_p_value_bootstrap).all()
        assert np.isnan(gof.ad_p_value_bootstrap).all()
        assert (gof.bootstrap_failed_share == 1).all()
        assert np.isfinite(gof.ks_statistic).all()

    def test_a_candidate_without_a_usable_fit_is_neither_simulated_nor_refitted(self, monkeypatch):
        values = _gamma(13, 2, 2)
        both = (Distribution.gamma, Distribution.pearson)
        reference = _gof(values, n_bootstrap=10, seed=8, spatial_time_major=True, candidates=both)
        score = suitability._score_fit
        refitted: list[Distribution] = []
        refit = suitability._refit_statistics

        def pearson_fails(distribution, *args):
            log_likelihood, failed = score(distribution, *args)
            return (
                (log_likelihood, np.ones_like(failed))
                if distribution is Distribution.pearson
                else (log_likelihood, failed)
            )

        def spy(distribution, *args):
            refitted.append(distribution)
            return refit(distribution, *args)

        monkeypatch.setattr(suitability, "_score_fit", pearson_fails)
        monkeypatch.setattr(suitability, "_refit_statistics", spy)
        gof = _gof(values, n_bootstrap=10, seed=8, spatial_time_major=True, candidates=both)
        assert set(refitted) == {Distribution.gamma}
        order = suitability._resolve_candidates("spi", both)
        failed, kept = order.index(Distribution.pearson), order.index(Distribution.gamma)
        assert np.isnan(gof.ks_p_value_bootstrap[failed]).all()
        assert np.isnan(gof.bootstrap_failed_share[failed]).all()
        np.testing.assert_array_equal(gof.ks_p_value_bootstrap[kept], reference.ks_p_value_bootstrap[kept])

        refitted.clear()
        monkeypatch.setattr(
            suitability,
            "_score_fit",
            lambda *args: (np.full(args[1].shape[1:], np.nan), np.ones(args[1].shape[1:], dtype=bool)),
        )
        none = _gof(values, n_bootstrap=10, seed=8, spatial_time_major=True, candidates=both)
        assert refitted == []
        assert np.isnan(none.ks_p_value_bootstrap).all()

    def test_the_same_seed_gives_an_identical_result_and_another_seed_does_not(self):
        values = _gamma(8, 2, 2)
        first = _gof(values, n_bootstrap=20, seed=11, spatial_time_major=True)
        again = _gof(values, n_bootstrap=20, seed=11, spatial_time_major=True)
        other = _gof(values, n_bootstrap=20, seed=12, spatial_time_major=True)
        for name in ("ks_p_value_bootstrap", "ad_p_value_bootstrap", "bootstrap_failed_share"):
            np.testing.assert_array_equal(getattr(first, name), getattr(again, name))
        assert not np.array_equal(first.ks_p_value_bootstrap, other.ks_p_value_bootstrap)

    def test_a_cell_gets_the_same_p_values_alone_as_in_a_grid(self):
        # the stream is keyed on the cell's own values, so neither its position nor its neighbours change it
        values = _gamma(9, 2, 3)
        grid = _gof(values, n_bootstrap=15, seed=4, spatial_time_major=True)
        alone = _gof(values[:, 1:2, 2:3], n_bootstrap=15, seed=4, spatial_time_major=True)
        np.testing.assert_array_equal(alone.ks_p_value_bootstrap[:, :, 0, 0], grid.ks_p_value_bootstrap[:, :, 1, 2])
        np.testing.assert_array_equal(alone.ad_p_value_bootstrap[:, :, 0, 0], grid.ad_p_value_bootstrap[:, :, 1, 2])

    @pytest.mark.parametrize("use_native", [False, True], ids=["python", "native"])
    def test_a_scaled_cell_gets_the_same_bootstrap_counts_in_every_layout(
        self, monkeypatch: pytest.MonkeyPatch, use_native: bool
    ) -> None:
        monkeypatch.setattr(compute, "_native", conftest.import_native() if use_native else None)
        grid = _grid(seed=3)
        options = {"n_bootstrap": 17, "seed": 6, "candidates": GAMMA}
        alone = _gof(grid.isel(lat=0, lon=1).values, 12, **options)
        for data in (grid, grid.chunk({"time": -1, "lat": 1, "lon": 1})):
            cell = _compare(data, 12, **options).isel(lat=0, lon=1)
            np.testing.assert_array_equal(cell.ks_p_value_bootstrap.values, alone.ks_p_value_bootstrap)
            np.testing.assert_array_equal(cell.ad_p_value_bootstrap.values, alone.ad_p_value_bootstrap)
            np.testing.assert_array_equal(cell.bootstrap_failed_share.values, alone.bootstrap_failed_share)

    def test_the_batch_size_does_not_change_the_result(self, monkeypatch):
        values = _gamma(10, 2, 2)
        whole = _gof(values, n_bootstrap=17, seed=6, spatial_time_major=True)
        monkeypatch.setattr(suitability, "_BOOTSTRAP_BATCH_ELEMENTS", 1)  # one replicate per batch
        singly = _gof(values, n_bootstrap=17, seed=6, spatial_time_major=True)
        np.testing.assert_array_equal(whole.ks_p_value_bootstrap, singly.ks_p_value_bootstrap)

    @pytest.mark.parametrize("use_native", [False, True], ids=["python", "native"])
    def test_rounded_gamma_counts_are_independent_of_batches_and_chunks(self, monkeypatch, use_native):
        if use_native and getattr(sys.flags, "context_aware_warnings", False):
            pytest.skip("context-aware warnings disable native dispatch")
        backend = conftest.NativeRecorder(conftest.import_native()) if use_native else None
        monkeypatch.setattr(compute, "_native", backend)
        grid = np.rint(_grid(seed=3) / 10.0) * 10.0
        grid[{"lat": 0, "lon": 0}] = np.nan
        grid[{"lat": 0, "lon": 1, "time": slice(24, None)}] = 0.0
        options = {"n_bootstrap": 17, "seed": 6, "candidates": GAMMA, "resolution": 10.0}
        success = []
        refit = suitability._refit_statistics

        def record_refit(*args):
            statistics, succeeded = refit(*args)
            success.append(succeeded)
            return statistics, succeeded

        monkeypatch.setattr(suitability, "_refit_statistics", record_refit)
        whole = _compare(grid, **options)
        if backend is not None:
            assert "gamma_parameters" in backend.calls
        batch_success = np.concatenate(success, axis=-1)
        assert batch_success[:, 0, 1].any()
        assert not batch_success[:, 0, 1].all()
        success.clear()
        monkeypatch.setattr(suitability, "_BOOTSTRAP_BATCH_ELEMENTS", 1)
        singly = _compare(grid, **options)
        np.testing.assert_array_equal(np.concatenate(success, axis=-1), batch_success)
        chunked = _compare(grid.chunk({"time": -1, "lat": 1, "lon": 1}), **options).compute()

        for result in (singly, chunked):
            xr.testing.assert_equal(result[list(BOOTSTRAP_VARIABLES)], whole[list(BOOTSTRAP_VARIABLES)])

    def test_seed_is_recorded_and_reproduces_a_run_that_drew_its_own(self):
        values = _gamma(11)
        drawn = _compare(values, n_bootstrap=12)
        recorded = drawn.metadata["climate_indices_rng_seed"]
        assert recorded.isdecimal()
        assert int(recorded) > 2**64  # a 128-bit seed, as a decimal string
        replay = _compare(values, n_bootstrap=12, seed=int(recorded))
        np.testing.assert_array_equal(
            drawn.goodness_of_fit.ks_p_value_bootstrap, replay.goodness_of_fit.ks_p_value_bootstrap
        )
        assert replay.metadata["climate_indices_rng_seed"] == recorded
        assert drawn.metadata["climate_indices_gof_bootstrap_samples"] == "12"
        assert _compare(values, n_bootstrap=12).metadata["climate_indices_rng_seed"] != recorded

    def test_the_resolution_rounds_the_draws_and_is_recorded(self):
        values = np.round(_gamma(12, 2, 2), 0) + 1.0
        continuous = _gof(values, n_bootstrap=30, seed=5, spatial_time_major=True)
        rounded = _compare(values, n_bootstrap=30, seed=5, resolution=1.0, spatial_time_major=True)
        assert not np.array_equal(continuous.ks_p_value_bootstrap, rounded.goodness_of_fit.ks_p_value_bootstrap)
        record = json.loads(rounded.metadata["climate_indices_provenance"])["parameters"]
        assert record["resolution"] == 1.0
        assert record["n_bootstrap"] == 30
        assert record["seed"] == "5"
        assert record["gof"] == ["ks", "ad"]

    def test_simulated_draws_are_rounded_and_keep_the_zero_mass(self):
        uniforms = np.random.default_rng(0).uniform(size=(5000, 1))
        p = {"alpha": np.array([2.0]), "beta": np.array([30.0]), "prob_zero": np.array([0.2])}
        draws = suitability._simulate(Distribution.gamma, uniforms, p, 0.5)
        assert np.allclose(draws / 0.5, np.round(draws / 0.5))
        assert abs((draws == 0).mean() - 0.2) < 0.03  # the zero mass of the fit, nothing else rounds to zero here
        assert (draws[draws > 0] >= 0.5).all()

    def test_a_coarse_unit_adds_no_zeros_beyond_the_fitted_zero_mass(self):
        # the library fits p0 on the rounded record, so it already counts what rounded to zero: a positive draw
        # that rounds to zero is set to the unit, and the simulated zero share stays p0
        uniforms = np.random.default_rng(1).uniform(size=(20000, 1))
        p = {"alpha": np.array([0.8]), "beta": np.array([3.0]), "prob_zero": np.array([0.15])}
        draws = suitability._simulate(Distribution.gamma, uniforms, p, 1.0)
        assert (draws == 0).mean() == pytest.approx((uniforms <= 0.15).mean(), abs=1e-12)
        assert (draws[draws > 0] >= 1.0).all()
        assert (draws == 1.0).mean() > 0.2  # the mass that would have rounded to zero now sits at the unit

    def test_a_family_without_a_zero_mass_rounds_its_draws_as_they_are(self):
        # a zero is an ordinary value for these families, so a draw that rounds to zero stays zero
        uniforms = np.random.default_rng(2).uniform(size=(5000, 1))
        p = {"loc": np.array([0.0]), "scale": np.array([1.0]), "shape": np.array([0.0])}
        draws = suitability._simulate(Distribution.loglogistic, uniforms, p, 2.0)
        assert (draws == 0).mean() > 0.2
        assert np.allclose(draws / 2.0, np.round(draws / 2.0))

    def test_a_zero_mass_family_keeps_negative_pearson_draws_and_moves_only_a_rounding_zero(self):
        # Pearson Type III has a real support below zero. A zero-mass clamp must not fold those draws onto the
        # least positive lattice point: only a draw that rounds to zero moves, so the zero share stays p0.
        uniforms = np.array([0.5, 0.01, 0.99]).reshape(3, 1)
        p = {"loc": np.array([0.0]), "scale": np.array([1.0]), "skew": np.array([0.0]), "prob_zero": np.array([0.0])}
        draws = suitability._simulate(Distribution.pearson, uniforms, p, 1.0)
        assert draws[0] == 1.0  # the draw that rounds to zero sits on the least positive lattice point
        assert draws[1] < 0.0
        assert draws[2] > 0.0

    def test_a_pearson_point_near_zero_stays_on_the_lattice(self) -> None:
        p = {"loc": np.array([0.0]), "scale": np.array([1.0]), "skew": np.array([0.0]), "prob_zero": np.array([0.0])}
        draws = suitability._simulate(Distribution.pearson, np.array([[0.5]]), p, 1.0, origin=-1e-10)
        np.testing.assert_array_equal(draws, [[-1e-10]])

    @pytest.mark.parametrize(("origin", "resolution", "least_positive"), [(3000.0, 0.3, 0.3), (1000.0, 6.0, 4.0)])
    def test_gamma_rounds_small_positive_draws_to_the_least_positive_lattice_point(
        self, origin: float, resolution: float, least_positive: float
    ) -> None:
        uniforms = np.array([[0.1], [0.21]])
        p = {"alpha": np.array([1.0]), "beta": np.array([1.0]), "prob_zero": np.array([0.2])}
        draws = suitability._simulate(Distribution.gamma, uniforms, p, resolution, origin)
        np.testing.assert_array_equal(draws[:, 0], [0.0, least_positive])

    def test_the_draws_lie_on_the_lattice_of_the_prepared_sample_not_on_multiples_of_the_unit(self):
        # a SPEI sample of values recorded in 3s carries +1000 per step, so it lies on 1000 + 3k, not on 3k
        uniforms = np.random.default_rng(3).uniform(size=(2000, 1))
        p = {"loc": np.array([1000.0]), "scale": np.array([20.0]), "shape": np.array([0.0])}
        draws = suitability._simulate(Distribution.loglogistic, uniforms, p, 3.0, origin=1000.0)
        np.testing.assert_allclose((draws - 1000.0) / 3.0, np.round((draws - 1000.0) / 3.0), atol=1e-9)
        assert not np.allclose(draws / 3.0, np.round(draws / 3.0))
        # a zero-mass family keeps its exact zeros and puts a positive draw that rounds to zero on the least positive point
        gamma = {"alpha": np.array([0.8]), "beta": np.array([3.0]), "prob_zero": np.array([0.15])}
        shifted = suitability._simulate(Distribution.gamma, uniforms, gamma, 3.0, origin=1000.0)
        assert (shifted == 0).mean() == pytest.approx((uniforms <= 0.15).mean(), abs=1e-12)
        assert (shifted[shifted > 0] >= 1.0).all()  # 1000 % 3 = 1, the least positive point of 1000 + 3k

    def test_a_refit_with_a_value_beyond_its_support_fails_by_the_rule_of_an_observed_fit(self, monkeypatch):
        # a GEV with shape 0.5 ends at loc + scale / shape = 2: the second replicate holds a 5, which the likelihood
        # would score as zero, so that refit is a failed one and the first, wholly inside, is not
        class Fit:
            parameters = {"loc": np.zeros((1, 2)), "scale": np.ones((1, 2)), "shape": np.full((1, 2), 0.5)}

        monkeypatch.setattr(compute.FittedDistribution, "resolve", lambda *args, **kwargs: Fit())
        draws = np.stack([np.linspace(-1.0, 1.5, 10), np.linspace(-1.0, 1.5, 10)], axis=-1)[:, np.newaxis, :]
        draws[3, 0, 1] = 5.0
        period = types.SimpleNamespace(start_year=START, end_year=START + 9)
        statistics, succeeded = suitability._refit_statistics(Distribution.gev, draws, period, MONTHLY)
        assert succeeded.tolist() == [[True, False]]
        assert np.isfinite(statistics["ks"][0, 0])
        assert np.isnan(statistics["ks"][0, 1])

    def test_a_refit_with_a_saturated_cdf_inside_its_support_still_succeeds(self, monkeypatch):
        # a GEV with shape -0.5 starts at loc + scale / shape = -2: -1.95 is inside, with an underflowed CDF but a
        # finite log-density, which is how an observed fit decides it is inside
        class Fit:
            parameters = {"loc": np.zeros((1, 1)), "scale": np.ones((1, 1)), "shape": np.full((1, 1), -0.5)}

        monkeypatch.setattr(compute.FittedDistribution, "resolve", lambda *args, **kwargs: Fit())
        draws = np.linspace(-1.0, 1.5, 10).reshape(10, 1, 1)
        draws[0, 0, 0] = -1.95
        period = types.SimpleNamespace(start_year=START, end_year=START + 9)
        p = {key: value[..., np.newaxis] for key, value in Fit.parameters.items()}
        assert (
            suitability._continuous_cdf(Distribution.gev, np.array([-1.95]), {k: v[0] for k, v in p.items()})[0] == 0.0
        )
        _, succeeded = suitability._refit_statistics(Distribution.gev, draws, period, MONTHLY)
        assert succeeded.tolist() == [[True]]

    def test_the_stream_depends_on_the_values_not_on_how_a_value_is_stored(self):
        sample = np.random.default_rng(4).gamma(2.0, 30.0, size=(YEARS, 3))
        sample[3, 0] = np.nan
        sample[5, 1] = 0.0
        variants = {"positive nan": sample, "negative nan": sample.copy(), "negative zero": sample.copy()}
        variants["negative nan"][3, 0] = -np.nan
        variants["negative zero"][5, 1] = -0.0
        assert variants["negative nan"].tobytes() != sample.tobytes()
        assert variants["negative zero"].tobytes() != sample.tobytes()
        variants["big-endian"] = sample.astype(">f8")

        def first_draws(values):
            return [generator.random(4) for generator in suitability._column_generators(7, values)]

        expected = first_draws(sample)
        for name, values in variants.items():
            np.testing.assert_array_equal(first_draws(values), expected, err_msg=name)

    def test_simulated_samples_add_no_warning_or_log_event_of_their_own(self, monkeypatch):
        values = _gamma(20, 2, 2)
        values[:, 0, 0] = np.nan  # a cell with no value: its refits have an empty column

        class Recorder:
            """Stands in for the module logger, so the test does not depend on how logging is configured."""

            def __init__(self) -> None:
                self.events: list[str] = []

            def bind(self, **_fields: Any) -> Recorder:
                return self

            def __getattr__(self, _level: str) -> Any:
                return lambda event, **_kwargs: self.events.append(event)

        def run(**kwargs: Any) -> tuple[list[str], list[str]]:
            recorder = Recorder()
            monkeypatch.setattr(compute, "_logger", recorder)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _compare(values, spatial_time_major=True, candidates=GAMMA, **kwargs)
            return sorted(str(w.message) for w in caught), recorder.events

        without_bootstrap = run()
        assert (
            "distribution_fitting_started" in without_bootstrap[1]
        )  # the observed fit logs, so the next line means something
        assert run(n_bootstrap=6, seed=1) == without_bootstrap

    def test_the_empty_column_warning_stays_on_the_normal_path_and_is_dropped_in_a_refit(self, monkeypatch):
        monkeypatch.setattr(
            compute, "_native", None
        )  # the native gamma fit does not warn; the Python one is under test
        values = np.arange(1.0, 361.0).reshape(30, 12)
        values[:, 0] = np.nan

        def empty_slice_warnings() -> list[str]:
            with warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
                warnings.simplefilter("always")
                compute.gamma_parameters(values, 1895, 1895, 1924, MONTHLY)
            return [str(w.message) for w in caught if "Mean of empty slice" in str(w.message)]

        assert empty_slice_warnings()  # the documented outcome for a column with no value
        with compute._fit_warnings_suppressed():
            assert not empty_slice_warnings()
        assert empty_slice_warnings()

    def test_the_fit_warning_switch_is_scoped_to_its_thread(self):
        entered, release = threading.Event(), threading.Event()

        def hold() -> None:
            with compute._fit_warnings_suppressed():
                entered.set()
                release.wait(10)

        holder = threading.Thread(target=hold)
        holder.start()
        try:
            assert entered.wait(10)
            assert compute._FIT_WARNINGS_ENABLED.get() is True  # another thread's block does not reach this one
        finally:
            release.set()
            holder.join()

    def test_a_unit_far_coarser_than_the_data_is_reported_as_failed_refits(self):
        values = _gamma(21)
        continuous = _gof(values, candidates=GAMMA, n_bootstrap=5, seed=1)
        rounded = _gof(values, candidates=GAMMA, n_bootstrap=5, seed=1, resolution=1e9)
        assert np.isfinite(continuous.ks_p_value_bootstrap).all()
        assert (continuous.bootstrap_failed_share == 0).all()
        assert np.isnan(rounded.ks_p_value_bootstrap).all()
        assert np.isnan(rounded.ad_p_value_bootstrap).all()
        assert (rounded.bootstrap_failed_share == 1).all()

    def test_library_fit_warnings_describe_the_data_not_the_simulated_samples(self):
        bimodal = np.where(np.random.default_rng(0).random((YEARS * 12,)) < 0.5, 5.0, 200.0)
        bimodal = bimodal + np.random.default_rng(1).uniform(0, 1, bimodal.shape)

        def count(**kwargs: Any) -> int:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _compare(bimodal, **kwargs)
            return sum(issubclass(w.category, GoodnessOfFitWarning) for w in caught)

        assert count(n_bootstrap=0) > 0  # the data itself is a poor gamma fit
        assert count(n_bootstrap=25, seed=1) == count(n_bootstrap=0)

    def test_the_fit_warning_switch_is_scoped_to_its_block(self):
        bimodal = np.where(np.random.default_rng(0).random((YEARS, 12)) < 0.5, 5.0, 200.0) + np.random.default_rng(
            1
        ).uniform(0, 1, (YEARS, 12))

        def poor_fit_warnings() -> int:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                compute.FittedDistribution.resolve(bimodal, Distribution.gamma, START, START, END, MONTHLY)
            return sum(issubclass(w.category, GoodnessOfFitWarning) for w in caught)

        assert poor_fit_warnings() > 0
        with compute._fit_warnings_suppressed():
            assert poor_fit_warnings() == 0
        assert poor_fit_warnings() > 0


class TestAcrossCells:
    def _gof(self, **kwargs: Any) -> suitability.GoodnessOfFit:
        return _gof(_gamma(13, 3, 4), spatial_time_major=True, candidates=GAMMA, **kwargs)

    def test_rejection_share_counts_the_cells_below_alpha(self):
        gof = self._gof(alpha=0.3)
        assert gof.rejection_share.shape == (1, 12, 2)
        expected = (gof.ks_p_value_naive <= 0.3).mean(axis=(2, 3))
        np.testing.assert_allclose(gof.rejection_share[..., 0], expected)
        assert np.isnan(gof.rejection_share[..., 1]).all()  # Anderson-Darling has no naive p-value

    def test_rejection_share_follows_the_bootstrap_p_value_once_there_is_one(self):
        gof = self._gof(n_bootstrap=19, seed=2, alpha=0.2)
        for column, p in enumerate((gof.ks_p_value_bootstrap, gof.ad_p_value_bootstrap)):
            np.testing.assert_allclose(gof.rejection_share[..., column], (p <= 0.2).mean(axis=(2, 3)))
        assert np.isfinite(gof.rejection_share).all()

    def test_rejection_share_leaves_out_the_cells_without_a_p_value(self):
        values = _gamma(14, 2, 2)
        values[:, 0, 0] = np.nan
        gof = _gof(values, spatial_time_major=True, alpha=0.5, candidates=GAMMA)
        finite = np.isfinite(gof.ks_p_value_naive)
        expected = ((gof.ks_p_value_naive <= 0.5) & finite).sum(axis=(2, 3)) / finite.sum(axis=(2, 3))
        np.testing.assert_allclose(gof.rejection_share[..., 0], expected)

    def test_benjamini_hochberg_matches_scipy_per_candidate_and_step(self):
        gof = self._gof(n_bootstrap=19, seed=3, bh_adjust=True)
        for p, adjusted in (
            (gof.ks_p_value_bootstrap, gof.ks_p_value_bh),
            (gof.ad_p_value_bootstrap, gof.ad_p_value_bh),
        ):
            for candidate in range(p.shape[0]):
                for step in range(p.shape[1]):
                    expected = scipy.stats.false_discovery_control(p[candidate, step].ravel())
                    np.testing.assert_allclose(adjusted[candidate, step].ravel(), expected)
        assert (gof.ks_p_value_bh >= gof.ks_p_value_bootstrap - 1e-12).all()

    def test_without_a_bootstrap_the_k_s_p_value_is_adjusted_and_anderson_darling_has_none(self):
        gof = self._gof(bh_adjust=True)
        expected = scipy.stats.false_discovery_control(gof.ks_p_value_naive[0, 0].ravel())
        np.testing.assert_allclose(gof.ks_p_value_bh[0, 0].ravel(), expected)
        assert gof.ad_p_value_bh is None

    def test_a_nan_p_value_stays_out_of_the_family(self):
        values = _gamma(15, 2, 3)
        values[:, 0, 0] = np.nan
        gof = _gof(values, spatial_time_major=True, bh_adjust=True, candidates=GAMMA)
        assert np.isnan(gof.ks_p_value_bh[:, :, 0, 0]).all()
        finite = gof.ks_p_value_naive[0, 0].ravel()
        expected = np.full(finite.shape, np.nan)
        expected[np.isfinite(finite)] = scipy.stats.false_discovery_control(finite[np.isfinite(finite)])
        np.testing.assert_allclose(gof.ks_p_value_bh[0, 0].ravel(), expected, equal_nan=True)

    def test_bh_needs_a_cell_axis(self):
        values = _gamma(0)
        folded = values.reshape(YEARS, 12)
        with pytest.raises(InvalidArgumentError, match="cell"):
            _compare(values, bh_adjust=True)
        with pytest.raises(InvalidArgumentError, match="cell"):
            _compare(folded, bh_adjust=True)


class TestArguments:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"gof": "ks"},
            {"gof": ("ks", "chi2")},
            {"gof": ("ks", "ks")},
            {"gof": [np.array(["ks"])]},
            {"gof": 5},
            {"n_bootstrap": -1},
            {"n_bootstrap": 2.5},
            {"n_bootstrap": True},
            {"n_bootstrap": "9"},
            {"n_bootstrap": 5, "resolution": 0.0},
            {"n_bootstrap": 5, "resolution": -0.1},
            {"n_bootstrap": 5, "resolution": float("nan")},
            {"n_bootstrap": 5, "resolution": float("inf")},
            {"n_bootstrap": 5, "resolution": "0.1"},
            {"n_bootstrap": 5, "resolution": True},
            {"n_bootstrap": 5, "resolution": 10**400},  # beyond the float range
            {"n_bootstrap": 5, "resolution": 1e-320},  # a subnormal: its reciprocal overflows
            {"gof": ("ad",), "bh_adjust": True},  # nothing to adjust without a bootstrap
            {"alpha": 0.0},
            {"alpha": 1.0},
            {"alpha": float("nan")},
            {"alpha": "0.05"},
            {"seed": -1},
            {"seed": 1.5},
            {"seed": True},
            {"n_bootstrap": 5, "seed": np.random.default_rng(1)},
            {"bh_adjust": 1},
            {"gof": (), "n_bootstrap": 5},
            {"gof": (), "bh_adjust": True},
        ],
    )
    def test_invalid_goodness_of_fit_arguments_raise(self, kwargs):
        values = _gamma(0, 2, 2)
        with pytest.raises(InvalidArgumentError):
            _compare(values, spatial_time_major=True, **kwargs)

    def test_a_generator_is_refused_with_its_reason(self):
        values = _gamma(0)
        generator = np.random.default_rng(1)
        with pytest.raises(InvalidArgumentError, match="does not expose the seed"):
            _compare(values, n_bootstrap=5, seed=generator)

    def test_numpy_integers_are_accepted(self):
        result = _compare(_gamma(0), n_bootstrap=np.int64(5), seed=np.int64(3))
        assert result.goodness_of_fit.n_bootstrap == 5
        assert result.metadata["climate_indices_rng_seed"] == "3"

    def test_numpy_scalars_are_recorded_as_plain_types(self):
        result = _compare(
            _gamma(0), n_bootstrap=3, seed=np.int64(1), alpha=np.float32(0.05), resolution=np.float32(0.5)
        )
        record = json.loads(result.metadata["climate_indices_provenance"])["parameters"]
        assert record["alpha"] == pytest.approx(0.05, rel=1e-6)
        assert record["resolution"] == 0.5
        assert record["bh_adjust"] is False

    def test_bh_with_only_anderson_darling_needs_a_bootstrap(self):
        gof = _gof(_gamma(0, 2, 2), gof=("ad",), n_bootstrap=5, seed=1, bh_adjust=True, spatial_time_major=True)
        assert np.isfinite(gof.ad_p_value_bh).any()

    def test_resolution_and_seed_are_ignored_without_a_bootstrap(self):
        result = _compare(_gamma(0), resolution=0.1, seed=4)
        assert "climate_indices_rng_seed" not in result.metadata
        assert "resolution" not in json.loads(result.metadata["climate_indices_provenance"])["parameters"]


# ---- the xarray / Dask route ------------------------------------------------------------------------------


GOF_VARIABLES = ("ks_statistic", "ks_p_value_naive", "ad_statistic")
BOOTSTRAP_VARIABLES = ("ks_p_value_bootstrap", "ad_p_value_bootstrap", "bootstrap_failed_share")


def _assert_matches_numpy(dataset: xr.Dataset, core: suitability.DistributionComparison) -> None:
    gof = core.goodness_of_fit
    assert gof is not None
    for name in GOF_VARIABLES + BOOTSTRAP_VARIABLES + ("ks_p_value_bh", "ad_p_value_bh"):
        expected = getattr(gof, name)
        if expected is None:
            assert name not in dataset
        else:
            np.testing.assert_allclose(dataset[name].values, expected, rtol=1e-12, equal_nan=True, err_msg=name)
    if gof.rejection_share is None:
        assert "rejection_share" not in dataset
    else:
        assert dataset["rejection_share"].dims == ("candidate", "month", "statistic")
        assert list(dataset["statistic"].values) == ["ks", "ad"]
        np.testing.assert_allclose(dataset["rejection_share"].values, gof.rejection_share, equal_nan=True)


class TestXarrayRoute:
    OPTIONS: dict[str, Any] = {"n_bootstrap": 12, "seed": 21, "bh_adjust": True, "alpha": 0.3}

    def test_variables_match_the_numpy_route(self):
        grid = _grid()
        dataset = _compare(grid, **self.OPTIONS)
        core = _compare(grid.values, spatial_time_major=True, **self.OPTIONS)
        _assert_matches_numpy(dataset, core)
        assert dataset["ks_statistic"].dims == ("candidate", "month", "lat", "lon")
        assert dataset["ks_p_value_bh"].dims == dataset["ks_statistic"].dims

    @pytest.mark.parametrize("lazy", [False, True])
    def test_an_all_missing_cell_has_no_p_value_and_stays_out_of_the_family(self, lazy):
        grid = _grid().copy()
        grid[{"lat": 0, "lon": 0}] = np.nan
        data = grid.chunk({"time": -1, "lat": 1, "lon": 1}) if lazy else grid
        dataset = _compare(data, **self.OPTIONS).compute()
        _assert_matches_numpy(dataset, _compare(grid.values, spatial_time_major=True, **self.OPTIONS))
        assert np.isnan(dataset["ks_p_value_bh"].isel(lat=0, lon=0)).all()
        assert np.isfinite(dataset["ks_p_value_bh"].sel(candidate="gamma").isel(lat=1, lon=1)).all()

    @pytest.mark.parametrize("gof", [("ks",), ("ad",)])
    def test_a_statistic_left_out_matches_the_numpy_route(self, gof):
        grid = _grid()
        options: dict[str, Any] = {"n_bootstrap": 6, "seed": 3, "gof": gof, "bh_adjust": True}
        _assert_matches_numpy(_compare(grid, **options), _compare(grid.values, spatial_time_major=True, **options))

    def test_bh_without_a_bootstrap_adjusts_the_naive_k_s_p_value_alone(self):
        grid = _grid()
        dataset = _compare(grid, bh_adjust=True)
        assert "ad_p_value_bh" not in dataset
        _assert_matches_numpy(dataset, _compare(grid.values, spatial_time_major=True, bh_adjust=True))

    def test_daily_periodicity_runs_over_the_day_of_year(self):
        times = pd.date_range(f"{START}-01-01", f"{END}-12-31", freq="D")
        series = xr.DataArray(
            np.random.default_rng(8).gamma(2.0, 30.0, size=(len(times), 2)),
            dims=("time", "lat"),
            coords={"time": times},
        )
        dataset = _compare(
            series, periodicity=compute.Periodicity.daily, candidates=GAMMA, n_bootstrap=3, seed=1, bh_adjust=True
        )
        assert dataset["ks_p_value_bootstrap"].dims == ("candidate", "dayofyear", "lat")
        assert dataset.sizes["dayofyear"] == 366
        assert dataset["rejection_share"].dims == ("candidate", "dayofyear", "statistic")
        assert np.isfinite(dataset["ks_p_value_bh"]).sum() > 0

    def test_default_call_adds_the_statistics_and_the_naive_p_value(self):
        dataset = _compare(_grid())
        for name in GOF_VARIABLES + ("rejection_share",):
            assert name in dataset
        for name in BOOTSTRAP_VARIABLES + ("ks_p_value_bh", "ad_p_value_bh"):
            assert name not in dataset
        assert dataset["ks_statistic"].attrs["long_name"].startswith("Kolmogorov-Smirnov")

    def test_an_empty_gof_adds_no_variable(self):
        dataset = _compare(_grid(), gof=())
        for name in GOF_VARIABLES + BOOTSTRAP_VARIABLES + ("rejection_share",):
            assert name not in dataset
        assert "climate_indices_gof_bootstrap_samples" not in dataset["aic"].attrs

    def test_a_single_series_has_no_across_cell_variable(self):
        series = _grid().isel(lat=1, lon=1).assign_coords(statistic="input")
        dataset = _compare(series, n_bootstrap=6, seed=2)
        assert "ks_p_value_bootstrap" in dataset
        assert "rejection_share" not in dataset
        assert dataset.coords["statistic"].item() == "input"
        _assert_matches_numpy(dataset, _compare(series.values, n_bootstrap=6, seed=2))
        with pytest.raises(InvalidArgumentError, match="cell"):
            _compare(series, bh_adjust=True)

    def test_two_dimensional_input_is_read_cell_by_cell(self):
        grid = _grid().isel(lon=0)
        dataset = _compare(grid, n_bootstrap=6, seed=2, bh_adjust=True, candidates=GAMMA)
        assert dataset["ks_p_value_bh"].dims == dataset["ks_statistic"].dims
        for cell in range(grid.sizes["lat"]):
            single = _compare(grid.isel(lat=cell).values, n_bootstrap=6, seed=2, candidates=GAMMA)
            np.testing.assert_allclose(
                dataset["ks_p_value_bootstrap"].isel(lat=cell).values, single.goodness_of_fit.ks_p_value_bootstrap
            )
        expected = scipy.stats.false_discovery_control(dataset["ks_p_value_bootstrap"].values[0, 0])
        np.testing.assert_allclose(dataset["ks_p_value_bh"].values[0, 0], expected)

    @pytest.mark.parametrize("chunks", [{"time": -1, "lat": 1, "lon": 1}, {"time": -1, "lat": 2, "lon": 1}])
    def test_dask_chunking_does_not_change_the_result(self, chunks):
        grid = _grid(seed=3)
        eager = _compare(grid, **self.OPTIONS)
        lazy = _compare(grid.chunk(chunks), **self.OPTIONS)
        assert lazy["ks_p_value_bh"].chunks is not None
        for name in GOF_VARIABLES + BOOTSTRAP_VARIABLES + ("ks_p_value_bh", "ad_p_value_bh", "rejection_share"):
            # a block's floating-point results differ from a whole grid's by rounding only; the stream is the same
            np.testing.assert_allclose(lazy[name].values, eager[name].values, rtol=1e-9, atol=1e-12, err_msg=name)

    def test_an_unseeded_call_records_one_seed_for_every_block(self):
        lazy = _compare(_grid(seed=4).chunk({"time": -1, "lat": 1, "lon": 1}), n_bootstrap=8).compute()
        seeds = {lazy[name].attrs["climate_indices_rng_seed"] for name in GOF_VARIABLES + BOOTSTRAP_VARIABLES}
        assert len(seeds) == 1
        replay = _compare(_grid(seed=4), n_bootstrap=8, seed=int(seeds.pop()))
        np.testing.assert_array_equal(replay["ks_p_value_bootstrap"].values, lazy["ks_p_value_bootstrap"].values)

    def test_attributes_name_the_bootstrap_and_the_seed(self):
        dataset = _compare(_grid(), n_bootstrap=7, seed=2**100 + 5, resolution=0.5)
        for name in ("aic", "ks_statistic", "ks_p_value_bootstrap", "rejection_share"):
            attrs = dataset[name].attrs
            assert attrs["climate_indices_gof_bootstrap_samples"] == "7", name
            assert attrs["climate_indices_rng_seed"] == str(2**100 + 5), name
        record = json.loads(dataset["ks_statistic"].attrs["climate_indices_provenance"])["parameters"]
        assert record["resolution"] == 0.5
        assert record["n_bootstrap"] == 7

    def test_netcdf_round_trip_keeps_the_values_and_a_seed_above_64_bits(self, tmp_path: Path):
        dataset = _compare(_grid(), n_bootstrap=5, seed=2**100 + 5, bh_adjust=True).compute()
        path = tmp_path / "gof.nc"
        dataset.to_netcdf(path, engine="h5netcdf")
        with xr.open_dataset(path, engine="h5netcdf") as reopened:
            for name in ("ks_statistic", "ks_p_value_bootstrap", "ks_p_value_bh", "rejection_share"):
                np.testing.assert_allclose(reopened[name].values, dataset[name].values, equal_nan=True)
            assert reopened["ks_p_value_bootstrap"].attrs["climate_indices_rng_seed"] == str(2**100 + 5)

    def test_an_input_dimension_named_like_an_output_is_refused(self):
        renamed = _grid().rename(lon="statistic")
        with pytest.raises(InvalidArgumentError, match="named like an output"):
            _compare(renamed)

    def test_bh_names_are_reserved_when_emitted(self):
        grid = _grid().assign_coords(ks_p_value_bh="input")
        assert _compare(grid).coords["ks_p_value_bh"].item() == "input"
        with pytest.raises(InvalidArgumentError, match="ks_p_value_bh"):
            _compare(grid, bh_adjust=True)

        grid = _grid().assign_coords(ad_p_value_bh="input")
        assert _compare(grid, bh_adjust=True).coords["ad_p_value_bh"].item() == "input"
        with pytest.raises(InvalidArgumentError, match="ad_p_value_bh"):
            _compare(grid, bh_adjust=True, n_bootstrap=3)

    def test_arguments_are_validated_on_this_route_too(self):
        grid = _grid()
        with pytest.raises(InvalidArgumentError):
            _compare(grid, n_bootstrap=-2)


# ---- size under the null: characterization of the bootstrap p-values ---------------------------------------


def _draws(distribution: Distribution, years: int, cells: tuple[int, ...], seed: int) -> tuple[np.ndarray, str, dict]:
    """A record whose every calendar step and cell is iid from ``distribution``, with its route and arguments."""
    shape = (years * 12, *cells)
    rng = np.random.default_rng(seed)
    if distribution is Distribution.gamma:
        return rng.gamma(2.0, 30.0, size=shape), "spi", {}
    if distribution is Distribution.pearson:
        return scipy.stats.pearson3.rvs(1.0, loc=100.0, scale=30.0, size=shape, random_state=rng), "spi", {}
    draw = (
        scipy.stats.fisk.rvs(c=1 / 0.3, scale=60.0, size=shape, random_state=rng)
        if distribution is Distribution.loglogistic
        else scipy.stats.genextreme.rvs(-0.1, loc=100.0, scale=30.0, size=shape, random_state=rng)
    )
    # PET of zero makes the water balance, and so the SPEI sample, the draw itself
    return draw, "spei", {"pet_mm": np.zeros(shape)}


N_BOOTSTRAP, ALPHA, ROUNDING = 99, 0.05, 10.0
# These fast regression checks do not replace the 399-draw acceptance experiment in the characterization script.
SIZE_REGRESSION_BOUNDS = (0.02, 0.09)


def _rejection_rates(
    distribution: Distribution, *, rounded: bool = False, resolution: float | None = None, seed: int = 0
):
    """Observed K-S and A^2 rejection rates among finite bootstrap p-values."""
    values, index, extra = _draws(distribution, YEARS, (6, 10), seed)
    if rounded:
        values = np.round(values / ROUNDING) * ROUNDING
    gof = _compare(
        values,
        index=index,
        candidates=(distribution,),
        n_bootstrap=N_BOOTSTRAP,
        seed=seed + 100,
        spatial_time_major=True,
        resolution=resolution,
        **extra,
    ).goodness_of_fit
    return np.array(
        [(p <= ALPHA).sum() / np.isfinite(p).sum() for p in (gof.ks_p_value_bootstrap, gof.ad_p_value_bootstrap)]
    )


class TestSizeUnderTheNull:
    """Fixed-seed regression checks on empirical size for the measured cases."""

    @pytest.mark.parametrize(
        "distribution",
        [Distribution.gamma, Distribution.pearson, Distribution.loglogistic, Distribution.gev],
        ids=lambda d: d.value,
    )
    def test_continuous_record(self, distribution):
        rates = _rejection_rates(distribution)
        low, high = SIZE_REGRESSION_BOUNDS
        assert ((low <= rates) & (rates <= high)).all(), rates

    def test_rounded_record_with_its_resolution(self):
        rates = _rejection_rates(Distribution.gamma, rounded=True, resolution=ROUNDING)
        low, high = SIZE_REGRESSION_BOUNDS
        assert ((low <= rates) & (rates <= high)).all(), rates

    def test_rounded_record_without_its_resolution_rejects_far_too_often(self):
        # a continuous null does not describe a record with ties, so the observed statistics look extreme
        rates = _rejection_rates(Distribution.gamma, rounded=True)
        assert (rates > 2 * ALPHA).all(), rates

    def test_naive_k_s_under_rejects_for_estimated_parameters(self):
        values = _gamma(30, 6, 10)
        gof = _gof(values, candidates=(Distribution.gamma,), spatial_time_major=True)
        tests = int(np.isfinite(gof.ks_p_value_naive).sum())
        rejected = int((gof.ks_p_value_naive <= ALPHA).sum())
        # a calibrated test rejects about 5% of the series, and 99.9% of the time at least this many
        assert rejected < scipy.stats.binom.interval(0.999, tests, ALPHA)[0]
        assert np.nanmean(gof.ks_p_value_naive) > 0.6  # uniform p-values would average 0.5


# ---- cross-check against SciPy for gamma ---------------------------------------------------------------


class TestAgainstScipy:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_gamma_bootstrap_agrees_with_scipy_goodness_of_fit_in_distribution(self, seed):
        # SciPy refits by maximum likelihood and the library by Thom's approximation, so the p-values agree to
        # Monte-Carlo and estimator error, not exactly. A bimodal sample is a poor gamma fit that both reject
        # at the 1 / 200 floor of 199 samples; a gamma sample is rejected by neither.
        rng = np.random.default_rng(100 + seed)
        samples = {
            "gamma": _gamma(seed)[0::12],
            "bimodal": np.concatenate([rng.normal(10, 1, 20), rng.normal(100, 5, 20)]),
        }
        for name, sample in samples.items():
            values = np.repeat(sample, 12)  # every calendar step holds the same sample
            ours = _gof(values, candidates=(Distribution.gamma,), n_bootstrap=199, seed=seed)
            reference = scipy.stats.goodness_of_fit(
                scipy.stats.gamma,
                sample,
                known_params={"loc": 0.0},
                statistic="ad",
                n_mc_samples=199,
                rng=np.random.default_rng(seed),
            )
            p_ours, p_reference = ours.ad_p_value_bootstrap[0, 0], reference.pvalue
            assert abs(p_ours - p_reference) < 0.1, name
            assert ours.ad_statistic[0, 0] == pytest.approx(reference.statistic, rel=0.05), name
            assert (p_ours < 0.05) == (p_reference < 0.05) == (name == "bimodal"), (name, p_ours, p_reference)
