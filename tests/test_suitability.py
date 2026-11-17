"""Tests for ``climate_indices.suitability.compare_distributions`` (NumPy core, #1584, ADR-0028)."""

from __future__ import annotations

import dataclasses
import json
import warnings
from typing import Any

import numpy as np
import pytest
import scipy.stats

import climate_indices
from climate_indices import compute, suitability
from climate_indices.drought import indices
from climate_indices.exceptions import (
    BetaFeatureWarning,
    CalibrationPeriodClampedWarning,
    ClimateIndicesWarning,
    DistributionFittingError,
    InvalidArgumentError,
    ShortCalibrationWarning,
)
from climate_indices.vocabulary import Distribution

MONTHLY = compute.Periodicity.monthly
START, CAL_START, CAL_END = 1950, 1951, 1990  # 40 calibration years of a 60-year record
YEARS = 60

pytestmark = pytest.mark.filterwarnings("ignore::climate_indices.exceptions.BetaFeatureWarning")


def _precip(seed: int = 0, years: int = YEARS, shape: tuple[int, ...] = ()) -> np.ndarray:
    """Seasonal gamma monthly precipitation, (years * 12, *shape)."""
    rng = np.random.default_rng(seed)
    season = 40.0 + 30.0 * np.sin(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    base = np.tile(season, years).reshape((years * 12,) + (1,) * len(shape))
    return rng.gamma(shape=2.0, scale=base / 2.0, size=(years * 12, *shape))


def _compare(values, scale=1, *, index: Any = "spi", **kwargs):
    kwargs.setdefault("data_start_year", START)
    kwargs.setdefault("calibration_year_initial", CAL_START)
    kwargs.setdefault("calibration_year_final", CAL_END)
    kwargs.setdefault("periodicity", MONTHLY)
    return suitability.compare_distributions(values, scale, index=index, **kwargs)


def _calibration_sample(values: np.ndarray, scale=1):
    """The scaled calibration sample, (years, 12), recomputed independently of the module."""
    scaled = compute.prepare_scaled(values, scale, MONTHLY)
    return scaled[CAL_START - START : CAL_END - START + 1]


def _spei_inputs(*, k: float, seed: int = 1, years: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """Precipitation and PET whose SPEI water balance is generalized-logistic with Hosking shape ``k``.

    k < 0: precipitation is a ``fisk`` draw and PET is zero. k > 0: precipitation is zero and
    PET is a ``fisk`` draw, so the balance ``1000 - PET`` is the reflected distribution.
    """
    draw = np.asarray(scipy.stats.fisk.rvs(c=1 / 0.3, scale=60.0, size=years * 12, random_state=seed), dtype=float)
    zeros = np.zeros(years * 12)
    return (draw, zeros) if k < 0 else (zeros, draw)


class TestSurface:
    def test_beta_warning(self):
        values = _precip()
        with pytest.warns(BetaFeatureWarning):
            suitability.compare_distributions(
                values,
                1,
                index="spi",
                data_start_year=START,
                calibration_year_initial=CAL_START,
                calibration_year_final=CAL_END,
                periodicity=MONTHLY,
            )

    def test_no_root_reexport(self):
        assert "suitability" not in climate_indices.__all__

    def test_result_is_frozen(self):
        result = _compare(_precip())
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.best = None  # type: ignore[misc]

    def test_results_do_not_compare_by_value(self):
        # the provenance record carries a creation timestamp, so value equality is not defined
        first, second = _compare(_precip()), _compare(_precip())
        assert first != second


class TestCandidates:
    def test_default_sets_in_enum_order(self):
        spi = _compare(_precip())
        assert spi.candidates == (Distribution.pearson, Distribution.gamma)
        precip, pet = _spei_inputs(k=-0.3)
        spei = _compare(precip, index="spei", pet_mm=pet)
        assert spei.candidates == (
            Distribution.pearson,
            Distribution.gamma,
            Distribution.loglogistic,
            Distribution.gev,
        )

    def test_user_candidates_are_sorted_to_enum_order(self):
        result = _compare(_precip(), candidates=(Distribution.gamma, Distribution.pearson))
        assert result.candidates == (Distribution.pearson, Distribution.gamma)

    @pytest.mark.parametrize("name", ["empirical", "beta", "t_location_scale", "normal"])
    def test_non_likelihood_candidate_raises(self, name):
        values, candidate = _precip(), Distribution(name)
        with pytest.raises(InvalidArgumentError):
            _compare(values, candidates=(candidate,))

    @pytest.mark.parametrize("distribution", [Distribution.loglogistic, Distribution.gev])
    def test_spei_only_families_rejected_outside_spei(self, distribution):
        values = _precip()
        with pytest.raises(InvalidArgumentError):
            _compare(values, candidates=(distribution,))

    def test_duplicate_or_empty_candidates_raise(self):
        values = _precip()
        with pytest.raises(InvalidArgumentError):
            _compare(values, candidates=(Distribution.gamma, Distribution.gamma))
        with pytest.raises(InvalidArgumentError):
            _compare(values, candidates=())


class TestArguments:
    def test_spei_requires_pet(self):
        values = _precip()
        with pytest.raises(InvalidArgumentError):
            _compare(values, index="spei")

    def test_pet_only_for_spei(self):
        values, pet = _precip(), _precip(1)
        with pytest.raises(InvalidArgumentError):
            _compare(values, pet_mm=pet)

    @pytest.mark.parametrize("option", [{"aggregation": "mean"}, {"clip_negatives": False}])
    def test_standardized_index_options_only_on_that_route(self, option):
        values = _precip()
        with pytest.raises(InvalidArgumentError):
            _compare(values, index="spi", **option)

    def test_bad_route_and_criterion(self):
        values = _precip()
        with pytest.raises(InvalidArgumentError):
            _compare(values, index="pdsi")
        with pytest.raises(InvalidArgumentError):
            _compare(values, criterion="hqic")

    def test_criterion_must_be_a_string(self):
        values, criterion = _precip(), np.array("aicc")
        with pytest.raises(InvalidArgumentError, match="criterion"):
            _compare(values, criterion=criterion)

    @pytest.mark.parametrize("option", [{"aggregation": ["sum"]}, {"clip_negatives": "false"}])
    def test_options_must_have_their_documented_types(self, option):
        values = _precip()
        with pytest.raises(InvalidArgumentError):
            _compare(values, index="standardized_index", **option)

    def test_pet_shape_must_line_up_with_the_cells(self):
        values, pet = _precip(11).reshape(YEARS, 12), np.zeros((YEARS, 12))
        with pytest.raises(InvalidArgumentError, match="pet_mm"):
            _compare(values, index="spei", pet_mm=pet.T)

    def test_pet_must_keep_the_precipitation_time_length(self):
        block, pet = _precip(13, shape=(2, 2)), np.full((1, 2, 2), 800.0)
        with pytest.raises(InvalidArgumentError, match="pet_mm"):
            _compare(block, index="spei", pet_mm=pet, spatial_time_major=True)

    def test_spatial_block_pet_broadcasts_within_the_cells(self):
        block = _precip(12, shape=(2, 2))
        series = _compare(block, index="spei", pet_mm=np.full(YEARS * 12, 800.0), spatial_time_major=True)
        singleton = _compare(block, index="spei", pet_mm=np.full((YEARS * 12, 1, 1), 800.0), spatial_time_major=True)
        assert series.best.shape == singleton.best.shape == (12, 2, 2)
        np.testing.assert_array_equal(series.best, singleton.best)

    def test_signed_series_needs_clipping(self):
        signed = _precip() - 40.0
        with pytest.raises(InvalidArgumentError):
            _compare(signed, index="standardized_index", clip_negatives=False)

    def test_hourly_rejected(self):
        values = _precip()
        with pytest.raises(InvalidArgumentError):
            _compare(values, periodicity=compute.Periodicity.hourly)

    def test_all_missing_input_raises(self):
        values = np.full(YEARS * 12, np.nan)
        with pytest.raises(ValueError, match="no valid"):
            _compare(values)


class TestLogLikelihood:
    """Log-likelihood against SciPy at the library's own parameters, rtol=1e-10 (ADR-0028 §2)."""

    def test_gamma_matches_scipy(self):
        values = _precip(3)
        result = _compare(values, 3)
        sample = _calibration_sample(values, 3)
        g = result.candidates.index(Distribution.gamma)
        params = result.parameters[Distribution.gamma]
        for step in range(12):
            x = sample[:, step]
            x = x[~np.isnan(x)]
            expected = scipy.stats.gamma.logpdf(x, a=params["alpha"][step], scale=params["beta"][step]).sum()
            np.testing.assert_allclose(result.log_likelihood[g, step], expected, rtol=1e-10)
            assert result.n[step] == x.size

    def test_pearson_matches_scipy(self):
        values = _precip(4)
        result = _compare(values)
        sample = _calibration_sample(values)
        p = result.candidates.index(Distribution.pearson)
        params = result.parameters[Distribution.pearson]
        for step in range(12):
            x = sample[:, step]
            expected = scipy.stats.pearson3.logpdf(
                x, params["skew"][step], loc=params["loc"][step], scale=params["scale"][step]
            ).sum()
            np.testing.assert_allclose(result.log_likelihood[p, step], expected, rtol=1e-10)

    def test_zero_mass_term(self):
        values = _precip(5)
        # force exact zeros into one calendar step of the calibration period
        zeroed = values.copy()
        zeroed[(CAL_START - START) * 12 + 2 : (CAL_START - START + 6) * 12 : 12] = 0.0
        result = _compare(zeroed)
        sample = _calibration_sample(zeroed)
        g = result.candidates.index(Distribution.gamma)
        params = result.parameters[Distribution.gamma]
        x = sample[:, 2]
        n0, n = int((x == 0).sum()), x.size
        assert n0 == 6
        p0 = params["prob_zero"][2]
        np.testing.assert_allclose(p0, n0 / n, rtol=1e-12)
        expected = (
            scipy.stats.gamma.logpdf(x[x > 0], a=params["alpha"][2], scale=params["beta"][2]).sum()
            + (n - n0) * np.log1p(-p0)
            + n0 * np.log(p0)
        )
        np.testing.assert_allclose(result.log_likelihood[g, 2], expected, rtol=1e-10)
        assert result.n_parameters[g, 2] == 3  # gamma 2 + p0
        assert result.n_parameters[g, 3] == 2  # a step without zeros has no p0 term

    def test_glo_negative_shape_matches_fisk(self):
        precip, pet = _spei_inputs(k=-0.3)
        result = _compare(precip, index="spei", pet_mm=pet, calibration_year_final=START + 149)
        params = result.parameters[Distribution.loglogistic]
        assert np.all(params["shape"] < 0)
        sample = compute.prepare_scaled(precip - pet + 1000.0, 1, MONTHLY, clip_negatives=False)
        sample = sample[CAL_START - START :]
        loc_, alpha, k = params["loc"], params["scale"], params["shape"]
        c = result.candidates.index(Distribution.loglogistic)
        for step in range(12):
            expected = scipy.stats.fisk.logpdf(
                sample[:, step],
                c=1 / abs(k[step]),
                loc=loc_[step] + alpha[step] / k[step],
                scale=alpha[step] / abs(k[step]),
            ).sum()
            np.testing.assert_allclose(result.log_likelihood[c, step], expected, rtol=1e-10)

    def test_glo_positive_shape_matches_reflected_fisk(self):
        precip, pet = _spei_inputs(k=0.3)
        result = _compare(precip, index="spei", pet_mm=pet, calibration_year_final=START + 149)
        params = result.parameters[Distribution.loglogistic]
        assert np.all(params["shape"] > 0)
        sample = compute.prepare_scaled(precip - pet + 1000.0, 1, MONTHLY, clip_negatives=False)
        sample = sample[CAL_START - START :]
        loc_, alpha, k = params["loc"], params["scale"], params["shape"]
        c = result.candidates.index(Distribution.loglogistic)
        for step in range(12):
            # GLO(xi, alpha, k) reflects to GLO(-xi, alpha, -k): -X is a k < 0 GLO, which fisk covers
            expected = scipy.stats.fisk.logpdf(
                -sample[:, step],
                c=1 / k[step],
                loc=-loc_[step] - alpha[step] / k[step],
                scale=alpha[step] / k[step],
            ).sum()
            np.testing.assert_allclose(result.log_likelihood[c, step], expected, rtol=1e-10)


class TestCriteria:
    def test_formulas(self):
        result = _compare(_precip(6))
        ll, k = result.log_likelihood, result.n_parameters.astype(float)
        n = result.n.astype(float)
        ok = ~result.fit_failed  # a failed candidate has NaN criteria by contract
        assert ok.any()
        np.testing.assert_allclose(result.aic[ok], (2 * k - 2 * ll)[ok], rtol=1e-12)
        np.testing.assert_allclose(result.bic[ok], (k * np.log(n) - 2 * ll)[ok], rtol=1e-12)
        np.testing.assert_allclose(result.aicc[ok], (result.aic + 2 * k * (k + 1) / (n - k - 1))[ok], rtol=1e-12)
        assert np.all(np.isnan(result.aic[~ok]))

    def test_aicc_nan_when_n_not_above_k_plus_one(self):
        # 3 calibration years: n = 3 <= k + 1 = 3 for gamma without zeros
        values = _precip(7)
        with pytest.warns(ShortCalibrationWarning):
            result = _compare(values, candidates=(Distribution.gamma,), calibration_year_final=CAL_START + 2)
        assert np.all(np.isnan(result.aicc))
        assert np.all(np.isfinite(result.aic))
        assert np.all(result.best == -1)
        assert np.all(~result.fit_failed)
        with pytest.warns(ShortCalibrationWarning):
            by_aic = _compare(
                values, candidates=(Distribution.gamma,), calibration_year_final=CAL_START + 2, criterion="aic"
            )
        assert np.all(by_aic.best == 0)

    @pytest.mark.parametrize("criterion", ["aicc", "aic", "bic"])
    def test_best_is_argmin_of_requested_criterion(self, criterion):
        result = _compare(_precip(8), criterion=criterion)
        crit = getattr(result, criterion)
        key = np.where(np.isnan(crit), np.inf, crit)
        expected = np.where(np.isnan(crit).all(axis=0), -1, np.argmin(key, axis=0))
        np.testing.assert_array_equal(result.best, expected)
        assert result.criterion == criterion
        ordered = np.sort(key, axis=0)
        both = (~np.isnan(crit)).sum(axis=0) == 2
        assert both.any()
        np.testing.assert_allclose(result.delta_to_runner_up[both], (ordered[1] - ordered[0])[both], rtol=1e-12)
        assert np.all(np.isnan(result.delta_to_runner_up[~both]))


class TestOutsideSupport:
    """A sample value outside a fitted support has zero likelihood: failed, never ranked (ADR-0028 section 4)."""

    def test_zero_likelihood_candidate_is_failed_but_visible(self):
        result = _compare(_precip(0))
        p = result.candidates.index(Distribution.pearson)
        cells = np.flatnonzero(np.isneginf(result.log_likelihood[p]))
        assert cells.size  # Pearson's L-moment lower bound can sit above the sample minimum
        step = int(cells[0])
        assert result.fit_failed[p, step]
        assert np.isnan([result.aic[p, step], result.aicc[p, step], result.bic[p, step]]).all()
        assert result.best[step] == result.candidates.index(Distribution.gamma)
        assert np.isnan(result.delta_to_runner_up[step])  # one eligible candidate has no runner-up

    def test_sole_zero_likelihood_candidate_leaves_no_best(self):
        both = _compare(_precip(0))
        step = int(np.flatnonzero(np.isneginf(both.log_likelihood[0]))[0])
        only = _compare(_precip(0), candidates=(Distribution.pearson,))
        assert only.best[step] == -1


class TestSelectBest:
    def test_ties_go_to_the_earlier_candidate(self):
        crit = np.array([[5.0, 3.0], [5.0, 3.0], [9.0, 3.0]])
        best, delta = suitability._select_best(crit, np.ones_like(crit, dtype=bool))
        np.testing.assert_array_equal(best, [0, 0])
        np.testing.assert_array_equal(delta, [0.0, 0.0])

    def test_ineligible_candidates_are_skipped(self):
        crit = np.array([[1.0, np.nan], [4.0, np.nan], [2.0, 7.0]])
        eligible = np.array([[False, False], [True, False], [True, True]])
        best, delta = suitability._select_best(crit, eligible)
        np.testing.assert_array_equal(best, [2, 2])
        assert delta[0] == 2.0
        assert np.isnan(delta[1])  # a single eligible candidate has no runner-up

    def test_none_eligible_is_minus_one(self):
        crit = np.full((2, 3), np.nan)
        best, delta = suitability._select_best(crit, np.zeros_like(crit, dtype=bool))
        np.testing.assert_array_equal(best, [-1, -1, -1])
        assert np.all(np.isnan(delta))

    def test_all_infinite_picks_first_eligible(self):
        crit = np.array([[np.inf], [np.inf]])
        best, _ = suitability._select_best(crit, np.array([[False], [True]]))
        np.testing.assert_array_equal(best, [1])


class TestRoundTrip:
    """The returned parameters reproduce the direct index call bit for bit."""

    @pytest.mark.parametrize("scale", [1, 3])
    @pytest.mark.parametrize("distribution", [Distribution.gamma, Distribution.pearson])
    def test_spi(self, distribution, scale):
        values = _precip(9)
        result = _compare(values, scale)
        direct = indices.spi(values, scale, distribution, START, CAL_START, CAL_END, MONTHLY)
        reused = indices.spi(
            values,
            scale,
            distribution,
            START,
            CAL_START,
            CAL_END,
            MONTHLY,
            fitting_params=result.parameters[distribution],
        )
        np.testing.assert_array_equal(reused, direct)

    @pytest.mark.parametrize("distribution", [Distribution.gamma, Distribution.pearson])
    def test_standardized_index(self, distribution):
        values = _precip(10)
        result = _compare(values, 2, index="standardized_index", aggregation="mean")
        direct = indices.standardized_index(
            values, 2, distribution, START, CAL_START, CAL_END, MONTHLY, aggregation="mean"
        )
        reused = indices.standardized_index(
            values,
            2,
            distribution,
            START,
            CAL_START,
            CAL_END,
            MONTHLY,
            aggregation="mean",
            fitting_params=result.parameters[distribution],
        )
        np.testing.assert_array_equal(reused, direct)

    @pytest.mark.parametrize(
        "distribution", [Distribution.gamma, Distribution.pearson, Distribution.loglogistic, Distribution.gev]
    )
    def test_spei(self, distribution):
        precip = _precip(11)
        pet = _precip(12) * 0.8
        result = _compare(precip, 3, index="spei", pet_mm=pet)
        direct = indices.spei(precip, pet, 3, distribution, MONTHLY, START, CAL_START, CAL_END)
        reused = indices.spei(
            precip,
            pet,
            3,
            distribution,
            MONTHLY,
            START,
            CAL_START,
            CAL_END,
            fitting_params=result.parameters[distribution],
        )
        np.testing.assert_array_equal(reused, direct)


class TestEdgeCases:
    @pytest.mark.parametrize("candidates", [None, (Distribution.loglogistic,)])
    def test_spei_exact_zero_balance_is_scored_within_fitted_support(self, candidates):
        pet = np.repeat(np.linspace(980.0, 1020.0, 41), 12)
        precip = np.zeros_like(pet)
        result = _compare(
            precip,
            index="spei",
            pet_mm=pet,
            candidates=candidates,
            calibration_year_initial=START,
            calibration_year_final=START + 40,
        )
        glo = result.candidates.index(Distribution.loglogistic)
        params = result.parameters[Distribution.loglogistic]
        sample = (1000.0 - pet).reshape(41, 12)
        np.testing.assert_allclose(params["shape"], 0.0, atol=1e-12)
        expected = scipy.stats.logistic.logpdf(sample, loc=params["loc"], scale=params["scale"]).sum(axis=0)
        np.testing.assert_allclose(result.log_likelihood[glo], expected, rtol=1e-10)
        assert not result.fit_failed[glo].any()
        assert np.isfinite(result.aicc[glo]).all()
        np.testing.assert_array_equal(result.n_parameters[glo], np.full(12, 3))
        if candidates is not None:
            np.testing.assert_array_equal(result.best, np.zeros(12, dtype=int))

    def test_pearson_failure_is_recorded_not_substituted(self):
        values = _precip(15)
        # leave only three non-zero calibration values in step 7: Pearson needs four
        cal_rows = np.arange((CAL_START - START) * 12 + 7, (CAL_END - START + 1) * 12, 12)
        values[cal_rows[3:]] = 0.0
        result = _compare(values)
        p = result.candidates.index(Distribution.pearson)
        assert result.fit_failed[p, 7]
        assert not np.delete(result.fit_failed[p], 7).any()
        assert result.best[7] != p
        params = result.parameters[Distribution.pearson]
        assert (params["loc"][7], params["scale"][7], params["skew"][7]) == (0.0, 0.0, 0.0)
        assert np.isnan(result.log_likelihood[p, 7])
        assert np.isnan(result.aicc[p, 7])

    def test_nan_cell_in_spatial_block(self):
        block = _precip(16, shape=(2, 2))
        block[:, 0, 0] = np.nan
        result = _compare(block, spatial_time_major=True)
        assert result.best.shape == (12, 2, 2)
        assert result.log_likelihood.shape == (2, 12, 2, 2)
        assert np.all(result.n[:, 0, 0] == 0)
        assert result.fit_failed[:, :, 0, 0].all()
        assert np.all(result.best[:, 0, 0] == -1)
        assert np.all(np.isnan(result.delta_to_runner_up[:, 0, 0]))
        # a healthy cell equals the same series compared on its own
        alone = _compare(block[:, 1, 1])
        np.testing.assert_allclose(result.log_likelihood[:, :, 1, 1], alone.log_likelihood, rtol=1e-12)
        np.testing.assert_array_equal(result.best[:, 1, 1], alone.best)

    def test_fixture_series_with_missing_values(self, precips_mm_monthly, data_year_start_monthly):
        result = _compare(
            precips_mm_monthly.flatten(),
            3,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=1981,
            calibration_year_final=2010,
        )
        assert result.best.shape == (12,)
        assert np.all(result.best >= 0)


class TestWarnings:
    def test_short_calibration_period_warns(self):
        # the fits warn about a calibration period under 30 years, as the indices do
        values = _precip(17)
        with pytest.warns(ShortCalibrationWarning, match="30"):
            _compare(values, calibration_year_final=CAL_START + 19)

    def test_short_cell_sample_in_a_long_period_warns_once(self):
        values = _precip(17)
        # 12 of 40 calibration years missing in step 5: n = 28 although the period spans 40 years
        values[(CAL_START - START) * 12 + 5 : (CAL_START - START + 12) * 12 : 12] = np.nan
        with pytest.warns(ShortCalibrationWarning, match="shared") as caught:
            _compare(values)
        assert len([w for w in caught if "shared calibration values" in str(w.message)]) == 1

    def test_clamped_window_is_reported(self):
        values = _precip(22)
        with pytest.warns(CalibrationPeriodClampedWarning):
            _compare(values, calibration_year_initial=1900, calibration_year_final=2100)

    def test_warning_filters_are_untouched(self):
        # suppressing warnings by mutating the process-wide filters is unsafe across threads
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            before = list(warnings.filters)
            _compare(_precip(23))
            assert list(warnings.filters) == before

    def test_thirty_years_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ShortCalibrationWarning)
            _compare(_precip(18), calibration_year_final=CAL_START + 29)

    def test_long_timescale_warns(self):
        values = _precip(19)
        with pytest.warns(ClimateIndicesWarning, match="overlap"):
            _compare(values, 13)

    def test_scale_twelve_does_not_warn_about_dependence(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*overlap.*")
            _compare(_precip(19), 12)

    def test_daily_threshold_is_in_days(self):
        rng = np.random.default_rng(20)
        daily = rng.gamma(2.0, 2.0, size=40 * 366)
        kwargs = {
            "periodicity": compute.Periodicity.daily,
            "data_start_year": 1951,
            "calibration_year_initial": 1951,
            "calibration_year_final": 1990,
        }
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*overlap.*")
            _compare(daily, 90, **kwargs)  # 13..365 days is not flagged for daily data


class TestMetadata:
    def test_adr_section_6_attributes(self):
        result = _compare(_precip(21), 3, criterion="bic")
        meta = result.metadata
        assert json.loads(meta["climate_indices_candidates"]) == ["pearson", "gamma"]
        assert json.loads(meta["climate_indices_fitting_methods"]) == {
            "pearson": "l_moments",
            "gamma": "thom_mle_approximation",
        }
        assert meta["climate_indices_criterion"] == "bic"
        assert meta["climate_indices_log_likelihood_at"] == "fitted_estimates"
        assert meta["scale"] == "3"
        assert meta["calibration_year_initial"] == str(CAL_START)
        assert meta["calibration_year_final"] == str(CAL_END)
        assert "climate_indices_version" in meta
        assert "climate_indices_provenance" in meta
        # the GoF and RNG attributes belong to #1586
        assert "climate_indices_rng_seed" not in meta
        assert "climate_indices_gof_bootstrap_samples" not in meta

    def test_calibration_years_are_the_clamped_window(self):
        values = _precip(22)
        with pytest.warns(CalibrationPeriodClampedWarning):
            result = _compare(values, calibration_year_initial=1900, calibration_year_final=2100)
        assert result.metadata["calibration_year_initial"] == str(START)
        assert result.metadata["calibration_year_final"] == str(START + YEARS - 1)


class TestRecovery:
    """Synthetic characterization: how often the generating family wins. Reported, loosely asserted."""

    def test_gamma_data_prefers_gamma_more_often_than_not(self, record_property):
        wins = []
        for seed in range(6):
            result = _compare(
                _precip(100 + seed, years=80, shape=(4, 2)),
                1,
                spatial_time_major=True,
                data_start_year=1900,
                calibration_year_initial=1900,
                calibration_year_final=1979,
            )
            wins.append(np.mean(result.best == result.candidates.index(Distribution.gamma)))
        rate = float(np.mean(wins))
        record_property("gamma_recovery_rate", rate)
        assert rate > 0.85  # measured about 0.96 over 30 seeds; chance is 0.5


class TestParameterCounts:
    def test_counts_per_family_with_and_without_zero_mass(self):
        precip, pet = _spei_inputs(k=-0.3)
        spei = _compare(precip, index="spei", pet_mm=pet, calibration_year_final=START + 149)
        # pearson, gamma, loglogistic, gev: no zero mass is counted for this sample
        assert (spei.n_parameters[:, 0] == [3, 2, 3, 3]).all()
        zeroed = _precip(5)
        zeroed[(CAL_START - START) * 12 + 2 : (CAL_START - START + 6) * 12 : 12] = 0.0
        spi = _compare(zeroed)
        assert spi.n_parameters[0, 2] == 4  # Pearson with p0
        assert spi.n_parameters[0, 3] == 3  # Pearson without p0


class TestGloDensity:
    @pytest.mark.parametrize("k", [0.0, 5e-7, -5e-7])
    def test_logistic_limit(self, k):
        x, one = np.linspace(-3.0, 3.0, 7), np.ones(1)
        got = suitability._glo_logpdf(x, 0.5 * one, 2.0 * one, k * one)
        np.testing.assert_allclose(got, scipy.stats.logistic.logpdf(x, 0.5, 2.0), rtol=1e-5)

    def test_outside_support_is_minus_infinity(self):
        one = np.ones(1)
        # k > 0 bounds the support above at loc + scale / k = 4.5
        out = suitability._glo_logpdf(np.array([4.4, 4.5, 6.0]), 0.5 * one, 2.0 * one, 0.5 * one)
        assert np.isfinite(out[0])
        assert np.isneginf(out[1:]).all()

    def test_negative_shape_bounds_the_support_below(self):
        one = np.ones(1)
        # k < 0 bounds the support below at loc + scale / k = -3.5
        out = suitability._glo_logpdf(np.array([-4.0, -3.5, -3.0]), 0.5 * one, 2.0 * one, -0.5 * one)
        assert np.isneginf(out[:2]).all()
        assert np.isfinite(out[2])

    def test_shape_of_one_or_more_is_unusable(self):
        params = {"loc": np.zeros(3), "scale": np.ones(3), "shape": np.array([0.5, 1.0, -1.5])}
        np.testing.assert_array_equal(suitability._usable(Distribution.loglogistic, params), [True, False, False])


class TestInputs:
    def test_masked_values_and_pet_equal_nan_filled(self):
        precip, pet = _precip(30), _precip(31) * 0.8
        precip_masked = np.ma.masked_array(precip, mask=np.zeros(precip.shape, bool))
        precip_masked[100:103] = np.ma.masked
        pet_masked = np.ma.masked_array(pet, mask=np.zeros(pet.shape, bool))
        pet_masked[200] = np.ma.masked
        filled_p, filled_pet = precip.copy(), pet.copy()
        filled_p[100:103], filled_pet[200] = np.nan, np.nan
        masked = _compare(precip_masked, index="spei", pet_mm=pet_masked)
        filled = _compare(filled_p, index="spei", pet_mm=filled_pet)
        np.testing.assert_array_equal(masked.log_likelihood, filled.log_likelihood)

    def test_integer_input_matches_float(self):
        ints = np.round(_precip(32) * 10).astype(np.int64)
        np.testing.assert_array_equal(_compare(ints).log_likelihood, _compare(ints.astype(float)).log_likelihood)

    @pytest.mark.parametrize(
        "bad",
        [
            np.full(YEARS * 12, "1.0"),
            np.zeros(YEARS * 12, dtype="datetime64[D]"),
            np.ones(YEARS * 12, dtype=complex),
            np.ma.masked_array(np.full(YEARS * 12, "1.0")),
            np.ma.masked_array(np.ones(YEARS * 12, dtype=complex)),
        ],
    )
    def test_non_numeric_input_raises(self, bad):
        with pytest.raises(InvalidArgumentError, match="numeric"):
            _compare(bad)

    def test_non_numeric_pet_raises(self):
        values, pet = _precip(), np.full(YEARS * 12, "1.0")
        with pytest.raises(InvalidArgumentError, match="pet_mm"):
            _compare(values, index="spei", pet_mm=pet)

    def test_infinite_input_raises(self):
        values = _precip(37)
        values[0] = np.inf
        with pytest.raises(InvalidArgumentError, match="infinite"):
            _compare(values)

    def test_inputs_are_not_mutated(self):
        precip, pet = _precip(33) - 10.0, _precip(34)
        precip_before, pet_before = precip.copy(), pet.copy()
        _compare(precip, index="standardized_index")  # negatives are clipped on a copy
        _compare(precip, index="spei", pet_mm=pet)
        np.testing.assert_array_equal(precip, precip_before)
        np.testing.assert_array_equal(pet, pet_before)

    def test_standardized_index_unclipped_equals_clipped_on_non_negative_data(self):
        values = _precip(35)
        clipped = _compare(values, index="standardized_index", clip_negatives=True)
        unclipped = _compare(values, index="standardized_index", clip_negatives=False)
        np.testing.assert_array_equal(clipped.log_likelihood, unclipped.log_likelihood)

    def test_signed_series_is_clipped_when_asked(self):
        signed = _precip(36) - 40.0
        result = _compare(signed, index="standardized_index", clip_negatives=True)
        clipped = _compare(np.clip(signed, 0.0, None), index="standardized_index")
        np.testing.assert_array_equal(result.log_likelihood, clipped.log_likelihood)

    @pytest.mark.parametrize("bad", ["gamma", ["gamma"], [None]])
    def test_candidates_must_be_distribution_members(self, bad):
        values = _precip()
        with pytest.raises(InvalidArgumentError, match="Distribution"):
            _compare(values, candidates=bad)

    def test_unhashable_index_raises_invalid_argument(self):
        values = _precip()
        with pytest.raises(InvalidArgumentError):
            _compare(values, index=["spi"])  # type: ignore[arg-type]  # an unhashable index is the point


class TestRoutesAndShapes:
    def test_daily_result_shape_and_dependence_warning(self):
        rng = np.random.default_rng(40)
        daily = rng.gamma(2.0, 2.0, size=40 * 366)
        kwargs = {
            "periodicity": compute.Periodicity.daily,
            "data_start_year": 1951,
            "calibration_year_initial": 1951,
            "calibration_year_final": 1990,
        }
        assert _compare(daily, 30, **kwargs).best.shape == (366,)
        with pytest.warns(ClimateIndicesWarning, match="overlap"):
            _compare(daily, 366, **kwargs)

    def test_two_dimensional_input_matches_the_flat_series(self):
        values = _precip(45)
        flat, framed = _compare(values), _compare(values.reshape(YEARS, 12))
        np.testing.assert_array_equal(framed.log_likelihood, flat.log_likelihood)
        np.testing.assert_array_equal(framed.best, flat.best)

    @pytest.mark.parametrize("distribution", [Distribution.gamma, Distribution.pearson])
    def test_spatial_block_round_trip(self, distribution):
        block = _precip(41, shape=(2, 2))
        result = _compare(block, 3, spatial_time_major=True)
        direct = indices.spi(block, 3, distribution, START, CAL_START, CAL_END, MONTHLY, spatial_time_major=True)
        reused = indices.spi(
            block,
            3,
            distribution,
            START,
            CAL_START,
            CAL_END,
            MONTHLY,
            fitting_params=result.parameters[distribution],
            spatial_time_major=True,
        )
        np.testing.assert_array_equal(reused, direct)


class TestFitRaises:
    def test_fitting_error_is_recorded_not_substituted(self, monkeypatch):
        def refuse(cls, *args, **kwargs):
            raise DistributionFittingError("no fit")

        monkeypatch.setattr(compute.FittedDistribution, "resolve", classmethod(refuse))
        result = _compare(_precip(42))
        assert result.fit_failed.all()
        assert np.all(result.best == -1)
        assert all(np.isnan(array).all() for params in result.parameters.values() for array in params.values())


class TestProvenanceRecord:
    def test_spei_records_precipitation_shape_before_water_balance(self):
        precip = _precip().reshape(YEARS, 12)
        pet = np.zeros_like(precip)
        result = _compare(precip, index="spei", pet_mm=pet)
        record = json.loads(result.metadata["climate_indices_provenance"])
        inputs = {entry["role"]: entry for entry in record["inputs"]}
        assert inputs["values"]["shape"] == [YEARS, 12]
        assert inputs["pet_mm"]["shape"] == [YEARS, 12]

    @pytest.mark.parametrize("data_start_year", [1940, 1950])
    def test_record_holds_input_year_origin(self, data_start_year):
        result = _compare(_precip(years=80), data_start_year=data_start_year)
        record = json.loads(result.metadata["climate_indices_provenance"])["parameters"]
        assert record["data_start_year"] == data_start_year

    def test_record_holds_effective_options_and_clamped_window(self):
        precip, pet = _spei_inputs(k=-0.3)
        spei = _compare(precip, index="spei", pet_mm=pet)
        record = json.loads(spei.metadata["climate_indices_provenance"])["parameters"]
        assert record["clip_negatives"] is False  # SPEI clips precipitation itself and keeps the balance
        values = _precip(43)
        with pytest.warns(CalibrationPeriodClampedWarning):
            clamped = _compare(values, calibration_year_initial=1900, calibration_year_final=2100)
        record = json.loads(clamped.metadata["climate_indices_provenance"])["parameters"]
        assert record["calibration_year_initial"] == START
        assert record["calibration_year_final"] == START + YEARS - 1


def _gev_log_density(x: np.ndarray, loc: float, scale: float, shape: float) -> np.ndarray:
    """Hosking's GEV log-density from the closed form, ``-inf`` outside the support.

    With ``z = (x - loc) / scale`` and ``y = -log(1 - k z) / k`` (``y = z`` at ``k = 0``),
    ``log f = -log(scale) - (1 - k) y - exp(-y)``; the support is ``k z < 1``. Written here
    from the distribution function, not from SciPy, so it checks ``genextreme.logpdf``. ``log1p``
    matters: ``log(1 - k z)`` loses its digits to the rounding of ``1 - k z`` and, divided by a
    tiny ``k``, is wrong in the seventh digit at ``k = 1e-9``.
    """
    z = (x - loc) / scale
    argument = 1.0 - shape * z
    with np.errstate(all="ignore"):
        y = z if shape == 0.0 else -np.log1p(-shape * np.where(argument > 0.0, z, 0.0)) / shape
        density = -np.log(scale) - (1.0 - shape) * y - np.exp(-y)
    return np.where((shape == 0.0) | (argument > 0.0), density, -np.inf)


class TestGevCandidate:
    """GEV joins the SPEI candidates (#1583), scored at its L-moment estimates like the log-logistic."""

    @pytest.mark.parametrize("shape", [-0.4, -1e-9, 0.0, 1e-9, 0.4])
    def test_log_density_matches_the_closed_form(self, shape):
        x = np.linspace(-1.0, 8.0, 91)
        params = {"loc": np.full(91, 2.0), "scale": np.full(91, 1.5), "shape": np.full(91, shape)}
        ours = suitability._log_density(Distribution.gev, x, params)
        np.testing.assert_allclose(ours, _gev_log_density(x, 2.0, 1.5, shape), rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("shape", [-0.4, 0.0, 0.4])
    def test_density_is_the_derivative_of_the_distribution_function(self, shape):
        """exp(log f) equals dF/dx, with F from the transform (a different code path from the density)."""
        x = np.array([0.5, 1.0, 2.0, 3.0])
        step = 1e-6
        upper = compute._gev_fit(x + step, 2.0, 1.5, shape, output_scale="probability")
        lower = compute._gev_fit(x - step, 2.0, 1.5, shape, output_scale="probability")
        params = {"loc": np.full(4, 2.0), "scale": np.full(4, 1.5), "shape": np.full(4, shape)}
        density = np.exp(suitability._log_density(Distribution.gev, x, params))
        # a central difference of a smooth CDF is accurate to about step**2 plus rounding / step
        np.testing.assert_allclose(density, (upper - lower) / (2 * step), rtol=1e-6)

    def test_outside_the_support_is_minus_infinity_on_both_branches(self):
        two = np.ones(2)
        # k > 0 bounds the support above at loc + scale / k = 4.5; k < 0 bounds it below at -3.5
        above = suitability._log_density(
            Distribution.gev, np.array([4.4, 6.0]), {"loc": 0.5 * two, "scale": 2.0 * two, "shape": 0.5 * two}
        )
        below = suitability._log_density(
            Distribution.gev, np.array([-4.0, -3.0]), {"loc": 0.5 * two, "scale": 2.0 * two, "shape": -0.5 * two}
        )
        assert np.isfinite(above[0]) and np.isneginf(above[1])
        assert np.isneginf(below[0]) and np.isfinite(below[1])

    def test_usable_follows_the_transform_predicate(self):
        params = {
            "loc": np.array([0.0, 0.0, 0.0, np.nan]),
            "scale": np.array([1.0, 0.0, 1.0, 1.0]),
            "shape": np.array([1.5, 0.1, np.nan, 0.1]),
        }
        # a shape of 1.5 is usable (unlike the log-logistic); a zero scale marks an unfitted step
        usable = suitability._usable(Distribution.gev, params)
        np.testing.assert_array_equal(usable, [True, False, False, False])
        # and it is the predicate the transform applies: a position it cannot transform is NaN
        transformed = compute._gev_fit(
            np.ones(4), params["loc"], params["scale"], params["shape"], output_scale="probability"
        )
        np.testing.assert_array_equal(np.isfinite(transformed), usable)

    def test_a_family_without_a_likelihood_raises_rather_than_scoring_as_log_logistic(self):
        params = {"loc": np.zeros(1), "scale": np.ones(1), "shape": np.zeros(1)}
        with pytest.raises(ValueError, match="no likelihood"):
            suitability._log_density(Distribution.empirical, np.zeros(1), params)
        with pytest.raises(ValueError, match="no likelihood"):
            suitability._usable(Distribution.empirical, params)

    def test_log_likelihood_is_the_summed_closed_form_at_the_returned_parameters(self):
        precip, pet = _spei_inputs(k=-0.3)
        result = _compare(precip, index="spei", pet_mm=pet, calibration_year_final=START + 149)
        position = result.candidates.index(Distribution.gev)
        params = result.parameters[Distribution.gev]
        scaled = compute.prepare_scaled(indices._spei_water_balance(precip, pet), 1, MONTHLY, clip_negatives=False)
        sample = scaled[CAL_START - START : START + 149 - START + 1]
        assert not result.fit_failed[position].any()
        for step in range(12):
            expected = _gev_log_density(
                sample[:, step], params["loc"][step], params["scale"][step], params["shape"][step]
            )
            assert result.log_likelihood[position, step] == pytest.approx(expected.sum(), rel=1e-10)

    def test_records_its_method_and_is_ranked_with_the_others(self):
        precip, pet = _spei_inputs(k=-0.3)
        result = _compare(precip, index="spei", pet_mm=pet, calibration_year_final=START + 149)
        assert json.loads(result.metadata["climate_indices_fitting_methods"])["gev"] == "l_moments"
        assert json.loads(result.metadata["climate_indices_candidates"]) == ["pearson", "gamma", "loglogistic", "gev"]
        assert result.log_likelihood.shape[0] == 4
        assert set(np.unique(result.best)) <= set(range(4))

    def test_exact_zero_balance_is_scored_as_a_continuous_value(self):
        """GEV has no zero mass: a balance of exactly 0 is an ordinary observation inside its support."""
        pet = np.repeat(np.linspace(980.0, 1020.0, 41), 12)
        result = _compare(
            np.zeros_like(pet),
            index="spei",
            pet_mm=pet,
            candidates=(Distribution.gev,),
            calibration_year_initial=START,
            calibration_year_final=START + 40,
        )
        params = result.parameters[Distribution.gev]
        sample = (1000.0 - pet).reshape(41, 12)
        for step in range(12):
            expected = _gev_log_density(
                sample[:, step], params["loc"][step], params["scale"][step], params["shape"][step]
            )
            assert result.log_likelihood[0, step] == pytest.approx(expected.sum(), rel=1e-10)
        assert not result.fit_failed.any()
        np.testing.assert_array_equal(result.n_parameters[0], np.full(12, 3))
        np.testing.assert_array_equal(result.best, np.zeros(12, dtype=int))

    def test_a_constant_calendar_step_has_no_gev_fit(self):
        """A zero L-scale cannot be fitted: that step is failed and NaN, with no replacement, and others score."""
        precip = _precip(3).copy()
        precip[2::12] = 50.0  # the water balance is constant at calendar step 2
        result = _compare(precip, index="spei", pet_mm=np.full_like(precip, 30.0))
        position = result.candidates.index(Distribution.gev)
        params = result.parameters[Distribution.gev]
        assert result.fit_failed[position, 2]
        assert np.isnan(result.log_likelihood[position, 2])
        # the gamma estimator still returns a (degenerate) fit for a constant sample, so it may take `best`
        assert result.best[2] != position
        assert all(params[key][2] == 0.0 for key in ("loc", "scale", "shape"))
        assert (~result.fit_failed[position]).any()

    def test_a_sample_outside_the_fitted_support_does_not_rank(self):
        """Zero likelihood is a failed candidate: -inf log-likelihood, NaN criteria, and never `best`."""
        pet = np.random.default_rng(0).gamma(0.5, 30.0, 12 * 12)  # a short skewed record: GEV shape above 1
        result = _compare(
            np.zeros_like(pet),
            index="spei",
            pet_mm=pet,
            calibration_year_initial=START,
            calibration_year_final=START + 11,
        )
        position = result.candidates.index(Distribution.gev)
        outside = np.where(np.isneginf(result.log_likelihood[position]))[0]
        assert outside.size, "the seeded record should leave a value outside the fitted GEV support"
        assert result.fit_failed[position, outside].all()
        assert np.isnan(result.aicc[position, outside]).all()
        assert (result.best[outside] != position).all()
        assert not result.fit_failed[result.best[outside], outside].any()

    def test_a_spatial_block_scores_each_cell_and_fails_only_the_constant_one(self):
        precip = _precip(7, shape=(2, 2)).copy()  # (time, 2, 2): three dimensions make it a spatial block
        precip[2::12, 0, 0] = 50.0  # cell (0, 0) has a constant balance at calendar step 2
        result = _compare(precip, index="spei", pet_mm=np.full_like(precip, 30.0), spatial_time_major=True)
        position = result.candidates.index(Distribution.gev)
        assert result.log_likelihood.shape == (4, 12, 2, 2)
        assert result.fit_failed[position, 2, 0, 0]
        assert result.best[2, 0, 0] != position
        # a neighbouring cell is still scored
        assert (~result.fit_failed[position, :, 1, 1]).any()
