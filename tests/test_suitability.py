"""Tests for ``climate_indices.suitability.compare_distributions`` (NumPy core, #1584, ADR-0028)."""

from __future__ import annotations

import dataclasses
import json
import warnings

import numpy as np
import pytest
import scipy.stats

import climate_indices
from climate_indices import compute, suitability
from climate_indices.drought import indices
from climate_indices.exceptions import (
    BetaFeatureWarning,
    ClimateIndicesWarning,
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


def _compare(values, scale=1, *, index="spi", **kwargs):
    kwargs.setdefault("data_start_year", START)
    kwargs.setdefault("calibration_year_initial", CAL_START)
    kwargs.setdefault("calibration_year_final", CAL_END)
    kwargs.setdefault("periodicity", MONTHLY)
    return suitability.compare_distributions(values, scale, index=index, **kwargs)


def _calibration_sample(values, scale=1):
    """The scaled calibration sample, (years, 12), recomputed independently of the module."""
    scaled = compute.prepare_scaled(values, scale, MONTHLY)
    return scaled[CAL_START - START : CAL_END - START + 1]


def _spei_inputs(*, k: float, seed: int = 1, years: int = 150):
    """Precipitation and PET whose SPEI water balance is generalized-logistic with Hosking shape ``k``.

    k < 0: precipitation is a ``fisk`` draw and PET is zero. k > 0: precipitation is zero and
    PET is a ``fisk`` draw, so the balance ``1000 - PET`` is the reflected distribution.
    """
    draw = scipy.stats.fisk.rvs(c=1 / 0.3, scale=60.0, size=years * 12, random_state=seed)
    zeros = np.zeros(years * 12)
    return (draw, zeros) if k < 0 else (zeros, draw)


class TestSurface:
    def test_beta_warning(self):
        with pytest.warns(BetaFeatureWarning):
            suitability.compare_distributions(
                _precip(),
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


class TestCandidates:
    def test_default_sets_in_enum_order(self):
        spi = _compare(_precip())
        assert spi.candidates == (Distribution.pearson, Distribution.gamma)
        precip, pet = _spei_inputs(k=-0.3)
        spei = _compare(precip, index="spei", pet_mm=pet)
        assert spei.candidates == (Distribution.pearson, Distribution.gamma, Distribution.loglogistic)

    def test_user_candidates_are_sorted_to_enum_order(self):
        result = _compare(_precip(), candidates=(Distribution.gamma, Distribution.pearson))
        assert result.candidates == (Distribution.pearson, Distribution.gamma)

    @pytest.mark.parametrize("name", ["empirical", "beta", "t_location_scale", "normal"])
    def test_non_likelihood_candidate_raises(self, name):
        with pytest.raises(InvalidArgumentError):
            _compare(_precip(), candidates=(Distribution(name),))

    def test_loglogistic_rejected_outside_spei(self):
        with pytest.raises(InvalidArgumentError):
            _compare(_precip(), candidates=(Distribution.loglogistic,))

    def test_duplicate_or_empty_candidates_raise(self):
        with pytest.raises(InvalidArgumentError):
            _compare(_precip(), candidates=(Distribution.gamma, Distribution.gamma))
        with pytest.raises(InvalidArgumentError):
            _compare(_precip(), candidates=())


class TestArguments:
    def test_spei_requires_pet(self):
        with pytest.raises(InvalidArgumentError):
            _compare(_precip(), index="spei")

    def test_pet_only_for_spei(self):
        with pytest.raises(InvalidArgumentError):
            _compare(_precip(), pet_mm=_precip(1))

    @pytest.mark.parametrize("option", [{"aggregation": "mean"}, {"clip_negatives": False}])
    def test_standardized_index_options_only_on_that_route(self, option):
        with pytest.raises(InvalidArgumentError):
            _compare(_precip(), index="spi", **option)

    def test_bad_route_and_criterion(self):
        with pytest.raises(InvalidArgumentError):
            _compare(_precip(), index="pdsi")
        with pytest.raises(InvalidArgumentError):
            _compare(_precip(), criterion="hqic")

    def test_signed_series_needs_clipping(self):
        signed = _precip() - 40.0
        with pytest.raises(InvalidArgumentError):
            _compare(signed, index="standardized_index", clip_negatives=False)

    def test_hourly_rejected(self):
        with pytest.raises(InvalidArgumentError):
            _compare(_precip(), periodicity=compute.Periodicity.hourly)

    def test_all_missing_input_raises(self):
        with pytest.raises(ValueError, match="no valid"):
            _compare(np.full(YEARS * 12, np.nan))


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
        np.testing.assert_allclose(result.aic, 2 * k - 2 * ll, rtol=1e-12)
        np.testing.assert_allclose(result.bic, k * np.log(n) - 2 * ll, rtol=1e-12)
        np.testing.assert_allclose(result.aicc, result.aic + 2 * k * (k + 1) / (n - k - 1), rtol=1e-12)

    def test_aicc_nan_when_n_not_above_k_plus_one(self):
        # 3 calibration years: n = 3 <= k + 1 = 3 for gamma without zeros
        with pytest.warns(ShortCalibrationWarning):
            result = _compare(_precip(7), candidates=(Distribution.gamma,), calibration_year_final=CAL_START + 2)
        assert np.all(np.isnan(result.aicc))
        assert np.all(np.isfinite(result.aic))
        assert np.all(result.best == -1)
        assert np.all(~result.fit_failed)
        with pytest.warns(ShortCalibrationWarning):
            by_aic = _compare(
                _precip(7), candidates=(Distribution.gamma,), calibration_year_final=CAL_START + 2, criterion="aic"
            )
        assert np.all(by_aic.best == 0)

    @pytest.mark.parametrize("criterion", ["aicc", "aic", "bic"])
    def test_best_is_argmin_of_requested_criterion(self, criterion):
        result = _compare(_precip(8), criterion=criterion)
        crit = getattr(result, criterion)
        np.testing.assert_array_equal(result.best, np.argmin(crit, axis=0))
        assert result.criterion == criterion
        ordered = np.sort(crit, axis=0)
        np.testing.assert_allclose(result.delta_to_runner_up, ordered[1] - ordered[0], rtol=1e-12)


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
        kwargs = {"aggregation": "mean"}
        direct = indices.standardized_index(values, 2, distribution, START, CAL_START, CAL_END, MONTHLY, **kwargs)
        reused = indices.standardized_index(
            values,
            2,
            distribution,
            START,
            CAL_START,
            CAL_END,
            MONTHLY,
            fitting_params=result.parameters[distribution],
            **kwargs,
        )
        np.testing.assert_array_equal(reused, direct)

    @pytest.mark.parametrize("distribution", [Distribution.gamma, Distribution.pearson, Distribution.loglogistic])
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
    def test_spei_exact_zero_balance_disqualifies_glo_in_that_cell_only(self):
        precip, pet = _precip(13), _precip(14)
        row = (CAL_START - START + 5) * 12 + 4
        precip[row], pet[row] = 0.0, 1000.0  # balance + 1000 offset is exactly 0
        result = _compare(precip, index="spei", pet_mm=pet)
        glo = result.candidates.index(Distribution.loglogistic)
        gamma = result.candidates.index(Distribution.gamma)
        assert result.fit_failed[glo, 4]
        assert not np.delete(result.fit_failed[glo], 4).any()
        assert result.best[4] != glo
        assert result.n_parameters[gamma, 4] == 3  # zero mass counted only where zeros exist
        assert result.n_parameters[gamma, 5] == 2

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
        assert np.isnan(result.log_likelihood[p, 7]) and np.isnan(result.aicc[p, 7])

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
    def test_short_calibration_warns_once(self):
        with pytest.warns(ShortCalibrationWarning) as caught:
            _compare(_precip(17), calibration_year_final=CAL_START + 19)
        assert len([w for w in caught if issubclass(w.category, ShortCalibrationWarning)]) == 1

    def test_thirty_years_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ShortCalibrationWarning)
            _compare(_precip(18), calibration_year_final=CAL_START + 29)

    def test_long_timescale_warns(self):
        with pytest.warns(ClimateIndicesWarning, match="overlap"):
            _compare(_precip(19), 13)

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
        with pytest.warns(ClimateIndicesWarning):
            result = _compare(_precip(22), calibration_year_initial=1900, calibration_year_final=2100)
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
        assert rate > 0.4  # loose floor: about 80 values per step cannot discriminate sharply
