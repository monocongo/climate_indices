"""Offline checks for the pre-registered plotting-position experiment (#1571)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import norm

from climate_indices import compute
from climate_indices.drought import indices
from scripts import earthengine_climate_engine_pilot as pilot


def test_tukey_known_ranks_ties_edges_and_missing() -> None:
    calibration = np.arange(1.0, 37.0)[:, None]
    values = np.concatenate(([-1.0], np.arange(1.0, 37.0), [37.0, np.nan]))[:, None]
    probability = pilot.tukey_probabilities(calibration, values)
    ranks = np.concatenate(([1], np.arange(1, 37), [36]))
    expected = (ranks - 1 / 3) / (36 + 1 / 3)
    np.testing.assert_allclose(norm.ppf(probability[:-1, 0]), norm.ppf(expected), atol=1e-14, rtol=0)
    assert np.isnan(probability[-1, 0])
    tied = np.array([0.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, np.nan])[:, None]
    assert np.isnan(pilot.tukey_probabilities(tied, np.array([[0.0]]))).all()
    tied[-1] = 9.0
    assert pilot.tukey_probabilities(tied, np.array([[0.0]]))[0, 0] == pytest.approx((2 / 3) / (10 + 1 / 3))


def test_tukey_matches_eddi_rank_semantics_with_documented_numerical_differences() -> None:
    calibration = np.arange(36.0)[:, None]
    values = np.array([[0.0], [12.0], [35.0], [np.nan]])
    ours = pilot.tukey_probabilities(calibration, values)
    below = np.count_nonzero(calibration[:, None, :] < values[None, :, :], axis=0)
    eddi_probability = (below + 0.66) / (36 + 0.33)
    eddi_probability[np.isnan(values)] = np.nan
    # Exact thirds versus NOAA's rounded constants differ by at most 0.0002 here.
    np.testing.assert_allclose(ours, eddi_probability, atol=0.0002, rtol=0, equal_nan=True)
    native = compute._native_tukey_probabilities(calibration, values, np.zeros(1))
    if native is not None:
        np.testing.assert_allclose(native, eddi_probability, atol=1e-14, rtol=0, equal_nan=True)
    data = xr.DataArray(
        np.repeat(np.arange(36.0), 12),
        dims="time",
        coords={"time": pd.date_range("1981-01-31", periods=432, freq="ME")},
    )
    result = pilot.tukey_index(data, 1, clip_negatives=False)
    expected = indices.eddi(data.values, 1, 1981, 1981, 2016, compute.Periodicity.monthly)
    # NOAA's rounded constants and Hastings approximation change z by under 0.005.
    np.testing.assert_allclose(result.values, expected, atol=0.005, rtol=0)


def test_compute_indices_sei_is_unchanged_and_eddi_is_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pilot, "SCALES", {1: "30d", 3: "90d"})
    random = np.random.default_rng(1571)
    dates = pd.date_range("1980-01-31", periods=540, freq="ME")
    dataset = xr.Dataset(
        {
            name: ("time", random.uniform(0, 100, 540), {"units": "mm"})
            for name in ("precipitation", "reference_et", "alfalfa_et")
        },
        coords={"time": dates},
    )
    original = pilot.compute_indices(dataset)
    explicit = pilot.compute_indices(dataset, "sei")
    tukey = pilot.compute_indices(dataset, "tukey")
    for key in original:
        np.testing.assert_array_equal(original[key].values, explicit[key].values)
        if key.startswith("eddi"):
            np.testing.assert_array_equal(original[key].values, tukey[key].values)
    for scale in pilot.SCALES:
        for index, data in (("spi", dataset.precipitation), ("spei", dataset.precipitation - dataset.reference_et)):
            expected = indices.standardized_index(
                data.values,
                scale,
                indices.Distribution.empirical,
                1980,
                1981,
                2016,
                compute.Periodicity.monthly,
                clip_negatives=index == "spi",
            )
            np.testing.assert_array_equal(original[f"{index}_{scale}"].values, expected)


def test_paired_metrics_use_common_pairs_and_split_dates() -> None:
    dates = pd.date_range("1981-01-31", periods=528, freq="ME")
    reference = xr.DataArray(np.arange(528.0), dims="time", coords={"time": dates})
    sei = reference + 0.05
    tukey = reference.copy()
    tukey[0] = np.nan
    result = pilot.paired_metrics(sei, tukey, reference)
    assert result["1981-2024"]["sei"]["n"] == 527
    assert result["1981-2016"]["sei"]["n"] == 431
    assert result["2017-2024"]["sei"]["n"] == 96
    assert result["1981-2024"]["change"]["bias"] == pytest.approx(-0.05)
    assert len(result["calendar_month_bias"]["tukey"]) == 12
    assert result["calendar_month_bias"]["tukey"] == [0.0] * 12


def test_baseline_control_stops_on_drift() -> None:
    baseline = {"eddi_1": {"n": 12.0, "bias": 0.01, "correlation": 0.99}}
    pilot.check_baseline(baseline, baseline)
    with pytest.raises(RuntimeError, match="baseline drift.*eddi_1.*bias"):
        pilot.check_baseline({"eddi_1": {"n": 12.0, "bias": 0.02, "correlation": 0.99}}, baseline)


def test_expected_reduction_comes_from_formulas() -> None:
    expected = pilot.expected_rank_reduction(36)
    assert expected == pytest.approx(0.053831430857023244, abs=1e-14)
