"""Gridded regression for consecutive zero precipitation (issue #582)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import norm

from climate_indices import compute, spi
from climate_indices.drought import indices


@pytest.mark.parametrize("zero_handling", ["classic", "center_of_mass", "mean_zero"])
def test_gridded_spi_three_month_zero_run(
    zero_handling: compute.ZeroHandling, caplog: pytest.LogCaptureFixture, python_backend: None
) -> None:
    """March has two identical positive totals, 27 zeros, and one missing total."""
    wet = np.random.default_rng(582).gamma(3, 20, size=(31, 12))
    dry = wet.copy()
    dry[:30, :3] = 0
    dry[:2, :3] = [0.25, 0.25, 0.5]  # no spread for gamma, fewer than four positives for Pearson
    dry[5, 1] = np.nan
    rain = xr.DataArray(
        np.column_stack((dry.ravel(), wet.ravel())),
        dims=("time", "cell"),
        coords={"time": pd.date_range("1990-01-01", periods=31 * 12, freq="MS")},
        attrs={"units": "mm"},
    )
    args = (1990, 1990, 2019, compute.Periodicity.monthly)
    with np.errstate(invalid="ignore", divide="ignore"):
        pearson = spi(rain, 3, indices.Distribution.pearson, *args, zero_handling=zero_handling)
        gamma = spi(rain, 3, indices.Distribution.gamma, *args, zero_handling=zero_handling)
    events = [record.msg for record in caplog.records if isinstance(record.msg, dict)]
    failures = [event for event in events if event.get("event") == "distribution_fitting_failures"]
    assert [(event["distribution"], event["failure_count"]) for event in failures] == [("pearson3", 1)]
    assert not any(event.get("event") == "distribution_fallback" for event in events)

    p0 = 27 / 29
    gamma_zero = {
        "classic": norm.ppf(p0),
        "center_of_mass": norm.ppf(p0 / 2),
        "mean_zero": -norm.pdf(norm.ppf(p0)) / p0,
    }
    zero_marches = np.delete(np.arange(2, 30), 3) * 12 + 2
    unfit_or_missing = np.array([0, 1, 5, 30]) * 12 + 2
    for result, distribution, zero_score in (
        (pearson, indices.Distribution.pearson, -3.09),
        (gamma, indices.Distribution.gamma, gamma_zero[zero_handling]),
    ):
        np.testing.assert_allclose(result.values[zero_marches, 0], zero_score, rtol=0, atol=1e-12)
        assert np.isnan(result.values[unfit_or_missing, 0]).all()
        assert np.isfinite(result.values[2:, 1]).all()
        neighbor = indices.spi(wet.ravel(), 3, distribution, *args, zero_handling=zero_handling)
        np.testing.assert_allclose(result.values[:, 1], neighbor, rtol=0, atol=1e-10)
