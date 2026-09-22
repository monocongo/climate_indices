"""Recipe and parity checks for the documented SRI and SSI paths (issue #1114).

`docs/standardized-hydrologic-indices.md` tells users to compute the
Standardized Runoff Index and Standardized Streamflow Index by calling
`indices.standardized_index()` on a runoff or streamflow series. These tests
run the documented calls, so a change to that signature or to the shared
pipeline fails here alongside the example.
"""

from __future__ import annotations

import numpy as np

from climate_indices import compute, indices

_START_YEAR = 1981
_END_YEAR = 2010
_MONTHS = (_END_YEAR - _START_YEAR + 1) * 12


def _monthly_series(scale: float, exponent: float) -> np.ndarray:
    """Return a deterministic, non-negative monthly series with a seasonal cycle."""
    months = np.arange(_MONTHS, dtype=float)
    seasonal = 1.0 + 0.6 * np.sin(2.0 * np.pi * (months % 12.0) / 12.0)
    return scale * seasonal * np.exp(exponent * np.sin(months / 7.0))


def test_sri_recipe_standardizes_runoff_with_the_spi_pipeline() -> None:
    """The SRI recipe is the SPI procedure applied to runoff, not a parallel fit."""
    runoff = _monthly_series(scale=50.0, exponent=0.4)

    sri = indices.standardized_index(
        runoff,
        scale=3,
        distribution=indices.Distribution.gamma,
        data_start_year=_START_YEAR,
        calibration_year_initial=_START_YEAR,
        calibration_year_final=_END_YEAR,
        periodicity=compute.Periodicity.monthly,
    )
    spi = indices.spi(
        runoff,
        3,
        indices.Distribution.gamma,
        _START_YEAR,
        _START_YEAR,
        _END_YEAR,
        compute.Periodicity.monthly,
    )

    np.testing.assert_array_equal(sri, spi)
    assert sri.shape == runoff.shape
    # the first scale-1 steps have no complete accumulation window, as in SPI
    assert np.isfinite(sri[2:]).all()
    assert abs(float(np.nanmean(sri))) < 0.1


def test_ssi_recipe_uses_the_generic_standardization_api_unchanged() -> None:
    """The SSI recipe calls standardized_index() with the documented arguments."""
    streamflow = _monthly_series(scale=20.0, exponent=0.8)

    ssi = indices.standardized_index(
        streamflow,
        scale=12,
        distribution=indices.Distribution.pearson,
        data_start_year=_START_YEAR,
        calibration_year_initial=_START_YEAR,
        calibration_year_final=_END_YEAR,
        periodicity=compute.Periodicity.monthly,
    )
    standardized = indices.standardized_index(
        streamflow,
        12,
        indices.Distribution.pearson,
        _START_YEAR,
        _START_YEAR,
        _END_YEAR,
        compute.Periodicity.monthly,
    )

    np.testing.assert_array_equal(ssi, standardized)
    assert ssi.shape == streamflow.shape
    assert np.isfinite(ssi[11:]).all()
    assert abs(float(np.nanmean(ssi))) < 0.1
