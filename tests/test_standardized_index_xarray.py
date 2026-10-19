"""The generic standardization seam's beta xarray path (ADR-0020, #1174).

Parity with the NumPy core, the aggregation seam, inference, metadata, and the
package-root re-export. Distribution behaviour is covered by the existing seam tests;
these pin what the adapter layer adds.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import climate_indices
from climate_indices import indices
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.compute import Periodicity
from climate_indices.exceptions import CoordinateValidationError, InvalidArgumentError

SCALE = 3
DATA_START_YEAR = 1985
CALIBRATION_START = 1990
CALIBRATION_END = 2019


def _series(seed: int = 0, years: int = 35) -> np.ndarray:
    """Return a bounded monthly capacity-factor-like series."""
    return np.random.default_rng(seed).uniform(0.05, 0.95, years * 12)


def _monthly(seed: int = 0, years: int = 35) -> xr.DataArray:
    values = _series(seed, years)
    time = xr.date_range(f"{DATA_START_YEAR}-01-01", periods=values.size, freq="MS")
    return xr.DataArray(values, dims="time", coords={"time": time}, attrs={"units": "1"})


def _gridded(seed: int = 0, years: int = 35) -> xr.DataArray:
    """Return a (time, lat, lon) cube with a spatial cell axis that is not a calendar length."""
    values = np.random.default_rng(seed).uniform(0.05, 0.95, (years * 12, 2, 3))
    time = xr.date_range(f"{DATA_START_YEAR}-01-01", periods=values.shape[0], freq="MS")
    return xr.DataArray(
        values, dims=("time", "lat", "lon"), coords={"time": time, "lat": [10.0, 20.0], "lon": [1.0, 2.0, 3.0]}
    )


def _numpy(
    values: np.ndarray,
    *,
    calibration_start: int = DATA_START_YEAR,
    calibration_end: int = DATA_START_YEAR + 34,
    distribution: indices.Distribution = indices.Distribution.empirical,
    **kwargs: object,
) -> np.ndarray:
    return indices.standardized_index(
        values,
        SCALE,
        distribution,
        DATA_START_YEAR,
        calibration_start,
        calibration_end,
        Periodicity.monthly,
        **kwargs,  # type: ignore[arg-type]
    )


def test_one_dimensional_series_matches_numpy() -> None:
    """A 1-D DataArray standardizes exactly as the NumPy core does."""
    series = _series()
    result = climate_indices.standardized_index(_monthly(), SCALE, indices.Distribution.empirical)

    np.testing.assert_array_equal(result.values, _numpy(series))
    assert result.dims == ("time",)
    np.testing.assert_array_equal(result.coords["time"].values, _monthly().coords["time"].values)


def test_inferred_calibration_period_matches_numpy() -> None:
    """Omitting the temporal parameters infers the full time range, as the other adapters do."""
    series = _series()
    result = climate_indices.standardized_index(_monthly(), SCALE, indices.Distribution.empirical)

    assert result.attrs["calibration_year_initial"] == DATA_START_YEAR
    assert result.attrs["calibration_year_final"] == DATA_START_YEAR + 34
    assert result.attrs["scale"] == SCALE
    np.testing.assert_array_equal(result.values, _numpy(series))


def test_explicit_temporal_parameters_win_over_inference() -> None:
    """An explicitly supplied Calibration Period is used as given."""
    result = climate_indices.standardized_index(
        _monthly(),
        SCALE,
        indices.Distribution.empirical,
        calibration_year_initial=CALIBRATION_START,
        calibration_year_final=CALIBRATION_END,
    )

    assert result.attrs["calibration_year_initial"] == CALIBRATION_START
    assert result.attrs["calibration_year_final"] == CALIBRATION_END
    np.testing.assert_array_equal(
        result.values, _numpy(_series(), calibration_start=CALIBRATION_START, calibration_end=CALIBRATION_END)
    )


def test_metadata_is_the_neutral_registry_entry() -> None:
    """The output carries the neutral entry: no discipline-specific name, no standard_name."""
    result = climate_indices.standardized_index(_monthly(), SCALE, indices.Distribution.empirical)
    entry = CF_METADATA["standardized_index"]

    assert result.attrs["long_name"] == entry["long_name"] == "Standardized index"
    assert result.attrs["units"] == entry["units"] == "1"
    assert "standard_name" not in result.attrs
    assert result.attrs["references"] == entry["references"]
    assert "climate_indices" in result.attrs["history"]


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_aggregation_is_honoured_and_recorded(aggregation: str) -> None:
    """Both operations reach the core, and the Dataset-equivalent provenance names the one used."""
    result = climate_indices.standardized_index(
        _monthly(),
        SCALE,
        indices.Distribution.empirical,
        aggregation=aggregation,  # type: ignore[arg-type]
    )
    expected = _numpy(_series(), aggregation=aggregation)

    np.testing.assert_array_equal(result.values, expected)
    assert result.attrs["aggregation"] == aggregation


def test_default_aggregation_is_recorded() -> None:
    """An omitted aggregation still names the operation behind the fit."""
    result = climate_indices.standardized_index(_monthly(), SCALE, indices.Distribution.empirical)

    assert result.attrs["aggregation"] == "sum"


def test_gridded_input_matches_a_per_cell_numpy_run() -> None:
    """Each grid cell is standardized independently with the same arguments."""
    cube = _gridded()
    result = climate_indices.standardized_index(cube, SCALE, indices.Distribution.empirical)

    assert result.dims == cube.dims
    for lat_index in range(cube.sizes["lat"]):
        for lon_index in range(cube.sizes["lon"]):
            expected = _numpy(cube.isel(lat=lat_index, lon=lon_index).values)
            np.testing.assert_array_equal(result.isel(lat=lat_index, lon=lon_index).values, expected)


def test_gridded_input_matches_numpy_with_tied_values() -> None:
    """Tied values keep the NumPy ranks, so a grid cell cannot drift from its own series."""
    tied = np.concatenate([np.random.default_rng(7).permutation(np.arange(1, 13) / 13) for _ in range(40)])
    time = xr.date_range(f"{DATA_START_YEAR}-01-01", periods=tied.size, freq="MS")
    cube = xr.DataArray(
        tied[:, None, None], dims=("time", "lat", "lon"), coords={"time": time, "lat": [1.0], "lon": [1.0]}
    )

    result = climate_indices.standardized_index(
        cube,
        12,
        indices.Distribution.empirical,
        calibration_year_initial=DATA_START_YEAR,
        calibration_year_final=DATA_START_YEAR + 39,
    )
    expected = indices.standardized_index(
        tied,
        12,
        indices.Distribution.empirical,
        DATA_START_YEAR,
        DATA_START_YEAR,
        DATA_START_YEAR + 39,
        Periodicity.monthly,
    )

    np.testing.assert_array_equal(result.values[:, 0, 0], expected)


def test_split_time_chunk_is_rejected() -> None:
    """A fitting index reads the whole time series at once, so a split chunk cannot be honoured."""
    with pytest.raises(CoordinateValidationError, match="chunk"):
        climate_indices.standardized_index(_monthly().chunk({"time": 120}), SCALE, indices.Distribution.empirical)


def test_source_nans_propagate_as_missing() -> None:
    """A missing source step stays missing in the output rather than shifting the series."""
    cube = _gridded()
    cube[0, 0, 0] = np.nan

    result = climate_indices.standardized_index(cube, SCALE, indices.Distribution.empirical)

    assert np.isnan(result.values[0, 0, 0])
    np.testing.assert_array_equal(result.values[:, 0, 1], _numpy(cube.values[:, 0, 1]))


def test_dask_backed_input_matches_the_eager_result() -> None:
    """A time-chunked Dask cube stays lazy and produces the same values."""
    cube = _gridded().chunk({"time": -1, "lat": 1})
    result = climate_indices.standardized_index(cube, SCALE, indices.Distribution.empirical)

    assert result.chunks is not None
    np.testing.assert_array_equal(
        result.compute().values,
        climate_indices.standardized_index(_gridded(), SCALE, indices.Distribution.empirical).values,
    )


def test_beta_distribution_accepts_a_bounded_series() -> None:
    """The bounded [0, 1] family reaches the seam through the xarray path."""
    result = climate_indices.standardized_index(_monthly(), SCALE, indices.Distribution.beta, aggregation="mean")

    assert np.isfinite(result.values[SCALE:]).all()
    assert result.attrs["distribution"] == "beta"


def test_probability_output_scale_is_unclipped() -> None:
    """The probability scale stays a probability rather than a clipped z-score."""
    result = climate_indices.standardized_index(
        _monthly(), SCALE, indices.Distribution.empirical, output_scale="probability"
    )

    assert np.nanmin(result.values) >= 0.0
    assert np.nanmax(result.values) <= 1.0


def test_invalid_aggregation_is_rejected_before_computing() -> None:
    """An unknown aggregation raises instead of silently behaving as the default."""
    with pytest.raises(ValueError, match="aggregation"):
        climate_indices.standardized_index(
            _monthly(),
            SCALE,
            indices.Distribution.empirical,
            aggregation="median",  # type: ignore[arg-type]
        )


def test_unsupported_distribution_is_rejected() -> None:
    """Log-logistic stays a SPEI-only surface, as on the NumPy path."""
    with pytest.raises(InvalidArgumentError, match="Unsupported distribution"):
        climate_indices.standardized_index(_monthly(), SCALE, indices.Distribution.loglogistic)


def test_package_root_reexport_differs_from_the_numpy_core() -> None:
    """The public function is the adapted one, and the core stays available under indices."""
    assert climate_indices.standardized_index is not indices.standardized_index
    assert "standardized_index" in climate_indices.__all__
