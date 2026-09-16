"""Tests for the spatial (per-block) kernel path added in #923.

The xarray adapter hands a 3-D and higher input to a kernel that accepts the time
core dimension alongside its cell dimensions, so gridded work costs one kernel call
per non-core block rather than one call per grid cell. These tests pin the no-loop
guarantee, the equivalence with the single-series path, and the NaN/shape contracts
the existing adapter tests already cover for the per-cell path.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import compute, indices
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.xarray_adapter import xarray_adapter

_CALIBRATION_START = 1981
_CALIBRATION_END = 2010


@pytest.fixture
def gridded_monthly_precip() -> xr.DataArray:
    """40 years of monthly precipitation over a 3 x 2 grid (time, lat, lon)."""
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(7)
    values = rng.gamma(shape=2.0, scale=2.0, size=(time.size, 3, 2))
    return xr.DataArray(
        values,
        coords={"time": time, "lat": [10.0, 20.0, 30.0], "lon": [0.0, 5.0]},
        dims=["time", "lat", "lon"],
    )


@pytest.fixture
def spatial_spi():
    """SPI adapter with the spatial kernel path enabled (as typed_public_api wires it)."""
    return xarray_adapter(
        cf_metadata=CF_METADATA["spi"],  # type: ignore[arg-type]
        index_display_name="SPI",
        spatial_kernel=True,
    )(indices.spi)


@pytest.fixture
def per_cell_spi():
    """SPI adapter without the spatial kernel path, i.e. one call per grid cell."""
    return xarray_adapter(
        cf_metadata=CF_METADATA["spi"],  # type: ignore[arg-type]
        index_display_name="SPI",
    )(indices.spi)


@pytest.fixture
def spatial_spei():
    """SPEI adapter with the spatial kernel path enabled."""
    return xarray_adapter(
        cf_metadata=CF_METADATA["spei"],  # type: ignore[arg-type]
        index_display_name="SPEI",
        additional_input_names=["pet_mm"],
        spatial_kernel=True,
    )(indices.spei)


@pytest.fixture
def per_cell_spei():
    """SPEI adapter without the spatial kernel path."""
    return xarray_adapter(
        cf_metadata=CF_METADATA["spei"],  # type: ignore[arg-type]
        index_display_name="SPEI",
        additional_input_names=["pet_mm"],
    )(indices.spei)


def _pointwise_spi(data: np.ndarray, scale: int, distribution: indices.Distribution) -> np.ndarray:
    """Compute SPI for every cell of a (time, lat, lon) array via the NumPy API."""
    result = np.empty(data.shape, dtype=float)
    for latitude in range(data.shape[1]):
        for longitude in range(data.shape[2]):
            result[:, latitude, longitude] = indices.spi(
                data[:, latitude, longitude],
                scale=scale,
                distribution=distribution,
                data_start_year=1980,
                calibration_year_initial=_CALIBRATION_START,
                calibration_year_final=_CALIBRATION_END,
                periodicity=compute.Periodicity.monthly,
            )
    return result


class TestSpatialKernelSkipsPerCellLoop:
    """The gridded path must not call the fitting kernel once per grid cell."""

    def test_spi_gamma_fits_once_for_gridded_input(self, gridded_monthly_precip, spatial_spi, monkeypatch):
        """A 3 x 2 grid runs the gamma fit once, not once per cell."""
        calls: list[tuple[int, ...]] = []
        original = compute.transform_fitted_gamma

        def counting_transform(values, *args, **kwargs):
            calls.append(np.shape(values))
            return original(values, *args, **kwargs)

        monkeypatch.setattr(compute, "transform_fitted_gamma", counting_transform)

        result = spatial_spi(
            gridded_monthly_precip,
            scale=3,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )

        assert result.shape == gridded_monthly_precip.shape
        assert len(calls) == 1, f"expected one vectorized fit, saw {len(calls)} calls"

    def test_spei_gamma_fits_once_for_gridded_input(self, gridded_monthly_precip, spatial_spei, monkeypatch):
        """SPEI over the same grid also runs the gamma fit once."""
        pet = xr.full_like(gridded_monthly_precip, 1.0)
        calls: list[tuple[int, ...]] = []
        original = compute.transform_fitted_gamma

        def counting_transform(values, *args, **kwargs):
            calls.append(np.shape(values))
            return original(values, *args, **kwargs)

        monkeypatch.setattr(compute, "transform_fitted_gamma", counting_transform)

        result = spatial_spei(
            gridded_monthly_precip,
            pet_mm=pet,
            scale=3,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )

        assert result.shape == gridded_monthly_precip.shape
        assert len(calls) == 1, f"expected one vectorized fit, saw {len(calls)} calls"

    def test_two_dimensional_input_keeps_per_cell_path(self, gridded_monthly_precip, spatial_spi, monkeypatch):
        """A (time, cell) input has one dimension to broadcast over and stays per cell."""
        calls: list[tuple[int, ...]] = []
        original = compute.transform_fitted_gamma

        def counting_transform(values, *args, **kwargs):
            calls.append(np.shape(values))
            return original(values, *args, **kwargs)

        monkeypatch.setattr(compute, "transform_fitted_gamma", counting_transform)

        transect = gridded_monthly_precip.isel(lon=0)
        result = spatial_spi(
            transect,
            scale=3,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )

        assert result.shape == transect.shape
        assert len(calls) == transect.shape[1]

    def test_kernel_without_spatial_path_keeps_per_cell_loop(self, gridded_monthly_precip, per_cell_spi, monkeypatch):
        """An adapter without the spatial kernel still calls the fitting kernel per cell."""
        calls: list[tuple[int, ...]] = []
        original = compute.transform_fitted_gamma

        def counting_transform(values, *args, **kwargs):
            calls.append(np.shape(values))
            return original(values, *args, **kwargs)

        monkeypatch.setattr(compute, "transform_fitted_gamma", counting_transform)

        per_cell_spi(
            gridded_monthly_precip,
            scale=3,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )

        cell_count = gridded_monthly_precip.shape[1] * gridded_monthly_precip.shape[2]
        assert len(calls) == cell_count

    def test_pearson_keeps_per_cell_loop(self, gridded_monthly_precip, spatial_spi, monkeypatch):
        """Pearson Type III's per-series L-moment fit still runs once per cell (#940)."""
        calls: list[tuple[int, ...]] = []
        original = compute.transform_fitted_pearson

        def counting_transform(values, *args, **kwargs):
            calls.append(np.shape(values))
            return original(values, *args, **kwargs)

        monkeypatch.setattr(compute, "transform_fitted_pearson", counting_transform)

        spatial_spi(
            gridded_monthly_precip,
            scale=3,
            distribution=indices.Distribution.pearson,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )

        cell_count = gridded_monthly_precip.shape[1] * gridded_monthly_precip.shape[2]
        assert len(calls) == cell_count


class TestSpatialKernelEquivalence:
    """Gridded output must match the single-series path."""

    @pytest.mark.parametrize("scale", [1, 3, 12])
    def test_spi_gamma_matches_pointwise(self, gridded_monthly_precip, spatial_spi, scale):
        """Every cell matches the NumPy API result for that cell's time series."""
        result = spatial_spi(
            gridded_monthly_precip,
            scale=scale,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )
        expected = _pointwise_spi(gridded_monthly_precip.values, scale, indices.Distribution.gamma)

        assert result.dims == gridded_monthly_precip.dims
        np.testing.assert_array_equal(np.isnan(result.values), np.isnan(expected))
        np.testing.assert_allclose(result.values, expected, atol=1e-8, rtol=1e-7, equal_nan=True)

    def test_spi_pearson_matches_pointwise(self, gridded_monthly_precip, spatial_spi):
        """The deferred Pearson path still produces per-cell-identical output."""
        result = spatial_spi(
            gridded_monthly_precip,
            scale=3,
            distribution=indices.Distribution.pearson,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )
        expected = _pointwise_spi(gridded_monthly_precip.values, 3, indices.Distribution.pearson)

        np.testing.assert_array_equal(np.isnan(result.values), np.isnan(expected))
        np.testing.assert_allclose(result.values, expected, atol=1e-8, rtol=1e-7, equal_nan=True)

    def test_spei_gamma_matches_pointwise(self, gridded_monthly_precip, spatial_spei):
        """SPEI reduces to the same result per cell, PET included."""
        pet = xr.full_like(gridded_monthly_precip, 0.5)
        result = spatial_spei(
            gridded_monthly_precip,
            pet_mm=pet,
            scale=6,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )

        expected = np.empty(gridded_monthly_precip.shape, dtype=float)
        for latitude in range(gridded_monthly_precip.shape[1]):
            for longitude in range(gridded_monthly_precip.shape[2]):
                expected[:, latitude, longitude] = indices.spei(
                    gridded_monthly_precip.values[:, latitude, longitude],
                    pet.values[:, latitude, longitude],
                    scale=6,
                    distribution=indices.Distribution.gamma,
                    periodicity=compute.Periodicity.monthly,
                    data_start_year=1980,
                    calibration_year_initial=_CALIBRATION_START,
                    calibration_year_final=_CALIBRATION_END,
                )

        np.testing.assert_array_equal(np.isnan(result.values), np.isnan(expected))
        np.testing.assert_allclose(result.values, expected, atol=1e-8, rtol=1e-7, equal_nan=True)

    def test_matches_per_cell_adapter(self, gridded_monthly_precip, spatial_spi, per_cell_spi):
        """The spatial and per-cell adapter paths agree, NaN layout included."""
        spatial_result = spatial_spi(
            gridded_monthly_precip,
            scale=12,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )
        per_cell_result = per_cell_spi(
            gridded_monthly_precip,
            scale=12,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )

        np.testing.assert_array_equal(np.isnan(spatial_result.values), np.isnan(per_cell_result.values))
        np.testing.assert_allclose(
            spatial_result.values,
            per_cell_result.values,
            atol=1e-8,
            rtol=1e-7,
            equal_nan=True,
        )

    def test_dask_input_matches_in_memory(self, gridded_monthly_precip, spatial_spi):
        """A Dask-backed grid produces the same values as an in-memory one."""
        chunked = gridded_monthly_precip.chunk({"time": -1, "lat": 2, "lon": 1})

        lazy_result = spatial_spi(
            chunked,
            scale=6,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )
        eager_result = spatial_spi(
            gridded_monthly_precip,
            scale=6,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )

        assert lazy_result.chunks is not None
        np.testing.assert_array_equal(np.isnan(lazy_result.values), np.isnan(eager_result.values))
        np.testing.assert_allclose(
            lazy_result.compute().values,
            eager_result.values,
            atol=1e-8,
            rtol=1e-7,
            equal_nan=True,
        )

    def test_nan_cells_match_per_cell_adapter(self, gridded_monthly_precip, spatial_spi, per_cell_spi):
        """All-missing and partially missing cells keep their NaN layout."""
        values = gridded_monthly_precip.values.copy()
        values[:, 2, 1] = np.nan
        values[:100, 1, 1] = np.nan
        values[5, 2, 0] = 0.0
        modified = gridded_monthly_precip.copy(data=values)

        spatial_result = spatial_spi(
            modified,
            scale=3,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )
        per_cell_result = per_cell_spi(
            modified,
            scale=3,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )

        assert np.all(np.isnan(spatial_result.values[:, 2, 1]))
        np.testing.assert_array_equal(np.isnan(spatial_result.values), np.isnan(per_cell_result.values))
        np.testing.assert_allclose(
            spatial_result.values,
            per_cell_result.values,
            atol=1e-8,
            rtol=1e-7,
            equal_nan=True,
        )

    def test_partial_final_year_matches_per_cell_adapter(self, spatial_spi, per_cell_spi):
        """A grid ending mid-year pads, computes, and trims back like the per-cell path."""
        time = pd.date_range("1980-01-01", periods=40 * 12 + 5, freq="MS")
        rng = np.random.default_rng(3)
        values = rng.gamma(shape=2.0, scale=2.0, size=(time.size, 2, 2))
        partial = xr.DataArray(
            values,
            coords={"time": time, "lat": [10.0, 20.0], "lon": [0.0, 5.0]},
            dims=["time", "lat", "lon"],
        )

        spatial_result = spatial_spi(
            partial,
            scale=3,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )
        per_cell_result = per_cell_spi(
            partial,
            scale=3,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )

        assert spatial_result.shape == partial.shape
        np.testing.assert_allclose(
            spatial_result.values,
            per_cell_result.values,
            atol=1e-8,
            rtol=1e-7,
            equal_nan=True,
        )

    def test_all_missing_grid_returns_missing(self, gridded_monthly_precip):
        """An entirely missing grid short-circuits to all-NaN output."""
        all_missing = np.full(gridded_monthly_precip.shape, np.nan)

        result = indices.spi(
            all_missing,
            scale=3,
            distribution=indices.Distribution.gamma,
            data_start_year=1980,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
            periodicity=compute.Periodicity.monthly,
        )

        assert result.shape == all_missing.shape
        assert np.all(np.isnan(result))

    def test_daily_grid_matches_per_cell_adapter(self, spatial_spi, per_cell_spi):
        """The 366-day calendar plan converts a whole grid, partial final year included."""
        time = pd.date_range("1980-01-01", "1999-06-30", freq="D")
        rng = np.random.default_rng(5)
        values = rng.gamma(shape=2.0, scale=2.0, size=(time.size, 2, 2))
        daily = xr.DataArray(
            values,
            coords={"time": time, "lat": [10.0, 20.0], "lon": [0.0, 5.0]},
            dims=["time", "lat", "lon"],
        )

        spatial_result = spatial_spi(
            daily,
            scale=30,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )
        per_cell_result = per_cell_spi(
            daily,
            scale=30,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )

        assert spatial_result.shape == daily.shape
        np.testing.assert_array_equal(np.isnan(spatial_result.values), np.isnan(per_cell_result.values))
        np.testing.assert_allclose(
            spatial_result.values,
            per_cell_result.values,
            atol=1e-8,
            rtol=1e-7,
            equal_nan=True,
        )


def _cell_series(data: np.ndarray):
    """Yield ((lat, lon), 1-D series) for every cell of a (time, lat, lon) array."""
    for latitude in range(data.shape[1]):
        for longitude in range(data.shape[2]):
            yield (latitude, longitude), data[:, latitude, longitude]


def _spatial_gamma_params(data: np.ndarray) -> dict[str, np.ndarray]:
    """Per-cell gamma fits packed as (period, lat, lon)."""
    alphas = np.empty((12, *data.shape[1:]))
    betas = np.empty((12, *data.shape[1:]))
    for (latitude, longitude), series in _cell_series(data):
        alphas[:, latitude, longitude], betas[:, latitude, longitude] = compute.gamma_parameters(
            series, 1980, _CALIBRATION_START, _CALIBRATION_END, compute.Periodicity.monthly
        )
    return {"alpha": alphas, "beta": betas}


def _spatial_pearson_params(data: np.ndarray) -> dict[str, np.ndarray]:
    """Per-cell Pearson Type III fits packed as (period, lat, lon)."""
    stacked = {key: np.empty((12, *data.shape[1:])) for key in ("prob_zero", "loc", "scale", "skew")}
    for (latitude, longitude), series in _cell_series(data):
        probabilities_of_zero, locs, scales, skews = compute.pearson_parameters(
            series, 1980, _CALIBRATION_START, _CALIBRATION_END, compute.Periodicity.monthly
        )
        stacked["prob_zero"][:, latitude, longitude] = probabilities_of_zero
        stacked["loc"][:, latitude, longitude] = locs
        stacked["scale"][:, latitude, longitude] = scales
        stacked["skew"][:, latitude, longitude] = skews
    return stacked


class TestSpatialFittingParameters:
    """Supplied fitting parameters must stay associated with their own cell (#944)."""

    @staticmethod
    def _assert_matches_pointwise(data, spatial_params, distribution, pointwise_params):
        result = indices.spi(
            data,
            scale=3,
            distribution=distribution,
            data_start_year=1980,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
            periodicity=compute.Periodicity.monthly,
            fitting_params=spatial_params,
        )

        for (latitude, longitude), series in _cell_series(data):
            expected = indices.spi(
                series,
                scale=3,
                distribution=distribution,
                data_start_year=1980,
                calibration_year_initial=_CALIBRATION_START,
                calibration_year_final=_CALIBRATION_END,
                periodicity=compute.Periodicity.monthly,
                fitting_params=pointwise_params(latitude, longitude),
            )
            np.testing.assert_allclose(result[:, latitude, longitude], expected, atol=1e-8, rtol=1e-7, equal_nan=True)

    def test_gamma_spatial_params_match_pointwise(self, gridded_monthly_precip):
        """Gamma parameters shaped (period, *cells) are sliced to the current cell."""
        data = gridded_monthly_precip.values
        params = _spatial_gamma_params(data)
        self._assert_matches_pointwise(
            data,
            params,
            indices.Distribution.gamma,
            lambda latitude, longitude: {key: value[:, latitude, longitude] for key, value in params.items()},
        )

    def test_gamma_period_params_are_shared_by_every_cell(self, gridded_monthly_precip):
        """Legacy (period,) gamma parameters broadcast over all cells, not the last cell axis."""
        data = gridded_monthly_precip.values
        alphas, betas = compute.gamma_parameters(
            data[:, 0, 0], 1980, _CALIBRATION_START, _CALIBRATION_END, compute.Periodicity.monthly
        )
        params = {"alpha": alphas, "beta": betas}
        self._assert_matches_pointwise(data, params, indices.Distribution.gamma, lambda *_: params)

    def test_pearson_spatial_params_match_pointwise(self, gridded_monthly_precip):
        """Pearson parameters shaped (period, *cells) follow the per-cell fit loop."""
        data = gridded_monthly_precip.values
        params = _spatial_pearson_params(data)
        self._assert_matches_pointwise(
            data,
            params,
            indices.Distribution.pearson,
            lambda latitude, longitude: {key: value[:, latitude, longitude] for key, value in params.items()},
        )
