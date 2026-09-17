"""Tests for the spatial (per-block) kernel path added in #923.

The xarray adapter hands a 3-D and higher input to a kernel that accepts the time
core dimension alongside its cell dimensions, so gridded work costs one kernel call
per non-core block rather than one call per grid cell. These tests pin the no-loop
guarantee, the equivalence with the single-series path, and the NaN/shape contracts
the existing adapter tests already cover for the per-cell path.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import compute, indices, typed_public_api
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.exceptions import DataShapeError, GoodnessOfFitWarning, InvalidArgumentError
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


@pytest.fixture
def spatial_eddi():
    """EDDI adapter with the spatial kernel path enabled (as typed_public_api wires it)."""
    return _spatial_adapter("eddi", indices.eddi)


@pytest.fixture
def per_cell_eddi():
    """EDDI adapter without the spatial kernel path, i.e. one call per grid cell."""
    return _spatial_adapter("eddi", indices.eddi, spatial_kernel=False)


@pytest.fixture
def spatial_percentage_of_normal():
    """Percentage-of-normal adapter with the spatial kernel path enabled."""
    return _spatial_adapter("percentage_of_normal", indices.percentage_of_normal)


@pytest.fixture
def per_cell_percentage_of_normal():
    """Percentage-of-normal adapter without the spatial kernel path."""
    return _spatial_adapter("percentage_of_normal", indices.percentage_of_normal, spatial_kernel=False)


def _grid_with_extra_months(extra_months: int, cells: tuple[int, int] = (3, 2)) -> xr.DataArray:
    """A (time, lat, lon) grid starting in 1980 and ending mid-year."""
    time = pd.date_range("1980-01-01", periods=40 * 12 + extra_months, freq="MS")
    rng = np.random.default_rng(23)
    values = rng.gamma(shape=2.0, scale=2.0, size=(time.size, *cells))
    return xr.DataArray(
        values,
        coords={"time": time, "lat": list(range(cells[0])), "lon": list(range(cells[1]))},
        dims=["time", "lat", "lon"],
    )


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
        """A 3 x 2 grid runs the gamma fit and transform once, not once per cell."""
        transforms: list[tuple[int, ...]] = []
        fits: list[tuple[int, ...]] = []
        original_transform = compute.transform_fitted_gamma
        original_fit = compute.gamma_parameters

        def counting_transform(values, *args, **kwargs):
            transforms.append(np.shape(values))
            return original_transform(values, *args, **kwargs)

        def counting_fit(values, *args, **kwargs):
            fits.append(np.shape(values))
            return original_fit(values, *args, **kwargs)

        monkeypatch.setattr(compute, "transform_fitted_gamma", counting_transform)
        monkeypatch.setattr(compute, "gamma_parameters", counting_fit)

        result = spatial_spi(
            gridded_monthly_precip,
            scale=3,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )

        assert result.shape == gridded_monthly_precip.shape
        assert len(transforms) == 1, f"expected one vectorized transform, saw {len(transforms)} calls"
        assert len(fits) == 1, f"expected one vectorized fit, saw {len(fits)} calls"
        # the fit sees the folded (years, periods, *cells) block, cells intact
        assert fits[0] == (40, 12, 3, 2)

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
            spatial_time_major=True,
        )

        assert result.shape == all_missing.shape
        assert np.all(np.isnan(result))

    def test_ambiguous_gridded_input_must_be_declared(self, gridded_monthly_precip):
        """A block whose first cell axis is the period length has to be declared.

        That shape reads equally as time-major (time, 12, *cells) and as the legacy
        (years, periods, *cells) layout, so it is the one gridded shape the core will
        not read by inference; declaring it takes the time-major reading, which is what
        the xarray adapter does for every block it packs.
        """
        ambiguous = np.asarray(gridded_monthly_precip.values[: 40 * 12, :1, :1]).repeat(12, axis=1)
        kwargs = {
            "scale": 3,
            "distribution": indices.Distribution.gamma,
            "data_start_year": 1980,
            "calibration_year_initial": _CALIBRATION_START,
            "calibration_year_final": _CALIBRATION_END,
            "periodicity": compute.Periodicity.monthly,
        }

        with pytest.raises(ValueError, match="ambiguous"):
            indices.spi(ambiguous, **kwargs)

        declared = indices.spi(ambiguous, spatial_time_major=True, **kwargs)

        assert declared.shape == ambiguous.shape

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


class TestSpatialPearsonDeferral:
    """The Pearson Type III path still fits one series per cell (#940)."""

    def test_spei_pearson_matches_pointwise(self, gridded_monthly_precip, spatial_spei):
        """Gridded SPEI with the deferred pearson fit matches the NumPy API per cell."""
        pet = xr.full_like(gridded_monthly_precip, 0.5)
        result = spatial_spei(
            gridded_monthly_precip,
            pet_mm=pet,
            scale=3,
            distribution=indices.Distribution.pearson,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )

        expected = np.empty(gridded_monthly_precip.shape, dtype=float)
        for latitude in range(gridded_monthly_precip.shape[1]):
            for longitude in range(gridded_monthly_precip.shape[2]):
                expected[:, latitude, longitude] = indices.spei(
                    gridded_monthly_precip.values[:, latitude, longitude],
                    pet.values[:, latitude, longitude],
                    scale=3,
                    distribution=indices.Distribution.pearson,
                    periodicity=compute.Periodicity.monthly,
                    data_start_year=1980,
                    calibration_year_initial=_CALIBRATION_START,
                    calibration_year_final=_CALIBRATION_END,
                )

        np.testing.assert_array_equal(np.isnan(result.values), np.isnan(expected))
        np.testing.assert_allclose(result.values, expected, atol=1e-8, rtol=1e-7, equal_nan=True)


class TestSpatialGoodnessOfFitParity:
    """The vectorized goodness-of-fit prefilter flags the same cells as the per-series check."""

    def test_poor_fit_counts_match_per_cell_path(self, spatial_spi, per_cell_spi):
        """A uniform per-period sample gives the same poor-fit count on both paths."""
        time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
        rng = np.random.default_rng(123)
        values = rng.uniform(0.1, 10.0, size=(time.size, 2, 3))
        grid = xr.DataArray(
            values,
            coords={"time": time, "lat": [10.0, 20.0], "lon": [0.0, 5.0, 10.0]},
            dims=["time", "lat", "lon"],
        )
        kwargs = {
            "scale": 1,
            "distribution": indices.Distribution.gamma,
            "calibration_year_initial": _CALIBRATION_START,
            "calibration_year_final": _CALIBRATION_END,
        }

        with warnings.catch_warnings(record=True) as spatial_warnings:
            warnings.simplefilter("always")
            spatial_spi(grid, **kwargs)
        with warnings.catch_warnings(record=True) as cell_warnings:
            warnings.simplefilter("always")
            per_cell_spi(grid, **kwargs)

        spatial_fits = [w for w in spatial_warnings if issubclass(w.category, GoodnessOfFitWarning)]
        cell_fits = [w for w in cell_warnings if issubclass(w.category, GoodnessOfFitWarning)]
        assert len(spatial_fits) == 1, "the spatial check aggregates one warning per call"
        # the same (time step, cell) pairs are flagged on both paths; the spatial warning
        # counts comparisons, while each per-cell warning counts its own time steps
        assert spatial_fits[0].message.poor_fit_count == sum(w.message.poor_fit_count for w in cell_fits)
        assert spatial_fits[0].message.total_steps == 12 * 2 * 3

    def test_daily_spei_grid_matches_per_cell_adapter(self, spatial_spei, per_cell_spei):
        """A daily grid with a PET secondary converts both inputs through the calendar plan."""
        time = pd.date_range("1980-01-01", "1999-06-30", freq="D")
        rng = np.random.default_rng(17)
        precip_values = rng.gamma(shape=2.0, scale=2.0, size=(time.size, 2, 2))
        pet_values = rng.gamma(shape=2.0, scale=1.0, size=(time.size, 2, 2))
        coords = {"time": time, "lat": [10.0, 20.0], "lon": [0.0, 5.0]}
        precip = xr.DataArray(precip_values, coords=coords, dims=["time", "lat", "lon"])
        pet = xr.DataArray(pet_values, coords=coords, dims=["time", "lat", "lon"])
        kwargs = {
            "pet_mm": pet,
            "scale": 30,
            "distribution": indices.Distribution.gamma,
            "calibration_year_initial": _CALIBRATION_START,
            "calibration_year_final": _CALIBRATION_END,
        }

        spatial_result = spatial_spei(precip, **kwargs)
        per_cell_result = per_cell_spei(precip, **kwargs)

        assert spatial_result.shape == precip.shape
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


class TestSpatialBlockContracts:
    """Input contracts that the spatial path has to keep from the per-cell path."""

    def test_one_dimensional_pet_secondary_broadcasts_over_cells(self, spatial_spei, per_cell_spei):
        """A single PET time series serves every cell, as it does on the per-cell path."""
        time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
        rng = np.random.default_rng(23)
        precip_values = rng.gamma(shape=2.0, scale=2.0, size=(time.size, 2, 2))
        precip = xr.DataArray(
            precip_values,
            coords={"time": time, "lat": [10.0, 20.0], "lon": [0.0, 5.0]},
            dims=["time", "lat", "lon"],
        )
        pet = rng.gamma(shape=2.0, scale=1.0, size=(time.size,))
        kwargs = {
            "pet_mm": pet,
            "scale": 3,
            "distribution": indices.Distribution.gamma,
            "calibration_year_initial": _CALIBRATION_START,
            "calibration_year_final": _CALIBRATION_END,
        }

        spatial_result = spatial_spei(precip, **kwargs)
        per_cell_result = per_cell_spei(precip, **kwargs)

        np.testing.assert_array_equal(np.isnan(spatial_result.values), np.isnan(per_cell_result.values))
        np.testing.assert_allclose(
            spatial_result.values,
            per_cell_result.values,
            atol=1e-8,
            rtol=1e-7,
            equal_nan=True,
        )

    def test_mismatched_parameter_cells_raise(self, gridded_monthly_precip):
        """A parameter array carrying the wrong cell dimensions is rejected, not ignored."""
        data = np.asarray(gridded_monthly_precip.values)
        params = {
            "prob_zero": np.zeros((12, 4, 4)),
            "loc": np.zeros((12, 4, 4)),
            "scale": np.ones((12, 4, 4)),
            "skew": np.zeros((12, 4, 4)),
        }

        with pytest.raises(ValueError, match="do not match the input's cells"):
            indices.spi(
                data,
                scale=3,
                distribution=indices.Distribution.pearson,
                data_start_year=1980,
                calibration_year_initial=_CALIBRATION_START,
                calibration_year_final=_CALIBRATION_END,
                periodicity=compute.Periodicity.monthly,
                fitting_params=params,
            )

    def test_kernel_without_the_spatial_contract_fails_loudly(self, gridded_monthly_precip):
        """An index registered as a spatial kernel must accept the declaration keyword."""

        def kernel_without_the_declaration(
            values,
            scale,
            data_start_year,
            calibration_start_year,
            calibration_end_year,
            periodicity,
        ):
            return values

        unregistered = xarray_adapter(index_display_name="PNP", spatial_kernel=True)(kernel_without_the_declaration)

        with pytest.raises(TypeError, match="spatial_time_major"):
            unregistered(
                gridded_monthly_precip,
                scale=3,
                data_start_year=1980,
                calibration_start_year=_CALIBRATION_START,
                calibration_end_year=_CALIBRATION_END,
                periodicity=compute.Periodicity.monthly,
            )

    def test_masked_pearson_cells_return_missing(self, gridded_monthly_precip):
        """A fully masked cell becomes NaN instead of the mask's fill value."""
        data = np.ma.masked_array(
            np.asarray(gridded_monthly_precip.values, dtype=float),
            mask=False,
        )
        data.mask[:, 1, 1] = True

        result = indices.spi(
            data,
            scale=3,
            distribution=indices.Distribution.pearson,
            data_start_year=1980,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
            periodicity=compute.Periodicity.monthly,
        )

        assert not np.ma.isMaskedArray(result)
        assert np.all(np.isnan(result[:, 1, 1]))
        assert np.isfinite(result[100, 0, 0])

    def test_incompatible_pet_cell_axes_raise(self, spatial_spei):
        """PET cell axes that cannot broadcast with the precipitation block are rejected."""
        precip = xr.DataArray(
            np.zeros((24, 2, 2)),
            coords={"time": pd.date_range("1980-01-01", periods=24, freq="MS"), "lat": [10.0, 20.0], "lon": [0.0, 5.0]},
            dims=["time", "lat", "lon"],
        )

        with pytest.raises(ValueError, match="Incompatible precipitation and PET arrays"):
            spatial_spei(
                precip,
                pet_mm=np.zeros((24, 3)),
                scale=3,
                distribution=indices.Distribution.gamma,
                calibration_year_initial=_CALIBRATION_START,
                calibration_year_final=_CALIBRATION_END,
            )


@pytest.fixture
def gridded_monthly_temps() -> xr.DataArray:
    """40 years of monthly mean temperatures over a 3 x 2 grid (time, lat, lon)."""
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(11)
    values = rng.uniform(-5.0, 28.0, size=(time.size, 3, 2))
    # a few missing months in one cell, to pin NaN propagation through the block
    values[10:13, 1, 1] = np.nan
    return xr.DataArray(
        values,
        coords={"time": time, "lat": [10.0, 20.0, 30.0], "lon": [0.0, 5.0]},
        dims=["time", "lat", "lon"],
    )


@pytest.fixture
def gridded_daily_temps() -> tuple[xr.DataArray, xr.DataArray]:
    """Three calendar years of daily tmin/tmax over a 2 x 2 grid (time, lat, lon)."""
    time = pd.date_range("2019-01-01", periods=1096, freq="D")
    rng = np.random.default_rng(12)
    tmin_values = rng.uniform(-5.0, 15.0, size=(time.size, 2, 2))
    # a missing stretch in one cell, to pin NaN propagation through the block path
    tmin_values[100:110, 1, 1] = np.nan
    tmax_values = tmin_values + rng.uniform(5.0, 15.0, size=(time.size, 2, 2))
    coords = {"time": time, "lat": [30.0, 45.0], "lon": [0.0, 10.0]}
    return (
        xr.DataArray(tmin_values, coords=coords, dims=["time", "lat", "lon"]),
        xr.DataArray(tmax_values, coords=coords, dims=["time", "lat", "lon"]),
    )


class TestSpatialPETKernels:
    """The PET adapters reach their NumPy kernels once per block (#941)."""

    def test_thornthwaite_runs_once_for_gridded_input(self, gridded_monthly_temps, monkeypatch):
        """A 3 x 2 grid reaches indices.pet once, as a (time, *cells) block."""
        from climate_indices.xarray_adapter import pet_thornthwaite

        blocks: list[tuple[int, ...]] = []
        original = indices.pet

        def counting_pet(values, *args, **kwargs):
            blocks.append(np.shape(values))
            return original(values, *args, **kwargs)

        monkeypatch.setattr(indices, "pet", counting_pet)

        result = pet_thornthwaite(gridded_monthly_temps, xr.DataArray([10.0, 20.0, 30.0], dims=["lat"]))

        assert result.shape == gridded_monthly_temps.shape
        assert blocks == [gridded_monthly_temps.shape]

    def test_hargreaves_runs_once_for_gridded_input(self, gridded_daily_temps, monkeypatch):
        """A 2 x 2 grid reaches eto.eto_hargreaves once, as a (time, *cells) block."""
        from climate_indices import eto
        from climate_indices.xarray_adapter import pet_hargreaves

        tmin, tmax = gridded_daily_temps
        blocks: list[tuple[int, ...]] = []
        original = eto.eto_hargreaves

        def counting_hargreaves(tmin_values, tmax_values, tmean_values, latitude_degrees, **kwargs):
            blocks.append(np.shape(tmean_values))
            return original(tmin_values, tmax_values, tmean_values, latitude_degrees, **kwargs)

        monkeypatch.setattr(eto, "eto_hargreaves", counting_hargreaves)

        result = pet_hargreaves(tmin, tmax, xr.DataArray([30.0, 45.0], dims=["lat"]))

        assert result.shape == tmin.shape
        # the kernel sees the all-leap (years, 366, *cells) block, every cell at once
        assert blocks == [(3 * 366, 2, 2)]

    def test_two_dimensional_input_keeps_per_cell_path(self, gridded_monthly_temps, monkeypatch):
        """A (time, cell) input has one dimension to broadcast over and stays per cell."""
        from climate_indices.xarray_adapter import pet_thornthwaite

        calls: list[tuple[int, ...]] = []
        original = indices.pet

        def counting_pet(values, *args, **kwargs):
            calls.append(np.shape(values))
            return original(values, *args, **kwargs)

        monkeypatch.setattr(indices, "pet", counting_pet)

        transect = gridded_monthly_temps.isel(lon=0)
        result = pet_thornthwaite(transect, 20.0)

        assert result.shape == transect.shape
        assert calls == [(transect.sizes["time"],)] * transect.sizes["lat"]

    def test_gridded_thornthwaite_matches_pointwise(self, gridded_monthly_temps):
        """Every cell matches the single-series result for that cell's latitude."""
        from climate_indices.xarray_adapter import pet_thornthwaite

        latitudes = gridded_monthly_temps.coords["lat"].values
        expected = np.full(gridded_monthly_temps.shape, np.nan)
        for latitude_index, latitude in enumerate(latitudes):
            for longitude_index in range(gridded_monthly_temps.sizes["lon"]):
                point = pet_thornthwaite(
                    gridded_monthly_temps.isel(lat=latitude_index, lon=longitude_index),
                    float(latitude),
                )
                expected[:, latitude_index, longitude_index] = point.values

        result = pet_thornthwaite(gridded_monthly_temps, xr.DataArray(latitudes, dims=["lat"]))

        assert result.dims == gridded_monthly_temps.dims
        np.testing.assert_array_equal(np.isnan(result.values), np.isnan(expected))
        np.testing.assert_allclose(result.values, expected, atol=1e-8, rtol=1e-7, equal_nan=True)

    def test_gridded_hargreaves_matches_pointwise(self, gridded_daily_temps):
        """Every cell matches the single-series result for that cell's latitude."""
        from climate_indices.xarray_adapter import pet_hargreaves

        tmin, tmax = gridded_daily_temps
        latitudes = tmin.coords["lat"].values
        expected = np.full(tmin.shape, np.nan)
        for latitude_index, latitude in enumerate(latitudes):
            for longitude_index in range(tmin.sizes["lon"]):
                point = pet_hargreaves(
                    tmin.isel(lat=latitude_index, lon=longitude_index),
                    tmax.isel(lat=latitude_index, lon=longitude_index),
                    float(latitude),
                )
                expected[:, latitude_index, longitude_index] = point.values

        result = pet_hargreaves(tmin, tmax, xr.DataArray(latitudes, dims=["lat"]))

        assert result.dims == tmin.dims
        np.testing.assert_array_equal(np.isnan(result.values), np.isnan(expected))
        np.testing.assert_allclose(result.values, expected, atol=1e-8, rtol=1e-7, equal_nan=True)

    def test_partial_year_hargreaves_block_is_not_padded(self, monkeypatch):
        """A block ending mid-year indexes its inputs in place instead of padding them."""
        from climate_indices import eto

        def no_padding(*args, **kwargs):
            raise AssertionError("Hargreaves padded a temperature input to a whole year")

        monkeypatch.setattr(np, "pad", no_padding)

        time_length = 2 * 366 + 100
        rng = np.random.default_rng(31)
        tmin = rng.uniform(-5.0, 15.0, size=(time_length, 2, 2))
        tmax = tmin + rng.uniform(5.0, 15.0, size=(time_length, 2, 2))
        tmean = (tmin + tmax) / 2.0

        block = eto.eto_hargreaves(tmin, tmax, tmean, np.full((2, 2), 30.0), spatial_time_major=True)
        scalar = eto.eto_hargreaves(tmin, tmax, tmean, 30.0, spatial_time_major=True)

        assert block.shape == tmin.shape
        np.testing.assert_array_equal(block, scalar)

    def test_gridded_thornthwaite_scalar_latitude_matches_pointwise(self, gridded_monthly_temps):
        """A scalar latitude reaches every cell of a gridded run."""
        from climate_indices.xarray_adapter import pet_thornthwaite

        expected = np.full(gridded_monthly_temps.shape, np.nan)
        for latitude_index in range(gridded_monthly_temps.sizes["lat"]):
            for longitude_index in range(gridded_monthly_temps.sizes["lon"]):
                point = pet_thornthwaite(
                    gridded_monthly_temps.isel(lat=latitude_index, lon=longitude_index),
                    25.0,
                )
                expected[:, latitude_index, longitude_index] = point.values

        result = pet_thornthwaite(gridded_monthly_temps, 25.0)

        np.testing.assert_array_equal(np.isnan(result.values), np.isnan(expected))
        np.testing.assert_allclose(result.values, expected, atol=1e-8, rtol=1e-7, equal_nan=True)

    def test_gridded_pet_dask_blocks_match_in_memory(self, gridded_monthly_temps, gridded_daily_temps):
        """Dask-backed gridded PET returns the same values as the in-memory block path."""
        from climate_indices.xarray_adapter import pet_hargreaves, pet_thornthwaite

        latitudes = xr.DataArray(gridded_monthly_temps.coords["lat"].values, dims=["lat"])
        expected_thornthwaite = pet_thornthwaite(gridded_monthly_temps, latitudes)
        chunked_thornthwaite = pet_thornthwaite(gridded_monthly_temps.chunk({"lat": 1}), latitudes)

        assert chunked_thornthwaite.chunks is not None
        np.testing.assert_allclose(
            chunked_thornthwaite.values,
            expected_thornthwaite.values,
            atol=1e-8,
            rtol=1e-7,
            equal_nan=True,
        )

        tmin, tmax = gridded_daily_temps
        daily_latitudes = xr.DataArray(tmin.coords["lat"].values, dims=["lat"])
        expected_hargreaves = pet_hargreaves(tmin, tmax, daily_latitudes)
        # a split time dimension exercises the rechunk the daily calendar needs
        chunked_hargreaves = pet_hargreaves(tmin.chunk({"time": 400}), tmax.chunk({"time": 400}), daily_latitudes)

        assert chunked_hargreaves.chunks is not None
        np.testing.assert_allclose(
            chunked_hargreaves.values,
            expected_hargreaves.values,
            atol=1e-8,
            rtol=1e-7,
            equal_nan=True,
        )

    def test_thornthwaite_transposed_cell_dims_match_pointwise(self):
        """A (time, lon, lat) grid gets each cell's own latitude, not a transposed reading."""
        from climate_indices.xarray_adapter import pet_thornthwaite

        time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
        rng = np.random.default_rng(22)
        temperatures = xr.DataArray(
            rng.uniform(-5.0, 28.0, size=(time.size, 2, 3)),
            coords={"time": time, "lon": [0.0, 10.0], "lat": [10.0, 20.0, 30.0]},
            dims=["time", "lon", "lat"],
        )
        latitudes = xr.DataArray([10.0, 20.0, 30.0], dims=["lat"])
        expected = np.full(temperatures.shape, np.nan)
        for latitude_index, latitude in enumerate([10.0, 20.0, 30.0]):
            for longitude_index in range(2):
                point = pet_thornthwaite(temperatures.isel(lon=longitude_index, lat=latitude_index), latitude)
                expected[:, longitude_index, latitude_index] = point.values

        result = pet_thornthwaite(temperatures, latitudes)

        assert result.dims == temperatures.dims
        np.testing.assert_allclose(result.values, expected, atol=1e-8, rtol=1e-7, equal_nan=True)

    def test_thornthwaite_partial_final_year_matches_pointwise(self):
        """A block ending mid-year matches the per-cell path without leaking padded months."""
        from climate_indices.xarray_adapter import pet_thornthwaite

        time = pd.date_range("1980-01-01", periods=24 + 5, freq="MS")
        rng = np.random.default_rng(23)
        temperatures = xr.DataArray(
            rng.uniform(-5.0, 28.0, size=(time.size, 2, 2)),
            coords={"time": time, "lat": [15.0, 35.0], "lon": [0.0, 10.0]},
            dims=["time", "lat", "lon"],
        )
        latitudes = xr.DataArray([15.0, 35.0], dims=["lat"])
        expected = np.full(temperatures.shape, np.nan)
        for latitude_index, latitude in enumerate([15.0, 35.0]):
            for longitude_index in range(2):
                point = pet_thornthwaite(temperatures.isel(lat=latitude_index, lon=longitude_index), latitude)
                expected[:, latitude_index, longitude_index] = point.values

        result = pet_thornthwaite(temperatures, latitudes)

        np.testing.assert_array_equal(np.isnan(result.values), np.isnan(expected))
        np.testing.assert_allclose(result.values, expected, atol=1e-8, rtol=1e-7, equal_nan=True)

    def test_dask_blocks_reach_the_kernel_once_per_block(self, gridded_monthly_temps, monkeypatch):
        """The Dask path hands each block over once, not once per grid cell."""
        from climate_indices.xarray_adapter import pet_thornthwaite

        calls: list[tuple[int, ...]] = []
        original = indices.pet

        def counting_pet(values, *args, **kwargs):
            calls.append(np.shape(values))
            return original(values, *args, **kwargs)

        monkeypatch.setattr(indices, "pet", counting_pet)

        chunked = gridded_monthly_temps.chunk({"lat": 1, "lon": 1})
        result = pet_thornthwaite(chunked, xr.DataArray([10.0, 20.0, 30.0], dims=["lat"])).compute()

        assert result.shape == gridded_monthly_temps.shape
        cell_count = gridded_monthly_temps.sizes["lat"] * gridded_monthly_temps.sizes["lon"]
        assert len(calls) == cell_count, f"expected one call per Dask block, saw {len(calls)}"
        assert all(shape[0] == gridded_monthly_temps.sizes["time"] for shape in calls)

    def test_block_latitude_that_cannot_broadcast_raises(self):
        """A block latitude that does not fit the cell dimensions is rejected, not read askew."""
        temperatures = np.full((24, 2, 3), 15.0)

        with pytest.raises(InvalidArgumentError, match="does not broadcast"):
            indices.pet(temperatures, np.array([10.0, 20.0]), 2000, spatial_time_major=True)

        # a per-latitude column does broadcast, and matches the per-cell path
        latitudes = np.full((2, 1), 20.0)
        result = indices.pet(temperatures, latitudes, 2000, spatial_time_major=True)

        assert result.shape == temperatures.shape
        for latitude_index in range(2):
            for longitude_index in range(3):
                np.testing.assert_allclose(
                    result[:, latitude_index, longitude_index],
                    indices.pet(temperatures[:, latitude_index, longitude_index], 20.0, 2000),
                    atol=1e-8,
                    rtol=1e-7,
                    equal_nan=True,
                )


def _spatial_adapter(metadata_key: str, kernel, *, spatial_kernel: bool = True):
    """Build the adapter a typed public API entry point wires for an index."""
    return xarray_adapter(
        cf_metadata=CF_METADATA[metadata_key],  # type: ignore[arg-type]
        index_display_name=metadata_key.upper(),
        spatial_kernel=spatial_kernel,
    )(kernel)


def _pointwise(values: np.ndarray, kernel) -> np.ndarray:
    """Apply a single-series NumPy kernel to every cell of a (time, lat, lon) array."""
    result = np.empty(values.shape, dtype=float)
    for latitude in range(values.shape[1]):
        for longitude in range(values.shape[2]):
            result[:, latitude, longitude] = kernel(values[:, latitude, longitude])
    return result


def _eddi_grid(values: np.ndarray, scale: int) -> np.ndarray:
    """EDDI per cell through the stable NumPy API."""
    return _pointwise(
        values,
        lambda series: indices.eddi(
            series,
            scale=scale,
            data_start_year=1980,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
            periodicity=compute.Periodicity.monthly,
        ),
    )


def _percentage_of_normal_grid(values: np.ndarray, scale: int) -> np.ndarray:
    """Percentage of normal per cell through the stable NumPy API."""
    return _pointwise(
        values,
        lambda series: indices.percentage_of_normal(
            series,
            scale=scale,
            data_start_year=1980,
            calibration_start_year=_CALIBRATION_START,
            calibration_end_year=_CALIBRATION_END,
            periodicity=compute.Periodicity.monthly,
        ),
    )


class TestSpatialEDDI:
    """EDDI ranks every cell of a (time, *cells) block in one pass (#942)."""

    def test_ranks_once_for_gridded_input(self, gridded_monthly_precip, monkeypatch):
        """A 3 x 2 grid reaches the ranking pass once, over the packed block."""
        shapes: list[tuple[int, ...]] = []
        original = indices._hastings_inverse_normal

        def counting_inverse_normal(probability):
            shapes.append(np.shape(probability))
            return original(probability)

        monkeypatch.setattr(indices, "_hastings_inverse_normal", counting_inverse_normal)

        result = typed_public_api.eddi(
            gridded_monthly_precip,
            scale=3,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )

        # the ranking pass sees the (years, periods, *cells) layout of the whole grid
        assert shapes == [(40, 12, 3, 2)]
        assert result.shape == gridded_monthly_precip.shape

    def test_matches_the_single_series_path(self, gridded_monthly_precip):
        """The block ranking is the per-series ranking, cell for cell."""
        spatial_eddi = _spatial_adapter("eddi", indices.eddi)

        result = spatial_eddi(
            gridded_monthly_precip,
            scale=3,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )

        np.testing.assert_array_equal(result.values, _eddi_grid(gridded_monthly_precip.values, scale=3))

    def test_partial_final_period_matches_per_cell_adapter(self, spatial_eddi, per_cell_eddi):
        """A grid ending mid-year pads, ranks, and trims back like the per-cell path."""
        partial = _grid_with_extra_months(5)

        spatial_result = spatial_eddi(
            partial,
            scale=3,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )
        per_cell_result = per_cell_eddi(
            partial,
            scale=3,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )

        assert spatial_result.shape == partial.shape
        np.testing.assert_array_equal(spatial_result.values, per_cell_result.values)

    def test_nan_cells_match_per_cell_adapter(self, gridded_monthly_precip, spatial_eddi, per_cell_eddi):
        """Missing observations keep their NaN positions and stay out of the climatology."""
        values = gridded_monthly_precip.values.copy()
        values[10:20, 1, 0] = np.nan
        missing_cell = gridded_monthly_precip.copy(data=values)

        spatial_result = spatial_eddi(
            missing_cell,
            scale=6,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )
        per_cell_result = per_cell_eddi(
            missing_cell,
            scale=6,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )

        np.testing.assert_array_equal(np.isnan(spatial_result.values), np.isnan(per_cell_result.values))
        np.testing.assert_array_equal(spatial_result.values, per_cell_result.values)
        assert np.all(np.isnan(spatial_result.values[10:20, 1, 0]))

    def test_daily_grid_matches_per_cell_adapter(self, spatial_eddi, per_cell_eddi):
        """The 366-day calendar plan ranks a whole grid, partial final year included."""
        time = pd.date_range("2000-01-01", "2004-06-30", freq="D")
        rng = np.random.default_rng(13)
        daily = xr.DataArray(
            rng.gamma(shape=2.0, scale=2.0, size=(time.size, 2, 2)),
            coords={"time": time, "lat": [10.0, 20.0], "lon": [0.0, 5.0]},
            dims=["time", "lat", "lon"],
        )

        spatial_result = spatial_eddi(
            daily,
            scale=30,
            calibration_year_initial=2000,
            calibration_year_final=2003,
        )
        per_cell_result = per_cell_eddi(
            daily,
            scale=30,
            calibration_year_initial=2000,
            calibration_year_final=2003,
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

    def test_dask_blocks_match_in_memory(self, gridded_monthly_precip, spatial_eddi):
        """A Dask-backed grid ranks the same values as an in-memory one."""
        chunked = gridded_monthly_precip.chunk({"time": -1, "lat": 2, "lon": 1})

        lazy_result = spatial_eddi(
            chunked,
            scale=6,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )
        eager_result = spatial_eddi(
            gridded_monthly_precip,
            scale=6,
            calibration_year_initial=_CALIBRATION_START,
            calibration_year_final=_CALIBRATION_END,
        )

        assert lazy_result.chunks is not None
        np.testing.assert_array_equal(lazy_result.compute().values, eager_result.values)

    def test_undeclared_block_raises_from_the_numpy_api(self, gridded_monthly_precip):
        """A 3-D array has to be declared as a time-major block, and EDDI says so."""
        with pytest.raises(DataShapeError, match="spatial_time_major"):
            indices.eddi(
                gridded_monthly_precip.values,
                scale=3,
                data_start_year=1980,
                calibration_year_initial=_CALIBRATION_START,
                calibration_year_final=_CALIBRATION_END,
                periodicity=compute.Periodicity.monthly,
            )


class TestSpatialPercentageOfNormal:
    """Percentage of normal divides every cell of a block by its own normals (#942)."""

    def test_divides_once_for_gridded_input(self, gridded_monthly_precip, monkeypatch):
        """A 3 x 2 grid reaches the preparation seam once, not once per grid cell."""
        shapes: list[tuple[int, ...]] = []
        original = compute.prepare_scaled

        def counting_prepare_scaled(*args, **kwargs):
            shapes.append(np.shape(args[0]))
            return original(*args, **kwargs)

        monkeypatch.setattr(compute, "prepare_scaled", counting_prepare_scaled)

        result = typed_public_api.percentage_of_normal(
            gridded_monthly_precip,
            scale=3,
            data_start_year=1980,
            calibration_start_year=_CALIBRATION_START,
            calibration_end_year=_CALIBRATION_END,
        )

        assert shapes == [(40 * 12, 3, 2)]
        assert result.shape == gridded_monthly_precip.shape

    def test_matches_the_single_series_path(self, gridded_monthly_precip):
        """The block normals are the per-series normals, cell for cell."""
        spatial_percentage_of_normal = _spatial_adapter("percentage_of_normal", indices.percentage_of_normal)

        result = spatial_percentage_of_normal(
            gridded_monthly_precip,
            scale=3,
            data_start_year=1980,
            calibration_start_year=_CALIBRATION_START,
            calibration_end_year=_CALIBRATION_END,
        )

        np.testing.assert_allclose(
            result.values,
            _percentage_of_normal_grid(gridded_monthly_precip.values, scale=3),
            atol=1e-8,
            rtol=1e-7,
            equal_nan=True,
        )

    def test_partial_final_period_matches_per_cell_adapter(
        self, spatial_percentage_of_normal, per_cell_percentage_of_normal
    ):
        """The trailing partial period divides by the normals of the periods it covers."""
        partial = _grid_with_extra_months(5)

        spatial_result = spatial_percentage_of_normal(
            partial,
            scale=3,
            data_start_year=1980,
            calibration_start_year=_CALIBRATION_START,
            calibration_end_year=_CALIBRATION_END,
        )
        per_cell_result = per_cell_percentage_of_normal(
            partial,
            scale=3,
            data_start_year=1980,
            calibration_start_year=_CALIBRATION_START,
            calibration_end_year=_CALIBRATION_END,
        )

        assert spatial_result.shape == partial.shape
        np.testing.assert_allclose(
            spatial_result.values,
            per_cell_result.values,
            atol=1e-8,
            rtol=1e-7,
            equal_nan=True,
        )

    def test_nan_cells_match_per_cell_adapter(
        self,
        gridded_monthly_precip,
        spatial_percentage_of_normal,
        per_cell_percentage_of_normal,
    ):
        """Missing observations keep their NaN positions and stay out of the normals."""
        values = gridded_monthly_precip.values.copy()
        values[10:20, 1, 0] = np.nan
        missing_cell = gridded_monthly_precip.copy(data=values)

        spatial_result = spatial_percentage_of_normal(
            missing_cell,
            scale=6,
            data_start_year=1980,
            calibration_start_year=_CALIBRATION_START,
            calibration_end_year=_CALIBRATION_END,
        )
        per_cell_result = per_cell_percentage_of_normal(
            missing_cell,
            scale=6,
            data_start_year=1980,
            calibration_start_year=_CALIBRATION_START,
            calibration_end_year=_CALIBRATION_END,
        )

        np.testing.assert_array_equal(np.isnan(spatial_result.values), np.isnan(per_cell_result.values))
        np.testing.assert_allclose(
            spatial_result.values,
            per_cell_result.values,
            atol=1e-8,
            rtol=1e-7,
            equal_nan=True,
        )

    def test_daily_grid_matches_per_cell_adapter(self, spatial_percentage_of_normal, per_cell_percentage_of_normal):
        """A daily grid divides each day by its own day-of-year normal."""
        time = pd.date_range("2000-01-01", "2004-06-30", freq="D")
        rng = np.random.default_rng(17)
        daily = xr.DataArray(
            rng.gamma(shape=2.0, scale=2.0, size=(time.size, 2, 2)),
            coords={"time": time, "lat": [10.0, 20.0], "lon": [0.0, 5.0]},
            dims=["time", "lat", "lon"],
        )

        spatial_result = spatial_percentage_of_normal(
            daily,
            scale=30,
            data_start_year=2000,
            calibration_start_year=2000,
            calibration_end_year=2003,
        )
        per_cell_result = per_cell_percentage_of_normal(
            daily,
            scale=30,
            data_start_year=2000,
            calibration_start_year=2000,
            calibration_end_year=2003,
        )

        assert spatial_result.shape == daily.shape
        np.testing.assert_allclose(
            spatial_result.values,
            per_cell_result.values,
            atol=1e-8,
            rtol=1e-7,
            equal_nan=True,
        )

    def test_dask_blocks_match_in_memory(self, gridded_monthly_precip, spatial_percentage_of_normal):
        """A Dask-backed grid divides the same values as an in-memory one."""
        chunked = gridded_monthly_precip.chunk({"time": -1, "lat": 2, "lon": 1})

        lazy_result = spatial_percentage_of_normal(
            chunked,
            scale=6,
            data_start_year=1980,
            calibration_start_year=_CALIBRATION_START,
            calibration_end_year=_CALIBRATION_END,
        )
        eager_result = spatial_percentage_of_normal(
            gridded_monthly_precip,
            scale=6,
            data_start_year=1980,
            calibration_start_year=_CALIBRATION_START,
            calibration_end_year=_CALIBRATION_END,
        )

        assert lazy_result.chunks is not None
        np.testing.assert_allclose(
            lazy_result.compute().values,
            eager_result.values,
            atol=1e-8,
            rtol=1e-7,
            equal_nan=True,
        )

    def test_all_missing_block_short_circuits(self):
        """An entirely missing block is returned as it arrived, without averaging."""
        block = np.full((24, 2, 2), np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("error")

            result = indices.percentage_of_normal(
                block,
                3,
                1980,
                1980,
                1981,
                compute.Periodicity.monthly,
                spatial_time_major=True,
            )

        assert result.shape == block.shape
        assert np.all(np.isnan(result))

    def test_undeclared_block_raises_from_the_numpy_api(self, gridded_monthly_precip):
        """A 3-D array has to be declared as a time-major block, and PNP says so."""
        with pytest.raises(DataShapeError, match="spatial_time_major"):
            indices.percentage_of_normal(
                gridded_monthly_precip.values,
                3,
                1980,
                _CALIBRATION_START,
                _CALIBRATION_END,
                compute.Periodicity.monthly,
            )
