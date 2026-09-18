"""Numerical equivalence between the Spatial Kernel path and the serial NumPy API (#930).

The serial reference here is the NumPy API applied cell by cell over a grid; the
vectorized side is the xarray adapter that scales, fits, ranks, or divides once per
Spatial Block. Both paths run the same core over the same values, so the results are
expected to be bit-for-bit identical, with one documented exception: Thornthwaite PET
reorders the same arithmetic inside its block and differs by a few float64 ULP.

Divergence beyond these bounds means the block path changed the science, and the
assertions below are the tripwire. Per-index shape, mask, and NaN contracts live in
`tests/test_spatial_kernel.py`, and the Palmer block path is pinned in
`tests/test_palmer_spatial.py`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

from climate_indices import compute, eddi, indices, percentage_of_normal, pet_thornthwaite, spei, spi

# The only path whose block form reorders arithmetic: the measured spread is 5.7e-14
# (about one float64 ULP) on the reference grid, asserted here with four orders of
# headroom so the bound still trips on any change that moves a cell's value.
_THORNTHWAITE_ATOL = 1e-12


def _pointwise(grid: xr.DataArray, kernel: Callable[[xr.DataArray, float], np.ndarray]) -> np.ndarray:
    """Apply a single-series kernel over every cell of a `(time, lat, lon)` grid."""
    expected = np.full(grid.shape, np.nan)
    latitudes = grid.coords["lat"].values
    for latitude_index, latitude in enumerate(latitudes):
        for longitude_index in range(grid.sizes["lon"]):
            expected[:, latitude_index, longitude_index] = kernel(
                grid.isel(lat=latitude_index, lon=longitude_index), float(latitude)
            )
    return expected


@pytest.mark.parametrize("distribution", [indices.Distribution.gamma, indices.Distribution.pearson])
def test_spi_block_path_matches_the_serial_numpy_api(
    gridded_monthly_precip_3d: xr.DataArray,
    calibration_year_start_monthly: int,
    calibration_year_end_monthly: int,
    distribution: indices.Distribution,
) -> None:
    """SPI over a Spatial Block equals SPI called once per cell, bit for bit."""
    scale = 6
    data_start_year = int(gridded_monthly_precip_3d.time.dt.year[0])

    result = spi(
        gridded_monthly_precip_3d,
        scale=scale,
        distribution=distribution,
        calibration_year_initial=calibration_year_start_monthly,
        calibration_year_final=calibration_year_end_monthly,
    )
    expected = _pointwise(
        gridded_monthly_precip_3d,
        lambda series, _latitude: indices.spi(
            np.asarray(series),
            scale=scale,
            distribution=distribution,
            data_start_year=data_start_year,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
            periodicity=compute.Periodicity.monthly,
        ),
    )

    np.testing.assert_array_equal(result.values, expected)


def test_spei_block_path_matches_the_serial_numpy_api(
    gridded_monthly_precip_3d: xr.DataArray,
    calibration_year_start_monthly: int,
    calibration_year_end_monthly: int,
) -> None:
    """SPEI over a Spatial Block equals SPEI called once per cell, bit for bit."""
    scale = 6
    data_start_year = int(gridded_monthly_precip_3d.time.dt.year[0])
    pet_mm = xr.full_like(gridded_monthly_precip_3d, 60.0)

    result = spei(
        gridded_monthly_precip_3d,
        pet_mm=pet_mm,
        scale=scale,
        distribution=indices.Distribution.gamma,
        calibration_year_initial=calibration_year_start_monthly,
        calibration_year_final=calibration_year_end_monthly,
    )
    expected = _pointwise(
        gridded_monthly_precip_3d,
        lambda series, _latitude: indices.spei(
            precips_mm=np.asarray(series),
            pet_mm=np.full(series.shape, 60.0),
            scale=scale,
            distribution=indices.Distribution.gamma,
            data_start_year=data_start_year,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
            periodicity=compute.Periodicity.monthly,
        ),
    )

    np.testing.assert_array_equal(result.values, expected)


def test_eddi_block_path_matches_the_serial_numpy_api(
    gridded_monthly_precip_3d: xr.DataArray,
    calibration_year_start_monthly: int,
    calibration_year_end_monthly: int,
) -> None:
    """EDDI over a Spatial Block equals EDDI called once per cell, bit for bit."""
    scale = 3
    data_start_year = int(gridded_monthly_precip_3d.time.dt.year[0])

    result = eddi(
        gridded_monthly_precip_3d,
        scale=scale,
        calibration_year_initial=calibration_year_start_monthly,
        calibration_year_final=calibration_year_end_monthly,
    )
    expected = _pointwise(
        gridded_monthly_precip_3d,
        lambda series, _latitude: indices.eddi(
            pet_values=np.asarray(series),
            scale=scale,
            data_start_year=data_start_year,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
            periodicity=compute.Periodicity.monthly,
        ),
    )

    np.testing.assert_array_equal(result.values, expected)


def test_percentage_of_normal_block_path_matches_the_serial_numpy_api(
    gridded_monthly_precip_3d: xr.DataArray,
    calibration_year_start_monthly: int,
    calibration_year_end_monthly: int,
) -> None:
    """Percentage of normal over a Spatial Block equals the per-cell result, bit for bit."""
    scale = 6
    data_start_year = int(gridded_monthly_precip_3d.time.dt.year[0])

    result = percentage_of_normal(
        gridded_monthly_precip_3d,
        scale=scale,
        calibration_start_year=calibration_year_start_monthly,
        calibration_end_year=calibration_year_end_monthly,
    )
    expected = _pointwise(
        gridded_monthly_precip_3d,
        lambda series, _latitude: indices.percentage_of_normal(
            values=np.asarray(series),
            scale=scale,
            data_start_year=data_start_year,
            calibration_start_year=calibration_year_start_monthly,
            calibration_end_year=calibration_year_end_monthly,
            periodicity=compute.Periodicity.monthly,
        ),
    )

    np.testing.assert_array_equal(result.values, expected)


def test_thornthwaite_block_path_stays_within_the_documented_tolerance(
    gridded_monthly_temp_3d: xr.DataArray,
) -> None:
    """Thornthwaite PET over a Spatial Block agrees with the per-cell result within one ULP."""
    latitudes = gridded_monthly_temp_3d.coords["lat"].values

    result = pet_thornthwaite(gridded_monthly_temp_3d, xr.DataArray(latitudes, dims=["lat"]))
    expected = _pointwise(
        gridded_monthly_temp_3d,
        lambda series, latitude: pet_thornthwaite(series, latitude).values,
    )

    np.testing.assert_allclose(result.values, expected, atol=_THORNTHWAITE_ATOL, rtol=0.0, equal_nan=True)


def test_dask_chunk_layout_does_not_change_the_result(
    gridded_monthly_precip_3d: xr.DataArray,
    dask_monthly_precip_3d: xr.DataArray,
    gridded_monthly_temp_3d: xr.DataArray,
    dask_monthly_temp_3d: xr.DataArray,
    calibration_year_start_monthly: int,
    calibration_year_end_monthly: int,
) -> None:
    """A Dask-backed grid matches the same grid in memory, block for block."""
    spi_eager = spi(
        gridded_monthly_precip_3d,
        scale=6,
        distribution=indices.Distribution.gamma,
        calibration_year_initial=calibration_year_start_monthly,
        calibration_year_final=calibration_year_end_monthly,
    )
    spi_lazy = spi(
        dask_monthly_precip_3d,
        scale=6,
        distribution=indices.Distribution.gamma,
        calibration_year_initial=calibration_year_start_monthly,
        calibration_year_final=calibration_year_end_monthly,
    )
    np.testing.assert_array_equal(spi_lazy.compute().values, spi_eager.values)

    latitudes = gridded_monthly_temp_3d.coords["lat"].values
    latitude = xr.DataArray(latitudes, dims=["lat"])
    thornthwaite_eager = pet_thornthwaite(gridded_monthly_temp_3d, latitude)
    thornthwaite_lazy = pet_thornthwaite(dask_monthly_temp_3d, latitude)
    np.testing.assert_allclose(
        thornthwaite_lazy.compute().values,
        thornthwaite_eager.values,
        atol=_THORNTHWAITE_ATOL,
        rtol=0.0,
        equal_nan=True,
    )
