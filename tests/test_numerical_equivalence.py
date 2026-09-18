"""Numerical equivalence between the Spatial Kernel path and the serial NumPy API (#930).

The serial reference here is the NumPy API applied cell by cell over a fully populated
grid; the vectorized side is the xarray adapter that scales, fits, ranks, or divides once
per Spatial Block. Both paths run the same core over the same values, so the results are
expected to be bit-for-bit identical, with one documented exception: Thornthwaite PET
reorders the same arithmetic inside its block and differs by a few float64 ULP.

Two boundaries that these bounds do not cover, on purpose:

* A block is fitted as a unit, so the Pearson fit's whole-block gamma fallback
  (`Distribution.pearson` via `indices.spi`) answers every cell in the block from one
  fit. That block semantics is documented in ADR-0009, not drift.
* Missing-data, partial-year, and daily-grid shapes keep their `atol=1e-8` equivalence
  contracts in `tests/test_spatial_kernel.py`, which also owns per-index shape, mask, and
  kernel-invocation contracts. The Palmer block path is pinned in
  `tests/test_palmer_spatial.py`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

from climate_indices import compute, eddi, indices, percentage_of_normal, pet_thornthwaite, spei, spi

# The one path whose block form reorders arithmetic: 1.14e-13 measured on the 5 x 6
# temperature fixture below (about 4 float64 ULP at the worst cell). The bound keeps
# roughly an order of magnitude of headroom and trips on any regression larger than
# itself; drift below the bound is accepted, not reported.
_THORNTHWAITE_ATOL = 1e-12

# (time series, cell latitude, matching secondary grid cell) -> single-series result
_Kernel = Callable[[xr.DataArray, float, "xr.DataArray | None"], np.ndarray]


def _pointwise(grid: xr.DataArray, kernel: _Kernel, secondary: xr.DataArray | None = None) -> np.ndarray:
    """Apply a single-series kernel over every cell of a `(time, lat, lon)` grid."""
    expected = np.full(grid.shape, np.nan)
    latitudes = grid.coords["lat"].values
    for latitude_index, latitude in enumerate(latitudes):
        for longitude_index in range(grid.sizes["lon"]):
            cell = secondary.isel(lat=latitude_index, lon=longitude_index) if secondary is not None else None
            expected[:, latitude_index, longitude_index] = kernel(
                grid.isel(lat=latitude_index, lon=longitude_index), float(latitude), cell
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
        lambda series, _latitude, _cell: indices.spi(
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
    # PET varies per cell, so a block path that pairs a cell with the wrong PET series fails here.
    pet_mm = gridded_monthly_precip_3d * 0.25 + 30.0

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
        lambda series, _latitude, pet_cell: indices.spei(
            precips_mm=np.asarray(series),
            pet_mm=np.asarray(pet_cell),
            scale=scale,
            distribution=indices.Distribution.gamma,
            data_start_year=data_start_year,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
            periodicity=compute.Periodicity.monthly,
        ),
        secondary=pet_mm,
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
        lambda series, _latitude, _cell: indices.eddi(
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
        lambda series, _latitude, _cell: indices.percentage_of_normal(
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
    """Thornthwaite PET over a Spatial Block agrees with the NumPy API within a few ULP."""
    data_start_year = int(gridded_monthly_temp_3d.time.dt.year[0])
    latitudes = gridded_monthly_temp_3d.coords["lat"].values

    result = pet_thornthwaite(gridded_monthly_temp_3d, xr.DataArray(latitudes, dims=["lat"]))
    expected = _pointwise(
        gridded_monthly_temp_3d,
        lambda series, latitude, _cell: indices.pet(np.asarray(series), latitude, data_start_year),
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

    eddi_eager = eddi(
        gridded_monthly_precip_3d,
        scale=3,
        calibration_year_initial=calibration_year_start_monthly,
        calibration_year_final=calibration_year_end_monthly,
    )
    eddi_lazy = eddi(
        dask_monthly_precip_3d,
        scale=3,
        calibration_year_initial=calibration_year_start_monthly,
        calibration_year_final=calibration_year_end_monthly,
    )
    np.testing.assert_array_equal(eddi_lazy.compute().values, eddi_eager.values)

    pnp_eager = percentage_of_normal(
        gridded_monthly_precip_3d,
        scale=6,
        calibration_start_year=calibration_year_start_monthly,
        calibration_end_year=calibration_year_end_monthly,
    )
    pnp_lazy = percentage_of_normal(
        dask_monthly_precip_3d,
        scale=6,
        calibration_start_year=calibration_year_start_monthly,
        calibration_end_year=calibration_year_end_monthly,
    )
    np.testing.assert_array_equal(pnp_lazy.compute().values, pnp_eager.values)

    # the shared dask temperature fixture is a single spatial chunk, so split it here
    latitudes = gridded_monthly_temp_3d.coords["lat"].values
    latitude = xr.DataArray(latitudes, dims=["lat"])
    thornthwaite_eager = pet_thornthwaite(gridded_monthly_temp_3d, latitude)
    thornthwaite_lazy = pet_thornthwaite(dask_monthly_temp_3d.chunk({"lat": 2, "lon": 3}), latitude)
    np.testing.assert_allclose(
        thornthwaite_lazy.compute().values,
        thornthwaite_eager.values,
        atol=_THORNTHWAITE_ATOL,
        rtol=0.0,
        equal_nan=True,
    )
