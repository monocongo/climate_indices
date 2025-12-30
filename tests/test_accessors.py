import numpy as np
import pytest
import xarray as xr

import climate_indices  # noqa: F401
from climate_indices import compute, indices


def test_spi_accessor_round_trip(tmp_path):
    values = np.array(
        [
            10.0,
            20.0,
            0.0,
            5.0,
            15.0,
            0.0,
            8.0,
            12.0,
            3.0,
            7.0,
            9.0,
            4.0,
            11.0,
            13.0,
            np.nan,
            6.0,
            14.0,
            2.0,
            1.0,
            16.0,
            18.0,
            0.0,
            5.0,
            9.0,
        ],
        dtype=float,
    )
    time = np.arange(values.size)
    da = xr.DataArray(values, dims=("time",), coords={"time": time}, name="precip")
    ds = xr.Dataset({"precip": da})

    nc_path = tmp_path / "precip.nc"
    ds.to_netcdf(nc_path, engine="h5netcdf")

    with xr.open_dataset(nc_path) as opened:
        precip = opened["precip"]
        spi_da = precip.indices.spi(
            scale=3,
            distribution="gamma",
            data_start_year=2000,
            calibration_year_initial=2000,
            calibration_year_final=2001,
            periodicity=compute.Periodicity.monthly,
        )

    expected = indices.spi(
        values,
        3,
        indices.Distribution.gamma,
        2000,
        2000,
        2001,
        compute.Periodicity.monthly,
    )

    assert isinstance(spi_da, xr.DataArray)
    assert spi_da.dims == ("time",)
    assert np.array_equal(spi_da["time"].values, time)
    assert spi_da.attrs["long_name"] == "Standardized Precipitation Index"
    assert spi_da.attrs["units"] == "unitless"
    np.testing.assert_allclose(spi_da.values, expected, equal_nan=True)


def test_spi_accessor_pearson_distribution():
    """Test accessor with Pearson Type III distribution."""
    values = np.array(
        [
            10.0,
            20.0,
            0.0,
            5.0,
            15.0,
            0.0,
            8.0,
            12.0,
            3.0,
            7.0,
            9.0,
            4.0,
            11.0,
            13.0,
            2.0,
            6.0,
            14.0,
            2.0,
            1.0,
            16.0,
            18.0,
            0.0,
            5.0,
            9.0,
        ],
        dtype=float,
    )
    time = np.arange(values.size)
    da = xr.DataArray(values, dims=("time",), coords={"time": time}, name="precip")

    spi_da = da.indices.spi(
        scale=3,
        distribution="pearson",
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2001,
        periodicity=compute.Periodicity.monthly,
    )

    expected = indices.spi(
        values,
        3,
        indices.Distribution.pearson,
        2000,
        2000,
        2001,
        compute.Periodicity.monthly,
    )

    assert isinstance(spi_da, xr.DataArray)
    assert spi_da.dims == ("time",)
    np.testing.assert_allclose(spi_da.values, expected, equal_nan=True)


def test_spi_accessor_2d_raises_valueerror():
    """Test that 2-D DataArray raises ValueError."""
    values = np.random.rand(12, 5)
    da = xr.DataArray(values, dims=("time", "lat"), name="precip")

    with pytest.raises(ValueError, match="1-D DataArray inputs only"):
        da.indices.spi(
            scale=3,
            distribution="gamma",
            data_start_year=2000,
            calibration_year_initial=2000,
            calibration_year_final=2000,
            periodicity=compute.Periodicity.monthly,
        )


@pytest.mark.parametrize("distribution_name", ["gamma", "pearson"])
def test_spi_xarray_with_dask_single_time_chunk(distribution_name):
    """Ensure spi_xarray works correctly with Dask-chunked DataArrays.

    This exercises the xarray.apply_ufunc/Dask path used by the public API and CLI.
    Time must be a single chunk for correctness.
    """
    values = np.array(
        [
            10.0,
            20.0,
            0.0,
            5.0,
            15.0,
            0.0,
            8.0,
            12.0,
            3.0,
            7.0,
            9.0,
            4.0,
            11.0,
            13.0,
            2.0,
            6.0,
            14.0,
            2.0,
            1.0,
            16.0,
            18.0,
            0.0,
            5.0,
            9.0,
        ],
        dtype=float,
    )
    time = np.arange(values.size)
    da = xr.DataArray(values, dims=("time",), coords={"time": time}, name="precip")

    # chunk with time=-1 keeps time as a single chunk (required for correctness)
    da_chunked = da.chunk({"time": -1})

    distribution = getattr(indices.Distribution, distribution_name)
    expected = indices.spi(
        values,
        3,
        distribution,
        2000,
        2000,
        2001,
        compute.Periodicity.monthly,
    )

    spi_da = indices.spi_xarray(
        da_chunked,
        scale=3,
        distribution=distribution,
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2001,
        periodicity=compute.Periodicity.monthly,
    )

    # spi_xarray may return a lazy Dask-backed DataArray; compute before comparison
    spi_da_computed = spi_da.compute()

    assert isinstance(spi_da, xr.DataArray)
    assert spi_da_computed.dims == ("time",)
    assert np.array_equal(spi_da_computed["time"].values, time)
    np.testing.assert_allclose(spi_da_computed.values, expected, equal_nan=True)


def test_spi_xarray_multiple_time_chunks_raises_valueerror():
    """Verify that spi_xarray fails with clear error when time is split into multiple chunks."""
    values = np.array(
        [
            10.0,
            20.0,
            0.0,
            5.0,
            15.0,
            0.0,
            8.0,
            12.0,
            3.0,
            7.0,
            9.0,
            4.0,
            11.0,
            13.0,
            2.0,
            6.0,
            14.0,
            2.0,
            1.0,
            16.0,
            18.0,
            0.0,
            5.0,
            9.0,
        ],
        dtype=float,
    )
    time = np.arange(values.size)
    da = xr.DataArray(values, dims=("time",), coords={"time": time}, name="precip")

    # chunk with time=12 splits time into multiple chunks (not supported)
    da_chunked = da.chunk({"time": 12})

    with pytest.raises(ValueError, match="multiple chunks.*core dimension"):
        indices.spi_xarray(
            da_chunked,
            scale=3,
            distribution=indices.Distribution.gamma,
            data_start_year=2000,
            calibration_year_initial=2000,
            calibration_year_final=2001,
            periodicity=compute.Periodicity.monthly,
        )
