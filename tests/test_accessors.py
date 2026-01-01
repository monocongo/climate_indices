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
    assert spi_da.attrs["standard_name"] == "spi"
    assert "Standardized Precipitation Index" in spi_da.attrs["long_name"]
    assert spi_da.attrs["units"] == "1"
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


def test_spi_accessor_2d_works():
    """Test that 2-D DataArray (time, lat) works correctly."""
    np.random.seed(42)
    # 24 months (2 years), 3 lat points
    values_2d = np.abs(np.random.randn(24, 3) * 10 + 20)
    time = np.arange(24)
    lat = np.array([30.0, 35.0, 40.0])

    da = xr.DataArray(
        values_2d,
        dims=("time", "lat"),
        coords={"time": time, "lat": lat},
        name="precip",
    )

    spi_da = da.indices.spi(
        scale=3,
        distribution="gamma",
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2001,
        periodicity=compute.Periodicity.monthly,
    )

    # verify output shape matches input
    assert spi_da.shape == da.shape
    assert spi_da.dims == ("time", "lat")

    # verify coordinates are preserved
    np.testing.assert_array_equal(spi_da["time"].values, time)
    np.testing.assert_array_equal(spi_da["lat"].values, lat)

    # verify CF-compliant attributes
    assert spi_da.attrs["standard_name"] == "spi"
    assert "Standardized Precipitation Index" in spi_da.attrs["long_name"]
    assert spi_da.attrs["units"] == "1"

    # verify values match per-pixel computation
    for i, _lat_val in enumerate(lat):
        expected = indices.spi(
            values_2d[:, i],
            3,
            indices.Distribution.gamma,
            2000,
            2000,
            2001,
            compute.Periodicity.monthly,
        )
        np.testing.assert_allclose(spi_da.isel(lat=i).values, expected, equal_nan=True, rtol=1e-10)


def test_spi_accessor_3d_works():
    """Test that 3-D DataArray (time, lat, lon) works correctly."""
    np.random.seed(123)
    # 24 months, 2 lat, 2 lon
    values_3d = np.abs(np.random.randn(24, 2, 2) * 10 + 20)
    time = np.arange(24)
    lat = np.array([30.0, 35.0])
    lon = np.array([-100.0, -95.0])

    da = xr.DataArray(
        values_3d,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="precip",
    )

    spi_da = da.indices.spi(
        scale=3,
        distribution="gamma",
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2001,
        periodicity=compute.Periodicity.monthly,
    )

    # verify output shape matches input
    assert spi_da.shape == da.shape
    assert spi_da.dims == ("time", "lat", "lon")

    # verify coordinates are preserved
    np.testing.assert_array_equal(spi_da["time"].values, time)
    np.testing.assert_array_equal(spi_da["lat"].values, lat)
    np.testing.assert_array_equal(spi_da["lon"].values, lon)

    # verify values match per-pixel computation
    for i in range(2):
        for j in range(2):
            expected = indices.spi(
                values_3d[:, i, j],
                3,
                indices.Distribution.gamma,
                2000,
                2000,
                2001,
                compute.Periodicity.monthly,
            )
            np.testing.assert_allclose(spi_da.isel(lat=i, lon=j).values, expected, equal_nan=True, rtol=1e-10)


def test_spi_accessor_3d_with_dask():
    """Test that 3-D DataArray with Dask chunking works correctly."""
    np.random.seed(456)
    # 24 months, 4 lat, 4 lon
    values_3d = np.abs(np.random.randn(24, 4, 4) * 10 + 20)
    time = np.arange(24)
    lat = np.linspace(30.0, 45.0, 4)
    lon = np.linspace(-100.0, -85.0, 4)

    da = xr.DataArray(
        values_3d,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="precip",
    )

    # chunk with time=-1 (required) and spatial chunks of 2
    da_chunked = da.chunk({"time": -1, "lat": 2, "lon": 2})

    spi_da = da_chunked.indices.spi(
        scale=3,
        distribution="gamma",
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2001,
        periodicity=compute.Periodicity.monthly,
    )

    # compute the lazy result
    spi_computed = spi_da.compute()

    # verify output shape matches input
    assert spi_computed.shape == da.shape
    assert spi_computed.dims == ("time", "lat", "lon")

    # verify coordinates are preserved
    np.testing.assert_array_equal(spi_computed["time"].values, time)
    np.testing.assert_array_equal(spi_computed["lat"].values, lat)
    np.testing.assert_array_equal(spi_computed["lon"].values, lon)

    # spot check a few pixels
    for i, j in [(0, 0), (1, 2), (3, 3)]:
        expected = indices.spi(
            values_3d[:, i, j],
            3,
            indices.Distribution.gamma,
            2000,
            2000,
            2001,
            compute.Periodicity.monthly,
        )
        np.testing.assert_allclose(spi_computed.isel(lat=i, lon=j).values, expected, equal_nan=True, rtol=1e-10)


def test_spi_accessor_multidim_multiple_time_chunks_raises():
    """Test that multi-dimensional arrays with multiple time chunks raise ValueError."""
    np.random.seed(789)
    values_2d = np.abs(np.random.randn(24, 3) * 10 + 20)
    da = xr.DataArray(
        values_2d,
        dims=("time", "lat"),
        coords={"time": np.arange(24), "lat": [30.0, 35.0, 40.0]},
        name="precip",
    )

    # chunk time into multiple pieces (not allowed)
    da_chunked = da.chunk({"time": 12, "lat": 1})

    with pytest.raises(ValueError, match="multiple chunks"):
        da_chunked.indices.spi(
            scale=3,
            distribution="gamma",
            data_start_year=2000,
            calibration_year_initial=2000,
            calibration_year_final=2001,
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
