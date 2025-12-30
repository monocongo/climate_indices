import numpy as np
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
