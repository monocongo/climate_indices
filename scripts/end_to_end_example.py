#!/usr/bin/env python3
"""Run with: uv run --with h5py --with zarr scripts/end_to_end_example.py."""

import shutil
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from climate_indices import compute, indices
from climate_indices.exceptions import CoordinateValidationError, InvalidArgumentError


def _validate_monthly_time(time_values: np.ndarray) -> pd.DatetimeIndex:
    """Return complete month-start or month-end timestamps."""
    try:
        time = pd.DatetimeIndex(time_values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CoordinateValidationError(
            "Time coordinate must contain supported datetime values.",
            coordinate_name="time",
            reason="not datetime-like",
        ) from exc
    if time.empty:
        raise CoordinateValidationError(
            "Time coordinate must not be empty.", coordinate_name="time", reason="empty coordinate"
        )
    if time.is_month_start.all():
        expected_time = pd.date_range(time[0], periods=time.size, freq="MS")
    elif time.is_month_end.all():
        expected_time = pd.date_range(time[0], periods=time.size, freq=pd.offsets.MonthEnd())
    else:
        expected_time = pd.DatetimeIndex([])
    if not time.equals(expected_time):
        raise CoordinateValidationError(
            "Time coordinate must be a complete, chronological sequence of monthly "
            "month-start or month-end timestamps.",
            coordinate_name="time",
            reason="non-monotonic, irregular, or gapped monthly timestamps",
        )
    return time


def clean_and_prepare_inputs(precip_path: Path, pet_path: Path, zarr_prepared_path: Path) -> None:
    """Prepare aligned monthly totals in mm for blockwise SPI/SPEI computation.

    Args:
        precip_path: NetCDF file containing ``pr(time, lat, lon)`` in mm.
        pet_path: NetCDF file containing ``pet(time, lat, lon)`` in mm.
        zarr_prepared_path: Prepared Zarr store to create or replace.
    """
    with (
        xr.open_dataset(precip_path, chunks={"time": 1}) as ds_precip,
        xr.open_dataset(pet_path, chunks={"time": 1}) as ds_pet,
    ):
        pr, pet = xr.align(ds_precip["pr"], ds_pet["pet"], join="exact")
        if pr.attrs.get("units") != "mm" or pet.attrs.get("units") != "mm":
            raise InvalidArgumentError(
                "Inputs must contain monthly totals with units='mm'.",
                argument_name="units",
                argument_value=f"pr={pr.attrs.get('units')!r}, pet={pet.attrs.get('units')!r}",
                valid_values="mm",
            )
        _validate_monthly_time(pr["time"].values)
        ds_clean = xr.Dataset({"precip": pr, "pet": pet, "wb": pr - pet}).transpose("time", "lat", "lon")
        ds_clean["wb"].attrs = {"long_name": "Precipitation minus PET, monthly total", "units": "mm"}
        # Full time series per block; 10x10 gives multiple tasks on the small example grid.
        ds_clean = ds_clean.drop_encoding().chunk({"time": -1, "lat": 10, "lon": 10})
        print(f"Saving prepared inputs: {zarr_prepared_path}")
        ds_clean.to_zarr(zarr_prepared_path, mode="w", zarr_format=2, consolidated=True)


# 2. Element-wise Wrapper Functions for climate_indices
def _spi_wrapper(
    precip_chunk,
    scale,
    distribution,
    data_start_year,
    calibration_start_year,
    calibration_end_year,
    periodicity: compute.Periodicity,
) -> np.ndarray:
    """
    Guaranteed to receive a numpy array with shape (time, lat, lon) where time is continuous,
    but lat and lon represent a subset block dictated by Dask.
    """
    # Allocate empty output matching input shape
    output = np.full(precip_chunk.shape, np.nan, dtype=np.float32)

    # Iterate over the spatial block slice passed to this worker
    for i in range(precip_chunk.shape[1]):
        for j in range(precip_chunk.shape[2]):
            time_series = precip_chunk[:, i, j]

            # Skip computation if entirely filled with NaNs (e.g., oceans)
            if np.isnan(time_series).all():
                continue

            output[:, i, j] = indices.spi(
                values=time_series,
                scale=scale,
                distribution=distribution,
                data_start_year=data_start_year,
                calibration_year_initial=calibration_start_year,
                calibration_year_final=calibration_end_year,
                periodicity=periodicity,
            )
    return output


def _spei_wrapper(
    precip_chunk,
    pet_chunk,
    scale,
    distribution,
    data_start_year,
    calibration_start_year,
    calibration_end_year,
    periodicity: compute.Periodicity,
) -> np.ndarray:
    """Wrapper for block-level SPEI calculation."""
    output = np.full(precip_chunk.shape, np.nan, dtype=np.float32)

    for i in range(precip_chunk.shape[1]):
        for j in range(precip_chunk.shape[2]):
            precip_time_series = precip_chunk[:, i, j]
            pet_time_series = pet_chunk[:, i, j]
            if np.isnan(precip_time_series).all():
                continue

            output[:, i, j] = indices.spei(
                precips_mm=precip_time_series,
                pet_mm=pet_time_series,
                scale=scale,
                distribution=distribution,
                data_start_year=data_start_year,
                calibration_year_initial=calibration_start_year,
                calibration_year_final=calibration_end_year,
                periodicity=periodicity,
            )
    return output


# 3. Concurrent Scaling Block Execution
def compute_indices_parallel(zarr_prepared_path: Path, output_zarr_path: Path, config: dict) -> None:
    """Compute SPI/SPEI in spatial blocks, retaining the complete time series.

    Args:
        zarr_prepared_path: Store produced by ``clean_and_prepare_inputs``.
        output_zarr_path: Output Zarr store to create or replace.
        config: Timescale, distributions, Periodicity, and data/calibration years.
    """
    with xr.open_zarr(zarr_prepared_path, consolidated=True) as ds:
        ds = ds.transpose("time", "lat", "lon").chunk({"time": -1})

        time = _validate_monthly_time(ds["time"].values)
        if time[0].month != 1 or time[-1].month != 12:
            raise CoordinateValidationError(
                "Prepared dataset must cover complete calendar years.",
                coordinate_name="time",
                reason="incomplete first or final year",
            )
        data_start_year = config["data_start_year"]
        if time[0].year != data_start_year:
            raise InvalidArgumentError(
                "config['data_start_year'] does not match the prepared dataset's first monthly timestamp.",
                argument_name="data_start_year",
                argument_value=str(data_start_year),
                valid_values=str(time[0].year),
            )
        data_end_year = time[-1].year
        cal_start_year, cal_end_year = config["cal_start_year"], config["cal_end_year"]
        if not (data_start_year <= cal_start_year <= cal_end_year <= data_end_year):
            raise InvalidArgumentError(
                "Calibration Period must fall within the prepared dataset's covered years.",
                argument_name="cal_start_year/cal_end_year",
                argument_value=f"{cal_start_year}-{cal_end_year}",
                valid_values=f"{data_start_year}-{data_end_year}",
            )

        kwargs = {
            "scale": config["scale"],
            "data_start_year": config["data_start_year"],
            "calibration_start_year": config["cal_start_year"],
            "calibration_end_year": config["cal_end_year"],
            "periodicity": config["periodicity"],
            "dtype": np.float32,
            "meta": np.empty((0, 0, 0), dtype=np.float32),
        }
        spi_array = da.map_blocks(_spi_wrapper, ds["precip"].data, distribution=config["distribution_spi"], **kwargs)
        spei_array = da.map_blocks(
            _spei_wrapper, ds["precip"].data, ds["pet"].data, distribution=config["distribution_spei"], **kwargs
        )
        ds_output = xr.Dataset(
            {
                f"spi_{config['scale']}": (("time", "lat", "lon"), spi_array),
                f"spei_{config['scale']}": (("time", "lat", "lon"), spei_array),
            },
            coords=ds.coords,
        )
        for name, long_name in (
            ("spi", "Standardized Precipitation Index"),
            ("spei", "Standardized Precipitation Evapotranspiration Index"),
        ):
            ds_output[f"{name}_{config['scale']}"].attrs = {
                "long_name": long_name,
                "scale": config["scale"],
                "units": "dimensionless",
            }
        print(f"Computing SPI/SPEI: {output_zarr_path}")
        # Write beside the target and swap on success so a failed run leaves a
        # previously completed store intact.
        tmp_path = output_zarr_path.with_name(output_zarr_path.name + ".tmp")
        shutil.rmtree(tmp_path, ignore_errors=True)
        ds_output.to_zarr(tmp_path, mode="w", zarr_format=2, consolidated=True)
        shutil.rmtree(output_zarr_path, ignore_errors=True)
        tmp_path.rename(output_zarr_path)


if __name__ == "__main__":
    from dask.distributed import Client

    pipeline_config = {
        "scale": 3,
        "distribution_spi": indices.Distribution.gamma,
        "distribution_spei": indices.Distribution.pearson,
        "periodicity": compute.Periodicity.monthly,
        "data_start_year": 1980,
        "cal_start_year": 1981,
        "cal_end_year": 2010,
    }
    data_dir = Path(__file__).resolve().parents[1] / "data" / "e2e"
    precip_nc = data_dir / "raw_precipitation.nc"
    pet_nc = data_dir / "raw_pet.nc"
    prepared_zarr = data_dir / "cache_prepared_input.zarr"
    final_output_zarr = data_dir / "climate_indices_output.zarr"

    with Client(n_workers=4, threads_per_worker=2, memory_limit="4GB", dashboard_address=None):
        clean_and_prepare_inputs(precip_nc, pet_nc, prepared_zarr)
        compute_indices_parallel(prepared_zarr, final_output_zarr, pipeline_config)
