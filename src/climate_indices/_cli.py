"""Shared helpers for the climate_indices command-line interfaces."""

import argparse
import logging

import xarray as xr

from climate_indices import compute, utils

# Retrieve logger and set desired logging level
_logger = utils.get_logger(__name__, logging.INFO)

_DEFAULT_SCALES_HELP = "Timestep scales over which the PNP, SPI, and SPEI values are to be computed"


def _prepare_file(netcdf_file: str, var_name: str) -> str:
    """
    Validate the dimensions of a NetCDF variable prior to processing.

    The file is returned unchanged; no dimensions are reordered.

    Args:
        netcdf_file: path of the NetCDF file to be validated.
        var_name: name of the variable whose dimensions are validated.

    Returns:
        The name of the NetCDF file containing correct dimensions.

    Raises:
        ValueError: if the variable's dimensions are not one of the supported forms.
    """

    # make sure we have the expected dimensions for the data type
    ds = xr.open_dataset(netcdf_file)
    dimensions = ds[var_name].dims

    # Validate dimensions based on data type
    if "division" in dimensions:
        # Climate divisions data
        if len(dimensions) == 1:
            expected_dims = {"division"}
        elif len(dimensions) == 2:
            expected_dims = {"division", "time"}
        else:
            message = f"Unsupported dimensions for climate division variable '{var_name}': {dimensions}"
            _logger.error(message)
            raise ValueError(message)
    else:
        # Gridded or timeseries data
        if len(dimensions) == 1:
            expected_dims = {"time"}
        elif len(dimensions) == 2:
            expected_dims = {"lat", "lon"}
        elif len(dimensions) == 3:
            expected_dims = {"lat", "lon", "time"}
        else:
            message = f"Unsupported dimensions for variable '{var_name}': {dimensions}"
            _logger.error(message)
            raise ValueError(message)

    # Validate that the actual dimensions match expected dimensions
    actual_dims = set(dimensions)
    if actual_dims != expected_dims:
        message = f"Invalid dimensions for variable '{var_name}': got {actual_dims}, expected {expected_dims}"
        _logger.error(message)
        raise ValueError(message)

    return netcdf_file


def _add_common_spi_arguments(
    parser: argparse.ArgumentParser,
    scales_help: str = _DEFAULT_SCALES_HELP,
) -> None:
    """
    Register the command line arguments shared by the climate_indices and SPI CLIs.

    Args:
        parser: parser to which the arguments are added.
        scales_help: help text for the ``--scales`` argument.
    """

    parser.add_argument(
        "--periodicity",
        help="Process input as either monthly or daily values",
        choices=[compute.Periodicity.monthly, compute.Periodicity.daily],
        type=compute.Periodicity.from_string,
        required=True,
    )
    parser.add_argument(
        "--scales",
        help=scales_help,
        type=int,
        nargs="*",
    )
    parser.add_argument(
        "--calibration_start_year",
        help="Initial year of the calibration period",
        type=int,
    )
    parser.add_argument("--calibration_end_year", help="Final year of calibration period", type=int)
    parser.add_argument(
        "--netcdf_precip",
        help="Precipitation NetCDF file to be used as input for indices computations",
    )
    parser.add_argument(
        "--var_name_precip",
        help="Precipitation variable name used in the precipitation NetCDF file",
    )
    parser.add_argument(
        "--output_file_base",
        help="Base output file path and name for the resulting output files",
        required=True,
    )
    parser.add_argument(
        "--multiprocessing",
        help="options for multiprocessing -- single core, all cores but one, or all cores",
        choices=["single", "all_but_one", "all"],
        required=False,
        default="all_but_one",
    )
