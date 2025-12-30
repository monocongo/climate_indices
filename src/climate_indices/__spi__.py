"""Command-line interface for performing SPI calculations"""

import argparse
import logging
import os
from datetime import datetime
from enum import Enum

import numpy as np
import xarray as xr

from climate_indices import compute, indices, utils

# variable names for the distribution fitting parameters
_FITTING_VARIABLES = ("alpha", "beta", "skew", "loc", "scale", "prob_zero")

# location of the package on GitHub (for documentation within NetCDFs)
_GITHUB_URL = "https://github.com/monocongo/climate_indices"

# Retrieve logger and set desired logging level
_logger = utils.get_logger(__name__, logging.INFO)


class InputType(Enum):
    """
    Enumeration type for differentiating between gridded, timeseries, and US
    climate division datasets.
    """

    grid = 1
    divisions = 2
    timeseries = 3


def _validate_args(args):
    """
    Validate the processing settings to confirm that proper argument
    combinations have been provided.

    :param args: an arguments object of the type returned by
        argparse.ArgumentParser.parse_args()
    :raise ValueError: if one or more of the command line arguments is invalid
    """

    def validate_dimensions(
        ds: xr.Dataset,
        var_name: str,
        variable_plain_name: str,
    ) -> InputType:
        """
        Function to verify that a variable's dimensions are in one of the expected
        dimension orders and if so then it will return the corresponding InputType.

        :param ds: xarray Dataset
        :param var_name: variable name
        :param variable_plain_name: plain English name/description of the variable
        :return: the InputType matching to the variable's dimensions
        :raises ValueError if the dimensions of the variable don't match with
            one of the expected dimension orders
        """

        # verify that the variable's dimensions are in the expected order
        dims = ds[var_name].dims
        if dims in expected_dimensions_grid:
            _input_type = InputType.grid
        elif dims in expected_dimensions_divisions:
            _input_type = InputType.divisions
        elif dims in expected_dimensions_timeseries:
            _input_type = InputType.timeseries
        else:
            mesg = (
                f"Invalid dimensions of the {variable_plain_name} "
                + f"variable: {dims}\nValid dimension names and "
                + f"order: {expected_dimensions_grid} or "
                + f"{expected_dimensions_divisions} or "
                + f"{expected_dimensions_timeseries}"
            )
            _logger.error(mesg)
            raise ValueError(mesg)

        return _input_type

    # the dimensions we expect to find for each data variable
    # (precipitation, temperature, and/or PET)
    expected_dimensions_divisions = [("time", "division"), ("division", "time")]
    expected_dimensions_grid = [("lat", "lon", "time"), ("time", "lat", "lon")]
    expected_dimensions_timeseries = [("time",)]

    # for fitting parameters we can either compute and save or load from file, but not both
    if args.save_params:
        if args.load_params:
            msg = "Both of the mutually exclusive fitting parameter file options were specified (both load and save)"
            _logger.error(msg)
            raise ValueError(msg)

        elif os.path.exists(args.save_params) and not args.overwrite:
            msg = "The distribution fitting parameters file to save is present and overwrite was not specified"
            _logger.error(msg)
            raise ValueError(msg)

    if args.load_params:
        # make sure the specified fitting parameters file exists
        if not os.path.exists(args.load_params):
            msg = f"The specified fitting parameters file {args.load_params} does not exist"
            _logger.error(msg)
            raise ValueError(msg)

        # open the fitting parameters file and make sure it looks reasonable
        with xr.open_dataset(args.load_params) as dataset_fittings:
            # confirm that all the fitting parameter variables are present
            missing_variables = []
            for var in _FITTING_VARIABLES:
                for scale in args.scales:
                    fitting_var_name_suffix = f"{scale}_{args.periodicity.unit()}"
                    fitting_var = "_".join((var, fitting_var_name_suffix))
                    if fitting_var not in dataset_fittings.variables:
                        missing_variables.append(fitting_var)
            if len(missing_variables) > 0:
                msg = (
                    "The following fitting parameter variables are expected "
                    "but not present in the specified fitting parameters "
                    f"dataset ({args.load_params}): {missing_variables}"
                )
                _logger.error(msg)
                raise ValueError(msg)

            # TODO compare against the lats_precip, lons_precip, etc.

    # make sure a precipitation file was specified
    if args.netcdf_precip is None:
        msg = "Missing the required precipitation file argument"
        _logger.error(msg)
        raise ValueError(msg)

    # make sure a precipitation variable name was specified
    if args.var_name_precip is None:
        msg = "Missing the precipitation variable name argument"
        _logger.error(msg)
        raise ValueError(msg)

    # validate the precipitation file itself
    with xr.open_dataset(args.netcdf_precip) as dataset_precip:
        # make sure we have a valid precipitation variable name
        if args.var_name_precip not in dataset_precip.variables:
            msg = (
                f"Invalid precipitation variable name: '{args.var_name_precip}'"
                + f"does not exist in precipitation file '{args.netcdf_precip}'"
            )
            _logger.error(msg)
            raise ValueError(msg)

        # verify that the precipitation variable's dimensions are in
        # the expected order, and if so then determine the input type
        input_type = validate_dimensions(
            dataset_precip,
            args.var_name_precip,
            "precipitation",
        )

    if args.scales is None:
        msg = "Missing one or more time scales (missing --scales argument)"
        _logger.error(msg)
        raise ValueError(msg)

    if any(n < 0 for n in args.scales):
        msg = "One or more negative scale specified within --scales argument"
        _logger.error(msg)
        raise ValueError(msg)

    return input_type


def _build_arguments(keyword_args):
    """
    Builds a dictionary of function arguments appropriate to the index to be computed.

    :param dict keyword_args:
    :return: dictionary of arguments keyed with names expected by the corresponding
        index computation function
    """

    function_arguments = {
        "data_start_year": keyword_args["data_start_year"],
        "scale": keyword_args["scale"],
        "distribution": keyword_args["distribution"],
        "calibration_year_initial": keyword_args["calibration_start_year"],
        "calibration_year_final": keyword_args["calibration_end_year"],
        "periodicity": keyword_args["periodicity"],
    }

    return function_arguments


def _get_variable_attributes(
    distribution: indices.Distribution,
    scale: int,
    periodicity: compute.Periodicity,
):
    attrs = {
        "long_name": "Standardized Precipitation Index ("
        f"{distribution.value.capitalize()}), "
        f"{scale}-{periodicity.unit()}",
        "valid_min": -3.09,
        "valid_max": 3.09,
    }

    return attrs


def build_dataset_fitting_grid(
    ds_example: xr.Dataset,
    periodicity: compute.Periodicity,
) -> xr.Dataset:
    """
    Builds a new Dataset object based on an example Dataset. Essentially copies
    the lat and lon values and sets these along with period-specific times
    as coordinates.

    :param ds_example:
    :param periodicity:
    :return:
    """

    if periodicity == compute.Periodicity.monthly:
        period_times = 12
    elif periodicity == compute.Periodicity.daily:
        period_times = 366
    else:
        raise ValueError(f"Unsupported periodicity: {periodicity}")

    usage_url = "https://climate-indices.readthedocs.io/en/latest/#spi-monthly"
    global_attrs = {
        "description": f"Distribution fitting parameters for various {periodicity.unit()} "
        f"scales computed from {periodicity} precipitation input "
        "by the climate_indices package available from "
        f"{_GITHUB_URL}. The variables contained herein are meant "
        "to be used as inputs for computing SPI datasets using "
        f"the climate_indices package. See {usage_url} for "
        "example usage.",
        "geospatial_lat_min": float(np.amin(ds_example.lat)),
        "geospatial_lat_max": float(np.amax(ds_example.lat)),
        "geospatial_lat_units": ds_example.lat.units,
        "geospatial_lon_min": float(np.amin(ds_example.lon)),
        "geospatial_lon_max": float(np.amax(ds_example.lon)),
        "geospatial_lon_units": ds_example.lon.units,
    }
    times = np.array(range(period_times))
    time_coord = periodicity.unit()
    coords = {
        "lat": ds_example.lat,
        "lon": ds_example.lon,
        time_coord: xr.DataArray(data=times, coords=[times], dims=time_coord),
    }
    ds_fitting_params = xr.Dataset(
        attrs=global_attrs,
        coords=coords,
    )

    return ds_fitting_params


def build_dataset_fitting_divisions(
    ds_example: xr.Dataset,
    periodicity: compute.Periodicity,
) -> xr.Dataset:
    """
    Builds a new Dataset object based on an example Dataset. Essentially copies
    the lat and lon values and sets these along with period-specific times
    as coordinates.

    :param ds_example:
    :param periodicity:
    :return:
    """

    if periodicity == compute.Periodicity.monthly:
        period_times = 12
    elif periodicity == compute.Periodicity.daily:
        period_times = 366
    else:
        raise ValueError(f"Unsupported periodicity: {periodicity}")

    usage_url = "https://climate-indices.readthedocs.io/en/latest/#spi-monthly"
    global_attrs = {
        "description": f"Distribution fitting parameters for various {periodicity.unit()} "
        f"scales computed from {periodicity} precipitation input "
        "by the climate_indices package available from "
        f"{_GITHUB_URL}. The variables contained herein are meant "
        "to be used as inputs for computing SPI datasets using "
        f"the climate_indices package. See {usage_url} for "
        "example usage.",
    }
    times = np.array(range(period_times))
    time_coord = periodicity.unit()
    coords = {
        "division": ds_example.division,
        time_coord: xr.DataArray(data=times, coords=[times], dims=time_coord),
    }
    ds_fitting_params = xr.Dataset(
        coords=coords,
        attrs=global_attrs,
    )

    return ds_fitting_params


def _compute_write_index(keyword_arguments):
    """
    Computes a climate index and writes the result into a corresponding NetCDF.

    :param keyword_arguments:
    :return:
    """

    _logger.info(f"Computing {keyword_arguments['periodicity']} SPI")

    # open the NetCDF files as an xarray DataSet object
    if "netcdf_precip" not in keyword_arguments:
        raise ValueError("Missing the 'netcdf_precip' keyword argument")
    if "input_type" not in keyword_arguments:
        raise ValueError("Missing the 'input_type' keyword argument")

    input_type = keyword_arguments["input_type"]
    ds_precip = xr.open_dataset(keyword_arguments["netcdf_precip"], chunks={"time": -1})

    # trim out all data variables from the dataset except the ones we'll need
    input_var_names = []
    if "var_name_precip" in keyword_arguments:
        input_var_names.append(keyword_arguments["var_name_precip"])
    # only keep the latitude variable if we're dealing with divisions
    if input_type == InputType.divisions:
        input_var_names.append("lat")
    for var in ds_precip.data_vars:
        if var not in input_var_names:
            ds_precip = ds_precip.drop_vars(var)

    if "var_name_precip" in keyword_arguments:
        precip_var_name = keyword_arguments["var_name_precip"]
    else:
        raise ValueError("No precipitation variable name was specified.")

    precip_da = ds_precip[precip_var_name]
    precip_unit = precip_da.units.lower()
    if precip_unit not in ("mm", "millimeters", "millimeter", "mm/dy"):
        if precip_unit in ("inches", "inch"):
            precip_da = precip_da * 25.4
        else:
            raise ValueError(f"Unsupported precipitation units: {precip_unit}")
    ds_precip[precip_var_name] = precip_da

    if input_type == InputType.grid:
        precip_da = precip_da.transpose("lat", "lon", "time")
    elif input_type == InputType.divisions:
        if "time" in precip_da.dims:
            precip_da = precip_da.transpose("division", "time")
    elif input_type == InputType.timeseries:
        precip_da = precip_da.transpose("time")
    else:
        raise ValueError(f"Invalid 'input_type' keyword argument: {input_type}")
    ds_precip[precip_var_name] = precip_da

    data_start_year = int(ds_precip["time"][0].dt.year)
    keyword_arguments["data_start_year"] = data_start_year

    periodicity = keyword_arguments["periodicity"]
    time_dim = "time"
    original_time = ds_precip[time_dim]
    original_time_len = len(original_time)

    if periodicity == compute.Periodicity.daily:
        initial_year = int(original_time[0].dt.year)
        final_year = int(original_time[-1].dt.year)
        total_years = final_year - initial_year + 1
        period_length = 366

        def _to_366(values_1d: np.ndarray) -> np.ndarray:
            return utils.transform_to_366day(values_1d, initial_year, total_years)

        precip_da = xr.apply_ufunc(
            _to_366,
            precip_da,
            input_core_dims=[[time_dim]],
            output_core_dims=[[time_dim]],
            output_sizes={time_dim: total_years * period_length},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        precip_da = precip_da.assign_coords({time_dim: np.arange(total_years * period_length)})

    elif periodicity == compute.Periodicity.monthly:
        period_length = 12
    else:
        raise ValueError(f"Unsupported periodicity argument value: {periodicity}")

    period_dim = periodicity.unit()
    ds_fitting = None
    if keyword_arguments["load_params"] is not None:
        ds_fitting = xr.open_dataset(keyword_arguments["load_params"])
    elif keyword_arguments["save_params"] is not None:
        if input_type == InputType.divisions:
            ds_fitting = build_dataset_fitting_divisions(ds_precip, periodicity)
        else:
            ds_fitting = build_dataset_fitting_grid(ds_precip, periodicity)

    fitting_var_attrs = {
        "alpha": {
            "description": "shape parameter of the gamma distribution (also "
            "referred to as the concentration) computed from "
            "the {scale}-month scaled precipitation values",
        },
        "beta": {
            "description": "1 / scale of the distribution (also referred to "
            "as the rate) computed from the {scale}-month "
            "scaled precipitation values",
        },
        "prob_zero": {"description": "probability of zero values within calibration period"},
        "loc": {
            "description": "loc parameter for Pearson Type III",
        },
        "scale": {
            "description": "scale parameter for Pearson Type III",
        },
        "skew": {
            "description": "skew parameter for Pearson Type III",
        },
    }

    for scale in keyword_arguments["scales"]:
        suffix = f"{scale}_{period_dim}"
        fitting_var_names = {var: f"{var}_{suffix}" for var in _FITTING_VARIABLES}

        fitting_params_gamma = None
        fitting_params_pearson = None

        if ds_fitting is not None:
            if keyword_arguments["load_params"] is not None:
                fitting_params_gamma = {
                    "alpha": ds_fitting[fitting_var_names["alpha"]],
                    "beta": ds_fitting[fitting_var_names["beta"]],
                }
                fitting_params_pearson = {
                    "prob_zero": ds_fitting[fitting_var_names["prob_zero"]],
                    "loc": ds_fitting[fitting_var_names["loc"]],
                    "scale": ds_fitting[fitting_var_names["scale"]],
                    "skew": ds_fitting[fitting_var_names["skew"]],
                }
            else:
                scaled_values = xr.apply_ufunc(
                    compute.sum_to_scale,
                    precip_da,
                    kwargs={"scale": scale},
                    input_core_dims=[[time_dim]],
                    output_core_dims=[[time_dim]],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[float],
                )

                def _gamma_params(values_1d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                    return compute.gamma_parameters(
                        values=values_1d,
                        data_start_year=data_start_year,
                        calibration_start_year=keyword_arguments["calibration_start_year"],
                        calibration_end_year=keyword_arguments["calibration_end_year"],
                        periodicity=periodicity,
                    )

                alphas, betas = xr.apply_ufunc(
                    _gamma_params,
                    scaled_values,
                    input_core_dims=[[time_dim]],
                    output_core_dims=[[period_dim], [period_dim]],
                    output_sizes={period_dim: period_length},
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[float, float],
                )

                def _pearson_params(values_1d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                    return compute.pearson_parameters(
                        values=values_1d,
                        data_start_year=data_start_year,
                        calibration_start_year=keyword_arguments["calibration_start_year"],
                        calibration_end_year=keyword_arguments["calibration_end_year"],
                        periodicity=periodicity,
                    )

                prob_zero, locs, scales, skews = xr.apply_ufunc(
                    _pearson_params,
                    scaled_values,
                    input_core_dims=[[time_dim]],
                    output_core_dims=[[period_dim], [period_dim], [period_dim], [period_dim]],
                    output_sizes={period_dim: period_length},
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[float, float, float, float],
                )

                if period_dim in ds_fitting.coords:
                    coord = ds_fitting[period_dim]
                    alphas = alphas.assign_coords({period_dim: coord})
                    betas = betas.assign_coords({period_dim: coord})
                    prob_zero = prob_zero.assign_coords({period_dim: coord})
                    locs = locs.assign_coords({period_dim: coord})
                    scales = scales.assign_coords({period_dim: coord})
                    skews = skews.assign_coords({period_dim: coord})

                alphas.name = fitting_var_names["alpha"]
                alphas.attrs = fitting_var_attrs["alpha"].copy()
                alphas.attrs["description"] = alphas.attrs["description"].format(scale=scale)
                ds_fitting[alphas.name] = alphas

                betas.name = fitting_var_names["beta"]
                betas.attrs = fitting_var_attrs["beta"].copy()
                betas.attrs["description"] = betas.attrs["description"].format(scale=scale)
                ds_fitting[betas.name] = betas

                prob_zero.name = fitting_var_names["prob_zero"]
                prob_zero.attrs = fitting_var_attrs["prob_zero"]
                ds_fitting[prob_zero.name] = prob_zero

                locs.name = fitting_var_names["loc"]
                locs.attrs = fitting_var_attrs["loc"]
                ds_fitting[locs.name] = locs

                scales.name = fitting_var_names["scale"]
                scales.attrs = fitting_var_attrs["scale"]
                ds_fitting[scales.name] = scales

                skews.name = fitting_var_names["skew"]
                skews.attrs = fitting_var_attrs["skew"]
                ds_fitting[skews.name] = skews

                fitting_params_gamma = {"alpha": alphas, "beta": betas}
                fitting_params_pearson = {
                    "prob_zero": prob_zero,
                    "loc": locs,
                    "scale": scales,
                    "skew": skews,
                }

        for distribution in [indices.Distribution.gamma, indices.Distribution.pearson]:
            _logger.info(
                f"Computing {scale}-{periodicity.unit()} SPI ({distribution.value.capitalize()})",
            )

            if distribution is indices.Distribution.gamma:
                fitting_params = fitting_params_gamma
            else:
                fitting_params = fitting_params_pearson

            spi_values = indices.spi_xarray(
                precip_da,
                scale,
                distribution,
                data_start_year,
                keyword_arguments["calibration_start_year"],
                keyword_arguments["calibration_end_year"],
                periodicity,
                fitting_params=fitting_params,
            )

            if periodicity == compute.Periodicity.daily:

                def _to_gregorian(values_1d: np.ndarray) -> np.ndarray:
                    return utils.transform_to_gregorian(values_1d, data_start_year)

                spi_values = xr.apply_ufunc(
                    _to_gregorian,
                    spi_values,
                    input_core_dims=[[time_dim]],
                    output_core_dims=[[time_dim]],
                    output_sizes={time_dim: original_time_len},
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[float],
                )
                spi_values = spi_values.assign_coords({time_dim: original_time})

            spi_var_name = f"spi_{distribution.value}_{scale}_{periodicity.unit()}"
            if input_type == InputType.divisions:
                ds_spi = build_dataset_spi_divisions(
                    ds_precip,
                    scale,
                    periodicity,
                    distribution,
                    data_start_year,
                    spi_var_name,
                    spi_values,
                )
            elif input_type == InputType.timeseries:
                global_attrs = {
                    "description": f"SPI for {scale}-{periodicity.unit()} scale computed "
                    f"from {periodicity} precipitation input "
                    "by the climate_indices package available from "
                    f"{_GITHUB_URL}.",
                }
                coords = {"time": original_time}
                ds_spi = xr.Dataset(
                    coords=coords,
                    attrs=global_attrs,
                )
                var_attrs = _get_variable_attributes(distribution, scale, periodicity)
                da_spi = xr.DataArray(
                    data=spi_values.data,
                    coords=coords,
                    dims=("time",),
                    name=spi_var_name,
                    attrs=var_attrs,
                )
                ds_spi[spi_var_name] = da_spi
            else:
                ds_spi = build_dataset_spi_grid(
                    ds_precip,
                    scale,
                    periodicity,
                    distribution,
                    data_start_year,
                    spi_var_name,
                    spi_values,
                )

            netcdf_file_name = keyword_arguments["output_file_base"] + "_" + spi_var_name + ".nc"
            ds_spi.to_netcdf(netcdf_file_name)

    if ds_fitting is not None and keyword_arguments["save_params"] is not None:
        ds_fitting.to_netcdf(keyword_arguments["save_params"])


def build_dataset_spi_grid(
    ds_example: xr.Dataset,
    scale: int,
    periodicity: compute.Periodicity,
    distribution: indices.Distribution,
    data_start_year: int,
    spi_var_name: str,
    spi_values: xr.DataArray,
) -> xr.Dataset:
    global_attrs = {
        "description": f"SPI for {scale}-{periodicity.unit()} scale computed "
        f"from {periodicity} precipitation input "
        "by the climate_indices package available from "
        f"{_GITHUB_URL}.",
        "geospatial_lat_min": float(np.amin(ds_example.lat)),
        "geospatial_lat_max": float(np.amax(ds_example.lat)),
        "geospatial_lat_units": ds_example.lat.units,
        "geospatial_lon_min": float(np.amin(ds_example.lon)),
        "geospatial_lon_max": float(np.amax(ds_example.lon)),
        "geospatial_lon_units": ds_example.lon.units,
    }
    coords = {
        "lat": ds_example.lat,
        "lon": ds_example.lon,
        "time": ds_example.time,
    }
    ds_spi = xr.Dataset(
        coords=coords,
        attrs=global_attrs,
    )

    index_values = spi_values.transpose("lat", "lon", "time")

    # create a new variable to contain the SPI values, assign into the dataset
    var_attrs = _get_variable_attributes(distribution, scale, periodicity)
    da_spi = xr.DataArray(
        data=index_values.data,
        coords=ds_example.coords,
        dims=("lat", "lon", "time"),
        name=spi_var_name,
        attrs=var_attrs,
    )
    ds_spi[spi_var_name] = da_spi

    return ds_spi


def build_dataset_spi_divisions(
    ds_example: xr.Dataset,
    scale: int,
    periodicity: compute.Periodicity,
    distribution: indices.Distribution,
    data_start_year: int,
    spi_var_name: str,
    spi_values: xr.DataArray,
) -> xr.Dataset:
    global_attrs = {
        "description": f"SPI for {scale}-{periodicity.unit()} scale computed "
        f"from {periodicity} precipitation input "
        "by the climate_indices package available from "
        f"{_GITHUB_URL}.",
    }
    coords = {
        "division": ds_example.division,
        "time": ds_example.time,
    }
    ds_spi = xr.Dataset(
        coords=coords,
        attrs=global_attrs,
    )

    index_values = spi_values.transpose("division", "time")

    # create a new variable to contain the SPI values, assign into the dataset
    var_attrs = _get_variable_attributes(distribution, scale, periodicity)
    da_spi = xr.DataArray(
        data=index_values.data,
        coords=ds_example.coords,
        dims=("division", "time"),
        name=spi_var_name,
        attrs=var_attrs,
    )
    ds_spi[spi_var_name] = da_spi

    return ds_spi


def _prepare_file(netcdf_file, var_name):
    """
    Determine if the NetCDF file has the expected lat, lon, and time dimensions,
    and if not correctly ordered then create a temporary NetCDF with dimensions
    in (lat, lon, time) order, otherwise just return the input NetCDF unchanged.

    :param str netcdf_file:
    :param str var_name:
    :return: name of the NetCDF file containing correct dimensions
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


def main():  # type: () -> None
    # This function is used to perform climate indices processing on NetCDF
    # gridded datasets.
    #
    # Example command line arguments for SPI only using monthly precipitation input:
    #
    # --periodicity monthly
    # --scales 1 2 3 6 9 12 24
    # --calibration_start_year 1998
    # --calibration_end_year 2016
    # --netcdf_precip example_data/nclimgrid_prcp_lowres.nc
    # --var_name_precip prcp
    # --output_file_base ~/data/test/spi/nclimgrid_lowres

    # # ==========================================================================
    # # UNCOMMENT THE BELOW FOR PROFILING
    # # ==========================================================================
    # import cProfile
    # import sys
    #
    # # if check avoids hackery when not profiling
    # if sys.modules['__main__'].__file__ == cProfile.__file__:
    #     import process_grid  # Imports you again (does *not* use cache or execute as __main__)
    #
    #     globals().update(vars(process_grid))  # Replaces current contents with newly imported stuff
    #     sys.modules['__main__'] = process_grid  # Ensures pickle lookups on __main__ find matching version
    # # ========== END OF PROFILING-SPECIFIC CODE ================================

    try:
        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        _logger.info("Start time:    %s", start_datetime)

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--periodicity",
            help="Process input as either monthly or daily values",
            choices=[compute.Periodicity.monthly, compute.Periodicity.daily],
            type=compute.Periodicity.from_string,
            required=True,
        )
        parser.add_argument(
            "--scales",
            help="Timestep scales over which the SPI values are to be computed",
            type=int,
            nargs="*",
        )
        parser.add_argument(
            "--calibration_start_year",
            help="Initial year of the calibration period",
            type=int,
        )
        parser.add_argument(
            "--calibration_end_year",
            help="Final year of calibration period",
            type=int,
        )
        parser.add_argument(
            "--netcdf_precip",
            type=str,
            help="Precipitation NetCDF file to be used as input for indices computations",
        )
        parser.add_argument(
            "--var_name_precip",
            type=str,
            help="Precipitation variable name used in the precipitation NetCDF file",
        )
        parser.add_argument(
            "--output_file_base",
            type=str,
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
        parser.add_argument(
            "--load_params",
            type=str,
            required=False,
            help="path to input NetCDF file (to be read) containing distribution fitting parameters",
        )
        parser.add_argument(
            "--save_params",
            type=str,
            required=False,
            help="path to output NetCDF file (to be written) containing distribution fitting parameters",
        )
        parser.add_argument(
            "--overwrite",
            default=False,
            action="store_true",
            help="overwrite existing files if they exist",
        )
        arguments = parser.parse_args()

        # validate the arguments and determine the input type
        input_type = _validate_args(arguments)

        if arguments.multiprocessing == "single":
            number_of_workers = 1
        elif arguments.multiprocessing == "all":
            number_of_workers = os.cpu_count() or 1
        else:  # default ("all_but_one")
            number_of_workers = (os.cpu_count() or 1) - 1
            if number_of_workers < 1:
                number_of_workers = 1

        # prepare precipitation NetCDF in case dimensions not (lat, lon, time) or if any coordinates are descending
        netcdf_precip = _prepare_file(arguments.netcdf_precip, arguments.var_name_precip)

        # keyword arguments used for the SPI function
        kwrgs = {
            "netcdf_precip": netcdf_precip,
            "var_name_precip": arguments.var_name_precip,
            "input_type": input_type,
            "scales": arguments.scales,
            "periodicity": arguments.periodicity,
            "calibration_start_year": arguments.calibration_start_year,
            "calibration_end_year": arguments.calibration_end_year,
            "output_file_base": arguments.output_file_base,
            "load_params": arguments.load_params,
            "save_params": arguments.save_params,
            "number_of_workers": number_of_workers,
            "overwrite": arguments.overwrite,
        }

        client = None
        if arguments.multiprocessing == "all":
            try:
                from dask.distributed import Client
            except ImportError as exc:
                raise RuntimeError("dask.distributed is required for --multiprocessing all") from exc
            client = Client(n_workers=number_of_workers)

        try:
            # compute and write SPI
            _compute_write_index(kwrgs)
        finally:
            if client is not None:
                client.close()

        # remove temporary file if one was created
        if netcdf_precip != arguments.netcdf_precip:
            os.remove(netcdf_precip)

        # report on the elapsed time
        end_datetime = datetime.now()
        _logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        _logger.info("Elapsed time:  %s", elapsed)

    except Exception:
        _logger.exception("Failed to complete", exc_info=True)
        raise


if __name__ == "__main__":
    """
    (please do not remove -- useful for running as a script when debugging)

    Example command line usage for US climate divisions:

     $ spi --scales 1 2 3 6 9 12 24
     --netcdf_precip ../example_climate_indices/example/input/nclimdiv.nc
     --output_file_base /home/data/test/nclimdiv
     --var_name_precip prcp
     --calibration_start_year 1951 --calibration_end_year 2010
     --multiprocessing all --periodicity monthly


    Example command line usage for gridded data (nClimGrid):

     $ spi --scales 1 2 3 6 9 12 24
     --netcdf_precip ../example_climate_indices/example/input/nclimgrid_prcp.nc
     --output_file_base /home/data/test/nclimgrid
     --var_name_precip prcp
     --calibration_start_year 1951 --calibration_end_year 2010
     --multiprocessing all --periodicity monthly
    """
    main()
