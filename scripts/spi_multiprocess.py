import argparse
from collections import Counter
from datetime import datetime
from enum import Enum
import logging
import multiprocessing
import os
from typing import Dict, List

from nco import Nco
import numpy as np
import xarray as xr

from climate_indices import compute, indices, utils

# variable names for the distribution fitting parameters
_FITTING_PARAMETER_VARIABLES = ("alpha", "beta")
# _FITTING_PARAMETER_VARIABLES = ("alpha", "beta", "skew", "loc", "scale", "prob_zero")

# shared memory array dictionary keys
_KEY_ARRAY = "array"
_KEY_SHAPE = "shape"
_KEY_LAT = "lat"
_KEY_RESULT = "result_array"

# global dictionary to contain shared arrays for use by worker processes
_global_shared_arrays = {}

# ------------------------------------------------------------------------------
# Retrieve logger and set desired logging level
_logger = utils.get_logger(__name__, logging.INFO)


# ------------------------------------------------------------------------------
class InputType(Enum):
    """
    Enumeration type for differentiating between gridded, timeseriesn and US
    climate division datasets.
    """

    grid = 1
    divisions = 2
    timeseries = 3


# ------------------------------------------------------------------------------
def init_worker(arrays_and_shapes):
    """
    Initialization function that assigns named arrays into the global variable.

    :param arrays_and_shapes: dictionary containing variable names as keys
        and two-element dictionaries containing RawArrays and associated shapes
        (i.e. each value of the dictionary is itself a dictionary with one key "array"
        and another key _KEY_SHAPE)
    :return:
    """

    global _global_shared_arrays
    _global_shared_arrays = arrays_and_shapes


# ------------------------------------------------------------------------------
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
    ):
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
            mesg = f"Invalid dimensions of the {variable_plain_name} " + \
                   f"variable: {dims}\nValid dimension names and " + \
                   f"order: {expected_dimensions_grid} or " + \
                   f"{expected_dimensions_divisions} or " + \
                   f"{expected_dimensions_timeseries}"
            _logger.error(mesg)
            raise ValueError(mesg)

        return _input_type

    # the dimensions we expect to find for each data variable
    # (precipitation, temperature, and/or PET)
    expected_dimensions_divisions = [("time", "division"), ("division", "time")]
    expected_dimensions_grid = [("lat", "lon", "time"), ("time", "lat", "lon")]
    expected_dimensions_timeseries = [("time",)]

    # for fitting parameters we can either compute and save or load from file, but not both
    if args.load_params and args.save_params:
        msg = "Both of the mutually exclusive fitting parameter "\
              "file options were specified (both load and save)"
        _logger.error(msg)
        raise ValueError(msg)

    if args.load_params:

        # make sure the specified fitting parameters file exists
        if not os.path.exists(args.load_params):
            msg = f"The specified fitting parameters file {args.load_params} "\
                  "does not exist"
            _logger.error(msg)
            raise ValueError(msg)

        # open the fitting parameters file and make sure it looks reasonable
        with xr.open_dataset(args.load_params) as dataset_fittings:

            # confirm that all the fitting parameter variables are present
            missing_variables = []
            for var in _FITTING_PARAMETER_VARIABLES:
                for scale in args.scales:
                    fitting_var = "_".join((var, str(scale).zfill(2)))
                    if fitting_var not in dataset_fittings.variables:
                        missing_variables.append(fitting_var)
            if len(missing_variables) > 0:
                msg = "The following fitting parameter variables are expected "\
                      "but not present in the specified fitting parameters "\
                      f"dataset ({args.load_params}): {missing_variables}"
                _logger.error(msg)
                raise ValueError(msg)

            # TODO compare against the lats_precip, lons_precip, etc.

    # make sure a precipitation file was specified
    if args.netcdf_precip is None:
        msg = "Missing the required precipitation file"
        _logger.error(msg)
        raise ValueError(msg)

    # make sure a precipitation variable name was specified
    if args.var_name_precip is None:
        msg = "Missing precipitation variable name"
        _logger.error(msg)
        raise ValueError(msg)

    # validate the precipitation file itself
    with xr.open_dataset(args.netcdf_precip) as dataset_precip:

        # make sure we have a valid precipitation variable name
        if args.var_name_precip not in dataset_precip.variables:
            msg = f"Invalid precipitation variable name: '{args.var_name_precip}'" + \
                  f"does not exist in precipitation file '{args.netcdf_precip}'"
            _logger.error(msg)
            raise ValueError(msg)

        # verify that the precipitation variable's dimensions are in
        # the expected order, and if so then determine the input type
        input_type = \
            validate_dimensions(
                dataset_precip,
                args.var_name_precip,
                "precipitation",
            )

        # get the values of the precipitation coordinate variables,
        # for comparison against those of the other data variables
        if input_type == InputType.grid:
            lats_precip = dataset_precip["lat"].values[:]
            lons_precip = dataset_precip["lon"].values[:]
        elif input_type == InputType.divisions:
            divisions_precip = dataset_precip["division"].values[:]
        # TODO what if input_type == InputType.timeseries?
        times_precip = dataset_precip["time"].values[:]

    if args.scales is None:
        msg = "Missing one or more time scales (missing --scales argument)"
        _logger.error(msg)
        raise ValueError(msg)

    if any(n < 0 for n in args.scales):
        msg = "One or more negative scale specified within --scales argument"
        _logger.error(msg)
        raise ValueError(msg)

    return input_type


# ------------------------------------------------------------------------------
def _log_status(args_dict):

    # get the scale increment for use in later log messages
    _logger.info(
        f"Computing {args_dict['scale']}-"
        f"{args_dict['periodicity'].unit()} SPI/"
        f"{args_dict['distribution'].value.capitalize()}",
    )

    return True


# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
def _get_variable_attributes(args_dict):

    long_name = "Standardized Precipitation Index ({dist} distribution), ".format(
        dist=args_dict["distribution"].value.capitalize()
    ) + "{scale}-{increment}".format(
        scale=args_dict["scale"], increment=args_dict["periodicity"].unit()
    )
    attrs = {"long_name": long_name, "valid_min": -3.09, "valid_max": 3.09}
    var_name = (
        "spi_"
        + args_dict["distribution"].value
        + "_"
        + str(args_dict["scale"]).zfill(2)
    )

    return var_name, attrs


# ------------------------------------------------------------------------------
def _drop_data_into_shared_arrays_grid(
        dataset_climatology: xr.Dataset,
        dataset_fitting: xr.Dataset,
        var_names_climate: List[str],
        var_names_fitting: List[str],
        periodicity: compute.Periodicity,
):
    """
    Copies arrays from an xarray Dataset into shared memory arrays.

    :param dataset_climatology: Dataset containing climatology value arrays
    :param dataset_fitting: Dataset containing distribution fitting parameter arrays
    :param var_names_climate: names of variables to be copied into shared memory
    :param var_names_fitting: names of variables to be copied into shared memory
    :param periodicity: monthly or daily
    """

    # get the data arrays we'll use later in the index computations
    global _global_shared_arrays
    expected_dims_3d_climate = (
        ("lat", "lon", "time"),
        ("lon", "lat", "time"),
    )
    expected_dims_3d_fitting = (
        ("lat", "lon", "month"),
        ("lon", "lat", "month"),
        ("lat", "lon", "day"),
        ("lon", "lat", "day"),
    )
    expected_dims_2d = (("lat", "lon"), ("lon", "lat"))
    expected_dims_1d = (("time",),)

    # copy all variables from climatology Dataset into shared memory arrays
    for var_name in var_names_climate:

        # confirm that the dimensions of the data array are valid
        dims = dataset_climatology[var_name].dims
        if len(dims) == 3:
            if dims not in expected_dims_3d_climate:
                message = f"Invalid dimensions for variable '{var_name}': {dims}"
                _logger.error(message)
                raise ValueError(message)
        elif len(dims) == 2:
            if dims not in expected_dims_2d:
                message = f"Invalid dimensions for variable '{var_name}': {dims}"
                _logger.error(message)
                raise ValueError(message)
        elif len(dims) == 1:
            if dims not in expected_dims_1d:
                message = f"Invalid dimensions for variable '{var_name}': {dims}"
                _logger.error(message)
                raise ValueError(message)

        # convert daily values into 366-day years
        if periodicity == compute.Periodicity.daily:
            initial_year = int(dataset_climatology["time"][0].dt.year)
            final_year = int(dataset_climatology["time"][-1].dt.year)
            total_years = final_year - initial_year + 1
            var_values = np.apply_along_axis(utils.transform_to_366day,
                                             len(dims) - 1,
                                             dataset_climatology[var_name].values,
                                             initial_year,
                                             total_years)

        else:  # assumed to be monthly
            var_values = dataset_climatology[var_name].values

        # create a shared memory array, wrap it as a numpy array and copy
        # copy the data (values) from this variable's DataArray
        shared_array = multiprocessing.Array("d", int(np.prod(var_values.shape)))
        shared_array_np = \
            np.frombuffer(shared_array.get_obj()).reshape(var_values.shape)
        np.copyto(shared_array_np, var_values)

        # add to the dictionary of arrays
        _global_shared_arrays[var_name] = {
            _KEY_ARRAY: shared_array,
            _KEY_SHAPE: var_values.shape,
        }

        # drop the variable from the xarray Dataset (we're assuming this frees the memory)
        dataset_climatology = dataset_climatology.drop(var_name)

    # copy all variables from fitting parameters Dataset into shared memory arrays
    for var_name in var_names_fitting:

        # confirm that the dimensions of the data array are valid
        dims = dataset_fitting[var_name].dims
        if len(dims) == 3:
            if dims not in expected_dims_3d_fitting:
                message = f"Invalid dimensions for variable '{var_name}': {dims}"
                _logger.error(message)
                raise ValueError(message)
        elif len(dims) == 2:
            if dims not in expected_dims_2d:
                message = f"Invalid dimensions for variable '{var_name}': {dims}"
                _logger.error(message)
                raise ValueError(message)
        elif len(dims) == 1:
            if dims not in expected_dims_1d:
                message = f"Invalid dimensions for variable '{var_name}': {dims}"
                _logger.error(message)
                raise ValueError(message)

        # convert daily values into 366-day years
        if periodicity == compute.Periodicity.daily:
            initial_year = int(dataset_climatology["time"][0].dt.year)
            final_year = int(dataset_climatology["time"][-1].dt.year)
            total_years = final_year - initial_year + 1
            var_values = np.apply_along_axis(utils.transform_to_366day,
                                             len(dims) - 1,
                                             dataset_fitting[var_name].values,
                                             initial_year,
                                             total_years)

        else:  # assumed to be monthly
            var_values = dataset_fitting[var_name].values

        # create a shared memory array, wrap it as a numpy array and copy
        # copy the data (values) from this variable's DataArray
        shared_array = multiprocessing.Array("d", int(np.prod(var_values.shape)))
        shared_array_np = \
            np.frombuffer(shared_array.get_obj()).reshape(var_values.shape)
        np.copyto(shared_array_np, var_values)

        # add to the dictionary of arrays
        _global_shared_arrays[var_name] = {
            _KEY_ARRAY: shared_array,
            _KEY_SHAPE: var_values.shape,
        }


# ------------------------------------------------------------------------------
def _drop_data_into_shared_arrays_divisions(
        dataset: xr.Dataset,
        var_names: List[str],
):
    # TODO add fitting variables as we've done in _drop_data_into_shared_arrays_grid
    """
    Copies arrays from an xarray Dataset into shared memory arrays.

    :param dataset:
    :param var_names: names of variables to be copied into shared memory
    """

    # get the data arrays we'll use later in the index computations
    global _global_shared_arrays
    expected_dims_2d = [
        ("division", "time"),
        ("time", "division"),
        ("division", "month"),
        ("month", "division"),
        ("division", "day"),
        ("day", "division"),
    ]
    expected_dims_1d = [("division",)]
    for var_name in var_names:

        # confirm that the dimensions of the data array are valid
        dims = dataset[var_name].dims
        if len(dims) == 2:
            if dims not in expected_dims_2d:
                message = f"Invalid dimensions for variable '{var_name}': {dims}"
                _logger.error(message)
                raise ValueError(message)
        elif len(dims) == 1:
            if dims not in expected_dims_1d:
                message = f"Invalid dimensions for variable '{var_name}': {dims}"
                _logger.error(message)
                raise ValueError(message)

        # create a shared memory array, wrap it as a numpy array and copy
        # copy the data (values) from this variable's DataArray
        shared_array = multiprocessing.Array("d", int(np.prod(dataset[var_name].shape)))
        shared_array_np = np.frombuffer(shared_array.get_obj()).reshape(dataset[var_name].shape)
        np.copyto(shared_array_np, dataset[var_name].values)

        # add to the dictionary of arrays
        _global_shared_arrays[var_name] = {
            _KEY_ARRAY: shared_array,
            _KEY_SHAPE: dataset[var_name].shape,
        }

        # drop the variable from the dataset (we're assuming this frees the memory)
        dataset = dataset.drop(var_name)


# ------------------------------------------------------------------------------
def _compute_write_index(keyword_arguments):
    """
    Computes a climate index and writes the result into a corresponding NetCDF.

    :param keyword_arguments:
    :return:
    """

    _log_status(keyword_arguments)

    # open the NetCDF files as an xarray DataSet object
    files = []
    if "netcdf_precip" in keyword_arguments:
        files.append(keyword_arguments["netcdf_precip"])
    if "input_type" not in keyword_arguments:
        raise ValueError("Missing the 'input_type' keyword argument")
    else:
        input_type = keyword_arguments["input_type"]
        if input_type == InputType.grid:
            chunks = {"lat": -1, "lon": -1}
        elif input_type == InputType.divisions:
            chunks = {"division": -1}
        elif input_type == InputType.timeseries:
            chunks = {"time": -1}
        else:
            raise ValueError(f"Invalid 'input_type' keyword argument: {input_type}")
    ds_precip = xr.open_mfdataset(files, chunks=chunks)

    # trim out all data variables from the dataset except the ones we'll need
    input_var_names = []
    if "var_name_precip" in keyword_arguments:
        input_var_names.append(keyword_arguments["var_name_precip"])
    # only keep the latitude variable if we're dealing with divisions
    if input_type == InputType.divisions:
        input_var_names.append("lat")
    for var in ds_precip.data_vars:
        if var not in input_var_names:
            ds_precip = ds_precip.drop(var)

    # add placeholder variables in the Dataset to hold the fitting parameters we'll compute
    if (("save_params" in keyword_arguments) and \
            (keyword_arguments["save_params"] is not None)) or \
            keyword_arguments["load_params"] is not None:

        if keyword_arguments["periodicity"] == compute.Periodicity.monthly:
            period_times = 12
        elif keyword_arguments["periodicity"] == compute.Periodicity.daily:
            period_times = 366
        else:
            raise ValueError(f"Unsupported periodicity: {keyword_arguments['periodicity']}")

        time_coord_name = keyword_arguments['periodicity'].unit()
        if input_type == InputType.grid:
            fitting_coords = {"lat": ds_precip.lat, "lon": ds_precip.lon, time_coord_name: range(period_times)}
            data_shape = (len(ds_precip.lat), len(ds_precip.lon), period_times)
        elif input_type == InputType.divisions:
            fitting_coords = {"division": ds_precip.division, time_coord_name: range(period_times)}
            data_shape = (len(ds_precip.division), period_times)
        elif input_type == InputType.timeseries:
            fitting_coords = {time_coord_name: range(period_times)}
            data_shape = (period_times,)
        else:
            raise ValueError(f"Invalid 'input_type' keyword argument: {input_type}")

        # open the dataset if it's already been written to file, otherwise create it
        if os.path.exists(keyword_arguments["save_params"]):
            ds_fitting = xr.open_dataset(keyword_arguments["save_params"])
        elif keyword_arguments["load_params"] is not None:
            ds_fitting = xr.open_dataset(keyword_arguments["load_params"])
        else:
            attrs_to_copy = [
                'Conventions',
                'ncei_template_version',
                'naming_authority',
                'standard_name_vocabulary',
                'institution',
                'geospatial_lat_min',
                'geospatial_lat_max',
                'geospatial_lon_min',
                'geospatial_lon_max',
                'geospatial_lat_units',
                'geospatial_lon_units',
            ]
            global_attrs = {key: value for (key, value) in ds_precip.attrs.items() if key in attrs_to_copy}
            ds_fitting = xr.Dataset(
                coords=fitting_coords,
                attrs=global_attrs,
            )

        # build DataArrays for the parameter fittings we'll compute
        # (only if not already in the Dataset when loading from existing file)
        fitting_var_names = {}
        fitting_var_name_suffix = f"{keyword_arguments['scale']}_{keyword_arguments['periodicity'].unit()}"
        if keyword_arguments["distribution"] == indices.Distribution.gamma:

            # add an empty DataArray to the fitting Dataset for alpha if not present
            alpha_var_name = f"alpha_{fitting_var_name_suffix}"
            fitting_var_names["alpha"] = alpha_var_name
            if (keyword_arguments["save_params"] is not None) or (alpha_var_name not in ds_fitting.data_vars):

                alpha_attrs = {
                    'description': 'shape parameter of the gamma distribution (also referred to as the concentration) ' + \
                                   f'computed from the {keyword_arguments["scale"]}-month scaled precipitation values',
                }
                da_alpha = xr.DataArray(
                    data=np.full(shape=data_shape, fill_value=np.NaN),
                    coords=fitting_coords,
                    dims=tuple(fitting_coords.keys()),
                    name=alpha_var_name,
                    attrs=alpha_attrs,
                )
                ds_fitting[alpha_var_name] = da_alpha

            # add an empty DataArray to the fitting Dataset for beta if not present
            beta_var_name = f"beta_{fitting_var_name_suffix}"
            fitting_var_names["beta"] = beta_var_name
            if (keyword_arguments["save_params"] is not None) or (beta_var_name not in ds_fitting.data_vars):

                beta_attrs = {
                    'description': '1 / scale of the distribution (also referred to as the rate) ' + \
                                   f'computed from the {keyword_arguments["scale"]}-month scaled precipitation values',
                }
                da_beta = xr.DataArray(
                    data=np.full(shape=data_shape, fill_value=np.NaN),
                    coords=fitting_coords,
                    dims=tuple(fitting_coords.keys()),
                    name=beta_var_name,
                    attrs=beta_attrs,
                )
                ds_fitting[beta_var_name] = da_beta

        elif keyword_arguments["distribution"] == indices.Distribution.pearson:

            # add an empty DataArray to the fitting Dataset for probability of zero if not present
            prob_zero_var_name = f"prob_zero_{fitting_var_name_suffix}"
            fitting_var_names["prob_zero"] = prob_zero_var_name
            if (keyword_arguments["save_params"] is not None) or \
                    (prob_zero_var_name not in ds_fitting.data_vars):

                prob_zero_attrs = {
                    'description': 'probability of zero values within calibration period',
                }
                da_prob_zero = xr.DataArray(
                    data=np.full(shape=data_shape, fill_value=np.NaN),
                    coords=fitting_coords,
                    dims=tuple(fitting_coords.keys()),
                    name=prob_zero_var_name,
                    attrs=prob_zero_attrs,
                )
                ds_fitting[prob_zero_var_name] = da_prob_zero

            # add an empty DataArray to the fitting Dataset for scale parameter if not present
            scale_var_name = f"scale_{fitting_var_name_suffix}"
            fitting_var_names["scale"] = scale_var_name
            if (keyword_arguments["save_params"] is not None) or \
                    (scale_var_name not in ds_fitting.data_vars):

                scale_attrs = {
                    'description': 'scale parameter for Pearson Type III',
                }
                scale_var_name = f"scale_{keyword_arguments['scale']}_{keyword_arguments['periodicity'].unit()}"
                da_scale = xr.DataArray(
                    data=np.full(shape=data_shape, fill_value=np.NaN),
                    coords=fitting_coords,
                    dims=tuple(fitting_coords.keys()),
                    name=scale_var_name,
                    attrs=scale_attrs,
                )
                ds_fitting[scale_var_name] = da_scale

            # add an empty DataArray to the fitting Dataset for skew parameter if not present
            skew_var_name = f"skew_{fitting_var_name_suffix}"
            fitting_var_names["skew"] = skew_var_name
            if (keyword_arguments["save_params"] is not None) or \
                    (skew_var_name not in ds_fitting.data_vars):

                skew_attrs = {
                    'description': 'skew parameter for Pearson Type III',
                }
                da_skew = xr.DataArray(
                    data=np.full(shape=data_shape, fill_value=np.NaN),
                    coords=fitting_coords,
                    dims=tuple(fitting_coords.keys()),
                    name=skew_var_name,
                    attrs=skew_attrs,
                )
                ds_fitting[skew_var_name] = da_skew

            # add an empty DataArray to the fitting Dataset for loc parameter if not present
            loc_var_name = f"loc_{fitting_var_name_suffix}"
            fitting_var_names["loc"] = loc_var_name
            if (keyword_arguments["save_params"] is not None) or \
                    (loc_var_name not in ds_fitting.data_vars):

                loc_attrs = {
                    'description': 'loc parameter for Pearson Type III',
                }
                da_loc = xr.DataArray(
                    data=np.full(shape=data_shape, fill_value=np.NaN),
                    coords=fitting_coords,
                    dims=tuple(fitting_coords.keys()),
                    name=loc_var_name,
                    attrs=loc_attrs,
                )
                ds_fitting[loc_var_name] = da_loc

    # get the initial year of the data
    data_start_year = int(ds_precip['time'][0].dt.year)
    keyword_arguments["data_start_year"] = data_start_year

    # the shape of output variables is assumed to match that
    # of the input, so use the precipitation variable's shape
    if "var_name_precip" in keyword_arguments:
        precip_var_name = keyword_arguments["var_name_precip"]
        output_dims = ds_precip[precip_var_name].dims

        # convert precipitation data into millimeters
        precip_unit = ds_precip[precip_var_name].units.lower()
        if precip_unit not in ("mm", "millimeters", "millimeter", "mm/dy"):
            if precip_unit in ("inches", "inch"):
                # inches to mm conversion (1 inch == 25.4 mm)
                ds_precip[precip_var_name].values *= 25.4
            else:
                raise ValueError(f"Unsupported precipitation units: {precip_unit}")

    else:
        raise ValueError("No precipitation variable name was specified.")

    if input_type == InputType.divisions:
        _drop_data_into_shared_arrays_divisions(ds_precip, input_var_names)
    else:
        _drop_data_into_shared_arrays_grid(
            ds_precip,
            ds_fitting,
            input_var_names,
            fitting_var_names.values(),
            keyword_arguments["periodicity"],
        )

    # use the precipitation shape as the output shape for the index values
    output_shape = ds_precip[keyword_arguments["var_name_precip"]].shape

    # build an arguments dictionary appropriate to the index we'll compute
    args = _build_arguments(keyword_arguments)

    # add an array to hold index computation results
    # to our dictionary of shared memory arrays
    if _KEY_RESULT not in _global_shared_arrays:
        _global_shared_arrays[_KEY_RESULT] = {
            _KEY_ARRAY: multiprocessing.Array("d", int(np.prod(output_shape))),
            _KEY_SHAPE: output_shape,
        }

    # compute the distribution fitting parameters if necessary (i.e. not loaded)
    if keyword_arguments["load_params"] is None:
        _parallel_fitting(
            keyword_arguments["distribution"],
            _global_shared_arrays,
            {"var_name_values": keyword_arguments["var_name_precip"]},
            fitting_var_names,
            input_type=input_type,
            number_of_workers=keyword_arguments["number_of_workers"],
            args=args,
        )

    # apply the SPI function along the time axis (axis=2)
    var_names_spi = fitting_var_names
    var_names_spi["var_name_precip"] = keyword_arguments["var_name_precip"]
    _parallel_spi(
        keyword_arguments["index"],
        _global_shared_arrays,
        var_names_spi,
        _KEY_RESULT,
        input_type=input_type,
        number_of_workers=keyword_arguments["number_of_workers"],
        args=args,
    )

    # get the name and attributes to use for the index variable in the output NetCDF
    output_var_name, output_var_attributes = \
        _get_variable_attributes(keyword_arguments)

    # get the shared memory results array and convert it to a numpy array
    array = _global_shared_arrays[_KEY_RESULT][_KEY_ARRAY]
    shape = _global_shared_arrays[_KEY_RESULT][_KEY_SHAPE]
    index_values = np.frombuffer(array.get_obj()).reshape(shape).astype(np.float32)

    # convert daily values into normal/Gregorian calendar years
    if keyword_arguments["periodicity"] == compute.Periodicity.daily:
        index_values = np.apply_along_axis(
            utils.transform_to_gregorian,
            len(output_dims) - 1,
            index_values,
            keyword_arguments["data_start_year"],
        )

    # create a new variable to contain the SPI values, assign into the dataset
    # TODO create a separate Dataset instead with only relevant attributes etc.
    #   rather than hijacking the precipitation dataset
    variable = xr.Variable(dims=output_dims,
                           data=index_values,
                           attrs=output_var_attributes)
    ds_precip[output_var_name] = variable

    # TODO set global attributes accordingly for this new dataset

    # remove all data variables except for the new variable
    for var_name in ds_precip.data_vars:
        if var_name != output_var_name:
            ds_precip = ds_precip.drop(var_name)

    # write the dataset as NetCDF
    netcdf_file_name = \
        keyword_arguments["output_file_base"] + "_" + output_var_name + ".nc"
    ds_precip.to_netcdf(netcdf_file_name)

    # if requested then we write the distribution fittings dataset to NetCDF
    if keyword_arguments["save_params"] is not None:

        # TODO copy the values from the fitting variable shared arrays into
        #  the fitting Dataset since we'll need to write it out later

        # ds_fitting.to_netcdf(keyword_arguments["save_params"])
        pass

    return netcdf_file_name, output_var_name


# ------------------------------------------------------------------------------
def _init_worker(shared_arrays_dict):

    global _global_shared_arrays
    _global_shared_arrays = shared_arrays_dict


# ------------------------------------------------------------------------------
def _parallel_spi(
        index: str,
        arrays_dict: dict,
        input_var_names: dict,
        output_var_name: str,
        input_type: InputType,
        number_of_workers: int,
        args,
):
    """
    TODO document this function

    :param str index:
    :param dict arrays_dict:
    :param dict input_var_names:
    :param str output_var_name:
    :param InputType input_type:
    :param number_of_workers: number of worker processes to use
    :param args:
    :return:
    """

    # find the start index of each sub-array we'll split out per worker process,
    # assuming the shape of the output array is the same as all input arrays
    shape = arrays_dict[output_var_name][_KEY_SHAPE]
    d, m = divmod(shape[0], number_of_workers)
    split_indices = list(range(0, ((d + 1) * (m + 1)), (d + 1)))
    if d != 0:
        split_indices += list(range(split_indices[-1] + d, shape[0], d))

    # build a list of parameters for each application of the function to an array chunk
    chunk_params = []

    # we have a single input array, create parameter dictionary objects
    # appropriate to the _apply_along_axis function, one per worker process
    for i in range(number_of_workers):
        params = {
            "index": index,
            "input_var_name": input_var_names["var_name_precip"],
            "output_var_name": output_var_name,
            "sub_array_start": split_indices[i],
            "input_type": input_type,
            "args": args,
        }
        if i < (number_of_workers - 1):
            params["sub_array_end"] = split_indices[i + 1]
        else:
            params["sub_array_end"] = None

        # set the distribution fitting variable name parameters
        if args["distribution"] is indices.Distribution.gamma:
            params["var_name_alpha"] = input_var_names["alpha"]
            params["var_name_beta"] = input_var_names["beta"]
        elif args["distribution"] is indices.Distribution.pearson:
            params["var_name_prob_zero"] = input_var_names["prob_zero"]
            params["var_name_loc"] = input_var_names["loc"]
            params["var_name_scale"] = input_var_names["scale"]
            params["var_name_skew"] = input_var_names["skew"]

        chunk_params.append(params)

    # instantiate a process pool
    with multiprocessing.Pool(processes=number_of_workers,
                              initializer=_init_worker,
                              initargs=(arrays_dict,)) as pool:

        pool.map(_apply_to_subarray_spi, chunk_params)


# ------------------------------------------------------------------------------
def _parallel_fitting(
        distribution: indices.Distribution,
        shared_arrays: Dict,
        input_var_names: Dict,
        output_var_names: Dict,
        input_type: InputType,
        number_of_workers: int,
        args: Dict,
):
    """
    TODO document this function

    :param distribution:
    :param Dict shared_arrays:
    :param Dict input_var_names:
    :param Dict output_var_names:
    :param InputType input_type:
    :param number_of_workers: number of worker processes to use
    :param args:
    :return:
    """

    # TODO somehow account for the calibration period, i.e. only compute
    #  fitting parameters on values within the calibration period

    # find the start index of each sub-array we'll split out per worker process,
    # assuming the shape of the output array is the same as all input arrays
    for output_var_name in output_var_names.values():
        shape = shared_arrays[output_var_name][_KEY_SHAPE]
        d, m = divmod(shape[0], number_of_workers)
        split_indices = list(range(0, ((d + 1) * (m + 1)), (d + 1)))
        if d != 0:
            split_indices += list(range(split_indices[-1] + d, shape[0], d))

    # build a list of parameters for each application of the function to an array chunk
    chunk_params = []
    for i in range(number_of_workers):
        params = {
            "input_var_name": input_var_names["var_name_values"],
            "output_var_names": output_var_names,
            "sub_array_start": split_indices[i],
            "input_type": input_type,
            "args": args,
        }
        if i < (number_of_workers - 1):
            params["sub_array_end"] = split_indices[i + 1]
        else:
            params["sub_array_end"] = None

        chunk_params.append(params)

    # instantiate a process pool
    with multiprocessing.Pool(processes=number_of_workers,
                              initializer=_init_worker,
                              initargs=(shared_arrays,)) as pool:

        if distribution == indices.Distribution.gamma:
            pool.map(_apply_to_subarray_gamma, chunk_params)
        elif distribution == indices.Distribution.pearson:
            pool.map(_apply_to_subarray_pearson, chunk_params)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")


# ------------------------------------------------------------------------------
def _apply_to_subarray_spi(params):
    """
    Like numpy.apply_along_axis(), but with arguments in a dict instead.
    Applicable for applying a function across subarrays of a single input array.

    This function is useful with multiprocessing.Pool().map(): (1) map() only
    handles functions that take a single argument, and (2) this function can
    generally be imported from a module, as required by map().

    :param dict params: dictionary of parameters including a function name,
        "func1d", start and stop indices for specifying the subarray to which
        the function should be applied, "sub_array_start" and "sub_array_end",
        a dictionary of arguments to be passed to the function, "args", and
        the key name of the shared array for output values, "output_var_name".
    """
    start_index = params["sub_array_start"]
    end_index = params["sub_array_end"]
    array = _global_shared_arrays[params["input_var_name"]][_KEY_ARRAY]
    shape = _global_shared_arrays[params["input_var_name"]][_KEY_SHAPE]
    np_array = np.frombuffer(array.get_obj()).reshape(shape)
    values_sub_array = np_array[start_index:end_index]
    args = params["args"]

    if args["distribution"] == indices.Distribution.gamma:

        array_alpha = _global_shared_arrays[params["var_name_alpha"]][_KEY_ARRAY]
        shape_alpha = _global_shared_arrays[params["var_name_alpha"]][_KEY_SHAPE]
        np_array_alpha = np.frombuffer(array_alpha.get_obj()).reshape(shape_alpha)
        sub_array_alpha = np_array_alpha[start_index:end_index]

        array_beta = _global_shared_arrays[params["var_name_beta"]][_KEY_ARRAY]
        shape_beta = _global_shared_arrays[params["var_name_beta"]][_KEY_SHAPE]
        np_array_beta = np.frombuffer(array_beta.get_obj()).reshape(shape_beta)
        sub_array_beta = np_array_beta[start_index:end_index]

        fitting_params = {
            "alpha": sub_array_alpha,
            "beta": sub_array_beta,
        }

    elif args["distribution"] == indices.Distribution.pearson:

        array_prob_zero = _global_shared_arrays[params["var_name_prob_zero"]][_KEY_ARRAY]
        shape_prob_zero = _global_shared_arrays[params["var_name_prob_zero"]][_KEY_SHAPE]
        np_array_prob_zero = np.frombuffer(array_prob_zero.get_obj()).reshape(shape_prob_zero)
        sub_array_prob_zero = np_array_prob_zero[start_index:end_index]

        array_loc = _global_shared_arrays[params["var_name_loc"]][_KEY_ARRAY]
        shape_loc = _global_shared_arrays[params["var_name_loc"]][_KEY_SHAPE]
        np_array_loc = np.frombuffer(array_loc.get_obj()).reshape(shape_loc)
        sub_array_loc = np_array_loc[start_index:end_index]

        array_scale = _global_shared_arrays[params["var_name_scale"]][_KEY_ARRAY]
        shape_scale = _global_shared_arrays[params["var_name_scale"]][_KEY_SHAPE]
        np_array_scale = np.frombuffer(array_scale.get_obj()).reshape(shape_scale)
        sub_array_scale = np_array_scale[start_index:end_index]

        array_skew = _global_shared_arrays[params["var_name_skew"]][_KEY_ARRAY]
        shape_skew = _global_shared_arrays[params["var_name_skew"]][_KEY_SHAPE]
        np_array_skew = np.frombuffer(array_skew.get_obj()).reshape(shape_skew)
        sub_array_skew = np_array_skew[start_index:end_index]

        fitting_params = {
            "prob_zero": sub_array_prob_zero,
            "loc": sub_array_loc,
            "scale": sub_array_scale,
            "skew": sub_array_skew,
        }

    else:
        raise ValueError(f"Unsupported distribution: {args['distribution']}")

    args["fitting_params"] = fitting_params

    # get the output shared memory array, convert to numpy, and get the subarray slice
    output_array = _global_shared_arrays[params["output_var_name"]][_KEY_ARRAY]
    computed_array = np.frombuffer(output_array.get_obj()).reshape(shape)[
        start_index:end_index
    ]

    for i, values in enumerate(values_sub_array):
        if params["input_type"] == InputType.grid:
            for j in range(values.shape[0]):

                if args["distribution"] == indices.Distribution.gamma:

                    fitting_params = {
                        "alpha": sub_array_alpha[i, j],
                        "beta": sub_array_beta[i, j],
                    }

                elif args["distribution"] == indices.Distribution.pearson:

                    fitting_params = {
                        "prob_zero": sub_array_prob_zero[i, j],
                        "loc": sub_array_loc[i, j],
                        "scale": sub_array_scale[i, j],
                        "skew": sub_array_skew[i, j],
                    }

                computed_array[i, j] = \
                    indices.spi(
                        values[j],
                        scale=args["scale"],
                        distribution=args["distribution"],
                        data_start_year=args["data_start_year"],
                        calibration_year_initial=args["calibration_year_initial"],
                        calibration_year_final=args["calibration_year_final"],
                        periodicity=args["periodicity"],
                        fitting_params=fitting_params,
                    )

        else:  # divisions

            if args["distribution"] == indices.Distribution.gamma:

                fitting_params = {
                    "alpha": sub_array_alpha[i],
                    "beta": sub_array_beta[i],
                }

            elif args["distribution"] == indices.Distribution.pearson:

                fitting_params = {
                    "prob_zero": sub_array_prob_zero[i],
                    "loc": sub_array_loc[i],
                    "scale": sub_array_scale[i],
                    "skew": sub_array_skew[i],
                }

            computed_array[i] = \
                indices.spi(
                    values_sub_array,
                    scale=args["scale"],
                    distribution=args["distribution"],
                    data_start_year=args["data_start_year"],
                    calibration_year_initial=args["calibration_year_initial"],
                    calibration_year_final=args["calibration_year_final"],
                    periodicity=args["periodicity"],
                    fitting_params=fitting_params,
                )


# ------------------------------------------------------------------------------
def _apply_to_subarray_gamma(params):
    """
    Applies the gamma fitting computation function across subarrays
    of the input (shared-memory) arrays.

    This function is useful with multiprocessing.Pool().map(): (1) map() only
    handles functions that take a single argument, and (2) this function can
    generally be imported from a module, as required by map().

    :param dict params: dictionary of parameters including a function name,
        "func1d", start and stop indices for specifying the subarray to which
        the function should be applied, "sub_array_start" and "sub_array_end",
        the variable names used for precipitation, alpha, and beta arrays,
        "var_name_precip", "var_name_alpha", and "var_name_beta", a dictionary
        of arguments to be passed to the function, "args", and the key name of
        the shared arrays for output values, "alpha_var_name" and "beta_var_name".
    """
    start_index = params["sub_array_start"]
    end_index = params["sub_array_end"]
    values_array_key = params["input_var_name"]
    alpha_array_key = params["output_var_names"]["alpha"]
    beta_array_key = params["output_var_names"]["beta"]

    values_shape = _global_shared_arrays[values_array_key][_KEY_SHAPE]
    values_array = _global_shared_arrays[values_array_key][_KEY_ARRAY]
    values_np_array = np.frombuffer(values_array.get_obj()).reshape(values_shape)
    sub_array_values = values_np_array[start_index:end_index]

    args = params["args"]

    # get the output shared memory arrays, convert to numpy, and get the subarray slices
    fitting_shape = _global_shared_arrays[alpha_array_key][_KEY_SHAPE]
    alpha_output_array = _global_shared_arrays[alpha_array_key][_KEY_ARRAY]
    alpha_np_array = np.frombuffer(alpha_output_array.get_obj()).reshape(fitting_shape)
    sub_array_alpha = alpha_np_array[start_index:end_index]

    beta_output_array = _global_shared_arrays[beta_array_key][_KEY_ARRAY]
    beta_np_array = np.frombuffer(beta_output_array.get_obj()).reshape(fitting_shape)
    sub_array_beta = beta_np_array[start_index:end_index]

    for i, values in enumerate(sub_array_values):
        if params["input_type"] == InputType.grid:
            for j in range(values.shape[0]):
                sub_array_alpha[i, j], sub_array_beta[i, j] = \
                compute.gamma_parameters(
                    values=values[j],
                    data_start_year=args["data_start_year"],
                    calibration_start_year=args["calibration_year_initial"],
                    calibration_end_year=args["calibration_year_final"],
                    periodicity=args["periodicity"]
                )
        else:  # divisions
            sub_array_alpha[i], sub_array_beta[i] = \
                compute.gamma_parameters(
                    values=values,
                    data_start_year=args["data_start_year"],
                    calibration_start_year=args["calibration_year_initial"],
                    calibration_end_year=args["calibration_year_final"],
                    periodicity=args["periodicity"]
                )


# ------------------------------------------------------------------------------
def _apply_to_subarray_pearson(params):
    """
    Applies the Pearson Type III distribution fitting computation function
    across subarrays of the input (shared-memory) arrays.

    This function is useful with multiprocessing.Pool().map(): (1) map() only
    handles functions that take a single argument, and (2) this function can
    generally be imported from a module, as required by map().

    :param dict params: dictionary of parameters including a function name,
        "func1d", start and stop indices for specifying the subarray to which
        the function should be applied, "sub_array_start" and "sub_array_end",
        the variable names used for precipitation, probability of zero, scale,
        skew, and loc arrays ("var_name_precip", "var_name_prob_zero",
        "var_name_scale", "var_name_skew", and "var_name_loc"), a dictionary of
        arguments to be passed to the function, "args", and the key name of
        the shared arrays for output values, "prob_zero_var_name" "scale_var_name",
        "skew_var_name", and "loc_var_name".
    """
    start_index = params["sub_array_start"]
    end_index = params["sub_array_end"]
    values_array_key = params["input_var_name"]
    prob_zero_array_key = params["output_var_names"]["prob_zero"]
    scale_array_key = params["output_var_names"]["scale"]
    skew_array_key = params["output_var_names"]["skew"]
    loc_array_key = params["output_var_names"]["loc"]

    values_shape = _global_shared_arrays[values_array_key][_KEY_SHAPE]
    values_array = _global_shared_arrays[values_array_key][_KEY_ARRAY]
    values_np_array = np.frombuffer(values_array.get_obj()).reshape(values_shape)
    sub_array_values = values_np_array[start_index:end_index]

    args = params["args"]

    # get the output shared memory arrays, convert to numpy, and get the subarray slices
    fitting_shape = _global_shared_arrays[prob_zero_array_key][_KEY_SHAPE]
    prob_zero_output_array = _global_shared_arrays[prob_zero_array_key][_KEY_ARRAY]
    prob_zero_np_array = np.frombuffer(prob_zero_output_array.get_obj()).reshape(fitting_shape)
    sub_array_prob_zero = prob_zero_np_array[start_index:end_index]

    scale_output_array = _global_shared_arrays[scale_array_key][_KEY_ARRAY]
    scale_np_array = np.frombuffer(scale_output_array.get_obj()).reshape(fitting_shape)
    sub_array_scale = scale_np_array[start_index:end_index]

    skew_output_array = _global_shared_arrays[skew_array_key][_KEY_ARRAY]
    skew_np_array = np.frombuffer(skew_output_array.get_obj()).reshape(fitting_shape)
    sub_array_skew = skew_np_array[start_index:end_index]

    loc_output_array = _global_shared_arrays[loc_array_key][_KEY_ARRAY]
    loc_np_array = np.frombuffer(loc_output_array.get_obj()).reshape(fitting_shape)
    sub_array_loc = loc_np_array[start_index:end_index]

    for i, values in enumerate(sub_array_values):
        if params["input_type"] == InputType.grid:
            for j in range(values.shape[0]):
                sub_array_prob_zero[i, j], sub_array_loc[i, j], sub_array_scale[i, j], sub_array_skew[i, j] = \
                    compute.pearson_parameters(values[j], args["periodicity"])
        else:  # divisions
            sub_array_prob_zero[i], sub_array_loc[i], sub_array_scale[i], sub_array_skew[i] = \
                compute.pearson_parameters(values, args["periodicity"])


# ------------------------------------------------------------------------------
def _prepare_file(netcdf_file, var_name):
    """
    Determine if the NetCDF file has the expected lat, lon, and time dimensions,
    and if not correctly ordered then create a temporary NetCDF with dimensions
    in (lat, lon, time) order, otherwise just return the input NetCDF unchanged.

    :param str netcdf_file:
    :param str var_name:
    :return: name of the NetCDF file containing correct dimensions
    """

    # make sure we have lat, lon, and time as variable dimensions, regardless of order
    ds = xr.open_dataset(netcdf_file)
    dimensions = ds[var_name].dims
    if "division" in dimensions:
        if len(dimensions) == 1:
            expected_dims = ("division",)
            dims = "division"
        elif len(dimensions) == 2:
            expected_dims = ("division", "time")
            dims = "division,time"
        else:
            raise ValueError(f"Unsupported dimensions for variable '{var_name}': {dimensions}")
    else:  # timeseries or gridded
        if len(dimensions) == 1:
            expected_dims = ("time",)
            dims = "time"
        elif len(dimensions) == 2:
            expected_dims = ("lat", "lon")
            dims = "lat,lon"
        elif len(dimensions) == 3:
            expected_dims = ("lat", "lon", "time")
            dims = "lat,lon,time"
        else:
            message = f"Unsupported dimensions for variable '{var_name}': {dimensions}"
            _logger.error(message)
            raise ValueError()

    if Counter(ds[var_name].dims) != Counter(expected_dims):
        message = f"Invalid dimensions for variable '{var_name}': {ds[var_name].dims}"
        _logger.error(message)
        raise ValueError(message)

    # perform reorder of dimensions if necessary
    if ds[var_name].dims != expected_dims:
        nco = Nco()
        netcdf_file = nco.ncpdq(input=netcdf_file,
                                options=[f'-a "{dims}"', "-O"])

    return netcdf_file


# ------------------------------------------------------------------------------
def main():  # type: () -> None

    # This function is used to perform climate indices processing on NetCDF
    # gridded datasets.
    #
    # Example command line arguments for SPI only using monthly precipitation input:
    #
    # --index spi
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
            "--index",
            type=str,
            help="Indices to compute",
            choices=["spi", "spei", "pnp", "scaled", "pet", "palmers", "all"],
            required=True,
        )
        parser.add_argument(
            "--periodicity",
            help="Process input as either monthly or daily values",
            choices=[compute.Periodicity.monthly, compute.Periodicity.daily],
            type=compute.Periodicity.from_string,
            required=True,
        )
        parser.add_argument(
            "--scales",
            help="Timestep scales over which the PNP, SPI, and SPEI values are to be computed",
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
            help="path to input NetCDF file (to be read) "
                 "containing distribution fitting parameters",
        )
        parser.add_argument(
            "--save_params",
            type=str,
            required=False,
            help="path to output NetCDF file (to be written) "
                 "containing distribution fitting parameters",
        )
        arguments = parser.parse_args()

        # validate the arguments and determine the input type
        input_type = _validate_args(arguments)

        if arguments.multiprocessing == "single":
            number_of_workers = 1
        elif arguments.multiprocessing == "all":
            number_of_workers = multiprocessing.cpu_count()
        else:  # default ("all_but_one")
            number_of_workers = multiprocessing.cpu_count() - 1

        # prepare precipitation NetCDF in case dimensions not (lat, lon, time) or if any coordinates are descending
        netcdf_precip = _prepare_file(arguments.netcdf_precip,
                                      arguments.var_name_precip)

        # run SPI computations for each scale/distribution in turn
        for scale in arguments.scales:
            for dist in indices.Distribution:

                # keyword arguments used for the SPI function
                kwrgs = {
                    "index": "spi",
                    "netcdf_precip": netcdf_precip,
                    "var_name_precip": arguments.var_name_precip,
                    "input_type": input_type,
                    "scale": scale,
                    "distribution": dist,
                    "periodicity": arguments.periodicity,
                    "calibration_start_year": arguments.calibration_start_year,
                    "calibration_end_year": arguments.calibration_end_year,
                    "output_file_base": arguments.output_file_base,
                    "load_params": arguments.load_params,
                    "save_params": arguments.save_params,
                    "number_of_workers": number_of_workers,
                }

                # compute and write SPI
                _compute_write_index(kwrgs)

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


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # (please do not remove -- useful for running as a script when debugging)
    #
    # Example command line usage for US climate divisions:
    #
    #  $ python climate_indices/__main__.py --index all --scales 1 2 3 6 9 12 24
    #  --netcdf_precip ../example_climate_indices/example/input/nclimdiv.nc
    #  --netcdf_temp ../example_climate_indices/example/input/nclimdiv.nc
    #  --netcdf_awc ../example_climate_indices/example/input/nclimdiv.nc
    #  --output_file_base /home/data/test/nclimdiv
    #  --var_name_precip prcp --var_name_temp tavg --var_name_awc awc
    #  --calibration_start_year 1951 --calibration_end_year 2010
    #  --multiprocessing all --periodicity monthly
    #
    #
    # Example command line usage for gridded data (nClimGrid):
    #
    #  $ python climate_indices/__main__.py --index all --scales 1 2 3 6 9 12 24
    #  --netcdf_precip ../example_climate_indices/example/input/nclimgrid_prcp.nc
    #  --netcdf_temp ../example_climate_indices/example/input/nclimgrid_tavg.nc
    #  --netcdf_awc ../example_climate_indices/example/input/nclimgrid_soil.nc
    #  --output_file_base /home/data/test/nclimgrid
    #  --var_name_precip prcp --var_name_temp tavg --var_name_awc awc
    #  --calibration_start_year 1951 --calibration_end_year 2010
    #  --multiprocessing all --periodicity monthly

    main()
