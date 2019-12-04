import argparse
from collections import Counter
from datetime import datetime
from enum import Enum
import logging
import multiprocessing
import os
from typing import List

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
def _drop_data_into_shared_arrays_grid(dataset: xr.Dataset,
                                       var_names: list,
                                       periodicity: compute.Periodicity,
                                       data_start_year: int):
    """
    Copies arrays from an xarray Dataset into shared memory arrays.

    :param dataset:
    :param var_names: names of variables to be copied into shared memory
    :param periodicity: monthly or daily
    :param data_start_year: initial year of the data
    """

    # get the data arrays we'll use later in the index computations
    global _global_shared_arrays
    expected_dims_3d = (
        ("lat", "lon", "time"),
        ("lon", "lat", "time"),
        ("lat", "lon", "month"),
        ("lon", "lat", "month"),
        ("lat", "lon", "day"),
        ("lon", "lat", "day"),
    )
    expected_dims_2d = (("lat", "lon"), ("lon", "lat"))
    expected_dims_1d = (("time",),)
    for var_name in var_names:

        # confirm that the dimensions of the data array are valid
        dims = dataset[var_name].dims
        if len(dims) == 3:
            if dims not in expected_dims_3d:
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
            initial_year = int(str(dataset["time"][0].data)[0:4])
            final_year = int(str(dataset["time"][-1].data)[0:4])
            total_years = final_year - initial_year + 1
            var_values = np.apply_along_axis(utils.transform_to_366day,
                                             len(dims) - 1,
                                             dataset[var_name].values,
                                             data_start_year,
                                             total_years)

        else:  # assumed to be monthly
            var_values = dataset[var_name].values

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
        dataset = dataset.drop(var_name)


# ------------------------------------------------------------------------------
def _drop_data_into_shared_arrays_divisions(
        dataset: xr.Dataset,
        var_names: List[str],
):
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
    dataset = xr.open_mfdataset(files, chunks=chunks)

    # trim out all data variables from the dataset except the ones we'll need
    input_var_names = []
    if "var_name_precip" in keyword_arguments:
        input_var_names.append(keyword_arguments["var_name_precip"])
    # only keep the latitude variable if we're dealing with divisions
    if input_type == InputType.divisions:
        input_var_names.append("lat")
    for var in dataset.data_vars:
        if var not in input_var_names:
            dataset = dataset.drop(var)

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
            fitting_coords = {"lat": dataset.lat, "lon": dataset.lon, time_coord_name: range(period_times)}
            data_shape = (len(dataset.lat), len(dataset.lon), period_times)
        elif input_type == InputType.divisions:
            fitting_coords = {"division": dataset.division, time_coord_name: range(period_times)}
            data_shape = (len(dataset.division), period_times)
        elif input_type == InputType.timeseries:
            fitting_coords = {time_coord_name: range(period_times)}
            data_shape = (period_times,)
        else:
            raise ValueError(f"Invalid 'input_type' keyword argument: {input_type}")

        # open the dataset if it's already been written to file, otherwise create it
        if os.path.exists(keyword_arguments["save_params"]):
            dataset_fitting = xr.open_dataset(keyword_arguments["save_params"])
        elif keyword_arguments["load_params"] is not None:
            dataset_fitting = xr.open_dataset(keyword_arguments["load_params"])
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
            global_attrs = {key: value for (key, value) in dataset.attrs.items() if key in attrs_to_copy}
            dataset_fitting = xr.Dataset(
                coords=fitting_coords,
                attrs=global_attrs,
            )

        if keyword_arguments["distribution"] == indices.Distribution.gamma:

            alpha_attrs = {
                'description': 'shape parameter of the gamma distribution (also referred to as the concentration) ' + \
                               f'computed from the {keyword_arguments["scale"]}-month scaled precipitation values',
            }
            alpha_var_name = f"alpha_{str(keyword_arguments['scale']).zfill(2)}"
            da_alpha = xr.DataArray(
                data=np.full(shape=data_shape, fill_value=np.NaN),
                coords=fitting_coords,
                dims=tuple(fitting_coords.keys()),
                name=alpha_var_name,
                attrs=alpha_attrs,
            )
            beta_attrs = {
                'description': '1 / scale of the distribution (also referred to as the rate) ' + \
                               f'computed from the {keyword_arguments["scale"]}-month scaled precipitation values',
            }
            beta_var_name = f"beta_{str(keyword_arguments['scale']).zfill(2)}"
            da_beta = xr.DataArray(
                data=np.full(shape=data_shape, fill_value=np.NaN),
                coords=fitting_coords,
                dims=tuple(fitting_coords.keys()),
                name=beta_var_name,
                attrs=beta_attrs,
            )
            dataset_fitting[alpha_var_name] = da_alpha
            dataset_fitting[beta_var_name] = da_beta

        elif keyword_arguments["distribution"] == indices.Distribution.pearson:

            prob_zero_attrs = {
                'description': 'probability of zero values within calibration period',
            }
            prob_zero_var_name = f"prob_zero_{keyword_arguments['scale']}_{keyword_arguments['periodicity'].unit()}"
            da_prob_zero = xr.DataArray(
                data=np.full(shape=data_shape, fill_value=np.NaN),
                coords=fitting_coords,
                dims=tuple(fitting_coords.keys()),
                name=prob_zero_var_name,
                attrs=prob_zero_attrs,
            )
            dataset_fitting[prob_zero_var_name] = da_prob_zero
            # TODO add DataArrays for scale, skew, and loc variables

    # get the initial year of the data
    data_start_year = int(str(dataset["time"].values[0])[0:4])
    keyword_arguments["data_start_year"] = data_start_year

    # the shape of output variables is assumed to match that of the input,
    # so use either precipitation or temperature variable's shape
    if "var_name_precip" in keyword_arguments:
        output_dims = dataset[keyword_arguments["var_name_precip"]].dims
    else:
        raise ValueError("Unable to determine output dimensions, no "
                         "precipitation variable name was specified.")

    # convert data into the appropriate units, if necessary
    # precipitation and PET should be in millimeters
    if "var_name_precip" in keyword_arguments:
        precip_var_name = keyword_arguments["var_name_precip"]
        precip_unit = dataset[precip_var_name].units.lower()
        if precip_unit not in ("mm", "millimeters", "millimeter", "mm/dy"):
            if precip_unit in ("inches", "inch"):
                # inches to mm conversion (1 inch == 25.4 mm)
                dataset[precip_var_name].values *= 25.4
            else:
                raise ValueError(f"Unsupported precipitation units: {precip_unit}")

    if input_type == InputType.divisions:
        _drop_data_into_shared_arrays_divisions(dataset, input_var_names)
    else:
        _drop_data_into_shared_arrays_grid(
            dataset,
            input_var_names,
            keyword_arguments["periodicity"],
            keyword_arguments["data_start_year"],
        )

    # use the temperature shape if computing PET, otherwise precipitation
    if keyword_arguments["index"] == "pet":
        output_shape = dataset[keyword_arguments["var_name_temp"]].shape
    else:
        output_shape = dataset[keyword_arguments["var_name_precip"]].shape

    # build an arguments dictionary appropriate to the index we'll compute
    args = _build_arguments(keyword_arguments)

    # add an array to hold results to the dictionary of arrays
    if _KEY_RESULT not in _global_shared_arrays:
        _global_shared_arrays[_KEY_RESULT] = {
            _KEY_ARRAY: multiprocessing.Array("d", int(np.prod(output_shape))),
            _KEY_SHAPE: output_shape,
        }

    # apply the SPI function along the time axis (axis=2)
    _parallel_process(
        keyword_arguments["index"],
        _global_shared_arrays,
        {"var_name_precip": keyword_arguments["var_name_precip"]},
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

    # create a new variable to contain the index values, assign into the dataset
    variable = xr.Variable(dims=output_dims,
                           data=index_values,
                           attrs=output_var_attributes)
    dataset[output_var_name] = variable

    # TODO set global attributes accordingly for this new dataset

    # remove all data variables except for the new variable
    for var_name in dataset.data_vars:
        if var_name != output_var_name:
            dataset = dataset.drop(var_name)

    # write the dataset as NetCDF
    netcdf_file_name = \
        keyword_arguments["output_file_base"] + "_" + output_var_name + ".nc"
    dataset.to_netcdf(netcdf_file_name)

    return netcdf_file_name, output_var_name


# ------------------------------------------------------------------------------
def _compute_write_fittings(keyword_arguments):
    """
    Computes a distribution fitting parameters and writes the result into a
    corresponding NetCDF.

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
    dataset = xr.open_mfdataset(files, chunks=chunks)

    # add placeholder variables in the Dataset to hold the fitting parameters we'll compute
    if ("save_params" in keyword_arguments) and \
            (keyword_arguments["save_params"] is not None) and \
            (keyword_arguments["index"] not in ("pet", "palmers")):

        if keyword_arguments["periodicity"] == compute.Periodicity.monthly:
            period_times = 12
            time_coord_name = "month"
        elif keyword_arguments["periodicity"] == compute.Periodicity.daily:
            period_times = 366
            time_coord_name = "day"
        else:
            raise ValueError(f"Unsupported periodicity: {keyword_arguments['periodicity']}")

        if input_type == InputType.grid:
            gamma_coords = {"lat": dataset.lat, "lon": dataset.lon, time_coord_name: range(period_times)}
            data_shape = (len(dataset.lat), len(dataset.lon), period_times)
        elif input_type == InputType.divisions:
            gamma_coords = {"division": dataset.division, time_coord_name: range(period_times)}
            data_shape = (len(dataset.division), period_times)
        elif input_type == InputType.timeseries:
            gamma_coords = {time_coord_name: range(period_times)}
            data_shape = (period_times,)
        else:
            raise ValueError(f"Invalid 'input_type' keyword argument: {input_type}")

        alpha_attrs = {
            'description': 'shape parameter of the gamma distribution (also referred to as the concentration) ' + \
                           f'computed from the {keyword_arguments["scale"]}-month scaled precipitation values',
        }
        alpha_var_name = f"alpha_{str(keyword_arguments['scale']).zfill(2)}"
        da_alpha = xr.DataArray(
            data=np.full(shape=data_shape, fill_value=np.NaN),
            coords=gamma_coords,
            dims=tuple(gamma_coords.keys()),
            name=alpha_var_name,
            attrs=alpha_attrs,
        )
        beta_attrs = {
            'description': '1 / scale of the distribution (also referred to as the rate) ' + \
                           f'computed from the {keyword_arguments["scale"]}-month scaled precipitation values',
        }
        beta_var_name = f"beta_{str(keyword_arguments['scale']).zfill(2)}"
        da_beta = xr.DataArray(
            data=np.full(shape=data_shape, fill_value=np.NaN),
            coords=gamma_coords,
            dims=tuple(gamma_coords.keys()),
            name=beta_var_name,
            attrs=beta_attrs,
        )
        dataset[alpha_var_name] = da_alpha
        dataset[beta_var_name] = da_beta

    # trim out all data variables from the dataset except the ones we'll need
    input_var_names = []
    if "var_name_precip" in keyword_arguments:
        input_var_names.append(keyword_arguments["var_name_precip"])
    # keep the parameter fitting variables if relevant
    if (("load_params" in keyword_arguments) or \
            ("save_params" in keyword_arguments)) and \
            (keyword_arguments["index"] not in ("pet", "palmers")):
        for fitting_param_name in _FITTING_PARAMETER_VARIABLES:
            fitting_param_var_name = "_".join((fitting_param_name, str(keyword_arguments["scale"]).zfill(2)))
            input_var_names.append(fitting_param_var_name)
    # only keep the latitude variable if we're dealing with divisions
    if input_type == InputType.divisions:
        input_var_names.append("lat")
    for var in dataset.data_vars:
        if var not in input_var_names:
            dataset = dataset.drop(var)

    # get the initial year of the data
    data_start_year = int(str(dataset["time"].values[0])[0:4])
    keyword_arguments["data_start_year"] = data_start_year

    # the shape of output variables is assumed to match that of the input,
    # so use either precipitation or temperature variable's shape
    if "var_name_precip" in keyword_arguments:
        output_dims = dataset[keyword_arguments["var_name_precip"]].dims
    else:
        raise ValueError("Unable to determine output dimensions, no "
                         "precipitation variable name was specified.")

    # convert data into the appropriate units, if necessary
    # precipitation and PET should be in millimeters
    if "var_name_precip" in keyword_arguments:
        precip_var_name = keyword_arguments["var_name_precip"]
        precip_unit = dataset[precip_var_name].units.lower()
        if precip_unit not in ("mm", "millimeters", "millimeter", "mm/dy"):
            if precip_unit in ("inches", "inch"):
                # inches to mm conversion (1 inch == 25.4 mm)
                dataset[precip_var_name].values *= 25.4
            else:
                raise ValueError(f"Unsupported precipitation units: {precip_unit}")

    if input_type == InputType.divisions:
        _drop_data_into_shared_arrays_divisions(dataset, input_var_names)
    else:
        _drop_data_into_shared_arrays_grid(
            dataset,
            input_var_names,
            keyword_arguments["periodicity"],
            keyword_arguments["data_start_year"],
        )

    # use the precipitation shape as the corresponding output array shape
    output_shape = dataset[keyword_arguments["var_name_precip"]].shape

    # build an arguments dictionary appropriate to the index we'll compute
    args = _build_arguments(keyword_arguments)

    # add output variable arrays into the shared memory arrays dictionary
    if _KEY_RESULT not in _global_shared_arrays:
        _global_shared_arrays[_KEY_RESULT] = {
            _KEY_ARRAY: multiprocessing.Array("d", int(np.prod(output_shape))),
            _KEY_SHAPE: output_shape,
        }

    # apply the SPI function along the time axis (axis=2)
    _parallel_process(
        keyword_arguments["index"],
        _global_shared_arrays,
        {"var_name_precip": keyword_arguments["var_name_precip"]},
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

    # create a new variable to contain the index values, assign into the dataset
    variable = xr.Variable(dims=output_dims,
                           data=index_values,
                           attrs=output_var_attributes)
    dataset[output_var_name] = variable

    # TODO set global attributes accordingly for this new dataset

    # remove all data variables except for the new variable
    for var_name in dataset.data_vars:
        if var_name != output_var_name:
            dataset = dataset.drop(var_name)

    # write the dataset as NetCDF
    netcdf_file_name = \
        keyword_arguments["output_file_base"] + "_" + output_var_name + ".nc"
    dataset.to_netcdf(netcdf_file_name)

    return netcdf_file_name, output_var_name


# ------------------------------------------------------------------------------
def _spi(precips, parameters):

    return indices.spi(values=precips,
                       scale=parameters["scale"],
                       distribution=parameters["distribution"],
                       data_start_year=parameters["data_start_year"],
                       calibration_year_initial=parameters["calibration_year_initial"],
                       calibration_year_final=parameters["calibration_year_final"],
                       periodicity=parameters["periodicity"])


# ------------------------------------------------------------------------------
def _init_worker(shared_arrays_dict):

    global _global_shared_arrays
    _global_shared_arrays = shared_arrays_dict


# ------------------------------------------------------------------------------
def _parallel_process(
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
    func1d = _spi

    # we have a single input array, create parameter dictionary objects
    # appropriate to the _apply_along_axis function, one per worker process
    for i in range(number_of_workers):
        params = {
            "index": index,
            "func1d": func1d,
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

        chunk_params.append(params)

    # instantiate a process pool
    with multiprocessing.Pool(processes=number_of_workers,
                              initializer=_init_worker,
                              initargs=(arrays_dict,)) as pool:

        pool.map(_apply_along_axis, chunk_params)


# ------------------------------------------------------------------------------
def _apply_along_axis(params):
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
    func1d = params["func1d"]
    start_index = params["sub_array_start"]
    end_index = params["sub_array_end"]
    array = _global_shared_arrays[params["input_var_name"]][_KEY_ARRAY]
    shape = _global_shared_arrays[params["input_var_name"]][_KEY_SHAPE]
    np_array = np.frombuffer(array.get_obj()).reshape(shape)
    sub_array = np_array[start_index:end_index]
    args = params["args"]

    if params["input_type"] == InputType.grid:
        axis_index = 2
    elif params["input_type"] == InputType.divisions:
        axis_index = 1
    elif params["input_type"] == InputType.timeseries:
        axis_index = 0
    else:
        raise ValueError(f"Invalid input type argument: {params['input_type']}")

    computed_array = np.apply_along_axis(func1d,
                                         axis=axis_index,
                                         arr=sub_array,
                                         parameters=args)

    output_array = _global_shared_arrays[params["output_var_name"]][_KEY_ARRAY]
    np_output_array = np.frombuffer(output_array.get_obj()).reshape(shape)
    np.copyto(np_output_array[start_index:end_index], computed_array)


# ------------------------------------------------------------------------------
def _apply_along_axis_double(params):
    """
    Like numpy.apply_along_axis(), but with arguments in a dict instead.
    Applicable for applying a function across subarrays of two input arrays.

    This function is useful with multiprocessing.Pool().map(): (1) map() only
    handles functions that take a single argument, and (2) this function can
    generally be imported from a module, as required by map().

    :param dict params: dictionary of parameters including a function name,
        "func1d", start and stop indices for specifying the subarray to which
        the function should be applied, "sub_array_start" and "sub_array_end",
        a dictionary of arguments to be passed to the function, "args", and
        the key name of the shared array for output values, "output_var_name".
    """

    func1d = params["func1d"]
    start_index = params["sub_array_start"]
    end_index = params["sub_array_end"]
    if params["index"] == "pet":
        first_array_key = params["var_name_temp"]
        second_array_key = params["var_name_lat"]
    elif params["index"] == "spei":
        first_array_key = params["var_name_precip"]
        second_array_key = params["var_name_pet"]
    else:
        raise ValueError("Unsupported index: {index}".format(index=params["index"]))

    shape = _global_shared_arrays[params["output_var_name"]][_KEY_SHAPE]
    first_array = _global_shared_arrays[first_array_key][_KEY_ARRAY]
    first_np_array = np.frombuffer(first_array.get_obj()).reshape(shape)
    sub_array_1 = first_np_array[start_index:end_index]
    if params["index"] == "pet":
        second_array = _global_shared_arrays[second_array_key][_KEY_ARRAY]
        second_np_array = np.frombuffer(second_array.get_obj()).reshape(shape[0])
    else:
        second_array = _global_shared_arrays[second_array_key][_KEY_ARRAY]
        second_np_array = np.frombuffer(second_array.get_obj()).reshape(shape)
    sub_array_2 = second_np_array[start_index:end_index]

    # get the output shared memory array, convert to numpy, and get the subarray slice
    output_array = _global_shared_arrays[params["output_var_name"]][_KEY_ARRAY]
    computed_array = np.frombuffer(output_array.get_obj()).reshape(shape)[
        start_index:end_index
    ]

    for i, (x, y) in enumerate(zip(sub_array_1, sub_array_2)):
        if params["input_type"] == InputType.grid:
            for j in range(x.shape[0]):
                if params["index"] == "pet":
                    # array x is temperature (2-D) and array y is lat (1-D)
                    computed_array[i, j] = func1d(x[j], y, parameters=params["args"])
                else:
                    # array x is precipitation and array y is PET (both 2-D)
                    computed_array[i, j] = func1d(x[j], y[j], parameters=params["args"])
        elif params["input_type"] == InputType.divisions:
            computed_array[i] = func1d(x, y, parameters=params["args"])
        else:
            raise ValueError(f"Unsupported input type: \'{params['input_type']}\'")


# ------------------------------------------------------------------------------
def _apply_along_axis_gamma(params):
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
        "var_name_precip", "var_name_pet", and "var_name_awc", a dictionary
        of arguments to be passed to the function, "args", and the key name of
        the shared arrays for output values, "alpha_var_name" and "beta_var_name".
    """
    func1d = params["func1d"]
    start_index = params["sub_array_start"]
    end_index = params["sub_array_end"]
    precip_array_key = params["var_name_precip"]
    alpha_array_key = params["var_name_alpha"]
    beta_array_key = params["var_name_beta"]

    shape = _global_shared_arrays[params["output_var_name"]][_KEY_SHAPE]
    precip_array = _global_shared_arrays[precip_array_key][_KEY_ARRAY]
    precip_np_array = np.frombuffer(precip_array.get_obj()).reshape(shape)
    sub_array_precip = precip_np_array[start_index:end_index]

    args = params["args"]

    # get the output shared memory arrays, convert to numpy, and get the subarray slices
    alpha_output_array = _global_shared_arrays[alpha_array_key][_KEY_ARRAY]
    alpha_np_array = np.frombuffer(alpha_output_array.get_obj()).reshape(shape)
    sub_array_alpha = alpha_np_array[start_index:end_index]

    beta_output_array = _global_shared_arrays[beta_array_key][_KEY_ARRAY]
    beta_np_array = np.frombuffer(beta_output_array.get_obj()).reshape(shape)
    sub_array_beta = beta_np_array[start_index:end_index]

    for i, precip in enumerate(sub_array_precip):
        if params["input_type"] == InputType.grid:
            for j in range(precip.shape[0]):
                sub_array_alpha[i, j], sub_array_beta[i, j] = \
                    compute.gamma_parameters(precip[j], parameters=args)
        else:  # divisions
            sub_array_alpha[i], sub_array_beta[i] = \
                compute.gamma_parameters(precip, parameters=args)


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
