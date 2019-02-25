import argparse
from collections import Counter
from datetime import datetime
import logging
import multiprocessing
import os

from nco import Nco
import numpy as np
import scipy.constants
import xarray as xr

from climate_indices import compute, indices

# the number of worker processes we'll use for process pools
_NUMBER_OF_WORKER_PROCESSES = multiprocessing.cpu_count() - 1

# shared memory array dictionary keys
_KEY_ARRAY = "array"
_KEY_SHAPE = "shape"
_KEY_LAT = "lat"
_KEY_RESULT = "result_array"
_KEY_RESULT_SCPDSI = "result_array_scpdsi"
_KEY_RESULT_PDSI = "result_array_pdsi"
_KEY_RESULT_PHDI = "result_array_phdi"
_KEY_RESULT_PMDI = "result_array_pmdi"
_KEY_RESULT_ZINDEX = "result_array_zindex"

# global dictionary to contain shared arrays for use by worker processes
_global_shared_arrays = {}

# ------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console as standard error
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
)
_logger = logging.getLogger(__name__)


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
    Validate the processing settings to confirm that proper argument combinations have been provided.

    :param args: an arguments object of the type returned by argparse.ArgumentParser.parse_args()
    :raise ValueError: if one or more of the command line arguments is invalid
    """

    # the dimensions we expect to find for each data variable (precipitation, temperature, and/or PET)
    expected_dimensions = [("time", "division"), ("division", "time")]

    # all indices except PET require a precipitation variable
    if args.index != "pet":

        # make sure a precipitation file was specified
        if args.netcdf_precip is None:
            msg = "Missing the required precipitation file"
            _logger.error(msg)
            raise ValueError(msg)

        # make sure a precipitation variable name was specified
        if args.var_name_precip is None:
            message = "Missing precipitation variable name"
            _logger.error(message)
            raise ValueError(message)

        # validate the precipitation variable
        with xr.open_dataset(args.netcdf_precip) as dataset_precip:

            # make sure we have a valid precipitation variable name
            if args.var_name_precip not in dataset_precip.variables:
                message = "Invalid precipitation variable name: '{var}' ".format(
                    var=args.var_name_precip
                ) + "does not exist in precipitation file '{file}'".format(
                    file=args.netcdf_precip
                )
                _logger.error(message)
                raise ValueError(message)

            # verify that the precipitation variable's dimensions are in the expected order
            dimensions = dataset_precip[args.var_name_precip].dims
            if dimensions not in expected_dimensions:
                message = "Invalid dimensions of the precipitation variable: {dims}, ".format(
                    dims=dimensions
                ) + "(expected names and order: {dims})".format(
                    dims=expected_dimensions
                )
                _logger.error(message)
                raise ValueError(message)

            # get the sizes of the division and time coordinate variables
            divisions_precip = dataset_precip["division"].values[:]
            times_precip = dataset_precip["time"].values[:]

    else:

        # PET requires a temperature file
        if args.netcdf_temp is None:
            msg = "Missing the required temperature file argument"
            _logger.error(msg)
            raise ValueError(msg)
        # don't allow a daily periodicity for PET (yet, this will be
        # possible once we have Hargreaves or a daily Thornthwaite)
        if args.periodicity is not compute.Periodicity.monthly:
            msg = (
                "Invalid periodicity argument for PET: "
                + "'{period}' -- only monthly is supported".format(
                    period=args.periodicity
                )
            )
            _logger.error(msg)
            raise ValueError(msg)

    # SPEI and Palmer require temperature and latitude variables in order to compute PET
    if args.index in ["spei", "scaled", "palmers"]:

        if args.netcdf_temp is None:

            if args.netcdf_pet is None:
                msg = "Missing the required temperature or PET files, neither were provided"
                _logger.error(msg)
                raise ValueError(msg)

            # validate the PET file
            with xr.open_dataset(args.netcdf_pet) as dataset_pet:

                # make sure we have a valid temperature variable name
                if args.var_name_pet is None:
                    message = "Missing PET variable name"
                    _logger.error(message)
                    raise ValueError(message)
                elif args.var_name_pet not in dataset_pet.variables:
                    message = "Invalid PET variable name: '{var_name}' ".format(
                        var_name=args.var_name_pet
                    ) + "does not exist in PET file '{file}'".format(
                        file=args.netcdf_pet
                    )
                    _logger.error(message)
                    raise ValueError(message)

                # verify that the temperature variable's dimensions are as expected
                dimensions = dataset_pet[args.var_name_pet].dims
                if dimensions not in expected_dimensions:
                    message = "Invalid dimensions of the PET variable: {dims}, ".format(
                        dims=dimensions
                    ) + "(expected names and order: {dims})".format(
                        dims=expected_dimensions
                    )
                    _logger.error(message)
                    raise ValueError(message)

                # verify that the coordinate variables match with those of the precipitation dataset
                if not np.array_equal(divisions_precip, dataset_pet["division"][:]):
                    message = (
                        "Precipitation and PET variables contain non-matching divisions"
                    )
                    _logger.error(message)
                    raise ValueError(message)
                elif not np.array_equal(times_precip, dataset_pet["time"][:]):
                    message = (
                        "Precipitation and PET variables contain non-matching times"
                    )
                    _logger.error(message)
                    raise ValueError(message)

        elif args.netcdf_pet is not None:

            # we can't have both temperature and PET files specified, no way to determine which to use
            msg = "Both temperature and PET files were specified, only one of these should be provided"
            _logger.error(msg)
            raise ValueError(msg)

        else:

            # validate the temperature file
            with xr.open_dataset(args.netcdf_temp) as dataset_temp:

                # make sure we have a valid temperature variable name
                if args.var_name_temp is None:
                    message = "Missing temperature variable name"
                    _logger.error(message)
                    raise ValueError(message)
                elif args.var_name_temp not in dataset_temp.variables:
                    message = "Invalid temperature variable name: '{var}' does ".format(
                        var=args.var_name_temp
                    ) + "not exist in temperature file '{file}'".format(
                        file=args.netcdf_temp
                    )
                    _logger.error(message)
                    raise ValueError(message)

                # verify that the latitude variable's dimensions are as expected
                dimensions = dataset_temp[args.var_name_temp].dims
                if dimensions not in expected_dimensions:
                    message = "Invalid dimensions of the temperature variable: {dims}, ".format(
                        dims=dimensions
                    ) + "(expected names and order: {dims})".format(
                        dims=expected_dimensions
                    )
                    _logger.error(message)
                    raise ValueError(message)

                # verify that the coordinate variables match with those of the precipitation dataset
                if not np.array_equal(divisions_precip, dataset_temp["division"][:]):
                    message = "Precipitation and temperature variables contain non-matching divisions"
                    _logger.error(message)
                    raise ValueError(message)
                elif not np.array_equal(times_precip, dataset_temp["time"][:]):
                    message = "Precipitation and temperature variables contain non-matching times"
                    _logger.error(message)
                    raise ValueError(message)

        # Palmers requires an available water capacity file
        if args.index in ["palmers"]:

            if args.netcdf_awc is None:

                msg = "Missing the required available water capacity file"
                _logger.error(msg)
                raise ValueError(msg)

            # validate the AWC file
            with xr.open_dataset(args.netcdf_awc) as dataset_awc:

                # make sure we have a valid PET variable name
                if args.var_name_awc is None:
                    message = "Missing the AWC variable name"
                    _logger.error(message)
                    raise ValueError(message)
                elif args.var_name_awc not in dataset_awc.variables:
                    message = "Invalid AWC variable name: '{var}' does not exist ".format(
                        var=args.var_name_awc
                    ) + "in AWC file '{file}'".format(
                        file=args.netcdf_awc
                    )
                    _logger.error(message)
                    raise ValueError(message)

                # verify that the AWC variable's dimensions are in the expected order
                dimensions = dataset_awc[args.var_name_awc].dims
                if len(dimensions) == 2:
                    expected_dimensions = [("division",)]
                if dimensions not in expected_dimensions:
                    message = "Invalid dimensions of the AWC variable: {dims}, ".format(
                        dims=dimensions
                    ) + "(expected names and order: {dims})".format(
                        dims=expected_dimensions
                    )
                    _logger.error(message)
                    raise ValueError(message)

                # verify that the lat and lon coordinate variable values
                # match with those of the precipitation dataset
                if not np.array_equal(divisions_precip, dataset_awc["division"][:]):
                    message = (
                        "Precipitation and AWC variables contain non-matching divisions"
                    )
                    _logger.error(message)
                    raise ValueError(message)

    if args.index in ["spi", "spei", "scaled", "pnp"]:

        if args.scales is None:
            message = (
                "Scaled indices (SPI, SPEI, and/or PNP) specified without including "
                + "one or more time scales (missing --scales argument)"
            )
            _logger.error(message)
            raise ValueError(message)

        if any(n < 0 for n in args.scales):
            message = "One or more negative scale specified within --scales argument"
            _logger.error(message)
            raise ValueError(message)


# ------------------------------------------------------------------------------
def _get_scale_increment(args_dict):

    if args_dict["periodicity"] == compute.Periodicity.daily:
        scale_increment = "day"
    elif args_dict["periodicity"] == compute.Periodicity.monthly:
        scale_increment = "month"
    else:
        raise ValueError(
            "Invalid periodicity argument: {}".format(args_dict["periodicity"])
        )

    return scale_increment


# ------------------------------------------------------------------------------
def _log_status(args_dict):

    # get the scale increment for use in later log messages
    if "scale" in args_dict:

        if "distribution" in args_dict:

            _logger.info(
                "Computing {scale}-{incr} {index}/{dist}".format(
                    scale=args_dict["scale"],
                    incr=_get_scale_increment(args_dict),
                    index=args_dict["index"].upper(),
                    dist=args_dict["distribution"].value.capitalize(),
                )
            )

        else:

            _logger.info(
                "Computing {scale}-{incr} {index}".format(
                    scale=args_dict["scale"],
                    incr=_get_scale_increment(args_dict),
                    index=args_dict["index"].upper(),
                )
            )

    else:

        _logger.info("Computing {index}".format(index=args_dict["index"].upper()))

    return True


# ------------------------------------------------------------------------------
def _build_arguments(keyword_args):
    """
    Builds a dictionary of function arguments appropriate to the index to be computed.

    :param dict keyword_args:
    :return: dictionary of arguments keyed with names expected by the corresponding
        index computation function
    """

    function_arguments = {"data_start_year": keyword_args["data_start_year"]}

    if keyword_args["index"] in ["spi", "spei"]:
        function_arguments["scale"] = keyword_args["scale"]
        function_arguments["distribution"] = keyword_args["distribution"]
        function_arguments["calibration_year_initial"] = keyword_args[
            "calibration_start_year"
        ]
        function_arguments["calibration_year_final"] = keyword_args[
            "calibration_end_year"
        ]
        function_arguments["periodicity"] = keyword_args["periodicity"]

    elif keyword_args["index"] == "pnp":
        function_arguments["scale"] = keyword_args["scale"]
        function_arguments["calibration_start_year"] = keyword_args[
            "calibration_start_year"
        ]
        function_arguments["calibration_end_year"] = keyword_args[
            "calibration_end_year"
        ]
        function_arguments["periodicity"] = keyword_args["periodicity"]

    elif keyword_args["index"] == "palmers":
        function_arguments["calibration_start_year"] = keyword_args[
            "calibration_start_year"
        ]
        function_arguments["calibration_end_year"] = keyword_args[
            "calibration_end_year"
        ]

    elif keyword_args["index"] != "pet":
        raise ValueError(
            "Index {index} not yet supported.".format(index=keyword_args["index"])
        )

    return function_arguments


# ------------------------------------------------------------------------------
def _get_variable_attributes(args_dict):

    if args_dict["index"] == "spi":

        long_name = "Standardized Precipitation Index ({dist} distribution), ".format(
            dist=args_dict["distribution"].value.capitalize()
        ) + "{scale}-{increment}".format(
            scale=args_dict["scale"], increment=_get_scale_increment(args_dict)
        )
        attrs = {"long_name": long_name, "valid_min": -3.09, "valid_max": 3.09}
        var_name = (
            "spi_"
            + args_dict["distribution"].value
            + "_"
            + str(args_dict["scale"]).zfill(2)
        )

    elif args_dict["index"] == "spei":

        long_name = "Standardized Precipitation Evapotranspiration Index ({dist} distribution), ".format(
            dist=args_dict["distribution"].value.capitalize()
        ) + "{scale}-{increment}".format(
            scale=args_dict["scale"], increment=_get_scale_increment(args_dict)
        )
        attrs = {"long_name": long_name, "valid_min": -3.09, "valid_max": 3.09}
        var_name = (
            "spei_"
            + args_dict["distribution"].value
            + "_"
            + str(args_dict["scale"]).zfill(2)
        )

    elif args_dict["index"] == "pnp":

        long_name = (
            "Percentage of Normal Precipitation, "
            + "{scale}-{increment}".format(
                scale=args_dict["scale"], increment=_get_scale_increment(args_dict)
            )
        )
        attrs = {"long_name": long_name, "valid_min": -1000.0, "valid_max": 1000.0}
        var_name = "pnp_" + str(args_dict["scale"]).zfill(2)

    elif args_dict["index"] == "pet":

        long_name = "Potential Evapotranspiration (Thornthwaite)"
        attrs = {
            "long_name": long_name,
            "valid_min": 0.0,
            "valid_max": 10000.0,
            "units": "millimeters",
        }
        var_name = "pet_thornthwaite"

    else:

        raise ValueError("Unsupported index: {index}".format(index=args_dict["index"]))

    return var_name, attrs


# ------------------------------------------------------------------------------
def _compute_write_index(keyword_arguments):
    """
    Computes a climate index and writes the result into a corresponding NetCDF.

    :param keyword_arguments:
    :return:
    """

    _log_status(keyword_arguments)

    # open the input NetCDF file as an xarray DataSet object
    files = []
    if "netcdf_precip" in keyword_arguments:
        files.append(keyword_arguments["netcdf_precip"])
    if "netcdf_temp" in keyword_arguments:
        files.append(keyword_arguments["netcdf_temp"])
    if "netcdf_pet" in keyword_arguments:
        files.append(keyword_arguments["netcdf_pet"])
    dataset = xr.open_mfdataset(files, chunks={"division": -1})

    # trim out all data variables from the dataset except the ones we'll need
    input_var_names = []
    if "var_name_precip" in keyword_arguments:
        input_var_names.append(keyword_arguments["var_name_precip"])
    if "var_name_temp" in keyword_arguments:
        input_var_names.append(keyword_arguments["var_name_temp"])
    if "var_name_pet" in keyword_arguments:
        input_var_names.append(keyword_arguments["var_name_pet"])
    for var in dataset.data_vars:
        if var not in input_var_names:
            if var != "lat":
                dataset = dataset.drop(var)
    # get the initial year of the data
    data_start_year = int(str(dataset["time"].values[0])[0:4])
    keyword_arguments["data_start_year"] = data_start_year

    # the shape of output variables is assumed to match that of the input,
    # so use either precipitation or temperature variable's shape
    if "var_name_precip" in keyword_arguments:
        output_shape = dataset[keyword_arguments["var_name_precip"]].shape
        output_dims = dataset[keyword_arguments["var_name_precip"]].dims
    elif "var_name_temp" in keyword_arguments:
        output_shape = dataset[keyword_arguments["var_name_temp"]].shape
        output_dims = dataset[keyword_arguments["var_name_temp"]].dims
    else:
        raise ValueError(
            "Unable to determine output shape, no precipitation "
            "or temperature variable name was specified."
        )

    # convert data into the appropriate units, if necessary
    # temperature should be in degrees Celsius
    # precipitation and PET should be in millimeters
    if "var_name_precip" in keyword_arguments:
        precip_var_name = keyword_arguments["var_name_precip"]
        precip_unit = dataset[precip_var_name].units.lower()
        if precip_unit not in ("mm", "millimeters", "millimeter"):
            if precip_unit in ("inches", "inch"):
                # inches to mm conversion (1 inch == 25.4 mm)
                dataset[precip_var_name].values *= 25.4
            else:
                raise ValueError(
                    "Unsupported precipitation units: {var}".format(
                        var=dataset[precip_var_name].units
                    )
                )
    if "var_name_temp" in keyword_arguments:
        temp_var_name = keyword_arguments["var_name_temp"]
        temp_unit = dataset[temp_var_name].units.lower()
        if temp_unit not in ("degrees_celsius", "celsius", "c"):
            if temp_unit in ("f", "fahrenheit"):
                dataset[temp_var_name].values = scipy.constants.convert_temperature(
                    dataset[temp_var_name].values, "f", "c"
                )
            elif temp_unit in ("k", "kelvin"):
                dataset[temp_var_name].values = scipy.constants.convert_temperature(
                    dataset[temp_var_name].values, "k", "c"
                )
            else:
                raise ValueError(
                    "Unsupported temperature units: {var}".format(
                        var=dataset[temp_var_name].units
                    )
                )

    # get the data arrays we'll use later in the index computations
    global _global_shared_arrays
    expected_dims_2d = [("division", "time"), ("time", "division")]
    expected_dims_1d = ["division"]
    for var_name in input_var_names:

        # confirm that the dimensions of the data array are valid
        dims = dataset[var_name].dims
        if len(dims) == 2:
            if dims not in expected_dims_2d:
                message = "Invalid dimensions for variable '{var_name}': {dims}".format(
                    var_name=var_name, dims=dims
                )
                _logger.error(message)
                raise ValueError(message)
        elif len(dims) == 1:
            if dims not in expected_dims_1d:
                message = "Invalid dimensions for variable '{var_name}': {dims}".format(
                    var_name=var_name, dims=dims
                )
                _logger.error(message)
                raise ValueError(message)

        # create a shared memory array, wrap it as a numpy array and copy
        # copy the data (values) from this variable's DataArray
        shared_array = multiprocessing.Array("d", int(np.prod(dataset[var_name].shape)))
        shared_array_np = np.frombuffer(shared_array.get_obj()).reshape(
            dataset[var_name].shape
        )
        np.copyto(shared_array_np, dataset[var_name].values)

        # add to the dictionary of arrays
        _global_shared_arrays[var_name] = {
            _KEY_ARRAY: shared_array,
            _KEY_SHAPE: dataset[var_name].shape,
        }

        # drop the variable from the dataset (we're assuming this frees the memory)
        dataset = dataset.drop(var_name)

    # build an arguments dictionary appropriate to the index we'll compute
    args = _build_arguments(keyword_arguments)

    # add output variable arrays into the shared memory arrays dictionary
    if keyword_arguments["index"] == "palmers":

        # read AWC data into shared memory array
        if ("netcdf_awc" not in keyword_arguments) or (
            "var_name_awc" not in keyword_arguments
        ):
            raise ValueError("Missing the AWC file and/or variable name argument(s)")

        awc_dataset = xr.open_dataset(keyword_arguments["netcdf_awc"])

        # create a shared memory array, wrap it as a numpy array and copy
        # copy the data (values) from this variable's DataArray
        var_name = keyword_arguments["var_name_awc"]
        shared_array = multiprocessing.Array(
            "d", int(np.prod(awc_dataset[var_name].shape))
        )
        shared_array_np = np.frombuffer(shared_array.get_obj()).reshape(
            awc_dataset[var_name].shape
        )
        np.copyto(shared_array_np, awc_dataset[var_name].values)

        # add to the dictionary of arrays
        _global_shared_arrays[var_name] = {
            _KEY_ARRAY: shared_array,
            _KEY_SHAPE: awc_dataset[var_name].shape,
        }

        # add shared memory arrays for computed Palmers to the dictionary of shared arrays
        if _KEY_RESULT_SCPDSI not in _global_shared_arrays:
            _global_shared_arrays[_KEY_RESULT_SCPDSI] = {
                _KEY_ARRAY: multiprocessing.Array("d", int(np.prod(output_shape))),
                _KEY_SHAPE: output_shape,
            }
        if _KEY_RESULT_PDSI not in _global_shared_arrays:
            _global_shared_arrays[_KEY_RESULT_PDSI] = {
                _KEY_ARRAY: multiprocessing.Array("d", int(np.prod(output_shape))),
                _KEY_SHAPE: output_shape,
            }
        if _KEY_RESULT_PHDI not in _global_shared_arrays:
            _global_shared_arrays[_KEY_RESULT_PHDI] = {
                _KEY_ARRAY: multiprocessing.Array("d", int(np.prod(output_shape))),
                _KEY_SHAPE: output_shape,
            }
        if _KEY_RESULT_PMDI not in _global_shared_arrays:
            _global_shared_arrays[_KEY_RESULT_PMDI] = {
                _KEY_ARRAY: multiprocessing.Array("d", int(np.prod(output_shape))),
                _KEY_SHAPE: output_shape,
            }
        if _KEY_RESULT_ZINDEX not in _global_shared_arrays:
            _global_shared_arrays[_KEY_RESULT_ZINDEX] = {
                _KEY_ARRAY: multiprocessing.Array("d", int(np.prod(output_shape))),
                _KEY_SHAPE: output_shape,
            }

        # apply the Palmers function along the time axis (axis=2)
        _parallel_process(
            keyword_arguments["index"],
            _global_shared_arrays,
            {
                "var_name_precip": keyword_arguments["var_name_precip"],
                "var_name_pet": keyword_arguments["var_name_pet"],
                "var_name_awc": keyword_arguments["var_name_awc"],
            },
            _KEY_RESULT_SCPDSI,
            args,
        )

        # get the computed SCPDSI data as an array of float32 values
        array = _global_shared_arrays[_KEY_RESULT_SCPDSI][_KEY_ARRAY]
        shape = _global_shared_arrays[_KEY_RESULT_SCPDSI][_KEY_SHAPE]
        scpdsi = np.frombuffer(array.get_obj()).reshape(shape).astype(np.float32)

        # get the computed PDSI data as an array of float32 values
        array = _global_shared_arrays[_KEY_RESULT_PDSI][_KEY_ARRAY]
        shape = _global_shared_arrays[_KEY_RESULT_PDSI][_KEY_SHAPE]
        pdsi = np.frombuffer(array.get_obj()).reshape(shape).astype(np.float32)

        # get the computed PHDI data as an array of float32 values
        array = _global_shared_arrays[_KEY_RESULT_PHDI][_KEY_ARRAY]
        shape = _global_shared_arrays[_KEY_RESULT_PHDI][_KEY_SHAPE]
        phdi = np.frombuffer(array.get_obj()).reshape(shape).astype(np.float32)

        # get the computed PMDI data as an array of float32 values
        array = _global_shared_arrays[_KEY_RESULT_PMDI][_KEY_ARRAY]
        shape = _global_shared_arrays[_KEY_RESULT_PMDI][_KEY_SHAPE]
        pmdi = np.frombuffer(array.get_obj()).reshape(shape).astype(np.float32)

        # get the computed Z-Index data as an array of float32 values
        array = _global_shared_arrays[_KEY_RESULT_ZINDEX][_KEY_ARRAY]
        shape = _global_shared_arrays[_KEY_RESULT_ZINDEX][_KEY_SHAPE]
        zindex = np.frombuffer(array.get_obj()).reshape(shape).astype(np.float32)

        # create a new variable to contain the SCPDSI values, assign into the dataset
        long_name = "Self-calibrated Palmer Drought Severity Index"
        scpdsi_attrs = {"long_name": long_name, "valid_min": -10.0, "valid_max": 10.0}
        var_name_scpdsi = "scpdsi"
        scpdsi_var = xr.Variable(dims=output_dims, data=scpdsi, attrs=scpdsi_attrs)
        dataset[var_name_scpdsi] = scpdsi_var

        # remove all data variables except for the new SCPDSI variable
        for var_name in dataset.data_vars:
            if var_name != var_name_scpdsi:
                dataset = dataset.drop(var_name)

        # TODO set global attributes accordingly for this new dataset

        # write the dataset as NetCDF
        netcdf_file_name = (
            keyword_arguments["output_file_base"] + "_" + var_name_scpdsi + ".nc"
        )
        dataset.to_netcdf(netcdf_file_name)

        # create a new variable to contain the PDSI values, assign into the dataset
        long_name = "Palmer Drought Severity Index"
        pdsi_attrs = {"long_name": long_name, "valid_min": -10.0, "valid_max": 10.0}
        var_name_pdsi = "pdsi"
        pdsi_var = xr.Variable(dims=output_dims, data=pdsi, attrs=pdsi_attrs)
        dataset[var_name_pdsi] = pdsi_var

        # remove all data variables except for the new PDSI variable
        for var_name in dataset.data_vars:
            if var_name != var_name_pdsi:
                dataset = dataset.drop(var_name)

        # TODO set global attributes accordingly for this new dataset

        # write the dataset as NetCDF
        netcdf_file_name = (
            keyword_arguments["output_file_base"] + "_" + var_name_pdsi + ".nc"
        )
        dataset.to_netcdf(netcdf_file_name)

        # create a new variable to contain the PHDI values, assign into the dataset
        long_name = "Palmer Hydrological Drought Index"
        phdi_attrs = {"long_name": long_name, "valid_min": -10.0, "valid_max": 10.0}
        var_name_phdi = "phdi"
        phdi_var = xr.Variable(dims=output_dims, data=phdi, attrs=phdi_attrs)
        dataset[var_name_phdi] = phdi_var

        # remove all data variables except for the new PHDI variable
        for var_name in dataset.data_vars:
            if var_name != var_name_phdi:
                dataset = dataset.drop(var_name)

        # TODO set global attributes accordingly for this new dataset

        # write the dataset as NetCDF
        netcdf_file_name = (
            keyword_arguments["output_file_base"] + "_" + var_name_phdi + ".nc"
        )
        dataset.to_netcdf(netcdf_file_name)

        # create a new variable to contain the PMDI values, assign into the dataset
        long_name = "Palmer Modified Drought Index"
        pmdi_attrs = {"long_name": long_name, "valid_min": -10.0, "valid_max": 10.0}
        var_name_pmdi = "pmdi"
        pmdi_var = xr.Variable(dims=output_dims, data=pmdi, attrs=pmdi_attrs)
        dataset[var_name_pmdi] = pmdi_var

        # remove all data variables except for the new PMDI variable
        for var_name in dataset.data_vars:
            if var_name != var_name_pmdi:
                dataset = dataset.drop(var_name)

        # TODO set global attributes accordingly for this new dataset

        # write the dataset as NetCDF
        netcdf_file_name = (
            keyword_arguments["output_file_base"] + "_" + var_name_pmdi + ".nc"
        )
        dataset.to_netcdf(netcdf_file_name)

        # create a new variable to contain the Z-Index values, assign into the dataset
        long_name = "Palmer Z-Index"
        zindex_attrs = {"long_name": long_name, "valid_min": -10.0, "valid_max": 10.0}
        var_name_zindex = "zindex"
        zindex_var = xr.Variable(dims=output_dims, data=zindex, attrs=zindex_attrs)
        dataset[var_name_zindex] = zindex_var

        # remove all data variables except for the new Z-Index variable
        for var_name in dataset.data_vars:
            if var_name != var_name_zindex:
                dataset = dataset.drop(var_name)

        # TODO set global attributes accordingly for this new dataset

        # write the dataset as NetCDF
        netcdf_file_name = (
            keyword_arguments["output_file_base"] + "_" + var_name_zindex + ".nc"
        )
        dataset.to_netcdf(netcdf_file_name)

    else:

        # add an array to hold results to the dictionary of arrays
        if _KEY_RESULT not in _global_shared_arrays:
            _global_shared_arrays[_KEY_RESULT] = {
                _KEY_ARRAY: multiprocessing.Array("d", int(np.prod(output_shape))),
                _KEY_SHAPE: output_shape,
            }

        if keyword_arguments["index"] in ["spi", "pnp"]:

            # apply the SPI function along the time axis (axis=2)
            _parallel_process(
                keyword_arguments["index"],
                _global_shared_arrays,
                {"var_name_precip": keyword_arguments["var_name_precip"]},
                _KEY_RESULT,
                args,
            )

        elif keyword_arguments["index"] == "spei":

            # apply the SPEI function along the time axis (axis=2)
            _parallel_process(
                keyword_arguments["index"],
                _global_shared_arrays,
                {
                    "var_name_precip": keyword_arguments["var_name_precip"],
                    "var_name_pet": keyword_arguments["var_name_pet"],
                },
                _KEY_RESULT,
                args,
            )

        elif keyword_arguments["index"] == "pet":

            # create a shared memory array, wrap it as a numpy array and copy
            # copy the data (values) from this variable's DataArray

            da_lat = dataset["lat"]
            shared_array = multiprocessing.Array("d", int(np.prod(da_lat.shape)))
            shared_array_np = np.frombuffer(shared_array.get_obj()).reshape(
                da_lat.shape
            )
            np.copyto(shared_array_np, da_lat.values)

            # add to the dictionary of arrays
            _global_shared_arrays[_KEY_LAT] = {
                _KEY_ARRAY: shared_array,
                _KEY_SHAPE: da_lat.shape,
            }

            # apply the PET function along the time axis (axis=2)
            _parallel_process(
                keyword_arguments["index"],
                _global_shared_arrays,
                {
                    "var_name_temp": keyword_arguments["var_name_temp"],
                    "var_name_lat": _KEY_LAT,
                },
                _KEY_RESULT,
                args,
            )

        else:
            raise ValueError(
                "Unsupported index: '{index}'".format(index=keyword_arguments["index"])
            )

        # get the name and attributes to use for the index variable in the output NetCDF
        output_var_name, output_var_attributes = _get_variable_attributes(
            keyword_arguments
        )

        # get the shared memory results array and convert it to a numpy array
        array = _global_shared_arrays[_KEY_RESULT][_KEY_ARRAY]
        shape = _global_shared_arrays[_KEY_RESULT][_KEY_SHAPE]
        index_values = np.frombuffer(array.get_obj()).reshape(shape).astype(np.float32)

        # create a new variable to contain the index values, assign into the dataset
        variable = xr.Variable(
            dims=output_dims, data=index_values, attrs=output_var_attributes
        )
        dataset[output_var_name] = variable

        # TODO set global attributes accordingly for this new dataset

        # remove all data variables except for the new variable
        for var_name in dataset.data_vars:
            if var_name != output_var_name:
                dataset = dataset.drop(var_name)

        # write the dataset as NetCDF
        netcdf_file_name = (
            keyword_arguments["output_file_base"] + "_" + output_var_name + ".nc"
        )
        dataset.to_netcdf(netcdf_file_name)

        return netcdf_file_name, output_var_name


# ------------------------------------------------------------------------------
def _pet(temperatures, latitude, parameters):
    return indices.pet(
        temperature_celsius=temperatures,
        latitude_degrees=latitude,
        data_start_year=parameters["data_start_year"],
    )


# ------------------------------------------------------------------------------
def _spi(precips, parameters):

    return indices.spi(
        values=precips,
        scale=parameters["scale"],
        distribution=parameters["distribution"],
        data_start_year=parameters["data_start_year"],
        calibration_year_initial=parameters["calibration_year_initial"],
        calibration_year_final=parameters["calibration_year_final"],
        periodicity=parameters["periodicity"],
    )


# ------------------------------------------------------------------------------
def _spei(precips, pet_mm, parameters):

    return indices.spei(
        precips_mm=precips,
        pet_mm=pet_mm,
        scale=parameters["scale"],
        distribution=parameters["distribution"],
        data_start_year=parameters["data_start_year"],
        calibration_year_initial=parameters["calibration_year_initial"],
        calibration_year_final=parameters["calibration_year_final"],
        periodicity=parameters["periodicity"],
    )


# ------------------------------------------------------------------------------
def _palmers(precips, pet_mm, awc, parameters):

    return indices.scpdsi(
        precip_time_series=precips,
        pet_time_series=pet_mm,
        awc=awc,
        data_start_year=parameters["data_start_year"],
        calibration_start_year=parameters["calibration_start_year"],
        calibration_end_year=parameters["calibration_end_year"],
    )


# ------------------------------------------------------------------------------
def _pnp(precips, parameters):

    return indices.percentage_of_normal(
        precips,
        scale=parameters["scale"],
        data_start_year=parameters["data_start_year"],
        calibration_start_year=parameters["calibration_start_year"],
        calibration_end_year=parameters["calibration_end_year"],
        periodicity=parameters["periodicity"],
    )


# ------------------------------------------------------------------------------
def _init_worker(shared_arrays_dict):

    global _global_shared_arrays
    _global_shared_arrays = shared_arrays_dict


# ------------------------------------------------------------------------------
def _parallel_process(index, arrays_dict, input_var_names, output_var_name, args):
    """
    TODO document this function
    :param index:
    :param arrays_dict:
    :param input_var_names:
    :param output_var_name:
    :param args:
    :return:
    """

    # find the start index of each sub-array we'll split out per worker process,
    # assuming the shape of the output array is the same as all input arrays
    shape = arrays_dict[output_var_name][_KEY_SHAPE]
    d, m = divmod(shape[0], _NUMBER_OF_WORKER_PROCESSES)
    split_indices = list(range(0, ((d + 1) * (m + 1)), (d + 1)))
    if d != 0:
        split_indices += list(range(split_indices[-1] + d, shape[0], d))

    # build a list of parameters for each application of the function to an array chunk
    chunk_params = []
    if index in ["spi", "pnp"]:

        if index == "spi":
            func1d = _spi
        else:
            func1d = _pnp

        # we have a single input array, create parameter dictionary objects
        # appropriate to the _apply_along_axis function, one per worker process
        for i in range(_NUMBER_OF_WORKER_PROCESSES):
            params = {
                "index": index,
                "func1d": func1d,
                "input_var_name": input_var_names["var_name_precip"],
                "output_var_name": output_var_name,
                "sub_array_start": split_indices[i],
                "args": args,
            }
            if i < (_NUMBER_OF_WORKER_PROCESSES - 1):
                params["sub_array_end"] = split_indices[i + 1]
            else:
                params["sub_array_end"] = None

            chunk_params.append(params)

    elif index == "spei":

        # we have two input arrays, create parameter dictionary objects
        # appropriate to the _apply_along_axis_double function, one per worker process
        for i in range(_NUMBER_OF_WORKER_PROCESSES):
            params = {
                "index": index,
                "func1d": _spei,
                "var_name_precip": input_var_names["var_name_precip"],
                "var_name_pet": input_var_names["var_name_pet"],
                "output_var_name": output_var_name,
                "sub_array_start": split_indices[i],
                "args": args,
            }
            if i < (_NUMBER_OF_WORKER_PROCESSES - 1):
                params["sub_array_end"] = split_indices[i + 1]
            else:
                params["sub_array_end"] = None

            chunk_params.append(params)

    elif index == "pet":

        # we have two input arrays, create parameter dictionary objects
        # appropriate to the _apply_along_axis_double function, one per worker process
        for i in range(_NUMBER_OF_WORKER_PROCESSES):
            params = {
                "index": index,
                "func1d": _pet,
                "var_name_temp": input_var_names["var_name_temp"],
                "var_name_lat": input_var_names["var_name_lat"],
                "output_var_name": output_var_name,
                "sub_array_start": split_indices[i],
                "args": args,
            }
            if i < (_NUMBER_OF_WORKER_PROCESSES - 1):
                params["sub_array_end"] = split_indices[i + 1]
            else:
                params["sub_array_end"] = None

            chunk_params.append(params)

    elif index == "palmers":

        # create parameter dictionary objects appropriate to
        # the _apply_along_axis_palmer function, one per worker process
        for i in range(_NUMBER_OF_WORKER_PROCESSES):
            params = {
                "index": index,
                "func1d": _palmers,
                "var_name_precip": input_var_names["var_name_precip"],
                "var_name_pet": input_var_names["var_name_pet"],
                "var_name_awc": input_var_names["var_name_awc"],
                "output_var_name": output_var_name,
                "sub_array_start": split_indices[i],
                "args": args,
            }
            if i < (_NUMBER_OF_WORKER_PROCESSES - 1):
                params["sub_array_end"] = split_indices[i + 1]
            else:
                params["sub_array_end"] = None

            chunk_params.append(params)

    else:
        raise ValueError("Unsupported index: {index}".format(index=index))

    # instantiate a process pool
    with multiprocessing.Pool(
        processes=_NUMBER_OF_WORKER_PROCESSES,
        initializer=_init_worker,
        initargs=(arrays_dict,),
    ) as pool:

        if index in ["spei", "pet"]:
            pool.map(_apply_along_axis_double, chunk_params)
        elif index == "palmers":
            pool.map(_apply_along_axis_palmers, chunk_params)
        else:
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

    computed_array = np.apply_along_axis(func1d, axis=1, arr=sub_array, parameters=args)

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
        computed_array[i] = func1d(x, y, parameters=params["args"])


# ------------------------------------------------------------------------------
def _apply_along_axis_palmers(params):
    """
    Applies the Palmer computation function across subarrays of
    the Palmer-specific input (shared-memory) arrays.

    This function is useful with multiprocessing.Pool().map(): (1) map() only
    handles functions that take a single argument, and (2) this function can
    generally be imported from a module, as required by map().

    :param dict params: dictionary of parameters including a function name,
        "func1d", start and stop indices for specifying the subarray to which
        the function should be applied, "sub_array_start" and "sub_array_end",
        the variable names used for precipitation, PET, and AWC arrays,
        "var_name_precip", "var_name_pet", and "var_name_awc", a dictionary
        of arguments to be passed to the function, "args", and the key name of
        the shared array for output values, "output_var_name".
    """
    func1d = params["func1d"]
    start_index = params["sub_array_start"]
    end_index = params["sub_array_end"]
    precip_array_key = params["var_name_precip"]
    pet_array_key = params["var_name_pet"]
    awc_array_key = params["var_name_awc"]

    shape = _global_shared_arrays[params["output_var_name"]][_KEY_SHAPE]
    precip_array = _global_shared_arrays[precip_array_key][_KEY_ARRAY]
    precip_np_array = np.frombuffer(precip_array.get_obj()).reshape(shape)
    sub_array_precip = precip_np_array[start_index:end_index]
    pet_array = _global_shared_arrays[pet_array_key][_KEY_ARRAY]
    pet_np_array = np.frombuffer(pet_array.get_obj()).reshape(shape)
    sub_array_pet = pet_np_array[start_index:end_index]
    awc_array = _global_shared_arrays[awc_array_key][_KEY_ARRAY]
    awc_np_array = np.frombuffer(awc_array.get_obj()).reshape(shape[0])
    sub_array_awc = awc_np_array[start_index:end_index]

    args = params["args"]

    # get the output shared memory arrays, convert to numpy, and get the subarray slices
    scpdsi_output_array = _global_shared_arrays[_KEY_RESULT_SCPDSI][_KEY_ARRAY]
    scpdsi = np.frombuffer(scpdsi_output_array.get_obj()).reshape(shape)[
        start_index:end_index
    ]

    pdsi_output_array = _global_shared_arrays[_KEY_RESULT_PDSI][_KEY_ARRAY]
    pdsi = np.frombuffer(pdsi_output_array.get_obj()).reshape(shape)[
        start_index:end_index
    ]

    phdi_output_array = _global_shared_arrays[_KEY_RESULT_PHDI][_KEY_ARRAY]
    phdi = np.frombuffer(phdi_output_array.get_obj()).reshape(shape)[
        start_index:end_index
    ]

    pmdi_output_array = _global_shared_arrays[_KEY_RESULT_PMDI][_KEY_ARRAY]
    pmdi = np.frombuffer(pmdi_output_array.get_obj()).reshape(shape)[
        start_index:end_index
    ]

    zindex_output_array = _global_shared_arrays[_KEY_RESULT_ZINDEX][_KEY_ARRAY]
    zindex = np.frombuffer(zindex_output_array.get_obj()).reshape(shape)[
        start_index:end_index
    ]

    for i, (precip, pet, awc) in enumerate(
        zip(sub_array_precip, sub_array_pet, sub_array_awc)
    ):
        scpdsi[i], pdsi[i], phdi[i], pmdi[i], zindex[i] = func1d(
            precip, pet, awc, parameters=args
        )


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
    if len(ds[var_name].dims) == 1:
        expected_dims = ("division",)
        dims = "division"
    elif len(ds[var_name].dims) == 2:
        expected_dims = ("division", "time")
        dims = "division,time"
    else:
        raise ValueError(
            "Unsupported dimensions for variable '{var_name}': {dims}".format(
                var_name=var_name, dims=ds[var_name].dims
            )
        )

    if Counter(ds[var_name].dims) != Counter(expected_dims):
        message = "Invalid dimensions for variable '{var_name}': {dims}".format(
            var_name=var_name, dims=ds[var_name].dims
        )
        _logger.error(message)
        raise ValueError(message)

    # perform reorder of dimensions if necessary
    if ds[var_name].dims != expected_dims:
        nco = Nco()
        netcdf_file = nco.ncpdq(
            input=netcdf_file, options=['-a "{dims}"'.format(dims=dims), "-O"]
        )

    return netcdf_file


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # This script is used to perform climate indices processing on NetCDF
    # gridded datasets.
    #
    # Example command line arguments for SPI only using monthly precipitation input:
    #
    # --index spi
    # --periodicity monthly
    # --scales 1 2 3 6 9 12 24
    # --calibration_start_year 1998
    # --calibration_end_year 2016
    # --netcdf_divs example_data/nclimgrid_prcp_lowres.nc
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
            "--calibration_end_year", help="Final year of calibration period", type=int
        )
        parser.add_argument(
            "--netcdf_precip",
            help="Precipitation NetCDF file to be used as input for indices computations",
        )
        parser.add_argument(
            "--var_name_precip",
            help="Precipitation variable name used in the precipitation NetCDF file",
        )
        parser.add_argument(
            "--netcdf_temp",
            help="Temperature NetCDF file to be used as input for indices computations",
        )
        parser.add_argument(
            "--var_name_temp",
            help="Temperature variable name used in the temperature NetCDF file",
        )
        parser.add_argument(
            "--netcdf_pet",
            help="PET NetCDF file to be used as input for SPEI and/or Palmer computations",
        )
        parser.add_argument(
            "--var_name_pet", help="PET variable name used in the PET NetCDF file"
        )
        parser.add_argument(
            "--netcdf_awc",
            help="Available water capacity NetCDF file to be used as input for the Palmer computations",
        )
        parser.add_argument(
            "--var_name_awc",
            help="Available water capacity variable name used in the AWC NetCDF file",
        )
        parser.add_argument(
            "--output_file_base",
            help="Base output file path and name for the resulting output files",
            required=True,
        )
        parser.add_argument(
            "--multiprocessing",
            help="Indices to compute",
            choices=["single", "all_but_one", "all"],
            required=False,
            default="all_but_one",
        )
        arguments = parser.parse_args()

        # validate the arguments
        _validate_args(arguments)

        if arguments.multiprocessing == "single":
            _NUMBER_OF_WORKER_PROCESSES = 1
        elif arguments.multiprocessing == "all":
            _NUMBER_OF_WORKER_PROCESSES = multiprocessing.cpu_count()
        else:  # default ("all_but_one")
            _NUMBER_OF_WORKER_PROCESSES = multiprocessing.cpu_count() - 1

        # compute SPI if specified
        if arguments.index in ["spi", "scaled", "all"]:

            # prepare the NetCDF in case the dimensions are not (time, divisions)
            netcdf_precip = _prepare_file(
                arguments.netcdf_precip, arguments.var_name_precip
            )

            # run SPI computations for each scale/distribution in turn
            for scale in arguments.scales:
                for dist in indices.Distribution:

                    # keyword arguments used for the SPI function
                    kwrgs = {
                        "index": "spi",
                        "netcdf_precip": netcdf_precip,
                        "var_name_precip": arguments.var_name_precip,
                        "scale": scale,
                        "distribution": dist,
                        "periodicity": arguments.periodicity,
                        "calibration_start_year": arguments.calibration_start_year,
                        "calibration_end_year": arguments.calibration_end_year,
                        "output_file_base": arguments.output_file_base,
                    }

                    # compute and write SPI
                    _compute_write_index(kwrgs)

            # remove temporary file if one was created
            if netcdf_precip != arguments.netcdf_precip:
                os.remove(netcdf_precip)

        if arguments.index in ["pet", "spei", "scaled", "palmers", "all"]:

            # run PET computation only if we've not been provided with a PET file
            if arguments.netcdf_pet is None:

                # prepare temperature NetCDF in case dimensions not (lat, lon, time)
                # or if coordinates are descending
                netcdf_temp = _prepare_file(
                    arguments.netcdf_temp, arguments.var_name_temp
                )

                # keyword arguments used for the PET function
                kwargs = {
                    "index": "pet",
                    "netcdf_temp": netcdf_temp,
                    "var_name_temp": arguments.var_name_temp,
                    "output_file_base": arguments.output_file_base,
                }

                # run PET computation, getting the PET file and corresponding variable name for later use
                arguments.netcdf_pet, arguments.var_name_pet = _compute_write_index(
                    kwargs
                )

                # remove temporary file
                if netcdf_temp != arguments.netcdf_temp:
                    os.remove(netcdf_temp)

        if arguments.index in ["spei", "scaled", "all"]:

            # prepare NetCDFs in case dimensions not (lat, lon, time) or if any coordinates are descending
            netcdf_precip = _prepare_file(
                arguments.netcdf_precip, arguments.var_name_precip
            )
            netcdf_pet = _prepare_file(arguments.netcdf_pet, arguments.var_name_pet)

            # run SPEI computations for each scale/distribution in turn
            for scale in arguments.scales:
                for dist in indices.Distribution:

                    # keyword arguments used for the SPI function
                    kwrgs = {
                        "index": "spei",
                        "netcdf_precip": netcdf_precip,
                        "var_name_precip": arguments.var_name_precip,
                        "netcdf_pet": netcdf_pet,
                        "var_name_pet": arguments.var_name_pet,
                        "scale": scale,
                        "distribution": dist,
                        "periodicity": arguments.periodicity,
                        "calibration_start_year": arguments.calibration_start_year,
                        "calibration_end_year": arguments.calibration_end_year,
                        "output_file_base": arguments.output_file_base,
                    }

                    # compute and write SPEI
                    _compute_write_index(kwrgs)

            # remove temporary file if one was created
            if netcdf_precip != arguments.netcdf_precip:
                os.remove(netcdf_precip)
            if netcdf_pet != arguments.netcdf_pet:
                os.remove(netcdf_pet)

        if arguments.index in ["pnp", "scaled", "all"]:

            # prepare precipitation NetCDF in case dimensions not (lat, lon, time) or if any coordinates are descending
            netcdf_precip = _prepare_file(
                arguments.netcdf_precip, arguments.var_name_precip
            )

            # run PNP computations for each scale in turn
            for scale in arguments.scales:

                # keyword arguments used for the SPI function
                kwrgs = {
                    "index": "pnp",
                    "netcdf_precip": netcdf_precip,
                    "var_name_precip": arguments.var_name_precip,
                    "scale": scale,
                    "periodicity": arguments.periodicity,
                    "calibration_start_year": arguments.calibration_start_year,
                    "calibration_end_year": arguments.calibration_end_year,
                    "output_file_base": arguments.output_file_base,
                }

                # compute and write PNP
                _compute_write_index(kwrgs)

            # remove temporary precipitation file if one was created
            if netcdf_precip != arguments.netcdf_precip:
                os.remove(netcdf_precip)

        if arguments.index in ["palmers", "all"]:

            # prepare NetCDFs in case dimensions not (lat, lon, time)
            netcdf_precip = _prepare_file(
                arguments.netcdf_precip, arguments.var_name_precip
            )
            netcdf_pet = _prepare_file(arguments.netcdf_pet, arguments.var_name_pet)
            netcdf_awc = _prepare_file(arguments.netcdf_awc, arguments.var_name_awc)

            # keyword arguments used for the SPI function
            kwrgs = {
                "index": "palmers",
                "netcdf_precip": netcdf_precip,
                "var_name_precip": arguments.var_name_precip,
                "netcdf_pet": netcdf_pet,
                "var_name_pet": arguments.var_name_pet,
                "netcdf_awc": netcdf_awc,
                "var_name_awc": arguments.var_name_awc,
                "calibration_start_year": arguments.calibration_start_year,
                "calibration_end_year": arguments.calibration_end_year,
                "output_file_base": arguments.output_file_base,
            }

            # compute and write Palmers
            _compute_write_index(kwrgs)

            # remove temporary files if they were created
            if netcdf_precip != arguments.netcdf_precip:
                os.remove(netcdf_precip)
            if netcdf_pet != arguments.netcdf_pet:
                os.remove(netcdf_pet)
            if netcdf_awc != arguments.netcdf_awc:
                os.remove(netcdf_awc)

        # report on the elapsed time
        end_datetime = datetime.now()
        _logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        _logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        _logger.exception("Failed to complete", exc_info=True)
        raise
