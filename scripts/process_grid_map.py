import argparse
from collections import Counter
from datetime import datetime
import logging
import multiprocessing
import os

from nco import Nco
import numpy as np
import xarray as xr

from climate_indices import compute, indices

# the number of worker processes we'll use for process pools
_NUMBER_OF_WORKER_PROCESSES = 1  # multiprocessing.cpu_count()

# ----------------------------------------------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console as standard error
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
)
_logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------
def _validate_args(args):
    """
    Validate the processing settings to confirm that proper argument combinations have been provided.

    :param args: an arguments object of the type returned by argparse.ArgumentParser.parse_args()
    :raise ValueError: if one or more of the command line arguments is invalid
    """

    # the dimensions we expect to find for each data variable (precipitation, temperature, and/or PET)
    expected_dimensions = [("lat", "lon", "time"), ("time", "lat", "lon")]

    # all indices except PET require a precipitation file
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

        # validate the precipitation file itself
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

            # get the sizes of the latitude and longitude coordinate variables
            lats_precip = dataset_precip["lat"].values[:]
            lons_precip = dataset_precip["lon"].values[:]
            times_precip = dataset_precip["time"].values[:]

    else:

        # PET requires a temperature file
        if args.netcdf_temp is None:
            msg = "Missing the required temperature file argument"
            _logger.error(msg)
            raise ValueError(msg)

        # don't allow a daily periodicity (yet, this will be possible once we have Hargreaves or a daily Thornthwaite)
        if args.periodicity is not compute.Periodicity.monthly:
            msg = (
                "Invalid periodicity argument for PET: "
                + "'{period}' -- only monthly is supported".format(
                    period=args.periodicity
                )
            )
            _logger.error(msg)
            raise ValueError(msg)

    # SPEI and Palmers require either a PET file or a temperature file in order to compute PET
    if args.index in ["spei", "scaled", "palmers"]:

        if args.netcdf_temp is None:

            if args.netcdf_pet is None:
                msg = "Missing the required temperature or PET files, neither were provided"
                _logger.error(msg)
                raise ValueError(msg)

            # validate the PET file
            with xr.open_dataset(args.netcdf_pet) as dataset_pet:

                # make sure we have a valid PET variable name
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

                # verify that the PET variable's dimensions are in the expected order
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
                if not np.array_equal(lats_precip, dataset_pet["lat"][:]):
                    message = (
                        "Precipitation and PET variables contain non-matching latitudes"
                    )
                    _logger.error(message)
                    raise ValueError(message)
                elif not np.array_equal(lons_precip, dataset_pet["lon"][:]):
                    message = "Precipitation and PET variables contain non-matching longitudes"
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

                # verify that the temperature variable's dimensions are in the expected order
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
                if not np.array_equal(lats_precip, dataset_temp["lat"][:]):
                    message = "Precipitation and temperature variables contain non-matching latitudes"
                    _logger.error(message)
                    raise ValueError(message)
                elif not np.array_equal(lons_precip, dataset_temp["lon"][:]):
                    message = "Precipitation and temperature variables contain non-matching longitudes"
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
                if (dimensions != ("lat", "lon")) and (
                    dimensions != expected_dimensions
                ):
                    message = "Invalid dimensions of the AWC variable: {dims}, ".format(
                        dims=dimensions
                    ) + "(expected names and order: {dims})".format(
                        dims=expected_dimensions
                    )
                    _logger.error(message)
                    raise ValueError(message)

                # verify that the lat and lon coordinate variables match with those of the precipitation dataset
                if not np.array_equal(lats_precip, dataset_awc["lat"][:]):
                    message = (
                        "Precipitation and AWC variables contain non-matching latitudes"
                    )
                    _logger.error(message)
                    raise ValueError(message)
                elif not np.array_equal(lons_precip, dataset_awc["lon"][:]):
                    message = "Precipitation and AWC variables contain non-matching longitudes"
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


# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
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
                    index=args_dict["index"].value.capitalize(),
                )
            )

    else:

        _logger.info("Computing {index}".format(index=args_dict["index"].upper()))

    return True


# ----------------------------------------------------------------------------------------------------------------------
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
    elif keyword_args["index"] != "pet":
        raise ValueError(
            "Index {index} not yet supported.".format(index=keyword_args["index"])
        )

    return function_arguments


# ----------------------------------------------------------------------------------------------------------------------
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

        long_name = "Percentage of Normal Precipitation" + "{scale}-{increment}".format(
            scale=args_dict["scale"], increment=_get_scale_increment(args_dict)
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


# ----------------------------------------------------------------------------------------------------------------------
def _compute_write_index(keyword_arguments):
    """
    Computes a climate index and writes the result into a corresponding NetCDF.

    :param keyword_arguments:
    :return:
    """

    _log_status(keyword_arguments)

    # open the precipitation NetCDF files as an xarray DataSet object
    files = []
    if "netcdf_precip" in keyword_arguments:
        files.append(keyword_arguments["netcdf_precip"])
    if "netcdf_temp" in keyword_arguments:
        files.append(keyword_arguments["netcdf_temp"])
    if "netcdf_pet" in keyword_arguments:
        files.append(keyword_arguments["netcdf_pet"])
    dataset = xr.open_mfdataset(files)

    # trim out all data variables from the dataset except the ones we'll need
    var_names = []
    if "var_name_precip" in keyword_arguments:
        var_names.append(keyword_arguments["var_name_precip"])
    if "var_name_temp" in keyword_arguments:
        var_names.append(keyword_arguments["var_name_temp"])
    if "var_name_pet" in keyword_arguments:
        var_names.append(keyword_arguments["var_name_pet"])
    for var in dataset.data_vars:
        if var not in var_names:
            dataset = dataset.drop(var)

    # get the initial year of the data
    data_start_year = int(str(dataset["time"].values[0])[0:4])
    keyword_arguments["data_start_year"] = data_start_year

    data_arrays = {}
    expected_dims = (("lat", "lon", "time"), ("lon", "lat", "time"))
    for var_name in var_names:

        data_array = dataset[var_name]
        if data_array.dims not in expected_dims:
            message = "Invalid dimensions for variable '{var_name}\`: {dims}".format(
                var_name=var_name, dims=data_array.dims
            )
            _logger.error(message)
            raise ValueError(message)
        data_arrays[var_name] = data_array

    # build an arguments dictionary appropriate to the index we'll compute
    args = _build_arguments(keyword_arguments)

    if keyword_arguments["index"] == "spi":

        # get the precipitation array, over which we'll compute the SPI
        da_precip = data_arrays[keyword_arguments["var_name_precip"]]

        # apply the SPI function along the time axis (axis=2)
        index_values = _parallel_apply_along_axis(
            spi, 2, da_precip.values, args, **keyword_arguments
        )

    elif keyword_arguments["index"] == "spei":

        # get the precipitation and PET arrays, over which we'll compute the SPEI
        da_precip = data_arrays[keyword_arguments["var_name_precip"]]
        da_pet = data_arrays[keyword_arguments["var_name_pet"]]

        # add the PET array as an argument to the arguments dictionary
        keyword_arguments["pet_array"] = da_pet.values

        # apply the SPEI function along the time axis (axis=2)
        index_values = _parallel_apply_along_axis(
            spei, 2, da_precip.values, args, **keyword_arguments
        )

    elif keyword_arguments["index"] == "pet":

        # get the temperature and latitude arrays, over which we'll compute PET
        da_temp = data_arrays[keyword_arguments["var_name_temp"]]

        # create a DataArray for latitudes with the same shape as temperature,
        # filling all lon/times with the lat value for the lat index
        da_lat_orig = dataset["lat"]
        da_lat = da_temp.copy(deep=True).load()
        for lat_index in range(da_lat_orig.size):
            da_lat[dict(lat=lat_index)] = da_lat_orig.values[lat_index]

        # add the latitudes array as an argument to the arguments dictionary
        keyword_arguments["lat_array"] = da_lat.values

        # apply the PET function along the time axis (axis=2)
        index_values = _parallel_apply_along_axis(
            pet, 2, da_temp.values, args, **keyword_arguments
        )

    elif keyword_arguments["index"] == "pnp":

        # get the precipitation array, over which we'll compute the PNP
        da_precip = data_arrays[keyword_arguments["var_name_precip"]]

        # apply the PNP function along the time axis (axis=2)
        index_values = _parallel_apply_along_axis(
            pnp, axis=2, arr=da_precip.values, arguments=args
        )

    else:

        raise ValueError(
            "Index {index} not yet supported.".format(index=keyword_arguments["index"])
        )

    # TODO set global attributes accordingly for this new dataset

    # get the name and attributes to use for the index variable in the output NetCDF
    variable_name, attributes = _get_variable_attributes(keyword_arguments)

    # here we assume all input data arrays share the dimensions of the computed
    # index values, so we just get the dimensions from the first one we find in the
    # dictionary of input data arrays
    dimensions = data_arrays[list(data_arrays.keys())[0]].dims

    # create a new variable to contain the index values, assign into the dataset
    variable = xr.Variable(dims=dimensions, data=index_values, attrs=attributes)
    dataset[variable_name] = variable

    # remove all data variables except for the new variable
    for var_name in dataset.data_vars:
        if var_name != variable_name:
            dataset = dataset.drop(var_name)

    # write the dataset as NetCDF
    netcdf_file_name = (
        keyword_arguments["output_file_base"] + "_" + variable_name + ".nc"
    )
    dataset.to_netcdf(netcdf_file_name)

    return netcdf_file_name, variable_name


# ----------------------------------------------------------------------------------------------------------------------
def pet(temps, parameters):

    return indices.pet(
        temps,
        latitude_degrees=parameters["latitude_degrees"],
        data_start_year=parameters["data_start_year"],
    )


# ----------------------------------------------------------------------------------------------------------------------
def spi(precips, parameters):

    return indices.spi(
        precips,
        scale=parameters["scale"],
        distribution=parameters["distribution"],
        data_start_year=parameters["data_start_year"],
        calibration_year_initial=parameters["calibration_year_initial"],
        calibration_year_final=parameters["calibration_year_final"],
        periodicity=parameters["periodicity"],
    )


# ----------------------------------------------------------------------------------------------------------------------
def spei(precips, parameters):

    return indices.spei(
        precips,
        pet_mm=parameters["pet_mm"],
        scale=parameters["scale"],
        distribution=parameters["distribution"],
        data_start_year=parameters["data_start_year"],
        calibration_year_initial=parameters["calibration_year_initial"],
        calibration_year_final=parameters["calibration_year_final"],
        periodicity=parameters["periodicity"],
    )


# ----------------------------------------------------------------------------------------------------------------------
def pnp(precips, parameters):

    return indices.percentage_of_normal(
        precips,
        scale=parameters["scale"],
        data_start_year=parameters["data_start_year"],
        calibration_start_year=parameters["calibration_start_year"],
        calibration_end_year=parameters["calibration_end_year"],
        periodicity=parameters["periodicity"],
    )


# ----------------------------------------------------------------------------------------------------------------------
def _parallel_apply_along_axis(func1d, axis, arr, args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple cores.

    :param func1d:
    :param axis:
    :param arr:
    :param args:
    :param kwargs:
    :return:
    """

    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # build a list of parameters for each application of the function to an array chunk
    chunk_params = []
    if kwargs["index"] in ["spi", "pnp"]:

        # we have a single input array
        for sub_arr in np.array_split(arr, _NUMBER_OF_WORKER_PROCESSES):
            params = {
                "func1d": func1d,
                "axis": effective_axis,
                "arr": sub_arr,
                "args": args,
                "kwargs": None,
            }
            chunk_params.append(params)

    elif kwargs["index"] == "spei":

        # we have a two input arrays (precipitation and PET)
        for sub_arr1, sub_arr2 in (
            np.array_split(arr, _NUMBER_OF_WORKER_PROCESSES),
            np.array_split(kwargs["pet_array"], _NUMBER_OF_WORKER_PROCESSES),
        ):

            # add the PET sub-array as the PET array argument expected by the SPEI function
            args["pet_mm"] = sub_arr2

            params = {
                "func1d": func1d,
                "axis": effective_axis,
                "arr": sub_arr1,
                "args": args,
                "kwargs": None,
            }
            chunk_params.append(params)

    elif kwargs["index"] == "pet":

        # we have a two input arrays (temperature and latitude)
        for sub_arr1, sub_arr2 in zip(
            np.array_split(arr, _NUMBER_OF_WORKER_PROCESSES),
            np.array_split(kwargs["lat_array"], _NUMBER_OF_WORKER_PROCESSES),
        ):

            # add the latitude sub-array as the latitude array argument expected by the PET function
            args["latitude_degrees"] = sub_arr2

            params = {
                "func1d": func1d,
                "axis": effective_axis,
                "arr": sub_arr1,
                "args": args,
                "kwargs": None,
            }
            chunk_params.append(params)

    # instantiate a process pool
    pool = multiprocessing.Pool(processes=_NUMBER_OF_WORKER_PROCESSES)

    """
     the function _unpacking_apply_along_axis() being applied in Pool.map() is separate
     so that subprocesses can import it, and is simply a thin wrapper that handles the
     fact that Pool.map() only takes a single argument:
    """

    individual_results = pool.map(_unpacking_apply_along_axis, chunk_params)

    # close the pool and wait on all processes to finish
    pool.close()
    pool.join()

    # concatenate all the individual result arrays back into a complete result array
    return np.concatenate(individual_results)


# ----------------------------------------------------------------------------------------------------------------------
def _unpacking_apply_along_axis(params):
    """
    Like numpy.apply_along_axis(), but and with arguments in a dict
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    func1d = params["func1d"]
    axis = params["axis"]
    arr = params["arr"]
    args = params["args"]
    return np.apply_along_axis(func1d, axis, arr, parameters=args)


# ----------------------------------------------------------------------------------------------------------------------
def _prepare_file(netcdf_file, var_name):

    # determine if coordinates are correctly ordered in ascending order
    ds = xr.open_dataset(netcdf_file)

    # make sure we have lat, lon, and time as variable dimensions
    expected_dims = ("lat", "lon", "time")
    if Counter(ds[var_name].dims) != Counter(expected_dims):
        message = "Invalid dimensions for precipitation variable: {dims}".format(
            dims=ds[var_name].dims
        )
        _logger.error(message)
        raise ValueError(message)

    # see if we need to reorder into (lat, lon, time)
    reorder_dims = ds[var_name].dims != expected_dims

    # see if we need to reverse the lat and/or lon dimensions
    dims = []
    reverse_dims = False
    for dim_name in ["lat", "lon"]:
        vals = ds[dim_name].values
        if np.all(vals[:-1] <= vals[1:]):
            dims.append(dim_name)
        else:
            reverse_dims = True
            dims.append("-" + dim_name)
    dims.append("time")

    # perform reorder and/or reversal of dimensions if necessary
    if reorder_dims or reverse_dims:
        dims = ",".join(dims)
        nco = Nco()
        netcdf_file = nco.ncpdq(
            input=netcdf_file, options=['-a \\"{dims}\\"'.format(dims=dims), "-O"]
        )

    return netcdf_file


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    This script is used to perform climate indices processing on gridded datasets in NetCDF.
    
    Example command line arguments for SPI only using monthly precipitation input:
    
    --index spi 
    --periodicity monthly
    --scales 1 2 3 6 9 12 24 
    --calibration_start_year 1998 
    --calibration_end_year 2016 
    --netcdf_precip example_data/nclimgrid_prcp_lowres.nc 
    --var_name_precip prcp 
    --output_file_base ~/data/test/spi/nclimgrid_lowres 
    """

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        _logger.info("Start time:    %s", start_datetime)

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--index",
            help="Indices to compute",
            choices=["spi", "spei", "pnp", "scaled", "pet", "palmers"],
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
        arguments = parser.parse_args()

        # validate the arguments
        _validate_args(arguments)

        # compute SPI if specified
        if arguments.index in ["spi", "scaled", "all"]:

            # prepare precipitation NetCDF in case dimensions not (lat, lon, time) or if any coordinates are descending
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

            # run SPI computations for each scale/distribution in turn
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

            # run SPI computations for each scale/distribution in turn
            for scale in arguments.scales:
                for dist in indices.Distribution:

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

            # remove temporary file if one was created
            if netcdf_precip != arguments.netcdf_precip:
                os.remove(netcdf_precip)

        if arguments.index in ["palmers", "all"]:

            # # TODO prepare input NetCDF files, ensure matching dimensions, etc.
            #
            # # keyword arguments used for the function we'll map
            # kwargs = {
            #     "netcdf_precip": arguments.netcdf_precip,
            #     "var_name_precip": arguments.var_name_precip,
            #     "netcdf_pet": arguments.netcdf_pet,
            #     "var_name_pet": arguments.var_name_pet,
            #     "netcdf_awc": arguments.netcdf_awc,
            #     "var_name_awc": arguments.var_name_awc,
            #     "calibration_start_year": arguments.calibration_start_year,
            #     "calibration_end_year": arguments.calibration_end_year,
            #     "periodicity": arguments.periodicity,
            #     "output_file_base": arguments.output_file_base,
            # }
            #
            # compute_write_palmers(kwargs)

            pass

        # report on the elapsed time
        end_datetime = datetime.now()
        _logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        _logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        _logger.exception("Failed to complete", exc_info=True)
        raise
