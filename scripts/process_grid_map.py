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
_NUMBER_OF_WORKER_PROCESSES = multiprocessing.cpu_count()

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
def compute_write_spi(kwrgs):

    # open the precipitation NetCDF as an xarray DataSet object
    dataset = xr.open_dataset(kwrgs["netcdf_precip"])

    # trim out all data variables from the dataset except the precipitation
    for var in dataset.data_vars:
        if var not in kwrgs["var_name_precip"]:
            dataset = dataset.drop(var)

    # get the initial year of the data
    data_start_year = int(str(dataset["time"].values[0])[0:4])
    kwrgs["data_start_year"] = data_start_year

    # get the scale increment for use in later log messages
    if kwrgs["periodicity"] == compute.Periodicity.daily:
        scale_increment = "day"
    elif kwrgs["periodicity"] == compute.Periodicity.monthly:
        scale_increment = "month"
    else:
        raise ValueError(
            "Invalid periodicity argument: {}".format(kwrgs["periodicity"])
        )

    _logger.info(
        "Computing {scale}-{incr} {index}/{dist}".format(
            scale=kwrgs["scale"],
            incr=scale_increment,
            index="SPI",
            dist=kwrgs["distribution"].value.capitalize(),
        )
    )

    # get the precipitation array, over which we'll compute the SPI
    da_precip = dataset[kwrgs["var_name_precip"]]

    # ensure we have our data array in either (lat, lon, time) or (lon, lat, time) orientation
    expected_dims = (("lat", "lon", "time"), ("lon", "lat", "time"))
    if da_precip.dims not in expected_dims:
        message = "Invalid dimensions for precipitation " "variable: {dims}".format(
            dims=da_precip.dims
        )
        _logger.error(message)
        raise ValueError(message)

    # # keyword arguments used for the function we'll apply to the data array
    # args_dict = {
    #     "scale": kwrgs["scale"],
    #     "distribution": kwrgs["distribution"],
    #     "data_start_year": data_start_year,
    #     "calibration_year_initial": kwrgs["calibration_start_year"],
    #     "calibration_year_final": kwrgs["calibration_end_year"],
    #     "periodicity": kwrgs["periodicity"],
    # }

    # apply the SPI function along the time axis (axis=2)
    spi_values = np.apply_along_axis(spi, axis=2, arr=da_precip.values, args=kwrgs)

    # TODO set global attributes accordingly for this new dataset

    # create a new variable to contain the SPI for the distribution/scale, assign into the dataset
    long_name = "Standardized Precipitation Index ({dist} distribution), ".format(
        dist=kwrgs["distribution"].value.capitalize()
    ) + "{scale}-{increment}".format(scale=kwrgs["scale"], increment=scale_increment)
    spi_attrs = {"long_name": long_name, "valid_min": -3.09, "valid_max": 3.09}
    var_name_spi = (
        "spi_" + kwrgs["distribution"].value + "_" + str(kwrgs["scale"]).zfill(2)
    )
    spi_var = xr.Variable(dims=da_precip.dims, data=spi_values, attrs=spi_attrs)
    dataset[var_name_spi] = spi_var

    # remove all data variables except for the new SPI variable
    for var_name in dataset.data_vars:
        if var_name != var_name_spi:
            dataset = dataset.drop(var_name)

    # write the dataset as NetCDF
    netcdf_file_name = kwrgs["output_file_base"] + "_" + var_name_spi + ".nc"
    dataset.to_netcdf(netcdf_file_name)

    return netcdf_file_name, var_name_spi


# ----------------------------------------------------------------------------------------------------------------------
def spi(precips, args):

    return indices.spi(
        precips,
        scale=args["scale"],
        distribution=args["distribution"],
        data_start_year=args["data_start_year"],
        calibration_year_initial=args["calibration_start_year"],
        calibration_year_final=args["calibration_end_year"],
        periodicity=args["periodicity"],
    )


# ----------------------------------------------------------------------------------------------------------------------
def run_multi_spi(
    netcdf_precip,
    var_name_precip,
    scales,
    periodicity,
    calibration_start_year,
    calibration_end_year,
    output_file_base,
):

    # create a process Pool for worker processes which will compute indices
    pool = multiprocessing.Pool(processes=_NUMBER_OF_WORKER_PROCESSES)

    # create an iterable of arguments specific to the function that we'll call within each worker process
    args = []
    for scale in scales:

        for dist in indices.Distribution:

            # keyword arguments used for the function we'll map
            kwrgs = {
                "netcdf_precip": netcdf_precip,
                "var_name_precip": var_name_precip,
                "scale": scale,
                "distribution": dist,
                "periodicity": periodicity,
                "calibration_start_year": calibration_start_year,
                "calibration_end_year": calibration_end_year,
                "output_file_base": output_file_base,
            }
            args.append(kwrgs)

    # map the arguments iterable to the compute function
    result = pool.map_async(compute_write_spi, args)

    # get/swallow the exception(s) thrown, if any
    result.get()

    # close the pool and wait on all processes to finish
    pool.close()
    pool.join()


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

            # run SPI with one process per scale/distribution
            run_multi_spi(
                netcdf_precip,
                arguments.var_name_precip,
                arguments.scales,
                arguments.periodicity,
                arguments.calibration_start_year,
                arguments.calibration_end_year,
                arguments.output_file_base,
            )

            # remove temporary file
            if netcdf_precip != arguments.netcdf_precip:
                os.remove(netcdf_precip)

        if arguments.index in ["pet", "spei", "scaled", "palmers", "all"]:

            # # run SPI with one process per scale/distribution
            # if arguments.netcdf_pet is None:
            #
            #     # prepare temperature NetCDF in case dimensions not (lat, lon, time) or if coordinates are descending
            #     netcdf_temp = _prepare_file(
            #         arguments.netcdf_temp, arguments.var_name_temp
            #     )
            #
            #     # keyword arguments used for the function we'll map
            #     kwargs = {
            #         "netcdf_temp": netcdf_temp,
            #         "var_name_temp": arguments.var_name_temp,
            #         "output_file_base": arguments.output_file_base,
            #     }
            #
            #     arguments.netcdf_pet, arguments.var_name_pet = compute_write_pet(kwargs)
            #
            #     # remove temporary file
            #     if netcdf_temp != arguments.netcdf_temp:
            #         os.remove(netcdf_temp)

            pass

        if arguments.index in ["spei", "scaled", "all"]:

            # # prepare NetCDFs in case dimensions not (lat, lon, time) or if any coordinates are descending
            # netcdf_precip = _prepare_file(
            #     arguments.netcdf_precip, arguments.var_name_precip
            # )
            # netcdf_pet = _prepare_file(arguments.netcdf_pet, arguments.var_name_pet)
            #
            # run_multi_spei(
            #     netcdf_precip,
            #     arguments.var_name_precip,
            #     netcdf_pet,
            #     arguments.var_name_pet,
            #     arguments.scales,
            #     arguments.periodicity,
            #     arguments.calibration_start_year,
            #     arguments.calibration_end_year,
            #     arguments.output_file_base,
            # )
            #
            # # remove temporary files
            # if netcdf_precip != arguments.netcdf_precip:
            #     os.remove(netcdf_precip)
            # if netcdf_pet != arguments.netcdf_pet:
            #     os.remove(netcdf_pet)

            pass

        if arguments.index in ["pnp", "scaled", "all"]:

            # # prepare NetCDF in case dimensions not (lat, lon, time) or if any coordinates are descending
            # netcdf_precip = _prepare_file(
            #     arguments.netcdf_precip, arguments.var_name_precip
            # )
            #
            # run_multi_pnp(
            #     netcdf_precip,
            #     arguments.var_name_precip,
            #     arguments.scales,
            #     arguments.periodicity,
            #     arguments.calibration_start_year,
            #     arguments.calibration_end_year,
            #     arguments.output_file_base,
            # )
            #
            # # remove temporary files
            # if netcdf_precip != arguments.netcdf_precip:
            #     os.remove(netcdf_precip)

            pass

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
