import argparse
from datetime import datetime
import logging

import numpy as np
import xarray as xr

from climate_indices import compute, indices

# ----------------------------------------------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
_logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------
def _validate_args(args):
    """
    Validate the processing settings to confirm that proper argument combinations have been provided.

    :param args: an arguments object of the type returned by argparse.ArgumentParser.parse_args()
    :raise ValueError: if one or more of the command line arguments is invalid
    """

    # the dimensions we expect to find for each data variable (precipitation, temperature, and/or PET)
    expected_dimensions = [('lat', 'lon', 'time'), ('time', 'lat', 'lon')]

    # all indices except PET require a precipitation file
    if args.index != 'pet':

        # make sure a precipitation file was specified
        if args.netcdf_precip is None:
            msg = 'Missing the required precipitation file'
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
                message = "Invalid precipitation variable name: '{var}' ".format(var=args.var_name_precip) + \
                          "does not exist in precipitation file '{file}'".format(file=args.netcdf_precip)
                _logger.error(message)
                raise ValueError(message)

            # verify that the precipitation variable's dimensions are in the expected order
            dimensions = dataset_precip[args.var_name_precip].dims
            if dimensions not in expected_dimensions:
                message = "Invalid dimensions of the precipitation variable: {dims}, ".format(dims=dimensions) + \
                          "(expected names and order: {dims})".format(dims=expected_dimensions)
                _logger.error(message)
                raise ValueError(message)

            # get the sizes of the latitude and longitude coordinate variables
            lats_precip = dataset_precip['lat'].values[:]
            lons_precip = dataset_precip['lon'].values[:]
            times_precip = dataset_precip['time'].values[:]

    else:

        # PET requires a temperature file
        if args.netcdf_temp is None:
            msg = 'Missing the required temperature file argument'
            _logger.error(msg)
            raise ValueError(msg)

        # don't allow a daily periodicity (yet, this will be possible once we have Hargreaves or a daily Thornthwaite)
        if args.periodicity != 'monthly':
            msg = "Invalid periodicity argument for PET: " + \
                "'{period}' -- only monthly is supported".format(period=args.periodicity)
            _logger.error(msg)
            raise ValueError(msg)

    # SPEI and Palmers require either a PET file or a temperature file in order to compute PET
    if args.index in ['spei', 'scaled', 'palmers']:

        if args.netcdf_temp is None:

            if args.netcdf_pet is None:
                msg = 'Missing the required temperature or PET files, neither were provided'
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
                    message = "Invalid PET variable name: '{var_name}' ".format(var_name=args.var_name_pet) + \
                              "does not exist in PET file '{file}'".format(file=args.netcdf_pet)
                    _logger.error(message)
                    raise ValueError(message)

                # verify that the PET variable's dimensions are in the expected order
                dimensions = dataset_pet[args.var_name_pet].dims
                if dimensions != expected_dimensions:
                    message = "Invalid dimensions of the PET variable: {dims}, ".format(dims=dimensions) + \
                              "(expected names and order: {dims})".format(dims=expected_dimensions)
                    _logger.error(message)
                    raise ValueError(message)

                # verify that the coordinate variables match with those of the precipitation dataset
                if not np.array_equal(lats_precip, dataset_pet['lat'][:]):
                    message = "Precipitation and PET variables contain non-matching latitudes"
                    _logger.error(message)
                    raise ValueError(message)
                elif not np.array_equal(lons_precip, dataset_pet['lon'][:]):
                    message = "Precipitation and PET variables contain non-matching longitudes"
                    _logger.error(message)
                    raise ValueError(message)
                elif not np.array_equal(times_precip, dataset_pet['time'][:]):
                    message = "Precipitation and PET variables contain non-matching times"
                    _logger.error(message)
                    raise ValueError(message)

        elif args.netcdf_pet is not None:

            # we can't have both temperature and PET files specified, no way to determine which to use
            msg = 'Both temperature and PET files were specified, only one of these should be provided'
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
                    message = "Invalid temperature variable name: '{var}' does ".format(var=args.var_name_temp) + \
                              "not exist in temperature file '{file}'".format(file=args.netcdf_temp)
                    _logger.error(message)
                    raise ValueError(message)

                # verify that the temperature variable's dimensions are in the expected order
                dimensions = dataset_temp[args.var_name_temp].dims
                if dimensions != expected_dimensions:
                    message = "Invalid dimensions of the temperature variable: {dims}, ".format(dims=dimensions) + \
                              "(expected names and order: {dims})".format(dims=expected_dimensions)
                    _logger.error(message)
                    raise ValueError(message)

                # verify that the coordinate variables match with those of the precipitation dataset
                if not np.array_equal(lats_precip, dataset_temp['lat'][:]):
                    message = "Precipitation and temperature variables contain non-matching latitudes"
                    _logger.error(message)
                    raise ValueError(message)
                elif not np.array_equal(lons_precip, dataset_temp['lon'][:]):
                    message = "Precipitation and temperature variables contain non-matching longitudes"
                    _logger.error(message)
                    raise ValueError(message)
                elif not np.array_equal(times_precip, dataset_temp['time'][:]):
                    message = "Precipitation and temperature variables contain non-matching times"
                    _logger.error(message)
                    raise ValueError(message)

        # Palmers requires an available water capacity file
        if args.index in ['palmers']:

            if args.netcdf_awc is None:

                msg = 'Missing the required available water capacity file'
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
                    message = "Invalid AWC variable name: '{var}' does not exist ".format(var=args.var_name_awc) + \
                              "in AWC file '{file}'".format(file=args.netcdf_awc)
                    _logger.error(message)
                    raise ValueError(message)

                # verify that the AWC variable's dimensions are in the expected order
                dimensions = dataset_awc[args.var_name_awc].dims
                if (dimensions != ('lat', 'lon')) and (dimensions != expected_dimensions):
                    message = "Invalid dimensions of the AWC variable: {dims}, ".format(dims=dimensions) + \
                              "(expected names and order: {dims})".format(dims=expected_dimensions)
                    _logger.error(message)
                    raise ValueError(message)

                # verify that the lat and lon coordinate variables match with those of the precipitation dataset
                if not np.array_equal(lats_precip, dataset_awc['lat'][:]):
                    message = "Precipitation and AWC variables contain non-matching latitudes"
                    _logger.error(message)
                    raise ValueError(message)
                elif not np.array_equal(lons_precip, dataset_awc['lon'][:]):
                    message = "Precipitation and AWC variables contain non-matching longitudes"
                    _logger.error(message)
                    raise ValueError(message)

    if args.index in ['spi', 'spei', 'scaled', 'pnp']:

        if args.scales is None:
            message = "Scaled indices (SPI, SPEI, and/or PNP) specified without including " + \
                      "one or more time scales (missing --scales argument)"
            _logger.error(message)
            raise ValueError(message)

        if any(n < 0 for n in args.scales):
            message = "One or more negative scale specified within --scales argument"
            _logger.error(message)
            raise ValueError(message)


# ----------------------------------------------------------------------------------------------------------------------
def spi_gamma(data_array,
              scale,
              start_year,
              calibration_year_initial,
              calibration_year_final,
              periodicity):

    original_shape = data_array.shape
    spi = indices.spi(data_array.values.squeeze(),
                      scale,
                      indices.Distribution.gamma,
                      start_year,
                      calibration_year_initial,
                      calibration_year_final,
                      periodicity)
    data_array.values = np.reshape(spi, newshape=original_shape)

    return data_array


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
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
    --output_file_base ~/data/cmorph/spi/cmorph 
    """

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        _logger.info("Start time:    %s", start_datetime)

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--index",
                            help="Indices to compute",
                            choices=['spi', 'spei', 'pnp', 'scaled', 'pet', 'palmers'],
                            required=True)
        parser.add_argument("--periodicity",
                            help="Process input as either monthly or daily values",
                            choices=[compute.Periodicity.monthly, compute.Periodicity.daily],
                            type=compute.Periodicity.from_string,
                            required=True)
        parser.add_argument("--scales",
                            help="Timestep scales over which the PNP, SPI, and SPEI values are to be computed",
                            type=int,
                            nargs='*')
        parser.add_argument("--calibration_start_year",
                            help="Initial year of the calibration period",
                            type=int)
        parser.add_argument("--calibration_end_year",
                            help="Final year of calibration period",
                            type=int)
        parser.add_argument("--netcdf_precip",
                            help="Precipitation NetCDF file to be used as input for indices computations")
        parser.add_argument("--var_name_precip",
                            help="Precipitation variable name used in the precipitation NetCDF file")
        parser.add_argument("--netcdf_temp",
                            help="Temperature NetCDF file to be used as input for indices computations")
        parser.add_argument("--var_name_temp",
                            help="Temperature variable name used in the temperature NetCDF file")
        parser.add_argument("--netcdf_pet",
                            help="PET NetCDF file to be used as input for SPEI and/or Palmer computations")
        parser.add_argument("--var_name_pet",
                            help="PET variable name used in the PET NetCDF file")
        parser.add_argument("--netcdf_awc",
                            help="Available water capacity NetCDF file to be used as input for the Palmer computations")
        parser.add_argument("--var_name_awc",
                            help="Available water capacity variable name used in the AWC NetCDF file")
        parser.add_argument("--output_file_base",
                            help="Base output file path and name for the resulting output files",
                            required=True)
        arguments = parser.parse_args()

        # validate the arguments
        _validate_args(arguments)

        # compute SPI if specified
        if arguments.index in ['spi', 'scaled']:

            # open the precipitation NetCDF as an xarray DataSet object
            dataset = xr.open_dataset(arguments.netcdf_precip) # , chunks={'lat': 1})

            # trim out all data variables from the dataset except the precipitation
            for var in dataset.data_vars:
                if var not in arguments.var_name_precip:
                    dataset = dataset.drop(var)

            # get the precipitation variable as an xarray DataArray object
            da_precip = dataset[arguments.var_name_precip]

            # get the initial year of the data
            data_start_year = int(str(da_precip['time'].values[0])[0:4])

            # stack the lat and lon dimensions into a new dimension named point, so at each lat/lon
            # we'll have a time series for the geospatial point
            da_precip = da_precip.stack(point=('lat', 'lon'))

            for timestep_scale in arguments.scales:

                if arguments.periodicity is compute.Periodicity.daily:
                    scale_increment = 'day'
                elif arguments.periodicity is compute.Periodicity.monthly:
                    scale_increment = 'month'
                else:
                    raise ValueError("Invalid periodicity argument: {}".format(arguments.periodicity))

                _logger.info('Computing {scale}-{incr} {index}'.format(scale=timestep_scale,
                                                                       incr=scale_increment,
                                                                       index='SPI'))

                # group the data by lat/lon point and apply the SPI/Gamma function to each time series group
                da_spi = da_precip.groupby('point').apply(spi_gamma,
                                                          scale=timestep_scale,
                                                          start_year=data_start_year,
                                                          calibration_year_initial=arguments.calibration_start_year,
                                                          calibration_year_final=arguments.calibration_end_year,
                                                          periodicity=arguments.periodicity)

                # unstack the array back into original dimensions
                da_spi = da_spi.unstack('point')

                # copy the original dataset since we'll be able to reuse most of the coordinates, attributes, etc.
                index_dataset = dataset.copy()

                # remove all data variables
                for var_name in index_dataset.data_vars:
                    index_dataset = index_dataset.drop(var_name)

                # TODO set global attributes accordingly for this new dataset

                # create a new variables to contain the SPI for the scale, assign into the dataset
                long_name = "Standardized Precipitation Index (Gamma distribution), "\
                            "{scale}-{increment}".format(scale=timestep_scale, increment=scale_increment)
                spi_var = xr.Variable(dims=da_spi.dims,
                                      data=da_spi,
                                      attrs={'long_name': long_name,
                                             'valid_min': -3.09,
                                             'valid_max': 3.09})
                var_name = "spi_gamma_" + str(timestep_scale).zfill(2)
                index_dataset[var_name] = spi_var

                # write the dataset as NetCDF
                index_dataset.to_netcdf(arguments.output_file_base + "_" + var_name + ".nc")

        # report on the elapsed time
        end_datetime = datetime.now()
        _logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        _logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        _logger.exception('Failed to complete', exc_info=True)
        raise
