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
        if args.periodicity is not compute.Periodicity.monthly:
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
                if dimensions not in expected_dimensions:
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
                if dimensions not in expected_dimensions:
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
def compute_write_spi(kwrgs):

    # open the precipitation NetCDF as an xarray DataSet object
    dataset = xr.open_dataset(kwrgs['netcdf_precip'])

    # trim out all data variables from the dataset except the precipitation
    for var in dataset.data_vars:
        if var not in kwrgs['var_name_precip']:
            dataset = dataset.drop(var)

    # get the initial year of the data
    data_start_year = int(str(dataset['time'].values[0])[0:4])

    # get the scale increment for use in later log messages
    if kwrgs['periodicity'] is compute.Periodicity.daily:
        scale_increment = 'day'
    elif kwrgs['periodicity'] is compute.Periodicity.monthly:
        scale_increment = 'month'
    else:
        raise ValueError("Invalid periodicity argument: {}".format(kwrgs['periodicity']))

    _logger.info("Computing {scale}-{incr} {index}/{dist}".format(scale=kwrgs['scale'],
                                                                  incr=scale_increment,
                                                                  index='SPI',
                                                                  dist=kwrgs['distribution'].value.capitalize()))

    # get the precipitation array, over which we'll compute the SPI
    da_precip = dataset[kwrgs['var_name_precip']]

    # stack the lat and lon dimensions into a new dimension named point, so at each lat/lon
    # we'll have a time series for the geospatial point, and group by these points
    da_precip_groupby = da_precip.stack(point=('lat', 'lon')).groupby('point')

    # keyword arguments used for the function we'll apply to the data array
    args_dict = {'scale': kwrgs['scale'],
                 'distribution': kwrgs['distribution'],
                 'data_start_year': data_start_year,
                 'calibration_year_initial': kwrgs['calibration_start_year'],
                 'calibration_year_final': kwrgs['calibration_end_year'],
                 'periodicity': kwrgs['periodicity']}

    # apply the SPI function to the data array
    da_spi = xr.apply_ufunc(indices.spi,
                            da_precip_groupby,
                            kwargs=args_dict)

    # unstack the array back into original dimensions
    da_spi = da_spi.unstack('point')

    # TODO set global attributes accordingly for this new dataset

    # create a new variable to contain the SPI for the distribution/scale, assign into the dataset
    long_name = "Standardized Precipitation Index ({dist} distribution), "\
                .format(dist=kwrgs['distribution'].value.capitalize()) + \
                "{scale}-{increment}".format(scale=kwrgs['scale'], increment=scale_increment)
    spi_attrs = {'long_name': long_name,
                 'valid_min': -3.09,
                 'valid_max': 3.09}
    var_name_spi = "spi_" + kwrgs['distribution'].value + "_" + str(kwrgs['scale']).zfill(2)
    spi_var = xr.Variable(dims=da_spi.dims,
                          data=da_spi,
                          attrs=spi_attrs)
    dataset[var_name_spi] = spi_var

    # remove all data variables except for the new SPI variable
    for var_name in dataset.data_vars:
        if var_name != var_name_spi:
            dataset = dataset.drop(var_name)

    # write the dataset as NetCDF
    netcdf_file_name = kwrgs['output_file_base'] + "_" + var_name_spi + ".nc"
    dataset.to_netcdf(netcdf_file_name)

    return netcdf_file_name, var_name_spi


# ----------------------------------------------------------------------------------------------------------------------
def compute_write_pnp(kwrgs):

    # open the precipitation NetCDF as an xarray DataSet object
    dataset = xr.open_dataset(kwrgs['netcdf_precip'])

    # trim out all data variables from the dataset except the precipitation
    for var in dataset.data_vars:
        if var not in kwrgs['var_name_precip']:
            dataset = dataset.drop(var)

    # get the initial year of the data
    data_start_year = int(str(dataset['time'].values[0])[0:4])

    # get the scale increment for use in later log messages
    if kwrgs['periodicity'] is compute.Periodicity.daily:
        scale_increment = 'day'
    elif kwrgs['periodicity'] is compute.Periodicity.monthly:
        scale_increment = 'month'
    else:
        raise ValueError("Invalid periodicity argument: {}".format(kwrgs['periodicity']))

    _logger.info("Computing {scale}-{incr} {index}".format(scale=kwrgs['scale'],
                                                           incr=scale_increment,
                                                           index='PNP'))

    # get the precipitation array, over which we'll compute the SPI
    da_precip = dataset[kwrgs['var_name_precip']]

    # stack the lat and lon dimensions into a new dimension named point, so at each lat/lon
    # we'll have a time series for the geospatial point, and group by these points
    da_precip_groupby = da_precip.stack(point=('lat', 'lon')).groupby('point')

    # keyword arguments used for the function we'll apply to the data array
    args_dict = {'scale': kwrgs['scale'],
                 'data_start_year': data_start_year,
                 'calibration_start_year': kwrgs['calibration_start_year'],
                 'calibration_end_year': kwrgs['calibration_end_year'],
                 'periodicity': kwrgs['periodicity']}

    # apply the PNP function to the data array
    da_pnp = xr.apply_ufunc(indices.percentage_of_normal,
                            da_precip_groupby,
                            kwargs=args_dict)

    # unstack the array back into original dimensions
    da_pnp = da_pnp.unstack('point')

    # TODO set global attributes accordingly for this new dataset

    # create a new variable to contain the SPI for the distribution/scale, assign into the dataset
    long_name = "Percentage of Normal Precipitation" + \
                "{scale}-{increment}".format(scale=kwrgs['scale'], increment=scale_increment)
    pnp_attrs = {'long_name': long_name,
                 'valid_min': -1000.0,
                 'valid_max': 1000.0}
    var_name_pnp = "pnp_" + "_" + str(kwrgs['scale']).zfill(2)
    pnp_var = xr.Variable(dims=da_pnp.dims,
                          data=da_pnp,
                          attrs=pnp_attrs)
    dataset[var_name_pnp] = pnp_var

    # remove all data variables except for the new PNP variable
    for var_name in dataset.data_vars:
        if var_name != var_name_pnp:
            dataset = dataset.drop(var_name)

    # write the dataset as NetCDF
    netcdf_file_name = kwrgs['output_file_base'] + "_" + var_name_pnp + ".nc"
    dataset.to_netcdf(netcdf_file_name)

    return netcdf_file_name, var_name_pnp


# ----------------------------------------------------------------------------------------------------------------------
def compute_write_spei(kwrgs):

    # open the precipitation and PET NetCDFs as a single xarray.DataSet object
    dataset_precip = xr.open_dataset(kwrgs['netcdf_precip'])
    dataset_pet = xr.open_dataset(kwrgs['netcdf_pet'])
    dataset = dataset_precip.merge(dataset_pet)

    # trim out all data variables from the dataset except the precipitation and PET
    for var in dataset.data_vars:
        if var not in [kwrgs['var_name_precip'], kwrgs['var_name_pet']]:
            dataset = dataset.drop(var)

    # get the initial year of the data
    data_start_year = int(str(dataset['time'].values[0])[0:4])

    # get the scale increment for use in later log messages
    if kwrgs['periodicity'] is compute.Periodicity.daily:
        scale_increment = 'day'
    elif kwrgs['periodicity'] is compute.Periodicity.monthly:
        scale_increment = 'month'
    else:
        raise ValueError("Invalid periodicity argument: {}".format(kwrgs['periodicity']))

    _logger.info("Computing {scale}-{incr} {index}/{dist}".format(scale=kwrgs['scale'],
                                                                  incr=scale_increment,
                                                                  index='SPEI',
                                                                  dist=kwrgs['distribution'].value.capitalize()))

    # get the precipitation and PET arrays, over which we'll compute the SPEI
    da_precip = dataset[kwrgs['var_name_precip']]
    da_pet = dataset[kwrgs['var_name_pet']]

    # stack the lat and lon dimensions into a new dimension named point, so at each lat/lon
    # we'll have a time series for the geospatial point, and group by these points
    da_precip_groupby = da_precip.stack(point=('lat', 'lon')).groupby('point')
    da_pet_groupby = da_pet.stack(point=('lat', 'lon')).groupby('point')

    # keyword arguments used for the function we'll apply to the data array
    args_dict = {'scale': kwrgs['scale'],
                 'distribution': kwrgs['distribution'],
                 'data_start_year': data_start_year,
                 'calibration_year_initial': kwrgs['calibration_start_year'],
                 'calibration_year_final': kwrgs['calibration_end_year'],
                 'periodicity': kwrgs['periodicity']}

    # apply the SPEI function to the data arrays
    da_spei = xr.apply_ufunc(indices.spei,
                             da_precip_groupby,
                             da_pet_groupby,
                             kwargs=args_dict)

    # unstack the array back into original dimensions
    da_spei = da_spei.unstack('point')

    # TODO set global attributes accordingly for this new dataset

    # create a new variable to contain the SPEI for the distribution/scale, assign into the dataset
    long_name = "Standardized Precipitation Evapotranspiration Index ({dist} distribution), "\
                .format(dist=kwrgs['distribution'].value.capitalize()) + \
                "{scale}-{increment}".format(scale=kwrgs['scale'], increment=scale_increment)
    spei_attrs = {'long_name': long_name,
                  'valid_min': -3.09,
                  'valid_max': 3.09}
    var_name_spei = "spei_" + kwrgs['distribution'].value + "_" + str(kwrgs['scale']).zfill(2)
    spei_var = xr.Variable(dims=da_spei.dims,
                           data=da_spei,
                           attrs=spei_attrs)
    dataset[var_name_spei] = spei_var

    # remove all data variables except for the new SPEI variable
    for var_name in dataset.data_vars:
        if var_name != var_name_spei:
            dataset = dataset.drop(var_name)

    # write the dataset as NetCDF
    netcdf_file_name = kwrgs['output_file_base'] + "_" + var_name_spei + ".nc"
    dataset.to_netcdf(netcdf_file_name)

    return netcdf_file_name, var_name_spei


# ----------------------------------------------------------------------------------------------------------------------
def compute_write_palmers(kwrgs):

    _logger.info("Computing {period} Palmers".format(period=kwrgs['periodicity']))

    # open the precipitation, PET, and AWC NetCDFs as a single xarray.DataSet object
    dataset_precip = xr.open_dataset(kwrgs['netcdf_precip'])
    dataset_pet = xr.open_dataset(kwrgs['netcdf_pet'])
    dataset_awc = xr.open_dataset(kwrgs['netcdf_awc'])
    dataset = dataset_precip.merge(dataset_pet)
    dataset = dataset.merge(dataset_awc)

    # trim out all data variables from the dataset except the precipitation, PET, and AWC
    for var in dataset.data_vars:
        if var not in [kwrgs['var_name_precip'], kwrgs['var_name_pet'], kwrgs['var_name_awc']]:
            dataset = dataset.drop(var)

    # get the initial year of the data
    data_start_year = int(str(dataset['time'].values[0])[0:4])

    # get the precipitation, PET, and AWC arrays
    da_precip = dataset[kwrgs['var_name_precip']]
    da_pet = dataset[kwrgs['var_name_pet']]

    # add a time dimension and duplicate the AWC value across all times for each lat/lon, in order
    # to have an array of the same size and dims as the precipitation and PET arrays, allowing for
    # the use of an AWC GroupBy that will correspond to the precipitation and PET GroupBys

    # create a DataArray with the same shape as temperature, fill all times with the AWC value for the lat/lon index
    da_awc_orig = dataset[kwrgs['var_name_awc']]
    da_awc = dataset[kwrgs['var_name_precip']].copy(deep=True)
    for lat_index in range(da_awc_orig['lat'].size):
        for lon_index in range(da_awc_orig['lon'].size):
            da_awc[dict(lat=lat_index, lon=lon_index)] = da_awc_orig[dict(lat=lat_index, lon=lon_index)]

    # stack the lat and lon dimensions into a new dimension named point, so at each lat/lon
    # we'll have a time series for the geospatial point, and group by these points
    da_precip_groupby = da_precip.stack(point=('lat', 'lon')).groupby('point')
    da_pet_groupby = da_pet.stack(point=('lat', 'lon')).groupby('point')
    da_awc_groupby = da_awc.stack(point=('lat', 'lon')).groupby('point')

    # keyword arguments used for the function we'll apply to the data array
    args_dict = {'data_start_year': data_start_year,
                 'calibration_start_year': kwrgs['calibration_start_year'],
                 'calibration_end_year': kwrgs['calibration_end_year']}

    # apply the self-calibrated Palmers function to the data arrays
    da_scpdsi, da_pdsi, da_phdi, da_pmdi, da_zindex = xr.apply_ufunc(indices.scpdsi,
                                                                     da_precip_groupby,
                                                                     da_pet_groupby,
                                                                     da_awc_groupby,
                                                                     output_core_dims=[[], [], [], [], []],
                                                                     kwargs=args_dict)

    # unstack the arrays back into original dimensions
    da_scpdsi = da_scpdsi.unstack('point')
    da_pdsi = da_pdsi.unstack('point')
    da_phdi = da_phdi.unstack('point')
    da_pmdi = da_pmdi.unstack('point')
    da_zindex = da_zindex.unstack('point')

    # TODO set global attributes accordingly for this new dataset

    # create a new variable to contain the SCPDSI values, assign into the dataset
    long_name = "Self-calibrated Palmer Drought Severity Index"
    scpdsi_attrs = {'long_name': long_name,
                    'valid_min': -10.0,
                    'valid_max': 10.0}
    var_name_scpdsi = "scpdsi"
    scpdsi_var = xr.Variable(dims=da_scpdsi.dims,
                             data=da_scpdsi,
                             attrs=scpdsi_attrs)
    dataset[var_name_scpdsi] = scpdsi_var

    # remove all data variables except for the new SCPDSI variable
    for var_name in dataset.data_vars:
        if var_name != var_name_scpdsi:
            dataset = dataset.drop(var_name)

    # write the dataset as NetCDF
    netcdf_file_name = kwrgs['output_file_base'] + "_" + var_name_scpdsi + ".nc"
    dataset.to_netcdf(netcdf_file_name)

    # create a new variable to contain the PDSI values, assign into the dataset
    long_name = "Palmer Drought Severity Index"
    pdsi_attrs = {'long_name': long_name,
                  'valid_min': -10.0,
                  'valid_max': 10.0}
    var_name_pdsi = "pdsi"
    pdsi_var = xr.Variable(dims=da_pdsi.dims,
                           data=da_pdsi,
                           attrs=pdsi_attrs)
    dataset[var_name_pdsi] = pdsi_var

    # remove all data variables except for the new PDSI variable
    for var_name in dataset.data_vars:
        if var_name != var_name_pdsi:
            dataset = dataset.drop(var_name)

    # write the dataset as NetCDF
    netcdf_file_name = kwrgs['output_file_base'] + "_" + var_name_pdsi + ".nc"
    dataset.to_netcdf(netcdf_file_name)

    # create a new variable to contain the PHDI values, assign into the dataset
    long_name = "Palmer Hydrological Drought Index"
    phdi_attrs = {'long_name': long_name,
                  'valid_min': -10.0,
                  'valid_max': 10.0}
    var_name_phdi = "phdi"
    phdi_var = xr.Variable(dims=da_phdi.dims,
                           data=da_phdi,
                           attrs=phdi_attrs)
    dataset[var_name_phdi] = phdi_var

    # remove all data variables except for the new PHDI variable
    for var_name in dataset.data_vars:
        if var_name != var_name_phdi:
            dataset = dataset.drop(var_name)

    # write the dataset as NetCDF
    netcdf_file_name = kwrgs['output_file_base'] + "_" + var_name_phdi + ".nc"
    dataset.to_netcdf(netcdf_file_name)

    # create a new variable to contain the PMDI values, assign into the dataset
    long_name = "Palmer Modified Drought Index"
    pmdi_attrs = {'long_name': long_name,
                  'valid_min': -10.0,
                  'valid_max': 10.0}
    var_name_pmdi = "pmdi"
    pmdi_var = xr.Variable(dims=da_pmdi.dims,
                           data=da_pmdi,
                           attrs=pmdi_attrs)
    dataset[var_name_pmdi] = pmdi_var

    # remove all data variables except for the new PMDI variable
    for var_name in dataset.data_vars:
        if var_name != var_name_pmdi:
            dataset = dataset.drop(var_name)

    # write the dataset as NetCDF
    netcdf_file_name = kwrgs['output_file_base'] + "_" + var_name_pmdi + ".nc"
    dataset.to_netcdf(netcdf_file_name)

    # create a new variable to contain the Z-Index values, assign into the dataset
    long_name = "Palmer Z-Index"
    zindex_attrs = {'long_name': long_name,
                    'valid_min': -10.0,
                    'valid_max': 10.0}
    var_name_zindex = "zindex"
    zindex_var = xr.Variable(dims=da_zindex.dims,
                             data=da_zindex,
                             attrs=zindex_attrs)
    dataset[var_name_zindex] = zindex_var

    # remove all data variables except for the new Z-Index variable
    for var_name in dataset.data_vars:
        if var_name != var_name_zindex:
            dataset = dataset.drop(var_name)

    # write the dataset as NetCDF
    netcdf_file_name = kwrgs['output_file_base'] + "_" + var_name_zindex + ".nc"
    dataset.to_netcdf(netcdf_file_name)

    return True


# ----------------------------------------------------------------------------------------------------------------------
def compute_write_pet(kwrgs):

    # open the temperature NetCDF as an xarray DataSet object
    dataset = xr.open_dataset(kwrgs['netcdf_temp'])

    # trim out all data variables from the dataset except the precipitation
    for var in dataset.data_vars:
        if var not in kwrgs['var_name_temp']:
            dataset = dataset.drop(var)

    # get the initial year of the data
    data_start_year = int(str(dataset['time'].values[0])[0:4])

    _logger.info("Computing PET")

    # get the temperature and latitude arrays, over which we'll compute the PET
    da_temp = dataset[kwrgs['var_name_temp']]

    # create a DataArray with the same shape as temperature, fill all lon/times with the lat value for the lat index
    da_lat_orig = dataset['lat']
    da_lat = dataset['tavg'].copy(deep=True)
    for lat_index in range(da_lat_orig.size):
        da_lat[dict(lat=lat_index)] = da_lat_orig.values[lat_index]

    # stack the lat and lon dimensions into a new dimension named point, so at each lat/lon
    # we'll have a time series for the geospatial point, and group by these points
    da_temp_groupby = da_temp.stack(point=('lat', 'lon')).groupby('point')
    da_lat_groupby = da_lat.stack(point=('lat', 'lon')).groupby('point')

    # keyword arguments used for the function we'll apply to the data arrays
    args_dict = {'data_start_year': data_start_year}

    # apply the PET function to the data arrays
    da_pet = xr.apply_ufunc(indices.pet,
                            da_temp_groupby,
                            da_lat_groupby,
                            kwargs=args_dict)

    # unstack the array back into original dimensions
    da_pet = da_pet.unstack('point')

    # TODO set global attributes accordingly for this new dataset

    # create a new variable to contain the PET values, assign into the dataset
    long_name = "Potential Evapotranspiration (Thornthwaite)"
    pet_attrs = {'long_name': long_name,
                 'valid_min': 0.0,
                 'valid_max': 10000.0,
                 'units': 'millimeters'}
    var_name_pet = "pet_thornthwaite"
    pet_var = xr.Variable(dims=da_pet.dims,
                          data=da_pet,
                          attrs=pet_attrs)
    dataset[var_name_pet] = pet_var

    # remove all data variables except for the new PET variable
    for var_name in dataset.data_vars:
        if var_name != var_name_pet:
            dataset = dataset.drop(var_name)

    # write the dataset as NetCDF
    netcdf_file_name = kwrgs['output_file_base'] + "_" + var_name_pet + ".nc"
    dataset.to_netcdf(netcdf_file_name)

    return netcdf_file_name, var_name_pet


# ----------------------------------------------------------------------------------------------------------------------
def run_multi_pnp(netcdf_precip,
                  var_name_precip,
                  scales,
                  periodicity,
                  calibration_start_year,
                  calibration_end_year,
                  output_file_base):

    # the number of worker processes we'll use in our process pool
    number_of_workers = multiprocessing.cpu_count()  # NOTE use 1 here when debugging for less butt hurt

    # create a process Pool for worker processes which will compute indices
    pool = multiprocessing.Pool(processes=number_of_workers)

    # create an iterable of arguments specific to the function that we'll call within each worker process
    args = []
    for scale in scales:

        # keyword arguments used for the function we'll map
        kwrgs = {'netcdf_precip': netcdf_precip,
                 'var_name_precip': var_name_precip,
                 'scale': scale,
                 'periodicity': periodicity,
                 'calibration_start_year': calibration_start_year,
                 'calibration_end_year': calibration_end_year,
                 'output_file_base': output_file_base}
        args.append(kwrgs)

    # map the arguments iterable to the compute function
    result = pool.map_async(compute_write_pnp, args)

    # get/swallow the exception(s) thrown, if any
    result.get()

    # close the pool and wait on all processes to finish
    pool.close()
    pool.join()


# ----------------------------------------------------------------------------------------------------------------------
def run_multi_spi(netcdf_precip,
                  var_name_precip,
                  scales,
                  periodicity,
                  calibration_start_year,
                  calibration_end_year,
                  output_file_base):

    # the number of worker processes we'll use in our process pool
    number_of_workers = multiprocessing.cpu_count()  # NOTE use 1 here when debugging for less butt hurt

    # create a process Pool for worker processes which will compute indices
    pool = multiprocessing.Pool(processes=number_of_workers)

    # create an iterable of arguments specific to the function that we'll call within each worker process
    args = []
    for scale in scales:

        for dist in indices.Distribution:

            # keyword arguments used for the function we'll map
            kwrgs = {'netcdf_precip': netcdf_precip,
                     'var_name_precip': var_name_precip,
                     'scale': scale,
                     'distribution': dist,
                     'periodicity': periodicity,
                     'calibration_start_year': calibration_start_year,
                     'calibration_end_year': calibration_end_year,
                     'output_file_base': output_file_base}
            args.append(kwrgs)

    # map the arguments iterable to the compute function
    result = pool.map_async(compute_write_spi, args)

    # get/swallow the exception(s) thrown, if any
    result.get()

    # close the pool and wait on all processes to finish
    pool.close()
    pool.join()


# ----------------------------------------------------------------------------------------------------------------------
def run_multi_spei(netcdf_precip,
                   var_name_precip,
                   netcdf_pet,
                   var_name_pet,
                   scales,
                   periodicity,
                   calibration_start_year,
                   calibration_end_year,
                   output_file_base):

    # the number of worker processes we'll use in our process pool
    number_of_workers = multiprocessing.cpu_count()  # NOTE use 1 here when debugging for less butt hurt

    # create a process Pool for worker processes which will compute indices
    pool = multiprocessing.Pool(processes=number_of_workers)

    # create an iterable of arguments specific to the function that we'll call within each worker process
    args = []
    for scale in scales:

        for dist in indices.Distribution:

            # keyword arguments used for the function we'll map
            kwrgs = {'netcdf_precip': netcdf_precip,
                     'var_name_precip': var_name_precip,
                     'netcdf_pet': netcdf_pet,
                     'var_name_pet': var_name_pet,
                     'scale': scale,
                     'distribution': dist,
                     'periodicity': periodicity,
                     'calibration_start_year': calibration_start_year,
                     'calibration_end_year': calibration_end_year,
                     'output_file_base': output_file_base}
            args.append(kwrgs)

    # map the arguments iterable to the compute function
    result = pool.map_async(compute_write_spei, args)

    # get/swallow the exception(s) thrown, if any
    result.get()

    # close the pool and wait on all processes to finish
    pool.close()
    pool.join()


# ----------------------------------------------------------------------------------------------------------------------
def _prepare_file(netcdf_file,
                  var_name):

    # determine if coordinates are correctly ordered in ascending order
    ds = xr.open_dataset(netcdf_file)

    # make sure we have lat, lon, and time as variable dimensions
    expected_dims = ('lat', 'lon', 'time')
    if Counter(ds[var_name].dims) != Counter(expected_dims):
        message = "Invalid dimensions for precipitation " \
                  "variable: {dims}".format(dims=ds[var_name].dims)
        _logger.error(message)
        raise ValueError(message)

    # see if we need to reorder into (lat,lon,time)
    reorder_dims = (ds[var_name].dims != expected_dims)

    # see if we need to reverse the lat and/or lon dimensions
    dims = []
    reverse_dims = False
    for dim_name in ['lat', 'lon']:
        vals = ds[dim_name].values
        if np.all(vals[:-1] <= vals[1:]):
            dims.append(dim_name)
        else:
            reverse_dims = True
            dims.append("-" + dim_name)
    dims.append('time')

    # perform reorder and/or reversal of dimensions if necessary
    if reorder_dims or reverse_dims:
        dims = ','.join(dims)
        nco = Nco()
        netcdf_file = nco.ncpdq(input_file=netcdf_file,
                                options=["-a \\\"{dims}\\\"".format(dims=dims), "-O"])

    return netcdf_file


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
    --output_file_base ~/data/test/spi/nclimgrid_lowres 
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
        if arguments.index in ['spi', 'scaled', 'all']:

            # prepare precipitation NetCDF in case dimensions not (lat, lon, time) or if any coordinates are descending
            netcdf_precip = _prepare_file(arguments.netcdf_precip,
                                          arguments.var_name_precip)

            # run SPI with one process per scale/distribution
            run_multi_spi(netcdf_precip,
                          arguments.var_name_precip,
                          arguments.scales,
                          arguments.periodicity,
                          arguments.calibration_start_year,
                          arguments.calibration_end_year,
                          arguments.output_file_base)

            # remove temporary file
            if netcdf_precip != arguments.netcdf_precip:
                os.remove(netcdf_precip)

        if arguments.index in ['pet', 'spei', 'scaled', 'palmers', 'all']:

            # run SPI with one process per scale/distribution
            if arguments.netcdf_pet is None:

                # prepare temperature NetCDF in case dimensions not (lat, lon, time) or if coordinates are descending
                netcdf_temp = _prepare_file(arguments.netcdf_temp,
                                            arguments.var_name_temp)

                # keyword arguments used for the function we'll map
                kwargs = {'netcdf_temp': netcdf_temp,
                          'var_name_temp': arguments.var_name_temp,
                          'output_file_base': arguments.output_file_base}

                arguments.netcdf_pet, arguments.var_name_pet = compute_write_pet(kwargs)

                # remove temporary file
                if netcdf_temp != arguments.netcdf_temp:
                    os.remove(netcdf_temp)

        if arguments.index in ['spei', 'scaled', 'all']:

            # prepare NetCDFs in case dimensions not (lat, lon, time) or if any coordinates are descending
            netcdf_precip = _prepare_file(arguments.netcdf_precip,
                                          arguments.var_name_precip)
            netcdf_pet = _prepare_file(arguments.netcdf_pet,
                                       arguments.var_name_pet)

            run_multi_spei(netcdf_precip,
                           arguments.var_name_precip,
                           netcdf_pet,
                           arguments.var_name_pet,
                           arguments.scales,
                           arguments.periodicity,
                           arguments.calibration_start_year,
                           arguments.calibration_end_year,
                           arguments.output_file_base)

            # remove temporary files
            if netcdf_precip != arguments.netcdf_precip:
                os.remove(netcdf_precip)
            if netcdf_pet != arguments.netcdf_pet:
                os.remove(netcdf_pet)

        if arguments.index in ['pnp', 'scaled', 'all']:

            # prepare NetCDF in case dimensions not (lat, lon, time) or if any coordinates are descending
            netcdf_precip = _prepare_file(arguments.netcdf_precip,
                                          arguments.var_name_precip)

            run_multi_pnp(netcdf_precip,
                          arguments.var_name_precip,
                          arguments.scales,
                          arguments.periodicity,
                          arguments.calibration_start_year,
                          arguments.calibration_end_year,
                          arguments.output_file_base)

            # remove temporary files
            if netcdf_precip != arguments.netcdf_precip:
                os.remove(netcdf_precip)

        if arguments.index in ['palmers', 'all']:

            # TODO prepare input NetCDF files, ensure matching dimensions, etc.

            # keyword arguments used for the function we'll map
            kwargs = {'netcdf_precip': arguments.netcdf_precip,
                      'var_name_precip': arguments.var_name_precip,
                      'netcdf_pet': arguments.netcdf_pet,
                      'var_name_pet': arguments.var_name_pet,
                      'netcdf_awc': arguments.netcdf_awc,
                      'var_name_awc': arguments.var_name_awc,
                      'calibration_start_year': arguments.calibration_start_year,
                      'calibration_end_year': arguments.calibration_end_year,
                      'periodicity': arguments.periodicity,
                      'output_file_base': arguments.output_file_base}

            compute_write_palmers(kwargs)

        # report on the elapsed time
        end_datetime = datetime.now()
        _logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        _logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        _logger.exception('Failed to complete', exc_info=True)
        raise
