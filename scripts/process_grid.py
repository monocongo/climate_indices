import argparse
from datetime import datetime
import logging
import math
import multiprocessing
import warnings

import netCDF4
import numpy as np

from climate_indices import compute, indices, utils
from scripts import netcdf_utils

# ----------------------------------------------------------------------------------------------------------------------
# static constants
_VALID_MIN = -10.0
_VALID_MAX = 10.0
_POSSIBLE_MM_UNITS = ['millimeters', 'millimeter', 'mm']
_POSSIBLE_INCH_UNITS = ['inches', 'inch']
_MM_TO_INCHES_FACTOR = 0.0393701

# ----------------------------------------------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
_logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------
# multiprocessing locks we'll use to synchronize I/O writes to NetCDF files, one per each output file
pet_lock = multiprocessing.Lock()
pdsi_lock = multiprocessing.Lock()
phdi_lock = multiprocessing.Lock()
pmdi_lock = multiprocessing.Lock()
zindex_lock = multiprocessing.Lock()
scpdsi_lock = multiprocessing.Lock()
spi_gamma_lock = multiprocessing.Lock()
spi_pearson_lock = multiprocessing.Lock()
spei_gamma_lock = multiprocessing.Lock()
spei_pearson_lock = multiprocessing.Lock()
pnp_lock = multiprocessing.Lock()

# ignore runtime warnings
warnings.simplefilter('ignore', Warning)


# ----------------------------------------------------------------------------------------------------------------------
class GridProcessor(object):             # pragma: no cover

    def __init__(self,
                 index,
                 periodicity,
                 output_file_base,
                 calibration_start_year,
                 calibration_end_year,
                 netcdf_precip,
                 var_name_precip,
                 netcdf_temp=None,
                 var_name_temp=None,
                 netcdf_pet=None,
                 var_name_pet=None,
                 netcdf_awc=None,
                 var_name_awc=None,
                 scales=None):

        # assign member values
        self.output_file_base = output_file_base
        self.netcdf_precip = netcdf_precip
        self.netcdf_temp = netcdf_temp
        self.netcdf_pet = netcdf_pet
        self.netcdf_awc = netcdf_awc
        self.var_name_precip = var_name_precip
        self.var_name_temp = var_name_temp
        self.var_name_pet = var_name_pet
        self.var_name_awc = var_name_awc
        self.scales = scales
        self.calibration_start_year = calibration_start_year
        self.calibration_end_year = calibration_end_year        
        self.index = index
        self.periodicity = periodicity
            
        # determine the file to use for coordinate specs (years range and lat/lon sizes), get relevant units
        if self.index == 'pet':
            
            # to compute PET we require a temperature input dataset
            if self.netcdf_temp is not None:
                
                # a PET file was not provided and we'll compute PET from temperature
                self.units_temp = netcdf_utils.variable_units(self.netcdf_temp, self.var_name_temp)

                # use the temperature file as the file that specifies the coordinate specs
                coordinate_specs_file = self.netcdf_temp
                
            else:
                message = 'A temperature file was not specified, required for PET computation'
                _logger.error(message)
                raise ValueError(message)
            
        else:

            # all other indices require precipitation
            if self.netcdf_precip is not None:

                self.units_precip = netcdf_utils.variable_units(self.netcdf_precip, self.var_name_precip)
            
            else:
                message = 'A precipitation file was not specified, required for all indices except PET'
                _logger.error(message)
                raise ValueError(message)

            # use the precipitation file as the file that specifies the coordinate specs
            coordinate_specs_file = self.netcdf_precip

            # SPI and PNP only require precipitation
            if self.index not in ['spi', 'pnp']:
                
                # PET needs to be either provided or computed from temperature
                if self.netcdf_pet is not None:
                    # a PET file was provided
                    self.units_pet = netcdf_utils.variable_units(self.netcdf_pet, self.var_name_pet)
                elif self.netcdf_temp is not None:
                    # a PET file was not provided and we'll compute PET from temperature
                    self.units_temp = netcdf_utils.variable_units(self.netcdf_temp, self.var_name_temp)
                else:
                    message = "Neither a PET nor a temperature file was specified, " \
                              "required for all indices except SPI and PNP"
                    _logger.error(message)
                    raise ValueError(message)
                    
                # AWC is only required for Palmers
                if self.index == 'palmers':
                    
                    if self.netcdf_awc is not None:
                        self.units_awc = netcdf_utils.variable_units(self.netcdf_awc, self.var_name_awc)
                        self.fill_value_awc = netcdf_utils.variable_fillvalue(self.netcdf_awc, self.var_name_awc)
                    else:
                        message = 'No AWC file was specified, required for Palmers indices'
                        _logger.error(message)
                        raise ValueError(message)
                        
        # get the coordinate related specs (time range, lat and lon sizes)
        self.data_start_year, self.data_end_year = netcdf_utils.initial_and_final_years(coordinate_specs_file)
        self.lat_size, self.lon_size = netcdf_utils.lat_and_lon_sizes(coordinate_specs_file)
        
        # initialize the NetCDF files used for Palmers output, scaled indices will have 
        # corresponding files initialized at each scale run
        if self.index == 'palmers':
        
            # place holders for the scaled NetCDFs, these files will be created 
            # and assigned to these variables at each scale's computational iteration
            self.netcdf_pdsi = self.output_file_base + '_pdsi.nc'
            self.netcdf_phdi = self.output_file_base + '_phdi.nc'
            self.netcdf_pmdi = self.output_file_base + '_pmdi.nc'
            self.netcdf_scpdsi = self.output_file_base + '_scpdsi.nc'
            self.netcdf_zindex = self.output_file_base + '_zindex.nc'
            
            netcdf_utils.initialize_netcdf_single_variable_grid(self.netcdf_pdsi,
                                                                self.netcdf_precip,
                                                                'pdsi',
                                                                'Palmer Drought Severity Index',
                                                                _VALID_MIN,
                                                                _VALID_MAX)
            netcdf_utils.initialize_netcdf_single_variable_grid(self.netcdf_phdi,
                                                                self.netcdf_precip,
                                                                'phdi',
                                                                'Palmer Hydrological Drought Index',
                                                                _VALID_MIN,
                                                                _VALID_MAX)
            netcdf_utils.initialize_netcdf_single_variable_grid(self.netcdf_pmdi,
                                                                self.netcdf_precip,
                                                                'pmdi',
                                                                'Palmer Modified Drought Index',
                                                                _VALID_MIN,
                                                                _VALID_MAX)
            netcdf_utils.initialize_netcdf_single_variable_grid(self.netcdf_scpdsi,
                                                                self.netcdf_precip,
                                                                'scpdsi',
                                                                'Self-calibrated Palmer Drought Severity Index',
                                                                _VALID_MIN,
                                                                _VALID_MAX)
            netcdf_utils.initialize_netcdf_single_variable_grid(self.netcdf_zindex,
                                                                self.netcdf_precip,
                                                                'zindex',
                                                                'Palmer Z-Index',
                                                                _VALID_MIN,
                                                                _VALID_MAX)

        elif self.index in ['spi', 'spei', 'pnp', 'scaled']:
        
            # place holders for the scaled NetCDFs, these files will be created as needed
            # and assigned to these variables at each scale's computational iteration
            self.netcdf_spi_gamma = ''
            self.netcdf_spi_pearson = ''
            self.netcdf_spei_gamma = ''
            self.netcdf_spei_pearson = ''
            self.netcdf_pnp = ''

    # ------------------------------------------------------------------------------------------------------------------
    def _initialize_scaled_netcdfs(self):

        # dictionary of index types to the NetCDF dataset files corresponding to the base index names and
        # day scales (this is the object we'll build and return from this function)
        netcdfs = {}

        # make a scale type substring to use within the variable long_name attributes
        scale_type = str(self.timestep_scale) + '-month scale'
        if self.periodicity is compute.Periodicity.daily:
            if self.index in ['spi', 'pnp']:
                scale_type = str(self.timestep_scale) + '-day scale'
            else:
                message = 'Incompatible periodicity -- only SPI and PNP are supported for daily time series'
                _logger.error(message)
                raise ValueError(message)
        elif self.periodicity is not compute.Periodicity.monthly:
            raise ValueError('Unsupported periodicity argument: %s' % self.periodicity)
        
        # dictionary of index types (ex. 'spi_gamma', 'spei_pearson', etc.) mapped to their corresponding long 
        # variable names, to be used within the respective NetCDFs as variable long_name attributes
        names_to_longnames = {}            
        if self.index == 'spi':
            names_to_longnames['spi_gamma'] = 'Standardized Precipitation Index (Gamma distribution), '
            names_to_longnames['spi_pearson'] = 'Standardized Precipitation Index (Pearson Type III distribution), '
        elif self.index == 'spei':
            names_to_longnames['spei_gamma'] = 'Standardized Precipitation Evapotranspiration ' \
                                               'Index (Gamma distribution), '
            names_to_longnames['spei_pearson'] = 'Standardized Precipitation Evapotranspiration ' \
                                                 'Index (Pearson Type III distribution), '
        elif self.index == 'pnp':
            names_to_longnames['pnp'] = 'Percentage of Normal Precipitation, '
        elif self.index == 'scaled':
            names_to_longnames['spi_gamma'] = 'Standardized Precipitation Index (Gamma distribution), '
            names_to_longnames['spi_pearson'] = 'Standardized Precipitation Index (Pearson Type III distribution), '
            names_to_longnames['spei_gamma'] = 'Standardized Precipitation Evapotranspiration ' \
                                               'Index (Gamma distribution), '
            names_to_longnames['spei_pearson'] = 'Standardized Precipitation Evapotranspiration ' \
                                                 'Index (Pearson Type III distribution), '
            names_to_longnames['pnp'] = 'Percentage of Normal Precipitation, '
        else:
            raise ValueError('Unsupported index: %s', self.index)

        # add the scale type to the end of each variable's long name
        names_to_longnames = {k: str(v + scale_type) for (k, v) in names_to_longnames.items()}
        
        # loop over the indices, creating an output NetCDF dataset for each
        for index_name, long_name in names_to_longnames.items():

            # use a separate valid min/max for PNP than for the other SP* indices
            if index_name == 'pnp':
                valid_min = np.float32(-1000.0)
                valid_max = np.float32(1000.0)
            else:
                valid_min = np.float32(-3.09)
                valid_max = np.float32(3.09)

            # create the variable name from the index and day scale
            variable_name = index_name + '_{0}'.format(str(self.timestep_scale).zfill(2))

            # create the NetCDF file path from the
            netcdf_file = self.output_file_base + '_' + variable_name + '.nc'

            # initialize the output NetCDF
            netcdf_utils.initialize_netcdf_single_variable_grid(netcdf_file,
                                                                self.netcdf_precip,
                                                                variable_name,
                                                                long_name.format(self.timestep_scale),
                                                                valid_min,
                                                                valid_max)

            # add the days scale index's NetCDF to the dictionary for the current index
            netcdfs[index_name] = netcdf_file

        # assign the NetCDF file paths to the corresponding member variables
        if self.index == 'spi':
            self.netcdf_spi_gamma = netcdfs['spi_gamma']
            self.netcdf_spi_pearson = netcdfs['spi_pearson']
        elif self.index == 'spei':
            self.netcdf_spei_gamma = netcdfs['spei_gamma']
            self.netcdf_spei_pearson = netcdfs['spei_pearson']
        elif self.index == 'pnp':
            self.netcdf_pnp = netcdfs['pnp']
        elif self.index == 'scaled':
            self.netcdf_spi_gamma = netcdfs['spi_gamma']
            self.netcdf_spi_pearson = netcdfs['spi_pearson']
            self.netcdf_spei_gamma = netcdfs['spei_gamma']
            self.netcdf_spei_pearson = netcdfs['spei_pearson']
            self.netcdf_pnp = netcdfs['pnp']

    # ------------------------------------------------------------------------------------------------------------------
    def run(self):

        # the number of worker processes we'll have in our process pool
        number_of_workers = multiprocessing.cpu_count()   # use 1 here for debugging
    
        # all index combinations/bundles except SPI and PNP will require PET, so compute it here if required
        if (self.netcdf_pet is None) and (self.index in ['pet', 'spei', 'scaled', 'palmers']):
        
            self.netcdf_pet = self.output_file_base + '_pet.nc'
            netcdf_utils.initialize_netcdf_single_variable_grid(self.netcdf_pet,
                                                                self.netcdf_temp,
                                                                'pet',
                                                                'Potential Evapotranspiration',
                                                                0.0,
                                                                10000.0,
                                                                'millimeters')

            # create a process Pool for worker processes which will compute indices over an entire latitude slice
            pool = multiprocessing.Pool(processes=number_of_workers)

            # map the latitude indices as an arguments iterable to the compute function
            result = pool.map_async(self._process_latitude_pet, range(self.lat_size))
    
            # get the exception(s) thrown, if any
            result.get()
    
            # close the pool and wait on all processes to finish
            pool.close()
            pool.join()

        # compute indices other than PET if requested
        if self.index != 'pet':
            
            if self.index in ['spi', 'spei', 'pnp', 'scaled']:
                
                for scale in self.scales:
                    
                    self.timestep_scale = scale
                    
                    self._initialize_scaled_netcdfs()
                    
                    # create a process Pool for worker processes which will compute indices over a full latitude slice
                    pool = multiprocessing.Pool(processes=number_of_workers)

                    # map the latitude indices as an arguments iterable to the compute function
                    result = pool.map_async(self._process_latitude_scaled, range(self.lat_size))
            
                    # get the exception(s) thrown, if any
                    result.get()
            
                    # close the pool and wait on all processes to finish
                    pool.close()
                    pool.join()
                
            elif self.index == 'palmers':
    
                # create a process Pool for worker processes which will compute indices over an entire latitude slice
                pool = multiprocessing.Pool(processes=number_of_workers)

                # map the latitude indices as an arguments iterable to the compute function
                result = pool.map_async(self._process_latitude_palmers, range(self.lat_size))
        
                # get the exception(s) thrown, if any
                result.get()
        
                # close the pool and wait on all processes to finish
                pool.close()
                pool.join()
                
            else:
                            
                raise ValueError('Unsupported index argument: %s' % self.index)

    # ------------------------------------------------------------------------------------------------------------------
    def _process_latitude_scaled(self, lat_index):
        """
        Processes the relevant scaled indices for a single latitude slice at a single scale.

        :param lat_index: the latitude index of the latitude slice that will be read from NetCDF, computed, and written
        """

        # open the precipitation NetCDF within a context manager
        with netCDF4.Dataset(self.netcdf_precip) as dataset_precip:

            # read the latitude slice of input precipitation
            lat_slice_precip = dataset_precip[self.var_name_precip][lat_index, :, :]   # assuming (lat, lon, time)

        if self.periodicity is compute.Periodicity.daily:

            # times are daily, transform to all leap year times (i.e. 366 days per year), 
            # so we fill Feb. 29th of each non-leap missing
            # TODO / FIXME move this up/out of here, should only need to be computed once
            total_years = self.data_end_year - self.data_start_year + 1

            # allocate an array to hold transformed time series where all years contain 366 days
            original_days_count = lat_slice_precip.shape[1]
            lat_slice_precip_all_leap = np.full((self.lon_size, total_years * 366), np.NaN)
            
            # at each longitude we have a time series of values, loop over these longitudes and transform each
            # corresponding time series to 366 day years representation (fill Feb 29 during non-leap years)
            # TODO work out how to apply this across the lon axis, to eliminate this loop (np.apply_along_axis()?)
            for lon_index in range(self.lon_size):
                
                # transform the data so it represents all years containing 366 days, 
                # with the Feb. 29th element containing a fill value during non-leap years
                lat_slice_precip_all_leap[lon_index, :] = utils.transform_to_366day(lat_slice_precip[lon_index, :],
                                                                                    self.data_start_year,
                                                                                    total_years)

            # use the all leap daily values as the latitude slice we'll work on
            lat_slice_precip = lat_slice_precip_all_leap
            
        # compute PNP if specified
        if self.index in ['pnp', 'scaled']:

            if self.periodicity is compute.Periodicity.daily:
                scale_increment = 'day'
            elif self.periodicity is compute.Periodicity.monthly:
                scale_increment = 'month'
            else:
                raise ValueError("Invalid periodicity: {}".format(self.periodicity))

            message = "Computing {scale}-{incr} ".format(scale=self.timestep_scale, incr=scale_increment) + \
                      "{index} for latitude index {lat}".format(index='PNP', lat=lat_index)
            _logger.info(message)

            # compute PNP across all longitudes of the latitude slice
            lat_slice_pnp = np.apply_along_axis(indices.percentage_of_normal,
                                                1,
                                                lat_slice_precip,
                                                self.timestep_scale,
                                                self.data_start_year,
                                                self.calibration_start_year,
                                                self.calibration_end_year,
                                                self.periodicity)

            if self.periodicity is compute.Periodicity.daily:

                # at each longitude we have a time series of values with a 366 day per year representation
                # (Feb. 29 during non-leap years is a fill value), loop over these longitudes and transform
                # each corresponding time series back to a normal Gregorian calendar
                lat_slice_pnp_gregorian = np.full((self.lon_size, original_days_count), np.NaN)
                for lon_index in range(lat_slice_precip.shape[0]):
                    
                    # transform the data so it represents mixed leap and non-leap years, i.e. normal Gregorian calendar
                    lat_slice_pnp_gregorian[lon_index, :] = utils.transform_to_gregorian(lat_slice_pnp[lon_index, :],
                                                                                         self.data_start_year)

                # use the transformed arrays as the lat slice we'll write to the output NetCDF
                lat_slice_pnp = lat_slice_pnp_gregorian
            
            # use relevant variable name
            pnp_variable_name = 'pnp_' + str(self.timestep_scale).zfill(2)

            # open the existing PNP NetCDF file for writing, copy the latitude 
            # slice into the PNP variable at the indexed latitude position
            pnp_lock.acquire()
            pnp_dataset = netCDF4.Dataset(self.netcdf_pnp, mode='a')
            pnp_dataset[pnp_variable_name][lat_index, :, :] = lat_slice_pnp   # assuming (lat, lon, time) orientation
            pnp_dataset.sync()
            pnp_dataset.close()
            pnp_lock.release()

        # compute SPI if specified
        if self.index in ['spi', 'scaled']:

            if self.periodicity is compute.Periodicity.daily:
                scale_increment = 'day'
            elif self.periodicity is compute.Periodicity.monthly:
                scale_increment = 'month'
            else:
                raise ValueError("Invalid periodicity: {}".format(self.periodicity))

            _logger.info('Computing {scale}-{incr} {index} for latitude index {lat}'.format(scale=self.timestep_scale,
                                                                                            incr=scale_increment, 
                                                                                            index='SPI', 
                                                                                            lat=lat_index))

            # compute SPI/Gamma across all longitudes of the latitude slice
            spi_gamma_lat_slice = np.apply_along_axis(indices.spi,
                                                      1,
                                                      lat_slice_precip,
                                                      self.timestep_scale,
                                                      indices.Distribution.gamma,
                                                      self.data_start_year,
                                                      self.calibration_start_year,
                                                      self.calibration_end_year,
                                                      self.periodicity)

            # compute SPI/Pearson across all longitudes of the latitude slice
            spi_pearson_lat_slice = np.apply_along_axis(indices.spi,
                                                        1,
                                                        lat_slice_precip,
                                                        self.timestep_scale,
                                                        indices.Distribution.pearson,
                                                        self.data_start_year,
                                                        self.calibration_start_year,
                                                        self.calibration_end_year,
                                                        self.periodicity)

            if self.periodicity is compute.Periodicity.daily:

                # at each longitude we have a time series of values with a 366 day per year representation 
                # (Feb. 29 during non-leap years is a fill value), loop over these longitudes and transform each 
                # corresponding time series back to a normal Gregorian calendar
                lat_slice_spi_gamma = np.full((self.lon_size, original_days_count), np.NaN)
                lat_slice_spi_pearson = np.full((self.lon_size, original_days_count), np.NaN)
                for lon_index in range(lat_slice_precip.shape[0]):
                    
                    # transform the data so it represents mixed leap and non-leap years, i.e. normal Gregorian calendar
                    lat_slice_spi_gamma[lon_index, :] = utils.transform_to_gregorian(spi_gamma_lat_slice[lon_index, :],
                                                                                     self.data_start_year)
                    lat_slice_spi_pearson[lon_index, :] = \
                        utils.transform_to_gregorian(spi_pearson_lat_slice[lon_index, :], self.data_start_year)

                # use these transformed arrays as the lat slices we'll write to the output NetCDF
                spi_gamma_lat_slice = lat_slice_spi_gamma
                spi_pearson_lat_slice = lat_slice_spi_pearson

            # use relevant variable names
            spi_gamma_variable_name = 'spi_gamma_' + str(self.timestep_scale).zfill(2)
            spi_pearson_variable_name = 'spi_pearson_' + str(self.timestep_scale).zfill(2)

            # open the existing SPI/Gamma NetCDF file for writing, copy the latitude
            # slice into the SPI variable at the indexed latitude position
            spi_gamma_lock.acquire()
            spi_gamma_dataset = netCDF4.Dataset(self.netcdf_spi_gamma, mode='a')
            spi_gamma_dataset[spi_gamma_variable_name][lat_index, :, :] = spi_gamma_lat_slice   # (lat, lon, time)
            spi_gamma_dataset.sync()
            spi_gamma_dataset.close()
            spi_gamma_lock.release()

            # open the existing SPI/Pearson NetCDF file for writing, copy the latitude 
            # slice into the SPI variable at the indexed latitude position
            spi_pearson_lock.acquire()
            spi_pearson_dataset = netCDF4.Dataset(self.netcdf_spi_pearson, mode='a')
            spi_pearson_dataset[spi_pearson_variable_name][lat_index, :, :] = spi_pearson_lat_slice   # (lat, lon, time)
            spi_pearson_dataset.sync()
            spi_pearson_dataset.close()
            spi_pearson_lock.release()

        # compute SPEI if specified
        if self.index in ['spei', 'scaled']:

            if self.periodicity is compute.Periodicity.daily:
                message = 'Daily SPEI not yet supported'
                _logger.error(message)
                raise ValueError(message)
            elif self.periodicity is compute.Periodicity.monthly:
                scale_increment = 'month'
                message = "Computing {scale}-{incr} ".format(scale=self.timestep_scale, incr=scale_increment) + \
                          "{index} for latitude index {lat}".format(index='SPEI', lat=lat_index)
                _logger.info(message)
                
            # open the PET NetCDF within a context manager (this PET file should be present, either provided initially
            # as a command line argument to the script or computed from temperature earlier in the processing chain)
            with netCDF4.Dataset(self.netcdf_pet) as dataset_pet:            
                
                # read the latitude slice of input PET
                lat_slice_pet = dataset_pet['pet'][lat_index, :, :]   # assuming (lat, lon, time) orientation

            # allocate latitude slices for SPEI output
            spei_gamma_lat_slice = np.full(lat_slice_precip.shape, np.NaN)
            spei_pearson_lat_slice = np.full(lat_slice_precip.shape, np.NaN)

            # compute SPEI for each longitude from the latitude slice where we have valid inputs
            for lon_index in range(self.lon_size):

                # get the time series values for this longitude
                precip_time_series = lat_slice_precip[lon_index, :]
                pet_time_series = lat_slice_pet[lon_index, :]

                # compute SPEI for the current longitude only if we have valid inputs
                if (not precip_time_series.mask.all()) and \
                   (not pet_time_series.mask.all()):

                    # compute SPEI/Gamma
                    spei_gamma_lat_slice[lon_index, :] = indices.spei(precip_time_series,
                                                                      self.timestep_scale,
                                                                      indices.Distribution.gamma,
                                                                      self.periodicity,
                                                                      self.data_start_year,
                                                                      self.calibration_start_year,
                                                                      self.calibration_end_year,
                                                                      pet_mm=pet_time_series)
               
                    # compute SPEI/Pearson
                    spei_pearson_lat_slice[lon_index, :] = indices.spei(precip_time_series,
                                                                        self.timestep_scale,
                                                                        indices.Distribution.pearson,
                                                                        self.periodicity,
                                                                        self.data_start_year,
                                                                        self.calibration_start_year,
                                                                        self.calibration_end_year,
                                                                        pet_mm=pet_time_series)

            # use relevant variable names
            spei_gamma_variable_name = 'spei_gamma_' + str(self.timestep_scale).zfill(2)
            spei_pearson_variable_name = 'spei_pearson_' + str(self.timestep_scale).zfill(2)

            # open the existing SPEI/Gamma NetCDF file for writing, copy the latitude slice 
            # into the SPEI variable at the indexed latitude position
            spei_gamma_lock.acquire()
            spei_gamma_dataset = netCDF4.Dataset(self.netcdf_spei_gamma, mode='a')
            spei_gamma_dataset[spei_gamma_variable_name][lat_index, :, :] = spei_gamma_lat_slice
            spei_gamma_dataset.sync()
            spei_gamma_dataset.close()
            spei_gamma_lock.release()

            # open the existing SPEI/Pearson NetCDF file for writing, copy the latitude slice
            # into the SPEI variable at the indexed latitude position
            spei_pearson_lock.acquire()
            spei_pearson_dataset = netCDF4.Dataset(self.netcdf_spei_pearson, mode='a')
            spei_pearson_dataset[spei_pearson_variable_name][lat_index, :, :] = spei_pearson_lat_slice
            spei_pearson_dataset.sync()
            spei_pearson_dataset.close()
            spei_pearson_lock.release()

    # ------------------------------------------------------------------------------------------------------------------
    def _process_latitude_pet(self, lat_index):
        """
        Processes PET for a single latitude slice.

        :param lat_index: the latitude index of the latitude slice that will be read from NetCDF, computed, and written
        """

        _logger.info('Computing %s PET for latitude index %s', self.periodicity, lat_index)

        # open the temperature NetCDF within a context manager
        with netCDF4.Dataset(self.netcdf_temp) as temp_dataset:

            # read the latitude slice of input temperature values
            temp_lat_slice = temp_dataset[self.var_name_temp][lat_index, :, :]   # assuming (lat, lon, time) orientation

            # TODO verify that values are in degrees Celsius, if not then convert
            
            # get the actual latitude value (assumed to be in degrees north) for the latitude slice
            latitude_degrees_north = temp_dataset['lat'][lat_index]

            if self.periodicity is compute.Periodicity.daily:

                pass  #placeholder until we work out daily Thornthwaite and/or Hargreaves
            
#                 # times are daily, transform to all leap year times (i.e. 366 days per year),
#                 # so we fill Feb 29th of each non-leap missing
#                 total_years = self.data_end_year - self.data_start_year + 1   # FIXME only needs to be computed once
#      
#                 # allocate an array to hold transformed time series where all years contain 366 days
#                 original_days_count = temp_lat_slice.shape[1]
#                 temp_lat_slice_all_leap = np.full((self.lon_size, total_years * 366), np.NaN)
#                 
#                 # at each longitude we have a time series of values, loop over these longitudes and transform each
#                 # corresponding time series to 366 day years representation (fill Feb 29 during non-leap years)
#                 for lon_index in range(self.lon_size):  # TODO eliminate loop, work out how to apply across lon axis
#                     
#                     # transform the data so it represents all years containing 366 days,
#                     # with Feb 29 containing fill value during non-leap years
#                     temp_lat_slice_all_leap[lon_index, :] = utils.transform_to_366day(temp_lat_slice[lon_index, :],
#                                                                                        self.data_start_year,
#                                                                                        total_years)
# 
#                 temp_lat_slice = temp_lat_slice_all_leap
# 
#                 # compute PET across all longitudes of the latitude slice
#                 pet_lat_slice = np.apply_along_axis(indices.pet_daily_hargreaves,
#                                                     0,
#                                                     temp_lat_slice,
#                                                     latitude_degrees=latitude_degrees_north,
#                                                     data_start_year=self.data_start_year)
#                 
#                 # at each longitude we have a time series of values with a 366 day per year representation
#                 # (Feb 29 during non-leap years is a fill value), loop over these longitudes and transform
#                 # each corresponding time series back to a normal Gregorian calendar
#                 lat_slice_pet = np.full((self.lon_size, original_days_count), np.NaN)
#                 for lon_index in range(pet_lat_slice.shape[0]):
#                     
#                     # transform data so it represents mixed leap and non-leap years, i.e. normal Gregorian calendar
#                     lat_slice_pet[lon_index, :] = utils.transform_to_gregorian(pet_lat_slice[lon_index, :],
#                                                                                self.data_start_year)
#                 pet_lat_slice = lat_slice_pet

            elif self.periodicity is compute.Periodicity.monthly:    # monthly

                # compute PET across all longitudes of the latitude slice
                pet_lat_slice = np.apply_along_axis(indices.pet,
                                                    1,
                                                    temp_lat_slice,
                                                    latitude_degrees=latitude_degrees_north,
                                                    data_start_year=self.data_start_year)

            else:
                raise ValueError("Invalid periodicity: {}".format(self.periodicity))

            # open the existing PET NetCDF file for writing, copy the latitude slice
            # into the PET variable at the indexed latitude position
            pet_lock.acquire()
            pet_dataset = netCDF4.Dataset(self.netcdf_pet, mode='a')
            # TODO make this more general to allow for other dimension orders, etc.
            pet_dataset['pet'][lat_index, :, :] = pet_lat_slice   # this assumes (lat, lon, time)
            pet_dataset.sync()
            pet_dataset.close()
            pet_lock.release()

    # ------------------------------------------------------------------------------------------------------------------
    def _process_latitude_palmers(self, lat_index):
        """
        Perform computation of Palmer indices on a latitude slice, i.e. all lat/lon locations for a single latitude.
        Each lat/lon will have its corresponding time series used as input, with a corresponding time series output
        for each index computed. The full latitude slice (all lons for the indexed lat) of index values will be written
        into the corresponding NetCDF.

        :param lat_index: index of the latitude in the NetCDF, valid range is [0..(total # of divisions - 1)]
        """

        _logger.info('Computing Palmers for latitude index %s', lat_index)

        # open the input NetCDFs
        with netCDF4.Dataset(self.netcdf_precip) as dataset_precip, \
             netCDF4.Dataset(self.netcdf_pet) as dataset_pet, \
             netCDF4.Dataset(self.netcdf_awc) as dataset_awc:

            # read the latitude slice of input precipitation and PET values
            # assumes (lat, lon, time) orientation
            lat_slice_precip = dataset_precip[self.var_name_precip][lat_index, :, :]    
            lat_slice_pet = dataset_pet[self.var_name_pet][lat_index, :, :]

            # determine the dimensionality of the AWC dataset, in case there is 
            # a missing time dimension, then get the AWC latitude slice accordingly
            awc_dims = dataset_awc[self.var_name_awc].dimensions
            if awc_dims == ('time', 'lat', 'lon'):
                awc_lat_slice = dataset_awc[self.var_name_awc][lat_index, :, 0].flatten()
            elif awc_dims == ('lat', 'lon'):
                awc_lat_slice = dataset_awc[self.var_name_awc][lat_index, :].flatten()
            else:
                message = 'Unable to read the available water capacity (AWC) values due to ' + \
                          'unsupported variable dimensions: {dims}'.format(dims=awc_dims)
                _logger.error(message)
                raise ValueError(message)
 
        # allocate arrays to contain a latitude slice of Palmer values
        pdsi_lat_slice = np.full(lat_slice_precip.shape, np.NaN)
        phdi_lat_slice = np.full(lat_slice_precip.shape, np.NaN)
        zindex_lat_slice = np.full(lat_slice_precip.shape, np.NaN)
        scpdsi_lat_slice = np.full(lat_slice_precip.shape, np.NaN)
        pmdi_lat_slice = np.full(lat_slice_precip.shape, np.NaN)
        
        # compute Palmer indices for each longitude from the latitude slice where we have valid inputs
        for lon_index in range(self.lon_size):
        
            # get the time series values for this longitude
            precip_time_series = lat_slice_precip[lon_index, :]
            pet_time_series = lat_slice_pet[lon_index, :]
            awc = awc_lat_slice[lon_index]
        
            # compute Palmer indices only if we have valid inputs
            if utils.is_data_valid(precip_time_series) and \
               utils.is_data_valid(pet_time_series) and \
               awc is not np.ma.masked and \
               not math.isnan(awc) and \
               not math.isclose(awc, self.fill_value_awc):
        
                # put precipitation and PET into inches, if not already
                if self.units_precip in _POSSIBLE_MM_UNITS:
                    precip_time_series = precip_time_series * _MM_TO_INCHES_FACTOR
                if self.units_pet in _POSSIBLE_MM_UNITS:
                    pet_time_series = pet_time_series * _MM_TO_INCHES_FACTOR
        
                # compute Palmer indices
                palmer_values = indices.scpdsi(precip_time_series,
                                               pet_time_series,
                                               awc,
                                               self.data_start_year,
                                               self.calibration_start_year,
                                               self.calibration_end_year)
        
                # add the values into the slice, first clipping all values to the valid range
                scpdsi_lat_slice[lon_index, :] = np.clip(palmer_values[0], _VALID_MIN, _VALID_MAX)
                pdsi_lat_slice[lon_index, :] = np.clip(palmer_values[1], _VALID_MIN, _VALID_MAX)
                phdi_lat_slice[lon_index, :] = np.clip(palmer_values[2], _VALID_MIN, _VALID_MAX)
                pmdi_lat_slice[lon_index, :] = np.clip(palmer_values[3], _VALID_MIN, _VALID_MAX)
                zindex_lat_slice[lon_index, :] = palmer_values[4]
        
        # open the existing PDSI NetCDF file for writing, copy the latitude slice 
        # into the PET variable at the indexed latitude position
        pdsi_lock.acquire()
        pdsi_dataset = netCDF4.Dataset(self.netcdf_pdsi, mode='a')
        pdsi_dataset['pdsi'][lat_index, :, :] = pdsi_lat_slice
        pdsi_dataset.sync()
        pdsi_dataset.close()
        pdsi_lock.release()
        
        # open the existing PHDI NetCDF file for writing, copy the latitude slice
        # into the PET variable at the indexed latitude position
        phdi_lock.acquire()
        phdi_dataset = netCDF4.Dataset(self.netcdf_phdi, mode='a')
        phdi_dataset['phdi'][lat_index, :, :] = phdi_lat_slice
        phdi_dataset.sync()
        phdi_dataset.close()
        phdi_lock.release()
        
        # open the existing Z-Index NetCDF file for writing, copy the latitude slice
        # into the Z-Index variable at the indexed latitude position
        zindex_lock.acquire()
        zindex_dataset = netCDF4.Dataset(self.netcdf_zindex, mode='a')
        zindex_dataset['zindex'][lat_index, :, :] = zindex_lat_slice
        zindex_dataset.sync()
        zindex_dataset.close()
        zindex_lock.release()
        
        # open the existing SCPDSI NetCDF file for writing, copy the latitude slice
        # into the scPDSI variable at the indexed latitude position
        scpdsi_lock.acquire()
        scpdsi_dataset = netCDF4.Dataset(self.netcdf_scpdsi, mode='a')
        scpdsi_dataset['scpdsi'][lat_index, :, :] = scpdsi_lat_slice
        scpdsi_dataset.sync()
        scpdsi_dataset.close()
        scpdsi_lock.release()
        
        # open the existing PMDI NetCDF file for writing, copy the latitude slice
        # into the PMDI variable at the indexed latitude position
        pmdi_lock.acquire()
        pmdi_dataset = netCDF4.Dataset(self.netcdf_pmdi, mode='a')
        pmdi_dataset['pmdi'][lat_index, :, :] = pmdi_lat_slice
        pmdi_dataset.sync()
        pmdi_dataset.close()
        pmdi_lock.release()


# ----------------------------------------------------------------------------------------------------------------------
def _validate_arguments(index,
                        periodicity,
                        netcdf_precip,
                        var_name_precip,
                        netcdf_temp=None,
                        var_name_temp=None,
                        netcdf_pet=None,
                        var_name_pet=None,
                        netcdf_awc=None,
                        var_name_awc=None,
                        scales=None):
    """
    Validate the processing settings to confirm that proper argument combinations have been provided.
    
    :param index:
    :param periodicity:
    :param netcdf_precip:
    :param var_name_precip:
    :param netcdf_temp:
    :param var_name_temp:
    :param netcdf_pet:
    :param var_name_pet:
    :param netcdf_awc:
    :param var_name_awc:
    :param scales:
    :return:
    :raise ValueError: if one or more of the command line arguments is invalid
    """
    
    # the dimensions we expect to find for each data variable (precipitation, temperature, and/or PET)
    expected_dimensions = ('lat', 'lon', 'time')
    
    # all indices except PET require a precipitation file
    if index != 'pet':
        
        # make sure a precipitation file was specified
        if netcdf_precip is None:
            msg = 'Missing the required precipitation file'
            _logger.error(msg)
            raise ValueError(msg)

        # make sure a precipitation variable name was specified
        if var_name_precip is None:
            message = "Missing precipitation variable name"
            _logger.error(message)
            raise ValueError(message)

        # validate the precipitation file itself        
        with netCDF4.Dataset(netcdf_precip) as dataset_precip:
            
            # make sure we have a valid precipitation variable name
            if var_name_precip not in dataset_precip.variables:
                message = "Invalid precipitation variable name: '{var}' ".format(var=var_name_precip) + \
                          "does not exist in precipitation file '{file}'".format(file=netcdf_precip)
                _logger.error(message)
                raise ValueError(message)
                
            # verify that the precipitation variable's dimensions are in the expected order
            dimensions = dataset_precip.variables[var_name_precip].dimensions
            if dimensions != expected_dimensions:
                message = "Invalid dimensions of the precipitation variable: {dims}, ".format(dims=dimensions) + \
                          "(expected names and order: {dims})".format(dims=expected_dimensions)
                _logger.error(message)
                raise ValueError(message)
            
            # get the sizes of the latitude and longitude coordinate variables
            lats_precip = dataset_precip.variables['lat'][:]
            lons_precip = dataset_precip.variables['lon'][:]
            times_precip = dataset_precip.variables['time'][:]

    else:
        
        # PET requires a temperature file
        if netcdf_temp is None:
            msg = 'Missing the required temperature file argument'
            _logger.error(msg)
            raise ValueError(msg)

        # don't allow a daily periodicity (yet, this will be possible once we have Hargreaves or a daily Thornthwaite)
        if periodicity is not compute.Periodicity.monthly:
            msg = "Invalid periodicity argument for PET: " + \
                "'{period}' -- only monthly is supported".format(period=periodicity)
            _logger.error(msg)
            raise ValueError(msg)
                            
    # SPEI and Palmers require either a PET file or a temperature file in order to compute PET  
    if index in ['spei', 'scaled', 'palmers']:
        
        if netcdf_temp is None: 
            
            if netcdf_pet is None:
                msg = 'Missing the required temperature or PET files, neither were provided'
                _logger.error(msg)
                raise ValueError(msg)
            
            # validate the PET file        
            with netCDF4.Dataset(netcdf_pet) as dataset_pet:
                
                # make sure we have a valid PET variable name
                if var_name_pet is None:
                    message = "Missing PET variable name"
                    _logger.error(message)
                    raise ValueError(message)
                elif var_name_pet not in dataset_pet.variables:
                    message = "Invalid PET variable name: '{var_name}' ".format(var_name=var_name_pet) + \
                              "does not exist in PET file '{file}'".format(file=netcdf_pet)
                    _logger.error(message)
                    raise ValueError(message)
                    
                # verify that the PET variable's dimensions are in the expected order
                dimensions = dataset_pet.variables[var_name_pet].dimensions
                if dimensions != expected_dimensions:
                    message = "Invalid dimensions of the PET variable: {dims}, ".format(dims=dimensions) + \
                              "(expected names and order: {dims})".format(dims=expected_dimensions)
                    _logger.error(message)
                    raise ValueError(message)
                
                # verify that the coordinate variables match with those of the precipitation dataset
                if not np.array_equal(lats_precip, dataset_pet.variables['lat'][:]):
                    message = "Precipitation and PET variables contain non-matching latitudes"
                    _logger.error(message)
                    raise ValueError(message)
                elif not np.array_equal(lons_precip, dataset_pet.variables['lon'][:]):
                    message = "Precipitation and PET variables contain non-matching longitudes"
                    _logger.error(message)
                    raise ValueError(message)
                elif not np.array_equal(times_precip, dataset_pet.variables['time'][:]):
                    message = "Precipitation and PET variables contain non-matching times"
                    _logger.error(message)
                    raise ValueError(message)

        elif netcdf_pet is not None: 

            # we can't have both temperature and PET files specified, no way to determine which to use            
            msg = 'Both temperature and PET files were specified, only one of these should be provided'
            _logger.error(msg)
            raise ValueError(msg)

        else:
            
            # validate the temperature file        
            with netCDF4.Dataset(netcdf_temp) as dataset_temp:
                
                # make sure we have a valid temperature variable name
                if var_name_temp is None:
                    message = "Missing temperature variable name"
                    _logger.error(message)
                    raise ValueError(message)
                elif var_name_temp not in dataset_temp.variables:
                    message = "Invalid temperature variable name: '{var}' does ".format(var=var_name_temp) + \
                              "not exist in temperature file '{file}'".format(file=netcdf_temp)
                    _logger.error(message)
                    raise ValueError(message)
                    
                # verify that the temperature variable's dimensions are in the expected order
                dimensions = dataset_temp.variables[var_name_temp].dimensions
                if dimensions != expected_dimensions:
                    message = "Invalid dimensions of the temperature variable: {dims}, ".format(dims=dimensions) + \
                              "(expected names and order: {dims})".format(dims=expected_dimensions)
                    _logger.error(message)
                    raise ValueError(message)
                
                # verify that the coordinate variables match with those of the precipitation dataset
                if not np.array_equal(lats_precip, dataset_temp.variables['lat'][:]):
                    message = "Precipitation and temperature variables contain non-matching latitudes"
                    _logger.error(message)
                    raise ValueError(message)
                elif not np.array_equal(lons_precip, dataset_temp.variables['lon'][:]):
                    message = "Precipitation and temperature variables contain non-matching longitudes"
                    _logger.error(message)
                    raise ValueError(message)
                elif not np.array_equal(times_precip, dataset_temp.variables['time'][:]):
                    message = "Precipitation and temperature variables contain non-matching times"
                    _logger.error(message)
                    raise ValueError(message)

        # Palmers requires an available water capacity file
        if index in ['palmers']:
        
            if netcdf_awc is None: 
                
                msg = 'Missing the required available water capacity file'
                _logger.error(msg)
                raise ValueError(msg)
                
            # validate the AWC file        
            with netCDF4.Dataset(netcdf_awc) as dataset_awc:
                
                # make sure we have a valid PET variable name
                if var_name_awc is None:
                    message = "Missing the AWC variable name"
                    _logger.error(message)
                    raise ValueError(message)
                elif var_name_awc not in dataset_awc.variables:
                    message = "Invalid AWC variable name: '{var}' does not exist ".format(var=var_name_awc) + \
                              "in AWC file '{file}'".format(file=netcdf_awc)
                    _logger.error(message)
                    raise ValueError(message)
                    
                # verify that the AWC variable's dimensions are in the expected order
                dimensions = dataset_awc.variables[var_name_awc].dimensions
                if (dimensions != ('lat', 'lon')) and (dimensions != expected_dimensions):
                    message = "Invalid dimensions of the AWC variable: {dims}, ".format(dims=dimensions) + \
                              "(expected names and order: {dims})".format(dims=expected_dimensions)
                    _logger.error(message)
                    raise ValueError(message)
                
                # verify that the lat and lon coordinate variables match with those of the precipitation dataset
                if not np.array_equal(lats_precip, dataset_awc.variables['lat'][:]):
                    message = "Precipitation and AWC variables contain non-matching latitudes"
                    _logger.error(message)
                    raise ValueError(message)
                elif not np.array_equal(lons_precip, dataset_awc.variables['lon'][:]):
                    message = "Precipitation and AWC variables contain non-matching longitudes"
                    _logger.error(message)
                    raise ValueError(message)

    if index in ['spi', 'spei', 'scaled', 'pnp']:
        
        if scales is None:
            message = "Scaled indices (SPI, SPEI, and/or PNP) specified without including " + \
                      "one or more time scales (missing --scales argument)"
            _logger.error(message)
            raise ValueError(message)
        
        if any(n < 0 for n in scales):
            message = "One or more negative scale specified within --scales argument"
            _logger.error(message)
            raise ValueError(message)


# ----------------------------------------------------------------------------------------------------------------------
def process_grid(index,
                 periodicity,
                 output_file_base,
                 calibration_start_year,
                 calibration_end_year,
                 netcdf_precip,
                 var_name_precip,
                 netcdf_temp=None,
                 var_name_temp=None,
                 netcdf_pet=None,
                 var_name_pet=None,
                 netcdf_awc=None,
                 var_name_awc=None,
                 scales=None):
    
    # validate the arguments
    _validate_arguments(index,
                        periodicity,
                        netcdf_precip,
                        var_name_precip,
                        netcdf_temp,
                        var_name_temp,
                        netcdf_pet,
                        var_name_pet,
                        netcdf_awc,
                        var_name_awc,
                        scales)
                
    # instantiate and run a grid processor object
    grid_processor = GridProcessor(index,
                                   periodicity,
                                   output_file_base,
                                   calibration_start_year,
                                   calibration_end_year,
                                   netcdf_precip,
                                   var_name_precip,
                                   netcdf_temp,
                                   var_name_temp,
                                   netcdf_pet,
                                   var_name_pet,
                                   netcdf_awc,
                                   var_name_awc,
                                   scales)
    grid_processor.run()


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    This module is used to perform climate indices processing on gridded datasets in NetCDF.

    Example command line arguments for SPI only using monthly precipitation input:
    
    --index spi 
    --periodicity monthly
    --scales 1 2 3 6 9 12 24 
    --calibration_start_year 1998 
    --calibration_end_year 2016 
    --netcdf_precip example_data/nclimgrid_prcp_lowres.nc 
    --var_name_precip prcp 
    --output_file_base ~/data/test/nclimgrid_lowres 
    """

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        _logger.info("Start time:    %s", start_datetime)

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--netcdf_precip",
                            help="Precipitation NetCDF file to be used as input")
        parser.add_argument("--var_name_precip",
                            help="Precipitation variable name used in the precipitation NetCDF file")
        parser.add_argument("--output_file_base",
                            help="Base output file path and name for the resulting output files",
                            required=True)
        parser.add_argument("--calibration_start_year",
                            help="Initial year of the calibration period",
                            type=int)
        parser.add_argument("--calibration_end_year",
                            help="Final year of calibration period",
                            type=int)
        parser.add_argument("--index",
                            help="Indices to compute",
                            choices=['spi', 'spei', 'pnp', 'scaled', 'pet', 'palmers'],
                            required=True)
        parser.add_argument("--periodicity",
                            help="Process input as either monthly or daily values",
                            choices=[compute.Periodicity.monthly, compute.Periodicity.daily],
                            type=compute.Periodicity.from_string,
                            required=True)
        parser.add_argument("--netcdf_temp",
                            help="Temperature NetCDF file to be used as input")
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
        parser.add_argument("--scales",
                            help="Timestep scales over which the PNP, SPI, and SPEI values are to be computed",
                            type=int,
                            nargs ='*')
        args = parser.parse_args()

        process_grid(args.index,
                     args.periodicity,
                     args.output_file_base,
                     args.calibration_start_year,
                     args.calibration_end_year,
                     args.netcdf_precip,
                     args.var_name_precip,
                     args.netcdf_temp,
                     args.var_name_temp,
                     args.netcdf_pet,
                     args.var_name_pet,
                     args.netcdf_awc,
                     args.var_name_awc,
                     args.scales)
        
        # report on the elapsed time
        end_datetime = datetime.now()
        _logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        _logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        _logger.exception('Failed to complete', exc_info=True)
        raise
