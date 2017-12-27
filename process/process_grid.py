import argparse
from datetime import datetime
import indices
import logging
import math
import multiprocessing
import netCDF4
import netcdf_utils
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------
# static constants
_VALID_MIN = -10.0
_VALID_MAX = 10.0

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
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

# ignore warnings
import warnings
warnings.simplefilter('ignore', Warning)

#-----------------------------------------------------------------------------------------------------------------------
class GridProcessor(object):             # pragma: no cover

    def __init__(self,
                 output_file_base,
                 netcdf_precip,
                 netcdf_temperature,
                 netcdf_soil,
                 var_name_precip,
                 var_name_temperature,
                 var_name_soil,
                 scale_months,
                 calibration_start_year,
                 calibration_end_year):

        self.output_file_base = output_file_base
        self.netcdf_precip = netcdf_precip
        self.netcdf_temperature = netcdf_temperature
        self.netcdf_soil = netcdf_soil
        self.var_name_precip = var_name_precip
        self.var_name_temperature = var_name_temperature
        self.var_name_soil = var_name_soil
        self.scale_months = scale_months
        self.calibration_start_year = calibration_start_year
        self.calibration_end_year = calibration_end_year        
        
        # TODO get the initial year from the precipitation NetCDF, for now use hard-coded value specific to nClimGrid
        self.data_start_year = 1895

        # the number of months used in scaled indices (for example this will be set to 6 for 6-month SPI, SPEI, and PNP)
        # this will need to be reset before the object is used for computing the scaled indices
        self.months = -1

        # place holders for the scaled NetCDFs, these should be created and assigned here for each months scale
        self.netcdf_spi_gamma = ''
        self.netcdf_spei_gamma = ''
        self.netcdf_spi_pearson = ''
        self.netcdf_spei_pearson = ''
        self.netcdf_pnp = ''

        # initialize the NetCDFs to be used as output files for the unscaled indices (Palmers and PET)
        self._initialize_unscaled_netcdfs(self.output_file_base,
                                          self.netcdf_precip)

    #-------------------------------------------------------------------------------------------------------------------
    def _validate_compatibility(self,
                                precipitation_dataset,
                                precipitation_var_name,
                                temperature_dataset,
                                temperature_var_name,
                                soil_dataset,
                                soil_var_name):

        # get the time, lat, and lon variables from the three datasets we want to validate against each other
        precip_time = precipitation_dataset.variables['time']
        precip_lat = precipitation_dataset.variables['lat']
        precip_lon = precipitation_dataset.variables['lon']
        temp_time = temperature_dataset.variables['time']
        temp_lat = temperature_dataset.variables['lat']
        temp_lon = temperature_dataset.variables['lon']
        awc_lat = soil_dataset.variables['lat']
        awc_lon = soil_dataset.variables['lon']

        # dataset names to be used in error messages
        precip_dataset_name = 'precipitation'
        temp_dataset_name = 'temperature'
        awc_dataset_name = 'available water capacity'

        # make sure that the datasets match in terms of coordinate variables
        if not np.allclose(precip_time[:], temp_time[:]):
            message = 'Mismatch of the time dimension between the {0} and {1} datasets'.format(precip_dataset_name,
                                                                                               temp_dataset_name)
            logger.error(message)
            raise ValueError(message)
        if not np.allclose(precip_lat[:], temp_lat[:]):
            message = 'Mismatch of the lat dimension between the {0} and {1} datasets'.format(precip_dataset_name,
                                                                                              temp_dataset_name)
            logger.error(message)
            raise ValueError(message)
        if not np.allclose(precip_lat[:], awc_lat[:], atol=1e-05, equal_nan=True):
            message = 'Mismatch of the lat dimension between the {0} and {1} datasets'.format(precip_dataset_name,
                                                                                              awc_dataset_name)
            logger.error(message)
            raise ValueError(message)
        if not np.allclose(precip_lon[:], temp_lon[:]):
            message = 'Mismatch of the lon dimension between the {0} and {1} datasets'.format(precip_dataset_name,
                                                                                              temp_dataset_name)
            logger.error(message)
            raise ValueError(message)
        if not np.allclose(precip_lon[:], awc_lon[:], atol=1e-05, equal_nan=True):
            message = 'Mismatch of the lon dimension between the {0} and {1} datasets'.format(precip_dataset_name,
                                                                                              awc_dataset_name)
            logger.error(message)
            raise ValueError(message)

        # make sure that each variable has (time, lat, lon) dimensions, in that order
        expected_dimensions = ('time', 'lat', 'lon')
        if not temperature_dataset.variables[temperature_var_name].dimensions == expected_dimensions:
            message = 'Unexpected dimensions for the {0} variable of the {1} dataset: {2}\nExpected dimensions are (\'time\', \'lat\', \'lon\')'.format(temperature_var_name, temp_dataset_name, temperature_dataset.variables[temperature_var_name].dimensions)
            logger.error(message)
            raise ValueError(message)
        if not precipitation_dataset.variables[precipitation_var_name].dimensions == expected_dimensions:
            message = 'Unexpected dimensions for the {0} variable of the {1} dataset: {2}\nExpected dimensions are (\'time\', \'lat\', \'lon\')'.format(precipitation_var_name, precip_dataset_name, precipitation_dataset.variables[precipitation_var_name].dimensions)
            logger.error(message)
            raise ValueError(message)
        if (not soil_dataset.variables[soil_var_name].dimensions == expected_dimensions) and \
            (not soil_dataset.variables[soil_var_name].dimensions == ('lat', 'lon')) and \
            (not soil_dataset.variables[soil_var_name].dimensions == ('lon', 'lat')):
            message = 'Unexpected dimensions for the {0} variable of the {1} dataset: {2}\nExpected dimensions are (\'time\', \'lat\', \'lon\')'.format(soil_var_name, awc_dataset_name, soil_dataset.variables[soil_var_name].dimensions)
            logger.error(message)
            raise ValueError(message)

    #-----------------------------------------------------------------------------------------------------------------------
    def _initialize_unscaled_netcdfs(self,
                                     base_file_path,
                                     template_netcdf):

        netcdf_file_pet = base_file_path + '_pet.nc'
        netcdf_file_pdsi = base_file_path + '_pdsi.nc'
        netcdf_file_phdi = base_file_path + '_phdi.nc'
        netcdf_file_zindex = base_file_path + '_zindex.nc'
        netcdf_file_scpdsi = base_file_path + '_scpdsi.nc'
        netcdf_file_pmdi = base_file_path + '_pmdi.nc'

        # min/max numbers for the Palmer indices
        valid_min = -10.0
        valid_max = 10.0

        # initialize separate NetCDF files for each variable
        netcdf_utils.initialize_netcdf_single_variable_grid(netcdf_file_pet,
                                                            template_netcdf,
                                                            'pet',
                                                            'Potential Evapotranspiration (PET), from Thornthwaite\'s equation',
                                                            0.0,
                                                            2000.0,
                                                            'millimeter')
        netcdf_utils.initialize_netcdf_single_variable_grid(netcdf_file_pdsi,
                                                            template_netcdf,
                                                            'pdsi',
                                                            'Palmer Drought Severity Index (PDSI)',
                                                            valid_min,
                                                            valid_max)
        netcdf_utils.initialize_netcdf_single_variable_grid(netcdf_file_phdi,
                                                            template_netcdf,
                                                            'phdi',
                                                            'Palmer Hydrological Drought Index (PHDI)',
                                                            valid_min,
                                                            valid_max)
        netcdf_utils.initialize_netcdf_single_variable_grid(netcdf_file_zindex,
                                                            template_netcdf,
                                                            'zindex',
                                                            'Palmer Z-Index',
                                                            valid_min,
                                                            valid_max)
        netcdf_utils.initialize_netcdf_single_variable_grid(netcdf_file_scpdsi,
                                                            template_netcdf,
                                                            'scpdsi',
                                                            'Self-calibrated Palmer Drought Severity Index (scPDSI)',
                                                            valid_min,
                                                            valid_max)
        netcdf_utils.initialize_netcdf_single_variable_grid(netcdf_file_pmdi,
                                                            template_netcdf,
                                                            'pmdi',
                                                            'Palmer Modified Drought Index (PMDI)',
                                                            valid_min,
                                                            valid_max)

        # assign the NetCDF file paths to the corresponding member variables
        self.netcdf_pet = netcdf_file_pet
        self.netcdf_pdsi = netcdf_file_pdsi
        self.netcdf_phdi = netcdf_file_phdi
        self.netcdf_zindex = netcdf_file_zindex
        self.netcdf_scpdsi = netcdf_file_scpdsi
        self.netcdf_pmdi = netcdf_file_pmdi

    #-----------------------------------------------------------------------------------------------------------------------
    def _set_scaling_months(self,
                            months):
        """
        Reset the instance's month scale, the scale that'll be used to computed scaled indices (SPI, SPEI, PNP)
        
        :param months: the number of months, should correspond to one of the values of self.month_scales 
        """
        self.months = months

    #-----------------------------------------------------------------------------------------------------------------------
    def _initialize_scaled_netcdfs(self,
                                   months_scale):

        # dictionary of index types to the NetCDF dataset files corresponding to the base index names and
        # month scales (this is the object we'll build and return from this function)
        netcdfs = {}

        # dictionary of index types mapped to their corresponding long variable names to be used within their respective NetCDFs
        indicators_to_longnames = {'pnp': 'Percent of normal precipitation, {0}-month average',
                                   'spi_gamma': 'Standard Precipitation Index (Gamma distribution), {0}-month scale',
                                   'spi_pearson': 'Standard Precipitation Index (Pearson Type III distribution), {0}-month scale',
                                   'spei_gamma': 'Standard Precipitation Evapotranspiration Index (Gamma distribution), {0}-month scale',
                                   'spei_pearson': 'Standard Precipitation Evapotranspiration Index (Pearson Type III distribution), {0}-month scale'}

        # loop over the indices, creating an output NetCDF dataset for each
        for index, long_name in indicators_to_longnames.items():

            # use a separate valid min/max for PNP than for the other SP* indices
            if index == 'pnp':
                valid_min = -10.0
                valid_max = 10.0
            else:
                valid_min = -3.09
                valid_max = 3.09

            # create the variable name from the index and month scale
            variable_name = index + '_{0}'.format(str(months_scale).zfill(2))

            # create the NetCDF file path from the
            netcdf_file = self.output_file_base + '_' + variable_name + '.nc'

            # initialize the output NetCDF dataset
            netcdf_utils.initialize_netcdf_single_variable_grid(netcdf_file,
                                                                self.netcdf_precip,
                                                                variable_name,
                                                                long_name.format(months_scale),
                                                                valid_min,
                                                                valid_max)

            # add the months scale index's NetCDF to the dictionary for the current index
            netcdfs[index] = netcdf_file

        # assign the NetCDF file paths to the corresponding member variables
        self.netcdf_spi_gamma = netcdfs['spi_gamma']
        self.netcdf_spei_gamma = netcdfs['spei_gamma']
        self.netcdf_spi_pearson = netcdfs['spi_pearson']
        self.netcdf_spei_pearson = netcdfs['spei_pearson']
        self.netcdf_pnp = netcdfs['pnp']

        # set the number of months so we'll know at which months scale the indices NetCDF files should be computed
        self.months = months_scale

    #-----------------------------------------------------------------------------------------------------------------------
    def run(self):

        # the number of worker processes we'll have in our process pool
        number_of_workers = 1#multiprocessing.cpu_count()

        # open the input NetCDF files for compatibility validation and to get the data's time range
        with netCDF4.Dataset(self.netcdf_precip) as dataset_precip, \
             netCDF4.Dataset(self.netcdf_temperature) as dataset_temp, \
             netCDF4.Dataset(self.netcdf_soil) as dataset_awc:

            # make sure the datasets are compatible dimensionally
            self._validate_compatibility(dataset_precip,
                                         self.var_name_precip,
                                         dataset_temp,
                                         self.var_name_temperature,
                                         dataset_awc,
                                         self.var_name_soil)

            # get the initial year of the input dataset(s)
            time_units = dataset_precip.variables['time']
            self.data_start_year = netCDF4.num2date(time_units[0], time_units.units).year

            # get the number of latitudes in the input dataset(s)
            lat_size = dataset_precip.variables['lat'].size

        #--------------------------------------------------------------------------------------------------------------
        # Create PET and Palmer index NetCDF files, computed from input temperature, precipitation, and soil constant.
        #--------------------------------------------------------------------------------------------------------------

        # create a process Pool for worker processes to compute PET and Palmer indices, passing arguments to an initializing function
        pool = multiprocessing.Pool(processes=number_of_workers)

        # map the latitude indices as an arguments iterable to the compute function
        result = pool.map_async(self._process_latitude_palmer, range(lat_size))

        # get the exception(s) thrown, if any
        result.get()

        # close the pool and wait on all processes to finish
        pool.close()
        pool.join()

        #----------------------------------------------------------------------------------------------------------
        # Take the PET and Palmer index NetCDF files, compress and move to destination directory.
        #----------------------------------------------------------------------------------------------------------

#         input_output_netcdfs = []
#         for index in ['pdsi', 'phdi', 'scpdsi', 'zindex']:
#
#             # convert the Palmer files to compressed NetCDF4 and move to the destination directory
#             indicator_tuple = (unscaled_netcdfs[index], os.sep.join([destination_dir, index, unscaled_netcdfs[index]]))
#             input_output_netcdfs.append(indicator_tuple)
#
#         pool = multiprocessing.Pool(processes=number_of_workers)
#
#         # create an arguments iterable containing the input and output NetCDFs, map it to the convert function
#         result = pool.map_async(netcdf_utils.convert_and_move_netcdf, input_output_netcdfs)
#
#         # get the exception(s) thrown, if any
#         result.get()
#
#         # close the pool and wait on all processes to finish
#         pool.close()
#         pool.join()

        # compute the scaled indices (PNP, SPI, and SPEI)
        for months in self.scale_months:

            # initialize the output NetCDFs for this month scale
            self._initialize_scaled_netcdfs(months)

            # set the instance's scale size (number of months over which SPI, etc. will be scaled)
            self._set_scaling_months(months)

            # map the latitude indices as an arguments iterable to the compute function (reuse the same pool)
            result = pool.map_async(self._process_latitude_spi_spei_pnp, range(lat_size))

            # get the exception(s) thrown, if any
            result.get()

            # close the pool and wait on all processes to finish
            pool.close()
            pool.join()

#             # convert the SPI, SPEI, and PNP files to compressed NetCDF4 and move to the destination directory
#             input_output_netcdfs = [(scaled_netcdfs['spi_gamma'], '/nidis/test/nclimgrid/spi_gamma/' + scaled_netcdfs['spi_gamma']),
#                                     (scaled_netcdfs['spi_pearson'], '/nidis/test/nclimgrid/spi_pearson/' + scaled_netcdfs['spi_pearson']),
#                                     (scaled_netcdfs['spei_gamma'], '/nidis/test/nclimgrid/spei_gamma/' + scaled_netcdfs['spei_gamma']),
#                                     (scaled_netcdfs['spei_pearson'], '/nidis/test/nclimgrid/spei_pearson/' + scaled_netcdfs['spei_pearson']),
#                                     (scaled_netcdfs['pnp'], '/nidis/test/nclimgrid/pnp/' + scaled_netcdfs['pnp'])]
#
#             pool = multiprocessing.Pool(processes=number_of_workers)
#
#             # create an arguments iterable containing the input and output NetCDFs, map it to the convert function
#             result = pool.map_async(netcdf_utils.convert_and_move_netcdf, input_output_netcdfs)
#
#             # get the exception(s) thrown, if any
#             result.get()
#
#             # close the pool and wait on all processes to finish
#             pool.close()
#             pool.join()
#
#         # convert the PET file to compressed NetCDF4 and move into the destination directory
#         netcdf_utils.convert_and_move_netcdf((unscaled_netcdfs['pet'], '/nidis/test/nclimgrid/pet/' + unscaled_netcdfs['pet']))
#

    #-------------------------------------------------------------------------------------------------------------------
    def _process_latitude_palmer(self, lat_index):
        """
        Perform computation of Palmer indices on a latitude slice, i.e. all lat/lon locations for a single latitude.
        Each lat/lon will have its corresponding time series used as input, with a corresponding time series output
        for each index computed.

        :param lat_index: index of the latitude in the NetCDF, valid range is [0..(total # of divisions - 1)]
        """

        logger.info('Computing PET and Palmers for latitude index %s', lat_index)

        # open the input NetCDFs
        with netCDF4.Dataset(self.netcdf_precip) as precip_dataset, \
             netCDF4.Dataset(self.netcdf_temperature) as temp_dataset, \
             netCDF4.Dataset(self.netcdf_soil) as awc_dataset:

            # read the latitude slice of input temperature values
            temperature_lat_slice = temp_dataset[self.var_name_temperature][:, lat_index, :]    # assuming (time, lat, lon) orientation

            # get the actual latitude value (assumed to be in degrees north) for the latitude slice specified by the index
            latitude_degrees_north = temp_dataset['lat'][lat_index]

            # use the numpyapply_along_axis() function for computing indices such as PET that take a single time series
            # array as input (i.e. each longitude's time series is the initial 1-D array argument to the function we'll apply)


            # compute PET across all longitudes of the latitude slice
            pet_lat_slice = np.apply_along_axis(indices.pet,
                                                0,
                                                temperature_lat_slice,
                                                latitude_degrees=latitude_degrees_north,
                                                data_start_year=self.data_start_year)

            # open the existing PET NetCDF file for writing, copy the latitude slice into the PET variable at the indexed latitude position
            pet_lock.acquire()
            pet_dataset = netCDF4.Dataset(self.netcdf_pet, mode='a')
            pet_dataset['pet'][:, lat_index, :] = pet_lat_slice
            pet_dataset.sync()
            pet_dataset.close()
            pet_lock.release()

            # determine the dimensionality of the AWC dataset, in case there is a missing time
            # dimension and/or a switched lat/lon, then get the AWC latitude slice accordingly
            awc_dims = awc_dataset[self.var_name_soil].dimensions
            if awc_dims == ('time', 'lat', 'lon'):
                awc_lat_slice = awc_dataset[self.var_name_soil][:, lat_index, :].flatten() # assuming (time, lat, lon) orientation
            elif awc_dims == ('lat', 'lon'):
                awc_lat_slice = awc_dataset[self.var_name_soil][lat_index, :].flatten()    # assuming (lat, lon) orientation
            elif awc_dims == ('time', 'lon', 'lat'):
                awc_lat_slice = awc_dataset[self.var_name_soil][:, :, lat_index].flatten() # assuming (time, lon, lat) orientation
            elif awc_dims == ('lon', 'lat'):
                awc_lat_slice = awc_dataset[self.var_name_soil][:, lat_index].flatten()    # assuming (lon, lat) orientation
            else:
                message = 'Unable to read the soil constant (AWC) values due to unsupported AWC variable dimensions: {0}'.format(awc_dims)
                logger.error(message)
                raise ValueError(message)

            # read the latitude slice of input precipitation and available water capacity values
            precip_lat_slice = precip_dataset[self.var_name_precip][:, lat_index, :]    # assuming (time, lat, lon) orientation
            awc_fill_value = awc_dataset[self.var_name_soil]._FillValue

            # allocate arrays to contain a latitude slice of Palmer values
            lon_size = temp_dataset['lon'].size
            time_size = temp_dataset['time'].size
            lat_slice_shape = (time_size, 1, lon_size)
            pdsi_lat_slice = np.full(lat_slice_shape, np.NaN)
            phdi_lat_slice = np.full(lat_slice_shape, np.NaN)
            zindex_lat_slice = np.full(lat_slice_shape, np.NaN)
            scpdsi_lat_slice = np.full(lat_slice_shape, np.NaN)
            pmdi_lat_slice = np.full(lat_slice_shape, np.NaN)

            # compute Palmer indices for each longitude from the latitude slice where we have valid inputs
            for lon_index in range(lon_size):

                logger.info('Computing Palmers for longitude index %s', lon_index)

                # get the time series values for this longitude
                precip_time_series = precip_lat_slice[:, lon_index]
                pet_time_series = pet_lat_slice[:, lon_index]
                awc = awc_lat_slice[lon_index]

                # compute Palmer indices only if we have valid inputs
                if self._is_data_valid(precip_time_series) and \
                   self._is_data_valid(pet_time_series) and \
                   awc is not np.ma.masked and \
                   not math.isnan(awc) and \
                   not math.isclose(awc, awc_fill_value):

                    # put precipitation into inches, if not already
                    mm_to_inches_multiplier = 0.0393701
                    possible_mm_units = ['millimeters', 'millimeter', 'mm']
                    if precip_dataset[self.var_name_precip].units in possible_mm_units:
                        precip_time_series = precip_time_series * mm_to_inches_multiplier

                    # PET is in mm, convert to inches
                    pet_time_series = pet_time_series * mm_to_inches_multiplier

                    # compute Palmer indices
                    palmer_values = indices.scpdsi(precip_time_series,
                                                   pet_time_series,
                                                   awc,
                                                   self.data_start_year,
                                                   self.calibration_start_year,
                                                   self.calibration_end_year)

                    # add the values into the slice, first clipping all values to the valid range
                    scpdsi_lat_slice[:, 0, lon_index] = np.clip(palmer_values[0], _VALID_MIN, _VALID_MAX)
                    pdsi_lat_slice[:, 0, lon_index] = np.clip(palmer_values[1], _VALID_MIN, _VALID_MAX)
                    phdi_lat_slice[:, 0, lon_index] = np.clip(palmer_values[2], _VALID_MIN, _VALID_MAX)
                    pmdi_lat_slice[:, 0, lon_index] = np.clip(palmer_values[3], _VALID_MIN, _VALID_MAX)
                    zindex_lat_slice[:, 0, lon_index] = palmer_values[4]

            # open the existing PDSI NetCDF file for writing, copy the latitude slice into the PET variable at the indexed latitude position
            pdsi_lock.acquire()
            pdsi_dataset = netCDF4.Dataset(self.netcdf_pdsi, mode='a')
            pdsi_dataset['pdsi'][:, lat_index, :] = pdsi_lat_slice
            pdsi_dataset.sync()
            pdsi_dataset.close()
            pdsi_lock.release()

            # open the existing PHDI NetCDF file for writing, copy the latitude slice into the PET variable at the indexed latitude position
            phdi_lock.acquire()
            phdi_dataset = netCDF4.Dataset(self.netcdf_phdi, mode='a')
            phdi_dataset['phdi'][:, lat_index, :] = phdi_lat_slice
            phdi_dataset.sync()
            phdi_dataset.close()
            phdi_lock.release()

            # open the existing Z-Index NetCDF file for writing, copy the latitude slice into the Z-Index variable at the indexed latitude position
            zindex_lock.acquire()
            zindex_dataset = netCDF4.Dataset(self.netcdf_zindex, mode='a')
            zindex_dataset['zindex'][:, lat_index, :] = zindex_lat_slice
            zindex_dataset.sync()
            zindex_dataset.close()
            zindex_lock.release()

            # open the existing SCPDSI NetCDF file for writing, copy the latitude slice into the scPDSI variable at the indexed latitude position
            scpdsi_lock.acquire()
            scpdsi_dataset = netCDF4.Dataset(self.netcdf_scpdsi, mode='a')
            scpdsi_dataset['scpdsi'][:, lat_index, :] = scpdsi_lat_slice
            scpdsi_dataset.sync()
            scpdsi_dataset.close()
            scpdsi_lock.release()

            # open the existing PMDI NetCDF file for writing, copy the latitude slice into the PMDI variable at the indexed latitude position
            pmdi_lock.acquire()
            pmdi_dataset = netCDF4.Dataset(self.netcdf_pmdi, mode='a')
            pmdi_dataset['pmdi'][:, lat_index, :] = pmdi_lat_slice
            pmdi_dataset.sync()
            pmdi_dataset.close()
            pmdi_lock.release()

    #-------------------------------------------------------------------------------------------------------------------
    def _process_latitude_spi_spei_pnp(self, lat_index):
        '''
        Processes scaled indices (SPI, SPEI, and PNP) for a single latitude slice.

        :param lat_index:
        '''

        logger.info('Computing SPI, SPEI, and PNP for latitude index %s', lat_index)

        # open the input NetCDFs
        with netCDF4.Dataset(self.netcdf_precip) as precip_dataset, \
             netCDF4.Dataset(self.netcdf_pet) as pet_dataset:

            # read the latitude slice of input precipitation and PET values
            precip_lat_slice = precip_dataset[self.var_name_precip][:, lat_index, :]   # assuming (time, lat, lon) orientation
            pet_lat_slice = pet_dataset['pet'][:, lat_index, :]   # assuming (time, lat, lon) orientation

            # allocate arrays to contain a latitude slice of Palmer values
            lon_size = precip_dataset['lon'].size
            time_size = precip_dataset['time'].size
            lat_slice_shape = (time_size, 1, lon_size)

            # compute SPI/Gamma across all longitudes of the latitude slice
            spi_gamma_lat_slice = np.apply_along_axis(indices.spi_gamma,
                                                      0,
                                                      precip_lat_slice,
                                                      self.months)

            # compute SPI/Pearson across all longitudes of the latitude slice
            spi_pearson_lat_slice = np.apply_along_axis(indices.spi_pearson,
                                                        0,
                                                        precip_lat_slice,
                                                        self.months,
                                                        self.data_start_year,
                                                        self.calibration_start_year,
                                                        self.calibration_end_year)

            # compute PNP across all longitudes of the latitude slice
            pnp_lat_slice = np.apply_along_axis(indices.percentage_of_normal,
                                                0,
                                                precip_lat_slice,
                                                self.months,
                                                self.data_start_year,
                                                self.calibration_start_year,
                                                self.calibration_end_year)

            # allocate latitude slices for SPEI output
            spei_gamma_lat_slice = np.full(lat_slice_shape, np.NaN)
            spei_pearson_lat_slice = np.full(lat_slice_shape, np.NaN)

            # compute SPEI for each longitude from the latitude slice where we have valid inputs
            for lon_index in range(lon_size):

                # get the time series values for this longitude
                precip_time_series = precip_lat_slice[:, lon_index]
                pet_time_series = pet_lat_slice[:, lon_index]

                # compute SPEI for the current longitude only if we have valid inputs
                if (not precip_time_series.mask.all()) and \
                   (not pet_time_series.mask.all()):

                    # compute SPEI/Gamma
                    spei_gamma_lat_slice[:, 0, lon_index] = indices.spei_gamma(self.months,
                                                                               precip_time_series,
                                                                               pet_mm=pet_time_series)

                    # compute SPEI/Pearson
                    spei_pearson_lat_slice[:, 0, lon_index] = indices.spei_pearson(self.months,
                                                                                   self.data_start_year,
                                                                                   precip_time_series,
                                                                                   pet_mm=pet_time_series,
                                                                                   calibration_year_initial=self.calibration_start_year,
                                                                                   calibration_year_final=self.calibration_end_year)

            # use the same variable name within both Gamma and Pearson NetCDFs
            #TODO update this for separate 'spi_gamma_<months>' and 'spi_pearson_<months>' instead
            spi_gamma_variable_name = 'spi_gamma_' + str(self.months).zfill(2)
            spi_pearson_variable_name = 'spi_pearson_' + str(self.months).zfill(2)
            spei_gamma_variable_name = 'spei_gamma_' + str(self.months).zfill(2)
            spei_pearson_variable_name = 'spei_pearson_' + str(self.months).zfill(2)
            pnp_variable_name = 'pnp_' + str(self.months).zfill(2)

            # open the existing SPI/Gamma NetCDF file for writing, copy the latitude slice into the SPI variable at the indexed latitude position
            spi_gamma_lock.acquire()
            spi_gamma_dataset = netCDF4.Dataset(self.netcdf_spi_gamma, mode='a')
            spi_gamma_dataset[spi_gamma_variable_name][:, lat_index, :] = spi_gamma_lat_slice
            spi_gamma_dataset.sync()
            spi_gamma_dataset.close()
            spi_gamma_lock.release()

            # open the existing SPI/Pearson NetCDF file for writing, copy the latitude slice into the SPI variable at the indexed latitude position
            spi_pearson_lock.acquire()
            spi_pearson_dataset = netCDF4.Dataset(self.netcdf_spi_pearson, mode='a')
            spi_pearson_dataset[spi_pearson_variable_name][:, lat_index, :] = spi_pearson_lat_slice
            spi_pearson_dataset.sync()
            spi_pearson_dataset.close()
            spi_pearson_lock.release()

            # open the existing SPEI/Gamma NetCDF file for writing, copy the latitude slice into the SPEI variable at the indexed latitude position
            spei_gamma_lock.acquire()
            spei_gamma_dataset = netCDF4.Dataset(self.netcdf_spei_gamma, mode='a')
            spei_gamma_dataset[spei_gamma_variable_name][:, lat_index, :] = spei_gamma_lat_slice
            spei_gamma_dataset.sync()
            spei_gamma_dataset.close()
            spei_gamma_lock.release()

            # open the existing SPEI/Pearson NetCDF file for writing, copy the latitude slice into the SPEI variable at the indexed latitude position
            spei_pearson_lock.acquire()
            spei_pearson_dataset = netCDF4.Dataset(self.netcdf_spei_pearson, mode='a')
            spei_pearson_dataset[spei_pearson_variable_name][:, lat_index, :] = spei_pearson_lat_slice
            spei_pearson_dataset.sync()
            spei_pearson_dataset.close()
            spei_pearson_lock.release()

            # open the existing PNP NetCDF file for writing, copy the latitude slice into the PNP variable at the indexed latitude position
            pnp_lock.acquire()
            pnp_dataset = netCDF4.Dataset(self.netcdf_pnp, mode='a')
            pnp_dataset[pnp_variable_name][:, lat_index, :] = pnp_lat_slice
            pnp_dataset.sync()
            pnp_dataset.close()
            pnp_lock.release()

    #-------------------------------------------------------------------------------------------------------------------
    def _is_data_valid(self, data):
        """
        Returns whether or not an array is valid, i.e. a supported array type (ndarray or MaskArray) which is not all-NaN.

        :param data: data object, expected as either numpy.ndarry or numpy.ma.MaskArray
        :return True if array is non-NaN for at least one element and is an array type valid for processing by other modules
        :rtype: boolean
        """

        # make sure we're not dealing with all NaN values
        if np.ma.isMaskedArray(data):

            valid_flag = bool(data.count())

        elif isinstance(data, np.ndarray):

            valid_flag = not np.all(np.isnan(data))

        else:
            logger.warning('Invalid data type passed for precipitation data')
            valid_flag = False

        return valid_flag

#-----------------------------------------------------------------------------------------------------------------------
def process_grid(output_file_base,     # pragma: no cover
                 precip_file,
                 temp_file,
                 awc_file,
                 precip_var_name,
                 temp_var_name,
                 awc_var_name,
                 month_scales,
                 calibration_start_year,
                 calibration_end_year):
    """
    Performs indices processing from gridded NetCDF inputs.
    
    :param output_file_base:
    :param precip_file: 
    :param temp_file: 
    :param awc_file: 
    :param precip_var_name:
    :param temp_var_name:
    :param awc_var_name:
    :param month_scales:
    :param calibration_start_year:
    :param calibration_end_year:
    """

    # perform the processing
    grid_processor = GridProcessor(output_file_base,
                                   precip_file,
                                   temp_file,
                                   awc_file,
                                   precip_var_name,
                                   temp_var_name,
                                   awc_var_name,
                                   month_scales,
                                   calibration_start_year,
                                   calibration_end_year)
    grid_processor.run()

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    """
    This module is used to perform climate indices processing on gridded datasets in NetCDF.
    """

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        logger.info("Start time:    %s", start_datetime)

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--precip_file",
                            help="Precipitation dataset file (NetCDF) to be used as input for SPI, SPEI, and PNP computations",
                            required=True)
        parser.add_argument("--precip_var_name",
                            help="Precipitation variable name used in the precipitation NetCDF file",
                            required=True)
        parser.add_argument("--temp_file",
                            help="Temperature dataset file (NetCDF) to be used as input for PET and SPEI computations",
                            required=True)
        parser.add_argument("--temp_var_name",
                            help="Temperature variable name used in the temperature NetCDF file",
                            required=True)
        parser.add_argument("--awc_file",
                            help="Temperature dataset file (NetCDF) to be used as input for the PDSI computation",
                            required=False)
        parser.add_argument("--awc_var_name",
                            help="Available water capacity variable name used in the available water capacity NetCDF file",
                            required=False)
        parser.add_argument("--output_file_base",
                            help="Base output file path and name for the resulting output files",
                            required=True)
        parser.add_argument("--month_scales",
                            help="Month scales over which the PNP, SPI, and SPEI values are to be computed",
                            type=int,
                            nargs = '*',
                            choices=range(1, 73),
                            required=True)
        parser.add_argument("--calibration_start_year",
                            help="Initial year of calibration period",
                            type=int,
                            choices=range(1870, start_datetime.year + 1),
                            required=True)
        parser.add_argument("--calibration_end_year",
                            help="Final year of calibration period",
                            type=int,
                            choices=range(1870, start_datetime.year + 1),
                            required=True)
        args = parser.parse_args()

        # perform the processing
        process_grid(args.output_file_base,
                     args.precip_file,
                     args.temp_file,
                     args.awc_file,
                     args.precip_var_name,
                     args.temp_var_name,
                     args.awc_var_name,
                     args.month_scales,
                     args.calibration_start_year,
                     args.calibration_end_year)

        # report on the elapsed time
        end_datetime = datetime.now()
        logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        logger.exception('Failed to complete', exc_info=True)
        raise
