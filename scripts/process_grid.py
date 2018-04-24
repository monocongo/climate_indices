import argparse
from datetime import datetime
import logging
import multiprocessing
import netCDF4
import netcdf_utils
import numpy as np

from indices_python import indices, utils

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
spi_gamma_lock = multiprocessing.Lock()
spi_pearson_lock = multiprocessing.Lock()

# ignore warnings
import warnings
warnings.simplefilter('ignore', Warning)

#-----------------------------------------------------------------------------------------------------------------------
class GridProcessor(object):             # pragma: no cover

    def __init__(self,
                 output_file_base,
                 netcdf_precip,
                 netcdf_temperature,
                 netcdf_pet,
                 netcdf_soil,
                 var_name_precip,
                 var_name_temperature,
                 var_name_pet,
                 var_name_soil,
                 scale,
                 data_start_year,
                 data_end_year,
                 calibration_start_year,
                 calibration_end_year,
                 index_bundle,
                 time_series_type):

        self.output_file_base = output_file_base
        self.netcdf_precip = netcdf_precip
        self.netcdf_pet = netcdf_pet
        self.netcdf_temperature = netcdf_temperature
        self.netcdf_soil = netcdf_soil
        self.var_name_precip = var_name_precip
        self.var_name_pet = var_name_pet
        self.var_name_temperature = var_name_temperature
        self.var_name_soil = var_name_soil
        self.scale = scale
        self.calibration_start_year = calibration_start_year
        self.calibration_end_year = calibration_end_year        
        self.index_bundle = index_bundle
        self.time_series_type = time_series_type
        
        #FIXME
        #TODO get the initial year from the precipitation NetCDF (SPI, PNP) or temperature NetCDF (PET) 
        # or both (SPEI, Palmers), and if both validate that the two match
        # for example:
        # self.data_start_year, self.data_end_year = netcdf_utils.get_data_years(self.netcdf_precip, 
        #                                                                        self.netcdf_temperature,
        #                                                                        self.netcdf_pet) 
        self.data_start_year = data_start_year
        self.data_end_year = data_end_year
        
        if self.index_bundle != 'palmers':

            # the number of time steps used for scaled indices (for example the will be set to 30 for 30-day SPI, SPEI, and/or PNP)
            # this will need to be set each time this grid processor object is used for computing the scaled indices
            self.scale = scale

            # place holders for the scaled NetCDFs, these files will be created 
            # and assigned to these variables at each scale's computational iteration
            # (not all of these will be used unless index bundle == 'scaled' which implies all scaled indices
            self.netcdf_spi_gamma = ''
            self.netcdf_spi_pearson = ''
            self.netcdf_spei_gamma = ''
            self.netcdf_spei_pearson = ''
            self.netcdf_pnp = ''
    
            self._initialize_scaled_netcdfs(self)
        
        elif self.index_bundle == 'palmers':
        
            # place holders for the scaled NetCDFs, these files will be created 
            # and assigned to these variables at each scale's computational iteration
            self.netcdf_pdsi = ''
            self.netcdf_scpdsi = ''
            self.netcdf_phdi = ''
            self.netcdf_pmdi = ''
            self.netcdf_zindex = ''
            
            self._initialize_palmer_netcdfs(self)

        else:
            
            raise ValueError('Unsupported index_bundle argument: %s' % index_bundle)
        
        # set the data start and end years and get the list of latitude range
        self.data_start_year, self.data_end_year, lat_size = netcdf_utils.years_and_sizes(self.netcdf_precip)
                
    #-----------------------------------------------------------------------------------------------------------------------
    def _reset_scale(self,
                     scale):
        """
        Reset the instance's scale, the scale that'll be used to computed scaled indices (SPI, SPEI, and/or PNP).
        
        :param scale: the number of time steps to use as the scale factor when computing scaled indices 
        """
        self.scale = scale
        self._initialize_scaled_netcdfs(self)

    #-----------------------------------------------------------------------------------------------------------------------
    def _initialize_scaled_netcdfs(self):
#                                    scale,
#                                    time_series_type):

        # dictionary of index types to the NetCDF dataset files corresponding to the base index names and
        # day scales (this is the object we'll build and return from this function)
        netcdfs = {}

        # make a scale type substring to use within variable long_name attributes
        scale_type = str(self.scale) + '-month scale'
        if self.time_series_type == 'daily':
            scale_type = str(self.scale) + '-day scale'
        elif self.time_series_type != 'monthly':
            raise ValueError('Unsupported time series type argument: %s' % self.time_series_type)
        
        # dictionary of index types (ex. 'spi_gamma', 'spei_pearson', etc.) mapped to their corresponding long 
        # variable names, to be used within the respective NetCDFs as variable long_name attributes
        names_to_longnames = {}            
        if self.index_bundle == 'spi':
            names_to_longnames['spi_gamma'] = 'Standardized Precipitation Index (Gamma distribution), ' + scale_type
            names_to_longnames['spi_pearson'] = 'Standardized Precipitation Index (Pearson Type III distribution), ' + scale_type
        elif self.index_bundle == 'spei':
            names_to_longnames['spei_gamma'] = 'Standardized Precipitation Evapotranspiration Index (Gamma distribution), ' + scale_type
            names_to_longnames['spei_pearson'] = 'Standardized Precipitation Evapotranspiration Index (Pearson Type III distribution), ' + scale_type
        elif self.index_bundle == 'pnp':
            names_to_longnames['pnp'] = 'Percentage of Normal Precipitation, ' + scale_type
        if self.index_bundle == 'scaled':
            names_to_longnames['spi_gamma'] = 'Standardized Precipitation Index (Gamma distribution), ' + scale_type
            names_to_longnames['spi_pearson'] = 'Standardized Precipitation Index (Pearson Type III distribution), ' + scale_type
            names_to_longnames['spei_gamma'] = 'Standardized Precipitation Evapotranspiration Index (Gamma distribution), ' + scale_type
            names_to_longnames['spei_pearson'] = 'Standardized Precipitation Evapotranspiration Index (Pearson Type III distribution), ' + scale_type
            names_to_longnames['pnp'] = 'Percentage of Normal Precipitation, ' + scale_type
        else:
            raise ValueError('Unsupported index bundle: %s', self.index_bundle)

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
            variable_name = index_name + '_{0}'.format(str(scale).zfill(2))

            # create the NetCDF file path from the
            netcdf_file = self.output_file_base + '_' + variable_name + '.nc'

            # initialize the output NetCDF
            netcdf_utils.initialize_netcdf_single_variable_grid(netcdf_file,
                                                                self.netcdf_precip,
                                                                variable_name,
                                                                long_name.format(scale),
                                                                valid_min,
                                                                valid_max)

            # add the days scale index's NetCDF to the dictionary for the current index
            netcdfs[index_name] = netcdf_file

        # assign the NetCDF file paths to the corresponding member variables
        if self.index_bundle == 'spi':
            self.netcdf_spi_gamma = netcdfs['spi_gamma']
            self.netcdf_spi_pearson = netcdfs['spi_pearson']
        #FIXME finish for other scaled indices

        # set the number of days used to scale the indices
        self.scale = scale

    #-----------------------------------------------------------------------------------------------------------------------
    def run(self):

        # the number of worker processes we'll have in our process pool
        number_of_workers = multiprocessing.cpu_count()   # use 1 here for debugging
    
        # create a process Pool for worker processes to compute PET and Palmer indices, passing arguments to an initializing function
        pool = multiprocessing.Pool(processes=number_of_workers)

        # all index combinations/bundles except SPI and PNP will require PET, so compute it here if temperature provided
        if self.index_bundle in ['pet', 'spei', 'scaled', 'palmers']:
        
        if self.index_bundle in ['spi', 'spei', 'pnp', 'scaled']:
            
            compute_function = self._process_latitude_scaled
            
        elif self.index_bundle == 'palmers':

            compute_function = self._process_latitude_palmers
            
        else:
                        
            raise ValueError('Unsupported index_bundle argument: %s' % self.index_bundle)

        # map the latitude indices as an arguments iterable to the compute function
        result = pool.map_async(compute_function, range(lat_size))

        # get the exception(s) thrown, if any
        result.get()

        # close the pool and wait on all processes to finish
        pool.close()
        pool.join()

    #-------------------------------------------------------------------------------------------------------------------
    def _process_latitude_scaled(self, lat_index):
        '''
        Processes the relevant scaled indices for a single latitude slice at a single scale.

        :param lat_index: the latitude index of the latitude slice that will be read from NetCDF, computed, and written
        '''

        if self.time_series_type == 'daily':
            scale_increment = 'day'
        elif self.time_series_type == 'monthly':
            scale_increment = 'month'
        logger.info('Computing %s-%s %s for latitude index %s', self.scale, scale_increment, lat_index)

        # open the input NetCDFs
        with netCDF4.Dataset(self.netcdf_precip) as precip_dataset:

            # read the latitude slice of input precipitation and PET values
            precip_lat_slice = precip_dataset[self.var_name_precip][lat_index, :, :]   # assuming (lat, lon, time) orientation

            if self.time_series_type == 'daily':

                # times are daily, transform to all leap year times (i.e. 366 days per year), so we fill Feb 29th of each non-leap missing
                total_years = self.data_end_year - self.data_start_year + 1   # FIXME move this out of here, only needs to be computed once
    
                # allocate an array to hold transformed time series where all years contain 366 days
                original_days_count = precip_lat_slice.shape[1]
                total_lons = precip_lat_slice.shape[0]
                precip_lat_slice_all_leap = np.full((total_lons, total_years * 366), np.NaN)
                
                # at each longitude we have a time series of values, loop over these longitudes and transform each
                # corresponding time series to 366 day years representation (fill Feb 29 during non-leap years)
                for lon_index in range(total_lons):  # TODO work out how to apply this across the lon axis, to eliminate this loop
                    
                    # transform the data so it represents all years containing 366 days, with Feb 29 containing fill value during non-leap years
                    precip_lat_slice_all_leap[lon_index, :] = utils.transform_to_366day(precip_lat_slice[lon_index, :],
                                                                                       self.data_start_year,
                                                                                       total_years)

                precip_lat_slice = precip_lat_slice_all_leap
                
                        
            # compute SPI/Gamma across all longitudes of the latitude slice
            spi_gamma_lat_slice = np.apply_along_axis(indices.spi_gamma,
                                                      1,
                                                      precip_lat_slice,
                                                      self.scale,
                                                      time_series_type=self.time_series_type)

            # compute SPI/Pearson across all longitudes of the latitude slice
            spi_pearson_lat_slice = np.apply_along_axis(indices.spi_pearson,
                                                        1,
                                                        precip_lat_slice,
                                                        self.scale,
                                                        self.data_start_year,
                                                        self.calibration_start_year,
                                                        self.calibration_end_year)

            if self.time_series_type == 'daily':

                # at each longitude we have a time series of values with a 366 day per year representation (Feb 29 during non-leap years
                # is a fill value), loop over these longitudes and transform each corresponding time series back to a normal Gregorian calendar
                lat_slice_spi_gamma = np.full((total_lons, original_days_count), np.NaN)
                lat_slice_spi_pearson = np.full((original_days_count, total_lons), np.NaN)
                for lon_index in range(precip_lat_slice.shape[0]):
                    
                    # transform the data so it represents mixed leap and non-leap years, i.e. normal Gregorian calendar
                    lat_slice_spi_gamma[lon_index, :] = utils.transform_to_gregorian(spi_gamma_lat_slice[lon_index, :],
                                                                                     self.data_start_year,
                                                                                     total_years)
                    lat_slice_spi_pearson[lon_index, :] = utils.transform_to_gregorian(spi_pearson_lat_slice[lon_index, :],
                                                                                       self.data_start_year,
                                                                                       total_years)

                # make the lat slices we'll write be these transformed arrays
                spi_gamma_lat_slice = lat_slice_spi_gamma
                spi_pearson_lat_slice = lat_slice_spi_pearson

            # use the same variable name within both Gamma and Pearson NetCDFs
            spi_gamma_variable_name = 'spi_gamma_' + str(self.scale).zfill(2)
            spi_pearson_variable_name = 'spi_pearson_' + str(self.scale).zfill(2)

            # open the existing SPI/Gamma NetCDF file for writing, copy the latitude slice into the SPI variable at the indexed latitude position
            spi_gamma_lock.acquire()
            spi_gamma_dataset = netCDF4.Dataset(self.netcdf_spi_gamma, mode='a')
            spi_gamma_dataset[spi_gamma_variable_name][lat_index, :, :] = spi_gamma_lat_slice   # (lat, lon, time)
            spi_gamma_dataset.sync()
            spi_gamma_dataset.close()
            spi_gamma_lock.release()

            # open the existing SPI/Pearson NetCDF file for writing, copy the latitude slice into the SPI variable at the indexed latitude position
            spi_pearson_lock.acquire()
            spi_pearson_dataset = netCDF4.Dataset(self.netcdf_spi_pearson, mode='a')
            spi_pearson_dataset[spi_pearson_variable_name][lat_index, :, :] = spi_pearson_lat_slice   # (lat, lon, time)
            spi_pearson_dataset.sync()
            spi_pearson_dataset.close()
            spi_pearson_lock.release()

#-----------------------------------------------------------------------------------------------------------------------
#TODO merge into netcdf_utils
def _initialize_netcdf_single_variable_grid(file_path,              # pragma: no cover
                                            template_netcdf,
                                            variable_name,
                                            variable_long_name,
                                            valid_min,
                                            valid_max,
                                            variable_units=None,
                                            fill_value=np.float32(np.NaN)):
    '''
    This function is used to initialize and return a netCDF4.Dataset object, containing a single data variable having 
    dimensions (lat, lon, time). The input data values array is assumed to be a 3-D array with indices corresponding to 
    the variable dimensions. The latitude, longitude, and time values are copied from the template NetCDF, which is 
    assumed to have dimension sizes matching to the axes of the variable values array. Global attributes are also copied 
    from the template NetCDF.
    
    :param file_path: the file path/name of the NetCDF Dataset object returned by this function
    :param template_dataset: an existing/open NetCDF Dataset object which will be used as a template for the Dataset
                             that will be created by this function
    :param variable_name: the variable name which will be used to identify the main data variable within the Dataset
    :param variable_long_name: the long name attribute of the main data variable within the Dataset
    :param variable_units: string specifying the units of the variable 
    :param valid_min: the minimum value to which the main data variable of the resulting Dataset(s) will be clipped
    :param valid_max: the maximum value to which the main data variable of the resulting Dataset(s) will be clipped
    :param fill_value: the fill value to use for main data variable of the resulting Dataset(s)
    :return: an open netCDF4.Dataset object
    '''

    with netCDF4.Dataset(template_netcdf, 'r') as template_dataset:
 
        # get the template's dimension sizes
        lat_size = template_dataset.variables['lat'].size
        lon_size = template_dataset.variables['lon'].size
        time_size = template_dataset.variables['time'].size
    
        # make a basic set of variable attributes
        variable_attributes = {'valid_min' : valid_min,
                               'valid_max' : valid_max,
                               'long_name' : variable_long_name}
        if variable_units is not None:
            variable_attributes['units'] = variable_units
            
        # open the dataset as a NetCDF in write mode
        dataset = netCDF4.Dataset(file_path, 'w')
        
        # copy the global attributes from the input
        # TODO/FIXME add/modify global attributes to correspond with the actual dataset
        dataset.setncatts(template_dataset.__dict__)
        
        # create the lat, lon, and time dimensions
        dataset.createDimension('lat', lat_size)
        dataset.createDimension('lon', lon_size)
        dataset.createDimension('time', time_size)
    
        # get the appropriate data types to use for the variables
        lat_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables['lat'])
        lon_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables['lon'])
        time_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables['time'])
        data_dtype = netcdf_utils.find_netcdf_datatype(fill_value)
    
        # create the variables
        lat_variable = dataset.createVariable('lat', lat_dtype, ('lat',))
        lon_variable = dataset.createVariable('lon', lon_dtype, ('lon',))
        time_variable = dataset.createVariable('time', time_dtype, ('time',))
        data_variable = dataset.createVariable(variable_name,
                                               data_dtype,
                                               ('lat', 'lon','time',),
                                               fill_value=fill_value, 
                                               zlib=False)
    
        # set the variables' attributes
        time_variable.setncatts(template_dataset.variables['time'].__dict__)
        lat_variable.setncatts(template_dataset.variables['lat'].__dict__)
        lon_variable.setncatts(template_dataset.variables['lon'].__dict__)
        data_variable.setncatts(variable_attributes)
    
        # set the coordinate variables' values
        lat_variable[:] = template_dataset.variables['lat'][:]
        lon_variable[:] = template_dataset.variables['lon'][:]
        time_variable[:] = template_dataset.variables['time'][:]

#-----------------------------------------------------------------------------------------------------------------------
def process_grid(output_file_base,     # pragma: no cover
                 precip_file,
                 precip_var_name,
                 day_scales,
                 calibration_start_year,
                 calibration_end_year,
                 index_bundle,
                 time_series_type):
    
    """
    Performs indices processing from gridded NetCDF inputs.
    
    :param output_file_base:
    :param precip_file: 
    :param precip_var_name:
    :param day_scales:
    :param calibration_start_year:
    :param calibration_end_year:
    :param index_bundle:
    :param time_series_type: 
    """

    # perform the processing
    grid_processor = GridProcessor(output_file_base,
                                   precip_file,
                                   precip_var_name,
                                   day_scales,
                                   calibration_start_year,
                                   calibration_end_year,
                                   index_bundle,
                                   time_series_type)
    
    grid_processor.run()

#-----------------------------------------------------------------------------------------------------------------------
def _process_spi(precip_file,
                 precip_var_name,
                 output_file_base,
                 scale,
                 calibration_start_year,
                 calibration_end_year,
                 time_series_type):
    """
    Processes SPI for a precipitation dataset for both gamma and Pearson fittings at a single scale. 
     
    :param precip_file: precipitation NetCDF, data variables expected to be in (time, lat, lon) order or similar
    :param precip_var_name: variable name for precipitation in the NetCDF
    :param output_file_base: base file path/name for final result SPI files
    :param scale: number of time steps over which the index scales, integer
    :param calibration_start_year: initial year for calibration period
    :param calibration_end_year: final year for calibration period
    :param time_series_type: process values as either monthly or daily values
    """
        
    # log what we're doing (and validate the time series type argument at the same time)
    if time_series_type == 'daily':
        scale_type = str(scale) + '-day'
    elif time_series_type == 'monthly':
        scale_type = str(scale) + '-month'
    else:
        raise ValueError('Unsupported time series type argument: %s', time_series_type)
    logger.info('Processing %s SPI for precipitation file:  %s\n', scale_type, precip_file)
    
    # verify that the dimensions are in the expected order
    with netCDF4.Dataset(precip_file) as dataset_precip:
        
        dimensions = dataset_precip.variables[precip_var_name].dimensions
        if dimensions != ('lat', 'lon', 'time'):
            message = "Dimensions of precipitation variable not in the expected lat/lon/time order: %s" % dimensions
            logger.error(message)
            raise ValueError(message)
        
    process_grid(output_file_base,
                 precip_file,
                 precip_var_name,
                 [scale],
                 calibration_start_year,
                 calibration_end_year,
                 'spi',
                 time_series_type)
        
    # log what we're doing
    logger.info('Output files: {0}\n              {1}'.format(output_file_base + '_spi_gamma_{0}.nc'.format(str(scale).zfill(2)),
                                                              output_file_base + '_spi_pearson_{0}.nc'.format(str(scale).zfill(2))))
    
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
        parser.add_argument("--output_file_base",
                            help="Base output file path and name for the resulting output files",
                            required=True)
        parser.add_argument("--scales",
                            help="Month scales over which the PNP, SPI, and SPEI values are to be computed",
                            type=int,
                            nargs = '*',
                            choices=range(1, 720),
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
        parser.add_argument("--index_bundle",
                            help="Indices to compute",
                            choices=['spi', 'spei', 'pnp', 'scaled', 'pet', 'palmers', 'full'],
                            default='spi',    #TODO use 'full' as the default once all indices are functional
                            required=False)
        parser.add_argument("--time_series_type",
                            help="Process input as either monthly or daily values",
                            choices=['monthly', 'daily'],
                            required=True)
        args = parser.parse_args()

        
        '''
        Example command line arguments for SPI only:
        
        --precip_file /tmp/jadams/cmorph_daily_prcp_199801_201707.nc --precip_var_name prcp --output_file_base ~/data/cmorph/spi/cmorph --day_scales 1 2 3 6 9 12 24 --calibration_start_year 1998 --calibration_end_year 2016 --index_bundle spi /tmp/jadams
        '''

        if args.index_bundle == 'spi':
            
            # loop over each days scale, performing processing for each in turn
            for scale in args.scales:
            
                # process SPI using full input file at once
                _process_spi(args.precip_file,
                             args.precip_var_name,
                             args.output_file_base,
                             scale,
                             args.calibration_start_year,
                             args.calibration_end_year,
                             args.time_series_type)
            
        # report on the elapsed time
        end_datetime = datetime.now()
        logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        logger.exception('Failed to complete', exc_info=True)
        raise
