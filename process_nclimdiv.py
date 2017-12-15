import argparse
from datetime import datetime
import indices
import logging
import multiprocessing
import netCDF4
import netcdf_utils
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
# static constants
_VALID_MIN = -10.0
_VALID_MAX = 10.0

#-----------------------------------------------------------------------------------------------------------------------
# multiprocessing lock we'll use to synchronize I/O writes to NetCDF files, one per each output file
lock = multiprocessing.Lock()

#-----------------------------------------------------------------------------------------------------------------------
class DivisionsProcessor(object):

    def __init__(self, 
                 input_file,
                 output_file,
                 var_name_precip,
                 var_name_temperature,
                 var_name_soil,
                 month_scales,
                 calibration_start_year,
                 calibration_end_year):
        
        '''
        Constructor method.
        
        :param input_file: 
        :param output_file: 
        :param var_name_precip: 
        :param var_name_temperature: 
        :param var_name_soil: 
        :param month_scales:
        :param calibration_start_year:
        :param calibration_end_year:   
        '''
    
        self.input_file = input_file
        self.output_file = output_file
        self.var_name_precip = var_name_precip
        self.var_name_temperature = var_name_temperature
        self.var_name_soil = var_name_soil
        self.scale_months = month_scales
        self.calibration_start_year = calibration_start_year
        self.calibration_end_year = calibration_end_year        

        # TODO get the initial year from the precipitation NetCDF, for now use hard-coded value specific to nClimGrid
        self.data_start_year = 1895
        
        # create and populate the NetCDF we'll use to contain our results of a call to run()
        self._initialize_netcdf()

    #-----------------------------------------------------------------------------------------------------------------------
    def _initialize_netcdf(self):
        """
        This function is used to initialize and return a netCDF4.Dataset object containing all variables 
        to be computed for a climate divisions climatology.
        """
     
        # use NaNs as our default fill/missing value
        fill_value=np.float32(np.NaN)
     
        # open the NetCDF datasets within a context manager
        with netCDF4.Dataset(self.input_file) as template_dataset, \
             netCDF4.Dataset(self.output_file, 'w') as new_dataset:
      
            # get the template's dimension sizes
            divisions_count = template_dataset.variables['division'].size
         
            # copy the global attributes from the input
            # TODO/FIXME add/modify global attributes to correspond with the actual dataset
            new_dataset.setncatts(template_dataset.__dict__)
             
            # use "ClimDiv-1.0" as the Conventions setting in order to facilitate visualization by the NOAA Weather and Climate Toolkit
            new_dataset.setncattr("Conventions", "ClimDiv-1.0")
             
            # create the time, x, and y dimensions
            new_dataset.createDimension('time', None)
            new_dataset.createDimension('division', divisions_count)
         
            # get the appropriate data types to use for the variables
            time_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables['time'])
            divisions_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables['division'])
            data_dtype = netcdf_utils.find_netcdf_datatype(fill_value)
         
            # create the coordinate variables
            time_units = new_dataset.createVariable('time', time_dtype, ('time',))
            division_variable = new_dataset.createVariable('division', divisions_dtype, ('division',))
     
            # set the coordinate variables' attributes and values
            time_units.setncatts(template_dataset.variables['time'].__dict__)
            time_units[:] = template_dataset.variables['time'][:]
            division_variable.setncatts(template_dataset.variables['division'].__dict__)
            division_variable[:] = template_dataset.variables['division'][:]
     
            # create a variable for each unscaled index
            unscaled_indices = ['pet', 'pdsi', 'phdi', 'pmdi', 'zindex', 'scpdsi']
            for index in unscaled_indices:
                if index == 'pet':
                     
                    variable_name = 'pet'
                    variable_attributes = {'standard_name': 'pet',
                                           'long_name': 'Potential Evapotranspiration (PET), from Thornthwaite\'s equation',
                                           'valid_min': 0.0,
                                           'valid_max': 2000.0,
                                           'units': 'millimeter'}
                 
                elif index == 'pdsi':
                     
                    variable_name = 'pdsi'
                    variable_attributes = {'standard_name': 'pdsi',
                                           'long_name': 'Palmer Drought Severity Index (PDSI)',
                                           'valid_min': -10.0,
                                           'valid_max': 10.0}
                 
                elif index == 'scpdsi':
                     
                    variable_name = 'scpdsi'
                    variable_attributes = {'standard_name': 'scpdsi',
                                           'long_name': 'Self-calibrated Palmer Drought Severity Index (PDSI)',
                                           'valid_min': -10.0,
                                           'valid_max': 10.0}
                 
                elif index == 'phdi':
                     
                    variable_name = 'phdi'
                    variable_attributes = {'standard_name': 'phdi',
                                           'long_name': 'Palmer Hydrological Drought Index (PHDI)',
                                           'valid_min': -10.0,
                                           'valid_max': 10.0}
                 
                elif index == 'pmdi':
                     
                    variable_name = 'pmdi'
                    variable_attributes = {'standard_name': 'pmdi',
                                           'long_name': 'Palmer Modified Drought Index (PMDI)',
                                           'valid_min': -10.0,
                                           'valid_max': 10.0}
                 
                elif index == 'zindex':
                     
                    variable_name = 'zindex'
                    variable_attributes = {'standard_name': 'zindex',
                                           'long_name': 'Palmer Z-Index',
                                           'valid_min': -10.0,
                                           'valid_max': 10.0}
     
                else:
     
                    message = '{0} is an unsupported index type'.format(index)
                    logger.error(message)
                    raise ValueError(message)
     
                # create variables with scale month
                data_variable = new_dataset.createVariable(variable_name,
                                                           data_dtype,
                                                           ('division', 'time',),
                                                           fill_value=fill_value, 
                                                           zlib=False)
                data_variable.setncatts(variable_attributes)
     
            # create a variable for each scaled index
            scaled_indices = ['pnp', 'spi_gamma', 'spi_pearson', 'spei_gamma', 'spei_pearson']
            for index in scaled_indices:
                for months in self.scale_months:
                     
                    variable_name = index + '_{}'.format(str(months).zfill(2))
                     
                    if index == 'spi_gamma':
                 
                        variable_attributes = {'standard_name': variable_name,
                                               'long_name': 'SPI (Gamma), {}-month scale'.format(months),
                                               'valid_min': -3.09,
                                               'valid_max': 3.09}
     
                    elif index == 'spi_pearson':
                 
                        variable_attributes = {'standard_name': variable_name,
                                               'long_name': 'SPI (Pearson), {}-month scale'.format(months),
                                               'valid_min': -3.09,
                                               'valid_max': 3.09}
     
                    elif index == 'spei_gamma':
                 
                        variable_attributes = {'standard_name': variable_name,
                                               'long_name': 'SPEI (Gamma), {}-month scale'.format(months),
                                               'valid_min': -3.09,
                                               'valid_max': 3.09}
     
                    elif index == 'spei_pearson':
                 
                        variable_attributes = {'standard_name': variable_name,
                                               'long_name': 'SPEI (Pearson), {}-month scale'.format(months),
                                               'valid_min': -3.09,
                                               'valid_max': 3.09}
     
                    elif index == 'pnp':
                 
                        variable_attributes = {'standard_name': variable_name,
                                               'long_name': 'Percent average precipitation, {}-month scale'.format(months),
                                               'valid_min': 0,
                                               'valid_max': 10.0,
                                               'units': 'percent of average'}
     
                    else:
     
                        message = '{0} is an unsupported index type'.format(index)
                        logger.error(message)
                        raise ValueError(message)
     
                    # create month scaled variable
                    data_variable = new_dataset.createVariable(variable_name,
                                                               data_dtype,
                                                               ('division', 'time',),
                                                               fill_value=fill_value, 
                                                               zlib=False)
                    data_variable.setncatts(variable_attributes)
     
    #-----------------------------------------------------------------------------------------------------------------------
    def _compute_and_write_division(self, division_index):
        """
        Computes indices for a single division, writing the output into NetCDF.
        
        :param division_index: 
        """
        
        # open the NetCDF files 
        with netCDF4.Dataset(self.input_file) as input_dataset:
            
            division_id = input_dataset['division'][division_index]
            
            # only process divisions within CONUS, 101 - 4809
            if division_id > 4899:
                return
            
            logger.info('Processing indices for division %s', division_id)
        
            # read the division of input temperature values 
            temperature = input_dataset[self.var_name_temperature][division_index, :]    # assuming (divisions, time) orientation
            
            # initialize the latitude outside of the valid range, in order to use this within a conditional below to verify a valid latitude
            latitude = -100.0  
    
            # latitudes are only available for certain divisions, make sure we have one for this division index
            if division_index < input_dataset['lat'][:].size:
                
                # get the actual latitude value (assumed to be in degrees north) for the latitude slice specified by the index
                latitude = input_dataset['lat'][division_index]
    
            # only proceed if the latitude value is within valid range            
            if not np.isnan(latitude) and (latitude < 90.0) and (latitude > -90.0):
                
                # convert temperatures from Fahrenheit to Celsius, if necessary
                temperature_units = input_dataset[self.var_name_temperature].units
                if temperature_units in ['degree_Fahrenheit', 'degrees Fahrenheit', 'degrees F', 'fahrenheit', 'Fahrenheit', 'F']:
                    
                    temperature = np.apply_along_axis(self._f2c, 0, temperature)
    
                elif temperature_units not in ['degree_Celsius', 'degrees Celsius', 'degrees C', 'celsius', 'Celsius', 'C']:
                    
                    raise ValueError('Unsupported temperature units: \'%s\'', temperature_units)
        
                # use the numpy.apply_along_axis() function for computing indices such as PET that take a single time series
                # array as input (i.e. each division's time series is the initial 1-D array argument to the function we'll apply)
                
                logger.info('\tComputing PET for division %s', division_id)
    
                # compute PET across all longitudes of the latitude slice
                pet_time_series = indices.pet(temperature, 
                                              latitude_degrees=latitude, 
                                              data_start_year=self.data_start_year)
            
                # the above returns PET in millimeters, note this for further consideration
                pet_units = 'millimeter'
                
                # write the PET values to NetCDF        
                lock.acquire()
                with netCDF4.Dataset(self.output_file, 'a') as output_dataset:
                    output_dataset['pet'][division_index, :] = np.reshape(pet_time_series, (1, pet_time_series.size))
                    output_dataset.sync()
                lock.release()
    
            else:
                
                pet_time_series = np.full(temperature.shape, np.NaN)
                pet_units = None
    
            # read the division's input precipitation and available water capacity values
            precip_time_series = input_dataset[self.var_name_precip][division_index, :]   # assuming (divisions, time) orientation
            
            if division_index < input_dataset[self.var_name_soil][:].size:
                awc = input_dataset[self.var_name_soil][division_index]               # assuming (divisions) orientation
                awc += 1   # AWC values need to include top inch, values from the soil file do not, so we add top inch here
            else:
                awc = np.NaN
                
            # allocate arrays to contain a latitude slice of Palmer values
            time_size = input_dataset['time'].size
            division_shape = (time_size)
            pdsi = np.full(division_shape, np.NaN)
            phdi = np.full(division_shape, np.NaN)
            pmdi = np.full(division_shape, np.NaN)
            scpdsi = np.full(division_shape, np.NaN)
            zindex = np.full(division_shape, np.NaN)
        
            # compute SPI and SPEI for the current division only if we have valid inputs
            if not np.isnan(precip_time_series).all():
                
                # put precipitation into inches if not already
                mm_to_inches_multiplier = 0.0393701
                possible_mm_units = ['millimeters', 'millimeter', 'mm']
                if input_dataset[self.var_name_precip].units in possible_mm_units:
                    precip_time_series = precip_time_series * mm_to_inches_multiplier
        
                if not np.isnan(pet_time_series).all():
                
                    # compute Palmer indices if we have valid inputs
                    if not np.isnan(awc):
                            
                        # if PET is in mm, convert to inches
                        if pet_units in possible_mm_units:
                            pet_time_series = pet_time_series * mm_to_inches_multiplier
        
                        # PET is in mm, convert to inches since the Palmer uses imperial units
                        pet_time_series = pet_time_series * mm_to_inches_multiplier
        
                        logger.info('\tComputing PDSI for division %s', division_id)
    
                        # compute Palmer indices
                        palmer_values = indices.scpdsi(precip_time_series,
                                                       pet_time_series,
                                                       awc,
                                                       self.data_start_year,
                                                       self.calibration_start_year,
                                                       self.calibration_end_year)
            
                        scpdsi = palmer_values[0]
                        pdsi = palmer_values[1]
                        phdi = palmer_values[2]
                        pmdi = palmer_values[3]
                        zindex = palmer_values[4]
        
                        # write the PDSI values to NetCDF
                        lock.acquire()
                        with netCDF4.Dataset(self.output_file, 'a') as output_dataset:
                            output_dataset['pdsi'][division_index, :] = np.reshape(pdsi, (1, pdsi.size))
                            output_dataset['phdi'][division_index, :] = np.reshape(phdi, (1, phdi.size))
                            output_dataset['pmdi'][division_index, :] = np.reshape(pmdi, (1, pmdi.size))
                            output_dataset['scpdsi'][division_index, :] = np.reshape(pdsi, (1, scpdsi.size))
                            output_dataset['zindex'][division_index, :] = np.reshape(zindex, (1, zindex.size))
                            output_dataset.sync()
                        lock.release()
        
                    # process the SPI and SPEI at the specified month scales
                    for months in self.scale_months:
                        
                        logger.info('\tComputing SPI/SPEI/PNP at %s-month scale for division %s', months, division_id)
    
                        #TODO ensure that the precipitation and PET values are using the same units
                        
                        # compute SPEI/Gamma
                        spei_gamma = indices.spei_gamma(months,
                                                        precip_time_series,
                                                        pet_mm=pet_time_series)
    
                        # compute SPEI/Pearson
                        spei_pearson = indices.spei_pearson(months,
                                                            self.data_start_year,
                                                            precip_time_series,
                                                            pet_mm=pet_time_series,
                                                            calibration_year_initial=self.calibration_start_year,
                                                            calibration_year_final=self.calibration_end_year)
                         
                        # compute SPI/Gamma
                        spi_gamma = indices.spi_gamma(precip_time_series, 
                                                      months)
                 
                        # compute SPI/Pearson
                        spi_pearson = indices.spi_pearson(precip_time_series, 
                                                          months,
                                                          self.data_start_year,
                                                          self.calibration_start_year, 
                                                          self.calibration_end_year)        
            
                        # compute PNP
                        pnp = indices.percentage_of_normal(precip_time_series, 
                                                           months,
                                                           self.data_start_year,
                                                           self.calibration_start_year, 
                                                           self.calibration_end_year)        
        
                        # create variable names which should correspond to the appropriate scaled index output variables
                        scaled_name_suffix = str(months).zfill(2)
                        spei_gamma_variable_name = 'spei_gamma_' + scaled_name_suffix
                        spei_pearson_variable_name = 'spei_pearson_' + scaled_name_suffix
                        spi_gamma_variable_name = 'spi_gamma_' + scaled_name_suffix
                        spi_pearson_variable_name = 'spi_pearson_' + scaled_name_suffix
                        pnp_variable_name = 'pnp_' + scaled_name_suffix
        
                        # write the SPI, SPEI, and PNP values to NetCDF        
                        lock.acquire()
                        with netCDF4.Dataset(self.output_file, 'a') as output_dataset:
                            output_dataset[spei_gamma_variable_name][division_index, :] =   np.reshape(spei_gamma, (1, spei_gamma.size))
                            output_dataset[spei_pearson_variable_name][division_index, :] = np.reshape(spei_pearson, (1, spei_pearson.size))
                            output_dataset[spi_gamma_variable_name][division_index, :] =    np.reshape(spi_gamma, (1, spi_gamma.size))
                            output_dataset[spi_pearson_variable_name][division_index, :] =  np.reshape(spi_pearson, (1, spi_pearson.size))
                            output_dataset[pnp_variable_name][division_index, :] =          np.reshape(pnp, (1, pnp.size))
                            output_dataset.sync()
                        lock.release()

    #-----------------------------------------------------------------------------------------------------------------------
    def _f2c(self, t):
        '''
        Converts a temperature value from Fahrenheit to Celsius
        '''
        return (t-32)*5.0/9

    #-------------------------------------------------------------------------------------------------------------------
    def run(self):
        
        # initialize the output NetCDF that will contain the computed indices
        with netCDF4.Dataset(self.input_file) as input_dataset:
            
            # get the initial and final year of the input datasets
            time_variable = input_dataset.variables['time']
            self.data_start_year = netCDF4.num2date(time_variable[0], time_variable.units).year
 
            # get the number of divisions in the input dataset(s)
            divisions_count = input_dataset.variables['division'].size
        
        #--------------------------------------------------------------------------------------------------------------
        # Create PET and Palmer index NetCDF files, computed from input temperature, precipitation, and soil constant.
        # Compute SPI, SPEI, and PNP at all specified month scales.
        #--------------------------------------------------------------------------------------------------------------

        # create a process Pool for worker processes to compute indices for each division
        pool = multiprocessing.Pool(processes=1)#multiprocessing.cpu_count())
          
        # map the divisions indices as an arguments iterable to the compute function
        result = pool.map_async(self._compute_and_write_division, range(divisions_count))
                  
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
        
#         # compute the scaled indices (PNP, SPI, and SPEI)
#         for months in self.scale_months:
#  
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
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    """
    This module is used to perform climate indices processing on nClimGrid datasets in NetCDF.
    """

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        logger.info("Start time:    %s", start_datetime)

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_file", 
                            help="Input dataset file (NetCDF) containing temperature, precipitation, and soil values for PDSI, SPI, SPEI, and PNP computations", 
                            required=True)
        parser.add_argument("--precip_var_name", 
                            help="Precipitation variable name used in the input NetCDF file", 
                            required=True)
        parser.add_argument("--temp_var_name", 
                            help="Temperature variable name used in the input NetCDF file", 
                            required=True)
        parser.add_argument("--awc_var_name", 
                            help="Available water capacity variable name used in the input NetCDF file", 
                            required=False)
        parser.add_argument("--output_file",
                            help=" Output file path and name",
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
        divisions_processor = DivisionsProcessor(args.input_file,
                                                 args.output_file,
                                                 args.precip_var_name,
                                                 args.temp_var_name,
                                                 args.awc_var_name,
                                                 args.month_scales,
                                                 args.calibration_start_year,
                                                 args.calibration_end_year)
        divisions_processor.run()
        
        # report on the elapsed time
        end_datetime = datetime.now()
        logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        logger.exception('Failed to complete', exc_info=True)
        raise
    