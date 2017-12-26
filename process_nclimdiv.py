import argparse
from datetime import datetime
import indices
from ingest import ingest_nclimdiv
import logging
import multiprocessing
import netCDF4
import netcdf_utils
import numba
import numpy as np
from process import process_nclimdiv
import random

#-----------------------------------------------------------------------------------------------------------------------
# set up matplotlib to use the Agg backend, in order to remove any dependencies on an X server
import matplotlib
import pdinew
import process
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
# ignore warnings
import warnings
warnings.simplefilter('ignore', Warning)

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
                 divisions_file,
                 var_name_precip,
                 var_name_temperature,
                 var_name_soil,
                 month_scales,
                 calibration_start_year,
                 calibration_end_year,
                 use_orig_pe=False):
        
        '''
        Constructor method.
        
        :param divisions_file: 
        :param var_name_precip: 
        :param var_name_temperature: 
        :param var_name_soil: 
        :param month_scales:
        :param calibration_start_year:
        :param calibration_end_year:   
        '''
    
        self.divisions_file = divisions_file
        self.var_name_precip = var_name_precip
        self.var_name_temperature = var_name_temperature
        self.var_name_soil = var_name_soil
        self.scale_months = month_scales
        self.calibration_start_year = calibration_start_year
        self.calibration_end_year = calibration_end_year        
        self.use_orig_pe = use_orig_pe
        
        # TODO get the initial year from the precipitation NetCDF, for now use hard-coded value specific to nClimDiv  pylint: disable=fixme
        self.data_start_year = 1895
        
#         # create and populate the NetCDF we'll use to contain our results of a call to run()
#         self._initialize_netcdf()

    #-----------------------------------------------------------------------------------------------------------------------
    def _initialize_netcdf(self):
        """
        This function is used to initialize and return a netCDF4.Dataset object containing all variables 
        to be computed for a climate divisions climatology.
        """
     
        # use NaNs as our default fill/missing value
        fill_value=np.float32(np.NaN)
     
        # open the NetCDF datasets within a context manager
        with netCDF4.Dataset(self.divisions_file, 'a') as new_dataset:
      
            data_dtype = netcdf_utils.find_netcdf_datatype(fill_value)
         
            # create a variable for each unscaled index
            unscaled_indices = ['pet', 'pdsi', 'phdi', 'pmdi', 'zindex', 'scpdsi']
            for variable_name in unscaled_indices:
                
                # get the attributes based on the name
                variable_attributes = _variable_attributes(variable_name)

                # create variables with scale month
                data_variable = new_dataset.createVariable(variable_name,
                                                           data_dtype,
                                                           ('division', 'time',),
                                                           fill_value=fill_value, 
                                                           zlib=False)
                data_variable.setncatts(variable_attributes)
     
            # create a variable for each scaled index
            scaled_indices = ['pnp', 'spi_gamma', 'spi_pearson', 'spei_gamma', 'spei_pearson']
            for scaled_index in scaled_indices:
                for months in self.scale_months:
                     
                    variable_name = scaled_index + '_{}'.format(str(months).zfill(2))
                     
                    # get the attributes based on the name and number of scale months
                    variable_attributes = _variable_attributes(scaled_index, months)
                    
                    # create month scaled variable
                    data_variable = new_dataset.createVariable(variable_name,
                                                               data_dtype,
                                                               ('division', 'time',),
                                                               fill_value=fill_value, 
                                                               zlib=False)
                    data_variable.setncatts(variable_attributes)
     
    #-----------------------------------------------------------------------------------------------------------------------
    def _compute_and_write_division(self, div_index):
        """
        Computes indices for a single division, writing the output into NetCDF.
        
        :param div_index: 
        """
        
        # open the NetCDF files 
        with netCDF4.Dataset(self.divisions_file, 'a') as divisions_dataset:
            
            climdiv_id = divisions_dataset['division'][div_index]
            
            # only process divisions within CONUS, 101 - 4809
            if climdiv_id > 4899:
                return
            
            logger.info('Processing indices for division %s', climdiv_id)
        
            # read the division of input temperature values 
            temperature = divisions_dataset[self.var_name_temperature][div_index, :]    # assuming (divisions, time) orientation
            
            # initialize the latitude outside of the valid range, in order to use this within a conditional below to verify a valid latitude
            latitude = -100.0  
    
            # latitudes are only available for certain divisions, make sure we have one for this division index
            if div_index < divisions_dataset['lat'][:].size:
                
                # get the actual latitude value (assumed to be in degrees north) for the latitude slice specified by the index
                latitude = divisions_dataset['lat'][div_index]
    
            # only proceed if the latitude value is within valid range            
            if not np.isnan(latitude) and (latitude < 90.0) and (latitude > -90.0):
                
                # convert temperatures from Fahrenheit to Celsius, if necessary
                temperature_units = divisions_dataset[self.var_name_temperature].units
                if temperature_units in ['degree_Fahrenheit', 'degrees Fahrenheit', 'degrees F', 'fahrenheit', 'Fahrenheit', 'F']:
                    
                    # TODO make sure this application of the ufunc is any faster  pylint: disable=fixme
                    temperature = _f2c(temperature)
#                     temperature = np.apply_along_axis(self._f2c, 0, temperature)
    
                elif temperature_units not in ['degree_Celsius', 'degrees Celsius', 'degrees C', 'celsius', 'Celsius', 'C']:
                    
                    raise ValueError('Unsupported temperature units: \'{0}\''.format(temperature_units))
        
                # use the numpy.apply_along_axis() function for computing indices such as PET that take a single time series
                # array as input (i.e. each division's time series is the initial 1-D array argument to the function we'll apply)
                
                logger.info('\tComputing PET for division %s', climdiv_id)
    
                # either use the original NCDC method or Thornthwaite method (default) for PET calculation
                if self.use_orig_pe:

                    # get B and H, originally taken from climate divisions soil constants file
                    B = divisions_dataset['B'][div_index]
                    H = divisions_dataset['H'][div_index]
                    pet_time_series = pdinew.potential_evapotranspiration(temperature,
                                                                          latitude,
                                                                          self.data_start_year,
                                                                          B,
                                                                          H)

                else:

                    # compute PET across all longitudes of the latitude slice
                    # Thornthwaite PE
                    pet_time_series = indices.pet(temperature, 
                                                  latitude_degrees=latitude, 
                                                  data_start_year=self.data_start_year)
                            
                # the above returns PET in millimeters, note this for further consideration
                pet_units = 'millimeter'
                
                # write the PET values to NetCDF        
                lock.acquire()
                divisions_dataset['pet'][div_index, :] = np.reshape(pet_time_series, (1, pet_time_series.size))
                divisions_dataset.sync()
                lock.release()
    
            else:
                
                pet_time_series = np.full(temperature.shape, np.NaN)
                pet_units = None
    
            # read the division's input precipitation and available water capacity values
            precip_time_series = divisions_dataset[self.var_name_precip][div_index, :]   # assuming (divisions, time) orientation
            
            if div_index < divisions_dataset[self.var_name_soil][:].size:
                awc = divisions_dataset[self.var_name_soil][div_index]               # assuming (divisions) orientation
                awc += 1   # AWC values need to include top inch, values from the soil file do not, so we add top inch here
            else:
                awc = np.NaN
                
            # allocate arrays to contain a latitude slice of Palmer values
            time_size = divisions_dataset['time'].size
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
                if divisions_dataset[self.var_name_precip].units in possible_mm_units:
                    precip_time_series = precip_time_series * mm_to_inches_multiplier
        
                if not np.isnan(pet_time_series).all():
                
                    # compute Palmer indices if we have valid inputs
                    if not np.isnan(awc):
                            
                        # if PET is in mm, convert to inches
                        if pet_units in possible_mm_units:
                            pet_time_series = pet_time_series * mm_to_inches_multiplier
        
                        # PET is in mm, convert to inches since the Palmer uses imperial units
                        pet_time_series = pet_time_series * mm_to_inches_multiplier
        
                        logger.info('\tComputing PDSI for division %s', climdiv_id)
    
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
                        divisions_dataset['pdsi'][div_index, :] = np.reshape(pdsi, (1, pdsi.size))
                        divisions_dataset['phdi'][div_index, :] = np.reshape(phdi, (1, phdi.size))
                        divisions_dataset['pmdi'][div_index, :] = np.reshape(pmdi, (1, pmdi.size))
                        divisions_dataset['scpdsi'][div_index, :] = np.reshape(pdsi, (1, scpdsi.size))
                        divisions_dataset['zindex'][div_index, :] = np.reshape(zindex, (1, zindex.size))
                        divisions_dataset.sync()
                        lock.release()
        
                    # process the SPI and SPEI at the specified month scales
                    for months in self.scale_months:
                        
                        logger.info('\tComputing SPI/SPEI/PNP at %s-month scale for division %s', months, climdiv_id)
    
                        #TODO ensure that the precipitation and PET values are using the same units  pylint: disable=fixme
                        
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
                        divisions_dataset[spei_gamma_variable_name][div_index, :] =   np.reshape(spei_gamma, (1, spei_gamma.size))
                        divisions_dataset[spei_pearson_variable_name][div_index, :] = np.reshape(spei_pearson, (1, spei_pearson.size))
                        divisions_dataset[spi_gamma_variable_name][div_index, :] =    np.reshape(spi_gamma, (1, spi_gamma.size))
                        divisions_dataset[spi_pearson_variable_name][div_index, :] =  np.reshape(spi_pearson, (1, spi_pearson.size))
                        divisions_dataset[pnp_variable_name][div_index, :] =          np.reshape(pnp, (1, pnp.size))
                        divisions_dataset.sync()
                        lock.release()

    #-------------------------------------------------------------------------------------------------------------------
    def run(self):
        
        # initialize the output NetCDF that will contain the computed indices
        with netCDF4.Dataset(self.divisions_file) as input_dataset:
            
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
#        
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
#@numba.jit
def _variable_attributes(index_name,
                         months=None):

    """
    Finds correct variable attributes for climate indices that will be computed by this processor.
    
    :param index_name: name of index for which attributes are requested
    :param months: for month-scaled indices a number of months to use as scale
    :return: dictionary of attribute names to values 
    """
    if index_name == 'pet':
          
        variable_name = 'pet'
        variable_attributes = {'standard_name': 'pet',
                               'long_name': 'Potential Evapotranspiration (PET), from Thornthwaite\'s equation',
                               'valid_min': 0.0,
                               'valid_max': 2000.0,
                               'units': 'millimeter'}
      
    elif index_name == 'pdsi':
          
        variable_name = 'pdsi'
        variable_attributes = {'standard_name': 'pdsi',
                               'long_name': 'Palmer Drought Severity Index (PDSI)',
                               'valid_min': -10.0,
                               'valid_max': 10.0}
      
    elif index_name == 'scpdsi':
          
        variable_name = 'scpdsi'
        variable_attributes = {'standard_name': 'scpdsi',
                               'long_name': 'Self-calibrated Palmer Drought Severity Index (PDSI)',
                               'valid_min': -10.0,
                               'valid_max': 10.0}
      
    elif index_name == 'phdi':
          
        variable_name = 'phdi'
        variable_attributes = {'standard_name': 'phdi',
                               'long_name': 'Palmer Hydrological Drought Index (PHDI)',
                               'valid_min': -10.0,
                               'valid_max': 10.0}
      
    elif index_name == 'pmdi':
          
        variable_name = 'pmdi'
        variable_attributes = {'standard_name': 'pmdi',
                               'long_name': 'Palmer Modified Drought Index (PMDI)',
                               'valid_min': -10.0,
                               'valid_max': 10.0}
      
    elif index_name == 'zindex':
          
        variable_name = 'zindex'
        variable_attributes = {'standard_name': 'zindex',
                               'long_name': 'Palmer Z-Index',
                               'valid_min': -10.0,
                               'valid_max': 10.0}

    else:

        # use the scale months in the variable name        
        variable_name = index_name + '_{}'.format(str(months).zfill(2))
    
        if index_name == 'pnp':
        
            variable_attributes = {'standard_name': variable_name,
                                   'long_name': 'Percent average precipitation, {}-month scale'.format(months),
                                   'valid_min': 0,
                                   'valid_max': 10.0,
                                   'units': 'percent of average'}

        elif index_name == 'spi_gamma':
        
            variable_attributes = {'standard_name': variable_name,
                                   'long_name': 'SPI (Gamma), {}-month scale'.format(months),
                                   'valid_min': -3.09,
                                   'valid_max': 3.09}
        
        elif index_name == 'spi_pearson':
        
            variable_attributes = {'standard_name': variable_name,
                                   'long_name': 'SPI (Pearson), {}-month scale'.format(months),
                                   'valid_min': -3.09,
                                   'valid_max': 3.09}
        
        elif index_name == 'spei_gamma':
        
            variable_attributes = {'standard_name': variable_name,
                                   'long_name': 'SPEI (Gamma), {}-month scale'.format(months),
                                   'valid_min': -3.09,
                                   'valid_max': 3.09}
        
        elif index_name == 'spei_pearson':
        
            variable_attributes = {'standard_name': variable_name,
                                   'long_name': 'SPEI (Pearson), {}-month scale'.format(months),
                                   'valid_min': -3.09,
                                   'valid_max': 3.09}

        else:
        
            message = '{0} is an unsupported index type'.format(index_name)
            logger.error(message)
            raise ValueError(message)

    return variable_attributes
    
#-----------------------------------------------------------------------------------------------------------------------
@numba.vectorize([numba.float64(numba.float64),
                  numba.float32(numba.float32)])
def _f2c(t):
    '''
    Converts a temperature value from Fahrenheit to Celsius
    '''
    return (t-32)*5.0/9

#-----------------------------------------------------------------------------------------------------------------------
def _rmse(predictions, targets):
    """
    Root mean square error
    
    :param predictions: np.ndarray
    :param targets: np.ndarray
    :return: np.ndarray
    """
    return np.sqrt(((predictions - targets) ** 2).mean())

#-----------------------------------------------------------------------------------------------------------------------
def _plot_and_save_histogram(difference_values,
                             number_of_bins,
                             range_lower, 
                             range_upper,
                             index_name,
                             climdiv_id,
                             title,
                             output_filepath):
    
    # plot a histogram of the differences
    plt.gcf().clear()
    plt.hist(difference_values[:], bins=number_of_bins, range=(range_lower, range_upper))
    plt.title(title + ': {0}, Division {1}'.format(index_name, climdiv_id))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
    # save to file
    logger.info('Saving histogram plot for index %s to file %s', index_name, output_filepath)
    plt.savefig(output_filepath)

#-----------------------------------------------------------------------------------------------------------------------
def _plot_and_save_lines(expected,
                         actual,
                         difference_values,
                         rmse,
                         climdiv_id,
                         varname,
                         output_filepath):

    # set figure size to (x, y)
    plt.figure(figsize=(30, 6))
    
    # plot the values and differences
    x = np.arange(difference_values.size)
    ax = plt.axes()
    ax.set_ylim([-5, 5])
    plt.axhline()
    expected_line, = plt.plot(x, expected, color='blue', label='NCEI (expected)')
    actual_line, = plt.plot(x, actual, color='yellow', linestyle='--', label='NIDIS (actual)')
    diffs_line, = plt.plot(x, difference_values, color='red', label='Difference')
    plt.legend(handles=[expected_line, actual_line, diffs_line], loc='upper left')
    plt.title('Comparison for division {0}: {1}     (RMSE: {2})'.format(climdiv_id, varname, rmse))
    plt.xlabel("months")
    plt.ylabel("value")
    
    plt.subplots_adjust(left=0.02, right=0.99, top=0.9, bottom=0.1)
    
    # save to file
    logger.info('Saving histogram plot for index %s to file %s', varname, output_filepath)
    plt.savefig(output_filepath, bbox_inches='tight')

#     plt.show()
    plt.close()

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
        parser.add_argument("--out_file", 
                            help="NetCDF output file containing variables composed from the input data and soil files", 
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
        parser.add_argument("--orig_pe", 
                            help="Use the original NCDC method for calculating potential evapotranspiration (PE) used in original Fortran", 
                            type=bool,
                            default=False,
                            required=False)
        args = parser.parse_args()

        # variable names used within the monthly NetCDF
        temp_var_name = 'tavg'
        precip_var_name = 'prcp'
        awc_var_name = 'awc'
        
#         # perform an ingest of the NCEI nClimDiv datasets for input (temperature  
#         # and precipitation) plus monthly computed indices for comparison
#         ingest_nclimdiv.ingest_netcdf_latest(args.out_file,
#                                              temp_var_name,
#                                              precip_var_name,
#                                              awc_var_name)

        # perform the processing
        process.process_nclimdiv(args.out_file,
                                 precip_var_name,
                                 temp_var_name,
                                 awc_var_name,
                                 args.month_scales,
                                 args.calibration_start_year,
                                 args.calibration_end_year,
                                 use_orig_pe=True)
        
#         # perform the processing
#         divisions_processor = DivisionsProcessor(args.out_file,
#                                                  precip_var_name,
#                                                  temp_var_name,
#                                                  awc_var_name,
#                                                  args.month_scales,
#                                                  args.calibration_start_year,
#                                                  args.calibration_end_year,
#                                                  use_orig_pe=True)
#         divisions_processor.run()
        
        # open the NetCDF files
        with netCDF4.Dataset(args.out_file, 'a') as dataset:

            # variable names for variables to diff from the two datasets
            comparison_arrays = {'PDSI': ('cmb_pdsi', 'pdsi'),
                                 'PHDI': ('cmb_phdi', 'phdi'),
                                 'PMDI': ('cmb_pmdi', 'pmdi'),
                                 'Z-Index': ('cmb_zndx', 'zindex')}
            for index, var_names in comparison_arrays.items():
                    
                # TODO validate that the two variables exist, have compatible dimensions/units, etc., all of which is assumed below  pylint: disable=fixme

                logger.info('Computing differences on variable %s', index)
            
                # allocate an array for the differences for this variable
                diffs = {}
                
                # common title for plots
                histogram_title = 'CMB vs. NIDIS: '
     
                # count the number of divisions we've analyzed in order to get a mean for various statistics such as RMSE
                divs_analyzed = 0
                rmse_sum = 0.0
                
                for division_index, division_id in enumerate(dataset.variables['division'][:]):
                 
                    # only process divisions within CONUS, 101 - 4809
                    if division_id > 4899:
                        continue
                    divs_analyzed += 1
                    
                    logger.info('Computing diffs for climate division ID: %s', division_id)
                    
                    # get the variable var_names for the month, mask the NaNs (data assumed to be in (division, time) dimension order)
                    data_CMB = np.ma.masked_invalid(dataset.variables[var_names[0]][division_index, :], copy=False)
                    data_NIDIS = np.ma.masked_invalid(dataset.variables[var_names[1]][division_index, :], copy=False)
             
                    # get the difference of the two, add into the differences array at the correct slot for this division
                    differences = data_CMB - data_NIDIS
                    diffs[division_index] = differences

                    # get the RMSE for the two sets of values
                    error = _rmse(data_NIDIS, data_CMB)
                    rmse_sum += error
     
                    # plot the differences as a histogram and save to file
                    _plot_and_save_histogram(differences,
                                             80,   # number_of_bins
                                             -2,   # lower range
                                             2,    # upper range
                                             index,
                                             division_id,
                                             histogram_title,
                                             'C:/home/data/nclimdiv/diffs_histogram_{0}_{1}.png'.format(var_names[1], division_id))
     
                    # plot and save line graphs showing correlation of values and differences
                    _plot_and_save_lines(data_NIDIS,
                                         data_CMB,
                                         differences,
                                         error,
                                         division_id,
                                         index,
                                         'C:/home/data/nclimdiv/diffs_line_{0}_{1}.png'.format(var_names[1], division_id))
                    
                    # add to the differences dictionary with this division ID key 
                    diffs[division_id] = differences

                # make sure that the variable name isn't already in use
                diff_variable_name = 'diffs_' + index
                if diff_variable_name in dataset.variables.keys():

                    variable = dataset.variables[diff_variable_name]
                    
                else:
                    
                    # get the NetCDF datatype applicable to the data array we'll store in the variable
                    random_array = random.choice(list(diffs.values()))
                    netcdf_data_type = netcdf_utils.find_netcdf_datatype(random_array[0])
                    
                    # create the variable, set the attributes
                    variable = dataset.createVariable(diff_variable_name, 
                                                      netcdf_data_type, 
                                                      ('division', 'time',), 
                                                      fill_value=np.NaN)
#                     variable.setncatts(variable_attributes)
                
                # get the total number of time steps
                times_size = dataset.variables['time'][:].size
                
                # loop over each existing division and add the corresponding data array, if one was provided
                for division_index, division_id in enumerate(list(dataset.variables['division'][:])):
                    
                    # make sure we have a data array of monthly values for this division
                    if division_index in diffs.keys():
        
                        # make sure the array has the expected number of time steps 
                        data_array = diffs[division_index]
                        if data_array.size == times_size:
                        
                            # assign the array into the current division's slot in the variable
                            variable[division_index, :] = np.reshape(data_array, (1, times_size))
        
                        else:
        
                            logger.info('Unexpected size of data array for division index {0} -- '.format(division_index) + 
                                        'expected {0} time steps but the array contains {1}'.format(times_size, data_array.size))

            # report summary statistics
            print('\nMean RMSE: {0}'.format(rmse_sum / divs_analyzed))
            
        # report on the elapsed time
        end_datetime = datetime.now()
        logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        logger.exception('Failed to complete', exc_info=True)
        raise
    