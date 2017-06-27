import argparse
from datetime import datetime
import indices
import logging
import multiprocessing
import netCDF4
import netcdf_utils
import numpy as np
import os
import subprocess
import sys

# set up a basic, global logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

# multiprocessing locks we'll use to synchronize I/O writes to NetCDF files, one per each output file
lock = multiprocessing.Lock()

# ignore warnings
import warnings
warnings.simplefilter('ignore', Warning)

#-----------------------------------------------------------------------------------------------------------------------
def init_process(worker_input_netcdf,
                 worker_output_netcdf,
                 worker_temp_var_name,
                 worker_precip_var_name,
                 worker_awc_var_name,
                 worker_scale_months,
                 worker_data_start_year,
                 worker_data_end_year,
                 worker_calibration_start_year, 
                 worker_calibration_end_year):
    
    # put the arguments into the global namespace
    global input_netcdf, \
           output_netcdf, \
           temp_var_name, \
           precip_var_name, \
           awc_var_name, \
           scale_months, \
           data_start_year, \
           data_end_year, \
           calibration_start_year, \
           calibration_end_year
           
    input_netcdf = worker_input_netcdf
    output_netcdf = worker_output_netcdf
    temp_var_name = worker_temp_var_name
    precip_var_name = worker_precip_var_name
    awc_var_name = worker_awc_var_name
    scale_months = worker_scale_months
    data_start_year = worker_data_start_year
    data_end_year = worker_data_end_year
    calibration_start_year = worker_calibration_start_year
    calibration_end_year = worker_calibration_end_year
    
#-----------------------------------------------------------------------------------------------------------------------
def initialize_netcdf(new_netcdf,
                      template_netcdf,
                      month_scales=[1, 2, 3, 6, 12, 24]):
    '''
    This function is used to initialize and return a netCDF4.Dataset object.
    
    :param new_netcdf: the file path/name of the new NetCDF Dataset object to be created and returned by this function
    :param template_netcdf: an existing/open NetCDF Dataset object which will be used as a template for the Dataset
                            that will be created by this function
    :param month_scales: some of the indicators this script computes, such as SPI, SPEI, and PNP, are typically computed for multiple
                         month scales (i.e. 1-month, 3-month, 6-month, etc.), these can be specified here as a list of integers 
    '''

    # use NaNs as our default fill/missing value
    fill_value=np.float32(np.NaN)

    # open the NetCDF datasets within a context manager
    with netCDF4.Dataset(template_netcdf) as template_dataset, \
         netCDF4.Dataset(new_netcdf, 'w') as new_dataset:
 
        # get the template's dimension sizes
        divisions_size = template_dataset.variables['division'].size
    
        # copy the global attributes from the input
        # TODO/FIXME add/modify global attributes to correspond with the actual dataset
        new_dataset.setncatts(template_dataset.__dict__)
        
        # use "ClimDiv-1.0" as the Conventions setting in order to facilitate visualization by the NOAA Weather and Climate Toolkit
        new_dataset.setncattr("Conventions", "ClimDiv-1.0")
        
        # create the time, x, and y dimensions
        new_dataset.createDimension('time', None)
        new_dataset.createDimension('division', divisions_size)
    
        # get the appropriate data types to use for the variables
        time_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables['time'])
        divisions_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables['division'])
        data_dtype = netcdf_utils.find_netcdf_datatype(fill_value)
    
        # create the coordinate variables
        time_variable = new_dataset.createVariable('time', time_dtype, ('time',))
        division_variable = new_dataset.createVariable('division', divisions_dtype, ('division',))

        # set the coordinate variables' attributes and values
        time_variable.setncatts(template_dataset.variables['time'].__dict__)
        time_variable[:] = template_dataset.variables['time'][:]
        division_variable.setncatts(template_dataset.variables['division'].__dict__)
        division_variable[:] = template_dataset.variables['division'][:]

        # create a variable for each unscaledindicator
        unscaled_indicators = ['pet', 'pdsi', 'phdi', 'pmdi', 'zindex', 'scpdsi']
        for indicator in unscaled_indicators:
            if indicator == 'pet':
                
                variable_name = 'pet'
                variable_attributes = {'standard_name': 'pet',
                                       'long_name': 'Potential Evapotranspiration (PET), from Thornthwaite\'s equation',
                                       'valid_min': 0.0,
                                       'valid_max': 2000.0,
                                       'units': 'millimeter'}
            
            elif indicator == 'pdsi':
                
                variable_name = 'pdsi'
                variable_attributes = {'standard_name': 'pdsi',
                                       'long_name': 'Palmer Drought Severity Index (PDSI)',
                                       'valid_min': -10.0,
                                       'valid_max': 10.0}
            
            elif indicator == 'scpdsi':
                
                variable_name = 'scpdsi'
                variable_attributes = {'standard_name': 'scpdsi',
                                       'long_name': 'Self-calibrated Palmer Drought Severity Index (PDSI)',
                                       'valid_min': -10.0,
                                       'valid_max': 10.0}
            
            elif indicator == 'phdi':
                
                variable_name = 'phdi'
                variable_attributes = {'standard_name': 'phdi',
                                       'long_name': 'Palmer Hydrological Drought Index (PHDI)',
                                       'valid_min': -10.0,
                                       'valid_max': 10.0}
            
            elif indicator == 'pmdi':
                
                variable_name = 'pmdi'
                variable_attributes = {'standard_name': 'pmdi',
                                       'long_name': 'Palmer Modified Drought Index (PMDI)',
                                       'valid_min': -10.0,
                                       'valid_max': 10.0}
            
            elif indicator == 'zindex':
                
                variable_name = 'zindex'
                variable_attributes = {'standard_name': 'zindex',
                                       'long_name': 'Palmer Z-Index',
                                       'valid_min': -10.0,
                                       'valid_max': 10.0}

            # create variables with scale month
            data_variable = new_dataset.createVariable(variable_name,
                                                       data_dtype,
                                                       ('division', 'time',),
                                                       fill_value=fill_value, 
                                                       zlib=False)
            data_variable.setncatts(variable_attributes)

        # create a variable for each scaled indicator
        scaled_indicators = ['pnp', 'spi_gamma', 'spi_pearson', 'spei_gamma', 'spei_pearson']
        for indicator in scaled_indicators:
            for months in month_scales:
                
                variable_name = indicator + '_{}'.format(str(months).zfill(2))
                
                if indicator == 'spi_gamma':
            
                    variable_attributes = {'standard_name': variable_name,
                                           'long_name': 'SPI (Gamma), {}-month scale'.format(months),
                                           'valid_min': -3.09,
                                           'valid_max': 3.09}

                elif indicator == 'spi_pearson':
            
                    variable_attributes = {'standard_name': variable_name,
                                           'long_name': 'SPI (Pearson), {}-month scale'.format(months),
                                           'valid_min': -3.09,
                                           'valid_max': 3.09}

                elif indicator == 'spei_gamma':
            
                    variable_attributes = {'standard_name': variable_name,
                                           'long_name': 'SPEI (Gamma), {}-month scale'.format(months),
                                           'valid_min': -3.09,
                                           'valid_max': 3.09}

                elif indicator == 'spei_pearson':
            
                    variable_attributes = {'standard_name': variable_name,
                                           'long_name': 'SPEI (Pearson), {}-month scale'.format(months),
                                           'valid_min': -3.09,
                                           'valid_max': 3.09}

                elif indicator == 'pnp':
            
                    variable_attributes = {'standard_name': variable_name,
                                           'long_name': 'Percent average precipitation, {}-month scale'.format(months),
                                           'valid_min': 0,
                                           'valid_max': 10.0,
                                           'units': 'percent of average'}

                # create month scaled variable 
                data_variable = new_dataset.createVariable(variable_name,
                                                           data_dtype,
                                                           ('division', 'time',),
                                                           fill_value=fill_value, 
                                                           zlib=False)
                data_variable.setncatts(variable_attributes)
        
#-----------------------------------------------------------------------------------------------------------------------
def construct_nco_command(netcdf_operator):

    # set the data directory path appropriate to the current platform
    if ((sys.platform == 'linux') or (sys.platform == 'linux2')):
        nco_home = '/home/james.adams/anaconda3/bin'
        suffix = ''
#         # to_null = ' >/dev/null 2>&1'  # use this if NCO error/warning/info messages become problematic
#         to_null = ''
    else:  # Windows
        nco_home = 'C:/nco'
        suffix = '.exe --no_tmp_fl'
#         # to_null = ' >NUL 2>NUL'  # use this if NCO error/warning/info messages become problematic
#         to_null = ''

    # get the proper executable path for the NCO command that'll be used to perform the concatenation operation
    normalized_executable_path = os.path.normpath(nco_home)
    return os.path.join(os.sep, normalized_executable_path, netcdf_operator) + suffix # + to_null

#-----------------------------------------------------------------------------------------------------------------------
def convert_and_move_netcdf(input_and_output_netcdfs):

    input_netcdf = input_and_output_netcdfs[0]
    output_netcdf = input_and_output_netcdfs[1]

    # get the proper executable path for the NCO command that'll be used to perform the conversion/compression 
    ncks = construct_nco_command('ncks')

    # build and run the command used to convert the file into a compressed NetCDF4 file
    convert_and_compress_command = ncks + ' -O -4 -L 4 -h ' + input_netcdf + ' ' + output_netcdf
    logger.info('Converting the temporary/work NetCDF file [{0}] into a compressed NetCDF4 file [{1}]'\
                .format(input_netcdf, output_netcdf))
    logger.info('NCO conversion/compression command:  {0}'.format(convert_and_compress_command))
    subprocess.call(convert_and_compress_command, shell=True)
    
    # remove the temporary/work file which will no longer needed
    logger.info('Removing the temporary/work file [{0}]'.format(input_netcdf))
    os.remove(input_netcdf)

#-----------------------------------------------------------------------------------------------------------------------
def f2c(t):
    '''
    Converts a temperature value from Fahrenheit to Celsius
    '''
    return (t-32)*5.0/9

#-----------------------------------------------------------------------------------------------------------------------
def compute_and_write_division(division_index):
    '''
    #TODO explain why this wrapper function is required (facilitates multiprocessing since the default way of mapping a function
    is with a single argument, the remaining arguments required are picked up from the global namespace in the function call below?)
    '''
    
    process_division(division_index,
                     input_netcdf,
                     output_netcdf,
                     temp_var_name,
                     precip_var_name,
                     awc_var_name,
                     scale_months,
                     data_start_year,
                     data_end_year,
                     calibration_start_year,
                     calibration_end_year)

#-----------------------------------------------------------------------------------------------------------------------
def process_division(division_index,
                     input_file,
                     output_file,
                     temp_var_name,
                     precip_var_name,
                     awc_var_name,
                     scale_months,
                     data_start_year,
                     data_end_year,
                     calibration_start_year,
                     calibration_end_year):
    
    # use a different name alias for the data_start_year in order to avoid conflicts with function arguments that use the same name
    initial_data_year = data_start_year
    
    # open the NetCDF files 
    with netCDF4.Dataset(input_file) as input_dataset:
        
        division_id = input_dataset['division'][division_index]
        logger.info('Processing indices for division {0}'.format(division_id))
    
        # read the division of input temperature values 
        temperature = input_dataset[temp_var_name][division_index, :]    # assuming (divisions, time) orientation
        
        # initialize the latitude outside of the valid range, in order to use this within a conditional below to verify a valid latitude
        latitude = -100.0  

        # latitudes are only available for certain divisions, make sure we have one for this division index
        if division_index < input_dataset['lat'][:].size:
            
            # get the actual latitude value (assumed to be in degrees north) for the latitude slice specified by the index
            latitude = input_dataset['lat'][division_index]

        # only proceed if the latitude value is within valid range            
        if not np.isnan(latitude) and (latitude < 90.0) and (latitude > -90.0):
            
            # convert temperatures from Fahrenheit to Celsius, if necessary
            temperature_units = input_dataset[temp_var_name].units
            if temperature_units in ['degree_Fahrenheit', 'fahrenheit', 'Fahrenheit', 'F']:
                
                temperature = np.apply_along_axis(f2c, 0, temperature)

            elif temperature_units not in ['degree_Celsius', 'celsius', 'Celsius', 'C']:
                
                raise ValueError('Unsupported temperature units: \'{0}\''.format(temperature_units))
    
            # use the numpyapply_along_axis() function for computing indicators such as PET that take a single time series 
            # array as input (i.e. each division's time series is the initial 1-D array argument to the function we'll apply)
            
            logger.info('\tComputing PET for division {0}'.format(division_id))

            # compute PET across all longitudes of the latitude slice
            pet_time_series = indices.pet(temperature, 
                                          latitude_degrees=latitude, 
                                          data_start_year=initial_data_year)
        
            # the above returns PET in millimeters, note this for further consideration
            pet_units = 'millimeter'
            
            # write the PET values to NetCDF        
            lock.acquire()
            with netCDF4.Dataset(output_file, 'a') as output_dataset:
                output_dataset['pet'][division_index, :] = np.reshape(pet_time_series, (1, pet_time_series.size))
                output_dataset.sync()
            lock.release()

        else:
            
            pet_time_series = np.full(temperature.shape, np.NaN)
            
        # read the division's input precipitation and available water capacity values 
        precip_time_series = input_dataset[precip_var_name][division_index, :]   # assuming (divisions, time) orientation
        
        if division_index < input_dataset[awc_var_name][:].size:
            awc = input_dataset[awc_var_name][division_index]               # assuming (divisions) orientation
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
            if input_dataset[precip_var_name].units in possible_mm_units:
                precip_time_series = precip_time_series * mm_to_inches_multiplier
    
            if not np.isnan(pet_time_series).all():
            
                # compute Palmer indicators if we have valid inputs
                if not np.isnan(awc):
                        
                    # if PET is in mm, convert to inches
                    if pet_units in possible_mm_units:
                        pet_time_series = pet_time_series * mm_to_inches_multiplier
    
                    # PET is in mm, convert to inches since the Palmer uses imperial units
                    pet_time_series = pet_time_series * mm_to_inches_multiplier
    
                    logger.info('\tComputing PDSI for division {0}'.format(division_id))

                    # compute Palmer indicators
                    palmer_values = indices.scpdsi(precip_time_series,
                                                   pet_time_series,
                                                   awc,
                                                   initial_data_year,
                                                   calibration_start_year,
                                                   calibration_end_year)
        
                    scpdsi = palmer_values[0]
                    pdsi = palmer_values[1]
                    phdi = palmer_values[2]
                    pmdi = palmer_values[3]
                    zindex = palmer_values[4]
    
                    # write the PDSI values to NetCDF
                    lock.acquire()
                    with netCDF4.Dataset(output_file, 'a') as output_dataset:
                        output_dataset['pdsi'][division_index, :] = np.reshape(pdsi, (1, pdsi.size))
                        output_dataset['phdi'][division_index, :] = np.reshape(phdi, (1, phdi.size))
                        output_dataset['pmdi'][division_index, :] = np.reshape(pmdi, (1, pmdi.size))
                        output_dataset['scpdsi'][division_index, :] = np.reshape(pdsi, (1, scpdsi.size))
                        output_dataset['zindex'][division_index, :] = np.reshape(zindex, (1, zindex.size))
                        output_dataset.sync()
                    lock.release()
    
                # process the SPI and SPEI at the specified month scales
                for months in scale_months:
                    
                    logger.info('\tComputing SPI/SPEI/PNP at {0}-month scale for division {1}'.format(months, division_id))

                    #TODO ensure that the precipitation and PET values are using the same units
                    
                    # compute SPEI/Gamma
                    spei_gamma = indices.spei_gamma(months,
                                                    precip_time_series,
                                                    pet_mm=pet_time_series)

                    # compute SPEI/Pearson
                    spei_pearson = indices.spei_pearson(months,
                                                        data_start_year,
                                                        precip_time_series,
                                                        pet_mm=pet_time_series,
                                                        calibration_year_initial=calibration_start_year,
                                                        calibration_year_final=calibration_end_year)
                     
                    # compute SPI/Gamma
                    spi_gamma = indices.spi_gamma(precip_time_series, 
                                                  months)
             
                    # compute SPI/Pearson
                    spi_pearson = indices.spi_pearson(precip_time_series, 
                                                      months,
                                                      data_start_year,
                                                      calibration_start_year, 
                                                      calibration_end_year)        
        
                    # compute PNP
                    pnp = indices.percentage_of_normal(precip_time_series, 
                                                       months,
                                                       data_start_year,
                                                       calibration_start_year, 
                                                       calibration_end_year)        
    
                    # create variable names which should correspond to the appropriate scaled indicator output variables
                    scaled_name_suffix = str(months).zfill(2)
                    spei_gamma_variable_name = 'spei_gamma_' + scaled_name_suffix
                    spei_pearson_variable_name = 'spei_pearson_' + scaled_name_suffix
                    spi_gamma_variable_name = 'spi_gamma_' + scaled_name_suffix
                    spi_pearson_variable_name = 'spi_pearson_' + scaled_name_suffix
                    pnp_variable_name = 'pnp_' + scaled_name_suffix
    
                    # write the SPI, SPEI, and PNP values to NetCDF        
                    lock.acquire()
                    with netCDF4.Dataset(output_file, 'a') as output_dataset:
                        output_dataset[spei_gamma_variable_name][division_index, :] = np.reshape(spei_gamma, (1, spei_gamma.size))
                        output_dataset[spei_pearson_variable_name][division_index, :] = np.reshape(spei_pearson, (1, spei_pearson.size))
                        output_dataset[spi_gamma_variable_name][division_index, :] = np.reshape(spi_gamma, (1, spi_gamma.size))
                        output_dataset[spi_pearson_variable_name][division_index, :] = np.reshape(spi_pearson, (1, spi_pearson.size))
                        output_dataset[pnp_variable_name][division_index, :] = np.reshape(pnp, (1, pnp.size))
                        output_dataset.sync()
                    lock.release()

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    '''
    This script performs climate indicator processing for US climate divisions.
    
    Indicators included are SPI and SPEI (both Gamma and Pearson Type III fittings), PET, PNP, and Palmers (PDSI, scPDSI, PHDI, 
    PMDI, and Z-Index).
    
    A single input NetCDF containing temperature, precipitation, latitude, and available water capacity variables for US climate divisions 
    is required.
      
    Example command line arguments: 
    
        --input_file C:/home/climdivs/climdivs_201701.nc 
        --precip_var_name prcp 
        --temp_var_name tavg 
        --awc_var_name awc 
        --output_file C:/home/climdivs/climdivs_201701_indices.nc 
        --month_scales 1 2 3 6 12 24 
        --calibration_start_year 1951 
        --calibration_end_year 2010
    '''

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        logger.info("Start time:    {0}".format(start_datetime, '%x'))

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

        # initialize the output NetCDF that will contain the computed indices
        initialize_netcdf(args.output_file, args.input_file, args.month_scales)
        
        # open the NetCDF files 
        with netCDF4.Dataset(args.input_file) as input_dataset:
             
            # get the initial and final year of the input datasets
            time_variable = input_dataset.variables['time']
            data_start_year = netCDF4.num2date(time_variable[0], time_variable.units).year
            data_end_year = netCDF4.num2date(time_variable[-1], time_variable.units).year
 
            # get the number of divisions in the input dataset(s)
            divisions_size = input_dataset.variables['division'].size
        
        # create a process Pool, with copies of the shared array going to each pooled/forked process
        pool = multiprocessing.Pool(processes=1,#multiprocessing.cpu_count(),
                                    initializer=init_process,
                                    initargs=(args.input_file,
                                              args.output_file,
                                              args.temp_var_name,
                                              args.precip_var_name,
                                              args.awc_var_name,
                                              args.month_scales,
                                              data_start_year,
                                              data_end_year,
                                              args.calibration_start_year,
                                              args.calibration_end_year))
 
        # map the divisions indices as an arguments iterable to the compute function
        result = pool.map_async(compute_and_write_division, range(divisions_size))
                  
        # get the exception(s) thrown, if any
        result.get()
              
        # close the pool and wait on all processes to finish
        pool.close()
        pool.join()

#         # convert and move the output file
#         convert_and_move_netcdf([args.output_file, args.output_file])
              
        # report on the elapsed time
        end_datetime = datetime.now()
        logger.info("End time:      {}".format(end_datetime, '%x'))
        elapsed = end_datetime - start_datetime
        logger.info("Elapsed time:  {}".format(elapsed, '%x'))

    except Exception as ex:
        logger.exception('Failed to complete', exc_info=True)
        raise