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

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
# multiprocessing locks we'll use to synchronize I/O writes to NetCDF files, one per each output file
lock = multiprocessing.Lock()

#-----------------------------------------------------------------------------------------------------------------------
# ignore warnings
import warnings
warnings.simplefilter('ignore', Warning)

#-----------------------------------------------------------------------------------------------------------------------
def init_process(worker_input_netcdf,
                 worker_output_netcdf,
                 worker_precip_var_name,
                 worker_pet_var_name,
                 worker_awc_var_name,
                 worker_scale_months,
                 worker_data_start_year,
                 worker_calibration_start_year, 
                 worker_calibration_end_year):
    
    # put the arguments into the global namespace
    global input_netcdf, \
           output_netcdf, \
           precip_var_name, \
           pet_var_name, \
           awc_var_name, \
           scale_months, \
           data_start_year, \
           calibration_start_year, \
           calibration_end_year
           
    input_netcdf = worker_input_netcdf
    output_netcdf = worker_output_netcdf
    precip_var_name = worker_precip_var_name
    pet_var_name = worker_pet_var_name
    awc_var_name = worker_awc_var_name
    scale_months = worker_scale_months
    data_start_year = worker_data_start_year
    calibration_start_year = worker_calibration_start_year
    calibration_end_year = worker_calibration_end_year
    
#-----------------------------------------------------------------------------------------------------------------------
def initialize_netcdf(output_netcdf_filepath,
                      template_netcdf,
                      month_scales):
    '''
    This function is used to initialize and return a netCDF4.Dataset object.
    
    :param output_netcdf_filepath: the file path/name of the NetCDF Dataset object returned by this function
    :param template_netcdf: an existing/open NetCDF Dataset object which will be used as a template for the Dataset
                            that will be created by this function
    :param month_scales: the various month scales for which scaled indices (SPI, SPEI, PNP) will be computed
    :return: None
    '''

    fill_value=np.float32(np.NaN)

    # SPI and SPEI have a valid/useful range of [-3.09, 3.09]
    valid_min = -3.09
    valid_max = 3.09

    with netCDF4.Dataset(template_netcdf) as template_dataset, \
         netCDF4.Dataset(output_netcdf_filepath, 'w') as output_dataset:
 
        # get the template's dimension sizes
        divisions_count = template_dataset.variables['division'].size
        times_size = template_dataset.variables['time'].size
    
        # copy the global attributes from the input
        # TODO/FIXME add/modify global attributes to correspond with the actual output dataset
        output_dataset.setncatts(template_dataset.__dict__)
        
        # create the time, x, and y dimensions
        output_dataset.createDimension('time', times_size)
#         output_dataset.createDimension('time', None)
        output_dataset.createDimension('division', divisions_count)
    
        # get the appropriate data types to use for the variables
        time_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables['time'])
        divisions_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables['division'])
        data_dtype = netcdf_utils.find_netcdf_datatype(fill_value)
    
        # create the coordinate variables
        chunk_sizes = [template_dataset.variables['time'].size]
        time_variable = output_dataset.createVariable('time', time_dtype, ('time',), chunksizes=chunk_sizes)
        division_variable = output_dataset.createVariable('division', divisions_dtype, ('division',))

        # set the coordinate variables' attributes and values
        time_variable.setncatts(template_dataset.variables['time'].__dict__)
        
        time_variable[:] = template_dataset.variables['time'][:]
        division_variable.setncatts(template_dataset.variables['division'].__dict__)
        division_variable[:] = template_dataset.variables['division'][:]

        data_chunk_sizes = [1, template_dataset.variables['time'].size]

        # create a variable for each unscaled index
        for index in ['pdsi', 'phdi', 'pmdi', 'zindex', 'scpdsi', 'ET', 'PR', 'R', 'RO', 'PRO', 'L', 'PL', 'pet']:
            
            if index == 'pdsi':
                
                variable_name = index
                variable_attributes = {'standard_name': variable_name,
                                       'long_name': 'Palmer Drought Severity Index (PDSI)',
                                       'valid_min': -10.0,
                                       'valid_max': 10.0}
            
            elif index == 'scpdsi':
                
                variable_name = index
                variable_attributes = {'standard_name': variable_name,
                                       'long_name': 'Self-calibrated Palmer Drought Severity Index (PDSI)',
                                       'valid_min': -10.0,
                                       'valid_max': 10.0}
            
            elif index == 'phdi':
                
                variable_name = index
                variable_attributes = {'standard_name': variable_name,
                                       'long_name': 'Palmer Hydrological Drought Index (PHDI)',
                                       'valid_min': -10.0,
                                       'valid_max': 10.0}
            
            elif index == 'pmdi':
                
                variable_name = index
                variable_attributes = {'standard_name': variable_name,
                                       'long_name': 'Palmer Modified Drought Index (PMDI)',
                                       'valid_min': -10.0,
                                       'valid_max': 10.0}
            
            elif index == 'zindex':
                
                variable_name = index
                variable_attributes = {'standard_name': variable_name,
                                       'long_name': 'Palmer Z-Index',
                                       'valid_min': -10.0,
                                       'valid_max': 10.0}

            elif index == 'pet':
                
                variable_name = index
                variable_attributes = {'standard_name': variable_name,
                                       'long_name': 'Potential Evapotranspiration',
                                       'units': 'millimeter',
                                       'valid_min': 0.0,
                                       'valid_max': 1000.0}

            else:
                
                variable_name = 'wb_' + index
                variable_attributes = {'standard_name': variable_name}
                
            # create variables with scale month
            data_variable = output_dataset.createVariable(variable_name,
                                                          data_dtype,
                                                          ('division', 'time',),
                                                          chunksizes=data_chunk_sizes,
                                                          fill_value=fill_value, 
                                                          zlib=False)
            data_variable.setncatts(variable_attributes)

        # create a variable for each scaled index
        scaled_indicators = ['pnp', 'spi_gamma', 'spi_pearson', 'spei_gamma', 'spei_pearson']
        for index in scaled_indicators:
            for months in month_scales:
                
                variable_name = index + '_{}'.format(str(months).zfill(2))
                
                if index == 'spi_gamma':
            
                    variable_attributes = {'standard_name': variable_name,
                                           'long_name': 'SPI (Gamma), {}-month scale'.format(months),
                                           'valid_min': valid_min,
                                           'valid_max': valid_max}

                elif index == 'spi_pearson':
            
                    variable_attributes = {'standard_name': variable_name,
                                           'long_name': 'SPI (Pearson), {}-month scale'.format(months),
                                           'valid_min': valid_min,
                                           'valid_max': valid_max}

                elif index == 'spei_gamma':
            
                    variable_attributes = {'standard_name': variable_name,
                                           'long_name': 'SPEI (Gamma), {}-month scale'.format(months),
                                           'valid_min': valid_min,
                                           'valid_max': valid_max}

                elif index == 'spei_pearson':
            
                    variable_attributes = {'standard_name': variable_name,
                                           'long_name': 'SPEI (Pearson), {}-month scale'.format(months),
                                           'valid_min': valid_min,
                                           'valid_max': valid_max}

                elif index == 'pnp':
            
                    variable_attributes = {'standard_name': variable_name,
                                           'long_name': 'Percent of normal precipitation, {}-month scale'.format(months),
                                           'valid_min': 0,
                                           'valid_max': 10.0,
                                           'units': 'percent of average'}

                # create month scaled variable 
                data_variable = output_dataset.createVariable(variable_name,
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
    Convenience function for use passing global variables as arguments to the process_division() function? To be honest 
    not sure why this is present, acts sort of as a shell around process_division(), perhaps required in order to use 
    the map_async() function which takes only a single argument.
    
    :param division_index: index for the climate division to be computed
    '''
    
    process_division(division_index,
                     input_netcdf,
                     output_netcdf,
                     precip_var_name,
                     pet_var_name,
                     awc_var_name,
                     scale_months,
                     data_start_year,
                     calibration_start_year,
                     calibration_end_year)

#-----------------------------------------------------------------------------------------------------------------------
def process_division(division_index,
                     input_file,
                     output_file,
                     precip_var_name,
                     pet_var_name,
                     awc_var_name,
                     scale_months,
                     data_start_year,
                     calibration_start_year,
                     calibration_end_year):
    '''
    :param division_index:
    :param input_file:
    :param output_file:   
    :param precip_var_name:   
    :param pet_var_name:   
    :param awc_var_name:   
    :param scale_months:   
    :param data_start_year:   
    :param calibration_start_year:   
    :param calibration_end_year:   
    '''
    
    logger.info('Computing indices for division index {0}'.format(division_index))
    
    # open the NetCDF file
    with netCDF4.Dataset(input_file) as input_dataset:
             
        # read the division's input precipitation, PET, and available water capacity values 
        precip_time_series = input_dataset[precip_var_name][division_index, :]   # assuming (divisions, time) orientation
        pet_time_series = input_dataset[pet_var_name][division_index, :]                  # assuming (divisions, time) orientation
        
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
        ET = np.full(division_shape, np.NaN)
        PR = np.full(division_shape, np.NaN)
        R = np.full(division_shape, np.NaN)
        RO = np.full(division_shape, np.NaN)
        PRO = np.full(division_shape, np.NaN)
        L = np.full(division_shape, np.NaN)
        PL = np.full(division_shape, np.NaN)
    
        # compute SPI and SPEI for the current division only if we have valid inputs
        if not np.isnan(precip_time_series).all():
            
            # put precipitation into inches if not already
            mm_to_inches_multiplier = 0.0393701
            possible_mm_units = ['millimeters', 'millimeter', 'mm']
            if input_dataset[precip_var_name].units in possible_mm_units:
                precip_time_series = precip_time_series * mm_to_inches_multiplier
    
            if not np.isnan(pet_time_series).all():
            
                # compute Palmer indices if we have valid inputs
                if not np.isnan(awc):
                        
                    # if PET is in mm, convert to inches
                    # TODO make reasoning for this more clear    
                    # put PET into inches if not already
                    if ('units' in input_dataset[pet_var_name].ncattrs()) and \
                       (input_dataset[pet_var_name].getncattr('units') in possible_mm_units):
                            pet_time_series = pet_time_series * mm_to_inches_multiplier
    
    #                 logger.debug('     Division index {0}'.format(lon_index))
    
                    logger.info('\tComputing PDSI for division index {0}'.format(division_index))

                    # compute Palmer indices
                    palmer_values = indices.scpdsi(precip_time_series,
                                                   pet_time_series,
                                                   awc,
                                                   data_start_year,
                                                   calibration_start_year,
                                                   calibration_end_year)
        
                    # pull out the individual variables into their own separate arrays
                    scpdsi = palmer_values[0]
                    pdsi = palmer_values[1]
                    phdi = palmer_values[2]
                    pmdi = palmer_values[3]
                    zindex = palmer_values[4]
                    ET = palmer_values[5]
                    PR = palmer_values[6]
                    R = palmer_values[7]
                    RO = palmer_values[8]
                    PRO = palmer_values[9]
                    L = palmer_values[10]
                    PL = palmer_values[11]
                    
                    # write the PDSI values to NetCDF
                    lock.acquire()
                    with netCDF4.Dataset(output_file, 'a') as output_dataset:
                        output_dataset['pdsi'][division_index, :] = np.reshape(pdsi, (1, pdsi.size))
                        output_dataset['phdi'][division_index, :] = np.reshape(phdi, (1, phdi.size))
                        output_dataset['pmdi'][division_index, :] = np.reshape(pmdi, (1, pmdi.size))
                        output_dataset['scpdsi'][division_index, :] = np.reshape(pdsi, (1, scpdsi.size))
                        output_dataset['zindex'][division_index, :] = np.reshape(zindex, (1, zindex.size))
                        output_dataset['wb_ET'][division_index, :] = np.reshape(ET, (1, ET.size))
                        output_dataset['wb_PR'][division_index, :] = np.reshape(PR, (1, PR.size))
                        output_dataset['wb_R'][division_index, :] = np.reshape(R, (1, R.size))
                        output_dataset['wb_RO'][division_index, :] = np.reshape(RO, (1, RO.size))
                        output_dataset['wb_PRO'][division_index, :] = np.reshape(PRO, (1, PRO.size))
                        output_dataset['wb_L'][division_index, :] = np.reshape(L, (1, L.size))
                        output_dataset['wb_PL'][division_index, :] = np.reshape(PL, (1, PL.size))
                        output_dataset['pet'][division_index, :] = np.reshape(pet_time_series, (1, pet_time_series.size))
                        output_dataset.sync()
                    lock.release()
    
                logger.info('\tComputing SPI/SPEI/PNP for division index {0}'.format(division_index))

                # process the SPI and SPEI at month scales
                for months in scale_months:
                    
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
    
                    # create variable names which should correspond to the appropriate scaled index output variables
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
    '''

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        logger.info("Start time:    {0}".format(start_datetime, '%x'))

        # parse the command line arguments
        parser = argparse.ArgumentParser()
#         parser.add_argument("--input_file", 
#                             help="Input dataset file (NetCDF) containing temperature, precipitation, and soil values for PDSI, SPI, SPEI, and PNP computations", 
#                             required=True)
#         parser.add_argument("--precip_var_name", 
#                             help="Precipitation variable name used in the input NetCDF file", 
#                             required=True)
#         parser.add_argument("--temp_var_name", 
#                             help="Temperature variable name used in the input NetCDF file", 
#                             required=True)
#         parser.add_argument("--awc_var_name", 
#                             help="Available water capacity variable name used in the input NetCDF file", 
#                             required=False)
#         parser.add_argument("--pet_var_name", 
#                             help="PET variable name used in the input NetCDF file", 
#                             required=True)
#         parser.add_argument("--output_file",
#                             help=" Output file path and name",
#                             required=True)
#         parser.add_argument("--month_scales",
#                             help="Month scales over which the PNP, SPI, and SPEI values are to be computed",
#                             type=int,
#                             nargs = '*',
#                             choices=range(1, 73),
#                             required=True)
#         parser.add_argument("--calibration_start_year",
#                             help="Initial year of calibration period",
#                             type=int,
#                             choices=range(1870, start_datetime.year + 1),
#                             required=True)
#         parser.add_argument("--calibration_end_year",
#                             help="Final year of calibration period",
#                             type=int,
#                             choices=range(1870, start_datetime.year + 1),
#                             required=True)
        args = parser.parse_args()

        # FIXME hard-coded for development/debugging only -- REMOVE!
        args.input_file = 'C:/home/climdivs/20170505/nclimdiv_cmbmonthly_20170505.nc'
        args.output_file = 'C:/home/climdivs/20170505/nclimdiv_nidis_petfromcmb_20170505_debug_01.nc'
        args.temp_var_name = 'tavg'
        args.precip_var_name = 'prcp'
        args.pet_var_name = 'pe60'  # PET variable name used by CMB
        args.awc_var_name = 'awc'
        args.month_scales = [1, 2, 3, 6, 9, 12, 24]
        args.calibration_start_year = 1931
        args.calibration_end_year = 1990
        
        # initialize the NetCDFs to be used as output files for the Palmer and PET indices,
        # getting dictionaries of index names mapped to corresponding NetCDF files        
        initialize_netcdf(args.output_file, args.input_file, args.month_scales)
        
        # open the NetCDF files 
        with netCDF4.Dataset(args.input_file) as input_dataset:
             
            # get the initial year of the input dataset
            time_variable = input_dataset.variables['time']
            data_start_year = netCDF4.num2date(time_variable[0], time_variable.units).year
 
            # get the number of divisions in the input dataset(s)
            divisions_count = input_dataset.variables['division'].size
        
        # create a process Pool, with copies of the shared array going to each pooled/forked process
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(),
                                    initializer=init_process,
                                    initargs=(args.input_file,
                                              args.output_file,
                                              args.precip_var_name,
                                              args.pet_var_name,
                                              args.awc_var_name,
                                              args.month_scales,
                                              data_start_year,
                                              args.calibration_start_year,
                                              args.calibration_end_year))
 
        # map the divisions indices as an arguments iterable to the compute function
        result = pool.map_async(compute_and_write_division, range(divisions_count))
                  
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
    