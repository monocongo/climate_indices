import argparse
from datetime import datetime
import indices
import logging
import multiprocessing
#import netCDF4
import netcdf_utils
import numpy as np
import os
import subprocess
import sys
from netCDF4 import Dataset, num2date

# set up a basic, global logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

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
def init_process_spi_spei_pnp(worker_precip_netcdf,
                              worker_pet_netcdf,
                              worker_precip_var_name,
                              worker_spi_gamma_netcdf,
                              worker_spi_pearson_netcdf,
                              worker_spei_gamma_netcdf,
                              worker_spei_pearson_netcdf,
                              worker_pnp_netcdf,
                              worker_scale_months,
                              worker_data_start_year,
                              worker_calibration_start_year, 
                              worker_calibration_end_year):
    
    # put the arguments into the global namespace
    global precip_netcdf, \
           pet_netcdf, \
           precip_var_name, \
           spi_gamma_netcdf, \
           spi_pearson_netcdf, \
           spei_gamma_netcdf, \
           spei_pearson_netcdf, \
           pnp_netcdf, \
           scale_months, \
           data_start_year, \
           calibration_start_year, \
           calibration_end_year
           
    precip_netcdf = worker_precip_netcdf
    pet_netcdf = worker_pet_netcdf
    precip_var_name = worker_precip_var_name
    spi_gamma_netcdf = worker_spi_gamma_netcdf
    spi_pearson_netcdf = worker_spi_pearson_netcdf
    spei_gamma_netcdf = worker_spei_gamma_netcdf
    spei_pearson_netcdf = worker_spei_pearson_netcdf
    pnp_netcdf = worker_pnp_netcdf
    scale_months = worker_scale_months
    data_start_year = worker_data_start_year
    calibration_start_year = worker_calibration_start_year
    calibration_end_year = worker_calibration_end_year
    
#-----------------------------------------------------------------------------------------------------------------------
def process_latitude_spi_spei_pnp(lat_index):
    '''
    '''
    
    logger.info('Computing SPI, SPEI, and PNP for latitude index {0}'.format(lat_index))
    
    # open the input NetCDFs
    with Dataset(precip_netcdf) as precip_dataset, \
         Dataset(pet_netcdf) as pet_dataset:

        valid_min = -3.09
        valid_max = 3.09
                                                  
        # read the latitude slice of input precipitation and PET values 
        precip_lat_slice = precip_dataset[precip_var_name][:, lat_index, :]   # assuming (time, lat, lon) orientation
        pet_lat_slice = pet_dataset['pet'][:, lat_index, :]   # assuming (time, lat, lon) orientation
        
        # allocate arrays to contain a latitude slice of Palmer values
        lon_size = precip_dataset['lon'].size
        time_size = precip_dataset['time'].size
        lat_slice_shape = (time_size, 1, lon_size)

        # compute SPI/Gamma across all longitudes of the latitude slice
        spi_gamma_lat_slice = np.apply_along_axis(indices.spi_gamma, 
                                                  0, 
                                                  precip_lat_slice, 
                                                  scale_months)
 
        # compute SPI/Pearson across all longitudes of the latitude slice
        spi_pearson_lat_slice = np.apply_along_axis(indices.spi_pearson, 
                                                    0, 
                                                    precip_lat_slice, 
                                                    scale_months,
                                                    data_start_year,
                                                    calibration_start_year, 
                                                    calibration_end_year)        
         
        # compute PNP across all longitudes of the latitude slice
        pnp_lat_slice = np.apply_along_axis(indices.percentage_of_normal, 
                                            0, 
                                            precip_lat_slice, 
                                            scale_months,
                                            data_start_year,
                                            calibration_start_year, 
                                            calibration_end_year)        
        
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
                spei_gamma_lat_slice[:, 0, lon_index] = indices.spei_gamma(scale_months,
                                                                           precip_time_series,
                                                                           pet_mm=pet_time_series)

                # compute SPEI/Pearson
                spei_pearson_lat_slice[:, 0, lon_index] = indices.spei_pearson(scale_months,
                                                                               data_start_year,
                                                                               precip_time_series,
                                                                               pet_mm=pet_time_series,
                                                                               calibration_year_initial=calibration_start_year,
                                                                               calibration_year_final=calibration_end_year)
                 
        # use the same variable name within both Gamma and Pearson NetCDFs
        #TODO update this for separate 'spi_gamma_<months>' and 'spi_pearson_<months>' instead
        spi_gamma_variable_name = 'spi_gamma_' + str(scale_months).zfill(2)
        spi_pearson_variable_name = 'spi_pearson_' + str(scale_months).zfill(2)
        spei_gamma_variable_name = 'spei_gamma_' + str(scale_months).zfill(2)
        spei_pearson_variable_name = 'spei_pearson_' + str(scale_months).zfill(2)
        pnp_variable_name = 'pnp_' + str(scale_months).zfill(2)
        
        # open the existing SPI/Gamma NetCDF file for writing, copy the latitude slice into the SPI variable at the indexed latitude position 
        spi_gamma_lock.acquire()
        spi_gamma_dataset = Dataset(spi_gamma_netcdf, mode='a')
        spi_gamma_dataset[spi_gamma_variable_name][:, lat_index, :] = spi_gamma_lat_slice
        spi_gamma_dataset.sync()
        spi_gamma_dataset.close()
        spi_gamma_lock.release()
 
        # open the existing SPI/Pearson NetCDF file for writing, copy the latitude slice into the SPI variable at the indexed latitude position 
        spi_pearson_lock.acquire()
        spi_pearson_dataset = Dataset(spi_pearson_netcdf, mode='a')
        spi_pearson_dataset[spi_pearson_variable_name][:, lat_index, :] = spi_pearson_lat_slice
        spi_pearson_dataset.sync()
        spi_pearson_dataset.close()
        spi_pearson_lock.release()

        # open the existing SPEI/Gamma NetCDF file for writing, copy the latitude slice into the SPEI variable at the indexed latitude position 
        spei_gamma_lock.acquire()
        spei_gamma_dataset = Dataset(spei_gamma_netcdf, mode='a')
        spei_gamma_dataset[spei_gamma_variable_name][:, lat_index, :] = spei_gamma_lat_slice
        spei_gamma_dataset.sync()
        spei_gamma_dataset.close()
        spei_gamma_lock.release()
 
        # open the existing SPEI/Pearson NetCDF file for writing, copy the latitude slice into the SPEI variable at the indexed latitude position 
        spei_pearson_lock.acquire()
        spei_pearson_dataset = Dataset(spei_pearson_netcdf, mode='a')
        spei_pearson_dataset[spei_pearson_variable_name][:, lat_index, :] = spei_pearson_lat_slice
        spei_pearson_dataset.sync()
        spei_pearson_dataset.close()
        spei_pearson_lock.release()

        # open the existing PNP NetCDF file for writing, copy the latitude slice into the PNP variable at the indexed latitude position 
        pnp_lock.acquire()
        pnp_dataset = Dataset(pnp_netcdf, mode='a')
        pnp_dataset[pnp_variable_name][:, lat_index, :] = pnp_lat_slice
        pnp_dataset.sync()
        pnp_dataset.close()
        pnp_lock.release()

        #TODO compute SPEI from precipitation and PET
         
#-----------------------------------------------------------------------------------------------------------------------
def init_palmer_process(worker_temp_netcdf,
                        worker_precip_netcdf,
                        worker_awc_netcdf,
                        worker_temp_var_name,
                        worker_precip_var_name,
                        worker_awc_var_name,
                        worker_pet_netcdf, 
                        worker_pdsi_netcdf, 
                        worker_phdi_netcdf, 
                        worker_zindex_netcdf, 
                        worker_scpdsi_netcdf, 
                        worker_pmdi_netcdf, 
                        worker_initial_data_year,
                        worker_calibration_start_year,
                        worker_calibration_end_year):
    '''
    This function is called by each process of a process Pool. It is used to pass values 
    into the global namespace which will facilitate their use by worker subprocesses.
     
    :param worker_temp_netcdf:
    :param worker_precip_netcdf: 
    :param worker_awc_netcdf: 
    :param worker_temp_var_name:
    :param worker_precip_var_name:
    :param worker_awc_var_name: 
    :param worker_pet_netcdf: 
    :param worker_pdsi_netcdf: 
    :param worker_phdi_netcdf: 
    :param worker_zindex_netcdf: 
    :param worker_scpdsi_netcdf: 
    :param worker_pmdi_netcdf: 
    :param worker_initial_data_year: 
    :param worker_calibration_start_year: 
    :param worker_calibration_end_year: 
    '''
     
    # put the arguments into the global namespace
    global temp_netcdf, \
           precip_netcdf, \
           awc_netcdf, \
           precip_var_name, \
           temp_var_name, \
           awc_var_name, \
           pet_netcdf, \
           pdsi_netcdf, \
           phdi_netcdf, \
           zindex_netcdf, \
           scpdsi_netcdf, \
           pmdi_netcdf, \
           initial_data_year, \
           calibration_start_year, \
           calibration_end_year
           
    temp_netcdf = worker_temp_netcdf
    precip_netcdf = worker_precip_netcdf
    awc_netcdf = worker_awc_netcdf
    temp_var_name = worker_temp_var_name
    precip_var_name = worker_precip_var_name
    awc_var_name = worker_awc_var_name
    pet_netcdf = worker_pet_netcdf
    pdsi_netcdf = worker_pdsi_netcdf
    phdi_netcdf = worker_phdi_netcdf
    zindex_netcdf = worker_zindex_netcdf
    scpdsi_netcdf = worker_scpdsi_netcdf
    pmdi_netcdf = worker_pmdi_netcdf
    initial_data_year = worker_initial_data_year
    calibration_start_year = worker_calibration_start_year
    calibration_end_year = worker_calibration_end_year

#-----------------------------------------------------------------------------------------------------------------------
def process_latitude_palmer(lat_index):
    '''
    '''
    
    logger.info('Computing PET and Palmers for latitude index {0}'.format(lat_index))
    
    # open the input NetCDFs
    with Dataset(precip_netcdf) as precip_dataset, \
         Dataset(temp_netcdf) as temp_dataset, \
         Dataset(awc_netcdf) as awc_dataset:
    
        # read the latitude slice of input temperature values 
        temperature_lat_slice = temp_dataset[temp_var_name][:, lat_index, :]    # assuming (time, lat, lon) orientation
        
        # get the actual latitude value (assumed to be in degrees north) for the latitude slice specified by the index
        latitude_degrees_north = temp_dataset['lat'][lat_index]
        
        # use the numpyapply_along_axis() function for computing indices such as PET that take a single time series 
        # array as input (i.e. each longitude's time series is the initial 1-D array argument to the function we'll apply)
        
        
        # compute PET across all longitudes of the latitude slice
        pet_lat_slice = np.apply_along_axis(indices.pet, 
                                            0, 
                                            temperature_lat_slice, 
                                            latitude_degrees=latitude_degrees_north, 
                                            data_start_year=initial_data_year)
    
        # open the existing PET NetCDF file for writing, copy the latitude slice into the PET variable at the indexed latitude position 
        pet_lock.acquire()
        pet_dataset = Dataset(pet_netcdf, mode='a')
        pet_dataset['pet'][:, lat_index, :] = pet_lat_slice
        pet_dataset.sync()
        pet_dataset.close()
        pet_lock.release()

        # read the latitude slice of input precipitation and available water capacity values 
        precip_lat_slice = precip_dataset[precip_var_name][:, lat_index, :]   # assuming (time, lat, lon) orientation
        awc_lat_slice = awc_dataset[awc_var_name][lat_index, :]             # assuming (lat, lon) orientation
        
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
            
            # get the time series values for this longitude
            precip_time_series = precip_lat_slice[:, lon_index]
            pet_time_series = pet_lat_slice[:, lon_index]
            awc = awc_lat_slice[lon_index]
            
            # compute Palmer indices if we have valid inputs
            if (not np.all(np.isnan(precip_time_series))) and \
               (not np.all(np.isnan(pet_time_series))) and \
               (not np.isnan(awc)):
                    
#                 logger.info('     Longitude index {0}'.format(lon_index))

                # put precipitation into inches if not already
                mm_to_inches_multiplier = 0.0393701
                possible_mm_units = ['millimeters', 'millimeter', 'mm']
                if precip_dataset[precip_var_name].units in possible_mm_units:
                    precip_time_series = precip_time_series * mm_to_inches_multiplier

                # PET is in mm, convert to inches
                pet_time_series = pet_time_series * mm_to_inches_multiplier

#                 logger.info('     Computing for longitude index {0}'.format(lon_index))

                # compute Palmer indices
                palmer_values = indices.scpdsi(precip_time_series,
                                               pet_time_series,
                                               awc,
                                               initial_data_year,
                                               calibration_start_year,
                                               calibration_end_year)
    
                scpdsi_lat_slice[:, 0, lon_index] = palmer_values[0]
                pdsi_lat_slice[:, 0, lon_index] = palmer_values[1]
                phdi_lat_slice[:, 0, lon_index] = palmer_values[2]
                pmdi_lat_slice[:, 0, lon_index] = palmer_values[3]
                zindex_lat_slice[:, 0, lon_index] = palmer_values[4]
        
        # open the existing PDSI NetCDF file for writing, copy the latitude slice into the PET variable at the indexed latitude position 
        pdsi_lock.acquire()
        pdsi_dataset = Dataset(pdsi_netcdf, mode='a')
        pdsi_dataset['pdsi'][:, lat_index, :] = pdsi_lat_slice
        pdsi_dataset.sync()
        pdsi_dataset.close()
        pdsi_lock.release()

        # open the existing PHDI NetCDF file for writing, copy the latitude slice into the PET variable at the indexed latitude position 
        phdi_lock.acquire()
        phdi_dataset = Dataset(phdi_netcdf, mode='a')
        phdi_dataset['phdi'][:, lat_index, :] = phdi_lat_slice
        phdi_dataset.sync()
        phdi_dataset.close()
        phdi_lock.release()

        # open the existing Z-Index NetCDF file for writing, copy the latitude slice into the PET variable at the indexed latitude position 
        zindex_lock.acquire()
        zindex_dataset = Dataset(zindex_netcdf, mode='a')
        zindex_dataset['zindex'][:, lat_index, :] = zindex_lat_slice
        zindex_dataset.sync()
        zindex_dataset.close()
        zindex_lock.release()

        # open the existing SCPDSI NetCDF file for writing, copy the latitude slice into the PET variable at the indexed latitude position 
        scpdsi_lock.acquire()
        scpdsi_dataset = Dataset(scpdsi_netcdf, mode='a')
        scpdsi_dataset['scpdsi'][:, lat_index, :] = scpdsi_lat_slice
        scpdsi_dataset.sync()
        scpdsi_dataset.close()
        scpdsi_lock.release()

        # open the existing PHDI NetCDF file for writing, copy the latitude slice into the PET variable at the indexed latitude position 
        pmdi_lock.acquire()
        pmdi_dataset = Dataset(pmdi_netcdf, mode='a')
        pmdi_dataset['pmdi'][:, lat_index, :] = pmdi_lat_slice
        pmdi_dataset.sync()
        pmdi_dataset.close()
        pmdi_lock.release()

#-----------------------------------------------------------------------------------------------------------------------
def initialize_netcdf(file_path,
                      template_netcdf,
                      variable_name,
                      variable_long_name,
                      valid_min,
                      valid_max,
                      variable_units=None,
                      fill_value=np.float32(np.NaN)):
    '''
    This function is used to initialize and return a netCDF4.Dataset object.
    
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

    with Dataset(template_netcdf, 'r') as template_dataset:
 
        # get the template's dimension sizes
        lat_size = template_dataset.variables['lat'].size
        lon_size = template_dataset.variables['lon'].size
    
        # make a basic set of variable attributes
        variable_attributes = {'valid_min' : valid_min,
                               'valid_max' : valid_max,
                               'long_name' : variable_long_name}
        if variable_units != None:
            variable_attributes['units'] = variable_units
            
        # open the dataset as a NetCDF in write mode
        dataset = Dataset(file_path, 'w')
        
        # copy the global attributes from the input
        # TODO/FIXME add/modify global attributes to correspond with the actual dataset
        dataset.setncatts(template_dataset.__dict__)
        
        # create the time, x, and y dimensions
        dataset.createDimension('time', None)
        dataset.createDimension('lat', lat_size)
        dataset.createDimension('lon', lon_size)
    
        # get the appropriate data types to use for the variables
        time_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables['time'])
        lat_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables['lat'])
        lon_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables['lon'])
        data_dtype = netcdf_utils.find_netcdf_datatype(fill_value)
    
        # create the variables
        time_variable = dataset.createVariable('time', time_dtype, ('time',))
        y_variable = dataset.createVariable('lat', lat_dtype, ('lat',))
        x_variable = dataset.createVariable('lon', lon_dtype, ('lon',))
        data_variable = dataset.createVariable(variable_name,
                                               data_dtype,
                                               ('time', 'lat', 'lon',),
                                               fill_value=fill_value, 
                                               zlib=False)
    
        # set the variables' attributes
        time_variable.setncatts(template_dataset.variables['time'].__dict__)
        y_variable.setncatts(template_dataset.variables['lat'].__dict__)
        x_variable.setncatts(template_dataset.variables['lon'].__dict__)
        data_variable.setncatts(variable_attributes)
    
        # set the coordinate variables' values
        time_variable[:] = template_dataset.variables['time'][:]
        y_variable[:] = template_dataset.variables['lat'][:]
        x_variable[:] = template_dataset.variables['lon'][:]

        # close the NetCDF
        dataset.close()

#-----------------------------------------------------------------------------------------------------------------------
def validate_compatibility(precip_dataset, 
                           temp_dataset,
                           awc_dataset):

    # get the time, lat, and lon variables from the three datasets we want to validate against each other
    precip_time = precip_dataset.variables['time']
    precip_lat = precip_dataset.variables['lat']
    precip_lon = precip_dataset.variables['lon']
    temp_time = temp_dataset.variables['time']
    temp_lat = temp_dataset.variables['lat']
    temp_lon = temp_dataset.variables['lon']
    awc_lat = awc_dataset.variables['lat']
    awc_lon = awc_dataset.variables['lon']
    
    # dataset names to be used in error messages
    precip_dataset_name = 'precipitation'
    temp_dataset_name = 'temperature'
    awc_dataset_name = 'available water capacity'

    # make sure that the datasets match in terms of coordinate variables
    if not np.allclose(precip_time[:], temp_time[:]):
        message = 'Mismatch of the time dimension between the {0} and {1} datasets'.format(precip_dataset_name, temp_dataset_name)
        logger.error(message)
        raise ValueError(message)
    if not np.allclose(precip_lat[:], temp_lat[:]):
        message = 'Mismatch of the lat dimension between the {0} and {1} datasets'.format(precip_dataset_name, temp_dataset_name)
        logger.error(message)
        raise ValueError(message)
    if not np.allclose(precip_lat[:], awc_lat[:], atol=1e-05, equal_nan=True):
        message = 'Mismatch of the lat dimension between the {0} and {1} datasets'.format(precip_dataset_name, awc_dataset_name)
        logger.error(message)
        raise ValueError(message)
    if not np.allclose(precip_lon[:], temp_lon[:]):
        message = 'Mismatch of the lon dimension between the {0} and {1} datasets'.format(precip_dataset_name, temp_dataset_name)
        logger.error(message)
        raise ValueError(message)
    if not np.allclose(precip_lon[:], awc_lon[:], atol=1e-05, equal_nan=True):
        message = 'Mismatch of the lon dimension between the {0} and {1} datasets'.format(precip_dataset_name, awc_dataset_name)
        logger.error(message)
        raise ValueError(message)

#-----------------------------------------------------------------------------------------------------------------------
def initialize_unscaled_netcdfs(base_file_path,
                                template_netcdf):
    
    pet_netcdf = base_file_path + '_pet.nc'
    pdsi_netcdf = base_file_path + '_pdsi.nc'
    phdi_netcdf = base_file_path + '_phdi.nc'
    zindex_netcdf = base_file_path + '_zindex.nc'
    scpdsi_netcdf = base_file_path + '_scpdsi.nc'
    pmdi_netcdf = base_file_path + '_pmdi.nc'
    valid_min = -10.0
    valid_max = 10.0
    
    initialize_netcdf(pet_netcdf,
                      template_netcdf,
                      'pet',
                      'Potential Evapotranspiration (PET), from Thornthwaite\'s equation',
                      0.0,
                      2000.0,
                      'millimeter')
    initialize_netcdf(pdsi_netcdf,
                      template_netcdf,
                      'pdsi',
                      'Palmer Drought Severity Index (PDSI)',
                      valid_min,
                      valid_max)
    initialize_netcdf(phdi_netcdf,
                      template_netcdf,
                      'phdi',
                      'Palmer Hydrological Drought Index (PHDI)',
                      valid_min,
                      valid_max)
    initialize_netcdf(zindex_netcdf,
                      template_netcdf,
                      'zindex',
                      'Palmer Z-Index',
                      valid_min,
                      valid_max)
    initialize_netcdf(scpdsi_netcdf,
                      template_netcdf,
                      'scpdsi',
                      'Self-calibrated Palmer Drought Severity Index (scPDSI)',
                      valid_min,
                      valid_max)
    initialize_netcdf(pmdi_netcdf,
                      template_netcdf,
                      'pmdi',
                      'Palmer Modified Drought Index (PMDI)',
                      valid_min,
                      valid_max)

    return {'pet': pet_netcdf,
            'pdsi': pdsi_netcdf,
            'phdi': phdi_netcdf,
            'zindex': zindex_netcdf,
            'pmdi': pmdi_netcdf,
            'scpdsi': scpdsi_netcdf}
    
#-----------------------------------------------------------------------------------------------------------------------
def initialize_scaled_netcdfs(base_file_path, 
                              scale_months, 
                              template_netcdf):
    
    # dictionary of index types to the NetCDF dataset files corresponding to the base index names and 
    # month scales (this is the object we'll build and return from this function)
    scaled_netcdfs = {}
    
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
        variable_name = index + '_{0}'.format(str(scale_months).zfill(2))

        # create the NetCDF file path from the 
        netcdf_file = base_file_path + '_' + variable_name + '.nc'
        
        # initialize the output NetCDF dataset
        initialize_netcdf(netcdf_file, 
                          template_netcdf,
                          variable_name,
                          long_name.format(scale_months),
                          valid_min,
                          valid_max)
    
        # add the months scale index's NetCDF to the dictionary for the current index
        scaled_netcdfs[index] = netcdf_file
        
    return scaled_netcdfs

#-----------------------------------------------------------------------------------------------------------------------
def construct_nco_command(netcdf_operator):
    '''
    This function constructs a NCO command appropriate to the platform where the code is running.
    
    :param netcdf_operator: the NCO command (eg. ncks, ncatted, etc.) to be called
    :return: executable command including full path, including platform-specific path separators
    :rtype: string   
    '''
    
    #TODO replace the hard-coded paths below with a function argument, the value of which is pulled from a command line option
    # set the NCO executable path appropriate to the current platform
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
if __name__ == '__main__':

    '''
    '''

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        logger.info("Start time:    {0}".format(start_datetime, '%x'))

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
        parser.add_argument("--destination_dir",
                            help="Destination directory where output indices files will be written",
                            required=True)
        
        args = parser.parse_args()

        # the number of worker processes we'll have in our process pool
        number_of_workers = multiprocessing.cpu_count()
        
        # initialize the NetCDFs to be used as output files for the Palmer and PET indices,
        # getting dictionaries of index names mapped to corresponding NetCDF files
        unscaled_netcdfs = initialize_unscaled_netcdfs(args.output_file_base, args.precip_file)
            
        # open the input NetCDF files for compatibility validation and to get the data's time range 
        with Dataset(args.precip_file) as precip_dataset, \
             Dataset(args.temp_file) as temp_dataset, \
             Dataset(args.awc_file) as awc_dataset:
              
            # make sure the datasets are compatible dimensionally
            validate_compatibility(precip_dataset, 
                                   temp_dataset, 
                                   awc_dataset)
              
            # get the initial year of the input dataset(s)
            time_variable = precip_dataset.variables['time']
            data_start_year = num2date(time_variable[0], time_variable.units).year
  
            # get the number of latitudes in the input dataset(s)
            lat_size = precip_dataset.variables['lat'].size
              
        # create a process Pool for worker processes to compute PET and Palmer indices, passing arguments to an initializing function
        pool = multiprocessing.Pool(processes=number_of_workers,
                                    initializer=init_palmer_process,
                                    initargs=(args.temp_file,
                                              args.precip_file,
                                              args.awc_file,
                                              args.temp_var_name,
                                              args.precip_var_name,
                                              args.awc_var_name,
                                              unscaled_netcdfs['pet'],
                                              unscaled_netcdfs['pdsi'],
                                              unscaled_netcdfs['phdi'],
                                              unscaled_netcdfs['zindex'],
                                              unscaled_netcdfs['scpdsi'],
                                              unscaled_netcdfs['pmdi'],
                                              data_start_year,
                                              args.calibration_start_year,
                                              args.calibration_end_year))
        
        # map the latitude indices as an arguments iterable to the compute function
        result = pool.map_async(process_latitude_palmer, range(lat_size))
                 
        # get the exception(s) thrown, if any
        result.get()
                 
        # close the pool and wait on all processes to finish
        pool.close()
        pool.join()
          
#         input_output_netcdfs = []
#         for index in ['pdsi', 'phdi', 'scpdsi', 'zindex']:
#             
#             # convert the Palmer files to compressed NetCDF4 and move to the destination directory
#             indicator_tuple = (unscaled_netcdfs[index], os.sep.join([args.destination_dir, index, unscaled_netcdfs[index]]))
#             input_output_netcdfs.append(indicator_tuple)
# 
#         pool = multiprocessing.Pool(processes=number_of_workers)
#             
#         # create an arguments iterable containing the input and output NetCDFs, map it to the convert function
#         result = pool.map_async(convert_and_move_netcdf, input_output_netcdfs)
#               
#         # get the exception(s) thrown, if any
#         result.get()
#               
#         # close the pool and wait on all processes to finish
#         pool.close()
#         pool.join()
        
#         # DEBUG ONLY -- REMOVE
#         debug_pet_file = args.output_file_base + '_pet.nc'
        
        # compute the scaled indices (PNP, SPI, and SPEI)
        for scale_months in args.month_scales:
 
            # initialize the output NetCDFs for SPI gamma and Pearson for the month scale
            scaled_netcdfs = initialize_scaled_netcdfs(args.output_file_base, scale_months, args.precip_file)
     
            # create a process Pool, initialize the global namespace to facilitate multiprocessing
            pool = multiprocessing.Pool(processes=number_of_workers,
                                        initializer=init_process_spi_spei_pnp,
                                        initargs=(args.precip_file,
                                                  unscaled_netcdfs['pet'],
                                                  args.precip_var_name,
                                                  scaled_netcdfs['spi_gamma'],
                                                  scaled_netcdfs['spi_pearson'],
                                                  scaled_netcdfs['spei_gamma'],
                                                  scaled_netcdfs['spei_pearson'],
                                                  scaled_netcdfs['pnp'],
                                                  scale_months,
                                                  data_start_year,
                                                  args.calibration_start_year,
                                                  args.calibration_end_year))
 
            # map the latitude indices as an arguments iterable to the compute function
            result = pool.map_async(process_latitude_spi_spei_pnp, range(lat_size))
              
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
#             result = pool.map_async(convert_and_move_netcdf, input_output_netcdfs)
#                 
#             # get the exception(s) thrown, if any
#             result.get()
#                 
#             # close the pool and wait on all processes to finish
#             pool.close()
#             pool.join()
         
#         # convert the PET file to compressed NetCDF4 and move into the destination directory
#         convert_and_move_netcdf((unscaled_netcdfs['pet'], '/nidis/test/nclimgrid/pet/' + unscaled_netcdfs['pet']))
         
        # report on the elapsed time
        end_datetime = datetime.now()
        logger.info("End time:      {}".format(end_datetime, '%x'))
        elapsed = end_datetime - start_datetime
        logger.info("Elapsed time:  {}".format(elapsed, '%x'))

    except Exception as ex:
        logger.exception('Failed to complete', exc_info=True)
        raise