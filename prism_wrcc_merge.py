import argparse
from datetime import datetime
import logging
import netCDF4
import netcdf_utils
import numpy as np
import os
from netCDF4 import Dataset, num2date
import urllib.request

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
def _initialize_dataset(file_path,
                        template_path):
    
    # open the output file, set its dimensions and variables, we'll return this object in an open state
    netcdf = Dataset(file_path, 'w')
    
    # open the template NetCDF, closed upon function completion
    with Dataset(template_path) as template_dataset:

        # copy the global attributes from the template
        netcdf.setncatts(template_dataset.__dict__)
    
        x_dim_name = 'longitude'
        y_dim_name = 'latitude'
        t_dim_name = 'day'
        x_size = template_dataset.variables[x_dim_name].size
        y_size = template_dataset.variables[y_dim_name].size
        t_size = template_dataset.variables[t_dim_name].size
            
        # create the time, x, and y dimensions
        netcdf.createDimension('time', None)  # use 'time' in the output although 'day' is the original time variable name
        netcdf.createDimension('lon', x_size)
        netcdf.createDimension('lat', y_size)
        
        # get the appropriate data types to use for the variables based on the values arrays
        time_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables[t_dim_name])
        x_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables[x_dim_name])
        y_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables[y_dim_name])
        data_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables['data'])
        
        # create the variables
        time_variable = netcdf.createVariable('time', time_dtype, ('time',))
        x_variable = netcdf.createVariable('lon', x_dtype, ('lon',))
        y_variable = netcdf.createVariable('lat', y_dtype, ('lat',))
        
        # set the variables' attributes
        time_variable.setncatts(template_dataset.variables[t_dim_name].__dict__)
        x_variable.setncatts(template_dataset.variables[x_dim_name].__dict__)
        y_variable.setncatts(template_dataset.variables[y_dim_name].__dict__)
        
        # set the coordinate variables' values
        x_variable[:] = template_dataset.variables[x_dim_name][:]
        y_variable[:] = template_dataset.variables[y_dim_name][:]
        
        # allocate an empty array for the times to be 12 times larger than the original monthly file
        time_variable[:] = np.full((t_size * 12), fill_value=np.NaN, dtype=time_dtype)
    
        # create placeholder variables for precipitation and temperature
        precip_variable = netcdf.createVariable('prcp', 
                                                data_dtype, 
                                                ('time', 'lat', 'lon',), 
                                                fill_value=np.NaN)
        precip_variable.units = 'millimeters'
        precip_variable.description = 'Accumulated precipitation'
        temp_variable = netcdf.createVariable('tavg', 
                                              data_dtype, 
                                              ('time', 'lat', 'lon',), 
                                              fill_value=np.NaN)
        temp_variable.units = 'Celsius'
        temp_variable.description = 'Mean temperature'
            
    return netcdf
    
#-----------------------------------------------------------------------------------------------------------------------
def merge_wrcc_prism(precip_file_base, 
                     temp_file_base,
                     output_file):
    
    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        logger.info("Start time:    {0}".format(start_datetime, '%x'))

        # open the first precip file as the template to use for output NetCDF initialization
        with _initialize_dataset(output_file,
                                 precip_file_base + '_1_PRISM.nc') as output_dataset:
              
            # loop over each calendar month, add values into the output dataset variables accordingly
            for month in range(1, 13):
                
                # we'll flip these flags if we download the associated files so we'll know to clean up once done
                cleanup_precip = False
                cleanup_temp = False

                # get the precipitation file if it's not on local disk
                precip_file = precip_file_base + '_{0}_PRISM.nc'.format(month)
                if not os.path.isfile(precip_file):
                    url = 'ftp://pubfiles.dri.edu/pub/mcevoy/WWDT_input/pon1_{0}_PRISM.nc'.format(month)
                    logger.info('Downloading from {0}'.format(url))
                    precip_file = urllib.request.urlretrieve(url)
                    logger.info('\tTemporary input data file: {0}'.format(precip_file))
                    cleanup_precip = True

                # get the temperature file if it's not on local disk
                temp_file = temp_file_base + '_{0}_PRISM.nc'.format(month)
                if not os.path.isfile(temp_file):
                    url = 'ftp://pubfiles.dri.edu/pub/mcevoy/WWDT_input/mdn1_{0}_PRISM.nc'.format(month)
                    logger.info('Downloading from {0}'.format(url))
                    temp_file = urllib.request.urlretrieve(url)
                    logger.info('\tTemporary input data file: {0}'.format(temp_file))
                    cleanup_temp = True

                # open the two input NetCDF files, closed automatically on completion of this loop steo                
                with Dataset(precip_file) as precip_dataset, \
                     Dataset(temp_file) as temp_dataset:

                    # make sure the times match up                          
                    times_precip = precip_dataset.variables['day'][:]
                    times_temp = temp_dataset.variables['day'][:]
                    if np.allclose(times_precip, times_temp):

                        logger.info('Assigning data for month: {0}'.format(month))
                                                
                        # add the times at every 12th time step (month) to correspond to the current calendar month
                        output_dataset.variables['time'][month - 1::12] = times_precip
                        
                        # assign values into the data variables at every 12th time step (month) to correspond to the current calendar month
                        output_dataset.variables['prcp'][month - 1::12] = precip_dataset.variables['data'][:]
                        output_dataset.variables['tavg'][month - 1::12] = temp_dataset.variables['data'][:]
        
                    else:
                        # the times didn't match, can't add a values array with incompatible dimensions
                        message = 'Incompatible time values found in temperature and precipitation files for month {0}'.format(month)
                        logger.error(message)
                        raise ValueError(message)

                # if we downloaded the files then remove them now        
                if cleanup_precip:
                    os.remove(precip_file)                  
                    logger.info('Removed temporary input data file: {0}'.format(precip_file))
                if cleanup_temp:
                    os.remove(temp_file)
                    logger.info('Removed temporary input data file: {0}'.format(temp_file))

        # report on the elapsed time
        end_datetime = datetime.now()
        logger.info("End time:      {}".format(end_datetime, '%x'))
        elapsed = end_datetime - start_datetime
        logger.info("Elapsed time:  {}".format(elapsed, '%x'))

    except Exception:
        logger.exception('Failed to complete', exc_info=True)
        raise
    
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    '''
    '''
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--precip_file_base", 
                        help="Precipitation dataset base file name to be used as input for merge", 
                        required=True)
    parser.add_argument("--temp_file_base", 
                        help="Temperature dataset base file name to be used as input for merge", 
                        required=True)
    parser.add_argument("--output_file",
                        help="Output file path and name",
                        required=True)
    args = parser.parse_args()
        
    merge_wrcc_prism(args.precip_file_base, args.temp_file_base, args.output_file)
