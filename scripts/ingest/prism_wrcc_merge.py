import argparse
from datetime import datetime
import logging
from netCDF4 import Dataset, num2date
import numpy as np
import os
import urllib

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
                        local_template_file):
    
    # open the output file, set its dimensions and variables, we'll return this object in an open state
    netcdf = Dataset(file_path, 'w')
    
    # open the template NetCDF, closed upon function completion
    with Dataset(local_template_file) as template_dataset:

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
        time_dtype = template_dataset.variables[t_dim_name].dtype
        x_dtype = template_dataset.variables[x_dim_name].dtype
        y_dtype = template_dataset.variables[y_dim_name].dtype
        
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
                
    return netcdf

#-----------------------------------------------------------------------------------------------------------------------
def _add_variable(dataset,
                  name,
                  dtype,
                  dims,
                  filler,
                  units,
                  description):
    
    # create placeholder variables for precipitation and temperature
    variable = dataset.createVariable(name, 
                                      dtype, 
                                      dims, 
                                      fill_value=filler)
    variable.units = units
    variable.description = description

#-----------------------------------------------------------------------------------------------------------------------
def _get_file(file_name,
              local_file):

    url = 'ftp://pubfiles.dri.edu/pub/mcevoy/WWDT_input/{0}'.format(file_name)
    logger.info('Downloading from %s', url)
    urllib.request.urlretrieve(url, local_file)
    logger.info('\tTemporary input data file: %s', local_file)

#-----------------------------------------------------------------------------------------------------------------------
def _merge_wrcc_prism(prism_inputs_dir,
                      prism_output_file,
                      name,
                      units,
                      description,
                      input_variable_name,
                      perform_cleanup=False):
    
    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        logger.info('Start time:    %s', start_datetime)

        # get the PRISM template file if it's not on local disk
        cleanup_template = False
        template_file_name = 'pon1_1_PRISM.nc'
        local_template_file = prism_inputs_dir + '/' + template_file_name
        if not os.path.isfile(local_template_file):
            _get_file(template_file_name, local_template_file)
            cleanup_template = perform_cleanup

#         # get the times for later use
#         with Dataset(local_template_file, 'r') as template_dataset:
#             times_template = template_dataset.variables['day'][:]

        with _initialize_dataset(prism_output_file, local_template_file) as output_dataset:
              
            # loop over each calendar month, add values into the output dataset variables accordingly
            for month in range(1, 13):
                
                # we'll flip this flag if we download the file so we'll know to clean up once done
                cleanup_file = False

                # get the PRISM input file if it's not on local disk
                file_name = '{0}_{1}_PRISM.nc'.format(input_variable_name, month)
                local_prism_file = prism_inputs_dir + '/' + file_name
                if not os.path.isfile(local_prism_file):
                    _get_file(file_name, local_prism_file)
                    cleanup_file = perform_cleanup

                # open the input NetCDF file, closed automatically on completion of this loop steo                
                with Dataset(local_prism_file) as prism_dataset:

                    times_prism = prism_dataset.variables['day'][:]

#                         # make sure the times match up 
#                         #TODO do this for lats/lons as well?
#                         if not np.allclose(times_template, times_prism):
#                             # the times didn't match, can't add a values array with incompatible dimensions
#                             message = 'Incompatible time values found in input data files for month {0}'.format(month)
#                             logger.error(message)
#                             raise ValueError(message)

                    # on the first month pass we'll create the corresponding variable in the output NetCDF
                    if month == 1:

                        # get the data type of the input, use this for the output's variable as well (no conversion)
                        variable_input = prism_dataset.variables['data']
                        dtype = variable_input.dtype
                        dims = variable_input.dimensions
                        filler = variable_input._FillValue

                        new_var_dims = []
                        for _, dim in enumerate(dims):
                            if dim == 'day':
                                new_var_dims.append('time')
                            elif dim == 'latitude':
                                new_var_dims.append('lat')
                            elif dim == 'longitude':
                                new_var_dims.append('lon')
                            else:
                                raise ValueError('Incompatible dimension: {0}'.format(dim))
                            
                        # add the variable into the output NetCDF
                        _add_variable(output_dataset, name, dtype, tuple(new_var_dims), filler, units, description)

                    logger.info('Assigning data for month: %s', month)
                                            
                    # add the times at every 12th time step (month) to correspond to the current calendar month
                    output_dataset.variables['time'][month - 1::12] = times_prism
                    
                    # assign values into the data variables at every 12th time step (month) to correspond to the current calendar month
                    output_dataset.variables[name][month - 1::12] = prism_dataset.variables['data'][:]

                # if we downloaded the file then remove it now        
                if cleanup_file:
                    os.remove(local_prism_file)                  
                    logger.info('Removed temporary input data file: %s', local_prism_file)

        # if we downloaded the template file then remove it now        
        if cleanup_template:
            os.remove(local_template_file)                  
            logger.info('Removed temporary template file: %s', local_template_file)

        # report on the elapsed time
        end_datetime = datetime.now()
        logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        logger.info("Elapsed time:  %s", elapsed)

    except Exception:
        logger.exception('Failed to complete', exc_info=True)
        raise
    
#-----------------------------------------------------------------------------------------------------------------------
def ingest_prism_from_wrcc_public_ftp(prism_inputs_dir,
                                      output_file_base):

    # dictionary of WRCC variable names to output PRISM dataset variable names and attributes
#     # DEBUG ONLY -- REMOVE
#     var_names_wrcc_prism = {'scpdsi': ['scpdsi', 'unitless', 'Self-calibrated Palmer Drought Severity Index']}
    var_names_wrcc_prism = {'pon1': ['prcp', 'millimeters', 'Accumulated precipitation'],
                            'mdn1': ['tavg', 'Celsius', 'Mean temperature'],
                            'pzi': ['zindex', 'unitless', 'Palmer Z-Index'], 
                            'pdsi': ['pdsi', 'unitless', 'Palmer Drought Severity Index'],
                            'scpdsi': ['scpdsi', 'unitless', 'Self-calibrated Palmer Drought Severity Index']}
    for variable_wrcc, variable_prism in var_names_wrcc_prism.items():

        output_file = output_file_base + '_{0}.nc'.format(variable_prism[0])
        
        # call the function that performs the merge
        _merge_wrcc_prism(prism_inputs_dir, 
                          output_file, 
                          variable_prism[0], 
                          variable_prism[1], 
                          variable_prism[2],
                          variable_wrcc)

        print('File for {0}:  {1}'.format(variable_prism[2], output_file))

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    '''
    Main function called when this module is called from a python command, i.e.
    
    python <this_code.py> --prism_inputs_dir <dir_with_prism_input_files> --output_file <out_file>

    :param args
    '''
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--prism_inputs_dir", 
                        help="PRISM input files from WRCC located in this directory", 
                        required=True)
    parser.add_argument("--output_file_base",
                        help="Output file path/name",
                        required=True)
    args = parser.parse_args()
    
    # perform an ingest from publicly available PRISM files produced by WWDT/WRCC/DRI provided by Dan McEvoy
    ingest_prism_from_wrcc_public_ftp(args.prism_inputs_dir,
                                      args.output_file_base)
