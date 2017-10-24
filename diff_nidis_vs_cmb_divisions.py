import compute
import datetime
import logging
import netCDF4
import netcdf_utils
import numpy as np
import sys
import utils

#-----------------------------------------------------------------------------------------------------------------------
# set up matplotlib to use the Agg backend, in order to remove any dependencies on an X server
import matplotlib
import argparse
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
def initialize_netcdf(new_netcdf,
                      template_netcdf,
                      vars=None):
    '''
    This function is used to initialize and return a netCDF4.Dataset object.
    
    :param new_netcdf: the file path/name of the new NetCDF Dataset object to be created and returned by this function
    :param template_netcdf: an existing/open NetCDF Dataset object which will be used as a template for the Dataset
                            that will be created and returned by this function
    :param vars: if present this is a dictionary of variable names (keys) to variable attributes/data of 
                 the original/initial variables to be loaded into the NetCDF 
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

        # create a variable for each variable listed in the dictionary of variable names to variable attributes/data
        if vars is not None:
            for variable_name, variable in vars.items():
                
                variable_attributes = variable[0]
                var_data = variable[1]

                # create variables with scale month
                data_variable = new_dataset.createVariable(variable_name,
                                                           data_dtype,
                                                           ('division', 'time',),
                                                           fill_value=fill_value, 
                                                           zlib=False)
                data_variable.setncatts(variable_attributes)
                data_variable[:] = var_data
        
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--nidis_file", 
                        help="NIDIS input dataset file (NetCDF) containing values for the named variable", 
                        required=True)
    parser.add_argument("--cmb_file", 
                        help="CMB input dataset file (NetCDF) containing values for the named variable", 
                        required=True)
    parser.add_argument("--output_file",
                        help=" Output file path and name",
                        required=True)
    parser.add_argument("--bins", 
                        help="Number of bins to use for histogram plots", 
                        required=True)
    parser.add_argument("--range_min", 
                        help="Minimum value of the range to use for histogram plots", 
                        required=True)
    parser.add_argument("--range_max", 
                        help="Maximum value of the range to use for histogram plots", 
                        required=True)
    parser.add_argument("--output_dir", 
                        help="Output directory for monthly differences line graph plots", 
                        required=True)
    args = parser.parse_args()

    # get command line arguments
    netcdf_file_CMB = args.cmb_file
    netcdf_file_NIDIS = args.nidis_file
    netcdf_file_OUT = args.output_file
    number_of_bins = int(args.bins)
    range_upper = int(args.range_max)
    range_lower = int(args.range_min)

    divisions_dim_name = 'division'

    # open the NetCDF files
    with netCDF4.Dataset(netcdf_file_CMB) as dataset_CMB, \
         netCDF4.Dataset(netcdf_file_NIDIS) as dataset_NIDIS, \
         netcdf_utils.initialize_dataset_climdivs(netcdf_file_OUT,
                                                  dataset_CMB,
                                                  divisions_dim_name) as dataset_OUT:

        # variable names for variables to diff from the two datasets
        vars = {'PDSI': ('pdsi.index', 'pdsi'),
                'PHDI': ('phdi.index', 'phdi'),
                'PMDI': ('pmdi.index', 'pmdi'),
                'Z-Index': ('z.index', 'zindex')}
        for var_name, values in vars.items():
                    
            # TODO validate that the two variables exist, have compatible dimensions/units, etc., all of which is assumed below

            logger.info('Computing differences on variable {0}'.format(var_name))
            
            # allocate an array for the differences for this variable
            diffs = {}
            
            size = dataset_CMB.variables['division'][:].size
            
            for division_index in range(dataset_CMB.variables['division'][:].size):
             
                print('Computing diffs for division {0}'.format(division_index))
                
                # get the variable values for the month, mask the NaNs (data assumed to be in (division, time) dimension order)
                data_CMB = np.ma.masked_invalid(dataset_CMB.variables[values[0]][division_index, :], copy=False)
                data_NIDIS = np.ma.masked_invalid(dataset_NIDIS.variables[values[1]][division_index, :], copy=False)
         
                # get the difference of the two
                diffs[division_index] = data_CMB - data_NIDIS
 
            # add the variable into the dataset and add the data into the variable
            netcdf_utils.add_variable_climdivs(netcdf_file_OUT, var_name, dataset_NIDIS.variables[values[1]].__dict__, diffs)

            # get the variable values for the month, mask the NaNs (data assumed to be in (division, time) dimension order)
            data_CMB = np.ma.masked_invalid(dataset_CMB.variables[values[0]][:, :], copy=False)
            data_NIDIS = np.ma.masked_invalid(dataset_NIDIS.variables[values[1]][:, :], copy=False)
            
            # get the difference of the two
            diffs = data_CMB - data_NIDIS
            
#             # keep the original shape
#             original_shape = diffs.shape
#
#             # get just the unmasked values, as a 1-D (flattened) array
#             diffs = diffs[~diffs.mask]
# 
#             # reshape back to original shape
#             diffs = np.reshape(diffs, original_shape)
            
            # plot the differences on a monthly basis
    
            # reshape from (divisions, months) to (divisions, years, 12)
            diffs = utils.reshape_to_divs_years_months(diffs)
            
            # common title
            histogram_title = 'CMB vs. NIDIS: '

            # loop over each month
            for i in range(12):
    
                # get the month as a string
                month = datetime.datetime.strptime(str(i + 1), "%m").strftime("%b")
    
                # plot a histogram of the differences
                plt.gcf().clear()
                plt.hist(diffs[:, :, i], bins=number_of_bins, range=(range_lower, range_upper))
                plt.title(histogram_title + ': {0}/{1}'.format(var_name, month))
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                file_name = args.output_dir + os.sep + 'diffs_cmb_nidis_{0}_'.format(var_name) + str(i + 1).zfill(2) + '.png'
                logger.info('Saving plot for month {0} as file {1}'.format(i + 1, file_name))
                plt.savefig(file_name)
