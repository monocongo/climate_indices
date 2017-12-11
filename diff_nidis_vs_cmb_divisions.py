import argparse
import logging
import netCDF4
import netcdf_utils
import numpy as np
import os

#-----------------------------------------------------------------------------------------------------------------------
# set up matplotlib to use the Agg backend, in order to remove any dependencies on an X server
import matplotlib
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
                      variables=None):
    '''
    This function is used to initialize and return a netCDF4.Dataset object.
    
    :param new_netcdf: the file path/name of the new NetCDF Dataset object to be created and returned by this function
    :param template_netcdf: an existing/open NetCDF Dataset object which will be used as a template for the Dataset
                            that will be created and returned by this function
    :param variables: if present this is a dictionary of variable names (keys) to variable attributes/data of 
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

        # set the coordinate variables' attributes and var_names
        time_variable.setncatts(template_dataset.variables['time'].__dict__)
        time_variable[:] = template_dataset.variables['time'][:]
        division_variable.setncatts(template_dataset.variables['division'].__dict__)
        division_variable[:] = template_dataset.variables['division'][:]

        # create a variable for each variable listed in the dictionary of variable names to variable attributes/data
        if variables is not None:
            for variable_name, variable in variables.items():
                
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
def _rmse(predictions, targets):
    """
    Root mean square error
    
    :param predictions: np.ndarray
    :param targets: np.ndarray
    :return: np.ndarray
    """
    return np.sqrt(((predictions - targets) ** 2).mean())

#-----------------------------------------------------------------------------------------------------------------------
def _plot_and_save_histograms(diffs,
                              number_of_bins,
                              range_lower, 
                              range_upper,
                              index,
                              division_id,
                              output_dir,
                              title):
    
    # plot a histogram of the differences
    plt.gcf().clear()
    plt.hist(diffs[:], bins=number_of_bins, range=(range_lower, range_upper))
    plt.title(title + ': {0}, Division {1}'.format(index, division_id))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    file_name = output_dir + os.sep + 'nclimdiv_diffs_{0}_{1}'.format(index, division_id) + '.png'
    logger.info('Saving plot for index {0} as file {1}'.format(index, file_name))
    plt.savefig(file_name)

#-----------------------------------------------------------------------------------------------------------------------
def _plot_and_save_lines(expected,
                         actual,
                         diffs,
                         division_id,
                         varname,
                         output_dir):

    error = _rmse(actual, expected)
    
    # set figure size to (x, y)
    plt.figure(figsize=(30, 6))
    
    # plot the values and differences
    x = np.arange(diffs.size)
    ax = plt.axes()
    ax.set_ylim([-5, 5])
    plt.axhline()
    expected_line, = plt.plot(x, expected, color='blue', label='NCEI (expected)')
    actual_line, = plt.plot(x, actual, color='yellow', linestyle='--', label='NIDIS (actual)')
    diffs_line, = plt.plot(x, diffs, color='red', label='Difference')
    plt.legend(handles=[expected_line, actual_line, diffs_line], loc='upper left')
    plt.title('Comparison for division {0}: {1}     (RMSE: {2})'.format(division_id, varname, error))
    plt.xlabel("months")
    plt.ylabel("value")
    
    plt.subplots_adjust(left=0.02, right=0.99, top=0.9, bottom=0.1)
    
    file_name = output_dir + os.sep + '{0}_div_{1}'.format(varname, division_id) + '.png'
    logger.info('Saving plot for variable/division {0}/{1} as file {2}'.format(varname, division_id, file_name))
    plt.savefig(file_name, bbox_inches='tight')
                
#     plt.show()
    plt.close()

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--nidis_file", 
                        help="NIDIS input dataset file (NetCDF) containing var_names for the named variable", 
                        required=True)
    parser.add_argument("--cmb_file", 
                        help="CMB input dataset file (NetCDF) containing var_names for the named variable", 
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
        comparison_arrays = {'PDSI': ('cmb_pdsi', 'pdsi'),
                'PHDI': ('cmb_phdi', 'phdi'),
                'PMDI': ('cmb_pmdi', 'pmdi'),
                'Z-Index': ('cmb_zndx', 'zindex')}
        for index, var_names in comparison_arrays.items():
                    
            # TODO validate that the two variables exist, have compatible dimensions/units, etc., all of which is assumed below

            logger.info('Computing differences on variable {0}'.format(index))
            
            # allocate an array for the differences for this variable
            diffs = {}
            
            size = dataset_CMB.variables['division'][:].size
            
            # common title for plots
            histogram_title = 'CMB vs. NIDIS: '
 
            for division_index, division_id in enumerate(dataset_CMB.variables['division'][:]):
             
                logger.info('Computing diffs for climate division ID: {0}'.format(division_id))
                
                # get the variable var_names for the month, mask the NaNs (data assumed to be in (division, time) dimension order)
                data_CMB = np.ma.masked_invalid(dataset_CMB.variables[var_names[0]][division_index, :], copy=False)
                data_NIDIS = np.ma.masked_invalid(dataset_NIDIS.variables[var_names[1]][division_index, :], copy=False)
         
                # get the difference of the two
                differences = data_CMB - data_NIDIS
                diffs[division_index] = differences
 
#                 # plot the differences as a histogram and save to file
#                 _plot_and_save_histograms(differences,
#                                           number_of_bins,
#                                           range_lower, 
#                                           range_upper,
#                                           index,
#                                           division_id,
#                                           args.output_dir,
#                                           histogram_title)

                # plot and save line graphs showing correlation of values and differences
                _plot_and_save_lines(data_NIDIS,
                                     data_CMB,
                                     differences,
                                     division_id,
                                     index,
                                     args.output_dir)

            # add the variable into the dataset and add the data into the variable
            netcdf_utils.add_variable_climdivs(netcdf_file_OUT, index, dataset_NIDIS.variables[var_names[1]].__dict__, diffs)

#             # get the variable values, mask the NaNs (data assumed to be in (division, time) dimension order)
#             data_CMB = np.ma.masked_invalid(dataset_CMB.variables[var_names[0]][:, :], copy=False)
#             data_NIDIS = np.ma.masked_invalid(dataset_NIDIS.variables[var_names[1]][:, :], copy=False)
#             
#             # get the difference of the two
#             diffs = data_CMB - data_NIDIS
#             
#             # plot the differences on a monthly basis
#     
#             # reshape from (divisions, months) to (divisions, years, 12)
#             diffs = utils.reshape_to_divs_years_months(diffs)
#             
#             # loop over each month
#             for i in range(12):
#     
#                 # get the month as a string
#                 month = datetime.datetime.strptime(str(i + 1), "%m").strftime("%b")
#     
#                 # plot a histogram of the differences
#                 plt.gcf().clear()
#                 plt.hist(diffs[:, :, i], bins=number_of_bins, range=(range_lower, range_upper))
#                 plt.title(histogram_title + ': {0}/{1}'.format(index, month))
#                 plt.xlabel("Value")
#                 plt.ylabel("Frequency")
#                 file_name = args.output_dir + os.sep + 'diffs_cmb_nidis_{0}_'.format(index) + str(i + 1).zfill(2) + '.png'
#                 logger.info('Saving plot for month {0} as file {1}'.format(i + 1, file_name))
#                 plt.savefig(file_name)
