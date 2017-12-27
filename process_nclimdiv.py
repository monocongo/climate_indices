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
import pdinew
from process import processor, process_divisions
import random

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
        
        # perform an ingest of the NCEI nClimDiv datasets for input (temperature  
        # and precipitation) plus monthly computed indices for comparison
        ingest_nclimdiv.ingest_netcdf_latest(args.out_file,
                                             temp_var_name,
                                             precip_var_name,
                                             awc_var_name)

        # perform the processing, using original NCDC PET calculation method
        process_divisions.process_divisions(args.out_file,
                                            precip_var_name,
                                            temp_var_name,
                                            awc_var_name,
                                            args.month_scales,
                                            args.calibration_start_year,
                                            args.calibration_end_year,
                                            use_orig_pe=True)
        
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
    