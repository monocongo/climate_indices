import argparse
from datetime import datetime
import logging
import multiprocessing
import netCDF4
import numpy as np
import random

from indices_python import netcdf_utils, utils
from indices_python.scripts.ingest import ingest_nclimgrid, ingest_prism
from indices_python.scripts.process import process_grid

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
def _plot_and_save_histogram(difference_values,        # pragma: no cover
                             number_of_bins,
                             range_lower, 
                             range_upper,
                             index_name,
                             grid_name,
                             title,
                             output_filepath):
    
    # plot a histogram of the differences
    plt.gcf().clear()
    plt.hist(difference_values[:], bins=number_of_bins, range=(range_lower, range_upper))
    plt.title(title + ': {0}, {1}'.format(index_name, grid_name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
    # save to file
    logger.info('Saving histogram plot for index %s to file %s', index_name, output_filepath)
    plt.savefig(output_filepath)

#-----------------------------------------------------------------------------------------------------------------------
def _plot_and_save_lines(expected,          # pragma: no cover
                         actual,
                         difference_values,
                         grid_name,
                         varname,
                         output_filepath):

    # get the RMSE for the two sets of values
    error = utils.rmse(actual, expected)
    
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
    plt.title('Comparison for {0}: {1}     (RMSE: {2})'.format(grid_name, varname, error))
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
    This module is used to perform climate indices processing on gridded datasets in NetCDF.
    """

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        logger.info("Start time:    %s", start_datetime)

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--grid", 
                            help="Valid values are \'nclimgrid\' and \'prism\'", 
                            required=True)
        parser.add_argument("--source_dir", 
                            help="Base directory under which are directories and files for precipitation and max/min/mean temperature", 
                            required=True)
        parser.add_argument("--output_dir", 
                            help="Directory under which the output NetCDF files will be written", 
                            required=True)
        args = parser.parse_args()

        # variable names used within the monthly NetCDF
        temp_var_name = 'tavg'
        precip_var_name = 'prcp'
        awc_var_name = 'awc'

        if args.grid == 'nclimgrid':

            # perform an ingest of the NCEI nClimGrid datasets for input (temperature  
            # and precipitation) plus soil constants (available water capacity)
            precip_file, temp_file, tmin_file, tmax_file = ingest_nclimgrid.ingest_to_netcdf(args.source_dir, args.output_dir)
            awc_file = args.output_dir + '/nclimgrid_soil.nc'
            utils.retrieve_file('https://github.com/monocongo/indices_python/blob/develop/example_inputs/nclimgrid_soil.nc', 
                                awc_file)

        elif args.grid == 'prism':
            
            # perform an ingest of the PRISM datasets for input (temperature  
            # and precipitation) plus soil constants (available water capacity)
            prism_file = ingest_prism.ingest_to_netcdf(args.output_dir, True)
            precip_file = prism_file
            temp_file = prism_file
            awc_file = args.output_dir + '/prism_soil.nc'
            utils.retrieve_file('https://github.com/monocongo/indices_python/blob/develop/example_inputs/prism_soil.nc', 
                                awc_file)
        
        else:
            
            error_message = 'Unsupported grid type: {0}'.format(args.grid)
            logger.error(error_message)
            raise ValueError(error_message)
        
        # perform the processing
        process_grid.process_grid(args.output_file_base,
                                  args.precip_file,
                                  args.temp_file,
                                  args.awc_file,
                                  args.precip_var_name,
                                  args.temp_var_name,
                                  args.awc_var_name,
                                  args.month_scales,
                                  args.calibration_start_year,
                                  args.calibration_end_year)

        # variable names for variables to diff from the two datasets
        comparison_arrays = {'PDSI': ('wrcc_pdsi', 'pdsi'),
                             'PHDI': ('wrcc_phdi', 'phdi'),
                             'PMDI': ('wrcc_pmdi', 'pmdi'),
                             'Z-Index': ('cmb_zndx', 'zindex')}
        for index, var_names in comparison_arrays.items():
                
            # TODO validate that the two variables exist, have compatible dimensions/units, etc., all of which is assumed below  pylint: disable=fixme

            logger.info('Computing differences on variable %s', index)
        
            # open the NetCDF files
            with netCDF4.Dataset('{0}/{1}_diffs_{2}.nc'.format(args.output_file_base, args.grid, var_names[1]), 'a') as dataset:
    
                # allocate an array for the differences for this variable
                diffs = np.full(dataset.variables[var_names[0]].shape(), np.NaN)
                
                # compute and plot difference values on a (calendar) monthly basis
                for month in range(0, 12):
                    
                    # get the variable values for the month, mask the NaNs, data assumed to be in (time, lat, lon) order
                    expected_values = np.ma.masked_invalid(dataset.variables[var_names[0]][month::12, :, :], copy=False)
                    actual_values = np.ma.masked_invalid(dataset.variables[var_names[1]][month::12, :, :], copy=False)
             
                    # get the difference of the two
                    differences = expected_values - actual_values
                    diffs[month::12] = differences
     
                    # plot the differences as a histogram and save to file
                    _plot_and_save_histogram(differences,
                                             80,   # number_of_bins
                                             -2,   # lower range
                                             2,    # upper range
                                             index,
                                             args.grid.upper(),
                                             'CMB vs. NIDIS,  Month: {0}'.format(month + 1),
                                             '{0}/histogram_{1}_{2}.png'.format(args.output_file_base, var_names[1], str(month + 1).zfill(2)))
     
                    # plot and save line graphs showing correlation of values and differences
                    _plot_and_save_lines(expected_values,
                                         actual_values,
                                         differences,
                                         index,
                                         args.grid.upper(),
                                         '{0}/line_{1}_{2}.png'.format(args.output_file_base, var_names[1], str(month + 1).zfill(2)))
                    
                    # add to the differences dictionary with this division ID key 
                    diffs[month::12] = differences
                    
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
                                                      ('time', 'lat', 'lon',), 
                                                      fill_value=np.NaN)
                
                # assign the array into the current division's slot in the variable
                variable[:,:,:] = diffs
        
        # report on the elapsed time
        end_datetime = datetime.now()
        logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        logger.exception('Failed to complete', exc_info=True)
        raise
