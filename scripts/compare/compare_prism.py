import argparse
import datetime
import logging
import multiprocessing
import netCDF4
import numpy as np

from indices_python import netcdf_utils, utils
from scripts.process import process_grid
from scripts.task import task_grid

#-----------------------------------------------------------------------------------------------------------------------
# set up matplotlib to use the Agg backend, in order to remove any dependencies on an X server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
_logger = logging.getLogger(__name__)

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
def _plot_and_save_histogram(difference_values,          # pragma: no cover
                             number_of_bins,
                             range_lower, 
                             range_upper,
                             climate_index_name,
                             title,
                             output_filepath):
    
    # plot a histogram of the differences
    plt.gcf().clear()
    plt.hist(difference_values[:], bins=number_of_bins, range=(range_lower, range_upper))
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
    # save to file
    _logger.info('Saving histogram plot for index %s to file %s', climate_index_name, output_filepath)
    plt.savefig(output_filepath)

#-----------------------------------------------------------------------------------------------------------------------
def _plot_and_save_lines(expected,             # pragma: no cover
                         actual,
                         difference_values,
                         rmserror,
                         percent_signchange,
                         varname,
                         output_filepath):

    # set figure size to (x, y)
    plt.figure(figsize=(48, 6))
    
    full_title = '{0} comparison for PRISM'.format(varname) + \
            '  (RMSE: % 5.2f,  percent with sign change: % 6.2f)' % (rmserror, percent_signchange)

    # plot the values and differences
    x = np.arange(difference_values.size)
    ax = plt.axes()
    ax.set_ylim([-5, 5])
    plt.axhline()
    expected_line, = plt.plot(x, expected, color='blue', label='WRCC (expected)')
    actual_line, = plt.plot(x, actual, color='yellow', linestyle='--', label='NIDIS (actual)')
    diffs_line, = plt.plot(x, difference_values, color='red', label='Difference')
    plt.legend(handles=[expected_line, actual_line, diffs_line], loc='upper left')
    plt.title(full_title)
    plt.xlabel('months')
    plt.ylabel('value')
    
    plt.subplots_adjust(left=0.02, right=0.99, top=0.9, bottom=0.1)
    
    # save to file
    _logger.info('Saving line plot for index %s to file %s', varname, output_filepath)
    plt.savefig(output_filepath, bbox_inches='tight')
    plt.close()

#-----------------------------------------------------------------------------------------------------------------------
def _get_variables(prism_dataset,
                   var_name,
                   diff_prefix,
                   rmse_prefix,
                   signchange_prefix):

    # use data type mapped to vanilla floats
    netcdf_data_type = netcdf_utils.find_netcdf_datatype(1.0)
    
    # get the NetCDF variable we'll use to populate with difference values for this index, create if does not exist
    diff_variable_name = diff_prefix + var_name
    if diff_variable_name in prism_dataset.variables.keys():
        
        # variable already exists
        diff_var = prism_dataset.variables[diff_variable_name]
        
    else:
        
        # create the variable, set the attributes
        diff_var = prism_dataset.createVariable(diff_variable_name, 
                                                    netcdf_data_type, 
                                                    ('time', 'lat', 'lon',), 
                                                    fill_value=np.NaN)
    
    # get the NetCDF variable we'll use to populate with RMSE value for this index, create if does not exist
    rmse_variable_name = rmse_prefix + var_name
    if rmse_variable_name in prism_dataset.variables.keys():
        
        # variable already exists
        rmse_var = prism_dataset.variables[rmse_variable_name]
        
    else:
        
        # create the (scalar) variable
        rmse_var = prism_dataset.createVariable(rmse_variable_name, 
                                                netcdf_data_type, 
                                                fill_value=np.NaN)
    
    # get the NetCDF variable we'll use to populate with % sign change value for this index, create if does not exist
    signchange_variable_name = signchange_prefix + var_name
    if signchange_variable_name in prism_dataset.variables.keys():
        
        # variable already exists
        signchange_var = prism_dataset.variables[signchange_variable_name]
        
    else:
        
        # create the (scalar) variable
        signchange_var = prism_dataset.createVariable(signchange_variable_name, 
                                                      netcdf_data_type, 
                                                      fill_value=np.NaN)

    return diff_var, rmse_var, signchange_var

#-----------------------------------------------------------------------------------------------------------------------
def _summary_analysis_plots(prism_dataset,
                            climate_index_name,
                            variable_name_wrcc,
                            variable_name_nidis,
                            diff_prefix,
                            rmse_prefix,
                            signchange_prefix,
                            output_dir):
    
    # get the variables used for diffs, RMSE, and percentage sign change for this climate index
    diff_variable, rmse_variable, signchange_variable = _get_variables(prism_dataset,
                                                                       variable_name_nidis,
                                                                       diff_prefix,
                                                                       rmse_prefix,
                                                                       signchange_prefix)
    
    # common title for plots
    histogram_title = 'WRCC vs. NIDIS: '
    
    _logger.info('Computing diffs, RMSE, and percentage sign change for PRISM')

    # get the variable var_names for the month, mask the NaNs (data assumed to be in (division, time) dimension order)
    data_WRCC = np.ma.masked_invalid(prism_dataset.variables[variable_name_wrcc][:, :, :], copy=False)
    data_NIDIS = np.ma.masked_invalid(prism_dataset.variables[variable_name_nidis][:, :, :], copy=False)

    # get the differences, assign the array into the current division's slot within the NetCDF variable
    differences = data_WRCC - data_NIDIS
    diff_variable[:, :, :] = np.reshape(differences, (1, differences.size))

    # get the RMSE for the two sets of values, sum for later averaging, and assign into NetCDF variable
    error = utils.rmse(data_NIDIS, data_WRCC)
    rmse_variable[:] = np.array([error])

    # compute the percentage sign change
    sign_changes = utils.sign_change(data_WRCC, data_NIDIS)
    percentage_sign_change = 100.0 * np.count_nonzero(sign_changes.flatten()) / sign_changes.size
    signchange_variable[:] = np.array([percentage_sign_change])

    # compute % change
    percent_bias = np.nanmean((data_NIDIS - data_WRCC) * 100.0 / data_WRCC)
        
    # display the division's RMSE and percentage sign change for the current index
    print('% s RMSE: % 5.2f    percentage sign change: % 4.1f  percent bias: % 4.1f' % (climate_index_name, 
                                                                                        error, 
                                                                                        percentage_sign_change,
                                                                                        percent_bias))

    # plot the differences as a histogram and save to file
    _plot_and_save_histogram(differences,
                             80,   # number_of_bins
                             -2,   # lower range
                             2,    # upper range
                             climate_index_name,
                             histogram_title,
                             output_dir + '/diffs_histogram_{0}.png'.format(variable_name_nidis))
    
    # plot and save line graphs showing correlation of values and differences
    _plot_and_save_lines(data_NIDIS,
                         data_WRCC,
                         differences,
                         error,
                         percentage_sign_change,
                         climate_index_name,
                         output_dir + '/diffs_line_{0}.png'.format(variable_name_nidis))

    # report summary statistics
    print('\nTotal Mean RMSE for {0}: {1}'.format(climate_index_name, error))

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    example command line invocation:
    
    $ python -u compare_prism.py --out_file C:/home/data/nclimdiv/nclimdiv_latest.nc \
                                 --month_scales 1 2 3 6 9 12 24 \
                                 --calibration_start_year 1931 \
                                 --calibration_end_year 1990

    This module is used to perform climate indices processing and analysis on PRISM datasets.
    
    Ingests from NetCDF (12 files, one per calendar month) to single, compressed, full period of record NetCDF.
    Processes indices from NetCDF input.
    Computes and stores differences compared to corresponding WRCC operational results.
    Plots differences as line graphs and histograms.
    """

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.datetime.now()
        _logger.info("Start time:    %s", start_datetime)

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--output_dir", 
                            help="Directory to contain NetCDF output files, one per variables computed", 
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
                            default=1931,
                            choices=range(1870, start_datetime.year + 1),
                            required=False)
        parser.add_argument("--calibration_end_year",
                            help="Final year of calibration period",
                            type=int,
                            default=1990,
                            choices=range(1870, start_datetime.year + 1),
                            required=False)
        args = parser.parse_args()

        # variable names used within the monthly NetCDF
        temp_var_name = 'tavg'
        precip_var_name = 'prcp'
        awc_var_name = 'awc'

#         # perform the ingest to NetCDF and indices processing, writing results back into input NetCDF (?, confirm this)
#         task_grid.ingest_and_process_indices('prism', 
#                                              None, 
#                                              args.output_dir, 
#                                              args.month_scales, 
#                                              args.calibration_start_year, 
#                                              args.calibration_end_year)
        
        # perform the indices processing, writing results back into input NetCDF (?, confirm this)
        use_original_pet = False
        diff_name_prefix = 'diffs_'
        rmse_name_prefix = 'rmse_'
        signchange_name_prefix = 'signchange_'
        output_dir = 'C:/home/data/prism/shiva_run_20171221/diff_plots'        
        process_grid.process_grid('C:/home/data/prism/prism_latest', 
                                  'C:/home/data/prism/shiva_run_20171221/PRISM_from_WRCC_prcp.nc', 
                                  'C:/home/data/prism/shiva_run_20171221/PRISM_from_WRCC_tavg.nc', 
                                  'C:/home/data/prism/shiva_run_20171221/prism_soil.nc', 
                                  precip_var_name, 
                                  temp_var_name, 
                                  awc_var_name, 
                                  args.month_scales,
                                  args.calibration_start_year,
                                  args.calibration_end_year)
        
        # open the NetCDF files
        with netCDF4.Dataset(args.out_file, 'a') as dataset:

            # variable names for variables to diff from the two datasets
            comparison_arrays = {'PDSI': ('wrcc_pdsi', 'pdsi'),
                                 'PHDI': ('wrcc_phdi', 'phdi'),
                                 'PMDI': ('wrcc_pmdi', 'pmdi'),
                                 'Z-Index': ('wrcc_zndx', 'zindex'),
                                 'SPI-1': ('wrcc_sp01', 'spi_pearson_01'),
                                 'SPI-12': ('wrcc_sp12', 'spi_pearson_12')}
            for index_name, var_names in comparison_arrays.items():
                    
                # TODO validate that the two variables exist, have compatible dimensions/units, etc., all of which is assumed below  pylint: disable=fixme
     
                _logger.info('Computing differences on variable %s', index_name)
                 
                # perform analysis, writing results back into NetCDF, with plots of specified divisions
                _summary_analysis_plots(dataset,
                                        index_name,
                                        var_names[0],
                                        var_names[1],
                                        diff_name_prefix,
                                        rmse_name_prefix,
                                        signchange_name_prefix,
                                        output_dir)

        # report on the elapsed time
        end_datetime = datetime.datetime.now()
        _logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        _logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        _logger.exception('Failed to complete', exc_info=True)
        raise
    