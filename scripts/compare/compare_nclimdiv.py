import argparse
import datetime
import logging
import multiprocessing
import netCDF4
import numpy as np

from indices_python import netcdf_utils, utils
from scripts.process import process_divisions
# from scripts.task import task_divisions

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
def _plot_and_save_histogram_divisional(difference_values,          # pragma: no cover
                                        number_of_bins,
                                        range_lower, 
                                        range_upper,
                                        climate_index_name,
                                        climdiv_id,
                                        title,
                                        output_filepath):

    full_title = title + ': {0}, Division {1}'.format(climate_index_name, climdiv_id)
    
    _plot_and_save_histogram(difference_values,          # pragma: no cover
                             number_of_bins,
                             range_lower, 
                             range_upper,
                             climate_index_name,
                             full_title,
                             output_filepath)

#-----------------------------------------------------------------------------------------------------------------------
def _plot_and_save_histogram_monthly(difference_values,          # pragma: no cover
                                     number_of_bins,
                                     range_lower, 
                                     range_upper,
                                     climate_index_name,
                                     month_name,
                                     plot_title,
                                     output_filepath):
    
    full_title = plot_title + ': {0}, Month {1}'.format(climate_index_name, month_name)
    
    _plot_and_save_histogram(difference_values,          # pragma: no cover
                             number_of_bins,
                             range_lower, 
                             range_upper,
                             climate_index_name,
                             full_title,
                             output_filepath)
    
#-----------------------------------------------------------------------------------------------------------------------
def _plot_and_save_lines(expected,             # pragma: no cover
                         actual,
                         difference_values,
                         title,
                         varname,
                         output_filepath,
                         x_label):

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
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("value")
    
    plt.subplots_adjust(left=0.02, right=0.99, top=0.9, bottom=0.1)
    
    # save to file
    _logger.info('Saving histogram plot for index %s to file %s', varname, output_filepath)
    plt.savefig(output_filepath, bbox_inches='tight')

#     plt.show()
    plt.close()

#-----------------------------------------------------------------------------------------------------------------------
def _plot_and_save_lines_divisional(expected,             # pragma: no cover
                                    actual,
                                    difference_values,
                                    rmse,
                                    percent_sign_change,
                                    climdiv_id,
                                    varname,
                                    output_filepath):

    title = '{0} comparison for division {1}  (RMSE: {2},  percent with sign change: {3})'.format(climdiv_id, 
                                                                                                  varname, 
                                                                                                  rmse, 
                                                                                                  percent_sign_change)

    _plot_and_save_lines(expected,             # pragma: no cover
                         actual,
                         difference_values,
                         title,
                         varname,
                         output_filepath,
                         "months")

#-----------------------------------------------------------------------------------------------------------------------
def _plot_and_save_lines_monthly(expected,             # pragma: no cover
                                 actual,
                                 difference_values,
                                 rmse,
                                 month_name,
                                 varname,
                                 output_filepath):

    title = 'Comparison for month {0}: {1}     (RMSE: {2})'.format(month_name, varname, rmse)

    _plot_and_save_lines(expected,             # pragma: no cover
                         actual,
                         difference_values,
                         title,
                         varname,
                         output_filepath,
                         "years")

#-----------------------------------------------------------------------------------------------------------------------
def _get_variables(divisions_dataset,
                   var_name,
                   diff_prefix,
                   rmse_prefix,
                   signchange_prefix):

    # use data type mapped to vanilla floats
    netcdf_data_type = netcdf_utils.find_netcdf_datatype(1.0)
    
    # get the NetCDF variable we'll use to populate with difference values for this index, create if does not exist
    diff_variable_name = diff_prefix + var_name
    if diff_variable_name in divisions_dataset.variables.keys():
        
        # variable already exists
        diff_var = divisions_dataset.variables[diff_variable_name]
        
    else:
        
        # create the variable, set the attributes
        diff_var = divisions_dataset.createVariable(diff_variable_name, 
                                                    netcdf_data_type, 
                                                    ('division', 'time',), 
                                                    fill_value=np.NaN)
    
    # get the NetCDF variable we'll use to populate with RMSE value for this index, create if does not exist
    rmse_variable_name = rmse_prefix + var_name
    if rmse_variable_name in divisions_dataset.variables.keys():
        
        # variable already exists
        rmse_var = divisions_dataset.variables[rmse_variable_name]
        
    else:
        
        # create the variable, set the attributes
        rmse_var = divisions_dataset.createVariable(rmse_variable_name, 
                                                    netcdf_data_type, 
                                                    ('division',),
                                                    fill_value=np.NaN)
    
    # get the NetCDF variable we'll use to populate with % sign change value for this index, create if does not exist
    signchange_variable_name = signchange_prefix + var_name
    if signchange_variable_name in divisions_dataset.variables.keys():
        
        # variable already exists
        signchange_var = divisions_dataset.variables[signchange_variable_name]
        
    else:
        
        # create the variable, set the attributes
        signchange_var = divisions_dataset.createVariable(signchange_variable_name, 
                                                          netcdf_data_type, 
                                                          ('division',),
                                                          fill_value=np.NaN)

    return diff_var, rmse_var, signchange_var

#-----------------------------------------------------------------------------------------------------------------------
def _monthly_analysis_plots(divs_dataset,       # pragma: no cover
                            variable_names,
                            climate_index_name,
                            divs_to_plot=None):
    pass
#                     # perform "per calendar month" analysis
#                     for i in range(0, 12):
#                         
#                         month = datetime.date(1900, i + 1, 1).strftime('%B')
#                         
#                         print('\tMonth: {0}'.format(month))
#     
#                         # get the variable var_names for the month, mask the NaNs (data assumed to be in (division, time) dimension order)
#                         data_CMB = np.ma.masked_invalid(divs_dataset.variables[variable_names[0]][:, i::12], copy=False)
#                         data_NIDIS = np.ma.masked_invalid(divs_dataset.variables[variable_names[1]][:, i::12], copy=False)
#      
#                         # get the difference of the two, add into the differences array at the correct slot for this division
#                         differences = data_CMB - data_NIDIS
#     
#                         # get the RMSE for the two sets of values
#                         error = utils.rmse(data_NIDIS, data_CMB)
#                         rmse_monthly[i] = error
#          
#                         # plot the differences as a histogram and save to file
#                         _plot_and_save_histogram_monthly(differences,
#                                                          80,   # number_of_bins
#                                                          -2,   # lower range
#                                                          2,    # upper range
#                                                          climate_index_name,
#                                                          month,
#                                                          histogram_title,
#                                                          'C:/home/data/nclimdiv/diffs_histogram_{0}_month{1}.png'.format(variable_names[1], i + 1))
#          
#                         # plot and save line graphs showing correlation of values and differences
#                         _plot_and_save_lines_monthly(data_NIDIS,
#                                                      data_CMB,
#                                                      differences,
#                                                      error,
#                                                      month,
#                                                      climate_index_name,
#                                                      'C:/home/data/nclimdiv/diffs_line_{0}_month{1}.png'.format(variable_names[1], month))
#
#-----------------------------------------------------------------------------------------------------------------------
def _divisional_analysis_plots(divs_dataset,
                               climate_index_name,
                               variable_names,
                               diff_prefix,
                               rmse_prefix,
                               signchange_prefix,
                               divs_to_plot=None):
    
    # get the variables used for diffs, RMSE, and percentage sign change for this climate index
    diff_variable, rmse_variable, signchange_variable = _get_variables(divs_dataset,
                                                                       variable_names[1],
                                                                       diff_prefix,
                                                                       rmse_prefix,
                                                                       signchange_prefix)
    
    # common title for plots
    histogram_title = 'CMB vs. NIDIS: '
    
    # count the number of divisions we've analyzed in order to get a mean for various statistics such as RMSE
    divs_analyzed = 0
    rmse_sum = 0.0
                
    # perform "per division" analysis
    for division_index, division_id in enumerate(divs_dataset.variables['division'][:]):

        # only process divisions within CONUS, 101 - 4809
        if division_id > 4899:

            continue

        _logger.info('Computing diffs, RMSE, and percentage sign change for climate division ID: %s', division_id)

        # keep count of the number of divisinos we've analyzed, for use with averaging later
        divs_analyzed += 1

        # get the variable var_names for the month, mask the NaNs (data assumed to be in (division, time) dimension order)
        data_CMB = np.ma.masked_invalid(divs_dataset.variables[variable_names[0]][division_index, :], copy=False)
        data_NIDIS = np.ma.masked_invalid(divs_dataset.variables[variable_names[1]][division_index, :], copy=False)

        # get the differences, assign the array into the current division's slot within the NetCDF variable
        differences = data_CMB - data_NIDIS
        diff_variable[division_index, :] = np.reshape(differences, (1, differences.size))

        # get the RMSE for the two sets of values, sum for later averaging, and assign into NetCDF variable
        error = utils.rmse(data_NIDIS, data_CMB)
        rmse_sum += error
        rmse_variable[division_index] = np.array([error])

        # compute the percentage sign change
        sign_changes = utils.sign_change(data_CMB, data_NIDIS)
        percentage_sign_change = 100.0 * np.count_nonzero(sign_changes.flatten()) / sign_changes.size
        signchange_variable[division_index] = np.array([percentage_sign_change])

        # display the division's RMSE and percentage sign change for the current index
        print('Division {0}:  {1} RMSE: {2}    % sign change: {3}'.format(division_id, 
                                                                          climate_index_name, 
                                                                          error, 
                                                                          percentage_sign_change))

        # only plot divisions in the list, if specified
        if divs_to_plot is not None and division_id not in divs_to_plot:

            continue

        else:

            # plot the differences as a histogram and save to file
            _plot_and_save_histogram_divisional(differences,
                                                80,   # number_of_bins
                                                -2,   # lower range
                                                2,    # upper range
                                                climate_index_name,
                                                division_id,
                                                histogram_title,
                                                output_dir + '/diffs_histogram_{0}_{1}.png'.format(variable_names[1], 
                                                                                                   division_id))

            # plot and save line graphs showing correlation of values and differences
            _plot_and_save_lines_divisional(data_NIDIS,
                                            data_CMB,
                                            differences,
                                            error,
                                            percentage_sign_change,
                                            division_id,
                                            climate_index_name,
                                            output_dir + '/diffs_line_{0}_{1}.png'.format(variable_names[1], 
                                                                                          division_id))
        # report summary statistics
        print('\nTotal Mean RMSE for {0}: {1}'.format(climate_index_name, rmse_sum / divs_analyzed))

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    example command line invocation:
    
    $ python -u --out_file C:/home/data/nclimdiv/nclimdiv_latest.nc \
                --month_scales 1 2 3 6 9 12 24 \
                --calibration_start_year 1931 \
                --calibration_end_year 1990

    $ python -u task_divisions.py --out_file C:/home/data/nclimdiv/nclimdiv_latest.nc --month_scales 1 2 3 6 9 12 24 --calibration_start_year 1931 --calibration_end_year 1990
        
    This module is used to perform climate indices processing on nClimGrid datasets in NetCDF.
    """

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.datetime.now()
        _logger.info("Start time:    %s", start_datetime)

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

        #DEBUG ONLY -- REMOVE
        divisions_to_plot = [405, 1309, 3405]
#         divisions_to_plot = None

        # settings for use of either original NCDC method or new Thornthwaite method for PET computation
#         diff_name_prefix = 'diffs_oldpet_'
#         rmse_name_prefix = 'rmse_oldpet_'
#         signchange_name_prefix = 'signchange_oldpet_'
#         use_original_pet = True
#         output_dir = 'C:/home/data/nclimdiv/diff_plots_oldpe'
#         # perform the processing, using original NCDC PET calculation method, writing results back into input NetCDF
#         task_divisions.ingest_and_process_indices(args.out_file, 
#                                                   temp_var_name, 
#                                                   precip_var_name, 
#                                                   awc_var_name, 
#                                                   args.month_scales,
#                                                   args.calibration_start_year,
#                                                   args.calibration_end_year,
#                                                   use_orig_pe=use_original_pet)
        
        # perform the processing, using original NCDC PET calculation method, writing results back into input NetCDF
        use_original_pet = False
        diff_name_prefix = 'diffs_newpet_'
        rmse_name_prefix = 'rmse_newpet_'
        signchange_name_prefix = 'signchange_newpet_'
        output_dir = 'C:/home/data/nclimdiv/diff_plots_newpe'        
#         process_divisions.process_divisions(args.out_file,
#                                             precip_var_name,
#                                             temp_var_name,
#                                             awc_var_name,
#                                             args.month_scales,
#                                             args.calibration_start_year,
#                                             args.calibration_end_year,
#                                             use_orig_pe=use_original_pet)
        # open the NetCDF files
        with netCDF4.Dataset(args.out_file, 'a') as dataset:

            # variable names for variables to diff from the two datasets
            comparison_arrays = {'PDSI': ('cmb_pdsi', 'pdsi'),
                                 'PHDI': ('cmb_phdi', 'phdi'),
                                 'PMDI': ('cmb_pmdi', 'pmdi'),
                                 'Z-Index': ('cmb_zndx', 'zindex'),
                                 'SPI-1': ('cmb_sp01', 'spi_pearson_01'),
                                 'SPI-12': ('cmb_sp12', 'spi_pearson_12')}
            for index_name, var_names in comparison_arrays.items():
                    
                # only do the below monthly comparisons if we're dealing with all divisions
                if divisions_to_plot is None:
 
                    # TODO validate that the two variables exist, have compatible dimensions/units, etc., all of which is assumed below  pylint: disable=fixme
     
                    _logger.info('Computing differences on variable %s', index_name)
                 
                    # perform monthly analysis and plots
                    _monthly_analysis_plots(dataset, var_names, index_name, divisions_to_plot)
                
                # perform analysis on each division, writing results back into NetCDF, with plots of specified divisions
                _divisional_analysis_plots(dataset,
                                           index_name,
                                           var_names,
                                           diff_name_prefix,
                                           rmse_name_prefix,
                                           signchange_name_prefix,
                                           divisions_to_plot)

        # report on the elapsed time
        end_datetime = datetime.datetime.now()
        _logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        _logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        _logger.exception('Failed to complete', exc_info=True)
        raise
    