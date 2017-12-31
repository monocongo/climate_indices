import argparse
import datetime
import logging
import netCDF4
import numpy as np
import palmer
import utils

"""
Code to perform indices processing on climate divisions datasets in NetCDF format, then computing differences 
of the computed indices data with monthly datasets computed operationally by NCEI. The differences are plotted 
as line graphs (including both expected and actual), differences histograms, and saved as difference variables 
within the NetCDF provided as input.
Example usage:

python -u <this_script> --input_file C:/home/data/nclimdiv/climdiv-climdv-v1.0.0-20170906.nc \
                        --precip_var_name prcp \
                        --temp_var_name tavg \
                        --awc_var_name awc
"""
#-----------------------------------------------------------------------------------------------------------------------
# set up matplotlib to use the Agg backend, in order to remove any dependencies on an X server,
# for example to get around the ImportError: cannot import name 'QtCore'
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
# absolute tolerance used for comparison of values, original values compared against are in hundredths
_TOLERANCE = 0.05

#-----------------------------------------------------------------------------------------------------------------------
def rmse(predictions, targets):
    """
    Root mean square error
    
    :param predictions: np.ndarray
    :param targets: np.ndarray
    :return: np.ndarray
    """
    return np.sqrt(((predictions - targets) ** 2).mean())

#-----------------------------------------------------------------------------------------------------------------------
def main():

    '''
    Run with arguments like this:
    
    --input_file C:/home/data/nclimdiv/nclimdiv-v1.0.0-20170707.nc    
    --precip_var_name pdat 
    --temp_var_name tdat 
    --awc_var_name awc 
    --output_dir C:/home/data/nclimdiv/compare_palmer

    '''

    try:

#         # FOR DEBUG/DEVELOPMENT ONLY -- REMOVE
#         _limit_counter = 0
#         _LIMIT = 10
        
        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_file", 
                            help="Input dataset file (NetCDF) containing temperature, precipitation, and soil values for palmer_PDSI, SPI, SPEI, and PNP computations", 
                            required=True)
        parser.add_argument("--precip_var_name", 
                            help="Precipitation variable name used in the input NetCDF file", 
                            required=True)
        parser.add_argument("--temp_var_name", 
                            help="Temperature variable name used in the input NetCDF file", 
                            required=True)
        parser.add_argument("--awc_var_name", 
                            help="Available water capacity variable name used in the input NetCDF file", 
                            required=True)
        parser.add_argument("--output_dir", 
                            help="Directory where result plot image files will be stored", 
                            required=True)
        args = parser.parse_args()

        #TODO replace below with CL arguments
        # calibration period years used operationally with pdinew.f
        calibration_begin_year = 1931
        calibration_end_year = 1990
        
        #TODO get this value from the NetCDF, compute from time values, etc.                        
        data_begin_year = 1895

        # open the NetCDF files 
        with netCDF4.Dataset(args.input_file) as input_dataset:

            # get the division IDs as a list
            division_ids = list(input_dataset.variables['division'][:])
            
            # read the temperature, precipitation, latitude and AWC for each division
            for division_index, division_id in enumerate(division_ids):
        
#                 # DEBUG ONLY -- RMEOVE
#                 if division_id > 102:
#                     break
                
#                 # FOR DEBUG/DEVELOPMENT ONLY -- REMOVE
                print('\n\n======================================================================\nDivision ID: {0}\n'.format(division_id))
                    
                # get the data for this division
                precip_timeseries = input_dataset.variables[args.precip_var_name][division_index, :]
                temp_timeseries = input_dataset.variables[args.temp_var_name][division_index, :]
                awc = input_dataset.variables[args.awc_var_name][division_index]
                latitude = input_dataset.variables['lat'][division_index]
                B = input_dataset.variables['B'][division_index]
                H = input_dataset.variables['H'][division_index]

                # get the expected/target values from the NetCDF
                expected_pdsi = input_dataset.variables['pdsi.index'][division_index, :]
                expected_phdi = input_dataset.variables['phdi.index'][division_index, :]
                expected_pmdi = input_dataset.variables['pmdi.index'][division_index, :]
                expected_zindex = input_dataset.variables['z.index'][division_index, :]
                
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # compute PDSI etc. using new Palmer code translated from Jacobi et al MatLab
                nidis_pdsi, nidis_phdi, nidis_pmdi, nidis_zindex = palmer.pdsi_from_climatology(precip_timeseries,
                                                                                                temp_timeseries,
                                                                                                awc + 1.0,  # original AWC value is underlying layer only, top layer is one inch so we add it here
                                                                                                latitude,
                                                                                                data_begin_year,
                                                                                                calibration_begin_year,
                                                                                                calibration_end_year,
                                                                                                B,
                                                                                                H)
 
                # dictionary of variable names to corresponding arrays of differences to facilitate looping below
                #TODO is nidis_*.flatten() necessary below?
                varnames_to_arrays = {'nidis_pdsi': (expected_pdsi, nidis_pdsi.flatten()),
                                      'nidis_phdi': (expected_phdi, nidis_phdi.flatten()),
                                      'nidis_pmdi': (expected_pmdi, nidis_pmdi.flatten()),
                                      'nidis_zindex': (expected_zindex, nidis_zindex.flatten()) }
    
                # we want to see all zero differences, if any non-zero differences exist then raise an alert
                for varname, array_tuple in varnames_to_arrays.items():
                        
                    print('Plotting differences for variable: {0}'.format(varname))

                    # overall RMSE for division
                    total_rmse = 0
                    
                    # loop over each calendar month
                    for i in range(0, 12):
                        
                        month = datetime.date(1900, i + 1, 1).strftime('%B')
                        
                        print('\tMonth: {0}'.format(month))
    
                        expected = array_tuple[0][i::12]
                        actual = array_tuple[1][i::12]
     
                        # get the RMSE
                        rmse = utils.rmse(expected, actual)
                        
                        total_rmse += rmse
                        
                        # plot the two data arrays and the differences                       
                        plot_diffs(expected,
                                   actual,
                                   division_id,
                                   varname,
                                   args.output_dir)
 
                    # average the monthly RMSE values
                    total_rmse /= 12
                    
    except Exception:
        logger.exception('Failed to complete', exc_info=True)
        raise
    
#-----------------------------------------------------------------------------------------------------------------------
def plot_diffs(expected,
               actual,
               division_id,
               varname,
               output_dir):

    diffs = expected - actual     
    error = rmse(actual, expected)
    
    # plot the values and differences
    x = np.arange(diffs.size)
    
    # set the Y-limit to range from -5 to 5, since this is the approximate range of values
    ax = plt.axes()
    ax.set_ylim([-5, 5])

    # the X values will correspond to a horizontal line axis
    plt.axhline()
    
    # set the size of the plot to be larger/wider than the default
    plt.rcParams["figure.figsize"] = [48, 8]
    
    # plot the lines, legend, title, labels, etc.
    expected_line, = plt.plot(x, expected, color='blue', label='NCEI (expected)')
    actual_line, = plt.plot(x, actual, color='yellow', linestyle='--', label='NIDIS (actual)')
    diffs_line, = plt.plot(x, diffs, color='red', label='Difference')
    plt.legend(handles=[expected_line, actual_line, diffs_line], loc='upper left')
    plt.title('Comparison for division {0}: {1}     (RMSE: {2})'.format(division_id, varname, error))
    plt.xlabel("months")
    plt.ylabel("value")
#     plt.show()

    # save the plot to file
    plt.savefig(output_dir + '/compare_{0}_{1}_origpet.png'.format(varname, division_id))
    
    plt.close()

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    
