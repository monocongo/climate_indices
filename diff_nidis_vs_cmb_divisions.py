import compute
import datetime
import logging
import netCDF4
import netcdf_utils
import numpy as np
import sys

#-----------------------------------------------------------------------------------------------------------------------
# set up matplotlib to use the Agg backend, in order to remove any dependencies on an X server
import matplotlib
import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

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
        vars = { 'ro66': 'wb_RO',
                 'pe60': 'pet',
                 'pl61': 'wb_PL',
                 'pr62': 'wb_PR',
                 'et63': 'wb_ET',
                 're65': 'wb_R',
                 'sl67': 'wb_L'}
        for var_CMB, var_NIDIS in vars.items():
                    
            # TODO validate that the two variables exist, have compatible dimensions/units, etc., all of which is assumed below

            # create a variable that'll be used to write into the output NetCDF
            var_name = var_CMB + '_minus_' + var_NIDIS
            
            logger.info('Computing differences on variables {0} (NIDIS/indices.py) and {1} (CMB)'.format(var_CMB, var_NIDIS))
            
            # get the variable values for the month, mask the NaNs (data assumed to be in (division, time) dimension order)
            data_CMB = np.ma.masked_invalid(dataset_CMB.variables[var_CMB][:, :], copy=False)
            data_NIDIS = np.ma.masked_invalid(dataset_NIDIS.variables[var_NIDIS][:, :], copy=False)
    
            # get the difference of the two
            diffs = data_CMB - data_NIDIS
    
            # add the variable into the dataset and add the data into the variable
            dataset_OUT = netcdf_utils.add_variable_climdivs(dataset_OUT, var_name, dataset_NIDIS.variables[var_NIDIS].__dict__)
            expected_shape = dataset_OUT.variables[var_name][:].shape
#             dataset_OUT.variables[var_name][:] = compute.reshape_to_divs_months(diffs, dataset_NIDIS.variables[divisions_dim_name][:].size)
#             dataset_OUT.variables[var_name][:] = np.swapaxes(diffs,0,1)
            dataset_OUT.variables[var_name][:] = diffs
            
            # get just the unmasked values, as a 1-D (flattened) array
            diffs = diffs[~diffs.mask]

            # plot the differences on a monthly basis
    
            # common title
            histogram_title = 'CMB vs. NIDIS: '

            # loop over each month
            for i in range(12):
    
                # get the month as a string
                month = datetime.datetime.strptime(str(i + 1), "%m").strftime("%b")
    
                # plot a histogram of the differences
                plt.gcf().clear()
                plt.hist(diffs, bins=number_of_bins, range=(range_lower, range_upper))
                plt.title(histogram_title + ': {0}/{1}'.format(var_name, month))
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                file_name = 'C:/home/climdivs/diff_histograms/cmb_{0}_minus_nidis_{1}_'.format(var_CMB, var_NIDIS) + str(i + 1).zfill(2) + '.png'
                logger.info('Saving plot for month {0} as file {1}'.format(i + 1, file_name))
                plt.savefig(file_name)
