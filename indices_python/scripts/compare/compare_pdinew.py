import argparse
import logging
import netCDF4
import numpy as np

from indices_python import pdinew

"""
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
def _rmse(predictions, targets):
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
    '''

    try:

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_file", 
                            help="Input dataset file (NetCDF) containing temperature, precipitation, and soil values for PDSI, SPI, SPEI, and PNP computations", 
                            required=True)
        parser.add_argument("--precip_var_name", 
                            help="Precipitation variable name used in the input NetCDF file", 
                            required=True)
        parser.add_argument("--temp_var_name", 
                            help="Temperature variable name used in the input NetCDF file", 
                            required=True)
        parser.add_argument("--awc_var_name", 
                            help="Available water capacity variable name used in the input NetCDF file", 
                            required=False)
        parser.add_argument("--output_dir", 
                            help="Directory where result plot image files will be stored", 
                            required=True)
        args = parser.parse_args()

        # open the NetCDF files 
        with netCDF4.Dataset(args.input_file) as input_dataset:

            # get the division IDs as a list
            division_ids = list(input_dataset.variables['division'][:])
            
            # read the temperature, precipitation, latitude and AWC for each division
            for division_index, division_id in enumerate(division_ids):
        
#                 # DEBUG ONLY -- RMEOVE
#                 if division_id < 1302:
#                     continue
                
                # FOR DEBUG/DEVELOPMENT ONLY -- REMOVE
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
                
                # calibration period years used operationally with pdinew.f
                calibration_begin_year = 1931
                calibration_end_year = 1990
 
                # compare the results of the pdinew.f (monthly NCEI results) against the values  
                # computed by the corresponding new Palmer implementation based on Jacobi 2013
                                  
                #TODO get these values out of the NetCDF, compute from time values, etc.                        
                data_begin_year = 1895


#                 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                 # Water balance accounting
#                 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                 # water balance intermediate values
#                 spdat = input_dataset.variables['spdat'][division_index, :]
#                 pedat = input_dataset.variables['pedat'][division_index, :]
#                 pldat = input_dataset.variables['pldat'][division_index, :]
#                 prdat = input_dataset.variables['prdat'][division_index, :]
#                 rdat = input_dataset.variables['rdat'][division_index, :]
#                 tldat = input_dataset.variables['tldat'][division_index, :]
#                 etdat = input_dataset.variables['etdat'][division_index, :]
#                 rodat = input_dataset.variables['rodat'][division_index, :]
#                 tdat = input_dataset.variables['tdat'][division_index, :]
#                 sssdat = input_dataset.variables['sssdat'][division_index, :]
#                 ssudat = input_dataset.variables['ssudat'][division_index, :]

#                 # calculate the negative tangent of the latitude which is used as an argument to the water balance function
#                 neg_tan_lat = -1 * math.tan(math.radians(latitude))
#     
#                 # compute water balance values using the function translated from the Fortran pdinew.f
#                 #NOTE keep this code in place in order to compute the PET used later, since the two have 
#                 # different PET algorithms and we want to compare PDSI using the same PET inputs 
#                 pdinew_pdat, pdinew_spdat, pdinew_pedat, pdinew_pldat, pdinew_prdat, pdinew_rdat, pdinew_tldat, \
#                     pdinew_etdat, pdinew_rodat, pdinew_prodat, pdinew_tdat, pdinew_sssdat, pdinew_ssudat = \
#                         pdinew._water_balance(temp_timeseries, precip_timeseries, awc + 1.0, neg_tan_lat, B, H)
# #                         pdinew._water_balance(temp_timeseries, precip_timeseries, awc, neg_tan_lat, B, H)
#                          
#                 # compare the values against the operational values produced monthly by pdinew.f
#                 spdat_diffs = spdat - pdinew_spdat.flatten()
#                 pedat_diffs = pedat - pdinew_pedat.flatten()
#                 pldat_diffs = pldat - pdinew_pldat.flatten()
#                 prdat_diffs = prdat - pdinew_prdat.flatten()
#                 rdat_diffs = rdat - pdinew_rdat.flatten()
#                 tldat_diffs = tldat - pdinew_tldat.flatten()
#                 etdat_diffs = etdat - pdinew_etdat.flatten()
#                 rodat_diffs = rodat - pdinew_rodat.flatten()
#                 tdat_diffs = tdat - pdinew_tdat.flatten()
#                 sssdat_diffs = sssdat - pdinew_sssdat.flatten()
#                 ssudat_diffs = ssudat - pdinew_ssudat.flatten()
#     
#                 # dictionary of variable names to corresponding arrays of differences                
#                 varnames_to_arrays = {'SP': (spdat_diffs, spdat, pdinew_spdat),
#                                       'PE': (pedat_diffs, pedat, pdinew_pedat),
#                                       'PL': (pldat_diffs, pldat, pdinew_pldat),
#                                       'PR': (prdat_diffs, prdat, pdinew_prdat),
#                                       'R': (rdat_diffs, rdat, pdinew_rdat),
#                                       'TL': (tldat_diffs, tldat, pdinew_tldat),
#                                       'ET': (etdat_diffs, etdat, pdinew_etdat),
#                                       'RO': (rodat_diffs, rodat, pdinew_rodat),
#                                       'T': (tdat_diffs, tdat, pdinew_tdat),
#                                       'SSS': (sssdat_diffs, sssdat, pdinew_sssdat),
#                                       'SSU': (ssudat_diffs, ssudat, pdinew_ssudat), }
#     
#                 # we want to see all zero differences, if any non-zero differences exist then raise an alert
#                 for varname, array_tuple in varnames_to_arrays.items():
# 
#                     print('Plotting differences for variable: {0}'.format(varname))
#                     
#                     diffs = array_tuple[0]
#                     expected = array_tuple[1]
#                     actual = array_tuple[2].flatten()
#                         
#                     _plot_diffs(expected,
#                                actual,
#                                division_id,
#                                varname)
 
    
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # PDSI and Z-Index
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
                # compute PDSI etc. using PDSI code translated from pdinew.f
                pdinew_PDSI, pdinew_PHDI, pdinew_PMDI, pdinew_Z, PET = pdinew.pdsi_from_climatology(precip_timeseries,
                                                                                                    temp_timeseries,
                                                                                                    awc,
                                                                                                    latitude,
                                                                                                    B,
                                                                                                    H,
                                                                                                    data_begin_year,
                                                                                                    calibration_begin_year,
                                                                                                    calibration_end_year)
                    
                # dictionary of variable names to corresponding arrays of differences to facilitate looping below
                varnames_to_arrays = {'pdinew_PDSI': (expected_pdsi, pdinew_PDSI.flatten()),
                                      'pdinew_PHDI': (expected_phdi, pdinew_PHDI),
                                      'pdinew_PMDI': (expected_pmdi, pdinew_PMDI),
                                      'pdinew_Z-INDEX': (expected_zindex, pdinew_Z.flatten()) }
    
                # we want to see all zero differences, if any non-zero differences exist then raise an alert
                for varname, array_tuple in varnames_to_arrays.items():
                        
                    print('Plotting differences for variable: {0}'.format(varname))

                    expected = array_tuple[0]
                    actual = array_tuple[1]
                        
                    _plot_diffs(expected,
                                actual,
                                division_id,
                                varname,
                                args.output_dir)
 
    except Exception:
        logger.exception('Failed to complete', exc_info=True)
        raise
    
#-----------------------------------------------------------------------------------------------------------------------
def _plot_diffs(expected,
                actual,
                division_id,
                varname,
                output_dir):

    diffs = expected - actual     
    error = _rmse(actual, expected)
    
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
#     plt.show()
    plt.savefig(output_dir + '/compare_cmb_vs_{0}_{1}.png'.format(varname, division_id))
    plt.close()

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    
