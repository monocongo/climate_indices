import argparse
import logging
import math
import netCDF4
import numpy as np
import os
import palmer
import pdinew

"""
Example usage:

python -u <this_script> --input_file C:/home/data/nclimdiv/climdiv-climdv-v1.0.0-20170906.nc \
                        --precip_var_name prcp \
                        --temp_var_name tavg \
                        --awc_var_name awc \
                        --output_dir C:/home/climdivs/pdi_differences
"""
#-----------------------------------------------------------------------------------------------------------------------
# set up matplotlib to use the Agg backend, in order to remove any dependencies on an X server,
# for example to get around the ImportError: cannot import name 'QtCore'
import matplotlib
#matplotlib.use('Agg')
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
                            help="Output directory for monthly differences line graph plots", 
                            required=True)
        args = parser.parse_args()

        # open the NetCDF files 
        with netCDF4.Dataset(args.input_file) as input_dataset:

            # get the division IDs as a list
            division_ids = list(input_dataset.variables['division'][:])
            
            # create a dictionary containing division IDs as keys and average differences as values
            divisions_to_differences = dict.fromkeys(division_ids)
                               
            # read the temperature, precipitation, latitude and AWC for each division
            for division_index, division_id in enumerate(division_ids):
        
#                 # DEBUG ONLY -- RMEOVE
#                 divs = [101, 205, 1405, 3002, 4501]
#                 if division_id not in divs:
#                     continue
                
#                 # FOR DEBUG/DEVELOPMENT ONLY -- REMOVE
                print('\n\n======================================================================\nDivision ID: {0}\n'.format(division_id))
#                 if _limit_counter > _LIMIT:
#                     break
#                 _limit_counter += 1
                    
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
                data_end_year = 2017

#                 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                 # compute palmer_PDSI etc. using new palmer_PDSI code translated from Jacobi et al MatLab code
#                 palmer_PDSI, palmer_PHDI, palmer_PMDI, palmer_Z = palmer.pdi_from_climatology(precip_timeseries,
#                                                                                               temp_timeseries,
#                                                                                               awc,
#                                                                                               latitude,
#                                                                                               data_begin_year,
#                                                                                               calibration_begin_year,
#                                                                                               calibration_end_year,
#                                                                                               expected_pdsi,
#                                                                                               B,
#                                                                                               H)
#  
#                 # find the differences between the new (Matlab-derived) and NCEI results
#                 pdsi_diffs = palmer_PDSI.flatten() - expected_pdsi
# #                 phdi_diffs = palmer_PHDI.flatten() - expected_phdi
# #                 pmdi_diffs = palmer_PMDI.flatten() - expected_pmdi
#                 zindex_diffs = palmer_Z.flatten() - expected_zindex
#                 
#                 # dictionary of variable names to corresponding arrays of differences to facilitate looping below
#                 varnames_to_arrays = {'palmer_PDSI': (pdsi_diffs, expected_pdsi, palmer_PDSI.flatten()),
# #                                       'palmer_PHDI': (phdi_diffs, expected_phdi, pdinew_PHDI),
# #                                       'palmer_PMDI': (pmdi_diffs, expected_pmdi, pdinew_PMDI),
#                                       'palmer_Z-INDEX': (zindex_diffs, expected_zindex, palmer_Z.flatten()) }
#     
#                 # we want to see all zero differences, if any non-zero differences exist then raise an alert
#                 zeros = np.zeros(pdsi_diffs.shape)
#                 for varname, array_tuple in varnames_to_arrays.items():
#                         
#                     diffs = array_tuple[0]
#                     expected = array_tuple[1]
#                     actual = array_tuple[2]
#  
#                     # plot the two data arrays and the differences                       
#                     plot_diffs(expected,
#                                actual,
#                                division_id,
#                                varname,
#                                args.output_dir)
#  
#                     # report if we see any significant differences
#                     if not np.allclose(diffs, zeros, atol=_TOLERANCE, equal_nan=True):
#                         logger.warn('Division {0}: Comparing new Palmer (pdinew.py) against '.format(division_id) + \
#                                     'operational pdinew.f: \nNon-matching values for {0}'.format(varname))
#                         offending_indices = np.where(abs(diffs) > _TOLERANCE)
# #                         non_offending_indices = np.where(abs(diffs) <= _TOLERANCE)
#                         nan_indices = np.where(actual is np.NaN)
#                         logger.warn('Time steps with NaN ({0}): {1}'.format(np.isnan(actual).sum(), nan_indices))
#                         logger.warn('Time steps with significant differences ({0}): {1}'.format(len(offending_indices[0]), offending_indices[0])) 
#                            
# #                         for i in offending_indices[0]:
# #                             
# #                             print('{0}  Expected:  {1}   Actual: {2}'.format(i, expected[i], actual[i]))
#                 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
                # calculate the negative tangent of the latitude which is used as an argument to the water balance function
                neg_tan_lat = -1 * math.tan(math.radians(latitude))
    
                # compute water balance values using the function translated from the Fortran pdinew.f
                #NOTE keep this code in place in order to compute the PET used later, since the two have 
                # different PET algorithms and we want to compare palmer_PDSI using the same PET inputs 
#                 pdinew_pdat, pdinew_spdat, pdinew_pedat, pdinew_pldat, pdinew_prdat, pdinew_rdat, pdinew_tldat, \
                pdinew_pdat, pdinew_pedat, pdinew_pldat, pdinew_prdat, pdinew_rdat, pdinew_tldat, \
                    pdinew_etdat, pdinew_rodat, pdinew_prodat, pdinew_tdat, pdinew_sssdat, pdinew_ssudat = \
                        pdinew._water_balance(temp_timeseries, precip_timeseries, awc, neg_tan_lat, B, H)
                         
                # compare the values against the operational values produced monthly by pdinew.f
                prodat_diffs = input_dataset.variables['spdat'][division_index, :] - pdinew_prodat.flatten()  # PRO and SP are synonymous in pdinew.f
                pedat_diffs = input_dataset.variables['pedat'][division_index, :] - pdinew_pedat.flatten()
                pldat_diffs = input_dataset.variables['pldat'][division_index, :] - pdinew_pldat.flatten()
                prdat_diffs = input_dataset.variables['prdat'][division_index, :] - pdinew_prdat.flatten()
                rdat_diffs = input_dataset.variables['rdat'][division_index, :] - pdinew_rdat.flatten()
                tldat_diffs = input_dataset.variables['tldat'][division_index, :] - pdinew_tldat.flatten()
                etdat_diffs = input_dataset.variables['etdat'][division_index, :] - pdinew_etdat.flatten()
                rodat_diffs = input_dataset.variables['rodat'][division_index, :] - pdinew_rodat.flatten()
                tdat_diffs = input_dataset.variables['tdat'][division_index, :] - pdinew_tdat.flatten()
                sssdat_diffs = input_dataset.variables['sssdat'][division_index, :] - pdinew_sssdat.flatten()
                ssudat_diffs = input_dataset.variables['ssudat'][division_index, :] - pdinew_ssudat.flatten()
    
                # dictionary of variable names to corresponding arrays of differences                
                varnames_to_arrays = {'PRO': prodat_diffs,
                                      'PE': pedat_diffs,
                                      'PL': pldat_diffs,
                                      'PR': prdat_diffs,
                                      'R': rdat_diffs,
                                      'TL': tldat_diffs,
                                      'ET': etdat_diffs,
                                      'RO': rodat_diffs,
                                      'T': tdat_diffs,
                                      'SSS': sssdat_diffs,
                                      'SSU': ssudat_diffs }
    
                # we want to see all zero differences, if any non-zero differences exist then raise an alert
                zeros = np.zeros(prodat_diffs.shape)
                for varname, diffs in varnames_to_arrays.items():
                    if not np.allclose(diffs, zeros, atol=_TOLERANCE, equal_nan=True):
                        logger.warn('Division {0}: Comparing pdinew.py against operational pdinew.f water balance: '.format(division_id) + \
                                    '\nNon-matching difference arrays for water balance variable: {0}'.format(varname))
                        offending_indices = np.where(abs(diffs) > _TOLERANCE)
                        logger.warn('Indices with significant differences: {0}'.format(offending_indices))
 
                # convert monthly temperatures from Fahrenheit to Celsius
                monthly_temps_celsius = (temp_timeseries - 32) * 5.0 / 9.0
 
                # compute PE using version from original Fortran
                palmer_PE = palmer._pdinew_potential_evapotranspiration(monthly_temps_celsius, 
                                                                        latitude,
                                                                        data_begin_year,
                                                                        B,
                                                                        H)
                 
                # we want to see all zero differences, if any non-zero differences exist then raise an alert
                diffs = input_dataset.variables['pedat'][division_index, :] - palmer_PE
                if not np.allclose(diffs, zeros, atol=_TOLERANCE, equal_nan=True):
                    logger.warn('Division {0}: Comparing pdinew.py against operational pdinew.f PET: '.format(division_id) + \
                                '\nNon-matching difference arrays for PET')
                    offending_indices = np.where(abs(diffs) > _TOLERANCE)
                    logger.warn('Indices with significant differences ({0}): {1}'.format(offending_indices[0].size, offending_indices[0]))
 
                # compute the water balance values using the new Python version derived from Jacobi et al Matlab palmer_PDSI
                palmer_ET, palmer_PR, palmer_R, palmer_RO, palmer_PRO, palmer_L, palmer_PL = \
                    palmer._water_balance(awc + 1.0, palmer_PE, precip_timeseries)
 
                    
                # compare the values against the values produced monthly by pdinew.py
                etdat_diffs = input_dataset.variables['etdat'][division_index, :] - palmer_ET
                tldat_diffs = input_dataset.variables['tldat'][division_index, :] - palmer_L
                pldat_diffs = input_dataset.variables['pldat'][division_index, :] - palmer_PL
                rdat_diffs = input_dataset.variables['rdat'][division_index, :] - palmer_R
                prdat_diffs = input_dataset.variables['prdat'][division_index, :] - palmer_PR
                rodat_diffs = input_dataset.variables['rodat'][division_index, :] - palmer_RO
    
                # dictionary of variable names to corresponding arrays of differences                
                varnames_to_arrays = {'palmer_PR': prdat_diffs,
                                      'palmer_R': rdat_diffs,
                                      'palmer_PL': pldat_diffs,
                                      'palmer_L': tldat_diffs,
                                      'palmer_ET': etdat_diffs,
                                      'palmer_RO': rodat_diffs}
    
                # we want to see all zero differences, if any non-zero differences exist then raise an alert
                zeros = np.zeros(pldat_diffs.shape)
                for varname, diffs in varnames_to_arrays.items():
                    if not np.allclose(diffs, zeros, atol=_TOLERANCE, equal_nan=True):
                        logger.warn('Division {0}: Comparing pdinew.py against operational pdinew.f water balance: '.
                                    format(division_id) + \
                                    '\nNon-matching difference arrays for water balance variable: {0}'.format(varname))
                        offending_indices = np.where(abs(diffs) > _TOLERANCE)
                        logger.warn('Indices with significant differences: {0}'.format(offending_indices[0]))
                            
                # compare the values against the operational values produced monthly by NCEI (assumed to be from pdinew.f)
                etdat_wb_diffs = input_dataset.variables['etdat'][division_index, :] - palmer_ET
                prdat_wb_diffs = input_dataset.variables['prdat'][division_index, :] - palmer_PR
                rdat_wb_diffs = input_dataset.variables['rdat'][division_index, :] - palmer_R
                rodat_wb_diffs = input_dataset.variables['rodat'][division_index, :] - palmer_RO
                ldat_wb_diffs = input_dataset.variables['tldat'][division_index, :] - palmer_L
                pldat_wb_diffs = input_dataset.variables['pldat'][division_index, :] - palmer_PL
                   
                # compare the differences of the two, these difference arrays should come out to all zeros
                et_diffs = etdat_wb_diffs - etdat_diffs
                pr_diffs = prdat_wb_diffs - prdat_diffs
                r_diffs = rdat_wb_diffs - rdat_diffs
                ro_diffs = rodat_wb_diffs - rodat_diffs
                l_diffs = ldat_wb_diffs - tldat_diffs
                pl_diffs = pldat_wb_diffs - pldat_diffs
                   
                # dictionary of variable names to corresponding arrays of differences                
                varnames_to_arrays = {'palmer_PL': pl_diffs,
                                      'palmer_PR': pr_diffs,
                                      'palmer_R': r_diffs,
                                      'palmer_L': l_diffs,
                                      'palmer_ET': et_diffs,
                                      'palmer_RO': rodat_diffs }
   
                # we want to see all zero differences, if any non-zero differences exist then raise an alert
                zeros = np.zeros(pl_diffs.shape)
                for varname, diffs in varnames_to_arrays.items():
                    if not np.allclose(diffs, zeros, atol=_TOLERANCE, equal_nan=True):
                        logger.warn('Division {0}: Comparing palmer.py against operational pdinew.f: '.format(division_id) + \
                                    '\nNon-matching difference arrays for water balance variable: {0}'.format(varname))
                        offending_indices = np.where(abs(diffs) > _TOLERANCE)
                        logger.warn('Indices with significant differences: {0}'.format(offending_indices))
 
                #NOTE we need to compute CAFEC coefficients for use later/below
                # compute palmer_PDSI etc. using translated functions from pdinew.f Fortran code
                pdinew_alpha, pdinew_beta, pdinew_delta, pdinew_gamma, t_ratio = pdinew._cafec_coefficients(precip_timeseries,
                                                                                                            pdinew_pedat,
                                                                                                            pdinew_etdat,
                                                                                                            pdinew_prdat,
                                                                                                            pdinew_rdat,
                                                                                                            pdinew_rodat,
                                                                                                            pdinew_prodat,
                                                                                                            pdinew_tldat,
                                                                                                            pdinew_pldat,
                                                                                                            data_begin_year,
                                                                                                            calibration_begin_year,
                                                                                                            calibration_end_year)
                    
                # compute the coefficients using the new function   palmer_ET, palmer_PR, palmer_R, palmer_RO, palmer_PRO, palmer_L, palmer_PL
                palmer_alpha, palmer_beta, palmer_gamma, palmer_delta = palmer._cafec_coefficients(precip_timeseries,
                                                                                                   palmer_PE,
                                                                                                   palmer_ET,
                                                                                                   palmer_PR,
                                                                                                   palmer_R,
                                                                                                   palmer_RO,
                                                                                                   palmer_PRO,
                                                                                                   palmer_L,
                                                                                                   palmer_PL,
                                                                                                   data_begin_year,
                                                                                                   calibration_begin_year,
                                                                                                   calibration_end_year)
 
                # look at the differences between the results of the old and new versions of the coefficients code                
                alpha_diffs = pdinew_alpha - palmer_alpha
                beta_diffs = pdinew_beta - palmer_beta
                gamma_diffs = pdinew_gamma - palmer_gamma
                delta_diffs = pdinew_delta - palmer_delta
                   
                # dictionary of variable names to corresponding arrays of differences                
                varnames_to_arrays = {'Alpha': alpha_diffs,
                                      'Beta': beta_diffs,
                                      'Gamma': gamma_diffs,
                                      'Delta': delta_diffs }
   
                # we want to see all zero differences, if any non-zero differences exist then raise an alert
                zeros = np.zeros(alpha_diffs.shape)
                for varname, diffs in varnames_to_arrays.items():
                    if not np.allclose(diffs, zeros, atol=_TOLERANCE, equal_nan=True):
                        logger.warn('Division {0}: Comparing new Palmer against operational pdinew.f: ' + \
                                    '\nNon-matching difference arrays for CAFEC coefficient: {1}'.format(division_id, varname))
                        offending_indices = np.where(abs(diffs) > _TOLERANCE)
                        logger.warn('Indices with significant differences: {0}'.format(offending_indices[0]))
   
                # compute the weighting factor (climatic characteristic) using the new version
                palmer_K = palmer._climatic_characteristic(palmer_alpha,
                                                           palmer_beta,
                                                           palmer_gamma,
                                                           palmer_delta,
                                                           precip_timeseries,
                                                           palmer_ET,
                                                           palmer_PE,
                                                           palmer_R,
                                                           palmer_PR,
                                                           palmer_RO,
                                                           palmer_PRO,
                                                           palmer_L,
                                                           palmer_PL,
                                                           data_begin_year,
                                                           calibration_begin_year,
                                                           calibration_end_year)
   
                # compute the weighting factor (climatic characteristic) using the version translated from pdinew.f
                pdinew_K = pdinew._climatic_characteristic(pdinew_alpha,
                                                           pdinew_beta,
                                                           pdinew_gamma,
                                                           pdinew_delta,
                                                           pdinew_pdat,
                                                           pdinew_pedat,
                                                           pdinew_prdat,
                                                           pdinew_prodat,
                                                           pdinew_pldat,
                                                           t_ratio,
                                                           data_begin_year,
                                                           calibration_begin_year,
                                                           calibration_end_year)
                   
                # look at the differences of the climatic characteristic results from the two implementations
                diffs = palmer_K - pdinew_K        
                zeros = np.zeros(diffs.shape)
                if not np.allclose(diffs, zeros, atol=_TOLERANCE, equal_nan=True):
                    logger.warn('Division {0}: Non-matching difference arrays for climatic characteristic: {1}'.format(division_id, varname))
                    offending_indices = np.where(abs(diffs) > _TOLERANCE)
                    #logger.warn('Time steps with significant differences: {0}'.format(offending_indices))
                    for i in offending_indices[0]:
                        print('{0}  Expected/NCEI:  {1}   Actual/NIDIS: {2}'.format(i, pdinew_K[i], palmer_K[i]))
    
                # compute the palmer_Z-Index using the new (Matlab derived) version
                palmer_Z = palmer._z_index(precip_timeseries,
                                           palmer_PE,
                                           palmer_ET,
                                           palmer_PR,
                                           palmer_R,
                                           palmer_RO,
                                           palmer_PRO,
                                           palmer_L,
                                           palmer_PL,
                                           data_begin_year,
                                           calibration_begin_year,
                                           calibration_end_year)
 
                pdinew_Z = pdinew._zindex_from_climatology(temp_timeseries, 
                                                           precip_timeseries, 
                                                           awc, 
                                                           neg_tan_lat, 
                                                           B, 
                                                           H,
                                                           data_begin_year,
                                                           calibration_begin_year,
                                                           calibration_end_year)

#                 # plot the values and differences of the two Z-Index results                 
#                 plot_diffs(pdinew_Z.flatten(),
#                            expected_zindex,
#                            division_id,
#                            'pdinew Z-Index',
#                            args.output_dir)
                 
#                 # compute palmer_PDSI etc. using palmer_PDSI code translated from pdinew.f
#                 pdinew_PDSI, pdinew_PHDI, pdinew_PMDI, pdinew_Z, PET = pdinew.pdsi_from_climatology(precip_timeseries,
#                                                                                                     temp_timeseries,
#                                                                                                     awc,
#                                                                                                     latitude,
#                                                                                                     B,
#                                                                                                     H,
#                                                                                                     data_begin_year,
#                                                                                                     calibration_begin_year,
#                                                                                                     calibration_end_year,
#                                                                                                     expected_pdsi)
#                     
#                 # find the differences between the new (Matlab-derived) and previous (Fortran-derived) versions
#                 pdsi_diffs = pdinew_PDSI.flatten() - expected_pdsi
#                 phdi_diffs = pdinew_PHDI.flatten() - expected_phdi
#                 pmdi_diffs = pdinew_PMDI.flatten() - expected_pmdi
#                 zindex_diffs = pdinew_Z.flatten() - expected_zindex
#                 
#                 # dictionary of variable names to corresponding arrays of differences to facilitate looping below
#                 varnames_to_arrays = {'pdinew_PDSI': (pdsi_diffs, expected_pdsi, pdinew_PDSI.flatten()),
#                                       'pdinew_PHDI': (phdi_diffs, expected_phdi, pdinew_PHDI.flatten()),
#                                       'pdinew_PMDI': (pmdi_diffs, expected_pmdi, pdinew_PMDI.flatten()),
#                                       'pdinew_Z-INDEX': (zindex_diffs, expected_zindex, pdinew_Z.flatten()) }
#     
#                 # we want to see all zero differences, if any non-zero differences exist then raise an alert
#                 zeros = np.zeros(pdsi_diffs.shape)
#                 for varname, array_tuple in varnames_to_arrays.items():
#                         
#                     diffs = array_tuple[0]
#                     expected = array_tuple[1]
#                     actual = array_tuple[2]
#                         
#                     plot_diffs(expected,
#                                actual,
#                                division_id,
#                                varname,
#                                args.output_dir)
#  
#                     if not np.allclose(diffs, zeros, atol=_TOLERANCE, equal_nan=True):
#                         logger.warn('Division {0}: Comparing new Palmer (pdinew.py) against '.format(division_id) + \
#                                     'operational pdinew.f: \nNon-matching values for {0}'.format(varname))
#                         offending_indices = np.where(abs(diffs) > _TOLERANCE)
# #                         non_offending_indices = np.where(abs(diffs) <= _TOLERANCE)
#                         nan_indices = np.where(actual is np.NaN)
#                         logger.warn('Time steps with NaN ({0}): {1}'.format(np.isnan(actual).sum(), nan_indices))
#                         logger.warn('Time steps with significant differences ({0}): {1}'.format(len(offending_indices[0]), offending_indices[0])) 
#                            
# #                         for i in offending_indices[0]:
# #                             
# #                             print('{0}  Expected:  {1}   Actual: {2}'.format(i, expected[i], actual[i]))
   
#                 # compute palmer_PDSI etc. using new palmer_PDSI code translated from Jacobi et al MatLab code
#                 palmer_PDSI, palmer_PHDI, palmer_PMDI, palmer_Z = palmer.pdsi_from_climatology(precip_timeseries,
#                                                                                                temp_timeseries,
#                                                                                                awc,
#                                                                                                latitude,
#                                                                                                data_begin_year,
#                                                                                                calibration_begin_year,
#                                                                                                calibration_end_year,
#                                                                                                expected_pdsi,
#                                                                                                B,
#                                                                                                H)
# 
#                 # find the differences between the new (Matlab-derived) and previous (Fortran-derived) versions
#                 pdsi_diffs = expected_pdsi - palmer_PDSI
#                 phdi_diffs = expected_phdi - palmer_PHDI
#                 pmdi_diffs = expected_pmdi - palmer_PMDI
#                 zindex_diffs = expected_zindex - palmer_Z
#                
#                 # dictionary of variable names to corresponding arrays of differences to facilitate looping below
#                 varnames_to_arrays = {'palmer_PDSI': (pdsi_diffs, expected_pdsi, palmer_PDSI),
#                                       'palmer_PHDI': (phdi_diffs, expected_phdi, palmer_PHDI),
#                                       'palmer_PMDI': (pmdi_diffs, expected_pmdi, palmer_PMDI),
#                                       'palmer_Z-INDEX': (zindex_diffs, expected_zindex, palmer_Z) }
#                 # we want to see all zero differences, if any non-zero differences exist then raise an alert
#                 zeros = np.zeros(pdsi_diffs.shape)
#                 for varname, array_tuple in varnames_to_arrays.items():
#                     
#                     diffs = array_tuple[0]
#                     expected = array_tuple[1]
#                     actual = array_tuple[2]
#                        
#                     # plot the differences
#                     plot_diffs(expected,
#                                actual,
#                                diffs,
#                                division_id,
#                                varname,
#                                args.output_dir)
                          
                # compute palmer_PDSI etc. using new palmer_PDSI code translated from Jacobi et al MatLab code
                palmer_SCPDSI, palmer_PDSI, palmer_PHDI, palmer_PMDI, palmer_Z = palmer.scpdsi_from_climatology(precip_timeseries,
                                                                                                                temp_timeseries,
                                                                                                                awc,
                                                                                                                latitude,
                                                                                                                data_begin_year,
                                                                                                                calibration_begin_year,
                                                                                                                calibration_end_year)

                # find the differences between the new (Matlab-derived) and previous (Fortran-derived) versions
                pdsi_diffs = expected_pdsi - palmer_PDSI
                phdi_diffs = expected_phdi - palmer_PHDI
                pmdi_diffs = expected_pmdi - palmer_PMDI
                zindex_diffs = expected_zindex - palmer_Z
                scpdsi_diffs = expected_pdsi - palmer_SCPDSI
               
                # dictionary of variable names to corresponding arrays of differences to facilitate looping below
                varnames_to_arrays = {'palmer_SCPDSI': (scpdsi_diffs, expected_pdsi, palmer_SCPDSI),
                                      'palmer_PDSI': (pdsi_diffs, expected_pdsi, palmer_PDSI),
                                      'palmer_PHDI': (phdi_diffs, expected_phdi, palmer_PHDI),
                                      'palmer_PMDI': (pmdi_diffs, expected_pmdi, palmer_PMDI),
                                      'palmer_Z-INDEX': (zindex_diffs, expected_zindex, palmer_Z) }
   
                # we want to see all zero differences, if any non-zero differences exist then raise an alert
                zeros = np.zeros(pdsi_diffs.shape)
                for varname, array_tuple in varnames_to_arrays.items():
                    
                    diffs = array_tuple[0]
                    expected = array_tuple[1]
                    actual = array_tuple[2]
                       
                    # plot the differences
                    plot_diffs(expected,
                               actual,
                               diffs,
                               division_id,
                               varname,
                               args.output_dir)

    except Exception as ex:
        logger.exception('Failed to complete', exc_info=True)
        raise
    
#-----------------------------------------------------------------------------------------------------------------------
def plot_diffs(expected,
               actual,
               diffs,
               division_id,
               varname,
               output_dir):

    error = rmse(actual, expected)
    
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
if __name__ == "__main__":
    main()
    
