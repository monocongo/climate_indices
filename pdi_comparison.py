import argparse
import logging
import math
import netCDF4
import numpy as np
import palmer
import pdinew
import utils

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
if __name__ == '__main__':

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
        args = parser.parse_args()

        # open the NetCDF files 
        with netCDF4.Dataset(args.input_file) as input_dataset:

            # read the temperature, precipitation, latitude and AWC for each division
            for division_index, division_id in enumerate(list(input_dataset.variables['division'][:])):
                
                # get the data for this division
                precip_timeseries = input_dataset.variables[args.precip_var_name][division_index, :]
                temp_timeseries = input_dataset.variables[args.temp_var_name][division_index, :]
                awc = input_dataset.variables[args.awc_var_name][division_index]
                latitude = input_dataset.variables['lat'][division_index]
                B = input_dataset.variables['B'][division_index]
                H = input_dataset.variables['H'][division_index]

                # calculate the negative tangent of the latitude which is used as an argument to the water balance function
                neg_tan_lat = -1 * math.tan(math.radians(latitude))
  
                # compute water balance values using the function translated from the Fortran pdinew.f
                #NOTE keep this code in place in order to compute the PET used later, since the two have 
                # different PET algorithms and we want to compare PDSI using the same PET inputs 
                pdat, spdat, pedat, pldat, prdat, rdat, tldat, etdat, rodat, prodat, tdat, sssdat, ssudat = \
                    pdinew._water_balance(temp_timeseries, precip_timeseries, awc, neg_tan_lat, B, H)
                       
                # compare the values against the operational values produced monthly by pdinew.f
                spdat_diffs = input_dataset.variables['spdat'][division_index, :] - spdat.flatten()
                pedat_diffs = input_dataset.variables['pedat'][division_index, :] - pedat.flatten()
                pldat_diffs = input_dataset.variables['pldat'][division_index, :] - pldat.flatten()
                prdat_diffs = input_dataset.variables['prdat'][division_index, :] - prdat.flatten()
                rdat_diffs = input_dataset.variables['rdat'][division_index, :] - rdat.flatten()
                tldat_diffs = input_dataset.variables['tldat'][division_index, :] - tldat.flatten()
                etdat_diffs = input_dataset.variables['etdat'][division_index, :] - etdat.flatten()
                rodat_diffs = input_dataset.variables['rodat'][division_index, :] - rodat.flatten()
                tdat_diffs = input_dataset.variables['tdat'][division_index, :] - tdat.flatten()
                sssdat_diffs = input_dataset.variables['sssdat'][division_index, :] - sssdat.flatten()
                ssudat_diffs = input_dataset.variables['ssudat'][division_index, :] - ssudat.flatten()
 
                # dictionary of variable names to corresponding arrays of differences                
                varnames_to_arrays = {'SP': spdat_diffs,
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
                zeros = np.zeros(spdat_diffs.shape)
                for varname, diffs_array in varnames_to_arrays.items():
                    tolerance = 0.01
                    if not np.allclose(diffs_array, zeros, atol=tolerance, equal_nan=True):
                        logger.warn('Division {0}: Comparing pdinew.py against operational pdinew.f: '.format(division_id) + \
                                    'Non-matching difference arrays for water balance variable: {0}'.format(varname))
                        offending_indices = np.where(abs(diffs_array) > tolerance)
                        logger.warn('Indices with significant differences: {0}'.format(offending_indices))
                          
                #NOTE we need this water balance call to stay uncommented in order to get the PRO value used later/below
                # compute the water balance values using the new Python version derived from Jacobi et al Matlab PDSI                                                                             
                ET, PR, R, RO, PRO, L, PL = palmer._water_balance(awc + 1.0, pedat, pdat)
                  
                # compare the values against the operational values produced monthly by pdinew.f
                etdat_wb_diffs = input_dataset.variables['etdat'][division_index, :] - ET
                prdat_wb_diffs = input_dataset.variables['prdat'][division_index, :] - PR
                rdat_wb_diffs = input_dataset.variables['rdat'][division_index, :] - R
                rodat_wb_diffs = input_dataset.variables['rodat'][division_index, :] - RO
                ldat_wb_diffs = input_dataset.variables['tldat'][division_index, :] - L
                pldat_wb_diffs = input_dataset.variables['pldat'][division_index, :] - PL
                 
                # compare the differences of the two, these difference arrays should come out to all zeros
                et_diffs = etdat_wb_diffs - etdat_diffs
                pr_diffs = prdat_wb_diffs - prdat_diffs
                r_diffs = rdat_wb_diffs - rdat_diffs
                ro_diffs = rodat_wb_diffs - rodat_diffs
                l_diffs = ldat_wb_diffs - tldat_diffs
                pl_diffs = pldat_wb_diffs - pldat_diffs
                 
                # dictionary of variable names to corresponding arrays of differences                
                varnames_to_arrays = {'PL': pl_diffs,
                                      'PR': pr_diffs,
                                      'R': r_diffs,
                                      'L': l_diffs,
                                      'ET': et_diffs,
                                      'RO': rodat_diffs }
 
                # we want to see all zero differences, if any non-zero differences exist then raise an alert
                zeros = np.zeros(pl_diffs.shape)
                for varname, diffs_array in varnames_to_arrays.items():
                    tolerance = 0.01
                    if not np.allclose(diffs_array, zeros, atol=tolerance, equal_nan=True):
                        logger.warn('Division {0}: Comparing palmer.py against operational pdinew.f: '.format(division_id) + \
                                    'Non-matching difference arrays for water balance variable: {0}'.format(varname))
                        offending_indices = np.where(abs(diffs_array) > tolerance)
                        logger.warn('Indices with significant differences: {0}'.format(offending_indices))

                # calibration period years used operationally with pdinew.f
                calibration_begin_year = 1931
                calibration_end_year = 1990
 
                # compare the results of the pdinew.f translated code (from pdinew.py) against the values  
                # computed by the corresponding new Palmer implementation based on Jacobi 2013
                                  
                #TODO get these values out of the NetCDF, compute from time values, etc.                        
                data_begin_year = 1895
                data_end_year = 2017

                #NOTE we need to compute CAFEC coefficients for use later/below
                # compute PDSI etc. using translated functions from pdinew.f Fortran code
                alpha, beta, delta, gamma, t_ratio = pdinew._cafec_coefficients(precip_timeseries,
                                                                                pedat,
                                                                                etdat,
                                                                                prdat,
                                                                                rdat,
                                                                                rodat,
                                                                                PRO,
                                                                                tldat,
                                                                                pldat,
                                                                                spdat,
                                                                                data_begin_year,
                                                                                calibration_begin_year,
                                                                                calibration_end_year)
                  
                # compute the coefficients using the new function
                new_alpha, new_beta, new_gamma, new_delta = palmer._cafec_coefficients(precip_timeseries,
                                                                                       pedat,
                                                                                       etdat,
                                                                                       prdat,
                                                                                       rdat,
                                                                                       rodat,
                                                                                       PRO,
                                                                                       tldat,
                                                                                       pldat,
                                                                                       data_begin_year,
                                                                                       calibration_begin_year,
                                                                                       calibration_end_year)
 
                # look at the differences between the results of the old and new versions of the coefficients code                
                alpha_diffs = alpha - new_alpha
                beta_diffs = beta - new_beta
                gamma_diffs = gamma - new_gamma
                delta_diffs = delta - new_delta
                 
                # dictionary of variable names to corresponding arrays of differences                
                varnames_to_arrays = {'Alpha': alpha_diffs,
                                      'Beta': beta_diffs,
                                      'Gamma': gamma_diffs,
                                      'Delta': delta_diffs }
 
                # we want to see all zero differences, if any non-zero differences exist then raise an alert
                zeros = np.zeros(alpha_diffs.shape)
                for varname, diffs_array in varnames_to_arrays.items():
                    if not np.allclose(diffs_array, zeros, atol=0.001, equal_nan=True):
                        logger.warn('Division {0}: Comparing new Palmer against operational pdinew.f: ' + \
                                    'Non-matching difference arrays for CAFEC coefficient: {1}'.format(division_id, varname))
 
                # compute the weighting factor (climatic characteristic) using the new version
                K = palmer._climatic_characteristic(alpha,
                                                    beta,
                                                    gamma,
                                                    delta,
                                                    pdat,
                                                    etdat,
                                                    pedat,
                                                    rdat,
                                                    prdat,
                                                    rodat,
                                                    PRO,
                                                    tldat,
                                                    pldat,
                                                    data_begin_year,
                                                    calibration_begin_year,
                                                    calibration_end_year)
 
                # compute the weighting factor (climatic characteristic) using the version translated from pdinew.f
                AK = pdinew._climatic_characteristic(alpha,
                                                     beta,
                                                     gamma,
                                                     delta,
                                                     pdat,
                                                     pedat,
                                                     prdat,
                                                     spdat,
                                                     pldat,
                                                     t_ratio,
                                                     data_begin_year,
                                                     calibration_begin_year,
                                                     calibration_end_year)
                 
                # look at the differences of the climatic characteristic results of the two implementations
                K_diffs = K - AK        
                zeros = np.zeros(K_diffs.shape)
                if not np.allclose(K_diffs, zeros, atol=0.001, equal_nan=True):
                    logger.warn('Division {0}: Non-matching difference arrays for climatic characteristic: {1}'.format(division_id, varname))
 
                # compute PDSI etc. using PDSI code translated from pdinew.f
                pdinew_PDSI, pdinew_PHDI, pdinew_PMDI, pdinew_Z = pdinew.pdsi_from_climatology(precip_timeseries,
                                                                                               temp_timeseries,
                                                                                               awc,
                                                                                               latitude,
                                                                                               B,
                                                                                               H,
                                                                                               data_begin_year,
                                                                                               data_end_year,
                                                                                               calibration_begin_year,
                                                                                               calibration_end_year)
                
                # compute PDSI etc. using new PDSI code translated from Jacobi et al MatLab code
                PDSI, PHDI, PMDI, zindex = palmer.pdsi_from_climatology(precip_timeseries,
                                                                        temp_timeseries,
                                                                        awc,
                                                                        latitude,
                                                                        data_begin_year,
                                                                        calibration_begin_year,
                                                                        calibration_end_year)
                
                # find the differences between the new (Matlab-derived) and previous (Fortran-derived) versions
                pdsi_diffs = pdinew_PDSI.flatten() - input_dataset.variables['pdsi.index'][division_index, :]
                phdi_diffs = pdinew_PHDI.flatten() - input_dataset.variables['phdi.index'][division_index, :]
                pmdi_diffs = pdinew_PMDI.flatten() - input_dataset.variables['pmdi.index'][division_index, :]
                zindex_diffs = pdinew_Z.flatten() - input_dataset.variables['z.index'][division_index, :]
            
                # dictionary of variable names to corresponding arrays of differences to facilitate looping below
                varnames_to_arrays = {'PDSI': pdsi_diffs,
                                      'PHDI': phdi_diffs,
                                      'PMDI': pmdi_diffs,
                                      'Z-INDEX': zindex_diffs }

                # we want to see all zero differences, if any non-zero differences exist then raise an alert
                zeros = np.zeros(pdsi_diffs.shape)
                for varname, diffs_array in varnames_to_arrays.items():
                    tolerance = 0.01
                    if not np.allclose(diffs_array, zeros, atol=tolerance, equal_nan=True):
                        logger.warn('Division {0}: Comparing new Palmer (pdinew.py) against operational pdinew.f: ' + \
                                    'Non-matching values for {1}'.format(division_id, varname))
                        offending_indices = np.where(abs(diffs_array) > 0)
                        logger.warn('Time steps with significant differences: {0}'.format(offending_indices))

                # find the differences between the new (Matlab-derived) and previous (Fortran-derived) versions
                pdsi_diffs = PDSI - input_dataset.variables['pdsi.index'][division_index, :]
                phdi_diffs = PHDI - input_dataset.variables['phdi.index'][division_index, :]
                pmdi_diffs = PMDI - input_dataset.variables['pmdi.index'][division_index, :]
                zindex_diffs = zindex - input_dataset.variables['z.index'][division_index, :]
            
                # dictionary of variable names to corresponding arrays of differences to facilitate looping below
                varnames_to_arrays = {'PDSI': pdsi_diffs,
                                      'PHDI': phdi_diffs,
                                      'PMDI': pmdi_diffs,
                                      'Z-INDEX': zindex_diffs }

                # we want to see all zero differences, if any non-zero differences exist then raise an alert
                zeros = np.zeros(pdsi_diffs.shape)
                for varname, diffs_array in varnames_to_arrays.items():
                    tolerance = 0.01
                    if not np.allclose(diffs_array, zeros, atol=tolerance, equal_nan=True):
                        logger.warn('Division {0}: Comparing new Palmer (palmer.py) against operational pdinew.f: ' + \
                                    'Non-matching values for {1}'.format(division_id, varname))
                        offending_indices = np.where(abs(diffs_array) > 0)
                        logger.warn('Time steps with significant differences: {0}'.format(offending_indices))

    except Exception as ex:
        logger.exception('Failed to complete', exc_info=True)
        raise