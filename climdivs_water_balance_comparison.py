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

                # compute water balance values using the function translated from the Fortran pdinew.f
                neg_tan_lat = -1 * math.tan(math.radians(latitude))
                pdat, spdat, pedat, pldat, prdat, rdat, tldat, etdat, rodat, tdat, sssdat, ssudat = pdinew._water_balance(temp_timeseries, precip_timeseries, awc, neg_tan_lat, B, H)
                    
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
                    if not np.allclose(diffs_array, zeros, atol=0.01, equal_nan=True):
                        logger.warn('Division {0}: Comparing pdinew.py against operational pdinew.f: ' + \
                                    'Non-matching difference arrays for water balance variable: {1}'.format(division_id, varname))
                        offending_indices = np.where(abs(diffs_array) > 0)
                        logger.warn('Indices with significant differences: {0}'.format(offending_indices))
                        
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
                    if not np.allclose(diffs_array, zeros, atol=0.001, equal_nan=True):
                        logger.warn('Division {0}: Comparing new Palmer against operational pdinew.f: ' + \
                                    'Non-matching difference arrays for water balance variable: {1}'.format(division_id, 
                                                                                                            varname))

                # calibration period years
                calibration_begin_year = 1931
                calibration_end_year = 1990
                data_begin_year = 1895

                # compare the results of the pdinew.f translated code (from pdinew.py) against the new values computed by the new Palmer implementation based on Jacobi 2013
                                 
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
                        
                # compute the PDI values from the version translated from pdinew.f
                #? how does PRO relate to PPR in the signature for _zindex_pdsi(), are they really the same? 
                pdinew_PDSI, pdinew_PHDI, pdinew_PMDI, pdinew_Z = pdinew._zindex_pdsi_pandas(precip_timeseries,
                                                                                      pedat,
                                                                                      prdat,
                                                                                      spdat,
                                                                                      pldat,
                                                                                      PRO,
                                                                                      alpha,
                                                                                      beta,
                                                                                      gamma,
                                                                                      delta,
                                                                                      AK, 1895, 2017)
                
                # compute using new PDSI code translated from Jacobi et al MatLab code
                PDSI, PHDI, PMDI, zindex = palmer.pdsi_from_climatology(precip_timeseries,
                                                                        temp_timeseries,
                                                                        awc,
                                                                        latitude,
                                                                        data_begin_year,
                                                                        calibration_begin_year,
                                                                        calibration_end_year)
                
                # find the differences between the new (Matlab-derived) and previous (Fortran-derived) versions
                pdsi_diffs = PDSI - pdinew_PDSI.flatten()
                phdi_diffs = PHDI - pdinew_PHDI.flatten()
                pmdi_diffs = PMDI - pdinew_PMDI.flatten()
                zindex_diffs = zindex - pdinew_Z.flatten()
            
    except Exception as ex:
        logger.exception('Failed to complete', exc_info=True)
        raise