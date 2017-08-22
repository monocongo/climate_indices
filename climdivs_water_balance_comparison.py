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
                
#                 zeros = np.zeros(et_diffs.shape)
#                 if np.allclose(et_diffs, zeros, atol=0.0005):
#                     logger.warn('Division {0}: Water balance differences for ET'.format(division_id))
#                 if np.allclose(pr_diffs, zeros, atol=0.0005):
#                     logger.warn('Division {0}: Water balance differences for PR'.format(division_id))
#                 if np.allclose(r_diffs, zeros, atol=0.0005):
#                     logger.warn('Division {0}: Water balance differences for R'.format(division_id))
#                 if np.allclose(ro_diffs, zeros, atol=0.0005):
#                     logger.warn('Division {0}: Water balance differences for RO'.format(division_id))
#                 if np.allclose(l_diffs, zeros, atol=0.0005):
#                     logger.warn('Division {0}: Water balance differences for L'.format(division_id))
#                 if np.allclose(pl_diffs, zeros, atol=0.0005):
#                     logger.warn('Division {0}: Water balance differences for PL'.format(division_id))

                # calibration period years
                calibration_begin_year = 1931
                calibration_end_year = 1990
                data_begin_year = 1895
                
                # compute using new PDSI from translated Jacobi et al MatLab code
                PDSI, PHDI, PMDI, zindex = palmer.pdsi_from_climatology(precip_timeseries,
                                                                        temp_timeseries,
                                                                        awc,
                                                                        latitude,
                                                                        data_begin_year,
                                                                        calibration_begin_year,
                                                                        calibration_end_year)
                
                # compute PDSI etc. using translated functions from pdinew.f Fortran code
                alpha, beta, gamma, delta, t_ratio = pdinew._cafec_coefficients(precip_timeseries,
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
                
                K_diffs = K - AK
                
                # compute the PDI values from the version translated from pdinew.f
                #? how does PRO relate to PPR in the signature for _zindex_pdsi(), are they really the same? 
                pdinew_PDSI, pdinew_PHDI, pdinew_PMDI, pdinew_Z = pdinew._zindex_pdsi(precip_timeseries,
                                                                                      pedat,
                                                                                      prdat,
                                                                                      spdat,
                                                                                      pldat,
                                                                                      PRO,
                                                                                      alpha,
                                                                                      beta,
                                                                                      gamma,
                                                                                      delta,
                                                                                      AK)
                
                # find the differences between the new (Matlab-derived) and previous (Fortran-derived) versions
                pdsi_diffs = PDSI - pdinew_PDSI.flatten()
                phdi_diffs = PHDI - pdinew_PHDI.flatten()
                pmdi_diffs = PMDI - pdinew_PMDI.flatten()
                zindex_diffs = zindex - pdinew_Z.flatten()
            
    except Exception as ex:
        logger.exception('Failed to complete', exc_info=True)
        raise