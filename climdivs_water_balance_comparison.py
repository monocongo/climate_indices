import argparse
import logging
import math
import netCDF4
import numpy as np
import palmer
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
                pdat, spdat, pedat, pldat, prdat, rdat, tldat, etdat, rodat, tdat, sssdat, ssudat = palmer.pdinew_water_balance(temp_timeseries, precip_timeseries, awc, neg_tan_lat, B, H)
                    
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
                ET, PR, R, RO, PRO, L, PL = palmer.water_balance(awc + 1.0, pedat, pdat)
                
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

                # compute using new PDSI from translated Jacobi et al MatLab code
                PDSI, PHDI, zindex = pdsi_from_climatology(precip_timeseries,
                                                           temp_timeseries,
                                                           awc,
                                                           latitude,
                                                           1895,
                                                           1931,
                                                           1990)
                
                # compute PDSI etc. using translated pdinew.f Fortran code
                alpha, beta, gamma, delta = palmer.cafec_coefficients_pdinew(precip_timeseries,
                                                                             pedat,
                                                                             etdat,
                                                                             prdat,
                                                                             rdat,
                                                                             rodat,
                                                                             prodat,
                                                                             tldat,
                                                                             pldat,
                                                                             data_start_year,
                                                                             calibration_start_year,
                                                                             calibration_end_year)
                pdinew_PDSI, pdinew_PHDI, pdinew_PMDI, pdinew_Z = palmer.pdinew_zindex_pdsi(precip_timeseries,
                                                                                            pedat,
                                                                                            prdat,
                                                                                            spdat,
                                                                                            pldat,
                                                                                            PPR,
                                                                                            alpha,
                                                                                            beta,
                                                                                            gamma,
                                                                                            delta)
                pass
            
    except Exception as ex:
        logger.exception('Failed to complete', exc_info=True)
        raise