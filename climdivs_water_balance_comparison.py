import argparse
from datetime import datetime
import indices
import logging
import math
import multiprocessing
import netCDF4
import netcdf_utils
import numpy as np
import palmer
import os
import subprocess
import sys

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
                
                precip_timeseries = input_dataset.variables[args.precip_var_name][division_index, :]
                temp_timeseries = input_dataset.variables[args.temp_var_name][division_index, :]
                awc = input_dataset.variables[args.awc_var_name][division_index]
                latitude = input_dataset.variables['lat'][division_index]
                B = input_dataset.variables['B'][division_index]
                H = input_dataset.variables['H'][division_index]

                neg_tan_lat = -1 * math.tan(latitude)
                pdat, spdat, pedat, pldat, prdat, rdat, tldat, etdat, rodat, tdat, sssdat, ssudat = palmer.new_water_balance(temp_timeseries, precip_timeseries, awc, neg_tan_lat, B, H)
                    
                pdinew_spdat = input_dataset.variables['spdat'][division_index, :]
                pdinew_pedat = input_dataset.variables['pedat'][division_index, :]
                pdinew_pldat = input_dataset.variables['pldat'][division_index, :]
                pdinew_prdat = input_dataset.variables['prdat'][division_index, :]
                pdinew_rdat = input_dataset.variables['rdat'][division_index, :]
                pdinew_tldat = input_dataset.variables['tldat'][division_index, :]
                pdinew_etdat = input_dataset.variables['etdat'][division_index, :]
                pdinew_rodat = input_dataset.variables['rodat'][division_index, :]
                pdinew_tdat = input_dataset.variables['tdat'][division_index, :]
                pdinew_sssdat = input_dataset.variables['sssdat'][division_index, :]
                pdinew_ssudat = input_dataset.variables['ssudat'][division_index, :]
                
    except Exception as ex:
        logger.exception('Failed to complete', exc_info=True)
        raise