import argparse
import datetime
import logging

from scripts.ingest import ingest_nclimdiv
from scripts.process import process_divisions

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
def ingest_and_process_indices(divisions_file,
                               temp_var_name,
                               precip_var_name,
                               awc_var_name,
                               scale_months,
                               calibration_start_year,
                               calibration_end_year,
                               use_orig_pe=False):

    # perform an ingest of the NCEI nClimDiv datasets for input (temperature  
    # and precipitation) plus monthly computed indices for comparison
    ingest_nclimdiv.ingest_netcdf_latest(divisions_file,
                                         temp_var_name,
                                         precip_var_name,
                                         awc_var_name)

    # perform the processing, using original NCDC PET calculation method, writing results back into input NetCDF
    process_divisions.process_divisions(divisions_file,
                                        precip_var_name,
                                        temp_var_name,
                                        awc_var_name,
                                        scale_months,
                                        calibration_start_year,
                                        calibration_end_year,
                                        use_orig_pe)
        
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    Script for monthly ingest and processing task.
    
    Example command line invocation:
    
    $ python -u --out_file C:/home/data/nclimdiv/nclimdiv_latest.nc \
                --month_scales 1 2 3 6 9 12 24 \
                --calibration_start_year 1931 \
                --calibration_end_year 1990

    This module is used to perform ingest and climate indices processing on nClimdiv datasets.
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

        # variable names used within the monthly NetCDF for NIDIS, add to args dict (for now)
        args.temp_var_name = 'tavg'
        args.precip_var_name = 'prcp'
        args.awc_var_name = 'awc'
        
        ingest_and_process_indices(args.out_file, 
                                   args.temp_var_name, 
                                   args.precip_var_name, 
                                   args.awc_var_name, 
                                   args.month_scales, 
                                   args.calibration_start_year, 
                                   args.calibration_end_year, 
                                   args.use_orig_pe)
        # report on the elapsed time
        end_datetime = datetime.datetime.now()
        _logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        _logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        _logger.exception('Failed to complete', exc_info=True)
        raise
    