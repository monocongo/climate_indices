import argparse
from datetime import datetime
import ftplib
import ingest_nclimdiv
from io import StringIO
import logging
import process_nclimdiv

#-----------------------------------------------------------------------------------------------------------------------
_TEMP_VAR_NAME = 'tavg'
_PRECIP_VAR_NAME = 'prcp'
_AWC_VAR_NAME = 'awc'
_DIVISION_VAR_NAME = 'division'

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
def _get_processing_date():
    '''
    Gets the processing date as specified in the file ftp://ftp.ncdc.noaa.gov/pub/data/cirs/climdiv/procdate.txt
    '''
    stringIO = StringIO()
    ftp = ftplib.FTP('ftp.ncdc.noaa.gov')   
    ftp.login() 
    ftp.cwd('pub/data/cirs/climdiv/')             
    ftp.retrlines('RETR procdate.txt', stringIO.write) 
    stringIO.seek(0)
    processing_date = stringIO.readline()
    return processing_date

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        logger.info("Start time:    {0}".format(start_datetime, '%x'))

        # get the date string we'll use for file identification
        processing_date = _get_processing_date()
    
#         # parse the command line arguments
#         parser = argparse.ArgumentParser()
#         parser.add_argument("--base_file_path", 
#                             help="Output file path up to the base file name. For example if this value is /abc/base then the ouput " + \
#                                  "file will be/abc/base_<processing_date>.nc", 
#                             required=True)
#         parser.add_argument("--month_scales",
#                             help="Month scales over which the PNP, SPI, and SPEI values are to be computed",
#                             type=int,
#                             nargs = '*',
#                             choices=range(1, 73),
#                             required=True)
#         parser.add_argument("--calibration_start_year",
#                             help="Initial year of calibration period",
#                             type=int,
#                             choices=range(1870, start_datetime.year + 1),
#                             required=True)
#         parser.add_argument("--calibration_end_year",
#                             help="Final year of calibration period",
#                             type=int,
#                             choices=range(1870, start_datetime.year + 1),
#                             required=True)
#         args = parser.parse_args()
        
        # the NetCDF file we want to write
        nclimdiv_netcdf = '{0}_{1}.nc'.format('C:/home/data/nclimdiv/nclimdiv', processing_date)
#         nclimdiv_netcdf = '{0}_{1}.nc'.format(args.base_file_path, processing_date)

        # ingest the nClimDiv datasets into a NetCDF
        ingest_nclimdiv.ingest_netcdf(nclimdiv_netcdf, 
                                      processing_date,        
                                      _TEMP_VAR_NAME,
                                      _PRECIP_VAR_NAME,
                                      _AWC_VAR_NAME)
        
        # come up with a file to use as the results NetCDF
        indices_netcdf = '{0}_{1}.nc'.format('C:/home/data/nclimdiv/nclimdiv_nidis', processing_date)

        # compute indices for the nClimDiv dataset we just ingested
        process_nclimdiv.process_nclimdiv(nclimdiv_netcdf, 
                                          indices_netcdf, 
                                          [1, 2, 3, 6, 9, 12, 24], 
                                          _TEMP_VAR_NAME,
                                          _PRECIP_VAR_NAME,
                                          _AWC_VAR_NAME,
                                          1931, 
                                          1990)
        
        print('\nNIDIS indices file: {0}'.format(indices_netcdf))
        
        # report on the elapsed time
        end_datetime = datetime.now()
        logger.info("End time:      {}".format(end_datetime, '%x'))
        elapsed = end_datetime - start_datetime
        logger.info("Elapsed time:  {}".format(elapsed, '%x'))

    except:
        logger.exception('Failed to complete', exc_info=True)
        raise
