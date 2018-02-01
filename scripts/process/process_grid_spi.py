import argparse
from datetime import datetime
import logging
import multiprocessing
import netCDF4
import numpy as np

from indices_python import indices, netcdf_utils

#-----------------------------------------------------------------------------------------------------------------------
# static constants
_VALID_MIN = -10.0
_VALID_MAX = 10.0

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
# multiprocessing locks we'll use to synchronize I/O writes to NetCDF files, one per each output file
spi_gamma_lock = multiprocessing.Lock()
spi_pearson_lock = multiprocessing.Lock()

# ignore warnings
import warnings
warnings.simplefilter('ignore', Warning)

#-----------------------------------------------------------------------------------------------------------------------
class GridProcessor(object):             # pragma: no cover

    def __init__(self,
                 output_file_base,
                 netcdf_precip,
                 var_name_precip,
                 scale_months,
                 calibration_start_year,
                 calibration_end_year):

        self.output_file_base = output_file_base
        self.netcdf_precip = netcdf_precip
        self.var_name_precip = var_name_precip
        self.scale_months = scale_months
        self.calibration_start_year = calibration_start_year
        self.calibration_end_year = calibration_end_year        
        
        # TODO get the initial year from the precipitation NetCDF, for now use hard-coded value specific to CMORPH
        self.data_start_year = 1998

        # the number of months used in scaled indices (for example this will be set to 6 for 6-month SPI)
        # this will need to be reset before the object is used for computing the scaled indices
        self.months = -1

        # place holders for the scaled NetCDFs, these should be created and assigned here for each months scale
        self.netcdf_spi_gamma = ''
        self.netcdf_spi_pearson = ''

    #-----------------------------------------------------------------------------------------------------------------------
    def _set_scaling_months(self,
                            months):
        """
        Reset the instance's month scale, the scale that'll be used to computed scaled indices (SPI)
        
        :param months: the number of months, should correspond to one of the values of self.month_scales 
        """
        self.months = months

    #-----------------------------------------------------------------------------------------------------------------------
    def _initialize_scaled_netcdfs(self,
                                   months_scale):

        valid_min = -3.09
        valid_max = 3.09

        # dictionary of index types to the NetCDF dataset files corresponding to the base index names and
        # month scales (this is the object we'll build and return from this function)
        netcdfs = {}

        # dictionary of index types mapped to their corresponding long variable names to be used within their respective NetCDFs
        indicators_to_longnames = {'spi_gamma': 'Standard Precipitation Index (Gamma distribution), {0}-month scale',
                                   'spi_pearson': 'Standard Precipitation Index (Pearson Type III distribution), {0}-month scale'}

        # loop over the indices, creating an output NetCDF dataset for each
        for index, long_name in indicators_to_longnames.items():

            # create the variable name from the index and month scale
            variable_name = index + '_{0}'.format(str(months_scale).zfill(2))

            # create the NetCDF file path from the
            netcdf_file = self.output_file_base + '_' + variable_name + '.nc'

            # initialize the output NetCDF dataset
            netcdf_utils.initialize_netcdf_single_variable_grid(netcdf_file,
                                                                self.netcdf_precip,
                                                                variable_name,
                                                                long_name.format(months_scale),
                                                                valid_min,
                                                                valid_max)

            # add the months scale index's NetCDF to the dictionary for the current index
            netcdfs[index] = netcdf_file

        # assign the NetCDF file paths to the corresponding member variables
        self.netcdf_spi_gamma = netcdfs['spi_gamma']
        self.netcdf_spi_pearson = netcdfs['spi_pearson']

        # set the number of months so we'll know at which months scale the indices NetCDF files should be computed
        self.months = months_scale

    #-----------------------------------------------------------------------------------------------------------------------
    def run(self):

        # the number of worker processes we'll have in our process pool
        number_of_workers = multiprocessing.cpu_count()

        # open the input NetCDF files for compatibility validation and to get the data's time range
        with netCDF4.Dataset(self.netcdf_precip) as dataset_precip:

            # get the initial year of the input dataset(s)
            time_units = dataset_precip.variables['time']
            self.data_start_year = netCDF4.num2date(time_units[0], time_units.units).year

            # get the number of latitudes in the input dataset(s)
            lat_size = dataset_precip.variables['lat'].size

        # compute the SPI
        for months in self.scale_months:

            # initialize the output NetCDFs for this month scale
            self._initialize_scaled_netcdfs(months)

            # set the instance's scale size (number of months over which SPI, etc. will be scaled)
            self._set_scaling_months(months)

            # create a process Pool for worker processes to compute PET and Palmer indices, passing arguments to an initializing function
            pool = multiprocessing.Pool(processes=number_of_workers)

            # map the latitude indices as an arguments iterable to the compute function (reuse the same pool)
            result = pool.map_async(self._process_latitude_spi, range(lat_size))

            # get the exception(s) thrown, if any
            result.get()

            # close the pool and wait on all processes to finish
            pool.close()
            pool.join()

#             # convert the SPI files to compressed NetCDF4 and move to the destination directory
#             input_output_netcdfs = [(scaled_netcdfs['spi_gamma'], '/nidis/test/nclimgrid/spi_gamma/' + scaled_netcdfs['spi_gamma']),
#                                     (scaled_netcdfs['spi_pearson'], '/nidis/test/nclimgrid/spi_pearson/' + scaled_netcdfs['spi_pearson'])]
#
#             pool = multiprocessing.Pool(processes=number_of_workers)
#
#             # create an arguments iterable containing the input and output NetCDFs, map it to the convert function
#             result = pool.map_async(netcdf_utils.convert_and_move_netcdf, input_output_netcdfs)
#
#             # get the exception(s) thrown, if any
#             result.get()
#
#             # close the pool and wait on all processes to finish
#             pool.close()
#             pool.join()
#
#         # convert the PET file to compressed NetCDF4 and move into the destination directory
#         netcdf_utils.convert_and_move_netcdf((unscaled_netcdfs['pet'], '/nidis/test/nclimgrid/pet/' + unscaled_netcdfs['pet']))
#

    #-------------------------------------------------------------------------------------------------------------------
    def _process_latitude_spi(self, lat_index):
        '''
        Processes SPI for a single latitude slice.

        :param lat_index:
        '''

        logger.info('Computing SPI for latitude index %s', lat_index)

        # open the input NetCDFs
        with netCDF4.Dataset(self.netcdf_precip) as precip_dataset:

            # read the latitude slice of input precipitation and PET values
            precip_lat_slice = precip_dataset[self.var_name_precip][:, lat_index, :]   # assuming (time, lat, lon) orientation

            # compute SPI/Gamma across all longitudes of the latitude slice
            spi_gamma_lat_slice = np.apply_along_axis(indices.spi_gamma,
                                                      0,
                                                      precip_lat_slice,
                                                      self.months)

            # compute SPI/Pearson across all longitudes of the latitude slice
            spi_pearson_lat_slice = np.apply_along_axis(indices.spi_pearson,
                                                        0,
                                                        precip_lat_slice,
                                                        self.months,
                                                        self.data_start_year,
                                                        self.calibration_start_year,
                                                        self.calibration_end_year)

            # use the same variable name within both Gamma and Pearson NetCDFs
            #TODO update this for separate 'spi_gamma_<months>' and 'spi_pearson_<months>' instead
            spi_gamma_variable_name = 'spi_gamma_' + str(self.months).zfill(2)
            spi_pearson_variable_name = 'spi_pearson_' + str(self.months).zfill(2)

            # open the existing SPI/Gamma NetCDF file for writing, copy the latitude slice into the SPI variable at the indexed latitude position
            spi_gamma_lock.acquire()
            spi_gamma_dataset = netCDF4.Dataset(self.netcdf_spi_gamma, mode='a')
            spi_gamma_dataset[spi_gamma_variable_name][:, lat_index, :] = spi_gamma_lat_slice
            spi_gamma_dataset.sync()
            spi_gamma_dataset.close()
            spi_gamma_lock.release()

            # open the existing SPI/Pearson NetCDF file for writing, copy the latitude slice into the SPI variable at the indexed latitude position
            spi_pearson_lock.acquire()
            spi_pearson_dataset = netCDF4.Dataset(self.netcdf_spi_pearson, mode='a')
            spi_pearson_dataset[spi_pearson_variable_name][:, lat_index, :] = spi_pearson_lat_slice
            spi_pearson_dataset.sync()
            spi_pearson_dataset.close()
            spi_pearson_lock.release()

#-----------------------------------------------------------------------------------------------------------------------
def process_grid(output_file_base,     # pragma: no cover
                 precip_file,
                 precip_var_name,
                 month_scales,
                 calibration_start_year,
                 calibration_end_year):
    """
    Performs indices processing from gridded NetCDF inputs.
    
    :param output_file_base:
    :param precip_file: 
    :param precip_var_name:
    :param month_scales:
    :param calibration_start_year:
    :param calibration_end_year:
    """

    # perform the processing
    grid_processor = GridProcessor(output_file_base,
                                   precip_file,
                                   precip_var_name,
                                   month_scales,
                                   calibration_start_year,
                                   calibration_end_year)
    grid_processor.run()

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    This module is used to perform climate indices processing on gridded datasets in NetCDF.

    Example command line usage:
    
    --precip_file /tmp/jadams/cmorph_monthly_prcp_1998_2017.nc --precip_var_name prcp --output_file_base /tmp/jadams/cmorph --month_scales 1 2 3 6 9 12 24 --calibration_start_year 1931 --calibration_end_year 2010
    """

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        logger.info("Start time:    %s", start_datetime)

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--precip_file",
                            help="Precipitation dataset file (NetCDF) to be used as input for SPI computations",
                            required=True)
        parser.add_argument("--precip_var_name",
                            help="Precipitation variable name used in the precipitation NetCDF file",
                            required=True)
        parser.add_argument("--output_file_base",
                            help="Base output file path and name for the resulting output files",
                            required=True)
        parser.add_argument("--month_scales",
                            help="Month scales over which the SPI values are to be computed",
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
        args = parser.parse_args()

        
        '''
        Example command line arguments:
        
        --precip_file /dev/shm/PRISM_from_WRCC_prcp.nc --precip_var_name prcp --temp_file /dev/shm/PRISM_from_WRCC_tavg.nc --temp_var_name tavg --awc_file /dev/shm/prism_soil.nc --awc_var_name awc --output_file_base /dev/shm/prism_from_WRCC --month_scales 1 2 3 --calibration_start_year 1931 --calibration_end_year 1990
        '''
        
        # perform the processing
        process_grid(args.output_file_base,
                     args.precip_file,
                     args.precip_var_name,
                     args.month_scales,
                     args.calibration_start_year,
                     args.calibration_end_year)

        # report on the elapsed time
        end_datetime = datetime.now()
        logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        logger.exception('Failed to complete', exc_info=True)
        raise
