import argparse
from datetime import date, datetime
import glob
import logging
import multiprocessing
import netCDF4
import numpy as np
import pandas as pd
import uuid
import utils

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

# #-----------------------------------------------------------------------------------------------------------------------
# def _compute_days(initial_year,
#                   initial_month,
#                   total_months,
#                   start_year=1800):
# 
#     '''
#     Computes the "number of days" equivalent of the regular, incremental monthly time steps from an initial year/month.
#     
#     :param initial_year: the initial year from which the increments should start
#     :param initial_month: the initial month from which the increments should start
#     :param total_months: the total number of monthly increments (time steps measured in days) to be computed
#     :param start_year: the start year from which the monthly increments (time steps measured in days) to be computed
#     :return: an array of time step increments, measured in days since midnight of January 1st of the start_year
#     :rtype: ndarray of ints 
#     '''
# 
#     # compute an offset from which the day values should begin (assuming that we're using "days since <start_date>" as time units) 
#     start_date = datetime(start_year, 1, 1)
# 
#     # initialize the list of day values we'll build
#     days = np.empty(total_months, dtype=int)
#     
#     # loop over all time steps (months)
#     for i in range(total_months):
#         
#         years = int((i + initial_month - 1) / 12)   # the number of years since the initial year 
#         months = int((i + initial_month - 1) % 12)  # the number of months since January
#         
#         # cook up a date for the current time step (month)
#         current_date = datetime(initial_year + years, 1 + months, 1)
#         
#         # leverage the difference between dates operation available with datetime objects
#         days[i] = (current_date - start_date).days
#     
#     return days
#     
#-----------------------------------------------------------------------------------------------------------------------
def _get_coordinate_values(ascii_file):

    '''
    This function takes a nCLimGrid ASCII file for a single month and extracts a list of lat and lon coordinate values 
    for the regular grid contained therein.
    
    :param ascii_file:
    :return: lats and lons respectively
    :rtype: two 1-D numpy arrays of floats  
    '''

    # create a dataframe from the file
    data_frame = pd.read_csv(ascii_file, delim_whitespace=True, names = ['lat', 'lon', 'value'])

    # successive lats and lons are separated by 1/24th of a degree (regular grid)
    increment = 1 / 24.
    
    # determine the minimum lat and lon values
    min_lat = min(data_frame.lat)
    min_lon = min(data_frame.lon)

    # create lat and lon index columns corresponding to the dataframe's lat and lon values
    # the index starts at the minimum, i.e. lat_index[0] == min_lat
    data_frame['lat_index'] = (np.round((data_frame.lat - min_lat) / increment)).astype(int)
    data_frame['lon_index'] = (np.round((data_frame.lon - min_lon) / increment)).astype(int)

    # the lat|lon indices start at zero so the number of lats|lons is the length of the index plus one
    lats_count = max(data_frame.lat_index) + 1
    lons_count = max(data_frame.lon_index) + 1

    # since we know the starting lat|lon and the increment between then we can 
    # create a full list of lat and lon values based on the number of lats|lons
    lat_values = (np.arange(lats_count) * increment) + min_lat
    lon_values = (np.arange(lons_count) * increment) + min_lon

    return lat_values, lon_values

#-----------------------------------------------------------------------------------------------------------------------
def _get_variable_attributes(var_name):

    '''
    This function builds a dictionary of variable attributes based on the variable name. Four variable names 
    are supported: 'prcp', 'tavg', 'tmin', and 'tmax'.
    
    :param var_name:
    :return: attributes relevant to the specified variable name
    :rtype: dictionary with string keys corresponding to attribute names specified by the NCEI NetCDF 
            template for gridded datasets (https://www.nodc.noaa.gov/data/formats/netcdf/v2.0/grid.cdl)
    '''
    
    # initialize the attributes dictionary with values applicable to all supported variable names
    attributes = {'coordinates': 'time lat lon',
                  'references': 'GHCN-Monthly Version 3 (Vose et al. 2011), NCEI/NOAA, https://www.ncdc.noaa.gov/ghcnm/v3.php'}
    
    # flesh out additional attributes, based on the variable type
    if var_name == 'prcp':
        attributes['long_name'] = 'Precipitation, monthly total'
        attributes['standard_name'] = 'precipitation_amount'
        attributes['units'] = 'millimeter'
        attributes['valid_min'] = np.float32(0.0)
        attributes['valid_max'] = np.float32(2000.0)
    else:
        attributes['standard_name'] = 'air_temperature'
        attributes['units'] = 'degree_Celsius'
        attributes['valid_min'] = np.float32(-100.0)
        attributes['valid_max'] = np.float32(100.0)
        if var_name == 'tavg':
            attributes['long_name'] = 'Temperature, monthly average of daily averages'
            attributes['cell_methods'] = 'mean'
        elif var_name == 'tmax':
            attributes['long_name'] = 'Temperature, monthly average of daily maximums'
            attributes['cell_methods'] = 'maximum'
        elif var_name == 'tmin':
            attributes['long_name'] = 'Temperature, monthly average of daily minimums'
            attributes['cell_methods'] = 'minimum'
        else:
            raise ValueError('The var_name argument \"{}\" is unsupported.'.format(var_name))

    return attributes
        
#-----------------------------------------------------------------------------------------------------------------------
def _build_netcdf(ascii_files, 
                  netcdf_file, 
                  var_name):

    '''
    This function builds a NetCDF file for a nClimGrid dataset defined by a set of ASCII files. The list of ASCII files
    is assumed to be sorted in ascending order with the first file representing the first time step.
    
    :param ascii_files: a list nClimGrid ASCII files for a single variable dataset
    :param netcdf_file: the NetCDF file (full path) to create and build out from the data contained in the ASCII files 
    :param var_name: name of the variable, supported variables are 'prcp', 'tavg', 'tmin' and 'tmax'
    :rtype: None
    '''
    
    # get the start/end months/years from the initial/final file names in the list, which is assumed to be sorted in
    # ascending order, and file names are assumed to be in the format <YYYYMM>.pnt, eg. "201004.pnt" for April 2010
    initial_year = int(ascii_files[0][-10:-6])
    initial_month = int(ascii_files[0][-6:-4])

    # use NaN as the fill value for missing data
    output_fill_value = np.float32(np.NaN)
            
    # determine the lat and lon coordinate values by extracting these from the initial ASCII  
    # file in our list (assumes that each ASCII file contains the same lat/lon coordinates)
    lat_values, lon_values = _get_coordinate_values(ascii_files[0])

    min_lat = np.float32(min(lat_values))
    max_lat = np.float32(max(lat_values))
    min_lon = np.float32(min(lon_values))
    max_lon = np.float32(max(lon_values))
    lat_units = 'degrees_north'
    lon_units = 'degrees_east'
    total_lats = lat_values.shape[0]
    total_lons = lon_values.shape[0]

    # build the NetCDF    
    with netCDF4.Dataset(netcdf_file, 'w') as dataset:
                 
        # create dimensions for a time series, 2-D dataset 
        dataset.createDimension('time', None)
        dataset.createDimension('lat', total_lats)
        dataset.createDimension('lon', total_lons)
 
        # set global group attributes
        dataset.date_created = str(datetime.now())
        dataset.date_modified = str(datetime.now())
        dataset.Conventions = 'CF-1.6, ACDD-1.3'
        dataset.ncei_template_version = 'NCEI_NetCDF_Grid_Template_v2.0'
        dataset.standard_name_vocabulary = 'Standard Name Table v35'
        dataset.institution = 'US DOC; NOAA; NESDIS; National Centers for Environmental Information'
        dataset.geospatial_lat_min = min_lat
        dataset.geospatial_lat_max = max_lat
        dataset.geospatial_lon_min = min_lon
        dataset.geospatial_lon_max = max_lon
        dataset.geospatial_lat_units = lat_units
        dataset.geospatial_lon_units = lon_units

        dataset.Metadata_Conventions = 'Unidata Dataset Discovery v1.0'
        dataset.title = 'nClimGrid'
        dataset.summary = 'Gridded, 1/24 degree (~4km) resolution, CONUS Climatology, derived from NCEI GHCN-Monthly'
        dataset.naming_authority = 'gov.noaa.ncei'
        dataset.acknowledgment = 'NCEI/NOAA'
        dataset.cdm_data_type = 'Grid'
        dataset.comment = dataset.summary
        dataset.coverage_content_type = 'referenceInformation'
        dataset.creator_institution = 'US DOC; NOAA; NESDIS; National Centers for Environmental Information'
        dataset.uuid = uuid.uuid4()

        # create a time coordinate variable with an increment per month of the period of record
        start_year = 1800
        total_timesteps = len(ascii_files)
        chunk_sizes = [total_timesteps]
        time_variable = dataset.createVariable('time', 'i4', ('time',), chunksizes=chunk_sizes)
        time_variable[:] = utils.compute_days(initial_year, total_timesteps, initial_month, start_year)
        time_variable.long_name = 'Time, in monthly increments'
        time_variable.standard_name = 'time'
        time_variable.calendar = 'gregorian'
        time_variable.units = 'days since ' + str(start_year) + '-01-01 00:00:00'
        time_variable.axis = 'T'

        # create the lat coordinate variable
        lat_variable = dataset.createVariable('lat', 'f4', ('lat',))
        lat_variable.standard_name = 'latitude'
        lat_variable.long_name = 'Latitude'
        lat_variable.units = lat_units
        lat_variable.axis = 'Y'
        lat_variable.valid_min = min_lat# - 0.0001
        lat_variable.valid_max = max_lat# + 0.0001
        lat_variable.units = lat_units
        lat_variable[:] = lat_values

        # create the lon coordinate variable
        lon_variable = dataset.createVariable('lon', 'f4', ('lon',))
        lon_variable.standard_name = 'longitude'
        lon_variable.long_name = 'Longitude'
        lon_variable.units = lon_units
        lon_variable.axis = 'X'
        lon_variable.valid_min = min_lon
        lon_variable.valid_max = max_lon
        lon_variable.units = lon_units
        lon_variable[:] = lon_values

        # create the data variable
        variable = dataset.createVariable(var_name, 
                                          'f4', 
                                          ('time', 'lat', 'lon'), 
                                          fill_value=output_fill_value,
                                          zlib=True,
                                          least_significant_digit=3)
        
        # set the variable's attributes
        variable.setncatts(_get_variable_attributes(var_name))

        # array to contain variable data values
        variable_data = np.full((time_variable.shape[0], total_lats, total_lons), output_fill_value, dtype=np.float32)

        # loop over the ASCII files in order to build the variable's data array
        for time_index, ascii_file in enumerate(ascii_files):

            # create a pandas dataframe from the file
            data_frame = pd.read_csv(ascii_file, delim_whitespace=True, names = ['lat', 'lon', 'value'])
        
            # successive lats and lons are separated by 1/24th of a degree (regular grid)
            increment = 1 / 24.
            
            # create lat and lon index columns corresponding to the dataframe's lat and lon values
            # the index starts at the minimum, i.e. lat_index[0] == min_lat
            data_frame['lat_index'] = (np.round((data_frame.lat - min_lat) / increment)).astype(int)
            data_frame['lon_index'] = (np.round((data_frame.lon - min_lon) / increment)).astype(int)
            
            # fill the data array with data values, using the lat|lon indices
            variable_data[time_index, data_frame['lat_index'], data_frame['lon_index']] = data_frame['value']

        # assign the data array to the data variable
        variable[:] = variable_data
        
    print('NetCDF file created successfully for variable \"{0}\":  {1}'.format(var_name, netcdf_file))
        
#-----------------------------------------------------------------------------------------------------------------------
def _ingest_nclimgrid_dataset(parameters):

    '''
    This function creates a NetCDF for the full period of record of an nClimGrid dataset.
    
    :param parameters: dictionary containing all required parameters, used instead of individual parameters since this 
                       function will be called from a process pool mapping which requires a single function argument 
    '''

    try:
        # determine the input directory containing the ASCII files for the variable based on the variable's name
        if parameters['variable_name'] == 'prcp':
            input_directory = parameters['source_dir'] + '/prcp'
        elif parameters['variable_name'] == 'tavg':
            input_directory = parameters['source_dir'] + '/tave'
        elif parameters['variable_name'] == 'tmax':
            input_directory = parameters['source_dir'] + '/tmax'
        elif parameters['variable_name'] == 'tmin':
            input_directory = parameters['source_dir'] + '/tmin'
        else:
            raise ValueError('The variable_name argument \"{}\" is unsupported.'.format(parameters['variable_name']))
    
        print('Ingesting ASCII files for variable \"{}\"'.format(parameters['variable_name']))
    
        # get a list of all *.pnt files in the specified directory 
        ascii_files = sorted(glob.glob('/'.join([input_directory, '*.pnt'])))

        # create the file names for the various output NetCDFs we'll create
        netcdf_file = parameters['out_dir'] + '/nclimgrid_' + parameters['base_start'] + '_' + parameters['base_end'] + '_' + parameters['variable_name'] + '.nc'
        output_netcdf_file = parameters['output_dir'] + '/nclimgrid_' + parameters['variable_name'] + '.nc'

        # create the base period NetCDF
        _build_netcdf(ascii_files, netcdf_file, parameters['variable_name'])
            
        print('Completed ingest for variable \"{0}\":  result NetCDF file: {1}'.format(parameters['variable_name'], output_netcdf_file))
    
    except:
        # catch all exceptions
        print('Failed to complete')
        raise

#-----------------------------------------------------------------------------------------------------------------------
def ingest_to_netcdf(source_dir,
                     output_dir):
    """
    Ingest ASCII files to NetCDF. Four files created: precipitation, minimum temperature, maximum temperature, 
    and mean temperature.
    
    :param source_dir:
    :param output_dir: directory where output files for each variable will be written
    """
     
#     # REMOVE once we have a method in place where these arguments can be provided outside of this function
#     source_dir = '/datasets/internal/nclimgrid'
#     output_dir = '/datasets/processed/nclimgrid/output'

    # get the current year and month, used to determine the start/end points for the base and incremental periods
    current_month = date.today().month

    '''
    If we're running this code during January then we'll ingest the "base" period of record, i.e. from January 1895 
    through December of the year which is two years before the current year, as well as the "incremental" period
    which includes all months after the base period to the previous month. For this initial monthly run of the year
    the base period files will be cached in the specified storage directory for use with subsequent monthly runs,
    eliminating the need to reingest that portion of the dataset, since the base period data is guaranteed to be
    updated only once a year, at the start of each calendar year.
    '''
    
    # create an iterable containing dictionaries of parameters, with one dictionary of parameters per variable, 
    # since there will be a separate ingest process per variable, with each process having its own set of parameters
    variables = ['prcp', 'tavg', 'tmin', 'tmax']
    params_list = []

    for variable_name in variables:
        params = {'source_dir': source_dir,
                  'current_month': current_month,
                  'variable_name': variable_name,
                  'output_dir': output_dir}
        params_list.append(params)

    # create a process pool, mapping the ingest process to the iterable of parameter lists
    pool = multiprocessing.Pool(min(len(variables), multiprocessing.cpu_count()))
    pool.map_async(_ingest_nclimgrid_dataset, params_list)
    
    # get the result exception, if any
    pool.close()
    pool.join()

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    try:
        
        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--source_dir", 
                            help="Base directory under which are directories and files for precipitation and max/min/mean temperature", 
                            required=True)
        parser.add_argument("--output_dir", 
                            help="Directory under which the output NetCDF files will be written", 
                            required=True)
        args = parser.parse_args()

        # perform ingest to NetCDF using public API
        ingest_to_netcdf(args.source_dir,
                               args.output_dir)
    except:
        # catch all exceptions
        print('Failed to complete')
        raise
