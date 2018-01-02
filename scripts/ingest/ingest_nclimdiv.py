from datetime import datetime
import io
import logging
import math
import netCDF4
import numpy as np
import os
import pandas as pd
import pycurl

from indices_python import utils

#-----------------------------------------------------------------------------------------------------------------------
_DIVISION_VAR_NAME = 'division'

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
_logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
def _get_processing_date():   # pragma: no cover

    buffer = io.BytesIO()
    c = pycurl.Curl()  
    c.setopt(c.URL, 'ftp://ftp.ncdc.noaa.gov/pub/data/cirs/climdiv/procdate.txt')
    c.setopt(c.WRITEDATA, buffer)
    c.perform()
    c.close()
    
    body = buffer.getvalue()
    
    # body is a byte string, we need to know the encoding in order to decode it
    return body.decode('iso-8859-1').rstrip()

#-----------------------------------------------------------------------------------------------------------------------
def _parse_climatology(date,                 # pragma: no cover
                       p_or_t='T'):

    # get the relevant ASCII file for US climate divisions from NCEI
    if p_or_t == 'P':
        #TODO replace this hard coded path with a function parameter, taken from command line
        file_url = 'ftp://ftp.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-pcpn'
        fill_value = -9.99
    elif p_or_t == 'T':
        #TODO replace this hard coded path with a function parameter, taken from command line
        file_url = 'ftp://ftp.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-tmpc'
        fill_value = -99.90
    else:
        raise ValueError('Invalid p_or_t argument: {}'.format(p_or_t))
        
    _logger.info("Parsing climatology (%s) from nClimDiv ASCII file: %s", p_or_t, file_url)
    
    # use a temporary file that we'll remove once no longer necessary
    tmp_file = "tmp_climatology_for_ingest_nclimdiv.txt"
    utils.retrieve_file(file_url + 'dv-v1.0.0-{0}'.format(date), tmp_file)
    div_file = open(tmp_file, 'r')
    
    # use a list of column names to clue in pandas as to the structure of the ASCII rows
    column_names = ['division_year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    # read the file into a pandas DataFrame object  
    results_df = pd.read_csv(div_file, 
                             delim_whitespace=True,
                             header=None,
                             names=column_names,
                             dtype={'division_year': str},
                             na_values=fill_value)
    
    # get rid of the temporary file used for the climatology data
    div_file.close()
    os.remove(tmp_file)
    
    # convert the fill values to NaNs
    #NOTE this may not be necessary, verify this
    results_df.applymap(lambda x: np.NaN if x == fill_value else x)
    
    # split the initial column into a division ID and year, both as integers 
    results_df['division_id'] = results_df['division_year'].str[0:4]
    results_df['year'] = results_df['division_year'].str[-4:]
    
    # convert the division ID and year columns to integer data type
    results_df[['division_id', 'year']] = results_df[['division_id', 'year']].apply(pd.to_numeric)
    
    # find the minimum and maximum years for each division, each as a Series with the division ID as index
    division_min_years = results_df.groupby(by='division_id').min().year
    division_max_years = results_df.groupby(by='division_id').max().year
    minmax_years_df = pd.DataFrame()
    minmax_years_df['min_year'] = division_min_years
    minmax_years_df['max_year'] = division_max_years
    
    # get a dictionary of division IDs mapped to a list of row numbers corresponding to the division
    division_groups = results_df.groupby(by='division_id').groups
    
    # for each division ID read the monthly values into a 2-D numpy array, and map the division ID to the array in a dictionary
    divs_to_arrays = {}
    for division_id, row_numbers in division_groups.items():
        rows = results_df.iloc[row_numbers]
        years_months_data = rows.as_matrix(columns=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        divs_to_arrays[division_id] = years_months_data.flatten()
    
    # get the division IDs mapped to their corresponding minimum and maximum years
    divs_to_minmax_years = minmax_years_df.to_dict('index')
    
    # return the two dictionaries we've built
    return divs_to_arrays, divs_to_minmax_years, division_min_years.min(), division_max_years.max()

#-----------------------------------------------------------------------------------------------------------------------
def _parse_results(div_file,                # pragma: no cover
                   intermediates=False):

    _logger.info("Parsing results from nClimDiv ASCII file: %s", div_file)
    
    # read the file into a pandas DataFrame    
    if intermediates:
        column_names = ['division_year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'fill_value']
    else:
        column_names = ['division_year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    results_df = pd.read_csv(div_file, 
                             delim_whitespace=True,
                             header=None,
                             names=column_names,
                             dtype={'division_year': str},
                             na_values=-99.99)
    
    # split the initial column into a division ID and year, both as integers
    if intermediates:
        results_df['division_id'] = results_df['division_year'].str[3:7]
    else:
        results_df['division_id'] = results_df['division_year'].str[0:4]
    results_df['year'] = results_df['division_year'].str[-4:]
    
    # convert the division ID and year columns to integer data type
    results_df[['division_id', 'year']] = results_df[['division_id', 'year']].apply(pd.to_numeric)
    
    # find the minimum and maximum years for each division, each as a Series with the division ID as index
    division_min_years = results_df.groupby(by='division_id').min().year
    division_max_years = results_df.groupby(by='division_id').max().year
    minmax_years_df = pd.DataFrame()
    minmax_years_df['min_year'] = division_min_years
    minmax_years_df['max_year'] = division_max_years
    
    # get a dictionary of division IDs mapped to a list of row numbers corresponding to the division
    division_groups = results_df.groupby(by='division_id').groups
    
    # for each division ID read the monthly values into a 2-D numpy array, and map the division ID to the array in a dictionary
    divs_to_arrays = {}
    for division_id, row_numbers in division_groups.items():
        rows = results_df.iloc[row_numbers]
        years_months_data = rows.as_matrix(columns=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        divs_to_arrays[division_id] = years_months_data.flatten()
        
    # get the division IDs mapped to their corresponding minimum and maximum years
    divs_to_minmax_years = minmax_years_df.to_dict('index')
    
    # return the two dictionaries we've built
    return divs_to_arrays, divs_to_minmax_years, division_min_years.min(), division_max_years.max()

#-----------------------------------------------------------------------------------------------------------------------
def _parse_soil_constants(soil_file,         # pragma: no cover
                          awc_var_name):
    '''
    Parse the soil constant file, reading both AWC and latitude values for each division.
    '''
    
    _logger.info("Parsing soil constants from AWC file: %s", soil_file)
    
    # use a list of column names to clue in pandas as to the structure of the ASCII rows
    column_names = ['division_id', awc_var_name, 'B', 'H', 'neg_tan_lat']
    columns_to_use = column_names

    # read the file into a pandas DataFrame object  
    results_df = pd.read_csv(soil_file, 
                             delim_whitespace=True,
                             header=None,
                             names=column_names,
                             usecols=columns_to_use,
                             dtype={'division_id': int})

    # make the division ID our index
    results_df.set_index('division_id', drop=True, inplace=True)
    
    # convert the negative tangent of latitude to latitude degrees north
    results_df['lat'] = results_df['neg_tan_lat'].apply(lambda x: math.degrees(math.atan(-1 * x)))

    # drop the column holding the negative tangent of the latitude, no longer needed
    results_df.drop('neg_tan_lat', axis=1, inplace=True)
    
    # produce a dictionary mapping divison IDs to corresponding AWC and latitude values
    divs_to_awc = results_df[awc_var_name].to_dict()
    divs_to_lats = results_df['lat'].to_dict()
    divs_to_bs = results_df['B'].to_dict()
    divs_to_hs = results_df['H'].to_dict()
    
    return divs_to_awc, divs_to_lats, divs_to_bs, divs_to_hs

#-----------------------------------------------------------------------------------------------------------------------
def _create_netcdf(output_netcdf,             # pragma: no cover
                   division_ids,
                   divisional_arrays,
                   divisional_minmax_years,
                   minmax_years,
                   divs_to_awc,
                   divs_to_lats,
                   divs_to_bs,
                   divs_to_hs,
                   total_months,
                   temp_var_name,
                   precip_var_name,
                   awc_var_name,
                   min_year):
    '''
    This function writes variable values for all climate divisions as a NetCDF.
    
    :param output_netcdf: input NetCDF containing indicator values
    '''
    
    with netCDF4.Dataset(output_netcdf, 'w') as dataset:

        # open the output file for writing, set its dimensions and variables
        dataset.createDimension('time', None)
        dataset.createDimension(_DIVISION_VAR_NAME, len(division_ids))

        # set some global group attributes
        dataset.title = 'US Climate Divisions'
        dataset.source = 'conversion from data set files provided by CMB'
        dataset.institution = 'National Centers for Environmental Information (NCEI), NESDIS, NOAA, U.S. Department of Commerce'
        dataset.standard_name_vocabulary = 'CF Standard Name Table (v26, 08 November 2013)'
        dataset.date_created = str(datetime.now())
        dataset.date_modified = str(datetime.now())
        dataset.Conventions = 'ClimDiv-1.0'  # suggested by Steve Ansari for support within the NOAA WCT

        # create a time coordinate variable with an increment per month of the period of record
        chunk_sizes = [total_months]
        time_units = dataset.createVariable('time', 'i4', ('time',), chunksizes=chunk_sizes)
        time_units.long_name = 'time'
        time_units.standard_name = 'time'
        time_units.calendar = 'gregorian'
        time_units.units = 'days since ' + str(min_year) + '-01-01 00:00:00'
        time_units[:] = utils.compute_days(min_year, total_months, units_start_year=min_year)

        # create the division ID coordinate variable
        division_variable = dataset.createVariable(_DIVISION_VAR_NAME, 'i4', (_DIVISION_VAR_NAME,))
        division_variable.long_name = 'US climate division ID' 
        division_variable.standard_name = 'division ID' 
        division_variable[:] = np.array(division_ids)

        # create a variable for each climatology and indicator
        variables = {}
        variable_names = divisional_arrays.keys()
        filler = np.NaN
        float_type = 'f8'
        for var_name in variable_names:
            
            # create the variable and store it in the dictionary under the variable name
            variable = dataset.createVariable(var_name, float_type, (_DIVISION_VAR_NAME, 'time'), fill_value=filler)
            variables[var_name] = variable
            
            # additional attributes on temperature and precipitation
            if var_name == temp_var_name:
                variable.long_name = 'temperature, monthly average' 
                variable.standard_name = 'temperature' 
                variable.units = 'Fahrenheit'
                variable.valid_min = -100.0
                variable.valid_max = 200.0
            elif var_name == precip_var_name:
                variable.long_name = 'precipitation, monthly total' 
                variable.standard_name = 'precipitation' 
                variable.units = 'inch' 
                variable.valid_min = 0.0
                variable.valid_max = 100.0

        # create the available water capacity variable
        awc_variable = dataset.createVariable(awc_var_name, float_type, (_DIVISION_VAR_NAME,), fill_value=filler)
        awc_variable.long_name = 'available water capacity' 
        awc_variable.standard_name = 'available water capacity' 
        awc_variable.units = 'inch' 
        awc_variable.valid_min = 0.0
        awc_variable.valid_max = 100.0

        # create the latitude variable
        lat_variable = dataset.createVariable('lat', float_type, (_DIVISION_VAR_NAME,), fill_value=filler)
        lat_variable.long_name = 'latitude' 
        lat_variable.standard_name = 'latitude' 
        lat_variable.units = 'degrees north' 
        lat_variable.valid_min = -90.0
        lat_variable.valid_max = 90.0

        # create the B variable
        b_variable = dataset.createVariable('B', float_type, (_DIVISION_VAR_NAME,), fill_value=filler)
        b_variable.long_name = 'B' 
        b_variable.standard_name = 'B' 

        # create the H variable
        h_variable = dataset.createVariable('H', float_type, (_DIVISION_VAR_NAME,), fill_value=filler)
        h_variable.long_name = 'H' 
        h_variable.standard_name = 'H' 

        # process each climatology and indicator variable
        for var_name in variable_names:
            
            # get the variable
            variable = variables[var_name]
            
            # get the divisional array for the variable
            divisional_array = divisional_arrays[var_name]
            
            # copy each division's values into the appropriate variable array slice
            for division_index, division_id in enumerate(divisional_array.keys()):
                
                if division_id in division_ids:
                    netcdf_division_index = division_ids.index(division_id)
                    
                    # make sure the division index isn't out of range for the current variable
                    if netcdf_division_index < variable[:].shape[0]:
                        
                        # pull out the values for the division, reshape to 2-D
                        values = divisional_array[division_id]
                        values = np.reshape(values, (1, values.size))
    
                        # determine time range, either copy all or a slice of the values into the variable
                        initial_year = divisional_minmax_years[var_name][division_id]['min_year']
                        final_year = divisional_minmax_years[var_name][division_id]['max_year']
                        min_year, max_year = minmax_years[var_name]
                        
                        # make sure our year range equals the data's actual year range, otherwise assign it to a slice of the variable
                        if (initial_year == min_year) and (final_year == max_year):
        
                            # copy the values into the variable
                            variable[netcdf_division_index, :] = values
        
                        else:  # we assume the shape of values is same or smaller in each dimension
                            # copy the values into the temperature variable using a time range over the appropriate months
                            start_index = (initial_year - min_year) * 12
                            end_index = total_months - ((max_year - final_year) * 12)
                            variable[netcdf_division_index, start_index:end_index] = values
    
        # copy each division's AWC and latitude values into the corresponding NetCDF variables
        for division_index, division_id in enumerate(division_ids):

            # since the division ID keys are originally from the climatological datasets (temperature and precipitation) 
            # then we'll need to check if a corresponding division exists for the AWC and latitude variables, as these 
            # may contain data for fewer divisions
            if division_id in divs_to_awc.keys():
                awc_value = divs_to_awc[division_id]
            if division_id in divs_to_lats.keys():
                lat_value = divs_to_lats[division_id]
            if division_id in divs_to_bs.keys():
                b_value = divs_to_bs[division_id]
            if division_id in divs_to_hs.keys():
                h_value = divs_to_hs[division_id]

            # set the AWC value for the division
            awc_variable[division_index] = awc_value
            lat_variable[division_index] = lat_value
            b_variable[division_index] = b_value
            h_variable[division_index] = h_value
     
#-----------------------------------------------------------------------------------------------------------------------
def ingest_netcdf_latest(output_netcdf,        # pragma: no cover
                         temp_var_name,
                         precip_var_name,
                         awc_var_name):
    """
    Ingests temperature and precipitation values for nClimDiv datasets. Uses a matching soil constants file 
    from open source indices_python github repository.
    
    :param output_netcdf: file path/name for resulting NetCDF this function will create and return
    :param temp_var_name: temperature variable name in the output NetCDF
    :param precip_var_name: precipitation variable name in the output NetCDF
    :param awc_var_name: available water capacity variable name in the output NetCDF
    """

    # log some timing info, used later for elapsed time
    start_datetime = datetime.now()
    _logger.info("Start time:    %s", start_datetime)

    try:

        # ingest the latest nClimDiv datasets using the processing date specified at the FTP location        
        _ingest_netcdf(output_netcdf,
                       _get_processing_date(),
                       temp_var_name,
                       precip_var_name,
                       awc_var_name)
    except:
        
        _logger.exception('Failed to complete', exc_info=True)
        raise

    # report on the elapsed time
    end_datetime = datetime.now()
    _logger.info("End time:      %s", end_datetime)
    elapsed = end_datetime - start_datetime
    _logger.info("Elapsed time:  %s", elapsed)

#-----------------------------------------------------------------------------------------------------------------------
def _ingest_netcdf(output_netcdf,           # pragma: no cover
                   release_date,
                   temp_var_name,
                   precip_var_name,
                   awc_var_name):
    """
    Ingests temperature and precipitation values for nClimDiv datasets. Uses a matching soil constants file 
    from open source indices_python github repository.
    
    :param output_netcdf: file path/name for resulting NetCDF this function will create and return
    """
    try:
        
        # parse the soil constant (available water capacity)
        soil_url = 'https://raw.githubusercontent.com/monocongo/indices_python/master/example_inputs/pdinew.soilconst'

        # use a temporary file that we'll remove once no longer necessary
        tmp_file = "tmp_soil_for_ingest_nclimdiv.txt"
        utils.retrieve_file(soil_url, tmp_file)
        soil_file = open(tmp_file, 'r')

        # parse the soil constant (available water capacity)
        divs_to_awc, divs_to_lats, divs_to_bs, divs_to_hs = _parse_soil_constants(soil_file, awc_var_name)
        
        # remove the soil file
        soil_file.close()
        os.remove(tmp_file)
        
        # parse both the precipitation and the temperature datasets
        p_divs_to_arrays, p_divs_to_minmax_years, p_min_year, p_max_year = _parse_climatology(release_date, p_or_t='P')
        t_divs_to_arrays, t_divs_to_minmax_years, t_min_year, t_max_year = _parse_climatology(release_date, p_or_t='T')
        
        # determine the number of times and divisions for each (should match?) 
        total_months = (p_max_year - p_min_year + 1) * 12
        if total_months == ((t_max_year - t_min_year + 1) * 12):
            
            # use the intersection set of division IDs (all IDs in both temperature and precipitation)
            division_ids = list(set(list(set(list(p_divs_to_arrays)).intersection(t_divs_to_arrays))).intersection(divs_to_awc))
                                
        else:
            raise ValueError("Unequal number of time steps between the two climatological datasets")
        
        # for each climatology or indicator we'll parse out dictionaries to contain a) divisional arrays (full time series 
        # arrays for all divisions), b) divisional min/max year ranges, and c) overall minimum/maximum years
        divisional_arrays = {temp_var_name: t_divs_to_arrays,
                             precip_var_name: p_divs_to_arrays}
        divisional_minmax_years = {temp_var_name: t_divs_to_minmax_years,
                                   precip_var_name: p_divs_to_minmax_years}
        variable_minmax_years = {temp_var_name: [t_min_year, t_max_year],
                                 precip_var_name: [p_min_year, p_max_year]}
        
        # parse the indicator datasets
        for variable in ['zndx', 'sp01', 'sp02', 'sp03', 'sp06', 'sp12', 'sp24', 'pdsi', 'phdi', 'pmdi']:
            
            # get the relevant US climate divisions ASCII file from NCEI
            #TODO replace this hard coded path with a function parameter, taken from command line  pylint: disable=fixme
            file_url = 'ftp://ftp.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-{0}dv-v1.0.0-{1}'.format(variable, release_date)
        
            # use a temporary file that we'll remove once no longer necessary
            tmp_file = "tmp_climatology_for_ingest_nclimdiv.txt"
            utils.retrieve_file(file_url, tmp_file)
            div_file = open(tmp_file, 'r')
    
            var_name = 'cmb_' + variable
            
            # parse the index values into corresponding dictionaries, arrays, etc.
            divisional_array, minmax_years, min_year, max_year = _parse_results(div_file)
            divisional_arrays[var_name] = divisional_array
            divisional_minmax_years[var_name] = minmax_years
            variable_minmax_years[var_name] = [min_year, max_year]
                                  
            # remove the climatology file
            div_file.close()
            os.remove(tmp_file)

        # write the values as NetCDF
        _create_netcdf(output_netcdf,
                       division_ids,
                       divisional_arrays,
                       divisional_minmax_years,
                       variable_minmax_years,
                       divs_to_awc,
                       divs_to_lats,
                       divs_to_bs,
                       divs_to_hs,
                       total_months,
                       temp_var_name,
                       precip_var_name,
                       awc_var_name,
                       p_min_year)
                
        print('\nMonthly nClimDiv NetCDF file: {0}'.format(output_netcdf))

    except:
        
        _logger.exception('Failed to complete', exc_info=True)
        raise

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    try:

#         # parse the command line arguments
#         parser = argparse.ArgumentParser()
#         parser.add_argument("--base_file_path", 
#                             help="Output file path up to the base file name. For example if this value is /abc/base then the ouput " + \
#                                  "file will be/abc/base_<processing_date>.nc", 
#                             required=True)
#         args = parser.parse_args()
        
        # the NetCDF file to write, result file of this script
#         nclimdiv_netcdf = '{0}_{1}.nc'.format(args.base_file_path, _get_processing_date())
        # TESTING ONLY -- REMOVE
        NCLIMDIV_FILE = '{0}_{1}.nc'.format('C:/home/data/nclimdiv/nclimdiv', _get_processing_date())

        ingest_netcdf_latest(NCLIMDIV_FILE,
                             'tavg',
                             'prcp',
                             'awc')
        
    except:
        _logger.exception('Failed to complete', exc_info=True)
        raise
    