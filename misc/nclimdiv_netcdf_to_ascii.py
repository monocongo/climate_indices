from collections import OrderedDict
import logging
import sys
import netCDF4
import numpy as np
import numpy.ma as ma
import pandas as pd

# set up a basic, global logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------
def create_ascii(input_netcdf_filepath, 
                 output_ascii_filepath,
                 state_abbrevs):
    '''
    This function writes indicator values for all climate divisions to its respective ASCII files,
    which is formatted as input for the NCEI monitoring monthly indicator processor.
    
    :param input_netcdf_filepath: input NetCDF containing indicator values
    :param output_ascii_filepath: output ASCII file for soil constant values
    :param state_abbrevs: list of state abbreviations
    '''
    
    with netCDF4.Dataset(input_netcdf_filepath, 'r') as input_dataset:
        
        # get the initial and final year of the input datasets
        time_variable = input_dataset.variables['time']
        data_start_year = netCDF4.num2date(time_variable[0], time_variable.units).year
        data_end_year = netCDF4.num2date(time_variable[-1], time_variable.units).year
        total_months = (data_end_year - data_start_year + 1) * 12
        divisions = input_dataset.variables['division'][:]
        total_values = divisions.size * total_months
        
        index_code_dict = {'spi_pearson_01': '71',
                           'spi_pearson_02': '72',
                           'spi_pearson_03': '73',
                           'spi_pearson_06': '74',
                           'spi_pearson_09': '75',
                           'spi_pearson_12': '76',
                           'spi_pearson_24': '77',
                           'pdsi': '05',
                           'phdi': '06',
                           'zindex': '07'}
        for variable in input_dataset.variables:

            if variable in index_code_dict.keys():
                index_code = index_code_dict[variable]
#             if variable == 'spi_pearson_01':
#                 index_code = '71'
#             elif variable == 'spi_pearson_02':
#                 index_code = '72'
#             elif variable == 'spi_pearson_03':
#                 index_code = '73'
#             elif variable == 'spi_pearson_06':
#                 index_code = '74'
#             elif variable == 'spi_pearson_09':
#                 index_code = '75'
#             elif variable == 'spi_pearson_12':
#                 index_code = '76'
#             elif variable == 'spi_pearson_24':
#                 index_code = '77'
#             elif variable == 'pdsi':
#                 index_code = '05'
#             elif variable == 'phdi':
#                 index_code = '06'
#             elif variable == 'zindex':
#                 index_code = '07'
            else:
                continue
            
            # get the data from NetCDF
            data = input_dataset.variables[variable][:]
            if data.size != total_values:
                raise ValueError('Unexpected variable size: {0}'.format(data.size))
                
            # flatten the data into a single dimension array, easier for iterating over
            data = np.ndarray.flatten(data)
    
            # UPDATE THIS IN PRODUCTION   
            output_ascii_filepath = 'new_climdiv-{0}dv-v1.0.0-20161104'.format(variable)

            # write the data as ASCII
            _write_data_as_ascii(data, 
                                 output_ascii_filepath, 
                                 input_dataset.variables['division'][:], 
                                 index_code, 
                                 state_abbrevs, 
                                 data_start_year)
    
#-----------------------------------------------------------------------------------------------------------------------
def _write_data_as_ascii(data, 
                         ascii_file,
                         divisions,
                         code,
                         state_abbrevs,
                         start_year=1895):
    '''
    This function writes a time series of monthly data values, assumed to begin at January of the start year
    and end at December of the end year, as ASCII text.
    
    :param data: UD climate divisions data
    :param ascii_file: output file to which this function will write ASCII values
    :param divisions: US climate divisions
    :param state_abbrevs: list of state abbreviations 
    :param start_year: first year of the dataset
    :param code: index code, such as PDSI  
    '''
    
    # reshape data into (divisions, years, months)
    years = int(len(data) / (divisions.size * 12))
    data = np.reshape(data, (divisions.size, years, 12))
    
    # open the ASCII file for writing
    file = open(ascii_file, "w")
    
    for division_index, division_id in enumerate(divisions):
        
        # loop over each year of the data array, assume monthly values starting at January of the first year and all full years
        for year in range(years):
            
            # write ID, year, etc.
            # fix the following line to conform to the current ASCII format (width of station ID, year, etc.)
            state_id_index = int(str(division_id)[-2:]) - 1
            state = state_abbrevs[state_id_index]
            padded_division_id = str(division_id).zfill(5)
            
            file.write('{0}{1}{2}{3}'.format(state, padded_division_id, code, str(start_year + year)))
            
            # write the twelve monthly values
            for month in range(12):
                value = data[division_index, year, month]
                if value is not ma.masked:
                    file.write(" {:8.2f}".format(value))
                else:
                    file.write(" {:8.2f}".format(-99.99))
            file.write('\n')
        
    file.close()
    
#-----------------------------------------------------------------------------------------------------------------------
def _parse_states(states_csv):
    """
    Parse a file of states into a list of state abbreviations.
    
    :param states_csv: space separated rows with initial column containing state abbreviations 
    """
    df = pd.read_csv(states_csv, 
                     delim_whitespace=True, 
                     usecols=[0], 
                     names=['state_abbrev'])
    state_abbrevs = df['state_abbrev'].tolist()
    state_abbrevs = list(OrderedDict.fromkeys(state_abbrevs).keys())
    return state_abbrevs
    
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    try:
        # get the command line arguments
        input_netcdf = sys.argv[1]
        output_ascii = sys.argv[2]
        states_file = sys.argv[3]
        
        # get the states that map to the climate divisions
        states = _parse_states(states_file)
        
        # call the function that creates ASCII files from data read from corresponding NetCDFs
        create_ascii(input_netcdf,
                                 output_ascii,
                                 states)
        
    except:
    
        logger.exception('Failed to complete: %s', sys.exc_info())
        raise
