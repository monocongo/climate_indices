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
def create_ascii_from_netcdf(input_netcdf, 
                             output_ascii,
                             states):
    '''
    This function writes indicator values for all climate divisions to its respective ASCII files,
    which is formatted as input for the NCEI monitoring monthly indicator processor.
    
    :param input_netcdf: input NetCDF containing indicator values
    :param output_ascii: output ASCII file for soil constant values
    :param states: 
    '''
    
    with netCDF4.Dataset(input_netcdf, 'r') as input_dataset:
        
        # get the initial and final year of the input datasets
        time_variable = input_dataset.variables['time']
        data_start_year = netCDF4.num2date(time_variable[0], time_variable.units).year
        data_end_year = netCDF4.num2date(time_variable[-1], time_variable.units).year
        total_months = (data_end_year - data_start_year + 1) * 12
        divisions = input_dataset.variables['division'][:]
        total_values = divisions.size * total_months
        
        for variable in input_dataset.variables:

            if variable == 'spi_pearson_01':
                code = '71'
            elif variable == 'spi_pearson_02':
                code = '72'
            elif variable == 'spi_pearson_03':
                code = '73'
            elif variable == 'spi_pearson_06':
                code = '74'
            elif variable == 'spi_pearson_09':
                code = '75'
            elif variable == 'spi_pearson_12':
                code = '76'
            elif variable == 'spi_pearson_24':
                code = '77'
            elif variable == 'pdsi':
                code = '05'
            elif variable == 'phdi':
                code = '06'
            elif variable == 'zindex':
                code = '07'
            else:
                continue
            
            # get the data from NetCDF
            data = input_dataset.variables[variable][:]
            if data.size != total_values:
                raise ValueError('Unexpected variable size: {0}'.format(data.size))
                
            # flatten the data into a single dimension array, easier for iterating over
            data = np.ndarray.flatten(data)
    
            # UPDATE THIS IN PRODUCTION   
            output_ascii = 'new_climdiv-{0}dv-v1.0.0-20161104'.format(variable)

            # write the data as ASCII
            _write_data_as_ascii(data, output_ascii, input_dataset.variables['division'][:], code, states, data_start_year, data_end_year)
    
#-----------------------------------------------------------------------------------------------------------------------
def _write_data_as_ascii(data, 
                         ascii_file,
                         divisions,
                         code,
                         states,
                         start_year=1895):
    '''
    This function writes a time series of monthly data values, assumed to begin at January of the start year
    and end at December of the end year, as ASCII text.
    
    :param data:
    :param ascii_file:
    :param divisions:
    :param start_year:  
    :param end_year:  
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
            state = states[state_id_index]
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
def _parse_states(states_file):

    df = pd.read_csv(states_file, 
                     delim_whitespace=True, 
                     usecols=[0], 
                     names=['state_abbrev'])
    states = df['state_abbrev'].tolist()
    states = list(OrderedDict.fromkeys(states).keys())
    return states
    
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
        create_ascii_from_netcdf(input_netcdf,
                                 output_ascii,
                                 states)
        
    except:
    
        logger.exception('Failed to complete: {}'.format(sys.exc_info()))
        raise
