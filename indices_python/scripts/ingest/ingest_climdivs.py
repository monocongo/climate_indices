import argparse
import logging
import netcdf_utils
import os
import math
import pandas as pd
import sys

# set up a basic, global logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------
def parse_variable(variable_file):
    '''
    Parses a data variable's ASCII file, returning 1) a dictionary of division IDs to the corresponding 1-D array
    of monthly values, 2) a dictionary of division IDs to the corresponding minimum and maximum years of the data
    array for the division, 3) the minimum year for all divisions, and 4) the maximum year for all divisions.
    
    :param variable_file
    :return:  
    '''
    
    # value used to represent missing values in the ASCII file
    fill_value = -99.99

    # use a list of column names to clue in pandas as to the structure of the ASCII rows
    column_names = ['division', 'year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    # read the file into a pandas DataFrame object  
    results_df = pd.read_csv(variable_file, 
                             delim_whitespace=True,
                             header=None,
                             names=column_names,
                             dtype={'division': int,
                                    'year': int},
                             na_values=fill_value)
    
    # find the minimum and maximum years for each division, each as a Series with the division ID as index
    division_min_years = results_df.groupby(by='division').min().year
    division_max_years = results_df.groupby(by='division').max().year
    minmax_years_df = pd.DataFrame()
    minmax_years_df['min_year'] = division_min_years
    minmax_years_df['max_year'] = division_max_years
    
    # get a dictionary of division IDs mapped to a list of row numbers corresponding to the division
    division_groups = results_df.groupby(by='division').groups
    
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
def parse_soil(soil_file):
    '''
    Parse the soil constant file, reading both AWC and latitude values for each division.
    
    :param soil_file: ASCII file containing one line per climate division and with fixed width fields:
                      (1) division ID, (2) available water capacity, (3) foo, (4) bar, and (5) the negative tangent 
                      of the latitude of the climate division's centroid
    :return two dictionaries, both with division IDs as keys, the first with AWCs as values, the second with latitudes as values
    '''

    # use a list of column names to clue in pandas as to the structure of the ASCII rows
    column_names = ['division_id', 'awc', 'B', 'H', 'neg_tan_lat']

    # read the file into a pandas DataFrame object  
    results_df = pd.read_fwf(soil_file, 
                             widths=[5, 8, 8, 8, 8],
                             header=None,
                             names=column_names,
                             usecols=column_names)

    
    # make the division ID our index
    results_df.set_index('division_id', drop=True, inplace=True)
    
    # convert the negative tangent of latitude to latitude degrees north
    results_df['lat'] = results_df['neg_tan_lat'].apply(lambda x: math.degrees(math.atan(-1 * x)))

    # drop the column holding the negative tangent of the latitude, no longer needed
    results_df.drop('neg_tan_lat', axis=1, inplace=True)
    
    # produce a dictionary mapping division IDs to corresponding AWC, latitude, B, and H values
    divisions_to_awc = results_df['awc'].to_dict()
    divisions_to_lats = results_df['lat'].to_dict()
    divisions_to_Bs = results_df['B'].to_dict()
    divisions_to_Hs = results_df['H'].to_dict()
    
    return divisions_to_awc, divisions_to_lats, divisions_to_Bs, divisions_to_Hs
    
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    try:

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_dir", 
                            help="Directory containing ASCII files with climatology and water balance values", 
                            required=True)
        parser.add_argument("--soil_file", 
                            help="ASCII file with available water capacity (AWC) and latitude values", 
                            required=True)
        parser.add_argument("--out_file", 
                            help="NetCDF output file containing variables composed from the input data and soil files", 
                            required=True)
        args = parser.parse_args()
        
        # parse the temperature to get the years range of the period of record, values array for each division, etc.
        divs_to_arrays, divs_to_minmax_years, min_year, max_year = parse_variable(os.sep.join((args.data_dir, 'tdat.div')))
        
        # create the output NetCDF using the divisions and times from the temperature dataset, 
        # under the assumption that the remaining datasets to be ingested will share these attributes
        netcdf_utils.create_dataset_climdivs(args.out_file,
                                             list(divs_to_arrays.keys()),
                                             min_year,
                                             (max_year - min_year + 1) * 12)
        
        # add the temperature variable into the NetCDF
        variable_attributes =  {'long_name': 'Temperature, average',
                                'standard_name': 'tavg',
                                'units': 'degrees Fahrentheit'}
        netcdf_utils.add_variable_climdivs_divstime(args.out_file,
                                                    'tdat',
                                                    variable_attributes,
                                                    divs_to_arrays)
        
        # parse the soil constant (available water capacity, lower level) and latitude for the divisions
        divisions_to_awc, divisions_to_lats, divisions_to_Bs, divisions_to_Hs = parse_soil(args.soil_file)

        # create variables for AWC and latitude
        variable_attributes =  {'long_name': 'Available water capacity',
                                'standard_name': 'awc',
                                'units': 'inches'}
        netcdf_utils.add_variable_climdivs_divs(args.out_file,
                                                'awc',
                                                variable_attributes,
                                                divisions_to_awc)
        variable_attributes =  {'long_name': 'Latitude',
                                'standard_name': 'latitude',
                                'units': 'degrees north'}
        netcdf_utils.add_variable_climdivs_divs(args.out_file,
                                                'lat',
                                                variable_attributes,
                                                divisions_to_lats)
        variable_attributes =  {'standard_name': 'B'}
        netcdf_utils.add_variable_climdivs_divs(args.out_file,
                                                'B',
                                                variable_attributes,
                                                divisions_to_Bs)
        variable_attributes =  {'standard_name': 'H'}
        netcdf_utils.add_variable_climdivs_divs(args.out_file,
                                                'H',
                                                variable_attributes,
                                                divisions_to_Hs)
        
        # create variable for the remaining intermediate (water balance) variables
        for variable_name in ['etdat', 'pdat', 'pedat', 'pldat', 'prdat', 'rdat', 'rodat', 'spdat', 'sssdat', 
                              'ssudat', 'tldat', 'cp.index', 'pdsi.index', 'phdi.index', 'pmdi.index', 'z.index', 
                              'x1dat',  'x2dat', 'x3dat', 'phat']:

            # TODO these attributes only applicable for water balance variables, 
            # update units etc. when ingesting all intermediates such as X1, X2, Z-index, etc.
            variable_attributes =  {'standard_name': variable_name,
                                    'units': 'inches'}

            # parse the ASCII file
            divs_to_arrays, divs_to_minmax_years, min_year, max_year = \
                parse_variable(os.sep.join((args.data_dir, variable_name + '.div')))

            # create the variable within the NetCDF
            netcdf_utils.add_variable_climdivs_divstime(args.out_file,
                                                        variable_name,
                                                        variable_attributes,
                                                        divs_to_arrays)
            
    except:
    
        logger.exception('Failed to complete: {}'.format(sys.exc_info()))
        raise
