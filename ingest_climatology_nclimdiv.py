import argparse
import logging
import netcdf_utils
import math
import pandas as pd
import sys

'''
run with arguments like below:

--temp_file C:/home/data/nclimdiv/climdiv-tmpcdv-v1.0.0-20170906 
--precip_file C:/home/data/nclimdiv/climdiv-pcpndv-v1.0.0-20170906 
--pdsi_file C:/home/data/nclimdiv/climdiv-pdsidv-v1.0.0-20170906 
--phdi_file C:/home/data/nclimdiv/climdiv-phdidv-v1.0.0-20170906 
--pmdi_file C:/home/data/nclimdiv/climdiv-pmdidv-v1.0.0-20170906 
--zindex_file C:/home/data/nclimdiv/climdiv-zndxdv-v1.0.0-20170906 
--soil_file C:/home/data/nclimdiv/pdinew.soilconst 
--out_file C:/home/data/nclimdiv/climdiv-climdv-v1.0.0-20170906.nc

'''
#-----------------------------------------------------------------------------
# set up a basic, global logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------
def parse_variable(variable_file):
    '''
    Parses a data variable's ASCII file, returning 1) a dictionary of division IDs 
    to the corresponding 1-D array of monthly values, 2) the minimum year for all 
    divisions, and 3) the maximum year for all divisions.
    
    :param variable_file
    :return:  
    '''
    
    # value used to represent missing values in the ASCII file
    fill_value = -99.99

    # use a list of column names to clue in pandas as to the structure of the ASCII rows
    column_names = ['division', 'year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    col_specs = [(0, 4), (6, 10), (10, 17), (17, 24), (24, 31), (31, 38), (38, 45), (45, 52), (52, 59), (59, 66), (66, 73), (73, 81), (81, 88), (88, -1)]

    # parse the ASCII file as fixed-width using the specs and column names set out above
    results_df = pd.read_fwf(variable_file, 
                             colspecs=col_specs,
                             names=column_names,
                             converters = {0: int, 1: int},  # similar to setting the dtype, which is not supported although listed in the docs for this function
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
    month_column_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    for division_id, row_numbers in division_groups.items():
        rows = results_df.iloc[row_numbers]
        years_months_data = rows.as_matrix(columns=month_column_names)
        divs_to_arrays[division_id] = years_months_data.flatten()
    
    # return the two dictionaries we've built
    return divs_to_arrays, division_min_years.min(), division_max_years.max()

#-----------------------------------------------------------------------------------------------------------------------
def parse_soil(soil_file):
    '''
    Parse the soil constant file, reading both AWC and latitude values for each division.
    
    :param soil_file: ASCII file containing one line per climate division and with fixed width fields:
                      (1) division ID, (2) available water capacity, (3) B, (4) H, and (5) the negative 
                      tangent of the latitude of the climate division's centroid
    :return two dictionaries, both with division IDs as keys, the first with AWCs as values, and the second 
            with latitudes as values
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
    divs_to_awc = results_df['awc'].to_dict()
    divs_to_lats = results_df['lat'].to_dict()
    divs_to_Bs = results_df['B'].to_dict()
    divs_to_Hs = results_df['H'].to_dict()
    
    return divs_to_awc, divs_to_lats, divs_to_Bs, divs_to_Hs

#-----------------------------------------------------------------------------------------------------------------------
def add_variable_to_netcdf(variable_file_ascii,
                           var_name,
                           long_name,
                           units,
                           out_file,
                           create=False):

    # parse the temperature to get the years range of the period of record, values array for each division, etc.
    divs_to_arrays, min_year, max_year = parse_variable(variable_file_ascii)
    
    # create the output NetCDF using the divisions and times from the temperature dataset, 
    # under the assumption that the remaining datasets to be ingested will share these attributes
    if create:
        netcdf_utils.create_dataset_climdivs(out_file,
                                             list(divs_to_arrays.keys()),
                                             min_year,
                                             (max_year - min_year + 1) * 12)
    
    # add the temperature variable into the NetCDF
    variable_attributes =  {'long_name': long_name,
                            'standard_name': var_name,
                            'units': units}
    netcdf_utils.add_variable_climdivs_divstime(out_file,
                                                var_name,
                                                variable_attributes,
                                                divs_to_arrays)
    
#-----------------------------------------------------------------------------------------------------------------------
def main(temp_file,
         precip_file,
         soil_file,
         pdsi_file,
         phdi_file,
         pmdi_file,
         zindex_file,
         out_file):

    # parse the temperature and precipitation files to NetCDF
    add_variable_to_netcdf(temp_file,
                           'tavg', 
                           'Temperature, average', 
                           'degrees Fahrenheit',
                           out_file,
                           True)
    add_variable_to_netcdf(precip_file,
                           'prcp', 
                           'Precipitation, cumulative', 
                           'millimeters',
                           out_file,
                           False)
    add_variable_to_netcdf(pdsi_file,
                           'pdsi.index', 
                           'PDSI from NCEI', 
                           'no units',
                           out_file,
                           False)
    add_variable_to_netcdf(phdi_file,
                           'phdi.index', 
                           'PHDI from NCEI', 
                           'no units',
                           out_file,
                           False)
    add_variable_to_netcdf(pmdi_file,
                           'pmdi.index', 
                           'PMDI from NCEI', 
                           'no units',
                           out_file,
                           False)
    add_variable_to_netcdf(zindex_file,
                           'z.index', 
                           'Z-Index from NCEI', 
                           'no units',
                           out_file,
                           False)
    
    # parse the soil constant (available water capacity, lower level) and latitude for the divisions
    divs_to_awc, divs_to_lats, divs_to_Bs, divs_to_Hs = parse_soil(soil_file)

    # create variables for AWC, latitude, B, and H
    variable_attributes =  {'long_name': 'Available water capacity',
                            'standard_name': 'awc',
                            'units': 'inches'}
    netcdf_utils.add_variable_climdivs_divs(out_file,
                                            'awc',
                                            variable_attributes,
                                            divs_to_awc)
    variable_attributes =  {'long_name': 'Latitude',
                            'standard_name': 'latitude',
                            'units': 'degrees north'}
    netcdf_utils.add_variable_climdivs_divs(out_file,
                                            'lat',
                                            variable_attributes,
                                            divs_to_lats)
    variable_attributes =  {'standard_name': 'B'}
    netcdf_utils.add_variable_climdivs_divs(out_file,
                                            'B',
                                            variable_attributes,
                                            divs_to_Bs)
    variable_attributes =  {'standard_name': 'H'}
    netcdf_utils.add_variable_climdivs_divs(out_file,
                                            'H',
                                            variable_attributes,
                                            divs_to_Hs)
    
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    '''
    Ingest nClimDiv files in ASCII format to a NetCDF version 4 file.
    
    Process:
    
    1. Download monthly data files from ftp://ftp.ncdc.noaa.gov/pub/data/cirs/climdiv/
    
    C:/home/data/nclimdiv/climdiv-tmpcdv-v1.0.0-20170906 
    C:/home/data/nclimdiv/climdiv-pcpndv-v1.0.0-20170906 
    
    2. Download soil constants file from https://github.com/monocongo/indices_python/tree/climdivs_comparison/example_inputs/pdinew.soilconst
    
    C:/home/data/nclimdiv/pdinew.soilconst
    
    3. Run the ingest code from https://github.com/monocongo/indices_python/tree/climdivs_comparison/ingest_climatology_nclimdiv.py
    
    python -u ingest_climatology_nclimdiv.py \
      --temp_file C:/home/data/nclimdiv/climdiv-tmpcdv-v1.0.0-20170906 \
      --precip_file C:/home/data/nclimdiv/climdiv-pcpndv-v1.0.0-20170906 \
      --soil_file C:/home/data/nclimdiv/pdinew.soilconst \
      --out_file C:/home/data/nclimdiv/climdiv-climdv-v1.0.0-20170906.nc
      
    Use the above result NetCDF file as input to NIDIS/NCEI Palmer code, etc.
    '''
    try:

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--temp_file", 
                            help="ASCII file with temperature values", 
                            required=True)
        parser.add_argument("--precip_file", 
                            help="ASCII file with precipitation values", 
                            required=True)
        parser.add_argument("--soil_file", 
                            help="ASCII file with available water capacity (AWC) and latitude values", 
                            required=True)
        parser.add_argument("--pdsi_file", 
                            help="ASCII file with PDSI values", 
                            required=True)
        parser.add_argument("--phdi_file", 
                            help="ASCII file with PHDI values", 
                            required=True)
        parser.add_argument("--pmdi_file", 
                            help="ASCII file with PMDI values", 
                            required=True)
        parser.add_argument("--zindex_file", 
                            help="ASCII file with Z-Index values", 
                            required=True)
        parser.add_argument("--out_file", 
                            help="NetCDF output file containing variables composed from the input data and soil files", 
                            required=True)
        args = parser.parse_args()

        main(args.temp_file, args.precip_file, args.soil_file, args.pdsi_file, args.phdi_file, args.pmdi_file, args.zindex_file, args.out_file)
                
    except:
    
        logger.exception('Failed to complete: {}'.format(sys.exc_info()))
        raise
