import argparse
from datetime import datetime
import logging
import netCDF4
import numpy as np
import math

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
def join_prism_soil(prism_climatology_file,
                    prism_soil_file,
                    soil_var_name):
    
    # open the NetCDF datasets within a context manager
    with netCDF4.Dataset(prism_climatology_file, 'a') as climatology_dataset, \
         netCDF4.Dataset(prism_soil_file) as soil_dataset:
      
        # get the soil file's data variable
        variable_soil = soil_dataset.variables[soil_var_name]
        
        # get the dimensions from the climatology file's variables, make sure these match with the soil dimensions
        if variable_soil.dimensions != ('lat', 'lon'):
                        
            raise ValueError('Incompatible dimensions')
            
        # make sure we have the same lat/lon values
        lats_soil = soil_dataset.variables['lat'][:]
        lons_soil = soil_dataset.variables['lon'][:]
        lats_climatology = climatology_dataset.variables['lat'][:]
        lons_climatology = climatology_dataset.variables['lon'][:]
        if not np.allclose(lats_soil, lats_climatology, atol=0.05):

            raise ValueError('Incompatible latitude values')
        
        if not np.allclose(lons_soil, lons_climatology, atol=0.05):

            raise ValueError('Incompatible longitude values')

        # make sure that the climatology file doesn't already contain a soil variable
        if soil_var_name not in climatology_dataset.variables:
            
            # create the soil constant (available water capacity) variable
            
            # create the variable, copying the values and attributes from the original soil file's variable
            variable = climatology_dataset.createVariable(soil_var_name, 
                                                          variable_soil.datatype, 
                                                          variable_soil.dimensions)
            variable.setncatts(variable_soil.__dict__)

        else:
            
            variable = climatology_dataset.variables[soil_var_name]
            
            if variable._FillValue is not variable_soil._FillValue and not (math.isnan(variable._FillValue) and math.isnan(variable._FillValue)):
                raise ValueError('Incompatible fill values')
            
            variable.units = variable_soil.units
            variable.least_significant_digit = variable_soil.least_significant_digit            
            variable.valid_min = variable_soil.valid_min            
            variable.valid_max = variable_soil.valid_max            
            variable.standard_name = variable_soil.standard_name            
            variable.long_name = variable_soil.long_name
            
        # copy the soil constant values from the soil file into the climatology file
        variable[:] = variable_soil[:]
                    
            
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    """
    This module is used to perform climate indices processing on nClimGrid datasets in NetCDF.
    """

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        logger.info("Start time:    %s", start_datetime)

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--climatology_file", 
                            help="Climatology dataset file (NetCDF) containing temperature, precipitation, and (possibly) soil values from PRISM", 
                            required=True)
        parser.add_argument("--soil_file", 
                            help="Soil dataset file (NetCDF) containing soil values from PRISM", 
                            required=True)
        parser.add_argument("--var_name_soil", 
                            help="Available water capacity variable name used within the soil NetCDF file", 
                            required=False)
        args = parser.parse_args()

        # perform the merge
        join_prism_soil(args.climatology_file, args.soil_file, args.var_name_soil)
        
        # report on the elapsed time
        end_datetime = datetime.now()
        logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        logger.exception('Failed to complete', exc_info=True)
        raise
