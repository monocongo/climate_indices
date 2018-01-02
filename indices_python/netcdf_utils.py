from datetime import datetime
import logging
import netCDF4
import numpy as np
import os
import random

from indices_python import utils

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global _logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
_logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
def convert_and_move_netcdf(input_and_output_netcdfs):   # pragma: no cover
    
    input_netcdf = input_and_output_netcdfs[0]
    output_netcdf = input_and_output_netcdfs[1]
  
    try:
        # use NCO bindings to make conversion/compression command    
        import nco
        nco = nco.Nco()
        nco.ncks(input=[input_netcdf, output_netcdf],
                 output=output_netcdf,
                 options=['-O', '-4', '-L 4', '-h'])
          
        # remove the temporary/work file which will no longer needed
        _logger.info('Removing the temporary/work file [%s]', input_netcdf)
        os.remove(input_netcdf)

    except ImportError:
    
        _logger.warning('NCO unavailable, skipping conversion/move')

# #-----------------------------------------------------------------------------------------------------------------------
# def _compute_days(initial_year,
#                   total_months,
#                   initial_month=1,
#                   units_start_year=1800):
#     '''
#     Computes the "number of days" equivalent for regular, incremental monthly time steps given an initial year/month.
#     Useful when using "days since <start_date>" as time units within a NetCDF dataset.
#     
#     :param initial_year: the initial year from which the day values should start, i.e. the first value in the output
#                         array will correspond to the number of days between January of this initial year since January 
#                         of the units start year
#     :param initial_month: the month within the initial year from which the day values should start, with 1: January, 2: February, etc.
#     :param total_months: the total number of monthly increments (time steps measured in days) to be computed
#     :param units_start_year: the start year from which the monthly increments are computed, with time steps measured
#                              in days since January of this starting year 
#     :return: an array of time step increments, measured in days since midnight of January 1st of the units start year
#     :rtype: ndarray of ints 
#     '''
# 
#     # compute an offset from which the day values should begin 
#     start_date = datetime(units_start_year, 1, 1)
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
#         # cook up a datetime object for the current time step (month)
#         current_date = datetime(initial_year + years, 1 + months, 1)
#         
#         # get the number of days since the initial date
#         days[i] = (current_date - start_date).days
#     
#     return days
#
#-----------------------------------------------------------------------------------------------------------------------
def find_netcdf_datatype(data_object):
    
    if isinstance(data_object, netCDF4.Variable):

        if data_object.dtype == 'float16':
            netcdf_datatype = 'f2'
        elif data_object.dtype == 'float32':
            netcdf_datatype = 'f4'
        elif data_object.dtype == 'float64':
            netcdf_datatype = 'f8'
        elif data_object.dtype == 'int16':
            netcdf_datatype = 'i2'
        elif data_object.dtype == 'int32':
            netcdf_datatype = 'i4'
        else:
            raise ValueError('Unsupported data type: {}'.format(data_object.dtype))
    
    elif isinstance(data_object, float):

        netcdf_datatype = 'f8'
        
    elif isinstance(data_object, np.float32):

        netcdf_datatype = 'f4'
        
    elif isinstance(data_object, np.float64):

        netcdf_datatype = 'f8'
        
    elif isinstance(data_object, int):

        netcdf_datatype = 'i4'
        
    elif isinstance(data_object, np.int16):

        netcdf_datatype = 'i2'
        
    elif isinstance(data_object, np.int32):

        netcdf_datatype = 'i4'
        
    else:
        raise ValueError('Unsupported argument type: {}'.format(type(data_object)))
    
    return netcdf_datatype
    
#-----------------------------------------------------------------------------------------------------------------------
def create_dataset_climdivs(file_path,     # pragma: no cover
                            division_ids,
                            initial_year,
                            total_months):
    
    # create/open the output file for writing, set its dimensions and coordinate variables
    with netCDF4.Dataset(file_path, 'w') as dataset:

        # set some global group attributes
        dataset.title = 'US Climate Divisions'
        dataset.source = 'conversion from data set files provided by CMB'
        dataset.institution = 'National Centers for Environmental Information (NCEI), NESDIS, NOAA, U.S. Department of Commerce'
        dataset.standard_name_vocabulary = 'CF Standard Name Table (v26, 08 November 2013)'
        dataset.date_created = str(datetime.now())
        dataset.date_modified = str(datetime.now())
        dataset.Conventions = 'ClimDiv-1.0'  # suggested by Steve Ansari for support within the NOAA WCT

        # create the time and division dimensions
        dataset.createDimension('time', None)
        dataset.createDimension('division', len(division_ids))
        
        # create the time and division coordinate variables
        int_dtype = 'i4'
        time_variable = dataset.createVariable('time', int_dtype, ('time',))
        divisions_variable = dataset.createVariable('division', int_dtype, ('division',))
        
        # set the coordinate variables' attributes
        units_start_year = 1800
        time_variable.setncatts({'long_name': 'time',
                                 'standard_name': 'time',
                                 'calendar': 'gregorian',
                                 'units': 'days since ' + str(units_start_year) + '-01-01 00:00:00'})
        divisions_variable.setncatts({'long_name': 'US climate division ID',
                                      'standard_name': 'division ID'})
        
        # set the coordinate variables' values
        time_variable[:] = utils.compute_days(initial_year, total_months, 1, units_start_year)
        divisions_variable[:] = np.array(sorted(division_ids), dtype=np.dtype(int))
    
#-----------------------------------------------------------------------------------------------------------------------
def initialize_netcdf_single_variable_grid(file_path,              # pragma: no cover
                                           template_netcdf,
                                           variable_name,
                                           variable_long_name,
                                           valid_min,
                                           valid_max,
                                           variable_units=None,
                                           fill_value=np.float32(np.NaN)):
    '''
    This function is used to initialize and return a netCDF4.Dataset object, containing a single data variable having 
    dimensions (time, lat, lon). The input data values array is assumed to be a 3-D array with indices corresponding to 
    the variable dimensions. The latitude, longitude, and time values are copied from the template NetCDF, which is 
    assumed to have dimension sizes matching to the axes of the variable values array. Global attributes are also copied 
    from the template NetCDF.
    
    :param file_path: the file path/name of the NetCDF Dataset object returned by this function
    :param template_dataset: an existing/open NetCDF Dataset object which will be used as a template for the Dataset
                             that will be created by this function
    :param variable_name: the variable name which will be used to identify the main data variable within the Dataset
    :param variable_long_name: the long name attribute of the main data variable within the Dataset
    :param variable_units: string specifying the units of the variable 
    :param valid_min: the minimum value to which the main data variable of the resulting Dataset(s) will be clipped
    :param valid_max: the maximum value to which the main data variable of the resulting Dataset(s) will be clipped
    :param fill_value: the fill value to use for main data variable of the resulting Dataset(s)
    :return: an open netCDF4.Dataset object
    '''

    with netCDF4.Dataset(template_netcdf, 'r') as template_dataset:
 
        # get the template's dimension sizes
        lat_size = template_dataset.variables['lat'].size
        lon_size = template_dataset.variables['lon'].size
    
        # make a basic set of variable attributes
        variable_attributes = {'valid_min' : valid_min,
                               'valid_max' : valid_max,
                               'long_name' : variable_long_name}
        if variable_units is not None:
            variable_attributes['units'] = variable_units
            
        # open the dataset as a NetCDF in write mode
        dataset = netCDF4.Dataset(file_path, 'w')
        
        # copy the global attributes from the input
        # TODO/FIXME add/modify global attributes to correspond with the actual dataset
        dataset.setncatts(template_dataset.__dict__)
        
        # create the time, x, and y dimensions
        dataset.createDimension('time', None)
        dataset.createDimension('lat', lat_size)
        dataset.createDimension('lon', lon_size)
    
        # get the appropriate data types to use for the variables
        time_dtype = find_netcdf_datatype(template_dataset.variables['time'])
        lat_dtype = find_netcdf_datatype(template_dataset.variables['lat'])
        lon_dtype = find_netcdf_datatype(template_dataset.variables['lon'])
        data_dtype = find_netcdf_datatype(fill_value)
    
        # create the variables
        time_variable = dataset.createVariable('time', time_dtype, ('time',))
        y_variable = dataset.createVariable('lat', lat_dtype, ('lat',))
        x_variable = dataset.createVariable('lon', lon_dtype, ('lon',))
        data_variable = dataset.createVariable(variable_name,
                                               data_dtype,
                                               ('time', 'lat', 'lon',),
                                               fill_value=fill_value, 
                                               zlib=False)
    
        # set the variables' attributes
        time_variable.setncatts(template_dataset.variables['time'].__dict__)
        y_variable.setncatts(template_dataset.variables['lat'].__dict__)
        x_variable.setncatts(template_dataset.variables['lon'].__dict__)
        data_variable.setncatts(variable_attributes)
    
        # set the coordinate variables' values
        time_variable[:] = template_dataset.variables['time'][:]
        y_variable[:] = template_dataset.variables['lat'][:]
        x_variable[:] = template_dataset.variables['lon'][:]

        # close the NetCDF
        dataset.close()

#-----------------------------------------------------------------------------------------------------------------------
def initialize_dataset(file_path,           # pragma: no cover
                       template_dataset,
                       x_dim_name,
                       y_dim_name,
                       data_variable_name=None,
                       data_variable_attributes=None,
                       data_fill_value=np.NaN):
    
    # make sure the data matches the dimensions
    y_size = template_dataset.variables[y_dim_name].size
    x_size = template_dataset.variables[x_dim_name].size
    
    # open the output file for writing, set its dimensions and variables
    netcdf = netCDF4.Dataset(file_path, 'w')

    # copy the global attributes from the template
    netcdf.setncatts(template_dataset.__dict__)
        
    # create the time, x, and y dimensions
    netcdf.createDimension('time', None)
    netcdf.createDimension(x_dim_name, x_size)
    netcdf.createDimension(y_dim_name, y_size)
    
    # get the appropriate data types to use for the variables based on the values arrays
    time_dtype = find_netcdf_datatype(template_dataset.variables['time'])
    x_dtype = find_netcdf_datatype(template_dataset.variables[x_dim_name])
    y_dtype = find_netcdf_datatype(template_dataset.variables[y_dim_name])
    
    # create the variables
    time_variable = netcdf.createVariable('time', time_dtype, ('time',))
    x_variable = netcdf.createVariable(x_dim_name, x_dtype, (x_dim_name,))
    y_variable = netcdf.createVariable(y_dim_name, y_dtype, (y_dim_name,))
    
    # set the variables' attributes
    time_variable.setncatts(template_dataset.variables['time'].__dict__)
    x_variable.setncatts(template_dataset.variables[x_dim_name].__dict__)
    y_variable.setncatts(template_dataset.variables[y_dim_name].__dict__)
    
    # set the coordinate variables' values
    time_variable[:] = template_dataset.variables['time'][:]
    x_variable[:] = template_dataset.variables[x_dim_name][:]
    y_variable[:] = template_dataset.variables[y_dim_name][:]

    if data_variable_name is not None:
        
        data_dtype = find_netcdf_datatype(data_fill_value)
        data_variable = netcdf.createVariable(data_variable_name, 
                                              data_dtype, 
                                              ('time', x_dim_name, y_dim_name,), 
                                              fill_value=data_fill_value)        
        if data_variable_attributes is not None:

            data_variable.setncatts(data_variable_attributes)

    return netcdf
    
#-----------------------------------------------------------------------------------------------------------------------
def initialize_dataset_climdivs(file_path,            # pragma: no cover
                                template_dataset,
                                divisions_dim_name,
                                data_variable_name=None,
                                data_variable_attributes=None,
                                data_fill_value=np.NaN):
    
    # make sure the data matches the dimensions
    divisions_size = template_dataset.variables[divisions_dim_name].size
    
    # open the output file for writing, set its dimensions and variables
    netcdf = netCDF4.Dataset(file_path, 'w')

    # copy the global attributes from the template
    netcdf.setncatts(template_dataset.__dict__)
        
    # copy the global attributes from the input
    # TODO/FIXME add/modify global attributes to correspond with the actual dataset
    netcdf.setncatts(template_dataset.__dict__)
    
    # use "ClimDiv-1.0" as the Conventions setting in order to facilitate visualization by the NOAA Weather and Climate Toolkit
    netcdf.setncattr("Conventions", "ClimDiv-1.0")
        
    # create the time, x, and y dimensions
    netcdf.createDimension('time', None)
    netcdf.createDimension(divisions_dim_name, divisions_size)
    
    # get the appropriate data types to use for the variables based on the values arrays
    time_dtype = find_netcdf_datatype(template_dataset.variables['time'])
    divisions_dtype = find_netcdf_datatype(template_dataset.variables[divisions_dim_name])
    
    # create the variables
    time_variable = netcdf.createVariable('time', time_dtype, ('time',))
    divisions_variable = netcdf.createVariable(divisions_dim_name, divisions_dtype, (divisions_dim_name,))
    
    # set the variables' attributes
    time_variable.setncatts(template_dataset.variables['time'].__dict__)
    divisions_variable.setncatts(template_dataset.variables[divisions_dim_name].__dict__)
    
    # set the coordinate variables' values
    time_variable[:] = template_dataset.variables['time'][:]
    divisions_variable[:] = template_dataset.variables[divisions_dim_name][:]

    if (data_variable_name is not None):
        
        data_dtype = find_netcdf_datatype(data_fill_value)
        data_variable = netcdf.createVariable(data_variable_name, 
                                              data_dtype, 
                                              ('time', divisions_dim_name,), 
                                              fill_value=data_fill_value)        
        data_variable.setncatts(data_variable_attributes)

    return netcdf
    
#-----------------------------------------------------------------------------------------------------------------------
def create_variable_grid(netcdf,                     # pragma: no cover
                         data_variable_name,
                         data_variable_attributes,
                         data_fill_value=np.NaN):
    
    #TODO fix these to come from the dataset's dimension attributes?
    x_dim_name = 'lon'
    y_dim_name = 'lat'
    
    # get the appropriate data types to use for the variables based on the values arrays
    data_dtype = find_netcdf_datatype(data_fill_value)
    
    # create the variable
    data_variable = netcdf.createVariable(data_variable_name, 
                                          data_dtype, 
                                          ('time', x_dim_name, y_dim_name,), 
                                          fill_value=data_fill_value)
    
    # set the variable's attributes
    data_variable.setncatts(data_variable_attributes)

    return netcdf

#-----------------------------------------------------------------------------------------------------------------------
def add_variable_climgrid(file_path,
                          variable_name,
                          variable_attributes,
                          variable_array,
                          y_dim_name='lat', 
                          x_dim_name='lon'):
    '''
    Adds a three-dimensional (time, y_dim, x_dim) variable to an existing climate divisions NetCDF. The variable is created 
    and populated with the provided data values. 
    
    :param file_path: existing NetCDF to which the variable will be added. This NetCDF is assumed to contain 
                      the dimensions "time", y_dim, and x_dim (corresponding to the arguments) as well as corresponding coordinate variables.
    :param variable_name: name of the new variable to be added, a variable with this name should not already exist
    :param variable_attributes: the attributes that should be assigned to the new variable
    :param variable_array: an array of values, with shape (times, y_dim, x_dim).
                           The number of elements within the time dimension of the arrays should match with the number  
                           of time steps of the existing NetCDF being added to (as specified by the time coordinate variable).
    '''
    
    #TODO/FIXME add checks of the array's dimensions to make sure they match with the variable's dimensions, etc.
    
    # open the output file in append mode for writing, set its dimensions and coordinate variables
    with netCDF4.Dataset(file_path, 'a') as dataset:

        # make sure that the variable name isn't already in use
        if variable_name in dataset.variables.keys():
            
            variable = dataset.variables[variable_name]
            
#             message = 'Variable name \'{0}\' is already being used within the NetCDF file \'{1}\''.format(variable_name, file_path)
#             _logger.error(message)
#             raise ValueError(message)
            
        else:

            # get the NetCDF datatype applicable to the data array we'll store in the variable
            netcdf_data_type = find_netcdf_datatype(np.NaN)
    
            # create the variable
            variable = dataset.createVariable(variable_name, 
                                              netcdf_data_type, 
                                              ('time', y_dim_name, x_dim_name), 
                                              fill_value=np.NaN)
            
        # set the attributes
        if variable_attributes is not None:
            variable.setncatts(variable_attributes)
        
        # assign the array into the variable
        variable[:, :, :] = variable_array
            
#-----------------------------------------------------------------------------------------------------------------------
def add_variable_climdivs(file_path,
                          variable_name,
                          variable_attributes,
                          divisions_to_arrays):
    '''
    Adds a two-dimensional (division, time) variable to an existing climate divisions NetCDF. The variable is created 
    and populated with the provided data values. 
    
    :param file_path: existing NetCDF to which the variable will be added. This NetCDF is assumed to contain 
                      the dimensions "division" and "time" as well as corresponding coordinate variables.
    :param variable_name: name of the new variable to be added, a variable with this name should not already exist
    :param variable_attributes: the attributes that should be assigned to the new variable
    :param divisions_to_arrays: a dictionary with division indices as keys and corresponding 1-D Numpy arrays as values.
                               The number of elements within the arrays should match with the number of time steps 
                               of the existing NetCDF being added to (as specified by the time coordinate variable).
    '''
    
    # get the NetCDF datatype applicable to the data array we'll store in the variable
    random_array = random.choice(list(divisions_to_arrays.values()))
    netcdf_data_type = find_netcdf_datatype(random_array[0])
    
    # open the output file in append mode for writing, set its dimensions and coordinate variables
    with netCDF4.Dataset(file_path, 'a') as dataset:

        # make sure that the variable name isn't already in use
        if variable_name in dataset.variables.keys():
            
            message = 'Variable name \'{0}\' is already being used within the NetCDF file \'{1}\''.format(variable_name, file_path)
            _logger.error(message)
            raise ValueError(message)
            
        # create the variable, set the attributes
        variable = dataset.createVariable(variable_name, 
                                          netcdf_data_type, 
                                          ('division', 'time',), 
                                          fill_value=np.NaN)
        variable.setncatts(variable_attributes)
    
        # get the total number of time steps
        times_size = dataset.variables['time'][:].size
        
        # loop over each existing division and add the corresponding data array, if one was provided
        for division_index in range(dataset.variables['division'][:].size()):
            
            # make sure we have a data array of monthly values for this division
            if division_index in divisions_to_arrays.keys():

                # make sure the array has the expected number of time steps 
                data_array = divisions_to_arrays[division_index]
                if data_array.size == times_size:
                
                    # assign the array into the current division's slot in the variable
                    variable[division_index, :] = np.reshape(data_array, (1, times_size))

                else:

                    _logger.info('Unexpected size of data array for division index {0} -- '.format(division_index) + 
                                'expected {0} time steps but the array contains {1}'.format(times_size, data_array.size))
            
#-----------------------------------------------------------------------------------------------------------------------
def add_variable_climdivs_divstime(file_path,
                                   variable_name,
                                   variable_attributes,
                                   divisions_to_arrays):
    '''
    Adds a two-dimensional (division, time) variable to an existing climate divisions NetCDF. The variable is created 
    and populated with the provided data values. 
    
    :param file_path: existing NetCDF to which the variable will be added. This NetCDF is assumed to contain 
                      the dimensions "division" and "time" as well as corresponding coordinate variables.
    :param variable_name: name of the new variable to be added, a variable with this name should not already exist
    :param variable_attributes: the attributes that should be assigned to the new variable
    :param divisions_to_arrays: a dictionary with division IDs as keys and corresponding 1-D Numpy arrays as values.
                               The number of elements within the arrays should match with the number of time steps 
                               of the existing NetCDF being added to (as specified by the time coordinate variable).
    '''
    
    # get the NetCDF datatype applicable to the data array we'll store in the variable
    random_array = random.choice(list(divisions_to_arrays.values()))
    netcdf_data_type = find_netcdf_datatype(random_array[0])
    
    # open the output file in append mode for writing, set its dimensions and coordinate variables
    with netCDF4.Dataset(file_path, 'a') as dataset:

        # make sure that the variable name isn't already in use
        if variable_name in dataset.variables.keys():
            
            message = 'Variable name \'{0}\' is already being used within the NetCDF file \'{1}\''.format(variable_name, file_path)
            _logger.error(message)
            raise ValueError(message)
            
        # create the variable, set the attributes
        variable = dataset.createVariable(variable_name, 
                                          netcdf_data_type, 
                                          ('division', 'time',), 
                                          fill_value=np.NaN)
        variable.setncatts(variable_attributes)
    
        # get the total number of time steps
        times_size = dataset.variables['time'][:].size
        
        # loop over each existing division and add the corresponding data array, if one was provided
        for division_index, division_id in enumerate(list(dataset.variables['division'][:])):
            
            # make sure we have a data array of monthly values for this division
            if division_id in divisions_to_arrays.keys():

                # make sure the array has the expected number of time steps 
                data_array = divisions_to_arrays[division_id]
                if data_array.size == times_size:
                
                    # assign the array into the current division's slot in the variable
                    variable[division_index, :] = np.reshape(data_array, (1, times_size))

                else:

                    _logger.info('Unexpected size of data array for division ID {0} -- '.format(division_id) + 
                                'expected {0} time steps but the array contains {1}'.format(times_size, data_array.size))
            
#-----------------------------------------------------------------------------------------------------------------------
def add_variable_climdivs_divs(file_path,
                               variable_name,
                               variable_attributes,
                               divisions_to_values):
    
    '''
    Adds a one-dimensional (division) variable to an existing climate divisions NetCDF. The variable is created 
    and populated with the provided data values. 
    
    :param file_path: existing NetCDF to which the variable will be added. This NetCDF is assumed to contain 
                      the dimensions "division" and "time" as well as corresponding coordinate variables.
    :param variable_name: name of the new variable to be added, a variable with this name should not already exist
    :param variable_attributes: the attributes that should be assigned to the new variable
    :param divisions_to_values: a dictionary with division IDs as keys and corresponding scalars as values.
    '''

    # get the NetCDF datatype applicable to the data array we'll store in the variable
    random_value = random.choice(list(divisions_to_values.values()))
    netcdf_data_type = find_netcdf_datatype(random_value)
    
    # open the output file in append mode for writing, set its dimensions and coordinate variables
    with netCDF4.Dataset(file_path, 'a') as dataset:

        # make sure that the variable name isn't already in use
        if variable_name in dataset.variables.keys():
            
            message = 'Variable name \'{0}\' is already being used within the NetCDF file \'{1}\''.format(variable_name, file_path)
            _logger.error(message)
            raise ValueError(message)
            
        # create the variable, set the attributes
        variable = dataset.createVariable(variable_name, 
                                          netcdf_data_type, 
                                          ('division',), 
                                          fill_value=np.NaN)
        variable.setncatts(variable_attributes)
    
        # loop over each existing division, add the corresponding data array, if one was provided
#         input_divisions = list(divisions_to_values.keys())
        for division_index, division_id in enumerate(list(dataset.variables['division'][:])):
            
            # make sure we have a value for this division
            if division_id in divisions_to_values.keys():
                
                # assign the value into the current division's slot in the variable
                variable[division_index] = divisions_to_values[division_id]
            
#-----------------------------------------------------------------------------------------------------------------------
def initialize_variable_climdivs(netcdf,                   # pragma: no cover
                                 data_variable_name,
                                 data_variable_attributes,
                                 data_fill_value=np.NaN):
    
    #TODO fix these to come from the dataset's dimension attributes?
    divisions_dim_name = 'division'
    
    # get the appropriate data types to use for the variables based on the values arrays
    data_dtype = find_netcdf_datatype(data_fill_value)
    
    # create the variable
    data_variable = netcdf.createVariable(data_variable_name, 
                                          data_dtype, 
                                          (divisions_dim_name, 'time',), 
                                          fill_value=data_fill_value)
    
    # set the variable's attributes
    data_variable.setncatts(data_variable_attributes)

    return netcdf
