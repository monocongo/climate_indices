from datetime import date, datetime
from dateutil import relativedelta
from subprocess import CalledProcessError, check_output
from zipfile import ZipFile
import os
import logging
import wget
import re
import sys

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
def _add_time_dimension(netcdf_files):         # pragma: no cover

    updated_files = []

    for netcdf_file in netcdf_files:

        logger.debug('Adding time record dimension info to %s', netcdf_file)

        # get the year and month from the file name
        year_index = -9   # TODO customize this to the expected index location, file name dependent
        month_index = -5  # TODO customize this to the expected index location, file name dependent
        year = int(netcdf_file[year_index:year_index+4])
        month = int(netcdf_file[month_index:month_index+2])

        logger.debug('Year: %s  Month: %s', year, month)

        # get the time value, in our case days since 1800-01-01
        delta = date(year, month, 1) - date(1800, 1, 1)
        days_since_1800 = delta.days

        logger.debug('Days since 1800: %s', days_since_1800)

        # execute a NCO script process on the NetCDF to add a time dimension
        time_dim_processor = 'C:\\nco\\ncap2.exe'  # Windows, laptop PC
        #time_dim_processor = '/usr/bin/ncap2'  # Linux, climgrid-dev
        time_dim_script = 'defdim("time",1);time[time]=' + str(days_since_1800) + ';time@long_name="Time";time@units="days since 1800-01-01 00:00:00"'
        time_dim_options = [ '--no_tmp_fl', '-D 3', '-O', '-s', time_dim_script ]

        # execute a NCO operator to set time as the record dimension
        record_dim_processor = 'C:\\nco\\ncks.exe'   # Windows, laptop PC
        #record_dim_processor = '/usr/bin/ncks'  # Linux, climgrid-dev
        record_dim_options = [ '--no_tmp_fl', '-D', '3', '-O', '--mk_rec_dmn', 'time' ]

        # execute a NCO operator to set time as the record dimension of all data variables
        variable_dim_processor = 'C:\\nco\\ncecat.exe'   # Windows, laptop PC
        #variable_dim_processor = '/usr/bin/ncecat'  # Linux, climgrid-dev
        variable_dim_options = [ '--no_tmp_fl', '-D', '3', '-O', '-u', 'time' ]

        # build the output file from the input file name
        base_name = netcdf_file.partition('.nc')[0]
        tmp1_file = base_name + '_tmp1.nc'
        tmp2_file = base_name + '_tmp2.nc'

        try:
 
            # call the NCO processor to add a time dimension
            result = check_output([time_dim_processor,
                                   time_dim_options[0],
                                   time_dim_options[1],
                                   time_dim_options[2],
                                   time_dim_options[3],
                                   time_dim_options[4],
                                   netcdf_file,
                                   tmp1_file])
            print(result)
            logger.debug('Added the time dimension/variable, new temporary file: %s', tmp1_file)
 
            # call the NCO operator to set time as the record dimension
            result = check_output([record_dim_processor,
                                   record_dim_options[0],
                                   record_dim_options[1],
                                   record_dim_options[2],
                                   record_dim_options[3],
                                   record_dim_options[4],
                                   record_dim_options[5],
                                   tmp1_file,
                                   tmp2_file])
            print(result) 
            logger.debug('Set time as the record dimension, new temporary file: %s', tmp2_file)
 
            # call the NCO operator to set time as the record dimension for all data variables
            result = check_output([variable_dim_processor,
                                   variable_dim_options[0],
                                   variable_dim_options[1],
                                   variable_dim_options[2],
                                   variable_dim_options[3],
                                   variable_dim_options[4],
                                   variable_dim_options[5],
                                   tmp2_file,
                                   netcdf_file])
 
            print(result) 
            logger.debug('Set time as the record dimension for data variables, result file: %s', netcdf_file)
 
            # remove the temporary files
            os.remove(tmp1_file)
            os.remove(tmp2_file)
 
            # add to the list of files we'll return
            updated_files.append(netcdf_file)
 
        except CalledProcessError as e:
 
            # show the error
            logger.error('NCO error messages: %s', e)

    return updated_files

#-----------------------------------------------------------------------------------------------------------------------
def _convert_files(zip_files,       # pragma: no cover
                   variable_name):

    # validate the argument
    if len(zip_files) <= 0:
        
        message = 'No files passed into the _convert_files() method'
        logger.error(message)
        raise ValueError(message)

    # sort the list of files into ascending order, we assume that this will put things into ascending 
    # chronological order since file names should be either ppt_<YYYY>/.zip or ppt_<YYYYMM>.zip 
    zip_files.sort()
    
    # the list of NetCDF files we'll return
    netcdf_files = []

    # the GDAL translator program we'll use to get the first version of NetCDF
    translator = 'C:\\Program Files\\QGIS 2.14\\bin\\gdal_translate.exe'   # Windows
#    translator = '/home/james.adams/software/gdal/bin/gdal_translate'   # Linux
    options = ['-of', 'netcdf']

    # go through each of the ZIP files
    for zip_file in zip_files:

        # list of BIL files we'll convert to NetCDF in a later step
        bil_files = []

        # get the zipped file's contents list, extract all necessary files
        zip_ref = ZipFile(zip_file, 'r')
        zipped_files = zip_ref.namelist()
        for zipped_file in zipped_files:

            # we want to extract the BIL, HDR, and PRJ files
            extension = zipped_file[-4:]
            if (extension == '.bil') or (extension == '.hdr') or (extension == '.prj'):

                logger.debug('\tExtracting ' + zipped_file + ' from compressed file: ' + zip_file)
                
                # extract the file from the zip file
                zip_ref.extract(zipped_file)

                logger.debug('\t\tExtracted %s', zipped_file)

                # add to the list of BIL files
                if extension == '.bil':

                    bil_files.append(zipped_file)

        # close the zip file
        zip_ref.close()

        # loop over each of the BIL files and convert it to NetCDF
        for bil_file in bil_files:

            # use the file name minus the .bil extension as the base file name
            base_name = bil_file.partition('.bil')[0]
            netcdf_file = base_name + '.nc'

            try:

                logger.info('\n\tConverting %s to %s\n\n', bil_file, netcdf_file)
                logger.info('\tGDAL command: \n\n\t%s', ' '.join([translator, options[0], options[1], bil_file, netcdf_file]))

                # call the GDAL translator to convert from BIL to NetCDF (this assumes the corresponding *.hdr and *.prj files are in place)
                result = check_output([translator, options[0], options[1], bil_file, netcdf_file])
                print(result) 

                # console status message
                logger.info('\t\tConverted')

                # add to the list of converted files
                netcdf_files.append(netcdf_file)

            except CalledProcessError as e:

                # show the error
                logger.error('GDAL translator error: %s', e.output)

            # remove the BIL file and corresponding HDR and PRJ files
            os.remove(base_name + '.bil')
            os.remove(base_name + '.hdr')
            os.remove(base_name + '.prj')
    
    # go through all NetCDF files and only keep the monthly files
    monthly_netcdf_files = []
    regex = re.compile(r'PRISM_.*_(\d{6})_bil.nc')   # monthly files will have six digits (YYYYMM), whereas the annual files will only contain four (YYYY)
    for netcdf_file in netcdf_files:
    
        match = regex.match(netcdf_file)
        if match is not None:
            
            # rename to prism_<var>_<YYYYMM>.nc
            new_file = 'prism_' + variable_name + '_' + match.group(1) + '.nc'
            os.rename(netcdf_file, new_file)
            
            # add to the list of NetCDF files we'll return
            monthly_netcdf_files.append(new_file)
        
        else:
            
            # the file must be an annual or non-NetCDF file, not needed so we remove it
            os.remove(netcdf_file)
            
    return monthly_netcdf_files

#-----------------------------------------------------------------------------------------------------------------------
def _download_var(clim_var,    # pragma: no cover
                 start_year):

    # we'll download from a directory containing the gridded data set for the specified climate variable
    base_url = 'http://services.nacse.org/prism/data/public/4km/' + clim_var + '/'

    # we'll go from January of the start year to previous month of the current year
    start = date(start_year, 1, 1)
    now = datetime.now()
    if now.month == 1:
        stop_month = 12
        stop_year = now.year - 1
    else:
        stop_month = now.month - 1
        stop_year = now.year
    stop = date(stop_year, stop_month, now.day)
#     start = date(1979, 1, 1)
#     stop = date(1981, 12, 1)

    # list of downloaded file names
    filenames = []

    # loop over all months from the start to end date
    while start <= stop:

        # based on the year we'll either process whole years together (pre-1980) or separately month by month
        if start.year > 1980:
            
            # use the month as part of the file name and a one-month delta increment since months after 1980 are packaged separately
            year_month = str(start.year) + str(start.month).zfill(2)
            increment = relativedelta.relativedelta(months=+1)
        
        else:

            # use only the year as part of the file name and a one-year time delta increment since all months for a year are packaged together before 1980
            year_month = str(start.year)
            increment = relativedelta.relativedelta(years=+1)
        
        # complete the URL now that we've specified the date portion appropriately
        url = base_url + year_month

        logger.debug('Downloading for year: %s', str(start.year))

        # download from the location, save as a *.zip file with the same name as the variable
        local_filename = clim_var + '_' + year_month + '.zip'
        filename = wget.download(url, out=local_filename)
        filenames.append(filename)

        logger.debug('\t\tDownloaded %s', filename)

        # increment our month counter
        start = start + increment

    return filenames

#-----------------------------------------------------------------------------------------------------------------------
def _concatenate(netcdf_files,   # pragma: no cover
                 output_file):

    # validate the argument
    if len(netcdf_files) <= 0:
        
        message = 'No files passed into the _concatenate() method'
        logger.error(message)
        raise ValueError(message)

    else:      

        logger.debug('\n\nFiles to be concatenated:  %s\n\n', ' '.join([str(netcdf) for netcdf in netcdf_files]))

    # set the path to the NCO executable we'll use to _concatenate the individual files into a single file
    nco_concatenator = 'C:\\nco\\ncrcat.exe'  # Windows
    #nco_concatenator = '/usr/bin/ncrcat'      # Linux

    # create a list of strings specifying the NCO concatenation command that will be invoked
    process = [nco_concatenator, '-O', '-h', '-D 2']
    for netcdf_file in netcdf_files:
        process.append(netcdf_file)
    process.append(output_file)

    try:
        # console status message
        logger.info('Concatenate command:  %s', ' '.join([str(p) for p in process]))
         
        # call the NCO concatenator process
        result = check_output(process)
        print(result)
        
        # console status message
        logger.info('\t\tConcatenated/result file:  %s', output_file)
 
        return output_file
    
    except Exception as e:
 
        # show the error
        logger.error('NCO error: %s', str(e))
        raise e

#-----------------------------------------------------------------------------------------------------------------------
def _build_single_netcdf(netcdf_files,    # pragma: no cover
                         output_file):
    """
    Build a single NetCDF from a list of monthly NetCDF files.
    
    :param netcdf_files: list of NetCDF files to be concatenated into the final NetCDF
    :param output_file: path/name of the output NetCDF file
    :param var_name: variable for which we are creating the NetCDF
    :return: the output file path/name
    """
    logger.debug('Building single NetCDF from files: %s', str(netcdf_files))

    netcdf_files = _add_time_dimension(netcdf_files)    
    
    netcdf_file = _concatenate(netcdf_files, output_file)

    logger.debug('\t\tResult: %s', netcdf_file)
    
    return output_file

#-----------------------------------------------------------------------------------------------------------------------
def _rename_variable(netcdf_file, var_name):   # pragma: no cover
    
    nco_util = 'C:\\nco\\ncrename.exe'  # Windows    
    #nco_util = '/usr/bin/ncrename'      # Linux    

    process = [nco_util, '-v', 'Band1,' + var_name, netcdf_file]
     
    try:
 
        logger.debug('\tVariable rename command:  %s', ' '.join([str(p) for p in process]))
         
        # call the NCO command to rename the variable
        result = check_output(process)
        print(result)
 
        # console status message
        logger.debug('\t\tVariable renamed successfully')
 
    except WindowsError as e:
 
        # show the error
        logger.error('NCO error: %s', str(e))

#-----------------------------------------------------------------------------------------------------------------------
def _fix_variable_attributes(netcdf_file,   # pragma: no cover
                             var_name):
    """
    Set attributes for precipitation and temperature variables.
    
    :param netcdf_file: NetCDF file in which either a precipitation or temperature variable is to have attributes added.
    :param var_name: should be either 'ppt' or 'tmean  
    """

    if (var_name == 'ppt'):
        
        standard_name = 'precipitation'
        long_name = 'Precipitation, total'
        units = 'millimeters'
        valid_max = 2000.0
        valid_min = 0.0
        
    elif (var_name == 'tmean'):
        
        standard_name = 'temperature'
        long_name = 'Temperature, mean'
        units = 'degrees, Celsius'
        valid_max = 100.0
        valid_min = -50.0

    else:
        
        message = 'Invalid var_name argument: ' + var_name
        logger.error(message)
        raise ValueError(message)
        
    # use NaN as the fill value
    fill_value = 'NaN'
    
    nco_util = 'C:\\nco\\ncatted.exe'  # Windows
    #nco_util = '/usr/bin/ncatted'      # Linux
     
    # create a list of processes corresponding to the various attributes we want to update for the variable
    processes = [[nco_util, '-O', '-a', 'standard_name,' + var_name + ',o,c,' + standard_name, netcdf_file],
                 [nco_util, '-O', '-a', 'long_name,' + var_name + ',o,c,' + long_name, netcdf_file],
                 [nco_util, '-O', '-a', 'valid_max,' + var_name + ',o,f,' + str(valid_max), netcdf_file],
                 [nco_util, '-O', '-a', 'valid_min,' + var_name + ',o,f,' + str(valid_min), netcdf_file],
                 [nco_util, '-O', '-a', 'units,' + var_name + ',o,c,' + units, netcdf_file],
                 [nco_util, '-O', '-a', '_FillValue,' + var_name + ',o,f,' + fill_value, netcdf_file]]
     
    # execute each of the attribute update processes
    for process in processes:
        try:
     
            logger.debug('\tAttribute update command:  %s', ' '.join([str(p) for p in process]))
             
            # call the NCO command to change the attribute value
            result = check_output(process)
            print(result)
     
            # console status message
            logger.debug('\t\tAttribute updated')
     
        except WindowsError as e:
     
            # show the error
            logger.error('NCO error: %s', e)

#-----------------------------------------------------------------------------------------------------------------------
def ingest_to_netcdf(output_dir,
                     clean=False):

    # change into the output directory where all downloaded, intermediate, and final files will be located
    os.chdir(output_dir)
    
    for var_name in ['ppt', 'tmean']:
        
        # download the compressed/zip files
        zip_files = _download_var(var_name, 1895)
             
        # DEBUG/DEVELOPMENT ONLY -- REMOVE
#         zip_files = glob.glob(work_directory + '/' + var_name + '_*.zip')
 
        # convert the zipped BIL files to NetCDF
        netcdf_files = _convert_files(zip_files, var_name)        
    
#         # DEBUG ONLY
#         netcdf_files = glob.glob('prism_ppt_*.nc')

        # combine all the individual NetCDF files into a single NetCDF
        netcdf_file = _build_single_netcdf(netcdf_files, 'prism_' + var_name + '.nc')
    
        # rename the variable
        _rename_variable(netcdf_file, var_name)
        
        # update the variable's attributes
        _fix_variable_attributes(netcdf_file, var_name)

        if var_name == 'ppt':
            precip_file = netcdf_file
        elif var_name == 'tmean':
            temp_file = netcdf_file
            
        logger.info('\nResult file: %s\n', netcdf_file)

        # clean up the downloaded and intermediate files
        if clean:
        
            for tmp_file in zip_files:
                os.remove(tmp_file)
            for tmp_file in netcdf_files:
                os.remove(tmp_file)

        return precip_file, temp_file
    
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    output_directory = sys.argv[1]
    ingest_to_netcdf(output_directory)
