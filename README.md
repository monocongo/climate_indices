

[![Build Status](https://travis-ci.org/monocongo/indices_python.svg?master)](https://travis-ci.org/monocongo)
[![CodeFactor](https://www.codefactor.io/repository/github/monocongo/indices_python/badge/master)](https://www.codefactor.io/repository/github/monocongo/indices_python/overview/master)
[![Coverage Status](https://coveralls.io/repos/github/monocongo/indices_python/badge.svg?branch=master)](https://coveralls.io/github/monocongo/indices_python?branch=master)
[![Dependency Status](https://gemnasium.com/badges/github.com/monocongo/indices_python.svg)](https://gemnasium.com/github.com/monocongo/indices_python)
<!--
[![Codeship Status for monocongo/indices_python](https://app.codeship.com/projects/0d711e30-ca42-0135-871a-72c36ec6d502/status?branch=master)](https://app.codeship.com/projects/261762)
-->

# Community Reference Climate Indices

This project contains Python implementations of various climate indices which provide a geographical and temporal picture of the severity of precipitation and temperature anomalies. We attempt to provide best-of-breed implementations of various climate indices commonly used for climate and drought monitoring, to provide a codebase that is available for development by the climate science community, and to facilitate the use of climate indices datasets computed in a standardized, reproducible, and transparent fashion.


Currently provided climate indices:

* [SPI](https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-index-spi), Standard Precipitation Index
* [SPEI](https://www.researchgate.net/publication/252361460_The_Standardized_Precipitation-Evapotranspiration_Index_SPEI_a_multiscalar_drought_index), Standard Precipitation Evapotranspiration Index
* [PET](https://www.ncdc.noaa.gov/monitoring-references/dyk/potential-evapotranspiration), Potential Evapotranspiration: computed using [Thornthwaite's equation](https://en.wikipedia.org/wiki/Potential_evaporation) 
* [PNP](http://www.droughtmanagement.info/percent-of-normal-precipitation/), Percentage of Normal Precipitation
* [PDSI](http://www.droughtmanagement.info/palmer-drought-severity-index-pdsi/), Palmer Drought Severity Index
* [scPDSI](http://www.droughtmanagement.info/self-calibrated-palmer-drought-severity-index-sc-pdsi/), Self-calibrated Palmer Drought Severity Index
* [PHDI](http://www.droughtmanagement.info/palmer-hydrological-drought-index-phdi/), Palmer Hydrological Drought Index
* [Z-Index](http://www.droughtmanagement.info/palmer-z-index/), Palmer moisture anomaly index (Z-index)
* [PMDI](https://climate.ncsu.edu/climate/climdiv), Palmer Modified Drought Index 

These initial Python implementations of these climate indices algorithms is being developed with the following goals in mind:

 - to provide an open source software package to compute a suite of climate indices commonly used for drought monitoring, with well documented code that is faithful to the literature, and which provides scientifically accurate results
  - to provide transparency into the operational code used for climate monitoring activities at NCEI, and reproducibility for any users of datasets computed from this package
 - to facilitate standardization and consensus on best-of-breed algorithms and accompanying implementations
 - to serve as an example of open source scientific development process, incorporating software engineering principles and programming best practices

## Environment set up
#### Python
This project's code is written for Python 3. It's recommended that you use an installation of the [Anaconda](https://www.continuum.io/why-anaconda) Python 3 distribution. The below instructions will be Anaconda specific, and initially aimed at Linux users.

For users without an existing Python/Anaconda installation we recommend either 
* installing the [Miniconda](https://conda.io/miniconda.html) (minimal Anaconda) distribution
or 
* installing the full [Anaconda](https://www.continuum.io/downloads) distribution

#### Dependencies
This library and the example processing scripts use the [netCDF4](https://unidata.github.io/netcdf4-python/), [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), and [numba](http://numba.pydata.org/) Python modules. The NetCDF Operators ([NCO](http://nco.sourceforge.net/)) software package is also useful for the processing scripts, and can optionally be installed as a Python module via conda.

A new Anaconda [environment](https://conda.io/docs/using/envs.html) containing all required modules can be created through the use of the provided `environment.yml` file, which specifies an environment named **_indices_reference_** containing all required modules:

`$ conda env create -f environment.yml`

The environment created by the above command can be activated using the following command:

`$ source activate indices_reference`

Once the *conda Python environment has been activated then subsequent Python commands will run in this environment where the package dependencies for this project are present.
 
For users who'd prefer to not utilize the above approach using the provided `environment.yml` file, the required module dependencies can be installed instead into an Anaconda environment piecemeal via multiple `conda install` commands:

`$ conda create --name <env_name> python=3` 

`$ source activate <env_name>` 

`$ conda install numba` 

`$ conda install scipy` 

`$ conda install netCDF4` 

`$ conda install pycurl`

## Getting started

1. clone this repository from the 'develop' branch
    
    `$ git clone https://github.com/monocongo/indices_python.git`

2. move into the source directory
    
    `$ cd indices_python`

4. (optionally) install the package into the local sitepkgs

    `$ python setup.py install`

## Testing

Initially all tests should be run for validation:

`$ python -m tests/unittest test_*.py`

If you run the above from the main branch and get an error then please send a report and/or add an issue, as all test should pass on the main branch.

## Example indices processing scripts

There are example climate indices processing scripts provided which compute the full suite of indices for various input dataset types. These process input files in the NetCDF format, and produce output NetCDF files in a corresponding format.

### nClimGrid 
The script `process_nclimgrid.py` is used to compute climate indices from [nClimGrid](https://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NCDC/Geoportal/iso/xml/C00332.xml&view=getDataView&header=none) input datasets. Usage of this script requires specifying the input file names and corresponding variable names for prcipitation, temperature, and soil constant datasets, as well as the month scales over which the scaled indices (SPI, SPEI, and PAP) are to be computed, plus the base output file name and the initial and final years of the calibration period. 

This script has the following required command line arguments:

| Option | Description |
| ------ | ----------- | 
| precip_file <file> | input NetCDF file containing nClimGrid precipitation dataset |
| precip_var_name <name> | name of the precipitation variable within the input nClimGrid precipitation dataset NetCDF |
| temp_file <file> | input NetCDF file containing nClimGrid temperature dataset |
| temp_var_name <name> | name of the temperature variable within the input nClimGrid temperature dataset NetCDF |
| awc_file <file> | input NetCDF file containing a soil constant (available water capacity of the soil) dataset NetCDF, should correspond dimensionally with the input nClimGrid temperature and precipitation datasets |
| awc_var_name <name> | name of the soil constant (available water capacity of the soil) variable within the input soil constant dataset NetCDF |
| output_file_base <path> | base file name for all output files, each computed index will have an output file whose name will begin with this base plus the index's abbreviation plus a month scale (if applicable), plus ".nc" as the extension (i.e. for SPI/Gamma at 3-month scale the resulting output file will be named \<output_file_base\>_spi_gamma_03.nc) |
| month_scales <space separated list of ints> | month scales over which the PAP, SPI, and SPEI values are to be computed, valid range is 1-72 months|
| calibration_start_year <year> | initial year of calibration period |
| calibration_end_year <year> | final year of calibration period |
| destination_dir <dir> | directory where the final output files will be located after all processing has completed |

**Example command line invocation**:

`$ nohup python -u process_nclimgrid.py 
      --precip_file example_inputs/nclimgrid_lowres_prcp.nc 
      --temp_file example_inputs/nclimgrid_lowres_tavg.nc 
      --awc_file example_inputs/nclimgrid_lowres_soil.nc 
      --precip_var_name prcp 
      --temp_var_name tavg 
      --awc_var_name awc 
      --month_scales 1 2 3 6 12 24
      --calibration_start_year 1931 
      --calibration_end_year 1990 
      --output_file_base nclimgrid_lowres
      --destination_dir /indices`

### nClimDiv 
The script `process_nclimdiv.py` is used to compute climate indices from [nClimDiv](https://www.ncdc.noaa.gov/monitoring-references/maps/us-climate-divisions.php) input datasets. Usage of this script requires specifying the input file name and corresponding variable names for precipitation, temperature, and soil constant datasets, as well as the month scales over which the scaled indices (SPI, SPEI, and PAP) are to be computed, plus the base output file name and the initial and final years of the calibration period. 

This script has the following required command line arguments:

| Option | Description |
| ------ | ----------- | 
| input_file <file> | input NetCDF file containing nClimDiv temperature, precipitation, and soil constant datasets |
| precip_var_name <name> | name of the precipitation variable within the input nClimGrid dataset NetCDF |
| temp_var_name <name> | name of the temperature variable within the input dataset NetCDF |
| awc_var_name <name> | name of the soil constant (available water capacity of the soil) variable within the input dataset NetCDF |
| output_file <path> | output file, each computed index will be included as a variable within this output file |
| month_scales <space separated list of ints> | month scales over which the PAP, SPI, and SPEI values are to be computed, valid range is 1-72 months|
| calibration_start_year <year> | initial year of calibration period |
| calibration_end_year <year> | final year of calibration period |

**Example command line invocation**:

`$ nohup python -u process_nclimdiv.py 
      --input_file example_inputs/nclimdiv_20170404.nc 
      --precip_var_name prcp 
      --temp_var_name tavg 
      --awc_var_name awc 
      --month_scales 1 2 3 6 12 24
      --calibration_start_year 1931 
      --calibration_end_year 1990 
      --output_file nclimdiv_20170404_indices.nc`

## Get involved
Please use, make suggestions, and contribute to this code. Without diverse participation and community adoption this project will not reach its potential. 

Are you aware of other indices that would be a good addition here? Can you find bottlenecks and help improve performance? Want to take the Pepsi Challenge comparing these implementations against others? Please fork the code and have at it, and/or contact us to see if we can help.

* Read our [contributing guidelines](https://github.com/monocongo/indices_python/blob/master/CONTRIBUTING.md)
* File an [issue](https://github.com/monocongo/indices_python/issues), or submit a pull request
* Send us an [email](mailto:james.adams@noaa.gov)

## Copyright and licensing
Please read more on our [license](LICENSE) page.




