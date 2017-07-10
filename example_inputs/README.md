**Master**
[![Build Status](https://img.shields.io/travis/github.com/monocongo/indices/master.svg)](https://travis-ci.org/github.com/monocongo/indices)
[![Test Coverage](https://img.shields.io/codecov/c/gitlab/github.com/monocongo/indices/master.svg)](https://codecov.io/github.com/monocongo/indices)
[![Code Climate](https://img.shields.io/codeclimate/github.com/monocongo/indices.svg)](https://github.com/monocongo/indices)
[![Dependencies](https://img.shields.io/gemnasium/github.com/monocongo/indices.svg)](https://gemnasium.com/github.com/monocongo/indices)

# Ingest and Palmer comparisons

This project contains Python ingest code for ASCII datasets for climate divisions and converts these into a NetCDF that can then be used for indices processing by Python implementations of various climate indices, including comparison against previous Fortran implementations translated to Python.


#### nClimDivs ASCII to NetCDF
The script `ingest_climdiv.py` is used to ingest climatology, intermediate water balance values, and Palmer indices from files produced operationally on a monthly basis from NCEI, examples of which are located in the example_input directory. 

This script has the following required command line arguments:

| Option | Description |
| ------ | ----------- | 
| data_dir <dir> | directory containing climate division ASCII files such as pldat.div, etdat.div, etc. |
| soil_file <file> | input ASCII file containing climate division soil constants dataset |
| out_file <file> | output file |

**Example command line invocation**:

`$ nohup python -u ingest_climdiv.py 
      --data_dir C:/home/palmer 
      --soil_file C:/home/palmer/soilconstdiv.txt 
      --out_file C:/home/palmer/climdivs_all.nc`

#### nClimDiv 
The script `climdivs_water_balance_comparison.py` is used to compute water balance accouting values from [nClimDiv](https://www.ncdc.noaa.gov/monitoring-references/maps/us-climate-divisions.php) input datasets, in NetCDF form after ingest using the process described above. Usage of this script requires specifying the input file name and corresponding variable names for precipitation, temperature, and soil constant datasets. 

This script has the following required command line arguments:

| Option | Description |
| ------ | ----------- | 
| input_file <file> | input NetCDF file containing nClimDiv temperature, precipitation, and soil constant datasets |
| precip_var_name <name> | name of the precipitation variable within the input nClimGrid dataset NetCDF |
| temp_var_name <name> | name of the temperature variable within the input dataset NetCDF |
| awc_var_name <name> | name of the soil constant (available water capacity of the soil) variable within the input dataset NetCDF |

**Example command line invocation**:

`$ nohup python -u climdivs_water_balance_comparison.py 
      --input_file example_inputs/nclimdiv_20170404.nc 
      --precip_var_name prcp 
      --temp_var_name tavg 
      --awc_var_name awc`

## Copyright and licensing
This project is in the public domain within the United States, and we waive worldwide copyright and related rights 
through [CCO universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/). Read more on our license page.
