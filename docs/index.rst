.. climate_indices documentation master file, created by
   sphinx-quickstart on Tue Feb 20 16:10:27 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. |Build| image:: https://travis-ci.org/monocongo/climate_indices.svg?master
   :target: https://travis-ci.org/monocongo
.. |CodeFactor| image:: https://www.codefactor.io/repository/github/monocongo/climate_indices/badge/master
   :target: https://www.codefactor.io/repository/github/monocongo/climate_indices/overview/master
.. |Coverage| image:: https://coveralls.io/repos/github/monocongo/climate_indices/badge.svg?branch=master
   :target: https://coveralls.io/github/monocongo/climate_indices?branch=master
.. |Dependencies| image:: https://gemnasium.com/badges/github.com/monocongo/climate_indices.svg

   :target: https://gemnasium.com/github.com/monocongo/climate_indices
.. |License| image:: https://img.shields.io/badge/license-Unlicense-green.svg
   :target: https://unlicense.org/

|Build| |Coverage| |CodeFactor| |License|

Climate Indices in Python
=============================================

This project contains Python implementations of various climate index algorithms which provide 
a geographical and temporal picture of the severity of precipitation and temperature anomalies
useful for climate monitoring and research.

The following indices are provided:

-  `SPI <https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-index-spi>`__,
   Standardized Precipitation Index, utilizing both gamma and Pearson Type III distributions
-  `SPEI <https://www.researchgate.net/publication/252361460_The_Standardized_Precipitation-Evapotranspiration_Index_SPEI_a_multiscalar_drought_index>`__,
   Standardized Precipitation Evapotranspiration Index, utilizing both gamma and Pearson Type III distributions
-  `PET <https://www.ncdc.noaa.gov/monitoring-references/dyk/potential-evapotranspiration>`__,
   Potential Evapotranspiration, utilizing either `Thornthwaite <http://dx.doi.org/10.2307/21073>`_ 
   or `Hargreaves <http://dx.doi.org/10.13031/2013.26773>`_ equations 
-  `PDSI <http://www.droughtmanagement.info/palmer-drought-severity-index-pdsi/>`__,
   Palmer Drought Severity Index
-  `scPDSI <http://www.droughtmanagement.info/self-calibrated-palmer-drought-severity-index-sc-pdsi/>`__,
   Self-calibrated Palmer Drought Severity Index
-  `PHDI <http://www.droughtmanagement.info/palmer-hydrological-drought-index-phdi/>`__,
   Palmer Hydrological Drought Index
-  `Z-Index <http://www.droughtmanagement.info/palmer-z-index/>`__,
   Palmer moisture anomaly index (Z-index)
-  `PMDI <https://climate.ncsu.edu/climate/climdiv>`__, Palmer Modified
   Drought Index
-  `PNP <http://www.droughtmanagement.info/percent-of-normal-precipitation/>`__,
   Percentage of Normal Precipitation

This Python implementation of the above climate index algorithms is being developed 
with the following goals in mind:

-  to provide an open source software package to compute a suite of
   climate indices commonly used for climate monitoring, with well
   documented code that is faithful to the relevant literature and
   which produces scientifically verifiable results
-  to provide a central, open location for participation and collaboration 
   for researchers, developers, and users of climate indices
-  to facilitate standardization and consensus on best-of-breed
   climate index algorithms and corresponding compliant implementations in Python
-  to provide transparency into the operational code used for climate
   monitoring activities at NCEI/NOAA, and consequent reproducibility 
   of published datasets computed from this package
-  to incorporate modern software engineering principles and programming 
   best practices


Getting started
---------------

The installation and configuration described below is 
performed using a bash shell, either on Linux, Windows, or MacOS.

Windows users will need to install and configure a bash shell in order
to follow the usage shown below. We recommended either 
`babun <https://babun.github.io/>`__ or `Cygwin <https://www.cygwin.com/>`__
for this purpose.

Download the code
^^^^^^^^^^^^^^^^^

Clone this repository:

``$ git clone https://github.com/monocongo/climate_indices.git``

Move into the source directory:

``$ cd climate_indices``

Within this directory, there are six subdirectories:

-  ``climate_indices``: main computational package
-  ``tests``: unit tests for the main package
-  ``scripts``: scripts and supporting utility modules used to perform processing of indices computations on climatological datasets (typically grids or US climate division datasets in NetCDF)
-  ``example_inputs``: example/reference datasets that can be used as inputs to the processing scripts
-  ``notebooks``: Jupyter Notebooks describing the internals of the computational modules
-  ``docs``: documentation files

Configure the Python environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This project's code is written in Python 3. It's recommended to use 
either the `Miniconda3 <https://conda.io/miniconda.html>`__ (minimal Anaconda) or 
`Anaconda3 <https://www.continuum.io/downloads>`__ distribution. The below instructions
will be Anaconda specific (although relevant to any Python `virtualenv <https://virtualenv.pypa.io/en/stable/>`_
), and assume the use of a bash shell.

A new Anaconda `environment <https://conda.io/docs/using/envs.html>`__ can be created
using the `conda <https://conda.io/docs/>`_ environment management system that comes
packaged with Anaconda. In the following examples, we'll use an environment named *indices_env* 
(any environment name can be used instead of *indices_env*) which will be created and
populated with all required dependencies through the use of the provided ``setup.py`` file.

First create the Python environment:

``$ conda create -n indices_env``

The environment created can now be 'activated':

``$ source activate indices_env``

Once the environment has been activated then subsequent Python commands will run 
in this environment where the package dependencies for this project are present.

Now the package can be added to the environment along with all required modules 
(dependencies) via `pip <https://pip.pypa.io/en/stable/>`_:

``$ pip install .``

Testing
-------

Initially, all tests should be run for validation:

``$ export NUMBA_DISABLE_JIT=1``

``$ python setup.py test``

``$ unset NUMBA_DISABLE_JIT``

If you run the above from the main branch and get an error then please
send a report and/or add an issue, as all tests should pass.

The Numba environment variable is set/unset in order to bypass Numba's
just-in-time compilation process, which significantly reduces testing times.

Indices Processing
----------------------------------

Included are scripts which interact with the core computational package to compute
one or more climate indices. These are ``process_grid.py`` which is used
to compute indices corresponding to gridded NetCDF datasets, and ``process_divisions.py``
which is used to compute indices corresponding to US climate division NetCDF datasets.

These Python scripts are written to be run via bash shell commands, i.e.

``$ python process_grid.py <options>``

The options for these scripts are described below:


+------------------------+-------------------------------------------------+
| Option                 | Description                                     |
+========================+=================================================+
| index                  | Which of the climate indices to compute.        |
|                        | Valid values are 'spi', 'spei', 'pnp', 'scaled',|
|                        | 'pet', and 'palmers'. 'scaled' indicates all    |
|                        | three scaled indices (SPI, SPEI, and PNP) and   |
|                        | 'palmers' indicates all Palmer indices (PDSI,   |
|                        | PHDI, PMDI, SCPDSI, and Z-Index).               |
|                        |                                                 |
|                        | **NOTE**: Only used for grid processing, as     |
|                        | the divisions processing will compute all       |
|                        | indices.                                        |
+------------------------+-------------------------------------------------+
| periodicity            | The periodicity of the input dataset files.     |
|                        | Valid values are 'monthly' and 'daily'.         |
|                        |                                                 |
|                        | **NOTE**: Only used for grid processing, and    |
|                        | only SPI and PNP support daily inputs.          |
+------------------------+-------------------------------------------------+
| netcdf_divs            | Input NetCDF file containing a US climate       |
|                        | divisions dataset with precipitation,           |
|                        | temperature, and AWC variables. Computed indices|
|                        | variables will be written into this file, with  |
|                        | existing indices variables (if any) overwritten |
|                        | with the computed values. This file serves as   |
|                        | both input and output for the divisions         |
|                        | processing.                                     |
|                        |                                                 |
|                        | **NOTE**: Only used for US climate divisions    |
|                        | processing.                                     |
+------------------------+-------------------------------------------------+
| netcdf_precip          | Input NetCDF file containing a                  |
|                        | precipitation dataset, required for all         |
|                        | indices except for PET. Requires the use of     |
|                        | **var_name_temp** in conjunction so as to       |
|                        | identify the NetCDF's precipitation variable.   |
|                        |                                                 |
|                        | **NOTE**: Only used for grid processing.        |
+------------------------+-------------------------------------------------+
| var_name_precip        | Name of the precipitation variable within       |
|                        | the input precipitation NetCDF.                 |
+------------------------+-------------------------------------------------+
| netcdf_temp            | Input NetCDF file containing a                  |
|                        | temperature dataset, required for PET.          |
|                        | If specified in conjunction with an index       |
|                        | specification of SPEI or Palmers then PET       |
|                        | will be computed and written as a side          |
|                        | effect, since these indices require PET.        |
|                        | This option is mutually exclusive with          |
|                        | **netcdf_pet/var_name_pet**, as either          |
|                        | temperature or PET is required as an input      |
|                        | (but not both) when computing SPEI and/or       |
|                        | Palmers. Requires the use of                    |
|                        | **var_name_temp** in conjunction so as to       |
|                        | identify the NetCDF's temperature variable.     |
|                        |                                                 |
|                        | **NOTE**: Only used for grid processing.        |
+------------------------+-------------------------------------------------+
| var_name_temp          | Name of the temperature variable within the     |
|                        | input temperature NetCDF.                       |
+------------------------+-------------------------------------------------+
| netcdf_pet             | Input NetCDF file containing a PET dataset,     |
|                        | required for SPEI and Palmers.                  |
|                        | This option is mutually exclusive with          |
|                        | **netcdf_temp/var_name_temp**, as either        |
|                        | temperature or PET is required as an input      |
|                        | (but not both) when computing SPEI and/or       |
|                        | Palmers. Requires the use of                    |
|                        | **var_name_pet** in conjunction so as to        |
|                        | identify the NetCDF's PET variable.             |
|                        |                                                 |
|                        | **NOTE**: Only used for grid processing.        |
+------------------------+-------------------------------------------------+
| var_name_pet           | Name of the PET variable within the input PET   |
|                        | NetCDF.                                         |
|                        |                                                 |
|                        | **NOTE**: Only used for grid processing.        |
+------------------------+-------------------------------------------------+
| netcdf_awc             | Input NetCDF file containing an available water |
|                        | capacity, required for Palmers. Requires the    |                    
|                        | use of **var_name_awc** in conjunction so as to |
|                        | identify the NetCDF's AWC variable.             |
|                        |                                                 |
|                        | **NOTE**: Only used for grid processing.        |
+------------------------+-------------------------------------------------+
| awc_var_name           | Name of the available water capacity variable   |
|                        | within the input AWC NetCDF.                    |
+------------------------+-------------------------------------------------+
| output_file_base       | Base file name for all grid processing          |
|                        | output files.                                   |
|                        |                                                 |
|                        | Each computed index will have a corresponding   |
|                        | output file whose name will begin with          |
|                        | this base name plus the index's                 |
|                        | abbreviation plus a month scale                 |
|                        | (if applicable), connected with underscores,    |
|                        | plus the '.nc' extension. For example           |
|                        | for SPI at 3-month scale                        |
|                        | the resulting output files will be              |
|                        | named **<output_file_base>_spi_gamma_03.nc**    |
|                        | and **<output_file_base>_spi_pearson_03.nc**.   |
|                        |                                                 |
|                        | **NOTE**: Only used for grid processing.        |
+------------------------+-------------------------------------------------+
| scales                 | Time step scales over which the PNP, SPI, and   |
|                        | SPEI values are to be computed. Required when   |
|                        | the **index** argument is 'spi', 'spei',        |
|                        | 'pnp', or 'scaled'. If the **periodicity**      |
|                        | option is specified then that option will infer |
|                        | that the scales used here either month or day   |
|                        | scales.                                         |
|                        |                                                 |
|                        | **NOTE**: When used for US climate divisions    |
|                        | processing this option specifies to month       |
|                        | scales.                                         |
+------------------------+-------------------------------------------------+
| calibration_start_year | Initial year of the calibration period.         |
+------------------------+-------------------------------------------------+
| calibration_end_year   | Final year of the calibration period            |
|                        | (inclusive).                                    |
+------------------------+-------------------------------------------------+

Example Command Line Invocations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

US Climate Divisions (all indices)
""""""""""""""""""""""""""""""""""

``$ python process_divisions.py --scales 3 6
--input_file ../example_inputs/nclimdiv.nc
--output_file /data/nclimdiv_indices.nc
--var_name_precip prcp
--var_name_temp tavg --var_name_awc awc
--calibration_start_year 1951 --calibration_end_year 2010``

The above command will compute all indices from an input NetCDF dataset containing
precipitation, temperature, and available water capacity variables (in this case,
the US Climate Divisions NetCDF dataset provided in the example inputs directory).
The input dataset is monthly data and the calibration period used will be
Jan. 1951 through Dec. 2010. The indices will be computed at 3-month and 6-month scales.
Upon completion the output NetCDF file (/data/nclimdiv_indices.nc
in this example) will contain variables for all computed indices:
`pet`, `pnp_03`, `pnp_06`, `spi_gamma_03`, `spi_gamma_06`, `spi_pearson_03`, `spi_pearson_06`,
`spei_gamma_03`, `spei_gamma_06`, `spei_pearson_03`, `spei_pearson_06`, `pdsi`, `phdi`, `pmdi`,
`scpdsi`, and `zindex`.

PET monthly
""""""""""""

``$ python process_grid.py --index pet --periodicity monthly --netcdf_temp  
../example_inputs/nclimgrid_lowres_tavg.nc --var_name_temp tavg --output_file_base 
<out_dir>/nclimgrid_lowres``

The above command will compute PET (potential evapotranspiration) using the 
Thornthwaite method from an input temperature dataset (in this case, the reduced 
resolution nClimGrid temperature dataset provided in the example inputs directory).
The input dataset is monthly data and the calibration period used will be Jan. 1951
through Dec. 2010. The output file will be `<out_dir>/nclimgrid_lowres_pet.nc`.

SPI daily
""""""""""

``$ python process_grid.py --index spi  --periodicity daily --netcdf_precip 
../example_inputs/cmorph_lowres_daily_conus_prcp.nc --var_name_precip 
prcp --output_file_base <out_dir>/cmorph_lowres_daily_conus --scales 30 90 
--calibration_start_year 1998 --calibration_end_year 2016``

The above command will compute SPI (standardized precipitation index, both gamma 
and Pearson Type III distributions) from an input precipitation dataset (in this case, 
the reduced resolution CMORPH precipitation dataset provided in the example inputs
directory). The input dataset is daily data and the calibration period used will be 
Jan. 1st, 1998 through Dec. 31st, 2016. The index will be computed at 30-day and 
90-day timescales. The output files will be `<out_dir>/cmorph_lowres_daily_conus_spi_gamma_30.nc`, 
`<out_dir>/cmorph_lowres_daily_conus_spi_gamma_90.nc`, 
`<out_dir>/cmorph_lowres_daily_conus_spi_pearson_30.nc`, and 
`<out_dir>/cmorph_lowres_daily_conus_spi_pearson_90.nc`.

SPI monthly
""""""""""""

``$ python process_grid.py --index spi --periodicity monthly --netcdf_precip 
../example_inputs/nclimgrid_lowres_prcp.nc --var_name_precip  prcp 
--output_file_base <out_dir>/nclimgrid_lowres --scales 6 12 
--calibration_start_year 1951 --calibration_end_year 2010``  

The above command will compute SPI (standardized precipitation index, both gamma and 
Pearson Type III distributions) from an input precipitation dataset (in this case, 
the reduced resolution nClimGrid precipitation dataset provided in the example inputs directory).
The input dataset is monthly data and the calibration period used will be 
Jan. 1951 through Dec. 2010. The index will be computed at 6-month and 12-month timescales. 
The output files will be `<out_dir>/nclimgrid_lowres_spi_gamma_06.nc`, 
`<out_dir>/nclimgrid_lowres_spi_gamma_12.nc`, `<out_dir>/nclimgrid_lowres_spi_pearson_06.nc`, 
and `<out_dir>/nclimgrid_lowres_spi_pearson_12.nc`.

SPEI monthly
"""""""""""""

``$ python process_grid.py --index spei --periodicity monthly --netcdf_precip 
../example_inputs/nclimgrid_lowres_prcp.nc --var_name_precip  prcp --netcdf_pet 
../example_inputs/nclimgrid_lowres_pet.nc --var_name_pet pet --output_file_base 
<out_dir>/nclimgrid_lowres --scales 9 18 --calibration_start_year 1951 --calibration_end_year 2010``  

The above command will compute SPEI (standardized precipitation evapotranspiration index, 
both gamma and Pearson Type III distributions) from input precipitation and potential evapotranspiration datasets 
(in this case, the reduced resolution nClimGrid precipitation and PET datasets provided in the example inputs directory). 
The input datasets are monthly data and the calibration period used will be Jan. 1951 through Dec. 2010. The index 
datasets will be computed at 9-month and 18-month timescales. The output files will be 
`<out_dir>/nclimgrid_lowres_spei_gamma_09.nc`, `<out_dir>/nclimgrid_lowres_spei_gamma_18.nc`,
`<out_dir>/nclimgrid_lowres_spei_pearson_09.nc`, and `<out_dir>/nclimgrid_lowres_spei_pearson_18.nc`.

Palmers monthly
""""""""""""""""
``$ python process_grid.py --index palmers --periodicity monthly --netcdf_precip 
../example_inputs/nclimgrid_lowres_prcp.nc --var_name_precip prcp --netcdf_pet 
../example_inputs/nclimgrid_lowres_pet.nc --var_name_pet pet --netcdf_awc 
../example_inputs/nclimgrid_lowres_soil.nc  --var_name_awc awc --output_file_base 
<out_dir>/nclimgrid_lowres --calibration_start_year 1951 --calibration_end_year 2010``

The above command will compute the Palmer drought indices: PDSI (original Palmer Drought Severity Index), 
PHDI (Palmer Hydrological Drought Index), PMDI (Palmer Modified Drought Index), Z-Index (Palmer Z-Index), 
and SCPDSI (Self-calibrated Palmer Drought Severity Index) from input precipitation, potential 
evapotranspiration, and available water capacity datasets (in this case, the reduced resolution nClimGrid 
precipitation, PET, and AWC datasets provided in the example inputs directory). The input datasets are monthly 
data and the calibration period used will be Jan. 1951 through Dec. 2010. The output files will be 
`<out_dir>/nclimgrid_lowres_pdsi.nc`, `<out_dir>/nclimgrid_lowres_phdi.nc`, 
`<out_dir>/nclimgrid_lowres_pmdi.nc`, `<out_dir>/nclimgrid_lowres_scpdsi.nc`, and `<out_dir>/nclimgrid_lowres_zindex.nc`.

Get involved
-------------

Please use, make suggestions, and contribute to this code. Without
diverse participation and community adoption this project will not reach
its potential.

Are you aware of other indices that would be a good addition here? Can
you identify bottlenecks and help optimize performance? Can you suggest new
ways of comparing these implementations against others (or other
criteria) in order to determine best-of-breed? Please fork the code and
have at it, and/or contact us to see if we can help.

-  Read our `contributing
   guidelines <https://github.com/monocongo/climate_indices/blob/master/CONTRIBUTING.md>`__
-  File an
   `issue <https://github.com/monocongo/climate_indices/issues>`__, or
   submit a `pull request <https://github.com/monocongo/climate_indices/pulls>`_

-  Send us an `email <mailto:monocongo@gmail.com>`__

Copyright and licensing
-----------------------

This is a developmental version of code that is originally developed at
NCEI/NOAA, official release version available on
`drought.gov <https://www.drought.gov/drought/python-climate-indices>`__.
This software is in the public domain, with no copyright. Please read more on our `license <UNLICENSE>`__ page.


