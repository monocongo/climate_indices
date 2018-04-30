.. climate_indices documentation master file, created by
   sphinx-quickstart on Tue Feb 20 16:10:27 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


|Build Status| |Coverage Status| |Dependency Status| |CodeFactor| |License| 

Community Reference Climate Indices in Python
=============================================

This project contains Python implementations of various climate indices
which provide a geographical and temporal picture of the severity of
precipitation and temperature anomalies. We attempt to provide
best-of-breed implementations of various climate indices commonly used
for climate and drought monitoring, to provide a codebase that is
available for development by the climate science community, and to
facilitate the use of climate indices datasets computed in a
standardized, reproducible, and transparent fashion.

Python implementations of the following climate index algorithms are provided:

-  `SPI <https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-index-spi>`__,
   Standardized Precipitation Index, utilizing either Gamma or Pearson Type III distributions
-  `SPEI <https://www.researchgate.net/publication/252361460_The_Standardized_Precipitation-Evapotranspiration_Index_SPEI_a_multiscalar_drought_index>`__,
   Standardized Precipitation Evapotranspiration Index, utilizing either Gamma or Pearson Type III distributions
-  `PET <https://www.ncdc.noaa.gov/monitoring-references/dyk/potential-evapotranspiration>`__,
   Potential Evapotranspiration, utilizing either `Thornthwaite <http://dx.doi.org/10.2307/21073>`_ or `Hargreaves <http://dx.doi.org/10.13031/2013.26773>`_ equations 
-  `PNP <http://www.droughtmanagement.info/percent-of-normal-precipitation/>`__,
   Percentage of Normal Precipitation
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

This Python implementation of the above climate index algorithms is
being developed with the following goals in mind:

-  to provide an open source software package to compute a suite of
   climate indices commonly used for drought monitoring, with well
   documented code that is faithful to the relevant literature and
   which produces scientifically valid results
-  to facilitate standardization and consensus on best-of-breed
   climate index algorithms including compliant implementations
-  to provide transparency into the operational code used for climate
   monitoring activities at NCEI, and reproducibility for users of
   datasets computed from this package
-  to serve as an example of open source scientific development,
   incorporating software engineering principles and programming best
   practices

Getting started
---------------

The configuration and usage described below shows the indices
computation module being installed and used via shell commands calling
scripts that perform data management and computation of climate indices
from provided inputs. Interaction with the module is assumed to be
performed using a bash shell, either on Linux, Windows, or MacOS.

Windows users will need to install and configure a bash shell in order
to follow the usage shown below. We recommended either 
`babun <https://babun.github.io/>`__ or `Cygwin <https://www.cygwin.com/>`__
for this purpose.

Download the code
^^^^^^^^^^^^^^^

Clone this repository:

``$ git clone https://github.com/monocongo/climate_indices.git``

Move into the source directory:

``$ cd climate_indices``

Within this directory, there are three primary subdirectories:

-  ``climate_indices``: main package
-  ``tests``: unit tests for the main package
-  ``scripts``: scripts and supporting utility modules used to perform processing of indices
computations on climatological datasets (typically grids or climate divisions datasets in NetCDF)

Configure the Python environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This project's code is written in Python 3. It's recommended to use 
either the `Miniconda3 <https://conda.io/miniconda.html>`__ (minimal Anaconda) or 
`Anaconda3 <https://www.continuum.io/downloads>`__ distribution. The below instructions
will be Anaconda specific (although relevant to any Python `virtualenv <https://virtualenv.pypa.io/en/stable/>`_
), and assume the use of a bash shell.

A new Anaconda `environment <https://conda.io/docs/using/envs.html>`__ can be created using the `conda <https://conda.io/docs/>`_ environment management system that comes packaged with Anaconda. In the following examples, we'll use an environment named *indices_env*
containing all required modules can be created and populated with all required dependencies through the use of the provided ``setup.py`` file:

``$ conda create -n indices_env``

The environment created can be 'activated' using the following command:

``$ source activate indices_env``

Once the environment has been activated then subsequent Python commands will run in this environment where the package
dependencies for this project are present.

Now the package can be added to the environment along with all required modules (dependencies) via `pip <https://pip.pypa.io/en/stable/>`_
:

``$ pip install .``

Testing
-------

Initially, all tests should be run for validation:

``$ export NUMBA_DISABLE_JIT=1``

``$ python setup.py test``

``$ unset NUMBA_DISABLE_JIT``

If you run the above from the main branch and get an error then please
send a report and/or add an issue, as all tests should pass.

The numba environment variable is set/unset in order to bypass the numba
just-in-time compilation process, which reduces testing times.

Example indices processing scripts
----------------------------------

There are example climate indices processing scripts provided which
compute the full suite of indices for various input dataset types. These
process input files in the NetCDF format, and produce output NetCDF
files in a corresponding format.

**nClimGrid**

The script ``process_grid.py`` (found under the ``scripts/process``
subdirectory) is used to compute climate indices from
`nClimGrid <https://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NCDC/Geoportal/iso/xml/C00332.xml&view=getDataView&header=none>`__
input datasets. Usage of this script requires specifying the input file
names and corresponding variable names for precipitation, temperature,
and soil constant datasets, as well as the month scales over which the
scaled indices (SPI, SPEI, and PAP) are to be computed, plus the base
output file name and the initial and final years of the calibration
period.

This script has the following required command line arguments:

+------------------------+---------------------------------------------+
| Option                 | Description                                 |
+========================+=============================================+
| precip_file            | input NetCDF file containing nClimGrid      |
|                        | precipitation dataset                       |
+------------------------+---------------------------------------------+
| precip_var_name        | name of the precipitation variable within   |
|                        | the input nClimGrid precipitation dataset   |
|                        | NetCDF                                      |
+------------------------+---------------------------------------------+
| temp_file              | input NetCDF file containing nClimGrid      |
|                        | temperature dataset                         |
+------------------------+---------------------------------------------+
| temp_var_name          | name of the temperature variable within the |
|                        | input nClimGrid temperature dataset NetCDF  |
+------------------------+---------------------------------------------+
| awc_file               | input NetCDF file containing a soil         |
|                        | constant (available water capacity of the   |
|                        | soil) dataset NetCDF, should correspond     |
|                        | dimensionally with the input nClimGrid      |
|                        | temperature and precipitation datasets      |
+------------------------+---------------------------------------------+
| awc_var_name           | name of the soil constant (available water  |
|                        | capacity of the soil) variable within the   |
|                        | input soil constant dataset NetCDF          |
+------------------------+---------------------------------------------+
| output_file_base       | base file name for all output files, each   |
|                        | computed index will have an output file     |
|                        | whose name will begin with this base plus   |
|                        | the index's abbreviation plus a month scale |
|                        | (if applicable), plus '.nc'
|                        | extension (i.e. for SPI/Gamma at 3-month    |
|                        | scale the resulting output file will be     |
|                        | named <output_file_base>_spi_gamma_03.nc)   |
+------------------------+---------------------------------------------+
| month_scales           | month scales over which the PAP, SPI, and   |
|                        | SPEI values are to be computed, valid range |
|                        | is 1-72 months                              |
+------------------------+---------------------------------------------+
| calibration_start_year | initial year of calibration period          |
+------------------------+---------------------------------------------+
| calibration_end_year   | final year of calibration period            |
+------------------------+---------------------------------------------+

*Example command line invocation*:

``$ nohup python -u process_grid.py --precip_file
example_inputs/nclimgrid_lowres_prcp.nc --temp_file
example_inputs/nclimgrid_lowres_tavg.nc --awc_file
example_inputs/nclimgrid_lowres_soil.nc --precip_var_name prcp
--temp_var_name tavg --awc_var_name awc --month_scales 1 2 3 6 12 24
--calibration_start_year 1931 --calibration_end_year 1990
--output_file_base nclimgrid_lowres``

**nClimDiv**

The script ``process_divisions.py`` (found under the ``scripts/process``
subdirectory) is used to compute climate indices from
`nClimDiv <https://www.ncdc.noaa.gov/monitoring-references/maps/us-climate-divisions.php>`__
input datasets. Usage of this script requires specifying the input file
name and corresponding variable names for precipitation, temperature,
and soil constant datasets, as well as the month scales over which the
scaled indices (SPI, SPEI, and PAP) are to be computed, plus the base
output file name and the initial and final years of the calibration
period.

This script has the following required command line arguments:

+------------------------+---------------------------------------------+
| Option                 | Description                                 |
+========================+=============================================+
| input_file             | input NetCDF file containing nClimDiv       |
|                        | temperature, precipitation, and soil        |
|                        | constant datasets, with output variables    |
|                        | added or updated for each computed index    |
+------------------------+---------------------------------------------+
| precip_var_name        | name of the precipitation variable within   |
|                        | the input nClimGrid dataset NetCDF          |
+------------------------+---------------------------------------------+
| temp_var_name          | name of the temperature variable within the |
|                        | input dataset NetCDF                        |
+------------------------+---------------------------------------------+
| awc_var_name           | name of the soil constant (available water  |
|                        | capacity of the soil) variable within the   |
|                        | input dataset NetCDF                        |
+------------------------+---------------------------------------------+
| month_scales           | month scales over which the PAP, SPI, and   |
|                        | SPEI values are to be computed, valid range |
|                        | is 1-72 months                              |
+------------------------+---------------------------------------------+
| calibration_start_year | initial year of calibration period          |
+------------------------+---------------------------------------------+
| calibration_end_year   | final year of calibration period            |
+------------------------+---------------------------------------------+

*Example command line invocation*:

``$ nohup python -u process_divisions.py --input_file
example_inputs/nclimdiv_20170404.nc --precip_var_name prcp --temp_var_name
tavg --awc_var_name awc --month_scales 1 2 3 6 12 24
--calibration_start_year 1931 --calibration_end_year 1990``

Get involved
------------

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
Please read more on our `license <LICENSE>`__ page.

.. |Build Status| image:: https://travis-ci.org/monocongo/climate_indices.svg?master
   :target: https://travis-ci.org/monocongo
.. |CodeFactor| image:: https://www.codefactor.io/repository/github/monocongo/climate_indices/badge/master
   :target: https://www.codefactor.io/repository/github/monocongo/climate_indices/overview/master
.. |Coverage Status| image:: https://coveralls.io/repos/github/monocongo/climate_indices/badge.svg?branch=master
   :target: https://coveralls.io/github/monocongo/climate_indices?branch=master
.. |Dependency Status| image:: https://gemnasium.com/badges/github.com/monocongo/climate_indices.svg
   :target: https://gemnasium.com/github.com/monocongo/climate_indices
.. |License| image:: https://img.shields.io/badge/License-BSD%203--Clause-green.svg
   :target: https://opensource.org/licenses/BSD-3-Clause

