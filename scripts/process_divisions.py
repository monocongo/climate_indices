import argparse
from datetime import datetime
import logging
import multiprocessing

import netCDF4
import numpy as np
import scipy.constants

from climate_indices import compute, indices
from scripts import netcdf_utils

# ----------------------------------------------------------------------------------------------------------------------
# set up a basic, global logger which will write to the console as standard error
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------
# static constants
_VALID_MIN = -10.0
_VALID_MAX = 10.0

# ----------------------------------------------------------------------------------------------------------------------
# multiprocessing lock we'll use to synchronize I/O writes to NetCDF files, one per each output file
lock_output = multiprocessing.Lock()


# ----------------------------------------------------------------------------------------------------------------------
class DivisionsProcessor(object):
    def __init__(
        self,
        input_file,
        output_file,
        var_name_precip,
        var_name_temperature,
        var_name_soil,
        month_scales,
        calibration_start_year,
        calibration_end_year,
        divisions=None,
    ):

        """
        Constructor method.

        :param input_file:
        :param var_name_precip:
        :param var_name_temperature:
        :param var_name_soil:
        :param month_scales:
        :param calibration_start_year:
        :param calibration_end_year:
        :param divisions:
        """

        self.input_file = input_file
        self.output_file = output_file
        self.var_name_precip = var_name_precip
        self.var_name_temperature = var_name_temperature
        self.var_name_soil = var_name_soil
        self.scale_months = month_scales
        self.calibration_start_year = calibration_start_year
        self.calibration_end_year = calibration_end_year
        self.divisions = divisions

        # TODO get the initial year from the precipitation NetCDF, for now use hard-coded value specific to nClimDiv
        self.data_start_year = 1895

        # create and populate the NetCDF we'll use to contain our results of a call to run()
        self._initialize_netcdf()

    # ------------------------------------------------------------------------------------------------------------------
    def _initialize_netcdf(self):

        netcdf_utils.initialize_netcdf_divisions(
            self.output_file,
            self.input_file,
            self.var_name_precip,
            _variable_info(self.scale_months),
            True,
        )

    # ------------------------------------------------------------------------------------------------------------------
    def _compute_and_write_division(self, div_index):
        """
        Computes indices for a single division, writing the output into NetCDF.

        :param div_index:
        """

        # only process specified divisions
        if self.divisions is not None and div_index not in self.divisions:
            return

        # open the NetCDF files
        with netCDF4.Dataset(self.input_file, "a") as input_divisions, netCDF4.Dataset(
            self.output_file, "a"
        ) as output_divisions:

            climdiv_id = input_divisions["division"][div_index]

            # only process divisions within CONUS, 101 - 4811
            if climdiv_id > 4811:
                return

            logger.info("Processing indices for division %s", climdiv_id)

            # read the division of input temperature values
            temperature = input_divisions[self.var_name_temperature][
                div_index, :
            ]  # assuming dims (divisions, time)

            # initialize the latitude outside of the valid range, in order to use
            # this within a conditional below to verify a valid latitude
            latitude = -100.0

            # latitudes are only available for certain divisions, make sure we have one for this division index
            if div_index < input_divisions["lat"][:].size:

                # get the actual latitude value (assumed to be in degrees north)
                # for the latitude slice specified by the index
                latitude = input_divisions["lat"][div_index]

            # only proceed if the latitude value is within valid range
            if not np.isnan(latitude) and (latitude < 90.0) and (latitude > -90.0):

                # convert temperatures from Fahrenheit to Celsius, if necessary
                temperature_units = input_divisions[self.var_name_temperature].units
                if temperature_units in [
                    "degree_Fahrenheit",
                    "degrees Fahrenheit",
                    "degrees F",
                    "fahrenheit",
                    "Fahrenheit",
                    "F",
                ]:

                    # TODO make sure this application of the ufunc is any faster  pylint: disable=fixme
                    temperature = scipy.constants.convert_temperature(
                        temperature, "F", "C"
                    )

                elif temperature_units not in [
                    "degree_Celsius",
                    "degrees Celsius",
                    "degrees C",
                    "celsius",
                    "Celsius",
                    "C",
                ]:

                    raise ValueError(
                        "Unsupported temperature units: '{0}'".format(temperature_units)
                    )

                # TODO instead use the numpy.apply_along_axis() function for computing indices such as PET
                # that take a single time series array as input (i.e. each division's time series is the initial
                # 1-D array argument to the function we'll apply)

                logger.info("\tComputing PET for division %s", climdiv_id)

                logger.info("\t\tCalculating PET using Thornthwaite method")

                # compute PET across all longitudes of the latitude slice
                # Thornthwaite PE
                pet_time_series = indices.pet(
                    temperature,
                    latitude_degrees=latitude,
                    data_start_year=self.data_start_year,
                )

                # the above returns PET in millimeters, note this for further consideration
                pet_units = "millimeter"

                # write the PET values to NetCDF
                lock_output.acquire()
                output_divisions["pet"][div_index, :] = np.reshape(
                    pet_time_series, (1, pet_time_series.size)
                )
                output_divisions.sync()
                lock_output.release()

            else:

                pet_time_series = np.full(temperature.shape, np.NaN)
                pet_units = None

            # read the division's input precipitation and available water capacity values
            precip_time_series = input_divisions[self.var_name_precip][
                div_index, :
            ]  # assuming dims (divisions, time)

            if div_index < input_divisions[self.var_name_soil][:].size:

                awc = input_divisions[self.var_name_soil][
                    div_index
                ]  # assuming (divisions) dims orientation

                # AWC values need to include top inch, values from the soil file do not, so we add top inch here
                awc += 1

            else:
                awc = np.NaN

            # compute SPI and SPEI for the current division only if we have valid inputs
            if not np.isnan(precip_time_series).all():

                # put precipitation into inches if not already
                mm_to_inches_multiplier = 0.0393701
                possible_mm_units = ["millimeters", "millimeter", "mm"]
                if input_divisions[self.var_name_precip].units in possible_mm_units:
                    precip_time_series = precip_time_series * mm_to_inches_multiplier

                # only compute Palmers if we have PET already
                if not np.isnan(pet_time_series).all():

                    # compute Palmer indices if we have valid inputs
                    if not np.isnan(awc):

                        # if PET is in mm, convert to inches
                        if pet_units in possible_mm_units:
                            pet_time_series = pet_time_series * mm_to_inches_multiplier

                        # PET is in mm, convert to inches since the Palmer uses imperial units
                        pet_time_series = pet_time_series * mm_to_inches_multiplier

                        logger.info("\tComputing PDSI for division %s", climdiv_id)

                        # compute Palmer indices
                        palmer_values = indices.scpdsi(
                            precip_time_series,
                            pet_time_series,
                            awc,
                            self.data_start_year,
                            self.calibration_start_year,
                            self.calibration_end_year,
                        )

                        # pull Palmer indices out of the returned array (for code clarity)
                        scpdsi = palmer_values[0]
                        pdsi = palmer_values[1]
                        phdi = palmer_values[2]
                        pmdi = palmer_values[3]
                        zindex = palmer_values[4]

                        # write the Palmer index values to NetCDF
                        lock_output.acquire()
                        output_divisions["pdsi"][div_index, :] = np.reshape(
                            pdsi, (1, pdsi.size)
                        )
                        output_divisions["phdi"][div_index, :] = np.reshape(
                            phdi, (1, phdi.size)
                        )
                        output_divisions["pmdi"][div_index, :] = np.reshape(
                            pmdi, (1, pmdi.size)
                        )
                        output_divisions["scpdsi"][div_index, :] = np.reshape(
                            pdsi, (1, scpdsi.size)
                        )
                        output_divisions["zindex"][div_index, :] = np.reshape(
                            zindex, (1, zindex.size)
                        )
                        output_divisions.sync()
                        lock_output.release()

                    # process the SPI, SPEI, and PNP at the specified month scales
                    for months in self.scale_months:
                        logger.info(
                            "\tComputing SPI/SPEI/PNP at %s-month scale for division %s",
                            months,
                            climdiv_id,
                        )

                        # TODO ensure that the precipitation and PET values are using the same units

                        # compute SPEI/Gamma
                        spei_gamma = indices.spei(
                            precip_time_series,
                            pet_time_series,
                            months,
                            indices.Distribution.gamma,
                            compute.Periodicity.monthly,
                            self.data_start_year,
                            self.calibration_start_year,
                            self.calibration_end_year,
                        )

                        # compute SPEI/Pearson
                        spei_pearson = indices.spei(
                            precip_time_series,
                            pet_time_series,
                            months,
                            indices.Distribution.pearson,
                            compute.Periodicity.monthly,
                            self.data_start_year,
                            self.calibration_start_year,
                            self.calibration_end_year,
                        )

                        # compute SPI/Gamma
                        spi_gamma = indices.spi(
                            precip_time_series,
                            months,
                            indices.Distribution.gamma,
                            self.data_start_year,
                            self.calibration_start_year,
                            self.calibration_end_year,
                            compute.Periodicity.monthly,
                        )

                        # compute SPI/Pearson
                        spi_pearson = indices.spi(
                            precip_time_series,
                            months,
                            indices.Distribution.pearson,
                            self.data_start_year,
                            self.calibration_start_year,
                            self.calibration_end_year,
                            compute.Periodicity.monthly,
                        )

                        # compute PNP
                        pnp = indices.percentage_of_normal(
                            precip_time_series,
                            months,
                            self.data_start_year,
                            self.calibration_start_year,
                            self.calibration_end_year,
                            compute.Periodicity.monthly,
                        )

                        # create variable names which should correspond to the appropriate scaled index output variables
                        scaled_name_suffix = str(months).zfill(2)
                        spei_gamma_variable_name = "spei_gamma_" + scaled_name_suffix
                        spei_pearson_variable_name = (
                            "spei_pearson_" + scaled_name_suffix
                        )
                        spi_gamma_variable_name = "spi_gamma_" + scaled_name_suffix
                        spi_pearson_variable_name = "spi_pearson_" + scaled_name_suffix
                        pnp_variable_name = "pnp_" + scaled_name_suffix

                        # write the SPI, SPEI, and PNP values to NetCDF
                        lock_output.acquire()
                        output_divisions[spei_gamma_variable_name][
                            div_index, :
                        ] = np.reshape(spei_gamma, (1, spei_gamma.size))
                        output_divisions[spei_pearson_variable_name][
                            div_index, :
                        ] = np.reshape(spei_pearson, (1, spei_pearson.size))
                        output_divisions[spi_gamma_variable_name][
                            div_index, :
                        ] = np.reshape(spi_gamma, (1, spi_gamma.size))
                        output_divisions[spi_pearson_variable_name][
                            div_index, :
                        ] = np.reshape(spi_pearson, (1, spi_pearson.size))
                        output_divisions[pnp_variable_name][div_index, :] = np.reshape(
                            pnp, (1, pnp.size)
                        )
                        output_divisions.sync()
                        lock_output.release()

    # ------------------------------------------------------------------------------------------------------------------
    def run(self):

        # initialize the output NetCDF that will contain the computed indices
        with netCDF4.Dataset(self.input_file) as input_dataset:

            # get the initial and final year of the input datasets
            time_variable = input_dataset.variables["time"]
            self.data_start_year = netCDF4.num2date(
                time_variable[0], time_variable.units
            ).year

            # get the number of divisions in the input dataset(s)
            divisions_count = input_dataset.variables["division"].size

        # --------------------------------------------------------------------------------------------------------------
        # Create PET and Palmer index NetCDF files, computed from input temperature, precipitation, and soil constant.
        # Compute SPI, SPEI, and PNP at all specified month scales.
        # --------------------------------------------------------------------------------------------------------------

        # create a process Pool for worker processes to compute indices for each division
        pool = multiprocessing.Pool(
            processes=multiprocessing.cpu_count()
        )  # use single process here when debugging

        # map the divisions indices as an arguments iterable to the compute function
        result = pool.map_async(
            self._compute_and_write_division, range(divisions_count)
        )

        # get the exception(s) thrown, if any
        result.get()

        # close the pool and wait on all processes to finish
        pool.close()
        pool.join()


# ----------------------------------------------------------------------------------------------------------------------
def _variable_info(month_scales):
    """

    :param month_scales:
    :return: for month-scaled indices (SPI, SPEI, and PNP) a list of the number of months to use as scale
    """

    # the dictionary of variable names (keys) to dictionaries of variable attributes (values) we'll populate and return
    variable_attributes = {}

    variable_attributes["pet"] = {
        "standard_name": "pet",
        "long_name": "Potential Evapotranspiration (PET), from Thornthwaite's equation",
        "valid_min": 0.0,
        "valid_max": 2000.0,
        "units": "millimeter",
    }
    variable_attributes["pdsi"] = {
        "standard_name": "pdsi",
        "long_name": "Palmer Drought Severity Index (PDSI)",
        "valid_min": -10.0,
        "valid_max": 10.0,
    }
    variable_attributes["scpdsi"] = {
        "standard_name": "scpdsi",
        "long_name": "Self-calibrated Palmer Drought Severity Index (PDSI)",
        "valid_min": -10.0,
        "valid_max": 10.0,
    }
    variable_attributes["phdi"] = {
        "standard_name": "phdi",
        "long_name": "Palmer Hydrological Drought Index (PHDI)",
        "valid_min": -10.0,
        "valid_max": 10.0,
    }
    variable_attributes["pmdi"] = {
        "standard_name": "pmdi",
        "long_name": "Palmer Modified Drought Index (PMDI)",
        "valid_min": -10.0,
        "valid_max": 10.0,
    }
    variable_attributes["zindex"] = {
        "standard_name": "zindex",
        "long_name": "Palmer Z-Index",
        "valid_min": -10.0,
        "valid_max": 10.0,
    }

    for months in month_scales:
        variable_name = "pnp_" + str(months).zfill(2)
        variable_attributes[variable_name] = {
            "standard_name": variable_name,
            "long_name": "Percent average precipitation, {}-month scale".format(months),
            "valid_min": 0,
            "valid_max": 10.0,
            "units": "percent of average",
        }
        variable_name = "spi_gamma_" + str(months).zfill(2)
        variable_attributes[variable_name] = {
            "standard_name": variable_name,
            "long_name": "SPI (Gamma), {}-month scale".format(months),
            "valid_min": -3.09,
            "valid_max": 3.09,
        }
        variable_name = "spi_pearson_" + str(months).zfill(2)
        variable_attributes[variable_name] = {
            "standard_name": variable_name,
            "long_name": "SPI (Pearson), {}-month scale".format(months),
            "valid_min": -3.09,
            "valid_max": 3.09,
        }
        variable_name = "spei_gamma_" + str(months).zfill(2)
        variable_attributes[variable_name] = {
            "standard_name": variable_name,
            "long_name": "SPEI (Gamma), {}-month scale".format(months),
            "valid_min": -3.09,
            "valid_max": 3.09,
        }
        variable_name = "spei_pearson_" + str(months).zfill(2)
        variable_attributes[variable_name] = {
            "standard_name": variable_name,
            "long_name": "SPEI (Pearson), {}-month scale".format(months),
            "valid_min": -3.09,
            "valid_max": 3.09,
        }

    return variable_attributes


# ----------------------------------------------------------------------------------------------------------------------
def process_divisions(
    input_file,
    output_file,
    precip_var_name,
    temp_var_name,
    awc_var_name,
    month_scales,
    calibration_start_year,
    calibration_end_year,
    divisions=None,
):

    """
    Performs indices processing from climate divisions inputs.

    :param input_file
    :param output_file
    :param precip_var_name
    :param temp_var_name
    :param awc_var_name
    :param month_scales
    :param calibration_start_year
    :param calibration_end_year
    :param divisions: list of divisions to compute, if None (default) then all divisions are included
    """

    # perform the processing
    divisions_processor = DivisionsProcessor(
        input_file,
        output_file,
        precip_var_name,
        temp_var_name,
        awc_var_name,
        month_scales,
        calibration_start_year,
        calibration_end_year,
        divisions,
    )
    divisions_processor.run()


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    This module is used to perform climate indices processing on nClimGrid datasets in NetCDF.
    """

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        logger.info("Start time:    %s", start_datetime)

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input_file",
            help="Input dataset file (NetCDF) containing temperature, precipitation, and soil "
            "values for PDSI, SPI, SPEI, and PNP computations",
            required=True,
        )
        parser.add_argument(
            "--var_name_precip",
            help="Precipitation variable name used in the input NetCDF file",
            required=True,
        )
        parser.add_argument(
            "--var_name_temp",
            help="Temperature variable name used in the input NetCDF file",
            required=True,
        )
        parser.add_argument(
            "--var_name_awc",
            help="Available water capacity variable name used in the input NetCDF file",
            required=False,
        )
        parser.add_argument("--output_file", help=" Output file path", required=True)
        parser.add_argument(
            "--scales",
            help="Month scales over which the PNP, SPI, and SPEI values are to be computed",
            type=int,
            nargs="*",
            choices=range(1, 73),
            required=True,
        )
        parser.add_argument(
            "--calibration_start_year",
            help="Initial year of calibration period",
            type=int,
            choices=range(1870, start_datetime.year + 1),
            required=True,
        )
        parser.add_argument(
            "--calibration_end_year",
            help="Final year of calibration period",
            type=int,
            choices=range(1870, start_datetime.year + 1),
            required=True,
        )
        parser.add_argument(
            "--divisions",
            help="Divisions for which the PNP, SPI, and SPEI values are to be computed "
            "(useful for specifying a short list of divisions",
            type=int,
            nargs="*",
            choices=range(101, 4811),
            required=False,
        )
        args = parser.parse_args()

        # perform the processing
        process_divisions(
            args.input_file,
            args.output_file,
            args.var_name_precip,
            args.var_name_temp,
            args.var_name_awc,
            args.scales,
            args.calibration_start_year,
            args.calibration_end_year,
            args.divisions,
        )

        # report on the elapsed time
        end_datetime = datetime.now()
        logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        logger.exception("Failed to complete", exc_info=True)
        raise
