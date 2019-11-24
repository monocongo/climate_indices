import argparse

import numpy as np
import xarray as xr

from climate_indices import compute, indices


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=str,
        help="Indices to compute",
        choices=["spi", "spei", "pnp", "scaled", "pet", "palmers", "all"],
        required=True,
    )
    parser.add_argument(
        "--periodicity",
        help="Process input as either monthly or daily values",
        choices=[compute.Periodicity.monthly, compute.Periodicity.daily],
        type=compute.Periodicity.from_string,
        required=True,
    )
    parser.add_argument(
        "--scales",
        help="Timestep scales over which the PNP, SPI, and SPEI values are to be computed",
        type=int,
        nargs="*",
    )
    parser.add_argument(
        "--calibration_start_year",
        help="Initial year of the calibration period",
        type=int,
    )
    parser.add_argument(
        "--calibration_end_year",
        help="Final year of calibration period",
        type=int,
    )
    parser.add_argument(
        "--netcdf_precip",
        type=str,
        help="Precipitation NetCDF file to be used as input for indices computations",
    )
    parser.add_argument(
        "--var_name_precip",
        type=str,
        help="Precipitation variable name used in the precipitation NetCDF file",
    )
    parser.add_argument(
        "--netcdf_temp",
        type=str,
        help="Temperature NetCDF file to be used as input for indices computations",
    )
    parser.add_argument(
        "--var_name_temp",
        type=str,
        help="Temperature variable name used in the temperature NetCDF file",
    )
    parser.add_argument(
        "--netcdf_pet",
        type=str,
        help="PET NetCDF file to be used as input for SPEI and/or Palmer computations",
    )
    parser.add_argument(
        "--var_name_pet",
        type=str,
        help="PET variable name used in the PET NetCDF file",
    )
    parser.add_argument(
        "--netcdf_awc",
        type=str,
        help="Available water capacity NetCDF file to be used as input for the Palmer computations",
    )
    parser.add_argument(
        "--var_name_awc",
        type=str,
        help="Available water capacity variable name used in the AWC NetCDF file",
    )
    parser.add_argument(
        "--output_file_base",
        type=str,
        help="Base output file path and name for the resulting output files",
        required=True,
    )
    parser.add_argument(
        "--multiprocessing",
        help="options for multiprocessing -- single core, all cores but one, or all cores",
        choices=["single", "all_but_one", "all"],
        required=False,
        default="all_but_one",
    )
    parser.add_argument(
        "--load_params",
        type=str,
        required=False,
        help="path to input NetCDF file (to be read) "
             "containing distribution fitting parameters",
    )
    parser.add_argument(
        "--save_params",
        type=str,
        required=False,
        help="path to output NetCDF file (to be written) "
             "containing distribution fitting parameters",
    )
    cli_args = vars(parser.parse_args())

    if cli_args["index"] == "spi":

        # read the precipitation into an array in lat, lon, time order
        ds = xr.open_dataset(cli_args['netcdf_precip'])
        da_prcp = ds[cli_args['var_name_precip']].transpose('lat', 'lon', 'time')

        # work out various dimensions and initial values
        initial_year = int(da_prcp['time'][0].dt.year)
        if cli_args['periodicity'] == compute.Periodicity.monthly:
            period_times = 12
        else:
            period_times = 366
        total_lats = da_prcp.shape[0]
        total_lons = da_prcp.shape[1]
        fitting_shape = (total_lats, total_lons, period_times)

        # compute alphas, betas, and SPI/gamma for each scale
        for scale in cli_args['scales']:
            alphas = np.full(shape=fitting_shape, fill_value=np.NaN)
            betas = np.full(shape=fitting_shape, fill_value=np.NaN)
            spi = np.full(shape=da_prcp.shape, fill_value=np.NaN)

            # loop over each grid cell
            for lat_index in range(total_lats):
                for lon_index in range(total_lons):

                    # get the values for the lat/lon grid cell
                    values = da_prcp[lat_index, lon_index]

                    # skip over this grid cell if all NaN values
                    if (np.ma.is_masked(values) and values.mask.all()) or np.all(np.isnan(values)):
                        continue

                    # scale to 3-month convolutions
                    scaled_values = \
                        compute.scale_values(
                            values,
                            scale=scale,
                            periodicity=cli_args['periodicity'],
                        )

                    # compute the fitting parameters on the scaled data
                    alphas[lat_index, lon_index], betas[lat_index, lon_index] = \
                        compute.gamma_parameters(
                            scaled_values,
                            data_start_year=initial_year,
                            calibration_start_year=1900,
                            calibration_end_year=2000,
                            periodicity=cli_args['periodicity'],
                        )

                    gamma_parameters = {
                        "alphas": alphas[lat_index, lon_index],
                        "betas": betas[lat_index, lon_index],
                    }
                    spi[lat_index, lon_index] = \
                        indices.spi(
                            values,
                            scale=scale,
                            distribution=indices.Distribution.gamma,
                            data_start_year=initial_year,
                            calibration_year_initial=1900,
                            calibration_year_final=2000,
                            periodicity=cli_args['periodicity'],
                            fitting_params=gamma_parameters,
                        )

            # TODO combine the alphas and betas into a single xarray
            #  Dataset and write as NetCDF for later use

            # create a new DataArray for this scale's SPI and write as NetCDF
            da_spi = da_prcp.copy(data=spi)

            # copy the original Dataset and drop all the variables
            ds_spi = ds.copy()
            for var in ds_spi.data_vars:
                ds_spi = ds_spi.drop(var)

            # add the SPI DataArray as the only data variable in the Dataset
            spi_var_name = f'spi_gamma_{str(scale).zfill(2)}'
            ds_spi[spi_var_name] = da_spi

            # TODO create attributes to describe the SPI variable

            # write the SPI as NetCDF
            ds_spi.to_netcdf(f"{cli_args['output_file_base']}_{spi_var_name}.nc")

    else:
        raise ValueError(f"Unsupported index computation: {cli_args['index']}")
