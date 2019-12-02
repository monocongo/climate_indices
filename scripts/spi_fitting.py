import argparse
import concurrent.futures
import logging
import multiprocessing
from multiprocessing import shared_memory
from typing import Dict, List

import numpy as np
import xarray as xr

from climate_indices import compute, indices, utils


# ------------------------------------------------------------------------------
# Retrieve logger and set desired logging level
_logger = utils.get_logger(__name__, logging.INFO)


# ------------------------------------------------------------------------------
def compute_spi_gamma(
        da_precip: xr.DataArray,
        da_alpha: xr.DataArray,
        da_beta: xr.DataArray,
        scale: int,
        periodicity: compute.Periodicity,
        calibration_year_initial: int,
        calibration_year_final: int,
) -> xr.DataArray:

    initial_year = int(da_precip['time'][0].dt.year)
    total_lats = da_precip.shape[0]
    total_lons = da_precip.shape[1]
    spi = np.full(shape=da_precip.shape, fill_value=np.NaN)

    for lat_index in range(total_lats):
        for lon_index in range(total_lons):

            # get the values for the lat/lon grid cell
            values = da_precip[lat_index, lon_index]

            # skip over this grid cell if all NaN values
            if (np.ma.is_masked(values) and values.mask.all()) or np.all(np.isnan(values)):
                continue

            gamma_parameters = {
                "alphas": da_alpha[lat_index, lon_index],
                "betas": da_beta[lat_index, lon_index],
            }

            # compute the SPI
            spi_values = \
                indices.spi(
                    values,
                    scale=scale,
                    distribution=indices.Distribution.gamma,
                    data_start_year=initial_year,
                    calibration_year_initial=calibration_year_initial,
                    calibration_year_final=calibration_year_final,
                    periodicity=compute.Periodicity.monthly,
                    fitting_params=gamma_parameters,
                )
            spi[lat_index, lon_index] = spi_values[0]

            # build a DataArray for this scale's SPI
    da_spi = xr.DataArray(
        data=spi,
        coords=da_precip.coords,
        dims=da_precip.dims,
        name=f"spi_gamma_{scale}_{periodicity.unit()}",
    )
    da_spi.attrs = {
        'description': f'SPI ({scale}-{periodicity.unit()} gamma) computed from '
                       f'{periodicity} precipitation data for the period '
                       f'{da_precip.time[0]} through {da_precip.time[-1]} using '
                       f'a calibration period from {calibration_year_initial} '
                       f'through {calibration_year_final}',
        'valid_min': -3.09,
        'valid_max': 3.09,
        'long_name': f'{scale}-{periodicity.unit()} SPI(gamma)',
        'calibration_year_initial': calibration_year_initial,
        'calibration_year_final': calibration_year_final,
    }

    return da_spi


# ------------------------------------------------------------------------------
def compute_gammas(
        da_precip: xr.DataArray,
        scale: int,
        calibration_year_initial,
        calibration_year_final,
        periodicity: compute.Periodicity,
) -> (xr.DataArray, xr.DataArray):

    initial_year = int(da_precip['time'][0].dt.year)
    if periodicity == compute.Periodicity.monthly:
        period_times = 12
    elif periodicity == compute.Periodicity.daily:
        period_times = 366
    total_lats = da_precip.shape[0]
    total_lons = da_precip.shape[1]
    fitting_shape = (total_lats, total_lons, period_times)
    alphas = np.full(shape=fitting_shape, fill_value=np.NaN)
    betas = np.full(shape=fitting_shape, fill_value=np.NaN)

    # loop over the grid cells and compute the gamma parameters for each
    for lat_index in range(total_lats):
        for lon_index in range(total_lons):

            # get the precipitation values for the lat/lon grid cell
            values = da_precip[lat_index, lon_index]

            # skip over this grid cell if all NaN values
            if (np.ma.is_masked(values) and values.mask.all()) or np.all(np.isnan(values)):
                continue

            # convolve to scale
            scaled_values = \
                compute.scale_values(
                    values,
                    scale=scale,
                    periodicity=periodicity,
                )

            # compute the fitting parameters on the scaled data
            alphas[lat_index, lon_index], betas[lat_index, lon_index] = \
                compute.gamma_parameters(
                    scaled_values,
                    data_start_year=initial_year,
                    calibration_start_year=calibration_year_initial,
                    calibration_end_year=calibration_year_final,
                    periodicity=periodicity,
                )

    gamma_coords = {"lat": ds_prcp.lat, "lon": ds_prcp.lon, periodicity.unit(): range(period_times)}
    alpha_attrs = {
        'description': 'shape parameter of the gamma distribution (also referred to as the concentration) ' + \
                       f'computed from the {scale}-month scaled precipitation values',
    }
    da_alpha = xr.DataArray(
        data=alphas,
        coords=gamma_coords,
        dims=tuple(gamma_coords.keys()),
        name=f"alpha_{scale}_{periodicity.unit()}",
        attrs=alpha_attrs,
    )
    beta_attrs = {
        'description': '1 / scale of the distribution (also referred to as the rate) ' + \
                       f'computed from the {scale}-month scaled precipitation values',
    }
    da_beta = xr.DataArray(
        data=betas,
        coords=gamma_coords,
        dims=tuple(gamma_coords.keys()),
        name=f"beta_{scale}_{periodicity.unit()}",
        attrs=beta_attrs,
    )

    return da_alpha, da_beta


# ------------------------------------------------------------------------------
def compute_gammas_shm(
        da_precip: xr.DataArray,
        scale: int,
        calibration_year_initial,
        calibration_year_final,
        periodicity: compute.Periodicity,
) -> (xr.DataArray, xr.DataArray):

    def compute_gamma_latlon_shm(
            args: Dict,
    ):

        # get the shared memory array for precipitation
        existing_shm_precip = shared_memory.SharedMemory(name=args['shm_name_precip'])
        shm_ary_prcp = np.ndarray(shape=args['shape'], dtype=args['dtype'], buffer=existing_shm_precip.buf)

        # get the precipitation values for the lat/lon grid cell
        values = shm_ary_prcp[args['lat_index'], args['lon_index']]

        # skip over this grid cell if all NaN values
        if (np.ma.is_masked(values) and values.mask.all()) or np.all(np.isnan(values)):
            return

        # convolve to scale
        scaled_values = \
            compute.scale_values(
                values,
                scale=args['scale'],
                periodicity=args['periodicity'],
            )

        # get the shared memory arrays for alpha and beta
        existing_shm_alpha = shared_memory.SharedMemory(name=args['shm_name_alpha'])
        shared_alphas = np.ndarray(shape=args['shape'], dtype=args['dtype'], buffer=existing_shm_alpha.buf)
        existing_shm_beta = shared_memory.SharedMemory(name=args['shm_name_beta'])
        shared_betas = np.ndarray(shape=args['shape'], dtype=args['dtype'], buffer=existing_shm_beta.buf)

        # compute the fitting parameters on the scaled data
        shared_alphas[args['lat_index'], args['lon_index']], shared_betas[args['lat_index'], args['lon_index']] = \
            compute.gamma_parameters(
                scaled_values,
                data_start_year=args['initial_year'],
                calibration_start_year=args['calibration_year_initial'],
                calibration_end_year=args['calibration_year_final'],
                periodicity=args['periodicity'],
            )

    initial_year = int(da_precip['time'][0].dt.year)
    if periodicity == compute.Periodicity.monthly:
        period_times = 12
    elif periodicity == compute.Periodicity.daily:
        period_times = 366
    total_lats = da_precip.shape[0]
    total_lons = da_precip.shape[1]
    fitting_shape = (total_lats, total_lons, period_times)
    alphas = np.full(shape=fitting_shape, fill_value=np.NaN)
    betas = np.full(shape=fitting_shape, fill_value=np.NaN)

    # create shared memory arrays that can be accessed from worker processes
    shm_prcp = shared_memory.SharedMemory(create=True, size=da_precip.data.nbytes)
    shared_prcp = np.ndarray(da_precip.shape, dtype=da_precip.dtype, buffer=shm_prcp.buf)
    shared_prcp[:, :, :] = da_precip[:, :, :]
    shm_name_prcp = shm_prcp.name
    shm_alpha = shared_memory.SharedMemory(create=True, size=alphas.data.nbytes)
    shared_alpha = np.ndarray(alphas.shape, dtype=alphas.dtype, buffer=shm_alpha.buf)
    shared_alpha[:, :, :] = alphas[:, :, :]
    alphas = shared_alpha
    shm_name_alpha = shm_alpha.name
    shm_beta = shared_memory.SharedMemory(create=True, size=betas.data.nbytes)
    shared_beta = np.ndarray(betas.shape, dtype=betas.dtype, buffer=shm_beta.buf)
    shared_beta[:, :, :] = betas[:, :, :]
    betas = shared_beta
    shm_name_beta = shm_beta.name

    # loop over the grid cells and compute the gamma parameters for each
    arguments_list = []
    for lat_index in range(total_lats):
        for lon_index in range(total_lons):

            arguments = {
                'lat_index': lat_index,
                'lon_index': lon_index,
                'shm_name_precip': shm_name_prcp,
                'shm_name_alpha': shm_name_alpha,
                'shm_name_beta': shm_name_beta,
                'dtype': shared_prcp.dtype,
                'shape': shared_prcp.shape,
                'initial_year': initial_year,
                'calibration_year_initial': calibration_year_initial,
                'calibration_year_final': calibration_year_final,
                'periodicity': periodicity,
            }
            arguments_list.append(arguments)

    # use a ProcessPoolExecutor to compute the gammas in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:

        # use the executor to map the gammas computation function to the iterable of arguments
        executor.map(compute_gamma_latlon_shm, arguments_list)

    # build the gamma DataArrays from the computed arrays
    gamma_coords = {"lat": ds_prcp.lat, "lon": ds_prcp.lon, periodicity.unit(): range(period_times)}
    alpha_attrs = {
        'description': 'shape parameter of the gamma distribution (also referred to as the concentration) ' + \
                       f'computed from the {scale}-month scaled precipitation values',
    }
    da_alphas = xr.DataArray(
        data=alphas,
        coords=gamma_coords,
        dims=tuple(gamma_coords.keys()),
        name=f"alpha_{scale}_{periodicity.unit()}",
        attrs=alpha_attrs,
    )
    beta_attrs = {
        'description': '1 / scale of the distribution (also referred to as the rate) ' + \
                       f'computed from the {scale}-month scaled precipitation values',
    }
    da_betas = xr.DataArray(
        data=betas,
        coords=gamma_coords,
        dims=tuple(gamma_coords.keys()),
        name=f"beta_{scale}_{periodicity.unit()}",
        attrs=beta_attrs,
    )

    return da_alphas, da_betas


# ------------------------------------------------------------------------------
def build_fitting_dataset(
        ds_base: xr.Dataset,
        periodicity: compute.Periodicity,
) -> xr.Dataset:

    if periodicity == compute.Periodicity.monthly:
        period_times = 12
    elif periodicity == compute.Periodicity.daily:
        period_times = 366
    else:
        raise ValueError(f"Unsupported periodicity: {periodicity}")

    attrs_to_copy = [
        'Conventions',
        'ncei_template_version',
        'naming_authority',
        'standard_name_vocabulary',
        'institution',
        'geospatial_lat_min',
        'geospatial_lat_max',
        'geospatial_lon_min',
        'geospatial_lon_max',
        'geospatial_lat_units',
        'geospatial_lon_units',
    ]
    global_attrs = {key: value for (key, value) in ds_base.attrs.items() if key in attrs_to_copy}
    global_attrs['description'] = \
        f"Distribution fitting parameters for various {periodicity.unit()} scales "\
        f"computed from {periodicity} precipitation input by the climate_indices "\
        "package available from https://github.com/monocongo/climate_indices. "\
        "The variables contain herein are meant to be used as inputs for computing "\
        "SPI and SPEI datasets using the climate_indices package. See "\
        "https://climate-indices.readthedocs.io/en/latest/#spi-monthly for "\
        "example usage."
    coords = {"lat": ds_base.lat, "lon": ds_base.lon, periodicity.unit(): range(period_times)}
    ds_fitting_params = xr.Dataset(
        coords=coords,
        attrs=global_attrs,
    )

    return ds_fitting_params


# ------------------------------------------------------------------------------
def build_spi_gamma_dataset(
        ds_base: xr.Dataset,
        periodicity: compute.Periodicity,
        scale: int,
) -> xr.Dataset:

    attrs_to_copy = [
        'Conventions',
        'ncei_template_version',
        'naming_authority',
        'standard_name_vocabulary',
        'institution',
        'geospatial_lat_min',
        'geospatial_lat_max',
        'geospatial_lon_min',
        'geospatial_lon_max',
        'geospatial_lat_units',
        'geospatial_lon_units',
    ]
    global_attrs = {key: value for (key, value) in ds_base.attrs.items() if key in attrs_to_copy}
    global_attrs['description'] = \
        f"SPI computed using a gamma distribution for {scale}-{periodicity.unit()} scale, "\
        f"computed from {periodicity} precipitation input by the climate_indices "\
        "package available from https://github.com/monocongo/climate_indices. "
    ds_spi_gamma = xr.Dataset(
        coords=ds_base.coords,
        attrs=global_attrs,
    )

    return ds_spi_gamma


# ------------------------------------------------------------------------------
def validate_compatibility(
        ds_prcp: xr.Dataset,
        ds_fitting: xr.Dataset,
        periodicity: compute.Periodicity,
        scales: List[int],
):

    latlon = {'lat', 'lon'}
    for ds in [ds_prcp, ds_fitting]:
        if not latlon.issubset(ds.dims.keys()):
            raise ValueError("The expected dimensions 'lat' and 'lon' were not found")
        if not latlon.issubset(ds_fitting.coords.keys()):
            raise ValueError("The expected coordinates 'lat' and 'lon' were not found")
    for dim in latlon:
        if not np.allclose(ds_prcp[dim], ds_fitting[dim]):
            raise ValueError(f"The '{dim}' dimension values vary between the two datasets")
    fitting_dims = {'lat', 'lon', periodicity.unit()}
    if not fitting_dims == ds_fitting.coords.keys():
        raise ValueError(f"Unexpected coordinates in the fitting parameters dataset")
    if not fitting_dims == ds_fitting.dims.keys():
        raise ValueError(f"Unexpected coordinates in the fitting parameters dataset")
    for scale in scales:
        variable_name_alpha = f"alpha_{scale}_{periodicity.unit()}"
        variable_name_beta = f"beta_{scale}_{periodicity.unit()}"
        for var_name in [variable_name_alpha, variable_name_beta]:
            if var_name not in ds_fitting.data_vars:
                raise ValueError(f"Missing the expected fitting variable {var_name}")


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=str,
        help="Indices to compute",
        choices=["spi"],
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
        ds_prcp = xr.open_dataset(cli_args['netcdf_precip'])
        da_prcp = ds_prcp[cli_args['var_name_precip']].transpose('lat', 'lon', 'time')

        # get the fitting parameters Dataset
        if cli_args['load_params']:

            ds_fitting = xr.open_dataset(cli_args['load_params'])
            validate_compatibility(ds_prcp, ds_fitting, cli_args['periodicity'], cli_args['scales'])

        else:

            ds_fitting = build_fitting_dataset(ds_prcp, cli_args['periodicity'])

        # compute alphas, betas, and SPI/gamma for each scale
        for scale in cli_args['scales']:

            # use standard variable names
            var_name_alpha = f"alpha_{scale}_{cli_args['periodicity'].unit()}"
            var_name_beta = f"beta_{scale}_{cli_args['periodicity'].unit()}"

            if cli_args['load_params']:
                da_alpha = ds_fitting[var_name_alpha]
                da_beta = ds_fitting[var_name_beta]
            else:
                _logger.info(f"Computing gamma parameters for {scale}-{cli_args['periodicity'].unit()}")
                da_alpha, da_beta = \
                    compute_gammas(
                        da_prcp,
                        scale,
                        calibration_year_initial=cli_args["calibration_start_year"],
                        calibration_year_final=cli_args["calibration_end_year"],
                        periodicity=cli_args["periodicity"],
                    )
                ds_fitting[var_name_alpha] = da_alpha
                ds_fitting[var_name_beta] = da_beta

            # compute the SPI/gamma for this scale
            _logger.info(f"Computing SPI for {scale}-{cli_args['periodicity'].unit()}")
            da_spi = compute_spi_gamma(
                da_prcp,
                da_alpha,
                da_beta,
                scale,
                cli_args["periodicity"],
                cli_args["calibration_start_year"],
                cli_args["calibration_end_year"],
            )

            # write the SPI as NetCDF
            ds_spi_gamma = build_spi_gamma_dataset(ds_prcp, cli_args['periodicity'], scale)
            var_name_spi = f"spi_gamma_{scale}_{cli_args['periodicity'].unit()}"
            ds_spi_gamma[var_name_spi] = da_spi
            ds_spi_gamma.to_netcdf(f"{cli_args['output_file_base']}_{var_name_spi}.nc")

        # if specified write the fitting parameters to file
        if cli_args['save_params']:
            ds_fitting.to_netcdf(cli_args['save_params'])

    else:
        raise ValueError(f"Unsupported index computation: {cli_args['index']}")
