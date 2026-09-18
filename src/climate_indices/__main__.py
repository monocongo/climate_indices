"""Command-line interface for climate indices processing"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, cast

import numpy as np
import scipy.constants
import xarray as xr

from climate_indices import compute, fire, indices, palmer, utils
from climate_indices._cli import _add_common_spi_arguments, _open_with_default_chunks
from climate_indices.exceptions import ConvergenceError, InsufficientDataError
from climate_indices.validation import DatasetLayout, detect_dataset_layout, expected_dimensions

# the number of worker processes we'll use for process pools
_NUMBER_OF_WORKER_PROCESSES = multiprocessing.cpu_count() - 1
# shared memory array dictionary keys
_KEY_ARRAY = "array"
_KEY_SHAPE = "shape"
_KEY_LAT = "lat"
_KEY_RESULT = "result_array"
_KEY_RESULT_PDSI = "result_array_pdsi"
_KEY_RESULT_PHDI = "result_array_phdi"
_KEY_RESULT_PMDI = "result_array_pmdi"
_KEY_RESULT_ZINDEX = "result_array_zindex"
_KEY_RESULT_SCPDSI = "result_array_scpdsi"

# global dictionary to contain shared arrays for use by worker processes
_global_shared_arrays: dict[str, Any] = {}

# Retrieve logger and set desired logging level
_logger = utils.get_logger(__name__, logging.INFO)


@dataclass(frozen=True)
class _InputContext:
    """
    A precipitation or temperature input's validated dimensions, used to check
    the companion inputs against it.

    Built from precipitation for every index but ``pet``, which is computed
    from temperature alone; only that route leaves ``latitudes``,
    ``longitudes``, and ``divisions`` unset, since temperature alone carries
    no coordinate data to compare companion inputs against.
    """

    input_type: DatasetLayout
    dimensions: tuple[Hashable, ...]
    times: np.ndarray
    latitudes: np.ndarray | None = None
    longitudes: np.ndarray | None = None
    divisions: np.ndarray | None = None


@dataclass
class _IndexRequest:
    """
    One index computation's inputs and output settings.

    Replaces the untyped dictionary the CLI used to rebuild at every call site:
    the fields a given index does not use stay ``None``, and the dataclass is
    mutable only because the inputs' start year is known once they are opened.
    """

    index: str
    output_file_base: str
    input_type: DatasetLayout
    periodicity: compute.Periodicity
    chunksizes: str
    netcdf_precip: str | None = None
    var_name_precip: str | None = None
    netcdf_temp: str | None = None
    var_name_temp: str | None = None
    netcdf_pet: str | None = None
    var_name_pet: str | None = None
    netcdf_awc: str | None = None
    var_name_awc: str | None = None
    scale: int | None = None
    distribution: indices.Distribution | None = None
    calibration_start_year: int | None = None
    calibration_end_year: int | None = None
    # the initial year of the inputs, read from them as the computation starts
    data_start_year: int | None = None

    @classmethod
    def from_arguments(
        cls,
        arguments: argparse.Namespace,
        *,
        index: str,
        input_type: DatasetLayout,
        scale: int | None = None,
        distribution: indices.Distribution | None = None,
    ) -> _IndexRequest:
        """
        Build a request for one index from the parsed command line arguments.

        param arguments: the parsed command line arguments
        param index: the index to compute, which may be a member of the
            ``--index`` value rather than the value itself
        param input_type: the input type determined by argument validation
        param scale: the time scale to compute, for a scaled index
        param distribution: the distribution to fit, for a fitted index
        return: the request those arguments describe
        """
        inputs = {name: getattr(arguments, name) for name in _registry_for(index).input_paths}
        return cls(
            index=index,
            output_file_base=arguments.output_file_base,
            input_type=input_type,
            periodicity=arguments.periodicity,
            chunksizes=arguments.chunksizes,
            netcdf_precip=inputs.get("netcdf_precip"),
            var_name_precip=inputs.get("var_name_precip"),
            netcdf_temp=inputs.get("netcdf_temp"),
            var_name_temp=inputs.get("var_name_temp"),
            netcdf_pet=inputs.get("netcdf_pet"),
            var_name_pet=inputs.get("var_name_pet"),
            netcdf_awc=inputs.get("netcdf_awc"),
            var_name_awc=inputs.get("var_name_awc"),
            calibration_start_year=arguments.calibration_start_year,
            calibration_end_year=arguments.calibration_end_year,
            scale=scale,
            distribution=distribution,
        )


@dataclass(frozen=True)
class _ComputeContext:
    """The opened inputs and output settings a registration's compute and write steps share."""

    request: _IndexRequest
    dataset: xr.Dataset
    output_dims: tuple[Hashable, ...]
    output_shape: tuple[int, ...]
    output_encodings: dict[str, Any] | None
    output_engine: Literal["h5netcdf"] | None
    arguments: dict[str, Any]
    # inputs a registration prepared alongside the request, e.g. Palmer's AWC
    prepared: xr.Dataset | None = None


def _accepted_dimensions(layout: DatasetLayout) -> tuple[tuple[Hashable, ...], ...]:
    """
    Every dimension order a variable in a dataset of this layout may use.

    A dataset holds the data variables, which carry the layout's time
    dimension, and per-location companions such as the division latitudes,
    which do not.

    param layout: the dataset layout the dimensions are accepted for
    return: the accepted dimension orders, in storage order
    """
    return (expected_dimensions(layout) or ()) + (expected_dimensions(layout, includes_time=False) or ())


def _validate_precipitation_input(args: argparse.Namespace) -> _InputContext:
    """
    Validate the precipitation input and derive the input type from it.

    param args: an arguments object of the type returned by
        argparse.ArgumentParser.parse_args()
    return: the validated input, for comparison against the companion inputs
    raise ValueError: if the precipitation input is missing or invalid
    """

    # make sure a precipitation file was specified
    if args.netcdf_precip is None:
        msg = "Missing the required precipitation file"
        _logger.error(msg)
        raise ValueError(msg)

    # make sure a precipitation variable name was specified
    if args.var_name_precip is None:
        msg = "Missing precipitation variable name"
        _logger.error(msg)
        raise ValueError(msg)

    with xr.open_dataset(args.netcdf_precip) as dataset_precip:
        # make sure we have a valid precipitation variable name
        if args.var_name_precip not in dataset_precip.variables:
            msg = (
                f"Invalid precipitation variable name: '{args.var_name_precip}'"
                + f"does not exist in precipitation file '{args.netcdf_precip}'"
            )
            _logger.error(msg)
            raise ValueError(msg)

        # verify that the precipitation variable's dimensions are in the expected order
        dimensions = dataset_precip[args.var_name_precip].dims
        layout = detect_dataset_layout(dimensions, "precipitation")

        # get the values of the precipitation coordinate variables,
        # for comparison against those of the other data variables
        latitudes = None
        longitudes = None
        divisions = None
        if layout == DatasetLayout.GRID:
            latitudes = dataset_precip["lat"].values[:]
            longitudes = dataset_precip["lon"].values[:]
        elif layout == DatasetLayout.DIVISIONS:
            divisions = dataset_precip["division"].values[:]
        times = dataset_precip["time"].values[:]

    return _InputContext(
        input_type=layout,
        dimensions=dimensions,
        times=times,
        latitudes=latitudes,
        longitudes=longitudes,
        divisions=divisions,
    )


def _validate_temperature_input(args: argparse.Namespace) -> _InputContext:
    """
    Validate the temperature input and derive the input type from it.

    Only PET is computed from temperature alone, so this is the input route for
    ``--index pet``; every other index derives its input type from
    precipitation.

    param args: an arguments object of the type returned by
        argparse.ArgumentParser.parse_args()
    return: the validated input
    raise ValueError: if the temperature input is missing or invalid
    """

    # PET requires a temperature file
    if args.netcdf_temp is None:
        msg = "Missing the required temperature file argument"
        _logger.error(msg)
        raise ValueError(msg)

    # don't allow a daily periodicity (yet, this will be
    # possible once we have Hargreaves or a daily Thornthwaite)
    if args.periodicity is not compute.Periodicity.monthly:
        msg = "Invalid periodicity argument for PET: " + f"'{args.periodicity}' -- only 'monthly' is supported"
        _logger.error(msg)
        raise ValueError(msg)

    with xr.open_dataset(args.netcdf_temp) as dataset_temp:
        # make sure we have a valid temperature variable name
        if args.var_name_temp not in dataset_temp.variables:
            msg = (
                f"Invalid temperature variable name: '{args.var_name_temp}'"
                + f" does not exist in temperature file '{args.netcdf_temp}'"
            )
            _logger.error(msg)
            raise ValueError(msg)

        # verify that the temperature variable's dimensions are in the expected order
        dimensions = dataset_temp[args.var_name_temp].dims
        layout = detect_dataset_layout(dimensions, "temperature")

        return _InputContext(
            input_type=layout,
            dimensions=dimensions,
            times=dataset_temp["time"].values[:],
        )


def _validate_matching_input_file(
    context: _InputContext,
    label: str,
    netcdf_file: str,
    var_name: str | None,
) -> None:
    """
    Validate a companion input file against the precipitation input.

    The companion variable must carry the same input type as the precipitation
    variable, and its coordinates and times must match it.

    param context: the validated precipitation input
    param label: the companion variable's label, e.g. "PET" or "temperature"
    param netcdf_file: path of the companion NetCDF file
    param var_name: name of the companion variable within the file
    raise ValueError: if the companion input is invalid or does not match
    """

    expected = expected_dimensions(context.input_type)
    if expected is None:
        msg = "Failed to determine the input type (gridded, timeseries, or US climate division)"
        _logger.error(msg)
        raise ValueError(msg)

    with xr.open_dataset(netcdf_file) as dataset:
        # make sure we have a valid variable name
        if var_name is None:
            msg = f"Missing {label} variable name"
            _logger.error(msg)
            raise ValueError(msg)
        if var_name not in dataset.variables:
            msg = f"Invalid {label} variable name: '{var_name}' does not exist in {label} file '{netcdf_file}'"
            _logger.error(msg)
            raise ValueError(msg)

        # verify that the variable's dimensions are in the expected order
        dimensions = dataset[var_name].dims
        if dimensions not in expected:
            msg = f"Invalid dimensions of the {label} variable: {dimensions}(expected names and order: {list(expected)}"
            _logger.error(msg)
            raise ValueError(msg)

        # verify that the coordinate variables match with those of the precipitation dataset
        if context.input_type == DatasetLayout.GRID:
            assert context.latitudes is not None
            assert context.longitudes is not None
            if not np.allclose(
                context.latitudes,
                dataset["lat"][:],
                atol=utils.get_tolerance(context.latitudes),
            ):
                msg = f"Precipitation and {label} variables contain non-matching latitudes"
                _logger.error(msg)
                raise ValueError(msg)
            if not np.allclose(
                context.longitudes,
                dataset["lon"][:],
                atol=utils.get_tolerance(context.longitudes),
            ):
                msg = f"Precipitation and {label} variables contain non-matching longitudes"
                _logger.error(msg)
                raise ValueError(msg)

        elif context.input_type == DatasetLayout.DIVISIONS:
            assert context.divisions is not None
            if not np.array_equal(context.divisions, dataset["division"][:]):
                msg = f"Precipitation and {label} variables contain non-matching division IDs"
                _logger.error(msg)
                raise ValueError(msg)

        # make sure times match
        if not np.array_equal(context.times, dataset["time"][:]):
            msg = f"Precipitation and {label} variables contain non-matching times"
            _logger.error(msg)
            raise ValueError(msg)


def _validate_pet_or_temperature_input(args: argparse.Namespace, context: _InputContext) -> None:
    """
    Validate the PET input, or the temperature input it is computed from.

    param args: an arguments object of the type returned by
        argparse.ArgumentParser.parse_args()
    param context: the validated precipitation input
    raise ValueError: if neither input is provided, both are provided, or the
        provided input is invalid
    """

    if args.netcdf_temp is None:
        if args.netcdf_pet is None:
            msg = "Missing the required temperature or PET files, neither were provided"
            _logger.error(msg)
            raise ValueError(msg)

        # validate the PET file
        _validate_matching_input_file(context, "PET", args.netcdf_pet, args.var_name_pet)

    elif args.netcdf_pet is not None:
        # we can't have both temperature and PET files specified,
        # no way to determine which to use
        msg = "Both temperature and PET files were specified, only one of these should be provided"
        _logger.error(msg)
        raise ValueError(msg)

    else:
        # temperature-derived PET is Thornthwaite PET, which is defined for
        # monthly means only
        if args.periodicity is not compute.Periodicity.monthly:
            msg = "Invalid periodicity argument for PET: " + f"'{args.periodicity}' -- only 'monthly' is supported"
            _logger.error(msg)
            raise ValueError(msg)

        # validate the temperature file
        _validate_matching_input_file(context, "temperature", args.netcdf_temp, args.var_name_temp)


def _validate_awc_input(args: argparse.Namespace, context: _InputContext) -> None:
    """
    Validate the available water capacity input against the precipitation input.

    param args: an arguments object of the type returned by
        argparse.ArgumentParser.parse_args()
    param context: the validated precipitation input
    raise ValueError: if the AWC input is missing or invalid
    """

    if args.netcdf_awc is None:
        msg = "Missing the required available water capacity file"
        _logger.error(msg)
        raise ValueError(msg)

    # validate the AWC file
    with xr.open_dataset(args.netcdf_awc) as dataset_awc:
        # make sure we have a valid AWC variable name
        if args.var_name_awc is None:
            msg = "Missing the AWC variable name"
            _logger.error(msg)
            raise ValueError(msg)
        if args.var_name_awc not in dataset_awc.variables:
            msg = (
                f"Invalid AWC variable name: '{args.var_name_awc}' " + f"does not exist in AWC file '{args.netcdf_awc}'"
            )
            _logger.error(msg)
            raise ValueError(msg)

        # verify that the AWC variable's dimensions are in the expected order
        dimensions = dataset_awc[args.var_name_awc].dims
        expected = expected_dimensions(context.input_type, includes_time=False)
        if expected is None:
            msg = "Failed to determine the input type (gridded or US climate division)"
            _logger.error(msg)
            raise ValueError(msg)

        if dimensions not in expected:
            msg = (
                f"Invalid dimensions of the AWC variable: {dimensions} "
                + f"(expected names and order: {list(expected)})"
            )
            _logger.error(msg)
            raise ValueError(msg)

        # verify that the coordinate variables match with those of the precipitation dataset
        if context.input_type == DatasetLayout.GRID:
            assert context.latitudes is not None
            assert context.longitudes is not None
            if not np.allclose(
                context.latitudes,
                dataset_awc["lat"][:],
                atol=utils.get_tolerance(context.latitudes),
            ):
                msg = "Precipitation and AWC variables contain non-matching latitudes"
                _logger.error(msg)
                raise ValueError(msg)
            if not np.allclose(
                context.longitudes,
                dataset_awc["lon"][:],
                atol=utils.get_tolerance(context.longitudes),
            ):
                msg = "Precipitation and AWC variables contain non-matching longitudes"
                _logger.error(msg)
                raise ValueError(msg)

        elif context.input_type == DatasetLayout.DIVISIONS:
            assert context.divisions is not None
            if not np.array_equal(context.divisions, dataset_awc["division"][:]):
                msg = "Precipitation and AWC variables contain non-matching division IDs"
                _logger.error(msg)
                raise ValueError(msg)


def _validate_scales(args: argparse.Namespace) -> None:
    """
    Validate the time scales of a scaled index.

    param args: an arguments object of the type returned by
        argparse.ArgumentParser.parse_args()
    raise ValueError: if no scales were provided or one is negative
    """

    if not args.scales:
        msg = (
            "Scaled indices (SPI, SPEI, and/or PNP) specified without "
            + "including one or more time scales (missing --scales argument)"
        )
        _logger.error(msg)
        raise ValueError(msg)

    if any(n < 0 for n in args.scales):
        msg = "One or more negative scale specified within --scales argument"
        _logger.error(msg)
        raise ValueError(msg)


def _validate_args(args: argparse.Namespace) -> DatasetLayout:
    """
    Validate the processing settings to confirm that proper argument
    combinations have been provided.

    Each registration behind the ``--index`` value -- one handler per computable
    index, and one per member of an aggregate such as ``all`` -- declares its own
    input requirements, so the checks here are driven by those declarations
    rather than by the index name.

    param args: an arguments object of the type returned by
        argparse.ArgumentParser.parse_args()
    raise ValueError: if one or more of the command line arguments is invalid
    """

    handlers = _handlers_for_index(args.index)

    for handler in handlers:
        if handler.validate_arguments is not None:
            handler.validate_arguments(args)

    # the input that determines the input type, and the shape companions must match
    if any(handler.requires_precip for handler in handlers):
        context = _validate_precipitation_input(args)
    else:
        context = _validate_temperature_input(args)

    # index-specific checks that need the precipitation input's shape
    for handler in handlers:
        if handler.validate_inputs is not None:
            handler.validate_inputs(args, context)

    if any(handler.requires_pet_or_temp for handler in handlers):
        _validate_pet_or_temperature_input(args, context)

    if any(handler.requires_awc for handler in handlers):
        _validate_awc_input(args, context)

    if any(handler.requires_scales for handler in handlers):
        _validate_scales(args)

    return context.input_type


# the increment each periodicity's log messages are expressed in
_SCALE_INCREMENTS: dict[compute.Periodicity, str] = {
    compute.Periodicity.daily: "day",
    compute.Periodicity.monthly: "month",
}


def _get_scale_increment(periodicity: compute.Periodicity) -> str:
    return _SCALE_INCREMENTS[periodicity]


def _log_status(request: _IndexRequest) -> None:
    # get the scale increment for use in later log messages
    if request.scale is None:
        _logger.info(f"Computing {request.index.upper()}")

    elif request.distribution is None:
        _logger.info(f"Computing {request.scale}-{_get_scale_increment(request.periodicity)} {request.index.upper()}")

    else:
        _logger.info(
            f"Computing {request.scale}-{_get_scale_increment(request.periodicity)} {request.index.upper()}/{request.distribution.value.capitalize()}"
        )


def _drop_data_into_shared_arrays_grid(
    dataset: xr.Dataset,
    var_names: list[str],
    accepted_dimensions: tuple[tuple[Hashable, ...], ...],
    periodicity: compute.Periodicity,
    data_start_year: int,
) -> tuple[int, ...]:
    output_shape = None

    # get the data arrays we'll use later in the index computations
    global _global_shared_arrays
    for var_name in var_names:
        # confirm that the dimensions of the data array are valid
        dims = dataset[var_name].dims
        if dims not in accepted_dimensions:
            message = f"Invalid dimensions for variable '{var_name}': {dims}"
            _logger.error(message)
            raise ValueError(message)

        # convert daily values into 366-day years
        if periodicity == compute.Periodicity.daily:
            initial_year = int(str(dataset["time"][0].data)[0:4])
            final_year = int(str(dataset["time"][-1].data)[0:4])
            total_years = final_year - initial_year + 1
            var_values = np.apply_along_axis(
                utils.transform_to_366day,
                len(dims) - 1,
                dataset[var_name].values,
                data_start_year,
                total_years,
            )

        else:  # assumed to be monthly
            var_values = dataset[var_name].values

        output_shape = var_values.shape

        # create a shared memory array, wrap it as a numpy array and
        # copy the data (values) from this variable's DataArray
        shared_array = multiprocessing.Array("d", int(np.prod(var_values.shape)))
        shared_array_np = np.frombuffer(shared_array.get_obj()).reshape(var_values.shape)  # type: ignore[call-overload]
        np.copyto(shared_array_np, var_values)

        # add to the dictionary of arrays
        _global_shared_arrays[var_name] = {
            _KEY_ARRAY: shared_array,
            _KEY_SHAPE: var_values.shape,
        }

        # drop the variable from the dataset (we're assuming this frees the memory)
        dataset = dataset.drop_vars(names=[var_name])

    assert output_shape is not None, "No variables processed; output shape is unknown"
    return output_shape


def _drop_data_into_shared_arrays_divisions(
    dataset: xr.Dataset,
    var_names: list[str],
    accepted_dimensions: tuple[tuple[Hashable, ...], ...],
) -> tuple[int, ...]:
    """
    Drop data into shared arrays for use in the index computations.

    :param dataset:
    :param var_names:
    :param accepted_dimensions: the dimension orders a variable in this
        dataset may use
    :return:
    """
    output_shape = None

    # get the data arrays we'll use later in the index computations
    global _global_shared_arrays
    for var_name in var_names:
        # confirm that the dimensions of the data array are valid
        dims = dataset[var_name].dims
        if dims not in accepted_dimensions:
            message = f"Invalid dimensions for variable '{var_name}': {dims}"
            _logger.error(message)
            raise ValueError(message)

        # create a shared memory array, wrap it as a numpy array and
        # copy the data (values) from this variable's DataArray
        shared_array = multiprocessing.Array("d", int(np.prod(dataset[var_name].shape)))
        shared_array_np = np.frombuffer(shared_array.get_obj()).reshape(dataset[var_name].shape)  # type: ignore[call-overload]
        np.copyto(shared_array_np, dataset[var_name].values)

        # add to the dictionary of arrays
        _global_shared_arrays[var_name] = {
            _KEY_ARRAY: shared_array,
            _KEY_SHAPE: dataset[var_name].shape,
        }

        # we know we'll want the output for divisions to be 2-D
        if len(dataset[var_name].shape) == 2:
            output_shape = dataset[var_name].shape

        # drop the variable from the dataset (we're assuming this frees the memory)
        dataset = dataset.drop_vars(names=[var_name])

    assert output_shape is not None, "No variables processed; output shape is unknown"
    return output_shape


# the chunking xr.open_mfdataset() uses per input type, collapsing the whole
# dimension so index kernels always see complete series/grids/divisions
_CHUNKS_BY_INPUT_TYPE: dict[DatasetLayout, dict[str, int]] = {
    DatasetLayout.GRID: {"lat": -1, "lon": -1},
    DatasetLayout.DIVISIONS: {"division": -1},
    DatasetLayout.TIMESERIES: {"time": -1},
}


def _input_chunksizes(dataset: xr.Dataset) -> tuple[tuple[int, ...], tuple[Any, ...]]:
    """
    Find the first input variable's chunk sizes, for copying onto the output.

    Note that the netcdf spec doesn't require that all data variables have the
    same chunk sizes.

    param dataset: the opened inputs
    return: the chunk sizes found and the dimensions they correspond to, or a
        pair of empty tuples if no variable is chunked
    """
    for da in dataset.data_vars.values():
        if not da.encoding.get("contiguous", True):
            # tuple of chunksizes, respectively by dimension
            chunksizes = da.encoding.get("chunksizes", ())
            if chunksizes:
                return chunksizes, da.dims
    return (), ()


def _trim_to_input_variables(request: _IndexRequest, dataset: xr.Dataset) -> tuple[xr.Dataset, list[str]]:
    """
    Keep only the data variables an index's request actually uses.

    param request: the index request being computed
    param dataset: the opened inputs
    return: the trimmed dataset and the kept variable names
    """
    input_var_names = [name for name in (request.var_name_precip, request.var_name_temp, request.var_name_pet) if name]
    # keep the latitude variable if we're dealing with divisions
    if request.input_type == DatasetLayout.DIVISIONS:
        input_var_names.append("lat")
    for var in dataset.data_vars:
        if var not in input_var_names:
            dataset = dataset.drop_vars(names=[var])
    return dataset, input_var_names


def _output_dims(request: _IndexRequest, dataset: xr.Dataset) -> tuple[Hashable, ...]:
    """
    The output variable's dimensions, matching whichever input carries them.

    The shape of output variables is assumed to match that of the input, so
    use either the precipitation or temperature variable's dimensions.

    param request: the index request being computed
    param dataset: the opened, trimmed inputs
    return: the output variable's dimensions
    raise ValueError: if neither a precipitation nor temperature variable name
        was specified
    """
    if request.var_name_precip is not None:
        return dataset[request.var_name_precip].dims
    if request.var_name_temp is not None:
        return dataset[request.var_name_temp].dims
    raise ValueError(
        "Unable to determine output dimensions, no precipitation or temperature variable name was specified."
    )


def _reordered_chunksizes(
    chunksizes: tuple[int, ...],
    source_dims: tuple[Any, ...],
    output_dims: tuple[Hashable, ...],
) -> tuple[int, ...]:
    """
    Reorder a copied chunk-size tuple from the source variable's dimension
    order to the output variable's.

    The copied chunksizes follow the source variable's dimension order, which
    can differ from the output variable's -- reorder by dimension name so that
    each chunk length corresponds to the correct output dimension.

    param chunksizes: chunk sizes found on the source variable
    param source_dims: the source variable's dimensions, in the same order
    param output_dims: the output variable's dimensions
    return: chunksizes reordered to match output_dims, or an empty tuple if
        the source and output dimensions don't match
    """
    if not chunksizes or source_dims == tuple(output_dims):
        return chunksizes
    chunksizes_by_dim = dict(zip(source_dims, chunksizes, strict=False))
    if set(chunksizes_by_dim) == set(output_dims):
        return tuple(chunksizes_by_dim[dim] for dim in output_dims)
    _logger.warning(
        "Ignoring '--chunksizes input': chunked variable dimensions %s do not match output dimensions %s",
        source_dims,
        output_dims,
    )
    return ()


def _normalize_precipitation_units(dataset: xr.Dataset, var_name: str | None) -> None:
    """
    Convert a precipitation variable's values to millimeters, in place.

    param dataset: the dataset holding the variable
    param var_name: name of the precipitation variable, or None when the
        index takes no precipitation input
    raise ValueError: if the variable's units are not millimeters, a daily
        rate, or inches
    """
    if var_name is None:
        return
    precip_unit = dataset[var_name].units.lower()
    if precip_unit in ("mm", "millimeters", "millimeter", "mm/dy"):
        return
    if precip_unit in ("inches", "inch"):
        # inches to mm conversion (1 inch == 25.4 mm)
        dataset[var_name].values *= 25.4
    else:
        raise ValueError(f"Unsupported precipitation units: {precip_unit}")


def _normalize_temperature_units(dataset: xr.Dataset, var_name: str | None) -> None:
    """
    Convert a temperature variable's values to degrees Celsius, in place.

    param dataset: the dataset holding the variable
    param var_name: name of the temperature variable, or None when the index
        takes no temperature input
    raise ValueError: if the variable's units are not Celsius, Fahrenheit, or
        Kelvin
    """
    if var_name is None:
        return
    temp_unit = dataset[var_name].units.lower()
    if temp_unit in ("degree_celsius", "degrees_celsius", "celsius", "c"):
        return
    if temp_unit in ("f", "fahrenheit", "degree_fahrenheit", "degrees_fahrenheit"):
        dataset[var_name].values = scipy.constants.convert_temperature(dataset[var_name].values, "f", "c")
    elif temp_unit in ("k", "kelvin"):
        dataset[var_name].values = scipy.constants.convert_temperature(dataset[var_name].values, "k", "c")
    else:
        raise ValueError(f"Unsupported temperature units: {temp_unit}")


def _normalize_pet_units(dataset: xr.Dataset, var_name: str | None) -> None:
    """
    Convert a PET variable's values to millimeters, in place.

    param dataset: the dataset holding the variable
    param var_name: name of the PET variable, or None when the index takes no
        PET input
    raise ValueError: if the variable's units are not millimeters or inches
    """
    if var_name is None:
        return
    pet_unit = dataset[var_name].units.lower()
    if pet_unit in ("mm", "millimeters", "millimeter"):
        return
    if pet_unit in ("inches", "inch"):
        # inches to mm conversion (1 inch == 25.4 mm)
        dataset[var_name].values *= 25.4
    else:
        raise ValueError(f"Unsupported PET units: {dataset[var_name].units}")


def _compute_write_index(request: _IndexRequest) -> tuple[str, str] | None:
    """
    Computes a climate index and writes the result into a corresponding NetCDF.

    param request: the index, inputs, and output settings to compute with
    return: the name of the output file and of the variable written into it, or
        None for an index that writes more than one output file
    """

    handler = _registry_for(request.index)

    _log_status(request)

    # open the NetCDF files as an xarray DataSet object
    files = [path for path in (request.netcdf_precip, request.netcdf_temp, request.netcdf_pet) if path is not None]
    chunks = _CHUNKS_BY_INPUT_TYPE.get(request.input_type)
    if chunks is None:
        raise ValueError(f"Unsupported input type: {request.input_type}")

    # Since multiple variables can be in the same file, de-duplicate the filelist.
    dataset = xr.open_mfdataset(list(set(files)), chunks=chunks)

    output_chunksizes: tuple[int, ...] = ()
    chunksizes_dims: tuple[Any, ...] = ()
    if request.chunksizes == "input":
        output_chunksizes, chunksizes_dims = _input_chunksizes(dataset)

    # trim out all data variables from the dataset except the ones we'll need
    dataset, input_var_names = _trim_to_input_variables(request, dataset)

    # get the initial year of the data
    request.data_start_year = int(str(dataset["time"].values[0])[0:4])

    output_dims = _output_dims(request, dataset)
    output_chunksizes = _reordered_chunksizes(output_chunksizes, chunksizes_dims, output_dims)

    # convert data into the appropriate units, if necessary
    # precipitation and PET should be in millimeters, temperature in Celsius
    _normalize_precipitation_units(dataset, request.var_name_precip)
    _normalize_temperature_units(dataset, request.var_name_temp)
    _normalize_pet_units(dataset, request.var_name_pet)

    # the Palmer routines take inches, whereas the conversions above normalize
    # precipitation and PET to millimeters for every other index; this runs
    # before the inputs are copied into shared memory below, so an invalid AWC
    # label is rejected without paying for those full-array copies
    prepared = handler.prepare_inputs(request, dataset) if handler.prepare_inputs is not None else None

    accepted_dimensions = _accepted_dimensions(request.input_type)
    if request.input_type == DatasetLayout.DIVISIONS:
        output_shape = _drop_data_into_shared_arrays_divisions(dataset, input_var_names, accepted_dimensions)
    else:
        output_shape = _drop_data_into_shared_arrays_grid(
            dataset,
            input_var_names,
            accepted_dimensions,
            request.periodicity,
            request.data_start_year,
        )

    output_encodings = {"chunksizes": output_chunksizes} if output_chunksizes else None
    # a chunksizes encoding is only honored by an HDF5-backed engine, and the
    # supported xarray versions still default to scipy when netCDF4 is absent
    output_engine: Literal["h5netcdf"] | None = "h5netcdf" if output_chunksizes else None

    context = _ComputeContext(
        request=request,
        dataset=dataset,
        output_dims=output_dims,
        output_shape=output_shape,
        output_encodings=output_encodings,
        output_engine=output_engine,
        prepared=prepared,
        arguments=handler.build_arguments(request) if handler.build_arguments is not None else {},
    )

    assert handler.compute is not None, f"the '{handler.index}' index does not compute from shared arrays"
    assert handler.write is not None, f"the '{handler.index}' index does not compute from shared arrays"

    handler.compute(context)
    return handler.write(context)


def _pet(temperatures: np.ndarray, latitude: float, parameters: dict[str, Any]) -> np.ndarray:
    return indices.pet(
        temperature_celsius=temperatures,
        latitude_degrees=latitude,
        data_start_year=parameters["data_start_year"],
    )


def _spi(precips: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return indices.spi(
        values=precips,
        scale=parameters["scale"],
        distribution=parameters["distribution"],
        data_start_year=parameters["data_start_year"],
        calibration_year_initial=parameters["calibration_year_initial"],
        calibration_year_final=parameters["calibration_year_final"],
        periodicity=parameters["periodicity"],
    )


def _spei(precips: np.ndarray, pet_mm: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return indices.spei(
        precips_mm=precips,
        pet_mm=pet_mm,
        scale=parameters["scale"],
        distribution=parameters["distribution"],
        data_start_year=parameters["data_start_year"],
        calibration_year_initial=parameters["calibration_year_initial"],
        calibration_year_final=parameters["calibration_year_final"],
        periodicity=parameters["periodicity"],
    )


def _pnp(precips: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return indices.percentage_of_normal(
        precips,
        scale=parameters["scale"],
        data_start_year=parameters["data_start_year"],
        calibration_start_year=parameters["calibration_start_year"],
        calibration_end_year=parameters["calibration_end_year"],
        periodicity=parameters["periodicity"],
    )


def _palmers(
    precips: np.ndarray,
    pet: np.ndarray,
    awc: float | np.ndarray,
    parameters: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # pdsi() vectorizes a whole grid chunk (ADR-0009); scpdsi() stays
    # per-location (ADR-0011) and is looped inside that chunk below. The grid
    # worker passes its block with spatial_time_major=True; the divisions worker's
    # per-location call leaves it unset and gets the legacy 1-D reading.
    spatial_time_major = parameters.get("spatial_time_major", False)
    computed_pdsi, computed_phdi, computed_pmdi, computed_zindex, _fitting_params = palmer.pdsi(
        precips,
        pet,
        awc,
        parameters["data_start_year"],
        parameters["calibration_start_year"],
        parameters["calibration_end_year"],
        spatial_time_major=spatial_time_major,
    )
    computed_scpdsi = _palmers_scpdsi(precips, pet, awc, parameters, spatial_time_major)
    return computed_pdsi, computed_phdi, computed_pmdi, computed_zindex, computed_scpdsi


def _palmers_scpdsi(
    precips: np.ndarray,
    pet: np.ndarray,
    awc: float | np.ndarray,
    parameters: dict[str, Any],
    spatial_time_major: bool,
) -> np.ndarray:
    """
    Compute the self-calibrating PDSI for every location in a Palmer block.

    ``scpdsi()`` rejects a spatial block and stays per-location (ADR-0011), so a
    grid chunk loops over its cells -- the CLI's multiprocessing already
    parallelizes chunks -- while a divisions call computes its one location
    directly. A location whose calibration cannot fit usable duration factors
    (``ConvergenceError``/``InsufficientDataError``) is left missing instead of
    aborting the run, matching the all-missing shortcut in ``palmer.pdsi()``.

    :param precips: precipitation block, time-major when spatial_time_major
    :param pet: PET block matching precips
    :param awc: available water capacity, scalar or per-cell
    :param parameters: the Palmer arguments built by the registration
    :param spatial_time_major: whether a 3+-D block is read as (time, *cells)
    :return: the scPDSI series, shaped like the input block
    """
    calibration_years = (
        parameters["data_start_year"],
        parameters["calibration_start_year"],
        parameters["calibration_end_year"],
    )
    if not spatial_time_major:
        try:
            return palmer.scpdsi(precips, pet, cast(float, awc), *calibration_years)[0]
        except (ConvergenceError, InsufficientDataError):
            return np.full(np.shape(precips), np.nan, dtype=float)

    cell_series = np.asarray(precips).reshape(precips.shape[0], -1)
    pet_series = np.asarray(pet).reshape(pet.shape[0], -1)
    # ADR-0009 lets a block call broadcast a scalar AWC across its cells
    awc_values = np.broadcast_to(np.ravel(np.asarray(awc)), (cell_series.shape[1],))
    computed_scpdsi = np.full(cell_series.shape, np.nan, dtype=float)
    for cell in range(cell_series.shape[1]):
        try:
            computed_scpdsi[:, cell] = palmer.scpdsi(
                cell_series[:, cell],
                pet_series[:, cell],
                float(awc_values[cell]),
                *calibration_years,
            )[0]
        except (ConvergenceError, InsufficientDataError):
            continue  # leave this location missing rather than abort the run
    return computed_scpdsi.reshape(np.asarray(precips).shape)


def _init_worker(shared_arrays_dict: dict[str, Any]) -> None:
    global _global_shared_arrays
    _global_shared_arrays = shared_arrays_dict


def _parallel_process(request: _IndexRequest, arguments: dict[str, Any]) -> None:
    """
    Apply the requested index's kernel across the shared-memory input arrays.

    The work is split along the first axis — latitude, or division — with one
    worker process per sub-array.

    :param request: the index request being computed
    :param arguments: the kernel's arguments, as the index's registration builds them
    """
    handler = _registry_for(request.index)
    assert handler.kernel is not None
    assert handler.worker is not None
    assert handler.input_array_keys is not None

    # find the start index of each sub-array we'll split out per worker process,
    # assuming the shape of the output array is the same as all input arrays
    shape = _global_shared_arrays[handler.output_keys[0]][_KEY_SHAPE]
    # if there are fewer chunks than the available number of processes
    # then only create the necessary number of tasks
    required_processes = min(shape[0], _NUMBER_OF_WORKER_PROCESSES)
    d, m = divmod(shape[0], required_processes)
    split_indices = list(range(0, ((d + 1) * (m + 1)), (d + 1)))
    if d != 0:
        split_indices += list(range(split_indices[-1] + d, shape[0], d))

    # build a list of parameters for each application of the kernel to an array chunk
    chunk_params = []
    for i in range(required_processes):
        chunk_params.append(
            {
                "func1d": handler.kernel,
                "input_var_names": handler.input_array_keys(request),
                "coordinate_input": handler.coordinate_input,
                "output_var_names": handler.output_keys,
                "sub_array_start": split_indices[i],
                "sub_array_end": split_indices[i + 1] if i < (required_processes - 1) else None,
                "input_type": request.input_type,
                "args": arguments,
            }
        )

    # instantiate a process pool
    with multiprocessing.Pool(
        processes=_NUMBER_OF_WORKER_PROCESSES,
        initializer=_init_worker,
        initargs=(_global_shared_arrays,),
    ) as pool:
        pool.map(handler.worker, chunk_params)


def _apply_along_axis(params: dict[str, Any]) -> None:
    """
    Like numpy.apply_along_axis(), but with arguments in a dict instead.
    Applicable for applying a function across subarrays of a single input array.

    This function is useful with multiprocessing.Pool().map(): (1) map() only
    handles functions that take a single argument, and (2) this function can
    generally be imported from a module, as required by map().

    :param dict params: dictionary of parameters including a function name,
        "func1d", start and stop indices for specifying the subarray to which
        the function should be applied, "sub_array_start" and "sub_array_end",
        a dictionary of arguments to be passed to the function, "args", the
        keys of the input and output shared arrays, "input_var_names" and
        "output_var_names", and the input type, "input_type".
    """
    func1d = params["func1d"]
    start_index = params["sub_array_start"]
    end_index = params["sub_array_end"]
    input_var_name = params["input_var_names"][0]
    output_var_name = params["output_var_names"][0]
    shape = _global_shared_arrays[input_var_name][_KEY_SHAPE]

    sub_array = _shared_array(input_var_name, shape)[start_index:end_index]
    axis_index = _TIME_AXIS_INDEX[params["input_type"]]
    computed_array = np.apply_along_axis(func1d, axis=axis_index, arr=sub_array, parameters=params["args"])

    np.copyto(_shared_array(output_var_name, shape)[start_index:end_index], computed_array)


def _apply_along_axis_double(
    params: dict[str, Any],
) -> None:
    """
    Like numpy.apply_along_axis(), but with arguments in a dict instead.
    Applicable for applying a function across subarrays of two input arrays.

    This function is useful with multiprocessing.Pool().map(): (1) map() only
    handles functions that take a single argument, and (2) this function can
    generally be imported from a module, as required by map().

    :param dict params: dictionary of parameters including a function name,
        "func1d", start and stop indices for specifying the subarray to which
        the function should be applied, "sub_array_start" and "sub_array_end",
        a dictionary of arguments to be passed to the function, "args", the keys
        of the two input arrays and of the output array, "input_var_names" and
        "output_var_names", the input type, "input_type", and whether the second
        input is a coordinate fixed per row rather than a per-cell value,
        "coordinate_input".
    :return: None
    """

    func1d = params["func1d"]
    start_index = params["sub_array_start"]
    end_index = params["sub_array_end"]
    first_array_key, second_array_key = params["input_var_names"]
    output_var_name = params["output_var_names"][0]
    coordinate_input = params["coordinate_input"]

    shape = _global_shared_arrays[output_var_name][_KEY_SHAPE]
    # a coordinate input has one value per row rather than per cell
    second_shape = (shape[0],) if coordinate_input else shape
    sub_array_1 = _shared_array(first_array_key, shape)[start_index:end_index]
    sub_array_2 = _shared_array(second_array_key, second_shape)[start_index:end_index]

    # get the output shared memory array, convert to numpy, and get the subarray slice
    computed_array = _shared_array(output_var_name, shape)[start_index:end_index]

    for i, (x, y) in enumerate(zip(sub_array_1, sub_array_2, strict=False)):
        if params["input_type"] == DatasetLayout.GRID:
            for j in range(x.shape[0]):
                second_value = y if coordinate_input else y[j]
                computed_array[i, j] = func1d(x[j], second_value, parameters=params["args"])
        elif params["input_type"] == DatasetLayout.DIVISIONS:
            computed_array[i] = func1d(x, y, parameters=params["args"])
        else:
            raise ValueError(f"Unsupported input type: '{params['input_type']}'")


def _apply_along_axis_palmers(params: dict[str, Any]) -> None:
    """
    Applies the Palmer computation across subarrays of the Palmer-specific
    input (shared-memory) arrays.

    A grid chunk is computed in one vectorized call over the whole
    (lat_chunk, lon, time) block through the supplied ``func1d``, which receives
    the block with a private ``spatial_time_major=True`` in its parameters, so the
    standard Palmer indices are read per ADR-0009 rather than
    computed per grid cell while the kernel loops scPDSI per location (ADR-0011);
    multiprocessing still parallelizes across chunks (ADR-0002). A divisions chunk
    has no cell-adjacency structure to batch, so it stays on the per-location loop.

    This function is useful with multiprocessing.Pool().map(): (1) map() only
    handles functions that take a single argument, and (2) this function can
    generally be imported from a module, as required by map().

    :param dict params: dictionary of parameters including a function name,
        "func1d", start and stop indices for specifying the subarray to which
        the function should be applied, "sub_array_start" and "sub_array_end",
        a dictionary of arguments to be passed to the function, "args", the keys
        of the precipitation, PET, and AWC input arrays, "input_var_names", and
        the keys of the PDSI, PHDI, PMDI, Z-Index, and scPDSI output arrays,
        "output_var_names".
    """
    func1d = params["func1d"]
    start_index = params["sub_array_start"]
    end_index = params["sub_array_end"]
    precip_array_key, pet_array_key, awc_array_key = params["input_var_names"]
    output_keys = params["output_var_names"]

    shape = _global_shared_arrays[output_keys[0]][_KEY_SHAPE]
    sub_array_precip = _shared_array(precip_array_key, shape)[start_index:end_index]
    sub_array_pet = _shared_array(pet_array_key, shape)[start_index:end_index]
    # available water capacity is fixed per location, without a time dimension
    awc_shape: tuple[Any, ...]
    if params["input_type"] == DatasetLayout.GRID:
        awc_shape = (shape[0], shape[1])
    else:  # divisions
        awc_shape = (shape[0],)
    sub_array_awc = _shared_array(awc_array_key, awc_shape)[start_index:end_index]

    args = params["args"]

    # get the output shared memory arrays, convert to numpy, and get the subarray slices
    pdsi = _shared_array(output_keys[0], shape)[start_index:end_index]
    phdi = _shared_array(output_keys[1], shape)[start_index:end_index]
    pmdi = _shared_array(output_keys[2], shape)[start_index:end_index]
    zindex = _shared_array(output_keys[3], shape)[start_index:end_index]
    scpdsi = _shared_array(output_keys[4], shape)[start_index:end_index]

    if params["input_type"] == DatasetLayout.GRID:
        # sub_array_precip/pet are (lat_chunk, lon, time); pdsi() wants a
        # time-major (time, *cells) block
        precip_block = np.moveaxis(sub_array_precip, -1, 0)
        pet_block = np.moveaxis(sub_array_pet, -1, 0)
        block_args = {**args, "spatial_time_major": True}
        block_pdsi, block_phdi, block_pmdi, block_zindex, block_scpdsi = func1d(
            precip_block,
            pet_block,
            sub_array_awc,
            parameters=block_args,
        )
        np.copyto(pdsi, np.moveaxis(block_pdsi, 0, -1))
        np.copyto(phdi, np.moveaxis(block_phdi, 0, -1))
        np.copyto(pmdi, np.moveaxis(block_pmdi, 0, -1))
        np.copyto(zindex, np.moveaxis(block_zindex, 0, -1))
        np.copyto(scpdsi, np.moveaxis(block_scpdsi, 0, -1))
    else:  # divisions
        for i, (precip, pet, awc) in enumerate(zip(sub_array_precip, sub_array_pet, sub_array_awc, strict=False)):
            pdsi[i], phdi[i], pmdi[i], zindex[i], scpdsi[i] = func1d(precip, pet, awc, parameters=args)


@dataclass(frozen=True)
class _IndexRegistration:
    """
    Everything the CLI needs in order to compute one index.

    A registration collects what used to be re-derived from the index name at
    each use: the input files and arguments it requires, the arguments its
    kernel takes, the output variable's name and attributes, the shared-memory
    arrays it reads and writes, the worker that applies the kernel, and -- for
    the top-level pipeline -- how the index is run. An index that computes
    through xarray instead of the shared-memory route registers only ``run``.

    Fields left ``None`` are ones the index in question does not use.
    """

    index: str
    run: Callable[[argparse.Namespace, DatasetLayout], None]
    # names of the request's input fields this index reads, declared so that
    # from_arguments() copies only the inputs the index actually consumes
    input_paths: tuple[str, ...] = ()
    requires_precip: bool = False
    requires_pet_or_temp: bool = False
    requires_awc: bool = False
    requires_scales: bool = False
    validate_arguments: Callable[[argparse.Namespace], None] | None = None
    validate_inputs: Callable[[argparse.Namespace, _InputContext], None] | None = None
    build_arguments: Callable[[_IndexRequest], dict[str, Any]] | None = None
    variable_attributes: Callable[[_IndexRequest], tuple[str, dict[str, Any]]] | None = None
    prepare_inputs: Callable[[_IndexRequest, xr.Dataset], xr.Dataset] | None = None
    prepare_arrays: Callable[[_IndexRequest, xr.Dataset], None] | None = None
    kernel: Callable[..., Any] | None = None
    input_array_keys: Callable[[_IndexRequest], tuple[str, ...]] | None = None
    output_keys: tuple[str, ...] = (_KEY_RESULT,)
    coordinate_input: bool = False
    worker: Callable[[dict[str, Any]], None] | None = None
    compute: Callable[[_ComputeContext], None] | None = None
    write: Callable[[_ComputeContext], tuple[str, str] | None] | None = None


# the five outputs the Palmer routines produce, in the order they are written
_PALMER_OUTPUTS = (
    (_KEY_RESULT_PDSI, "pdsi", "Palmer Drought Severity Index"),
    (_KEY_RESULT_PHDI, "phdi", "Palmer Hydrological Drought Index"),
    (_KEY_RESULT_PMDI, "pmdi", "Palmer Modified Drought Index"),
    (_KEY_RESULT_ZINDEX, "zindex", "Palmer Z-Index"),
    (_KEY_RESULT_SCPDSI, "scpdsi", "Self-calibrated Palmer Drought Severity Index"),
)

# the axis each input type's time dimension lies along
_TIME_AXIS_INDEX: dict[DatasetLayout, int] = {
    DatasetLayout.GRID: 2,
    DatasetLayout.DIVISIONS: 1,
    DatasetLayout.TIMESERIES: 0,
}

# the registrations run only after _validate_args() has filled the request, so
# these assertions are invariant checks rather than input validation; keep each
# message identical across the registrations that assert it
_UNVALIDATED_PRECIP = "the precipitation variable name was not validated"
_UNVALIDATED_PET = "the PET variable name was not validated"
_UNVALIDATED_TEMP = "the temperature variable name was not validated"
_UNVALIDATED_AWC = "the AWC variable name was not validated"
_UNVALIDATED_DISTRIBUTION = "the distribution was not validated"
_UNVALIDATED_SCALE = "the scale was not validated"


def _shared_array(name: str, shape: tuple[int, ...]) -> np.ndarray:
    """
    Return a shared-memory array's values as a numpy array of the given shape.

    :param str name: the shared arrays dictionary key
    :param tuple shape: the shape the array's values are mapped onto
    :return: the shared array's values
    """
    shared = _global_shared_arrays[name][_KEY_ARRAY]
    return np.frombuffer(shared.get_obj()).reshape(shape)


def _allocate_shared_array(name: str, shape: tuple[int, ...]) -> None:
    """
    Create a shared-memory array under the given key, for worker processes to fill.

    :param str name: the shared arrays dictionary key
    :param tuple shape: the shape of the values the array will hold
    """
    _global_shared_arrays[name] = {
        _KEY_ARRAY: multiprocessing.Array("d", int(np.prod(shape))),
        _KEY_SHAPE: shape,
    }


def _precipitation_array_key(request: _IndexRequest) -> tuple[str, ...]:
    assert request.var_name_precip is not None, _UNVALIDATED_PRECIP
    return (request.var_name_precip,)


def _precipitation_and_pet_array_keys(request: _IndexRequest) -> tuple[str, ...]:
    assert request.var_name_precip is not None, _UNVALIDATED_PRECIP
    assert request.var_name_pet is not None, _UNVALIDATED_PET
    return (request.var_name_precip, request.var_name_pet)


def _temperature_and_latitude_array_keys(request: _IndexRequest) -> tuple[str, ...]:
    assert request.var_name_temp is not None, _UNVALIDATED_TEMP
    return (request.var_name_temp, _KEY_LAT)


def _palmer_array_keys(request: _IndexRequest) -> tuple[str, ...]:
    assert request.var_name_precip is not None, _UNVALIDATED_PRECIP
    assert request.var_name_pet is not None, _UNVALIDATED_PET
    assert request.var_name_awc is not None, _UNVALIDATED_AWC
    return (request.var_name_precip, request.var_name_pet, request.var_name_awc)


def _spi_arguments(request: _IndexRequest) -> dict[str, Any]:
    return {
        "data_start_year": request.data_start_year,
        "scale": request.scale,
        "distribution": request.distribution,
        "calibration_year_initial": request.calibration_start_year,
        "calibration_year_final": request.calibration_end_year,
        "periodicity": request.periodicity,
    }


def _pnp_arguments(request: _IndexRequest) -> dict[str, Any]:
    return {
        "data_start_year": request.data_start_year,
        "scale": request.scale,
        "calibration_start_year": request.calibration_start_year,
        "calibration_end_year": request.calibration_end_year,
        "periodicity": request.periodicity,
    }


def _palmer_arguments(request: _IndexRequest) -> dict[str, Any]:
    return {
        "data_start_year": request.data_start_year,
        "calibration_start_year": request.calibration_start_year,
        "calibration_end_year": request.calibration_end_year,
    }


def _pet_arguments(request: _IndexRequest) -> dict[str, Any]:
    return {"data_start_year": request.data_start_year}


def _spi_variable_attributes(request: _IndexRequest) -> tuple[str, dict[str, Any]]:
    assert request.distribution is not None, _UNVALIDATED_DISTRIBUTION
    assert request.scale is not None, _UNVALIDATED_SCALE
    long_name = (
        f"Standardized Precipitation Index ({request.distribution.value.capitalize()} distribution), "
        + f"{request.scale}-{_get_scale_increment(request.periodicity)}"
    )
    attrs = {"long_name": long_name, "valid_min": -3.09, "valid_max": 3.09}
    var_name = "spi_" + request.distribution.value + "_" + str(request.scale).zfill(2)

    return var_name, attrs


def _spei_variable_attributes(request: _IndexRequest) -> tuple[str, dict[str, Any]]:
    assert request.distribution is not None, _UNVALIDATED_DISTRIBUTION
    assert request.scale is not None, _UNVALIDATED_SCALE
    long_name = (
        f"Standardized Precipitation Evapotranspiration Index ({request.distribution.value.capitalize()} distribution), "
        + f"{request.scale}-{_get_scale_increment(request.periodicity)}"
    )
    attrs = {"long_name": long_name, "valid_min": -3.09, "valid_max": 3.09}
    var_name = "spei_" + request.distribution.value + "_" + str(request.scale).zfill(2)

    return var_name, attrs


def _pnp_variable_attributes(request: _IndexRequest) -> tuple[str, dict[str, Any]]:
    assert request.scale is not None, _UNVALIDATED_SCALE
    long_name = "Percentage of Normal Precipitation, " + f"{request.scale}-{_get_scale_increment(request.periodicity)}"
    attrs = {"long_name": long_name, "valid_min": -1000.0, "valid_max": 1000.0}
    var_name = "pnp_" + str(request.scale).zfill(2)

    return var_name, attrs


def _pet_variable_attributes(request: _IndexRequest) -> tuple[str, dict[str, Any]]:
    long_name = "Potential Evapotranspiration (Thornthwaite)"
    attrs = {
        "long_name": long_name,
        "valid_min": 0.0,
        "valid_max": 10000.0,
        "units": "millimeters",
    }

    return "pet_thornthwaite", attrs


def _prepare_palmer_inputs(request: _IndexRequest, dataset: xr.Dataset) -> xr.Dataset:
    """
    Convert the Palmer inputs to the inches palmer.pdsi() takes.

    :param request: the index request being computed
    :param dataset: the opened inputs, converted in place
    :return: the opened available water capacity dataset
    :raise ValueError: if an input's units are unsupported
    """
    assert request.var_name_precip is not None, _UNVALIDATED_PRECIP
    assert request.var_name_pet is not None, _UNVALIDATED_PET

    if dataset[request.var_name_precip].units.lower() == "mm/dy":
        # a daily rate isn't the monthly accumulated depth palmer.pdsi() requires
        raise ValueError("Unsupported precipitation units for palmers: 'mm/dy' is a daily rate, not a monthly total")

    for var_name in (request.var_name_precip, request.var_name_pet):
        # out-of-place so integer-valued variables are promoted rather than rejected
        dataset[var_name].values = dataset[var_name].values / 25.4

    if request.netcdf_awc is None or request.var_name_awc is None:
        raise ValueError("Missing the AWC file and/or variable name argument(s)")

    awc_dataset = xr.open_dataset(request.netcdf_awc)

    # the Palmer routines take available water capacity in inches; an
    # absent units attribute is assumed to already be inches
    awc_var_name = request.var_name_awc
    awc_units = str(awc_dataset[awc_var_name].attrs.get("units", "")).strip().lower()
    if awc_units in ("mm", "millimeters", "millimeter"):
        awc_dataset[awc_var_name].values = awc_dataset[awc_var_name].values / 25.4
    elif awc_units and awc_units not in ("inch", "inches"):
        # !r so a units attribute holding newlines/control characters can't
        # forge log lines or alter terminal rendering when this is logged
        raise ValueError(f"Unsupported available water capacity units: {awc_units!r}")

    return awc_dataset


def _prepare_latitude_array(request: _IndexRequest, dataset: xr.Dataset) -> None:
    """
    Copy the latitude coordinate into a shared-memory array for the PET workers.

    :param request: the index request being computed
    :param dataset: the opened inputs
    """
    latitudes = dataset["lat"]
    _allocate_shared_array(_KEY_LAT, latitudes.shape)
    np.copyto(_shared_array(_KEY_LAT, latitudes.shape), latitudes.values)


def _compute_single_array(context: _ComputeContext) -> None:
    """
    Apply the index's kernel across the shared inputs, into one shared result array.

    :param context: the opened inputs and output settings of the request
    """
    handler = _registry_for(context.request.index)
    # an aggregate pipeline reuses _KEY_RESULT across indices, whose output
    # shapes can differ (e.g. transposed input dimensions), so reallocate rather
    # than reshape an incompatible buffer
    existing = _global_shared_arrays.get(_KEY_RESULT)
    if existing is None or existing[_KEY_SHAPE] != context.output_shape:
        _allocate_shared_array(_KEY_RESULT, context.output_shape)

    if handler.prepare_arrays is not None:
        handler.prepare_arrays(context.request, context.dataset)

    _parallel_process(context.request, context.arguments)


def _compute_palmers(context: _ComputeContext) -> None:
    """
    Apply the Palmer kernel across the shared inputs, into its five shared result arrays.

    :param context: the opened inputs and output settings of the request
    """
    request = context.request
    assert request.var_name_precip is not None, _UNVALIDATED_PRECIP
    assert request.var_name_pet is not None, _UNVALIDATED_PET
    assert request.var_name_awc is not None, _UNVALIDATED_AWC
    assert context.prepared is not None, "the AWC dataset is opened before the shared arrays are filled"

    # read AWC data into a shared memory array; already opened and unit-validated
    awc_array = context.prepared[request.var_name_awc]
    _allocate_shared_array(request.var_name_awc, awc_array.shape)
    np.copyto(_shared_array(request.var_name_awc, awc_array.shape), awc_array.values)

    # add shared memory arrays for the computed Palmers to the dictionary of shared arrays
    for key, _var_name, _long_name in _PALMER_OUTPUTS:
        if key not in _global_shared_arrays:
            _allocate_shared_array(key, context.output_shape)

    # TODO once we support daily Palmers then we'll need to convert values
    #  from a 366-day calendar back into a normal/Gregorian calendar
    _parallel_process(request, context.arguments)


def _write_single_output(context: _ComputeContext) -> tuple[str, str]:
    """
    Write a computed single-output index into its NetCDF file.

    :param context: the opened inputs and output settings of the request
    :return: the name of the file written and of the variable within it
    """
    request = context.request
    handler = _registry_for(request.index)
    assert handler.variable_attributes is not None, "a single-output index names its output variable"

    output_var_name, output_var_attributes = handler.variable_attributes(request)

    # get the shared memory results array and convert it to a numpy array
    index_values = _shared_array(handler.output_keys[0], context.output_shape).astype(float)

    # convert daily values into normal/Gregorian calendar years
    if request.periodicity == compute.Periodicity.daily:
        assert request.data_start_year is not None, "the inputs' start year is read when they are opened"
        index_values = np.apply_along_axis(
            utils.transform_to_gregorian,
            len(context.output_dims) - 1,
            index_values,
            request.data_start_year,
        )

    # create a new variable to contain the index values, assign into the dataset
    dataset = context.dataset
    variable = xr.Variable(
        dims=context.output_dims,
        data=index_values,
        attrs=output_var_attributes,
        encoding=context.output_encodings,
    )
    dataset[output_var_name] = variable

    # TODO set global attributes accordingly for this new dataset

    # remove all data variables except for the new variable
    drop_var_names = [var_name for var_name in dataset.data_vars if var_name != output_var_name]
    if len(drop_var_names):
        dataset = dataset.drop_vars(names=drop_var_names)

    # write the dataset as NetCDF
    netcdf_file_name = request.output_file_base + "_" + output_var_name + ".nc"
    dataset.to_netcdf(netcdf_file_name, engine=context.output_engine)

    return netcdf_file_name, output_var_name


def _write_palmer_outputs(context: _ComputeContext) -> None:
    """
    Write each of the computed Palmer outputs into its own NetCDF file.

    :param context: the opened inputs and output settings of the request
    """
    dataset = context.dataset
    for key, var_name, long_name in _PALMER_OUTPUTS:
        # get the shared memory results array and convert it to a numpy array
        index_values = _shared_array(key, context.output_shape).astype(float)
        attrs: dict[str, Any] = {"long_name": long_name}
        if var_name != "scpdsi":
            # scPDSI's percentile rescaling has no hard bound, unlike the
            # historical (and conservative) range kept for the standard outputs
            attrs |= {"valid_min": -10.0, "valid_max": 10.0}

        # create a new variable for this output and assign it into the dataset
        variable = xr.Variable(
            dims=context.output_dims,
            data=index_values,
            attrs=attrs,
            encoding=context.output_encodings,
        )
        dataset[var_name] = variable

        # TODO set global attributes accordingly for this new dataset

        # remove all data variables except for the new one
        drop_var_names = [name for name in dataset.data_vars if name != var_name]
        if len(drop_var_names):
            dataset = dataset.drop_vars(names=drop_var_names)

        # write the dataset as NetCDF
        netcdf_file_name = context.request.output_file_base + "_" + var_name + ".nc"
        dataset.to_netcdf(netcdf_file_name, engine=context.output_engine)


def _validate_kbdi_arguments(args: argparse.Namespace) -> None:
    """
    Validate that KBDI was given the arguments it can use.

    KBDI is computed for daily inputs only, through the fire module, and does not
    use the scale, calibration, PET, or AWC arguments of the other indices.

    :param args: an arguments object of the type returned by
        argparse.ArgumentParser.parse_args()
    :raise ValueError: if an argument KBDI cannot use was provided, or if the
        temperature input it requires is missing
    """
    if args.periodicity is not compute.Periodicity.daily:
        msg = "Invalid periodicity argument for KBDI: " + f"'{args.periodicity}' -- only 'daily' is supported"
        _logger.error(msg)
        raise ValueError(msg)

    if args.scales is not None:
        msg = "The --scales argument is not applicable to KBDI"
        _logger.error(msg)
        raise ValueError(msg)

    if args.calibration_start_year is not None or args.calibration_end_year is not None:
        msg = "The --calibration_start_year and --calibration_end_year arguments are not applicable to KBDI"
        _logger.error(msg)
        raise ValueError(msg)

    if args.netcdf_pet is not None or args.var_name_pet is not None:
        msg = "The --netcdf_pet and --var_name_pet arguments are not applicable to KBDI"
        _logger.error(msg)
        raise ValueError(msg)

    if args.netcdf_awc is not None or args.var_name_awc is not None:
        msg = "The --netcdf_awc and --var_name_awc arguments are not applicable to KBDI"
        _logger.error(msg)
        raise ValueError(msg)

    if args.netcdf_temp is None:
        msg = "Missing the required temperature file argument"
        _logger.error(msg)
        raise ValueError(msg)

    if args.var_name_temp is None:
        msg = "Missing temperature variable name"
        _logger.error(msg)
        raise ValueError(msg)


def _validate_kbdi_inputs(args: argparse.Namespace, context: _InputContext) -> None:
    """
    Validate that KBDI's maximum temperature input matches the precipitation input.

    :param args: an arguments object of the type returned by
        argparse.ArgumentParser.parse_args()
    :param context: the validated precipitation input
    :raise ValueError: if the temperature input is invalid or does not match
    """
    with xr.open_dataset(args.netcdf_temp) as dataset_temp:
        if args.var_name_temp not in dataset_temp.variables:
            msg = (
                f"Invalid temperature variable name: '{args.var_name_temp}'"
                + f" does not exist in temperature file '{args.netcdf_temp}'"
            )
            _logger.error(msg)
            raise ValueError(msg)

        dimensions_temp = dataset_temp[args.var_name_temp].dims
        # compare dimension names rather than storage order: either supported
        # order of each input is valid, and fire.kbdi() aligns by name
        if set(dimensions_temp) != set(context.dimensions):
            msg = (
                f"Invalid dimensions of the temperature variable: {dimensions_temp} "
                + f"(expected the precipitation variable dimensions: {context.dimensions})"
            )
            _logger.error(msg)
            raise ValueError(msg)

        if not np.array_equal(context.times, dataset_temp["time"].values[:]):
            msg = "Precipitation and temperature variables contain non-matching times"
            _logger.error(msg)
            raise ValueError(msg)

        if context.input_type == DatasetLayout.GRID:
            assert context.latitudes is not None
            assert context.longitudes is not None
            if not np.allclose(
                context.latitudes,
                dataset_temp["lat"][:],
                atol=utils.get_tolerance(context.latitudes),
            ):
                msg = "Precipitation and temperature variables contain non-matching latitudes"
                _logger.error(msg)
                raise ValueError(msg)

            if not np.allclose(
                context.longitudes,
                dataset_temp["lon"][:],
                atol=utils.get_tolerance(context.longitudes),
            ):
                msg = "Precipitation and temperature variables contain non-matching longitudes"
                _logger.error(msg)
                raise ValueError(msg)

        elif context.input_type == DatasetLayout.DIVISIONS:
            assert context.divisions is not None
            if not np.array_equal(context.divisions, dataset_temp["division"][:]):
                msg = "Precipitation and temperature variables contain non-matching division IDs"
                _logger.error(msg)
                raise ValueError(msg)


def _run_spi(arguments: argparse.Namespace, input_type: DatasetLayout) -> None:
    """
    Compute SPI for each requested scale and distribution.

    :param arguments: the parsed command line arguments
    :param input_type: the input type determined by argument validation
    """
    for scale in arguments.scales:
        for distribution in indices.Distribution:
            _compute_write_index(
                _IndexRequest.from_arguments(
                    arguments,
                    index="spi",
                    input_type=input_type,
                    scale=scale,
                    distribution=distribution,
                )
            )


def _run_spei(arguments: argparse.Namespace, input_type: DatasetLayout) -> None:
    """
    Compute SPEI for each requested scale and distribution.

    :param arguments: the parsed command line arguments
    :param input_type: the input type determined by argument validation
    """
    for scale in arguments.scales:
        for distribution in indices.Distribution:
            _compute_write_index(
                _IndexRequest.from_arguments(
                    arguments,
                    index="spei",
                    input_type=input_type,
                    scale=scale,
                    distribution=distribution,
                )
            )


def _run_pnp(arguments: argparse.Namespace, input_type: DatasetLayout) -> None:
    """
    Compute percentage of normal precipitation for each requested scale.

    :param arguments: the parsed command line arguments
    :param input_type: the input type determined by argument validation
    """
    for scale in arguments.scales:
        _compute_write_index(_IndexRequest.from_arguments(arguments, index="pnp", input_type=input_type, scale=scale))


def _run_pet(arguments: argparse.Namespace, input_type: DatasetLayout) -> None:
    """
    Compute PET from the temperature input, unless a PET input was provided.

    Within an aggregate index this runs before SPEI and Palmers, so that either
    can consume the computed PET values when the caller supplied only
    temperature.

    :param arguments: the parsed command line arguments
    :param input_type: the input type determined by argument validation
    """
    # a provided PET file substitutes for a computed one only when a later
    # index in the pipeline consumes it; --index pet always computes from
    # temperature
    if arguments.netcdf_pet is not None and arguments.index != "pet":
        return

    result = _compute_write_index(_IndexRequest.from_arguments(arguments, index="pet", input_type=input_type))
    assert result is not None, "PET computation should return file and variable name"
    arguments.netcdf_pet, arguments.var_name_pet = result


def _run_palmers(arguments: argparse.Namespace, input_type: DatasetLayout) -> None:
    """
    Compute the Palmer drought indices.

    :param arguments: the parsed command line arguments
    :param input_type: the input type determined by argument validation
    """
    _compute_write_index(_IndexRequest.from_arguments(arguments, index="palmers", input_type=input_type))


def _run_kbdi(arguments: argparse.Namespace, input_type: DatasetLayout) -> None:
    """
    Compute KBDI through the fire module's xarray API.

    KBDI does not route through _compute_write_index(), since the shared daily
    path reshapes daily inputs into 366-day years and coerces input units, both
    of which corrupt the KBDI recurrence.

    :param arguments: the parsed command line arguments
    :param input_type: the input type determined by argument validation
    """
    request = _IndexRequest.from_arguments(arguments, index="kbdi", input_type=input_type)
    assert request.netcdf_precip is not None and request.var_name_precip is not None
    assert request.netcdf_temp is not None and request.var_name_temp is not None

    # KBDI's recurrence is sequential over time but independent per grid
    # cell/division, and fire.kbdi() requires the time axis in a single Dask
    # chunk: keep time whole and chunk the spatial axes, so the multi-decade
    # daily inputs are never all resident at once
    if request.input_type == DatasetLayout.GRID:
        chunks: dict[str, Any] = {"lat": "auto", "lon": "auto", "time": -1}
    elif request.input_type == DatasetLayout.DIVISIONS:
        chunks = {"division": "auto", "time": -1}
    else:
        chunks = {"time": -1}

    with (
        _open_with_default_chunks(xr.open_dataset, request.netcdf_precip, chunks=chunks) as dataset_precip,
        _open_with_default_chunks(xr.open_dataset, request.netcdf_temp, chunks=chunks) as dataset_temp,
    ):
        kbdi_values = fire.kbdi(
            dataset_precip[request.var_name_precip],
            dataset_temp[request.var_name_temp],
            units=arguments.kbdi_units,
            initial_kbdi=arguments.kbdi_initial,
        )

        # the xarray route names the result after its precipitation
        # input; use the CF variable name the `units` argument selected
        kbdi_values.name = "kbdi_imperial" if arguments.kbdi_units == "imperial" else "kbdi"
        output_file = f"{request.output_file_base}_{kbdi_values.name}.nc"

        # honor --chunksizes input by copying the precipitation
        # variable's on-disk chunks to the output variable; a chunksizes
        # encoding is only honored by an HDF5-backed engine, and the
        # supported xarray versions still default to scipy when
        # netCDF4 is absent
        output_engine: Literal["h5netcdf"] | None = None
        if request.chunksizes == "input":
            input_chunksizes = dataset_precip[request.var_name_precip].encoding.get("chunksizes")
            if input_chunksizes:
                kbdi_values.encoding["chunksizes"] = input_chunksizes
                output_engine = "h5netcdf"

        _logger.info("Writing KBDI values to file: %s", output_file)
        kbdi_values.to_netcdf(output_file, engine=output_engine)


_INDEX_REGISTRY: dict[str, _IndexRegistration] = {
    "spi": _IndexRegistration(
        index="spi",
        run=_run_spi,
        input_paths=("netcdf_precip", "var_name_precip"),
        requires_precip=True,
        requires_scales=True,
        build_arguments=_spi_arguments,
        variable_attributes=_spi_variable_attributes,
        kernel=_spi,
        input_array_keys=_precipitation_array_key,
        worker=_apply_along_axis,
        compute=_compute_single_array,
        write=_write_single_output,
    ),
    "spei": _IndexRegistration(
        index="spei",
        run=_run_spei,
        input_paths=("netcdf_precip", "var_name_precip", "netcdf_pet", "var_name_pet"),
        requires_precip=True,
        requires_pet_or_temp=True,
        requires_scales=True,
        build_arguments=_spi_arguments,
        variable_attributes=_spei_variable_attributes,
        kernel=_spei,
        input_array_keys=_precipitation_and_pet_array_keys,
        worker=_apply_along_axis_double,
        compute=_compute_single_array,
        write=_write_single_output,
    ),
    "pnp": _IndexRegistration(
        index="pnp",
        run=_run_pnp,
        input_paths=("netcdf_precip", "var_name_precip"),
        requires_precip=True,
        requires_scales=True,
        build_arguments=_pnp_arguments,
        variable_attributes=_pnp_variable_attributes,
        kernel=_pnp,
        input_array_keys=_precipitation_array_key,
        worker=_apply_along_axis,
        compute=_compute_single_array,
        write=_write_single_output,
    ),
    "pet": _IndexRegistration(
        index="pet",
        run=_run_pet,
        input_paths=("netcdf_temp", "var_name_temp"),
        build_arguments=_pet_arguments,
        variable_attributes=_pet_variable_attributes,
        prepare_arrays=_prepare_latitude_array,
        kernel=_pet,
        input_array_keys=_temperature_and_latitude_array_keys,
        coordinate_input=True,
        worker=_apply_along_axis_double,
        compute=_compute_single_array,
        write=_write_single_output,
    ),
    "palmers": _IndexRegistration(
        index="palmers",
        run=_run_palmers,
        input_paths=(
            "netcdf_precip",
            "var_name_precip",
            "netcdf_pet",
            "var_name_pet",
            "netcdf_awc",
            "var_name_awc",
        ),
        requires_precip=True,
        requires_pet_or_temp=True,
        requires_awc=True,
        build_arguments=_palmer_arguments,
        prepare_inputs=_prepare_palmer_inputs,
        kernel=_palmers,
        input_array_keys=_palmer_array_keys,
        output_keys=tuple(key for key, _var_name, _long_name in _PALMER_OUTPUTS),
        worker=_apply_along_axis_palmers,
        compute=_compute_palmers,
        write=_write_palmer_outputs,
    ),
    "kbdi": _IndexRegistration(
        index="kbdi",
        run=_run_kbdi,
        input_paths=("netcdf_precip", "var_name_precip", "netcdf_temp", "var_name_temp"),
        requires_precip=True,
        validate_arguments=_validate_kbdi_arguments,
        validate_inputs=_validate_kbdi_inputs,
    ),
}

# the indices behind each --index value, in the order they are run: PET runs
# before the indices that consume its output when no PET input was provided
_INDEX_PIPELINES: dict[str, tuple[str, ...]] = {
    "spi": ("spi",),
    "spei": ("pet", "spei"),
    "pnp": ("pnp",),
    "scaled": ("spi", "pet", "spei", "pnp"),
    "pet": ("pet",),
    "palmers": ("pet", "palmers"),
    "kbdi": ("kbdi",),
    "all": ("spi", "pet", "spei", "pnp", "palmers"),
}


def _registry_for(index: str) -> _IndexRegistration:
    """
    Return the registration of one computable index.

    :param str index: the index's name
    :return: the index's registration
    :raise ValueError: if no index is registered under that name
    """
    try:
        return _INDEX_REGISTRY[index]
    except KeyError:
        raise ValueError(f"Unsupported index: '{index}'") from None


def _handlers_for_index(index: str) -> tuple[_IndexRegistration, ...]:
    """
    Return the registrations behind an ``--index`` value, in execution order.

    :param str index: the ``--index`` value, which may be an aggregate of
        several indices
    :return: the registrations of the indices that value runs
    :raise ValueError: if no pipeline is registered under that name
    """
    try:
        pipeline = _INDEX_PIPELINES[index]
    except KeyError:
        raise ValueError(f"Unsupported index: '{index}'") from None

    return tuple(_INDEX_REGISTRY[name] for name in pipeline)


def main(argv: Sequence[str] | None = None) -> None:
    """
    Perform climate indices processing on NetCDF datasets, which may be
    gridded, US climate division, or single-location time-series inputs.

    :param argv: command line arguments; defaults to ``sys.argv[1:]`` when omitted.

    Example command line arguments for SPI only using monthly precipitation input:

    --index spi
    --periodicity monthly
    --scales 1 2 3 6 9 12 24
    --calibration_start_year 1998
    --calibration_end_year 2016
    --netcdf_precip example_data/nclimgrid_prcp_lowres.nc
    --var_name_precip prcp
    --output_file_base ~/data/test/spi/nclimgrid_lowres
    """
    try:
        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        _logger.info("Start time:    %s", start_datetime)

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--index",
            help="Indices to compute",
            choices=list(_INDEX_PIPELINES),
            required=True,
        )
        _add_common_spi_arguments(parser)
        parser.add_argument(
            "--netcdf_temp",
            help="Temperature NetCDF file to be used as input for indices computations",
        )
        parser.add_argument(
            "--var_name_temp",
            help="Temperature variable name used in the temperature NetCDF file",
        )
        parser.add_argument(
            "--netcdf_pet",
            help="PET NetCDF file to be used as input for SPEI and/or Palmer computations",
        )
        parser.add_argument("--var_name_pet", help="PET variable name used in the PET NetCDF file")
        parser.add_argument(
            "--netcdf_awc",
            help="Available water capacity NetCDF file to be used as input for the Palmer computations",
        )
        parser.add_argument(
            "--var_name_awc",
            help="Available water capacity variable name used in the AWC NetCDF file",
        )
        parser.add_argument(
            "--kbdi_units",
            help="Units of the KBDI input and output values",
            choices=["metric", "imperial"],
            required=False,
            default="metric",
        )
        parser.add_argument(
            "--kbdi_initial",
            help="Initial KBDI value",
            type=float,
            required=False,
            default=0.0,
        )
        parser.add_argument(
            "--chunksizes",
            help="Output file chunksizes. Can be 'none' (default), or 'input' to match input chunks",
            choices=["none", "input"],
            required=False,
            default="none",
        )

        arguments = parser.parse_args(argv)

        process_climate_indices(arguments=arguments)

        # report the elapsed time
        end_datetime = datetime.now()
        _logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        _logger.info("Elapsed time:  %s", elapsed)

    except Exception:
        _logger.exception("Failed to complete", exc_info=True)
        raise


def process_climate_indices(
    arguments: argparse.Namespace,
) -> None:
    """
    Process climate indices based on the provided arguments.

    :param arguments: the parsed command line arguments
    :return: The results of the climate indices processing
    """

    # start each invocation with fresh shared arrays, so result storage
    # retained from an earlier invocation in this process (with a possibly
    # incompatible shape) is never reused
    global _global_shared_arrays
    _global_shared_arrays = {}

    try:
        # validate the arguments and determine the input type
        input_type = _validate_args(arguments)

        global _NUMBER_OF_WORKER_PROCESSES
        if arguments.multiprocessing == "single":
            _NUMBER_OF_WORKER_PROCESSES = 1
        elif arguments.multiprocessing == "all":
            _NUMBER_OF_WORKER_PROCESSES = multiprocessing.cpu_count()
        else:  # default ("all_but_one")
            _NUMBER_OF_WORKER_PROCESSES = multiprocessing.cpu_count() - 1

        # run every index behind the --index value, in pipeline order
        for handler in _handlers_for_index(arguments.index):
            handler.run(arguments, input_type)

    except Exception:
        _logger.exception("Failed to complete", exc_info=True)
        raise

    return None


if __name__ == "__main__":
    # (please do not remove -- useful for running as a script when debugging)

    main()
