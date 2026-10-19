"""Standardized Residual Load Index (SRLI) and the residual-load helper.

Residual load is demand minus renewable production. Both the helper and the index
describe the **climate-driven component** of supply and demand, not actual load: a
series of observed consumption also carries capacity changes, curtailment, outages,
markets, and behaviour. Residual load is signed, so SRLI is a thin wrapper over
:func:`climate_indices.indices.standardized_index` (ADR-0019) that admits only the
empirical (rank-based) distribution, the one path with no support for a negative value
to violate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices import compute, indices
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.energy._xarray import STANDARDIZED_VALIDATORS, apply_standardized
from climate_indices.exceptions import InputTypeError, InvalidArgumentError
from climate_indices.xarray_adapter import INFER_TIME_PARAMETERS, xarray_adapter


def _numeric(name: str, value: npt.ArrayLike, fit_shape: tuple[int, ...] | None = None) -> npt.NDArray[np.float64]:
    """Return a float64 copy with masked values as NaN, or raise for a non-numeric or infinite value.

    With ``fit_shape``, an operand that does not broadcast to it is rejected before the copy.
    """
    try:
        raw = np.asarray(value)
    except ValueError as error:  # a ragged sequence has no array form
        raise InputTypeError(f"{name} must be numeric.", expected_type=float, actual_type=type(value)) from error
    if raw.dtype.kind not in "iuf":
        raise InputTypeError(f"{name} must be numeric.", expected_type=float, actual_type=raw.dtype.type)
    if fit_shape is not None:
        try:
            fits = np.broadcast_shapes(raw.shape, fit_shape) == fit_shape
        except ValueError:
            fits = False
        if not fits:
            raise InvalidArgumentError(
                f"{name} must broadcast to the shape of demand.",
                argument_name=name,
                argument_value=f"shape {raw.shape} for demand of shape {fit_shape}",
                valid_values="Shapes broadcastable to the shape of demand",
            )
    values = np.asarray(np.ma.asarray(value, dtype=np.float64).filled(np.nan), dtype=np.float64)
    if np.any(np.isinf(values)):
        raise InvalidArgumentError(
            f"{name} must be finite or NaN.",
            argument_name=name,
            argument_value="infinite value",
            valid_values="Finite values or NaN",
        )
    return values


def residual_load(
    demand: npt.ArrayLike,
    capacity_factors: Sequence[npt.ArrayLike],
    installed_capacities: Sequence[npt.ArrayLike],
) -> npt.NDArray[np.float64]:
    """Form residual load as demand minus capacity-factor-weighted renewable production.

    ``residual = demand - sum(capacity_factor_i * installed_capacity_i)`` over the
    technologies, for example wind and solar. The result is signed: a step where
    production exceeds demand is negative and is never clipped to zero. It is the
    input that :func:`srli` standardizes.

    The installed capacities are the caller's assumptions and the helper cannot check
    them. Choose their units so that each product is in the units of ``demand``: for a
    demand series in MWh per time step, give each capacity in MW multiplied by the hours
    in one step. Capacity may change over time, as an array. Holding it fixed over a long
    record is itself an assumption, and the result is no more accurate than that choice.

    Args:
        demand: Energy demand, or a demand proxy, in the units of the products below.
            Masked and NaN values give NaN.
        capacity_factors: One array per technology, each within [0, 1] or NaN and
            broadcastable to the shape of ``demand``. A value outside [0, 1] usually
            means percentages were passed.
        installed_capacities: One entry per technology, in the order of
            ``capacity_factors``: a finite non-negative scalar, or an array
            broadcastable to the shape of ``demand`` for a capacity that changes.

    Returns:
        Float64 residual load in the shape of ``demand``.

    Raises:
        InputTypeError: If an input is not numeric.
        InvalidArgumentError: If there are no technologies or the two sequences differ in
            length, a capacity factor is outside [0, 1], a capacity is negative or not
            finite, a value is infinite, a technology's shape would enlarge the shape of
            ``demand``, or the result overflows to infinity.
    """
    if len(capacity_factors) == 0:
        raise InvalidArgumentError(
            "capacity_factors must hold at least one technology.",
            argument_name="capacity_factors",
            argument_value="no entries",
            valid_values="One array per technology, at least one",
        )
    if len(capacity_factors) != len(installed_capacities):
        raise InvalidArgumentError(
            "installed_capacities must hold one entry per entry of capacity_factors.",
            argument_name="installed_capacities",
            argument_value=f"{len(installed_capacities)} entries for {len(capacity_factors)} capacity factors",
            valid_values="As many capacities as capacity factors",
        )
    residual = _numeric("demand", demand)
    for factor_input, capacity_input in zip(capacity_factors, installed_capacities, strict=True):
        factor = _numeric("capacity_factors", factor_input, residual.shape)
        capacity = _numeric("installed_capacities", capacity_input, residual.shape)
        if np.any((factor < 0.0) | (factor > 1.0)):
            raise InvalidArgumentError(
                "capacity_factors must be within [0, 1].",
                argument_name="capacity_factors",
                argument_value=f"min {np.nanmin(factor)}, max {np.nanmax(factor)}",
                valid_values="[0, 1] or NaN",
            )
        if not np.all(np.isfinite(capacity) & (capacity >= 0.0)):
            raise InvalidArgumentError(
                "installed_capacities must be finite and non-negative.",
                argument_name="installed_capacities",
                argument_value="a negative or non-finite capacity",
                valid_values="Finite values of at least 0",
            )
        with np.errstate(over="ignore"):
            residual = residual - factor * capacity
    if np.any(np.isinf(residual)):
        raise InvalidArgumentError(
            "The residual load overflowed to infinity.",
            argument_name="installed_capacities",
            argument_value="a product too large for float64",
            valid_values="Capacities whose products stay within float64",
        )
    return np.asarray(residual)


def _validate_distribution(distribution: indices.Distribution) -> None:
    if distribution is not indices.Distribution.empirical:
        raise InvalidArgumentError(
            "SRLI supports only the empirical distribution: a parametric fit does not suit a signed series.",
            argument_name="distribution",
            argument_value=str(distribution),
            valid_values="Distribution.empirical",
        )


def _srli_numpy(
    values: np.ndarray,
    scale: int,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
    *,
    distribution: indices.Distribution = indices.Distribution.empirical,
    aggregation: compute.Aggregation = "sum",
    spatial_time_major: bool = False,
    output_scale: compute.OutputScale = "normal",
) -> np.ndarray:
    """Standardized Residual Load Index (SRLI) of Allen & Otero (2023).

    Standardizes a signed residual-load series, demand minus renewable production (see
    :func:`residual_load`), against its Calibration Period. High values mean
    higher-than-usual residual load, which is the stress direction: Allen & Otero class
    drought at the 90th, 95th, and 97.5th percentiles of the standard normal (1.28, 1.64,
    1.96). The index is the climate-driven component of residual load, not actual load.

    Negative values are kept as given, never clipped, because clipping residual load to
    zero would standardize a different series. Only the empirical distribution accepts a
    signed series, so any other ``distribution`` raises. Each value's probability is its
    rank among the valid Calibration Period values at its calendar step (the plotting
    position of :func:`climate_indices.indices.standardized_index`), so the choice of
    ``aggregation`` moves the accumulated value but not its rank. Allen & Otero recommend
    this distribution above roughly 100 observations; the index makes no fallback to a
    parametric fit below that. A constant Calibration Period series ties every value at
    the highest plotting position, so it reads as maximum stress at every step.

    Unlike :func:`srepi` and :func:`standardized_demand_index`, this function takes no
    ``fitting_params`` or ``zero_handling``: both belong to the parametric families.

    Args:
        values: 1-D array of signed residual load, with the layouts that
            :func:`climate_indices.indices.standardized_index` accepts.
        scale: Number of time steps in each window before standardization.
        data_start_year: Initial year of the input series.
        calibration_year_initial: Initial year of the Calibration Period.
        calibration_year_final: Final year of the Calibration Period.
        periodicity: ``compute.Periodicity.monthly`` or ``compute.Periodicity.daily``.
            Daily input must start on 1 January in the 366-slot year layout; convert a
            Gregorian array with :func:`climate_indices.utils.transform_to_366day`.
        distribution: ``Distribution.empirical``, the only accepted member.
        aggregation: How each window is combined, "sum" (the default) or "mean".
        spatial_time_major: Read a time-major block of independent series.
        output_scale: "normal" (the default), "probability", or "bounded".

    Returns:
        Unitless standardized values, shaped as
        :func:`climate_indices.indices.standardized_index` returns them. A calendar step
        with fewer than ten valid calibration values is NaN.

    Raises:
        InvalidArgumentError: If ``distribution`` is not empirical, or the seam rejects
            the arguments.
        ValueError: If ``aggregation`` is not "sum" or "mean".

    References:
        Allen, S., & Otero, N. (2023). Standardised indices to monitor energy droughts.
        Renewable Energy, 217, 119206. https://doi.org/10.1016/j.renene.2023.119206
    """
    _validate_distribution(distribution)
    return indices.standardized_index(
        values,
        scale,
        distribution,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        periodicity,
        spatial_time_major=spatial_time_major,
        output_scale=output_scale,
        aggregation=aggregation,
        clip_negatives=False,
    )


_wrapped_srli: Callable[..., npt.NDArray[np.float64] | xr.DataArray] = xarray_adapter(
    calendar="periodicity",
    inferred_parameters=INFER_TIME_PARAMETERS,
    argument_validators=(
        *STANDARDIZED_VALIDATORS,
        lambda args: _validate_distribution(args["distribution"]),
    ),
    timescale_parameter="scale",
    cf_metadata=CF_METADATA["srli"],  # type: ignore[arg-type]
    index_display_name="SRLI",
    calculation_metadata_keys=[
        "scale",
        "distribution",
        "calibration_year_initial",
        "calibration_year_final",
        "aggregation",
    ],
)(_srli_numpy)


@overload
def srli(
    values: xr.DataArray,
    scale: int,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    periodicity: compute.Periodicity | None = None,
    *,
    distribution: indices.Distribution = indices.Distribution.empirical,
    aggregation: compute.Aggregation = "sum",
    output_scale: compute.OutputScale = "normal",
) -> xr.DataArray: ...


@overload
def srli(
    values: npt.ArrayLike,
    scale: int,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
    *,
    distribution: indices.Distribution = indices.Distribution.empirical,
    aggregation: compute.Aggregation = "sum",
    spatial_time_major: bool = False,
    output_scale: compute.OutputScale = "normal",
) -> npt.NDArray[np.float64]: ...


def srli(
    values: npt.ArrayLike | xr.DataArray,
    scale: int,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    periodicity: compute.Periodicity | None = None,
    *,
    distribution: indices.Distribution = indices.Distribution.empirical,
    aggregation: compute.Aggregation = "sum",
    spatial_time_major: bool = False,
    output_scale: compute.OutputScale = "normal",
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Standardized Residual Load Index (SRLI) of Allen & Otero (2023).

    Standardizes a signed residual-load series, demand minus renewable production (see
    :func:`residual_load`), against its Calibration Period. High values mean
    higher-than-usual residual load, which is the stress direction: Allen & Otero class
    drought at the 90th, 95th, and 97.5th percentiles of the standard normal (1.28, 1.64,
    1.96). The index is the climate-driven component of residual load, not actual load.

    Negative values are kept as given, never clipped, because clipping residual load to
    zero would standardize a different series. Only the empirical distribution accepts a
    signed series, so any other ``distribution`` raises. Each value's probability is its
    rank among the valid Calibration Period values at its calendar step (the plotting
    position of :func:`climate_indices.indices.standardized_index`), so the choice of
    ``aggregation`` moves the accumulated value but not its rank. Allen & Otero recommend
    this distribution above roughly 100 observations; the index makes no fallback to a
    parametric fit below that. A constant Calibration Period series ties every value at
    the highest plotting position, so it reads as maximum stress at every step.

    Unlike :func:`srepi` and :func:`standardized_demand_index`, this function takes no
    ``fitting_params`` or ``zero_handling``: both belong to the parametric families.

    DataArray input takes the beta adapter path; temporal parameters are inferred from
    the time coordinate when omitted, the ``srli`` registry metadata is stamped, and
    each grid cell is standardized independently to match its NumPy series exactly.
    The adapter validates ``distribution`` at call time, before building a Dask graph.

    Args:
        values: 1-D array of signed residual load, or a DataArray with a time dimension
            plus optional cell dimensions, with the layouts that
            :func:`climate_indices.indices.standardized_index` accepts.
        scale: Number of time steps in each window before standardization.
        data_start_year: Initial year of the input series. Required for NumPy input;
            inferred for DataArray input.
        calibration_year_initial: Initial year of the Calibration Period. Required
            for NumPy input; inferred for DataArray input.
        calibration_year_final: Final year of the Calibration Period. Required for
            NumPy input; inferred for DataArray input.
        periodicity: ``compute.Periodicity.monthly`` or ``compute.Periodicity.daily``.
            Required for NumPy input; inferred for DataArray input. Daily NumPy input
            must start on 1 January in the 366-slot year layout; convert a Gregorian
            array with :func:`climate_indices.utils.transform_to_366day`.
        distribution: ``Distribution.empirical``, the only accepted member.
        aggregation: How each window is combined, "sum" (the default) or "mean".
        spatial_time_major: Read a time-major block of independent series. Only used
            for NumPy input.
        output_scale: "normal" (the default), "probability", or "bounded".

    Returns:
        Unitless standardized values: an array for NumPy input, a DataArray carrying
        the ``srli`` registry metadata and the input's coordinates for DataArray input.
        A calendar step with fewer than ten valid calibration values is NaN.

    Raises:
        InvalidArgumentError: If ``distribution`` is not empirical, or the seam rejects
            the arguments.
        ValueError: If ``aggregation`` is not "sum" or "mean".
        CoordinateValidationError: If a DataArray's time dimension is missing,
            non-monotonic, or its periodicity does not match ``periodicity``.

    References:
        Allen, S., & Otero, N. (2023). Standardised indices to monitor energy droughts.
        Renewable Energy, 217, 119206. https://doi.org/10.1016/j.renene.2023.119206
    """
    return apply_standardized(
        _wrapped_srli,
        values,
        scale=scale,
        data_start_year=data_start_year,
        calibration_year_initial=calibration_year_initial,
        calibration_year_final=calibration_year_final,
        periodicity=periodicity,
        distribution=distribution,
        aggregation=aggregation,
        spatial_time_major=spatial_time_major,
        output_scale=output_scale,
    )
