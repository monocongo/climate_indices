"""Standardized renewable production and demand indices (SREPI and its demand analogue).

Both describe the **climate-driven component** of supply or demand, not actual
generation or load: a series of observed output or consumption also carries
capacity changes, curtailment, outages, markets, and behaviour, so its scores are not
weather-only diagnostics. They are thin wrappers over :func:`climate_indices.indices.standardized_index`
(ADR-0019), with the empirical (rank-based) distribution as the default that Allen &
Otero (2023) recommend above roughly 100 observations. The beta distribution is
the alternative for a bounded [0, 1] series such as a capacity factor.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices import compute, indices
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.energy._xarray import STANDARDIZED_INDEX_VALIDATORS, apply_standardized
from climate_indices.exceptions import InvalidArgumentError
from climate_indices.xarray_adapter import INFER_TIME_PARAMETERS, xarray_adapter

# Allen & Otero (2023), section 3.1 and table 1: the 90th, 95th, and 97.5th percentiles of
# the standard normal (1.28, 1.64, 1.96) give the moderate, severe, and extreme classes.
# An SREPI at or below the negative of a value is a drought of that class; the paper
# flags that other cut-offs (for example 2.33 for extreme) are equally defensible.
SREPI_DROUGHT_THRESHOLDS: Mapping[str, float] = MappingProxyType({"moderate": -1.28, "severe": -1.64, "extreme": -1.96})

_DISTRIBUTIONS = (indices.Distribution.empirical, indices.Distribution.beta)


def _validate_distribution(distribution: indices.Distribution) -> None:
    if distribution not in _DISTRIBUTIONS:
        raise InvalidArgumentError(
            "Energy standardized indices support only the empirical and beta distributions.",
            argument_name="distribution",
            argument_value=str(distribution),
            valid_values="Distribution.empirical, Distribution.beta",
        )


def _standardize(
    values: np.ndarray,
    scale: int,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
    fitting_params: dict[str, Any] | None,
    *,
    distribution: indices.Distribution,
    aggregation: compute.Aggregation,
    spatial_time_major: bool,
    output_scale: compute.OutputScale,
    zero_handling: compute.ZeroHandling,
    pool_window_days: int,
) -> np.ndarray:
    _validate_distribution(distribution)
    # negatives are never clipped to zero: an empirical rank takes them as given and beta rejects them
    return indices.standardized_index(
        values,
        scale,
        distribution,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        periodicity,
        fitting_params,
        spatial_time_major=spatial_time_major,
        output_scale=output_scale,
        zero_handling=zero_handling,
        aggregation=aggregation,
        clip_negatives=False,
        pool_window_days=pool_window_days,
    )


def _srepi_numpy(
    values: np.ndarray,
    scale: int,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
    fitting_params: dict[str, Any] | None = None,
    *,
    distribution: indices.Distribution = indices.Distribution.empirical,
    aggregation: compute.Aggregation = "mean",
    spatial_time_major: bool = False,
    output_scale: compute.OutputScale = "normal",
    zero_handling: compute.ZeroHandling = "classic",
    pool_window_days: int = 0,
) -> np.ndarray:
    """Standardized Renewable Energy Production Index (SREPI) of Allen & Otero (2023).

    Standardizes a renewable production series, or a wind, PV, or combined capacity
    factor, against its Calibration Period. Low values mean weaker-than-usual supply;
    :data:`SREPI_DROUGHT_THRESHOLDS` holds the paper's drought classes. The index is the
    climate-driven component of supply, not actual generation.

    The default ``aggregation="mean"`` suits a capacity factor. A series of accumulated
    production must pass ``aggregation="sum"``; input meaning is never inferred from
    magnitudes. The beta distribution needs every source and windowed value within
    [0, 1], which in practice means a capacity factor with ``aggregation="mean"``;
    :func:`climate_indices.indices.standardized_index` rejects a value outside that
    range. All other arguments are those of that function.

    Args:
        values: 1-D array of production or capacity factors, with the layouts that
            :func:`climate_indices.indices.standardized_index` accepts.
        scale: Number of time steps in each window before standardization.
        data_start_year: Initial year of the input series.
        calibration_year_initial: Initial year of the Calibration Period.
        calibration_year_final: Final year of the Calibration Period.
        periodicity: ``compute.Periodicity.monthly``, ``compute.Periodicity.daily``, or
            ``compute.Periodicity.hourly``. Hourly input carries 8784 values per year and
            supports only ``Distribution.empirical``.
            Daily input must start on 1 January in the 366-slot year layout; convert a
            Gregorian array with :func:`climate_indices.utils.transform_to_366day`.
        fitting_params: Pre-computed beta fitting parameters; the empirical
            distribution has none.
        distribution: ``Distribution.empirical`` (the default) or ``Distribution.beta``.
        aggregation: How each window is combined, "mean" (the default) or "sum".
        spatial_time_major: Read a time-major block of independent series.
        output_scale: "normal" (the default), "probability", or "bounded".
        zero_handling: Placement of a zero value within the boundary mass (beta only),
            one of ``compute.ZeroHandling``: "classic", "center_of_mass", or "mean_zero".
        pool_window_days: Number of neighbouring days on each side of an hourly
            empirical step whose values join its calibration sample; zero (the default)
            keeps the shipped hour x day-of-year strata. Available only with
            ``Periodicity.hourly`` and ``Distribution.empirical`` (#1407).

    Returns:
        Unitless standardized values, shaped as
        :func:`climate_indices.indices.standardized_index` returns them. With the
        empirical distribution, a calendar step with fewer than ten valid calibration
        values is NaN.

    Raises:
        InvalidArgumentError: If ``distribution`` is neither empirical nor beta, or
            the seam rejects the arguments.
        ValueError: If ``aggregation`` or ``zero_handling`` is not a supported mode.

    References:
        Allen, S., & Otero, N. (2023). Standardised indices to monitor energy droughts.
        Renewable Energy, 217, 119206. https://doi.org/10.1016/j.renene.2023.119206
    """
    return _standardize(
        values,
        scale,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        periodicity,
        fitting_params,
        distribution=distribution,
        aggregation=aggregation,
        spatial_time_major=spatial_time_major,
        output_scale=output_scale,
        zero_handling=zero_handling,
        pool_window_days=pool_window_days,
    )


def _standardized_demand_index_numpy(
    values: np.ndarray,
    scale: int,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
    fitting_params: dict[str, Any] | None = None,
    *,
    distribution: indices.Distribution = indices.Distribution.empirical,
    aggregation: compute.Aggregation = "sum",
    spatial_time_major: bool = False,
    output_scale: compute.OutputScale = "normal",
    zero_handling: compute.ZeroHandling = "classic",
    pool_window_days: int = 0,
) -> np.ndarray:
    """Standardized demand index over energy demand or energy degree days (EDD).

    The analogue of :func:`srepi` for demand, as in Allen & Otero (2023). High values
    mean higher-than-usual demand. The index is the climate-driven component of
    demand, not actual load; a degree-day input such as
    :func:`climate_indices.energy.energy_degree_days` is its direct input.

    Windows are accumulated by default (``aggregation="sum"``). The beta distribution
    applies only to a bounded [0, 1] series; see :func:`srepi`.

    Args:
        values: 1-D array of demand or degree days, with the layouts that
            :func:`climate_indices.indices.standardized_index` accepts.
        scale: Number of time steps in each window before standardization.
        data_start_year: Initial year of the input series.
        calibration_year_initial: Initial year of the Calibration Period.
        calibration_year_final: Final year of the Calibration Period.
        periodicity: ``compute.Periodicity.monthly``, ``compute.Periodicity.daily``, or
            ``compute.Periodicity.hourly``. Hourly input carries 8784 values per year and
            supports only ``Distribution.empirical``.
            Daily input must start on 1 January in the 366-slot year layout; convert a
            Gregorian array with :func:`climate_indices.utils.transform_to_366day`.
        fitting_params: Pre-computed beta fitting parameters; the empirical
            distribution has none.
        distribution: ``Distribution.empirical`` (the default) or ``Distribution.beta``.
        aggregation: How each window is combined, "sum" (the default) or "mean".
        spatial_time_major: Read a time-major block of independent series.
        output_scale: "normal" (the default), "probability", or "bounded".
        zero_handling: Placement of a zero value within the boundary mass (beta only),
            one of ``compute.ZeroHandling``: "classic", "center_of_mass", or "mean_zero".
        pool_window_days: Number of neighbouring days on each side of an hourly
            empirical step whose values join its calibration sample; zero (the default)
            keeps the shipped hour x day-of-year strata. Available only with
            ``Periodicity.hourly`` and ``Distribution.empirical`` (#1407).

    Returns:
        Unitless standardized values, shaped as
        :func:`climate_indices.indices.standardized_index` returns them. With the
        empirical distribution, a calendar step with fewer than ten valid calibration
        values is NaN.

    Raises:
        InvalidArgumentError: If ``distribution`` is neither empirical nor beta, or
            the seam rejects the arguments.
        ValueError: If ``aggregation`` or ``zero_handling`` is not a supported mode.

    References:
        Allen, S., & Otero, N. (2023). Standardised indices to monitor energy droughts.
        Renewable Energy, 217, 119206. https://doi.org/10.1016/j.renene.2023.119206
    """
    return _standardize(
        values,
        scale,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        periodicity,
        fitting_params,
        distribution=distribution,
        aggregation=aggregation,
        spatial_time_major=spatial_time_major,
        output_scale=output_scale,
        zero_handling=zero_handling,
        pool_window_days=pool_window_days,
    )


_wrapped_srepi: Callable[..., npt.NDArray[np.float64] | xr.DataArray] = xarray_adapter(
    calendar="periodicity",
    inferred_parameters=INFER_TIME_PARAMETERS,
    argument_validators=(
        *STANDARDIZED_INDEX_VALIDATORS,
        lambda args: _validate_distribution(args["distribution"]),
    ),
    timescale_parameter="scale",
    cf_metadata=CF_METADATA["srepi"],  # type: ignore[arg-type]
    index_display_name="SREPI",
    calculation_metadata_keys=[
        "scale",
        "distribution",
        "calibration_year_initial",
        "calibration_year_final",
        "aggregation",
    ],
)(_srepi_numpy)

_wrapped_standardized_demand_index: Callable[..., npt.NDArray[np.float64] | xr.DataArray] = xarray_adapter(
    calendar="periodicity",
    inferred_parameters=INFER_TIME_PARAMETERS,
    argument_validators=(
        *STANDARDIZED_INDEX_VALIDATORS,
        lambda args: _validate_distribution(args["distribution"]),
    ),
    timescale_parameter="scale",
    cf_metadata=CF_METADATA["standardized_demand_index"],  # type: ignore[arg-type]
    index_display_name="Standardized demand index",
    calculation_metadata_keys=[
        "scale",
        "distribution",
        "calibration_year_initial",
        "calibration_year_final",
        "aggregation",
    ],
)(_standardized_demand_index_numpy)


@overload
def srepi(
    values: xr.DataArray,
    scale: int,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    periodicity: compute.Periodicity | None = None,
    fitting_params: dict[str, Any] | None = None,
    *,
    distribution: indices.Distribution = indices.Distribution.empirical,
    aggregation: compute.Aggregation = "mean",
    output_scale: compute.OutputScale = "normal",
    zero_handling: compute.ZeroHandling = "classic",
    pool_window_days: int = 0,
) -> xr.DataArray: ...


@overload
def srepi(
    values: npt.ArrayLike,
    scale: int,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
    fitting_params: dict[str, Any] | None = None,
    *,
    distribution: indices.Distribution = indices.Distribution.empirical,
    aggregation: compute.Aggregation = "mean",
    spatial_time_major: bool = False,
    output_scale: compute.OutputScale = "normal",
    zero_handling: compute.ZeroHandling = "classic",
    pool_window_days: int = 0,
) -> npt.NDArray[np.float64]: ...


def srepi(
    values: npt.ArrayLike | xr.DataArray,
    scale: int,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    periodicity: compute.Periodicity | None = None,
    fitting_params: dict[str, Any] | None = None,
    *,
    distribution: indices.Distribution = indices.Distribution.empirical,
    aggregation: compute.Aggregation = "mean",
    spatial_time_major: bool = False,
    output_scale: compute.OutputScale = "normal",
    zero_handling: compute.ZeroHandling = "classic",
    pool_window_days: int = 0,
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Standardized Renewable Energy Production Index (SREPI) of Allen & Otero (2023).

    Standardizes a renewable production series, or a wind, PV, or combined capacity
    factor, against its Calibration Period. Low values mean weaker-than-usual supply;
    :data:`SREPI_DROUGHT_THRESHOLDS` holds the paper's drought classes. The index is the
    climate-driven component of supply, not actual generation.

    The default ``aggregation="mean"`` suits a capacity factor. A series of accumulated
    production must pass ``aggregation="sum"``; input meaning is never inferred from
    magnitudes. The beta distribution needs every source and windowed value within
    [0, 1], which in practice means a capacity factor with ``aggregation="mean"``;
    :func:`climate_indices.indices.standardized_index` rejects a value outside that
    range. All other arguments are those of that function.

    DataArray input uses the stable adapter: temporal parameters are inferred from the
    time coordinate when omitted, the ``srepi`` registry metadata is stamped, and
    each grid cell is fitted independently to match its NumPy series exactly.
    The adapter passes the same ``fitting_params`` to every cell.

    Args:
        values: 1-D array of production or capacity factors, or a DataArray with a
            time dimension plus optional cell dimensions, with the layouts that
            :func:`climate_indices.indices.standardized_index` accepts.
        scale: Number of time steps in each window before standardization.
        data_start_year: Initial year of the input series. Required for NumPy input;
            inferred for DataArray input.
        calibration_year_initial: Initial year of the Calibration Period. Required
            for NumPy input; inferred for DataArray input.
        calibration_year_final: Final year of the Calibration Period. Required for
            NumPy input; inferred for DataArray input.
        periodicity: ``compute.Periodicity.monthly``, ``compute.Periodicity.daily``, or
            ``compute.Periodicity.hourly``. Hourly input carries 8784 values per year and
            supports only ``Distribution.empirical``.
            Required for NumPy input; inferred for DataArray input. Daily NumPy input
            must start on 1 January in the 366-slot year layout; convert a Gregorian
            array with :func:`climate_indices.utils.transform_to_366day`.
        fitting_params: Pre-computed beta fitting parameters; the empirical
            distribution has none.
        distribution: ``Distribution.empirical`` (the default) or ``Distribution.beta``.
        aggregation: How each window is combined, "mean" (the default) or "sum".
        spatial_time_major: Read a time-major block of independent series. Only used
            for NumPy input.
        output_scale: "normal" (the default), "probability", or "bounded".
        zero_handling: Placement of a zero value within the boundary mass (beta only),
            one of ``compute.ZeroHandling``: "classic", "center_of_mass", or "mean_zero".
        pool_window_days: Number of neighbouring days on each side of an hourly
            empirical step whose values join its calibration sample; zero (the default)
            keeps the shipped hour x day-of-year strata. Available only with
            ``Periodicity.hourly`` and ``Distribution.empirical`` (#1407).

    Returns:
        Unitless standardized values: an array for NumPy input, a DataArray carrying
        the ``srepi`` registry metadata and the input's coordinates for DataArray
        input. With the empirical distribution, a calendar step with fewer than ten
        valid calibration values is NaN.

    Raises:
        InvalidArgumentError: If ``distribution`` is neither empirical nor beta, or
            the seam rejects the arguments.
        ValueError: If ``aggregation`` or ``zero_handling`` is not a supported mode.
        CoordinateValidationError: If a DataArray's time dimension is missing,
            non-monotonic, or its periodicity does not match ``periodicity``.

    References:
        Allen, S., & Otero, N. (2023). Standardised indices to monitor energy droughts.
        Renewable Energy, 217, 119206. https://doi.org/10.1016/j.renene.2023.119206
    """
    return apply_standardized(
        _wrapped_srepi,
        values,
        scale=scale,
        data_start_year=data_start_year,
        calibration_year_initial=calibration_year_initial,
        calibration_year_final=calibration_year_final,
        periodicity=periodicity,
        fitting_params=fitting_params,
        distribution=distribution,
        aggregation=aggregation,
        spatial_time_major=spatial_time_major,
        output_scale=output_scale,
        zero_handling=zero_handling,
        pool_window_days=pool_window_days,
    )


@overload
def standardized_demand_index(
    values: xr.DataArray,
    scale: int,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    periodicity: compute.Periodicity | None = None,
    fitting_params: dict[str, Any] | None = None,
    *,
    distribution: indices.Distribution = indices.Distribution.empirical,
    aggregation: compute.Aggregation = "sum",
    output_scale: compute.OutputScale = "normal",
    zero_handling: compute.ZeroHandling = "classic",
    pool_window_days: int = 0,
) -> xr.DataArray: ...


@overload
def standardized_demand_index(
    values: npt.ArrayLike,
    scale: int,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
    fitting_params: dict[str, Any] | None = None,
    *,
    distribution: indices.Distribution = indices.Distribution.empirical,
    aggregation: compute.Aggregation = "sum",
    spatial_time_major: bool = False,
    output_scale: compute.OutputScale = "normal",
    zero_handling: compute.ZeroHandling = "classic",
    pool_window_days: int = 0,
) -> npt.NDArray[np.float64]: ...


def standardized_demand_index(
    values: npt.ArrayLike | xr.DataArray,
    scale: int,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    periodicity: compute.Periodicity | None = None,
    fitting_params: dict[str, Any] | None = None,
    *,
    distribution: indices.Distribution = indices.Distribution.empirical,
    aggregation: compute.Aggregation = "sum",
    spatial_time_major: bool = False,
    output_scale: compute.OutputScale = "normal",
    zero_handling: compute.ZeroHandling = "classic",
    pool_window_days: int = 0,
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Standardized demand index over energy demand or energy degree days (EDD).

    The analogue of :func:`srepi` for demand, as in Allen & Otero (2023). High values
    mean higher-than-usual demand. The index is the climate-driven component of
    demand, not actual load; a degree-day input such as
    :func:`climate_indices.energy.energy_degree_days` is its direct input.

    Windows are accumulated by default (``aggregation="sum"``). The beta distribution
    applies only to a bounded [0, 1] series; see :func:`srepi`. DataArray input takes
    the same per-cell adapter path, registry metadata, and shared ``fitting_params``.

    Args:
        values: 1-D array of demand or degree days, or a DataArray with a time
            dimension plus optional cell dimensions, with the layouts that
            :func:`climate_indices.indices.standardized_index` accepts.
        scale: Number of time steps in each window before standardization.
        data_start_year: Initial year of the input series. Required for NumPy input;
            inferred for DataArray input.
        calibration_year_initial: Initial year of the Calibration Period. Required
            for NumPy input; inferred for DataArray input.
        calibration_year_final: Final year of the Calibration Period. Required for
            NumPy input; inferred for DataArray input.
        periodicity: ``compute.Periodicity.monthly``, ``compute.Periodicity.daily``, or
            ``compute.Periodicity.hourly``. Hourly input carries 8784 values per year and
            supports only ``Distribution.empirical``.
            Required for NumPy input; inferred for DataArray input. Daily NumPy input
            must start on 1 January in the 366-slot year layout; convert a Gregorian
            array with :func:`climate_indices.utils.transform_to_366day`.
        fitting_params: Pre-computed beta fitting parameters; the empirical
            distribution has none.
        distribution: ``Distribution.empirical`` (the default) or ``Distribution.beta``.
        aggregation: How each window is combined, "sum" (the default) or "mean".
        spatial_time_major: Read a time-major block of independent series. Only used
            for NumPy input.
        output_scale: "normal" (the default), "probability", or "bounded".
        zero_handling: Placement of a zero value within the boundary mass (beta only),
            one of ``compute.ZeroHandling``: "classic", "center_of_mass", or "mean_zero".
        pool_window_days: Number of neighbouring days on each side of an hourly
            empirical step whose values join its calibration sample; zero (the default)
            keeps the shipped hour x day-of-year strata. Available only with
            ``Periodicity.hourly`` and ``Distribution.empirical`` (#1407).

    Returns:
        Unitless standardized values: an array for NumPy input, a DataArray carrying
        the ``standardized_demand_index`` registry metadata and the input's
        coordinates for DataArray input. With the empirical distribution, a calendar
        step with fewer than ten valid calibration values is NaN.

    Raises:
        InvalidArgumentError: If ``distribution`` is neither empirical nor beta, or
            the seam rejects the arguments.
        ValueError: If ``aggregation`` or ``zero_handling`` is not a supported mode.
        CoordinateValidationError: If a DataArray's time dimension is missing,
            non-monotonic, or its periodicity does not match ``periodicity``.

    References:
        Allen, S., & Otero, N. (2023). Standardised indices to monitor energy droughts.
        Renewable Energy, 217, 119206. https://doi.org/10.1016/j.renene.2023.119206
    """
    return apply_standardized(
        _wrapped_standardized_demand_index,
        values,
        scale=scale,
        data_start_year=data_start_year,
        calibration_year_initial=calibration_year_initial,
        calibration_year_final=calibration_year_final,
        periodicity=periodicity,
        fitting_params=fitting_params,
        distribution=distribution,
        aggregation=aggregation,
        spatial_time_major=spatial_time_major,
        output_scale=output_scale,
        zero_handling=zero_handling,
        pool_window_days=pool_window_days,
    )
