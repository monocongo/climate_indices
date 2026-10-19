"""Temperature-derated photovoltaic resource for a fixed reference configuration."""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt

from climate_indices._recurrence import _as_float_array
from climate_indices.exceptions import InvalidArgumentError
from climate_indices.logging_config import get_logger, log_calculation_failure
from climate_indices.performance import check_large_array_memory

_logger = get_logger(__name__)


def pv_capacity_factor(
    ssrd: npt.ArrayLike,
    temperature: npt.ArrayLike,
    wind_speed: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Calculate a temperature-derated photovoltaic (PV) capacity factor.

    This generic reference configuration describes the climate-driven component
    of solar supply, not actual generation or a plant model. Jerez et al. (2015),
    Methods equations 1–3, use monocrystalline silicon with these fixed terms::

        Tcell = 4.3 + 0.943 * temperature + 0.028 * ssrd - 1.528 * wind_speed
        PR = 1 - 0.005 * (Tcell - 25)
        capacity_factor = PR * ssrd / 1000

    The cell-temperature coefficients come from Chenni et al. (2007).
    Standard test conditions are 1000 W m-2 and a *cell* temperature of 25 °C,
    not an air temperature of 25 °C. Clipping to [0, 1] is project policy;
    Jerez's unclipped potential can exceed one. This model does not include
    panel tilt, irradiance transposition, optical losses, or inverter losses.
    Use pvlib for configuration-specific modelling.

    Args:
        ssrd: Surface downwelling solar radiation as a mean flux in W m-2.
            For accumulated ERA5 SSRD in J m-2, divide each interval's energy
            by its actual duration in seconds before calling. Do not divide
            data already expressed in W m-2. Align all three inputs to the
            same intervals before calling; the model is nonlinear, so convert
            at the native sampling interval before temporal averaging.
        temperature: Near-surface (2 m) air temperature in °C.
        wind_speed: Surface (10 m) wind speed in m s-1, not hub-height wind.
            All inputs must broadcast together. NaN and masked values are missing.

    Returns:
        A float64 NumPy array in the broadcast shape, including a zero-dimensional
        array for scalar inputs. Valid zero radiation gives zero. Negative
        radiation or wind, non-finite inputs, and overflowing results give NaN.
        The function does not change its inputs. DataArray dispatch is not provided.

    Raises:
        InputTypeError: If an input is non-numeric.
        InvalidArgumentError: If the input shapes cannot broadcast together.

    References:
        Jerez et al. (2015), Nature Communications 6, 10014.
        https://doi.org/10.1038/ncomms10014
        Chenni et al. (2007), Energy 32, 1724–1730.
        https://doi.org/10.1016/j.energy.2006.12.006
    """
    log = _logger.bind(index_type="pv_capacity_factor")
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics: dict[str, float] = {}
    try:
        try:
            radiation, air, wind = np.broadcast_arrays(
                _as_float_array(ssrd), _as_float_array(temperature), _as_float_array(wind_speed)
            )
        except ValueError as exc:
            raise InvalidArgumentError("ssrd, temperature, and wind_speed must broadcast together.") from exc
        log = log.bind(input_shape=radiation.shape, input_elements=radiation.size)
        # Budget for cell temperature, arithmetic and output buffers, and temporary masks.
        memory_metrics = (
            check_large_array_memory(radiation, air, wind, extra_bytes=radiation.size * (5 * radiation.itemsize + 3))
            or {}
        )
        valid = np.isfinite(radiation) & np.isfinite(air) & np.isfinite(wind) & (radiation >= 0) & (wind >= 0)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            cell = 4.3 + 0.943 * air + 0.028 * radiation - 1.528 * wind
            result = (radiation / 1000) * (1 - 0.005 * (cell - 25))
            result = np.where(radiation == 0, 0.0, result)
        result = np.asarray(np.where(valid & np.isfinite(result), np.clip(result, 0, 1), np.nan), dtype=np.float64)
        log.info(
            "calculation_completed",
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            output_shape=result.shape,
            **memory_metrics,
        )
        return result
    except Exception as exc:
        log_calculation_failure(log, exc, duration_ms=round((time.perf_counter() - t0) * 1000.0, 2), **memory_metrics)
        raise
