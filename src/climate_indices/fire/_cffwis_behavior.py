"""CFFWIS behavior indices and the Daily Severity Rating (#804).

ISI, BUI, and the Canadian FWI are concurrent-input derivatives of the
moisture codes, not recurrences: they carry no state, broadcast elementwise,
and have no missing-data policy of their own. Invalid inputs follow the
Fosberg/HDW convention -- NaN with a logged warning rather than a raise. DSR
is the power transform that makes the FWI seasonally averageable.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from climate_indices._recurrence import _as_float_array
from climate_indices.fire._cffwis_codes import (
    _FFMC_COEFFICIENT,
    _FFMC_MAXIMUM,
    _KILOMETERS_PER_HOUR_PER_METER_PER_SECOND,
    _broadcast_elementwise,
    _elementwise_result,
)


def _initial_spread_index(
    ffmc: npt.NDArray[np.float64],
    wind_speed_kilometers_per_hour: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Combine FFMC and wind into ISI (Van Wagner and Pickett, 1985, Eq. 24-26)."""
    moisture = _FFMC_COEFFICIENT * (101.0 - ffmc) / (59.5 + ffmc)
    wind_factor = np.exp(0.05039 * wind_speed_kilometers_per_hour)
    fine_fuel_factor = 91.9 * np.exp(-0.1386 * moisture) * (1.0 + moisture**5.31 / 49300000.0)
    return 0.208 * wind_factor * fine_fuel_factor


def _buildup_index(
    dmc: npt.NDArray[np.float64],
    dc: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Combine DMC and DC into BUI (Van Wagner and Pickett, 1985, Eq. 27)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        # exact-zero branch, as in the cffdrs reference: np.equal keeps the
        # deliberate equality out of the float-comparison lint rule
        combined = np.where(np.equal(dmc, 0.0) & np.equal(dc, 0.0), 0.0, 0.8 * dc * dmc / (dmc + 0.4 * dc))
        weight = np.where(np.equal(dmc, 0.0), 0.0, (dmc - combined) / dmc)
        characteristic = 0.92 + (0.0114 * dmc) ** 1.7
        reduced = np.maximum(dmc - characteristic * weight, 0.0)
    return np.where(combined < dmc, reduced, combined)


def _cffwis_fwi(
    isi: npt.NDArray[np.float64],
    bui: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Combine ISI and BUI into the Canadian FWI (Van Wagner and Pickett, 1985, Eq. 28-30)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        damped = 1000.0 / (25.0 + 108.64 / np.exp(0.023 * bui))
        initial = 0.1 * isi * np.where(bui > 80.0, damped, 0.626 * bui**0.809 + 2.0)
        exponentiated = np.exp(2.72 * (0.434 * np.log(initial)) ** 0.647)
    return np.where(initial <= 1.0, initial, exponentiated)


def _daily_severity_rating(fwi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Transform FWI into DSR (Van Wagner, 1987, Eq. 31)."""
    return 0.0272 * fwi**1.77


def initial_spread_index(
    ffmc: npt.ArrayLike,
    wind_speed_meters_per_second: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Compute the Initial Spread Index (ISI).

    The expected rate of fire spread immediately after ignition, from the
    Fine Fuel Moisture Code and the 10 m wind speed (Van Wagner and Pickett,
    1985). It carries no state and broadcasts elementwise, so any shape works.

    Args:
        ffmc: Fine Fuel Moisture Code, in [0, 101].
        wind_speed_meters_per_second: 10 m wind speed, meters per second,
            non-negative. Converted to the km/h the equations are written in
            here and nowhere else.

    Returns:
        ISI with the broadcast shape of the inputs. NaN where any input is
        NaN or non-finite, where FFMC lies outside [0, 101], or where wind
        speed is negative.

    Raises:
        InvalidArgumentError: If the inputs cannot be broadcast together.
    """
    ffmc_array, wind_array = _broadcast_elementwise(
        "initial_spread_index",
        ("ffmc", "wind_speed_meters_per_second"),
        ffmc,
        wind_speed_meters_per_second,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        wind_kilometers_per_hour = wind_array * _KILOMETERS_PER_HOUR_PER_METER_PER_SECOND
    invalid = (
        ~np.isfinite(ffmc_array)
        | ~np.isfinite(wind_kilometers_per_hour)
        | (ffmc_array < 0.0)
        | (ffmc_array > _FFMC_MAXIMUM)
        | (wind_kilometers_per_hour < 0.0)
    )
    return _elementwise_result(
        "initial_spread_index",
        (ffmc_array, wind_kilometers_per_hour),
        lambda: _initial_spread_index(ffmc_array, wind_kilometers_per_hour),
        invalid=invalid,
        invalid_description="values with FFMC outside [0, 101] or negative wind speed",
    )


def buildup_index(
    dmc: npt.ArrayLike,
    dc: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Compute the Buildup Index (BUI).

    A weighted combination of the Duff Moisture Code and the Drought Code
    (Van Wagner and Pickett, 1985) representing the fuel available for
    spreading. It carries no state and broadcasts elementwise.

    Args:
        dmc: Duff Moisture Code, non-negative.
        dc: Drought Code, non-negative.

    Returns:
        BUI with the broadcast shape of the inputs. NaN where any input is
        NaN or non-finite or where DMC or DC is negative.

    Raises:
        InvalidArgumentError: If the inputs cannot be broadcast together.
    """
    dmc_array, dc_array = _broadcast_elementwise("buildup_index", ("dmc", "dc"), dmc, dc)
    invalid = ~np.isfinite(dmc_array) | ~np.isfinite(dc_array) | (dmc_array < 0.0) | (dc_array < 0.0)
    return _elementwise_result(
        "buildup_index",
        (dmc_array, dc_array),
        lambda: _buildup_index(dmc_array, dc_array),
        invalid=invalid,
        invalid_description="values with negative or non-finite DMC or DC",
    )


def cffwis_fwi(
    isi: npt.ArrayLike,
    bui: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Compute the Canadian Fire Weather Index (FWI).

    The final CFFWIS output, combining the Initial Spread Index and the
    Buildup Index (Van Wagner and Pickett, 1985). ``cffwis_fwi`` is named to
    keep it distinct from the Fosberg index, :func:`fosberg_ffwi`; there is no
    ``fwi()``. It carries no state and broadcasts elementwise.

    Args:
        isi: Initial Spread Index, non-negative.
        bui: Buildup Index, non-negative.

    Returns:
        FWI with the broadcast shape of the inputs. NaN where any input is
        NaN or non-finite, or negative.

    Raises:
        InvalidArgumentError: If the inputs cannot be broadcast together.
    """
    isi_array, bui_array = _broadcast_elementwise("cffwis_fwi", ("isi", "bui"), isi, bui)
    invalid = ~np.isfinite(isi_array) | ~np.isfinite(bui_array) | (isi_array < 0.0) | (bui_array < 0.0)
    return _elementwise_result(
        "cffwis_fwi",
        (isi_array, bui_array),
        lambda: _cffwis_fwi(isi_array, bui_array),
        invalid=invalid,
        invalid_description="values with negative or non-finite ISI or BUI",
    )


def daily_severity_rating(cffwis_fwi: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Compute the Daily Severity Rating (DSR).

    A power transform of the Canadian FWI that makes seasonal averaging
    meaningful (Van Wagner, 1987), the form used for climatological work.
    It carries no state and broadcasts elementwise.

    Args:
        cffwis_fwi: Canadian FWI, non-negative. The parameter keeps the
            design-doc name so the transform is unambiguous about which FWI
            it consumes.

    Returns:
        DSR with the broadcast shape of the input. NaN where the FWI is NaN,
        non-finite, or negative.
    """
    fwi_array = _as_float_array(cffwis_fwi)
    invalid = ~np.isfinite(fwi_array) | (fwi_array < 0.0)
    return _elementwise_result(
        "daily_severity_rating",
        (fwi_array,),
        lambda: _daily_severity_rating(fwi_array),
        invalid=invalid,
        invalid_description="values with negative or non-finite FWI",
    )
