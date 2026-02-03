"""Compute palmer drought indices."""

import calendar
from collections.abc import Callable
from typing import Any, TypeVar, cast

import numba
import numpy as np

from climate_indices import eto, logging_config, utils

# Retrieve logger and set desired logging level
_logger = logging_config.get_logger(__name__)

# declare the function names that should be included in the public API for this module
__all__ = ["pdsi"]

AWCTOP = 1.0
K8_SIZE = 40
PHI = np.array(
    [
        -0.3865982,
        -0.2316132,
        -0.0378180,
        0.1715539,
        0.3458803,
        0.4308320,
        0.3916645,
        0.2452467,
        0.0535511,
        -0.15583436,
        -0.3340551,
        -0.4310691,
    ],
    dtype=float,
)
DAYS_IN_MONTH = np.array([31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0])
_CALIBRATION_CLIMATOLOGY_MIN_FRACTION = 0.5

_F = TypeVar("_F", bound=Callable[..., Any])


def _njit(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:
    def decorator(func: _F) -> _F:
        return cast(_F, numba.njit(*args, **kwargs)(func))

    return decorator


@_njit(cache=True)
def _get_awc_bot(awc: float, wctop: float) -> float:
    """
    Calculate available water capcity in bottom layer

    :param awc: available water capacity (total), in inches
    :param wctop: available water capacity (surface layer), in inches
    :return available water capacity (under layer), in inches
    :rtype: float
    """
    return max(awc - wctop, 0.0)


@_njit(cache=True)
def _calc_potential_loss(
    pet: float,
    ss: float,
    su: float,
    awc: float,
    wctop: float,
) -> float:
    """
    Calculate potential loss

    :param pet: potential evapotranspiration
    :param ss: surface layer water content, in inches
    :param su: under layer water content, in inches
    :param awc: available water capacity (total), in inches
    :param wctop: available water capacity (surface layer), in inches
    :return potential loss
    :rtype: float
    """
    awc_bot = _get_awc_bot(awc, wctop)
    if ss >= pet:
        return pet
    return min(ss + su, ((pet - ss) * su) / (awc_bot + wctop) + ss)


@_njit(cache=True)
def _calc_recharge(
    p: float,
    pet: float,
    ss: float,
    su: float,
    awc: float,
    wctop: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Calculate recharge, runoff, residual moisture, loss
    to both surface and under layers

    Depends on the starting moisture content and values of
    precipitation and evaporation.

    :param p: preciptiation, in inches
    :param pet: potential evapotranspiration
    :param ss: surface layer water content, in inches
    :param su: under layer water content, in inches
    :param awc: available water capacity (total), in inches
    :param wctop: available water capacity (surface layer), in inches
    :return a tuple of floats
        - et: evapotranspiration
        - tl: total loss
        - r: recharge
        - ro: runoff
        - sss: surface layer water content, in inches
        - ssu: under layer water content, in inches
    """
    awc_bot = _get_awc_bot(awc, wctop)

    # precipitation exceeds potential evaporation
    if p >= pet:
        et = pet
        tl = 0.0

        # excess precipitation recharges under layer as well as upper
        if (p - pet) > (wctop - ss):
            rs = wctop - ss
            sss = wctop

            # both layers can take the entire excess
            if (p - pet - rs) < (awc_bot - su):
                ru = p - pet - rs
                ro = 0.0

            # some runoff occurs
            else:
                ru = awc_bot - su
                ro = p - pet - rs - ru

            ssu = su + ru
            r = rs + ru

        # only top layer recharged
        else:
            r = p - pet
            sss = ss + p - pet
            ssu = su
            ro = 0.0
    # evaporation exceeds precipitation
    else:
        r = 0.0

        # evaporation from surface layer only
        if ss >= (pet - p):
            sl = pet - p
            sss = ss - sl
            ul = 0.0
            ssu = su

        # evaporation from both layers
        else:
            sl = ss
            sss = 0.0
            ul = min(su, (pet - p - sl) * su / awc)
            ssu = su - ul

        tl = sl + ul
        ro = 0.0
        et = p + sl + ul

    return et, tl, r, ro, sss, ssu


def _as_float_array(values: np.ndarray | np.ma.MaskedArray) -> np.ndarray:
    """
    Convert masked arrays to floats with NaNs for missing values.

    Args:
        values: Array-like input values.

    Returns:
        Array of floats with NaNs for missing values.
    """
    if np.ma.isMaskedArray(values):
        masked = cast(np.ma.MaskedArray, values)
        return np.asarray(masked.filled(np.nan), dtype=float)
    return np.asarray(values, dtype=float)


def _validate_monthly_climatology(values: np.ndarray | None, name: str) -> np.ndarray | None:
    """
    Validate a 12-month climatology array.

    Args:
        values: Candidate monthly climatology values.
        name: Name used in error messages.

    Returns:
        A flattened 12-element array or None.

    Raises:
        ValueError: If the array does not contain 12 values.
    """
    if values is None:
        return None
    flattened = np.asarray(values, dtype=float).reshape(-1)
    if flattened.size != 12:
        raise ValueError(f"{name} must contain 12 monthly values, got {flattened.size}.")
    return flattened


def _has_sufficient_climatology_data(
    values_2d: np.ndarray,
    min_fraction: float,
) -> bool:
    """
    Determine whether a calibration window has enough data for climatology.

    Args:
        values_2d: Monthly values shaped (years, 12).
        min_fraction: Minimum fraction of available values per month.

    Returns:
        True if each month meets the minimum data requirement.
    """
    if values_2d.size == 0:
        return False
    min_required = max(1, int(np.ceil(values_2d.shape[0] * min_fraction)))
    counts = np.sum(~np.isnan(values_2d), axis=0)
    return bool(np.all(counts >= min_required))


def _compute_monthly_climatology(
    values: np.ndarray,
    *,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    min_coverage_fraction: float = _CALIBRATION_CLIMATOLOGY_MIN_FRACTION,
) -> np.ndarray:
    """
    Compute monthly climatology from a monthly series.

    Args:
        values: Monthly time series.
        data_start_year: Start year of the series (January start).
        calibration_year_initial: Start year of the calibration window.
        calibration_year_final: End year of the calibration window.
        min_coverage_fraction: Minimum fraction of available values per month.

    Returns:
        12-element climatology array.
    """
    values_2d = utils.reshape_to_2d(values, 12)
    if data_start_year is None or calibration_year_initial is None or calibration_year_final is None:
        return np.asarray(np.nanmean(values_2d, axis=0), dtype=float)

    start_idx = max(calibration_year_initial - data_start_year, 0)
    end_idx = min(calibration_year_final - data_start_year, values_2d.shape[0] - 1)
    if start_idx > end_idx:
        return np.asarray(np.nanmean(values_2d, axis=0), dtype=float)

    window = values_2d[start_idx : end_idx + 1]
    if not _has_sufficient_climatology_data(window, min_coverage_fraction):
        return np.asarray(np.nanmean(values_2d, axis=0), dtype=float)
    return np.asarray(np.nanmean(window, axis=0), dtype=float)


def _fill_missing_with_climatology(
    values: np.ndarray,
    climatology: np.ndarray,
    original_length: int,
) -> np.ndarray:
    """
    Fill NaN values with corresponding monthly climatology values.

    Args:
        values: Monthly values to fill.
        climatology: 12-month climatology.
        original_length: Original length for trimming padding.

    Returns:
        Filled array with original length.
    """
    values_2d = utils.reshape_to_2d(values, 12)
    for year in range(values_2d.shape[0]):
        for month in range(12):
            if np.isnan(values_2d[year, month]):
                values_2d[year, month] = climatology[month]
    return values_2d.reshape(-1)[0:original_length]


def _normalize_latitude(latitude_degrees: float | np.ndarray | None) -> float | None:
    """
    Normalize a latitude input to a scalar float.

    Args:
        latitude_degrees: Latitude value(s) in degrees north.

    Returns:
        Scalar latitude or None if not provided.
    """
    if latitude_degrees is None:
        return None
    if isinstance(latitude_degrees, np.ndarray):
        if latitude_degrees.size == 0:
            return None
        return float(latitude_degrees.flat[0])
    return float(latitude_degrees)


def _is_leap_year(year: int, rule: str) -> bool:
    """
    Determine leap-year status using the requested rule.

    Args:
        year: Year to check.
        rule: "noaa" for divisible-by-4 only, or "gregorian".

    Returns:
        True if leap year based on rule.

    Raises:
        ValueError: If the rule is not supported.
    """
    rule_normalized = rule.strip().lower()
    if rule_normalized == "noaa":
        return (year % 4) == 0
    if rule_normalized == "gregorian":
        return calendar.isleap(year)
    raise ValueError(f"Unsupported leap_year_rule: {rule}")


def _compute_pet_fortran(
    temps_fahrenheit: np.ndarray,
    latitude_degrees: float | np.ndarray | None,
    b: float,
    h: float,
    tla: float | None,
    data_start_year: int,
    leap_year_rule: str,
    unit_scale: float,
) -> np.ndarray:
    """
    Compute PET using the Fortran (pdi.f) formulation.

    Args:
        temps_fahrenheit: Monthly mean temperatures in degrees Fahrenheit.
        latitude_degrees: Latitude in degrees north, used to derive TLA if not provided.
        b: Soil constant B (from NOAA soil constants).
        h: Soil constant H (from NOAA soil constants).
        tla: Negative tangent of latitude; derived from latitude if None.
        data_start_year: Initial year of the input series.
        leap_year_rule: "noaa" or "gregorian".
        unit_scale: Optional scale factor to apply to PET values.

    Returns:
        Monthly PET values in the Fortran units scaled by unit_scale.

    Raises:
        ValueError: If latitude is missing and tla is not provided.
    """
    latitude = _normalize_latitude(latitude_degrees)
    if tla is None:
        if latitude is None:
            raise ValueError("latitude_degrees is required to compute TLA for Fortran PET.")
        tla_value = -np.tan(np.deg2rad(latitude))
    else:
        tla_value = float(tla)

    temps_2d = utils.reshape_to_2d(temps_fahrenheit, 12)
    pet = np.full(temps_2d.shape, np.nan)

    for year_idx in range(temps_2d.shape[0]):
        year = data_start_year + year_idx
        is_leap = _is_leap_year(year, leap_year_rule)
        for month in range(12):
            t = temps_2d[year_idx, month]
            if np.isnan(t) or t <= 32.0:
                pet[year_idx, month] = 0.0
                continue

            dum = PHI[month] * tla_value
            with np.errstate(divide="ignore", invalid="ignore"):
                dk = np.arctan(np.sqrt(1.0 - dum * dum) / dum)
            if dk < 0.0:
                dk = np.pi + dk
            dk = (dk + 0.0157) / 1.57

            if t >= 80.0:
                pe = (np.sin(t / 57.3 - 0.166) - 0.76) * dk
            else:
                dum = np.log(t - 32.0)
                pe = np.exp(-3.863233 + b * 1.715598 - b * np.log(h) + b * dum) * dk

            days = DAYS_IN_MONTH[month] + (1.0 if (month == 1 and is_leap) else 0.0)
            pet[year_idx, month] = pe * days * unit_scale

    return pet.reshape(-1)[0 : temps_fahrenheit.size]


def _aggregate_daily_to_monthly(
    values: np.ndarray,
    data_start_year: int,
    leap_year_rule: str,
) -> np.ndarray:
    """
    Aggregate daily values into monthly totals.

    Args:
        values: Daily values for consecutive years.
        data_start_year: Initial year of the daily series.
        leap_year_rule: "noaa" or "gregorian".

    Returns:
        Monthly totals as a 1-D array.

    Raises:
        ValueError: If the daily series length is not aligned to full months.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("Daily values must be a 1-D array.")

    monthly_totals: list[float] = []
    index = 0
    year = data_start_year
    while index < values.size:
        is_leap = _is_leap_year(year, leap_year_rule)
        days_per_month = DAYS_IN_MONTH.copy()
        if is_leap:
            days_per_month[1] = 29.0
        for days in days_per_month:
            days_int = int(days)
            end = index + days_int
            if end > values.size:
                raise ValueError("Daily values length does not align with full months.")
            month_values = values[index:end]
            if np.all(np.isnan(month_values)):
                monthly_totals.append(np.nan)
            else:
                monthly_totals.append(float(np.nansum(month_values)))
            index = end
        year += 1

    return np.asarray(monthly_totals)


def _resolve_pet_source(
    pet_source: str,
    pet: np.ndarray | None,
    temperature_celsius: np.ndarray | None,
    latitude_degrees: float | np.ndarray | None,
    data_start_year: int,
    fortran_b: float | None,
    fortran_h: float | None,
    fortran_tla: float | None,
    fortran_unit_scale: float,
    hargreaves_tmin_celsius: np.ndarray | None,
    hargreaves_tmax_celsius: np.ndarray | None,
    hargreaves_tmean_celsius: np.ndarray | None,
    leap_year_rule: str,
) -> np.ndarray:
    """
    Resolve PET based on the configured source.

    Args:
        pet_source: "input", "thornthwaite", "fortran", or "hargreaves".
        pet: Precomputed PET values if provided.
        temperature_celsius: Monthly temperatures for Thornthwaite/Fortran PET.
        latitude_degrees: Latitude used in PET calculations.
        data_start_year: Initial year of the series.
        fortran_b: Fortran soil constant B.
        fortran_h: Fortran soil constant H.
        fortran_tla: Negative tangent of latitude for Fortran PET.
        fortran_unit_scale: Scale factor for Fortran PET output.
        hargreaves_tmin_celsius: Daily Tmin values for Hargreaves.
        hargreaves_tmax_celsius: Daily Tmax values for Hargreaves.
        hargreaves_tmean_celsius: Daily Tmean values for Hargreaves.
        leap_year_rule: "noaa" or "gregorian".

    Returns:
        Monthly PET values.

    Raises:
        ValueError: If required inputs are missing.
    """
    source = pet_source.strip().lower()
    if source == "input":
        if pet is None:
            raise ValueError("PET input is required when pet_source is 'input'.")
        return pet

    if source == "thornthwaite":
        if temperature_celsius is None:
            raise ValueError("temperature_celsius is required for Thornthwaite PET.")
        latitude = _normalize_latitude(latitude_degrees)
        if latitude is None:
            raise ValueError("latitude_degrees is required for Thornthwaite PET.")
        return eto.eto_thornthwaite(temperature_celsius, latitude, data_start_year)

    if source == "fortran":
        if temperature_celsius is None:
            raise ValueError("temperature_celsius is required for Fortran PET.")
        if fortran_b is None or fortran_h is None:
            raise ValueError("fortran_b and fortran_h are required for Fortran PET.")
        temps_fahrenheit = (np.asarray(temperature_celsius, dtype=float) * 9.0 / 5.0) + 32.0
        return _compute_pet_fortran(
            temps_fahrenheit=temps_fahrenheit,
            latitude_degrees=latitude_degrees,
            b=fortran_b,
            h=fortran_h,
            tla=fortran_tla,
            data_start_year=data_start_year,
            leap_year_rule=leap_year_rule,
            unit_scale=fortran_unit_scale,
        )

    if source == "hargreaves":
        if hargreaves_tmin_celsius is None or hargreaves_tmax_celsius is None or hargreaves_tmean_celsius is None:
            raise ValueError("Daily Tmin/Tmax/Tmean arrays are required for Hargreaves PET.")
        latitude = _normalize_latitude(latitude_degrees)
        if latitude is None:
            raise ValueError("latitude_degrees is required for Hargreaves PET.")
        daily_pet = eto.eto_hargreaves(
            hargreaves_tmin_celsius,
            hargreaves_tmax_celsius,
            hargreaves_tmean_celsius,
            latitude,
        )
        return _aggregate_daily_to_monthly(daily_pet, data_start_year, leap_year_rule)

    raise ValueError(f"Unsupported pet_source: {pet_source}")


def _calc_alpha(data: dict) -> None:
    """
    Calculate alpha parameters

    :param data: dictionary of parameters (intialized in pdsi)
    """
    data["alpha"] = np.zeros(data["petsum"].shape)
    for idx, pet in enumerate(data["petsum"]):
        if pet != 0:
            data["alpha"][idx] = data["etsum"][idx] / pet
        elif data["etsum"][idx] == 0:
            data["alpha"][idx] = 1.0


def _calc_beta(data: dict) -> None:
    """
    Calculate beta parameters

    :param data: dictionary of parameters (intialized in pdsi)
    """
    data["beta"] = np.zeros(data["prsum"].shape)
    for idx, pr in enumerate(data["prsum"]):
        if pr != 0:
            data["beta"][idx] = data["rsum"][idx] / pr
        elif data["rsum"][idx] == 0:
            data["beta"][idx] = 1.0


def _calc_gamma(data: dict) -> None:
    """
    Calculate gamma parameters

    :param data: dictionary of parameters (intialized in pdsi)
    """
    data["gamma"] = np.zeros(data["spsum"].shape)
    for idx, sp in enumerate(data["spsum"]):
        if sp != 0:
            data["gamma"][idx] = data["rosum"][idx] / sp
        elif data["rosum"][idx] == 0:
            data["gamma"][idx] = 1.0


def _calc_delta(data: dict) -> None:
    """
    Calculate delta parameters

    :param data: dictionary of parameters (intialized in pdsi)
    """
    data["delta"] = np.zeros(data["plsum"].shape)
    for idx, pl in enumerate(data["plsum"]):
        if pl != 0:
            data["delta"][idx] = data["tlsum"][idx] / pl


@_njit(cache=True)
def _calc_water_balances_numba(
    precips: np.ndarray,
    pet: np.ndarray,
    awc: float,
    awc_bot: float,
    wctop: float,
    n_years: int,
    calibration_year_initial_idx: int,
    calibration_year_final_idx: int,
    psum: np.ndarray,
    spsum: np.ndarray,
    petsum: np.ndarray,
    plsum: np.ndarray,
    prsum: np.ndarray,
    rsum: np.ndarray,
    tlsum: np.ndarray,
    etsum: np.ndarray,
    rosum: np.ndarray,
    spdat: np.ndarray,
    pldat: np.ndarray,
    prdat: np.ndarray,
    rdat: np.ndarray,
    tldat: np.ndarray,
    etdat: np.ndarray,
    rodat: np.ndarray,
    sssdat: np.ndarray,
    ssudat: np.ndarray,
) -> None:
    ss = wctop
    su = awc_bot
    for year in range(n_years):
        for month in range(12):
            p = precips[year, month]
            pet_value = pet[year, month]
            sp = ss + su
            pr = awc_bot + wctop - sp

            # Get potential loss
            pl = _calc_potential_loss(pet_value, ss, su, awc, wctop)

            # Calculate recharge, runoff, residual moisture, loss to both
            # surface and under layers, depending on starting moisture
            # content and values of precipitation and evaporation
            et, tl, r, ro, sss, ssu = _calc_recharge(p, pet_value, ss, su, awc, wctop)

            # update sums
            if calibration_year_initial_idx <= year <= calibration_year_final_idx:
                psum[month] += p
                spsum[month] += sp
                petsum[month] += pet_value
                plsum[month] += pl
                prsum[month] += pr
                rsum[month] += r
                tlsum[month] += tl
                etsum[month] += et
                rosum[month] += ro

            # set data
            spdat[year, month] = sp
            pldat[year, month] = pl
            prdat[year, month] = pr
            rdat[year, month] = r
            tldat[year, month] = tl
            etdat[year, month] = et
            rodat[year, month] = ro
            sssdat[year, month] = sss
            ssudat[year, month] = ssu

            # update soil moisture
            ss = sss
            su = ssu


def _calc_water_balances(data: dict) -> None:
    """
    Perform water balance calculations

    :param data: dictionary of parameters (intialized in pdsi)
    """
    _calc_water_balances_numba(
        data["precips"],
        data["pet"],
        data["awc"],
        data["awc_bot"],
        data["wctop"],
        data["n_years"],
        data["calibration_year_initial_idx"],
        data["calibration_year_final_idx"],
        data["psum"],
        data["spsum"],
        data["petsum"],
        data["plsum"],
        data["prsum"],
        data["rsum"],
        data["tlsum"],
        data["etsum"],
        data["rosum"],
        data["spdat"],
        data["pldat"],
        data["prdat"],
        data["rdat"],
        data["tldat"],
        data["etdat"],
        data["rodat"],
        data["sssdat"],
        data["ssudat"],
    )


def _calc_cafec_coefficients(data: dict) -> None:
    """
    Calculate CAFEC Coefficients

    :param data: dictionary of parameters (intialized in pdsi)
    """
    _calc_alpha(data)
    _calc_beta(data)
    _calc_gamma(data)
    _calc_delta(data)


def _calc_zindex_factors(data: dict) -> None:
    """
    Calculate Z-Index weighting factors (variable AK)

    trat is the 'T' ratio of average moisture demand
    to average moisture supply in month M

    :param data: dictionary of parameters (intialized in pdsi)
    """
    data["trat"] = (data["petsum"] + data["rsum"] + data["rosum"]) / (data["psum"] + data["tlsum"])


def _avg_calibration_sums(data: dict) -> None:
    """
    Average the sums over the calibration period

    :param data: dictionary of parameters (intialized in pdsi)
    """
    n_calb_years = data["n_calb_years"]
    data["psum"] = data["psum"] / n_calb_years
    data["spsum"] = data["spsum"] / n_calb_years
    data["petsum"] = data["petsum"] / n_calb_years
    data["plsum"] = data["plsum"] / n_calb_years
    data["prsum"] = data["prsum"] / n_calb_years
    data["rsum"] = data["rsum"] / n_calb_years
    data["tlsum"] = data["tlsum"] / n_calb_years
    data["etsum"] = data["etsum"] / n_calb_years
    data["rosum"] = data["rosum"] / n_calb_years


def _calc_kfactors(data: dict) -> None:
    """
    Calculate K Factors

    Reread monthly parameters for calculation of the 'K' monthly
    weighting factors used in z-index calculation

    :param data: dictionary of parameters (intialized in pdsi)
    """
    sabsd = np.zeros((12,))
    for year in range(data["calibration_year_initial_idx"], data["calibration_year_final_idx"] + 1):
        for month in range(12):
            phat = (
                data["alpha"][month] * data["pet"][year, month]
                + data["beta"][month] * data["prdat"][year, month]
                + data["gamma"][month] * data["spdat"][year, month]
                - data["delta"][month] * data["pldat"][year, month]
            )
            d = data["precips"][year, month] - phat
            sabsd[month] += abs(d)

    dbar = sabsd / data["n_calb_years"]
    akhat = 1.5 * np.log10((data["trat"] + 2.8) / dbar) + 0.5
    swtd = np.sum(dbar * akhat)
    data["ak"] = 17.67 * akhat / swtd


def _case(prob: float, x1: float, x2: float, x3: float) -> float:
    """
    Select the preliminary (or near-real time) PDSI

    Selects the PDSI from the given x values
    defined below and the probability (prob) of ending either a
    drought or wet spell.

    :param prob: the probability of ending either a drought
                 or wet spell
    :param x1: Index for incipient wet spells (always positive)
    :param x2: Index for incipient dry spells (always negative)
    :param x3: severity index for an established wet spell (positive)
               or drought (negative)
    :returns the selected pdsi (either preliminary or final)
    :rtype: float
    """

    # if x3 = 0 the index is near normal and either a dry or wet spell
    # exists. Choose the largest absolute value of x1 or x2
    if x3 == 0:
        if abs(x1) > abs(x2):
            return x1
        return x2

    # A weather spell is established and palm = x3 is final
    if (prob <= 0) or (prob >= 100):
        return x3

    pro = prob / 100
    if x3 <= 0:
        return (1.0 - pro) * x3 + pro * x1

    return (1.0 - pro) * x3 + pro * x2


def _assign(data: dict) -> None:
    """
    Assign x values

    :param data: dictionary of parameters (intialized in pdsi)
    """
    year = data["year"]
    month = data["month"]
    data["sx"][data["k8"]] = data["x"][year, month]
    isave = data["iass"]
    if data["k8"] == 0:
        data["pdsi"][year, month] = data["x"][year, month]
        data["phdi"][year, month] = data["px3"][year, month]
        if data["px3"][year, month] == 0:
            data["phdi"][year, month] = data["x"][year, month]

        data["wplm"][year, month] = _case(
            data["ppr"][year, month],
            data["px1"][year, month],
            data["px2"][year, month],
            data["px3"][year, month],
        )
        return

    # use all x3 values
    if data["iass"] == 3:
        for i in range(data["k8"]):
            data["sx"][i] = data["sx3"][i]

    # backtrack through arrays, storing assigned x1 (or x2)
    # in sx until it is zero, then switching to the other until
    # it is zero, etc
    else:
        for i in range(data["k8"] - 1, -1, -1):
            if isave == 2:
                if data["sx2"][i] == 0:
                    isave = 1
                    data["sx"][i] = data["sx1"][i]
                else:
                    isave = 2
                    data["sx"][i] = data["sx2"][i]
            else:
                if data["sx1"][i] == 0:
                    isave = 2
                    data["sx"][i] = data["sx2"][i]
                else:
                    isave = 1
                    data["sx"][i] = data["sx1"][i]

    # proper assignments to array sx have been made, output the mess
    for idx in range(data["k8"] + 1):
        j = int(data["indexj"][idx])
        m = int(data["indexm"][idx])
        data["pdsi"][j, m] = data["sx"][idx]
        data["phdi"][j, m] = data["px3"][j, m]

        if data["px3"][j, m] == 0:
            data["phdi"][j, m] = data["sx"][idx]

        data["wplm"][j, m] = _case(
            data["ppr"][j, m],
            data["px1"][j, m],
            data["px2"][j, m],
            data["px3"][j, m],
        )
    data["k8"] = 0
    # data["k8max"] = 0


def _statement_220(data: dict) -> None:
    """
    Save this month's calculated variables (v,pro,x1,x2,x3) for
    use with next month's data

    Translated from statement 220 in NCEI's pdi.f

    :param data: dictionary of parameters (intialized in pdsi)
    """
    year = data["year"]
    month = data["month"]
    data["v"] = data["pv"]
    data["pro"] = data["ppr"][year, month]
    data["x1"] = data["px1"][year, month]
    data["x2"] = data["px2"][year, month]
    data["x3"] = data["px3"][year, month]


def _statement_210(data: dict) -> None:
    """
    prob(end) returns to 0. A possible abatement has fizzled out,
    so we accept all stored values of x3

    Translated from statement 210 in NCEI's pdi.f

    :param data: dictionary of parameters (intialized in pdsi)
    """
    year = data["year"]
    month = data["month"]
    data["pv"] = 0.0
    data["px1"][year, month] = 0.0
    data["px2"][year, month] = 0.0
    data["ppr"][year, month] = 0.0
    data["px3"][year, month] = 0.897 * data["x3"] + data["z"][year, month] / 3.0
    data["x"][year, month] = data["px3"][year, month]

    if data["k8"] == 0:
        data["pdsi"][year, month] = data["x"][year, month]
        data["phdi"][year, month] = data["px3"][year, month]
        if data["px3"][year, month] == 0:
            data["phdi"][year, month] = data["x"][year, month]
        data["wplm"][year, month] = _case(
            data["ppr"][year, month],
            data["px1"][year, month],
            data["px2"][year, month],
            data["px3"][year, month],
        )
    else:
        data["iass"] = 3
        _assign(data)

    _statement_220(data)


def _statement_200(data: dict) -> None:
    """
    Continue x1 and x2 calculations
    if either indicates the start of a new wet or drought,
    and if the last wet or drought has ended, use x1 or x2
    as the new x3

    Translated from statement 200 in NCEI's pdi.f

    :param data: dictionary of parameters (intialized in pdsi)
    """
    year = data["year"]
    month = data["month"]
    data["px1"][year, month] = max(0, 0.897 * data["x1"] + data["z"][year, month] / 3.0)

    # if no existing wet spell or drought
    # x1 becomes the new x3
    if (data["px1"][year, month] >= 1) and (data["px3"][year, month] == 0):
        data["x"][year, month] = data["px1"][year, month]
        data["px3"][year, month] = data["px1"][year, month]
        data["px1"][year, month] = 0
        data["iass"] = 1
        _assign(data)
        _statement_220(data)
        return

    data["px2"][year, month] = min(0.0, 0.897 * data["x2"] + data["z"][year, month] / 3.0)

    # if no existing wet spell or drought x2 becomes the new x3
    if (data["px2"][year, month] <= -1) and (data["px3"][year, month] == 0):
        data["x"][year, month] = data["px2"][year, month]
        data["px3"][year, month] = data["px2"][year, month]
        data["px2"][year, month] = 0.0
        data["iass"] = 2
        _assign(data)
        _statement_220(data)
        return

    # No established drought (wet spell), but x3 = 0
    # so either (nonzero) x1 or x2 must be used as x3
    if data["px3"][year, month] == 0:
        if data["px1"][year, month] == 0:
            data["x"][year, month] = data["px2"][year, month]
            data["iass"] = 2
            _assign(data)
            _statement_220(data)
            return

        if data["px2"][year, month] == 0:
            data["x"][year, month] = data["px1"][year, month]
            data["iass"] = 1
            _assign(data)
            _statement_220(data)
            return

    # at this point there is no determed value to assign to x,
    # all the values of x1, x2, and x3 are saved. Ata a later
    # time x3 will reach a value where it is the value of x (pdsi).
    # At that time, the assign method backtracs through choosing
    # the appropriate x1 or x2 to be that month's x.
    # _logger.debug(f"no value assigned; will backtrack later k8:{data['k8']},y:{year},m:{month}")
    if data["k8"] >= data["sx"].shape[0] + 1:
        vals = [0] * (data["k8"] - data["sx"].shape[0] + 2)
        data["sx"] = np.append(data["sx"], vals)
        data["sx1"] = np.append(data["sx1"], vals)
        data["sx2"] = np.append(data["sx2"], vals)
        data["sx3"] = np.append(data["sx3"], vals)
        data["indexj"] = np.append(data["indexj"], vals)
        data["indexm"] = np.append(data["indexm"], vals)

    data["sx1"][data["k8"]] = data["px1"][year, month]
    data["sx2"][data["k8"]] = data["px2"][year, month]
    data["sx3"][data["k8"]] = data["px3"][year, month]
    data["x"][year, month] = data["px3"][year, month]
    data["k8"] += 1
    data["k8max"] = data["k8"]

    _statement_220(data)


def _statement_190(data: dict) -> None:
    """
    drought or wet continues, calculate prob(end) (variable ze)

    Translated from statement 190 in NCEI's pdi.f

    :param data: dictionary of parameters (intialized in pdsi)
    """
    year = data["year"]
    month = data["month"]
    if data["pro"] == 100:
        q = data["ze"]
    else:
        q = data["ze"] + data["v"]

    data["ppr"][year, month] = (data["pv"] / q) * 100

    if data["ppr"][year, month] >= 100:
        data["ppr"][year, month] = 100
        data["px3"][year, month] = 0
    else:
        data["px3"][year, month] = 0.897 * data["x3"] + data["z"][year, month] / 3.0

    _statement_200(data)


def _statement_180(data: dict) -> None:
    """
    drought abatement is possible

    Translated from statement 180 in NCEI's pdi.f

    :param data: dictionary of parameters (intialized in pdsi)
    """
    year = data["year"]
    month = data["month"]
    data["uw"] = data["z"][year, month] + 0.15
    data["pv"] = data["uw"] + max(data["v"], 0.0)

    # During a drought, PV <= 0 implies prob(end) has returned to 0
    if data["pv"] <= 0:
        _statement_210(data)
        return

    data["ze"] = -2.691 * data["x3"] - 1.5
    _statement_190(data)


def _statement_170(data: dict) -> None:
    """
    Wet spell abatement is possible

    Translated from statement 170 in NCEI's pdi.f

    :param data: dictionary of parameters (intialized in pdsi)
    """
    year = data["year"]
    month = data["month"]
    data["ud"] = data["z"][year, month] - 0.15
    data["pv"] = data["ud"] + min(data["v"], 0.0)

    # During a wet spell, PV >= 0 implies prob(end) has returned to 0
    if data["pv"] >= 0:
        _statement_210(data)
        return

    data["ze"] = -2.691 * data["x3"] + 1.5
    _statement_190(data)


def _calc_zindex(data: dict) -> None:
    """
    Calculate Z Index

    Reread monthly parameters for calculation of the 'K' monthly
    weighting factors used in z-index calculation

    :param data: dictionary of parameters (intialized in pdsi)
    """
    for year in range(data["n_years"]):
        for month in range(12):
            data["year"] = year
            data["month"] = month
            k8 = int(data["k8"])
            data["indexj"][k8] = year
            data["indexm"][k8] = month
            data["ze"] = 0.0
            data["ud"] = 0.0
            data["uw"] = 0.0
            cet = data["alpha"][month] * data["pet"][year, month]
            cr = data["beta"][month] * data["prdat"][year, month]
            cro = data["gamma"][month] * data["spdat"][year, month]
            cl = data["delta"][month] * data["pldat"][year, month]
            data["cp"][year, month] = cet + cr + cro - cl
            cd = data["precips"][year, month] - data["cp"][year, month]
            data["z"][year, month] = data["ak"][month] * cd

            # No abatement underway, wet or drought will end if -.5 <= X3 <= .5
            if (data["pro"] == 100) or (data["pro"] == 0):
                # End of drought or wet
                if -0.5 <= data["x3"] <= 0.5:
                    data["pv"] = 0.0
                    data["ppr"][year, month] = 0.0
                    data["px3"][year, month] = 0.0
                    # check for new wet or drought start
                    _statement_200(data)
                    continue
                # We are in a wet spell
                elif data["x3"] > 0.5:
                    # The wet spell intensifies
                    if data["z"][year, month] >= 0.15:
                        _statement_210(data)
                        continue
                    # The wet spell starts to abate (and may end)
                    else:
                        _statement_170(data)
                        continue
                # We are in a drought
                elif data["x3"] < -0.5:
                    # The drought intensifies
                    if data["z"][year, month] <= -0.15:
                        _statement_210(data)
                        continue
                    # The drought starts to abate (and may end)
                    else:
                        _statement_180(data)
                        continue

            # Abatement is underway
            else:
                # We are in a wet spell
                if data["x3"] > 0:
                    _statement_170(data)
                    continue
                # We are in a drought
                elif data["x3"] <= 0:
                    _statement_180(data)
                    continue

            _statement_170(data)
            continue


def _finish_up(data: dict) -> None:
    """
    Wet spell abatement is possible

    :param data: dictionary of parameters (intialized in pdsi)
    """
    for k8 in range(data["k8max"]):
        i = int(data["indexj"][k8])
        j = int(data["indexm"][k8])
        i_end = data["precips"].shape[0] - 1
        data["pdsi"][i, j] = data["x"][i, j]
        data["phdi"][i, j] = data["px3"][i, j]

        if data["px3"][i, j] == 0:
            data["phdi"][i, j] = data["x"][i, j]

        data["wplm"][i, j] = _case(
            data["ppr"][i_end, 11],
            data["px1"][i_end, 11],
            data["px2"][i_end, 11],
            data["px3"][i_end, 11],
        )


def _validate_fitting_params(data: dict[str, Any], fitting_params: dict[str, Any] | None) -> None:
    """
    Validate the fitting parameters

    :param data: dictionary of parameters (intialized in pdsi)
    :param fitting_params: dictionary of the fitted parameters
    """
    if fitting_params is None:
        data["calibrate"] = True
    else:
        data["calibrate"] = False
        for param in ["alpha", "beta", "gamma", "delta"]:
            if (
                param in fitting_params
                and isinstance(fitting_params[param], list | tuple | np.ndarray)
                and len(fitting_params[param]) == 12
            ):
                data[param] = np.array(fitting_params[param])
            else:
                data["calibrate"] = True
                break


def _initialize_data(
    precips: np.ndarray,
    pet: np.ndarray,
    awc: float,
    wctop: float,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Initialize the data dictionary used in Palmer computations.

    Args:
        precips: Monthly precipitation values.
        pet: Monthly PET values.
        awc: Available water capacity (total), in inches.
        wctop: Available water capacity (surface layer), in inches.
        data_start_year: Initial year of the input series (January start).
        calibration_year_initial: Initial year of the calibration period.
        calibration_year_final: Final year of the calibration period.
        fitting_params: Optional fitted CAFEC parameters.

    Returns:
        Initialized data dictionary.
    """
    data: dict[str, Any] = {}
    # reshape precipitation values to (years, 12)
    data["precips"] = utils.reshape_to_2d(precips, 12)
    data["pet"] = utils.reshape_to_2d(pet, 12)
    n_years = data["precips"].shape[0]
    data["awc"] = awc
    data["wctop"] = wctop
    data["awc_bot"] = _get_awc_bot(awc, wctop)
    data["n_years"] = n_years
    data["n_calb_years"] = calibration_year_final - calibration_year_initial + 1
    data["calibration_year_initial_idx"] = calibration_year_initial - data_start_year
    data["calibration_year_final_idx"] = calibration_year_final - data_start_year

    data["psum"] = np.zeros((12,))
    data["spsum"] = np.zeros((12,))
    data["petsum"] = np.zeros((12,))
    data["plsum"] = np.zeros((12,))
    data["prsum"] = np.zeros((12,))
    data["rsum"] = np.zeros((12,))
    data["tlsum"] = np.zeros((12,))
    data["etsum"] = np.zeros((12,))
    data["rosum"] = np.zeros((12,))
    data["spdat"] = np.full((n_years, 12), np.nan)
    data["pldat"] = np.full((n_years, 12), np.nan)
    data["prdat"] = np.full((n_years, 12), np.nan)
    data["rdat"] = np.full((n_years, 12), np.nan)
    data["tldat"] = np.full((n_years, 12), np.nan)
    data["etdat"] = np.full((n_years, 12), np.nan)
    data["rodat"] = np.full((n_years, 12), np.nan)
    data["sssdat"] = np.full((n_years, 12), np.nan)
    data["ssudat"] = np.full((n_years, 12), np.nan)

    data["v"] = 0.0
    data["pro"] = 0.0
    data["x1"] = 0.0
    data["x2"] = 0.0
    data["x3"] = 0.0
    data["k8"] = 0
    data["k8max"] = 0
    data["indexj"] = np.full((K8_SIZE,), np.nan)
    data["indexm"] = np.full((K8_SIZE,), np.nan)
    data["pdsi"] = np.full((n_years, 12), np.nan)
    data["phdi"] = np.full((n_years, 12), np.nan)
    data["z"] = np.full((n_years, 12), np.nan)
    data["wplm"] = np.full((n_years, 12), np.nan)
    data["cp"] = np.full((n_years, 12), np.nan)
    data["ppr"] = np.zeros((n_years, 12))
    data["px1"] = np.zeros((n_years, 12))
    data["px2"] = np.zeros((n_years, 12))
    data["px3"] = np.zeros((n_years, 12))
    data["sx"] = np.zeros((K8_SIZE,))
    data["sx1"] = np.zeros((K8_SIZE,))
    data["sx2"] = np.zeros((K8_SIZE,))
    data["sx3"] = np.zeros((K8_SIZE,))
    data["x"] = np.zeros((n_years, 12))

    _validate_fitting_params(data, fitting_params)

    return data


def pdsi(
    precips: np.ndarray,
    pet: np.ndarray | None,
    awc: float,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict[str, Any] | None = None,
    *,
    missing_policy: str = "climatology",
    precip_climatology: np.ndarray | None = None,
    pet_climatology: np.ndarray | None = None,
    temperature_climatology: np.ndarray | None = None,
    wctop: float = AWCTOP,
    pet_source: str = "input",
    temperature_celsius: np.ndarray | None = None,
    latitude_degrees: float | np.ndarray | None = None,
    fortran_b: float | None = None,
    fortran_h: float | None = None,
    fortran_tla: float | None = None,
    fortran_unit_scale: float = 1.0,
    hargreaves_tmin_celsius: np.ndarray | None = None,
    hargreaves_tmax_celsius: np.ndarray | None = None,
    hargreaves_tmean_celsius: np.ndarray | None = None,
    leap_year_rule: str = "noaa",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    """
    Compute the Palmer drought indices for a single monthly series.

    Args:
        precips: Monthly precipitation values.
        pet: Monthly PET values, if provided.
        awc: Available water capacity (total), in inches.
        data_start_year: Initial year of the input series (January start).
        calibration_year_initial: Initial year of the calibration period.
        calibration_year_final: Final year of the calibration period.
        fitting_params: Optional fitted CAFEC parameters.
        missing_policy: "climatology" to fill missing values or "strict" to keep NaNs.
        precip_climatology: Optional monthly precipitation climatology for fill.
        pet_climatology: Optional monthly PET climatology for fill.
        temperature_climatology: Optional monthly temperature climatology for PET fill.
        wctop: Available water capacity (surface layer), in inches.
        pet_source: "input", "thornthwaite", "fortran", or "hargreaves".
        temperature_celsius: Monthly temperatures for PET computations.
        latitude_degrees: Latitude in degrees north for PET computations.
        fortran_b: Fortran PET soil constant B.
        fortran_h: Fortran PET soil constant H.
        fortran_tla: Negative tangent of latitude for Fortran PET.
        fortran_unit_scale: Scale factor for Fortran PET output.
        hargreaves_tmin_celsius: Daily Tmin values for Hargreaves PET.
        hargreaves_tmax_celsius: Daily Tmax values for Hargreaves PET.
        hargreaves_tmean_celsius: Daily Tmean values for Hargreaves PET.
        leap_year_rule: "noaa" or "gregorian" for leap-year logic.

    Returns:
        Tuple of arrays: (PDSI, PHDI, PMDI/WPLM, Z-index) and fitted parameters.

    Raises:
        ValueError: If inputs are incompatible or required PET inputs are missing.
    """

    # Normalize to float arrays with NaN handling for masked arrays
    precips = _as_float_array(precips)
    pet_values = _as_float_array(pet) if pet is not None else None
    temperature_celsius_values = _as_float_array(temperature_celsius) if temperature_celsius is not None else None
    hargreaves_tmin_values = _as_float_array(hargreaves_tmin_celsius) if hargreaves_tmin_celsius is not None else None
    hargreaves_tmax_values = _as_float_array(hargreaves_tmax_celsius) if hargreaves_tmax_celsius is not None else None
    hargreaves_tmean_values = (
        _as_float_array(hargreaves_tmean_celsius) if hargreaves_tmean_celsius is not None else None
    )

    # Validate inputs
    # if we're passed all missing values then we can't compute anything,
    # so we return the same array of missing values
    if np.all(np.isnan(precips)):
        return precips, precips, precips, precips, None

    # clip any negative values to zero
    if np.amin(precips) < 0.0:
        _logger.warning("Input contains negative values -- all negatives clipped to zero")
        precips = np.clip(precips, a_min=0.0, a_max=None)

    # remember the original length of the input array, in order to facilitate
    # returning an array of the same size
    original_length = precips.size

    policy = missing_policy.strip().lower()
    if policy not in ("climatology", "strict"):
        raise ValueError("missing_policy must be either 'climatology' or 'strict'")

    precip_climatology = _validate_monthly_climatology(precip_climatology, "precip_climatology")
    pet_climatology = _validate_monthly_climatology(pet_climatology, "pet_climatology")
    temperature_climatology = _validate_monthly_climatology(temperature_climatology, "temperature_climatology")

    if policy == "climatology" and pet_source.strip().lower() in ("thornthwaite", "fortran"):
        if temperature_celsius_values is not None:
            if temperature_climatology is None:
                temperature_climatology = _compute_monthly_climatology(
                    temperature_celsius_values,
                    data_start_year=data_start_year,
                    calibration_year_initial=calibration_year_initial,
                    calibration_year_final=calibration_year_final,
                )
            temperature_celsius_values = _fill_missing_with_climatology(
                temperature_celsius_values,
                temperature_climatology,
                original_length,
            )

    pet = _resolve_pet_source(
        pet_source=pet_source,
        pet=pet_values,
        temperature_celsius=temperature_celsius_values,
        latitude_degrees=latitude_degrees,
        data_start_year=data_start_year,
        fortran_b=fortran_b,
        fortran_h=fortran_h,
        fortran_tla=fortran_tla,
        fortran_unit_scale=fortran_unit_scale,
        hargreaves_tmin_celsius=hargreaves_tmin_values,
        hargreaves_tmax_celsius=hargreaves_tmax_values,
        hargreaves_tmean_celsius=hargreaves_tmean_values,
        leap_year_rule=leap_year_rule,
    )

    # validate that the two input arrays are compatible
    if precips.size != pet.size:
        message = "Incompatible precipitation and PET arrays"
        _logger.error(message)
        raise ValueError(message)

    if policy == "climatology":
        if precip_climatology is None:
            precip_climatology = _compute_monthly_climatology(
                precips,
                data_start_year=data_start_year,
                calibration_year_initial=calibration_year_initial,
                calibration_year_final=calibration_year_final,
            )
        precips = _fill_missing_with_climatology(precips, precip_climatology, original_length)

        if pet_climatology is None:
            pet_climatology = _compute_monthly_climatology(
                pet,
                data_start_year=data_start_year,
                calibration_year_initial=calibration_year_initial,
                calibration_year_final=calibration_year_final,
            )
        pet = _fill_missing_with_climatology(pet, pet_climatology, original_length)

    # Initialize data
    data = _initialize_data(
        precips=precips,
        pet=pet,
        awc=awc,
        wctop=wctop,
        data_start_year=data_start_year,
        calibration_year_initial=calibration_year_initial,
        calibration_year_final=calibration_year_final,
        fitting_params=fitting_params,
    )

    # Water balance calcs
    _calc_water_balances(data)

    # Get Cafec coefficients
    if data["calibrate"]:
        _calc_cafec_coefficients(data)

    # Calculate Z-Index weighting factors (variable AK)
    _calc_zindex_factors(data)

    # Sum variables now become averages over the calibration period
    # (currently not used - uncomment if want to export later)
    # _avg_calibration_sums(data)

    # reread monthly parameters for calculation of the 'K' monthly
    # weighting factors used in z-index calculation
    _calc_kfactors(data)

    # Calculate the z-index (moisture anomaly) and pdsi (variable x)
    _calc_zindex(data)

    _finish_up(data)

    # Format values
    pdsi = data["pdsi"].flatten()[0:original_length]
    phdi = data["phdi"].flatten()[0:original_length]
    wplm = data["wplm"].flatten()[0:original_length]
    z = data["z"].flatten()[0:original_length]
    params = {key: data[key] for key in ["alpha", "beta", "gamma", "delta"]}

    # return results
    return pdsi, phdi, wplm, z, params
