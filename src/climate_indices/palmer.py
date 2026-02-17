"""Compute palmer drought indices.

This module contains:
- SECTION 1: NumPy core computation engine (existing palmer.pdsi function)
- SECTION 2: xarray adapter layer (palmer_xarray manual wrapper)

The xarray wrapper uses Pattern C (manual wrapper) rather than the @xarray_adapter
decorator because Palmer returns multiple outputs (4 arrays + params_dict) as an
xr.Dataset, which is incompatible with the single-output decorator design.
"""

from __future__ import annotations

import copy
import json
import time
import warnings
from typing import Any

import numpy as np
import xarray as xr

from climate_indices import utils
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.exceptions import CoordinateValidationError, InputAlignmentWarning
from climate_indices.logging_config import get_logger
from climate_indices.xarray_adapter import (
    _append_history,
    _build_history_entry,
    _infer_calibration_period,
    _infer_data_start_year,
    _validate_time_dimension,
    _validate_time_monotonicity,
)

_logger = get_logger(__name__)

# declare the function names that should be included in the public API for this module
__all__ = ["pdsi", "palmer_xarray"]

AWCTOP = 1.0
K8_SIZE = 40


def _get_awc_bot(awc: float) -> float:
    """
    Calculate available water capcity in bottom layer

    :param awc: available water capacity (total), in inches
    :return available water capacity (under layer), in inches
    :rtype: float
    """
    return max(awc - AWCTOP, 0.0)


def _calc_potential_loss(
    pet: float,
    ss: float,
    su: float,
    awc: float,
) -> float:
    """
    Calculate potential loss

    :param pet: potential evapotranspiration
    :param ss: surface layer water content, in inches
    :param su: under layer water content, in inches
    :param awc: available water capacity (total), in inches
    :return potential loss
    :rtype: float
    """
    awc_bot = _get_awc_bot(awc)
    if ss >= pet:
        return pet
    return min(ss + su, ((pet - ss) * su) / (awc_bot + AWCTOP) + ss)


def _calc_recharge(
    p: float,
    pet: float,
    ss: float,
    su: float,
    awc: float,
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
    :return a tuple of floats
        - et: evapotranspiration
        - tl: total loss
        - r: recharge
        - ro: runoff
        - sss: surface layer water content, in inches
        - ssu: under layer water content, in inches
    """
    awc_bot = _get_awc_bot(awc)

    # precipitation exceeds potential evaporation
    if p >= pet:
        et = pet
        tl = 0.0

        # excess precipitation recharges under layer as well as upper
        if (p - pet) > (AWCTOP - ss):
            rs = AWCTOP - ss
            sss = AWCTOP

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


def _calc_alpha(data: dict[str, Any]) -> None:
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


def _calc_beta(data: dict[str, Any]) -> None:
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


def _calc_gamma(data: dict[str, Any]) -> None:
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


def _calc_delta(data: dict[str, Any]) -> None:
    """
    Calculate delta parameters

    :param data: dictionary of parameters (intialized in pdsi)
    """
    data["delta"] = np.zeros(data["plsum"].shape)
    for idx, pl in enumerate(data["plsum"]):
        if pl != 0:
            data["delta"][idx] = data["tlsum"][idx] / pl


def _calc_water_balances(data: dict[str, Any]) -> None:
    """
    Perform water balance calculations

    :param data: dictionary of parameters (intialized in pdsi)
    """
    ss = AWCTOP
    su = data["awc_bot"]
    for year in range(data["n_years"]):
        for month in range(12):
            p = data["precips"][year, month]
            pet = data["pet"][year, month]
            sp = ss + su
            pr = data["awc_bot"] + AWCTOP - sp

            # Get potential loss
            pl = _calc_potential_loss(pet, ss, su, data["awc"])

            # Calculate recharge, runoff, residual moisture, loss to both
            # surface and under layers, depending on starting moisture
            # content and values of precipitation and evaporation
            et, tl, r, ro, sss, ssu = _calc_recharge(p, pet, ss, su, data["awc"])

            # update sums
            if data["calibration_year_initial_idx"] <= year <= data["calibration_year_final_idx"]:
                data["psum"][month] += p
                data["spsum"][month] += sp
                data["petsum"][month] += pet
                data["plsum"][month] += pl
                data["prsum"][month] += pr
                data["rsum"][month] += r
                data["tlsum"][month] += tl
                data["etsum"][month] += et
                data["rosum"][month] += ro

            # set data
            data["spdat"][year, month] = sp
            data["pldat"][year, month] = pl
            data["prdat"][year, month] = pr
            data["rdat"][year, month] = r
            data["tldat"][year, month] = tl
            data["etdat"][year, month] = et
            data["rodat"][year, month] = ro
            data["sssdat"][year, month] = sss
            data["ssudat"][year, month] = ssu

            # update soil moisture
            ss = sss
            su = ssu


def _calc_cafec_coefficients(data: dict[str, Any]) -> None:
    """
    Calculate CAFEC Coefficients

    :param data: dictionary of parameters (intialized in pdsi)
    """
    _calc_alpha(data)
    _calc_beta(data)
    _calc_gamma(data)
    _calc_delta(data)


def _calc_zindex_factors(data: dict[str, Any]) -> None:
    """
    Calculate Z-Index weighting factors (variable AK)

    trat is the 'T' ratio of average moisture demand
    to average moisture supply in month M

    :param data: dictionary of parameters (intialized in pdsi)
    """
    data["trat"] = (data["petsum"] + data["rsum"] + data["rosum"]) / (data["psum"] + data["tlsum"])


def _avg_calibration_sums(data: dict[str, Any]) -> None:
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


def _calc_kfactors(data: dict[str, Any]) -> None:
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


def _assign(data: dict[str, Any]) -> None:
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


def _statement_220(data: dict[str, Any]) -> None:
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


def _statement_210(data: dict[str, Any]) -> None:
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


def _statement_200(data: dict[str, Any]) -> None:
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


def _statement_190(data: dict[str, Any]) -> None:
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


def _statement_180(data: dict[str, Any]) -> None:
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


def _statement_170(data: dict[str, Any]) -> None:
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


def _calc_zindex(data: dict[str, Any]) -> None:
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


def _finish_up(data: dict[str, Any]) -> None:
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
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Initialize the data

    :param precips: time series of monthly precipitation values, in inches
    :param pet: time series of monthly PET values, in inches
    :param awc: available water capacity (soil constant), in inches
    :param data_start_year: initial year of the input precipitation and PET datasets,
                            both of which are assumed to start in January of this year
    :param calibration_start_year: initial year of the calibration period
    :param calibration_end_year: final year of the calibration period
    :param fitting_params: dictionary of the fitted parameters
    :return dictionary of intialized parameters
    :rtype: dict
    """
    data: dict[str, Any] = {}
    # reshape precipitation values to (years, 12)
    data["precips"] = utils.reshape_to_2d(precips, 12)
    data["pet"] = utils.reshape_to_2d(pet, 12)
    n_years = int(data["precips"].shape[0])
    data["awc"] = awc
    data["awc_bot"] = _get_awc_bot(awc)
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
    pet: np.ndarray,
    awc: float,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    """
    Compute the Palmer Drought Severity Index (PDSI),
    Palmer Hydrological Drought Index (PHDI),
    Palmer Modified Drought Index (PMDI), and
    Palmer Z-Index.

    :param precips: time series of monthly precipitation values, in inches
    :param pet: time series of monthly PET values, in inches
    :param awc: available water capacity (soil constant), in inches
    :param data_start_year: initial year of the input precipitation and PET datasets,
                            both of which are assumed to start in January of this year
    :param calibration_start_year: initial year of the calibration period
    :param calibration_end_year: final year of the calibration period
    :param fitting_params: dictionary of the fitted parameters
    :return: four numpy arrays containing PDSI, PHDI, PMDI, and Z-Index values respectively and
             a dictionary containing the fitted parameters (alpha, beta, gamma, and delta)
    :rtype: four numpy.ndarrays of the PDSI values and a dictionary of the fitted parameters
    """

    # bind structured logging context for this calculation
    log = _logger.bind(
        index_type="pdsi",
        awc=awc,
        data_start_year=data_start_year,
        calibration_year_initial=calibration_year_initial,
        calibration_year_final=calibration_year_final,
        input_shape=precips.shape,
        input_elements=precips.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()

    try:
        # if we're passed all missing values then we can't compute anything,
        # so we return the same array of missing values
        if (isinstance(precips, np.ma.MaskedArray) and precips.mask.all()) or np.all(np.isnan(precips)):
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info(
                "calculation_completed",
                duration_ms=round(duration_ms, 2),
                result="all_missing",
            )
            return precips, precips, precips, precips, None

        # validate that the two input arrays are compatible
        if precips.size != pet.size:
            message = "Incompatible precipitation and PET arrays"
            log.error("validation_failed", reason=message)
            raise ValueError(message)

        # clip any negative values to zero
        if np.amin(precips) < 0.0:
            log.warning("negative_values_clipped", field="precips")
            precips = np.clip(precips, a_min=0.0, a_max=None)

        # remember the original length of the input array, in order to facilitate
        # returning an array of the same size
        original_length = precips.size

        # Initialize data
        data = _initialize_data(
            precips=precips,
            pet=pet,
            awc=awc,
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
        pdsi_result = data["pdsi"].flatten()[0:original_length]
        phdi = data["phdi"].flatten()[0:original_length]
        wplm = data["wplm"].flatten()[0:original_length]
        z = data["z"].flatten()[0:original_length]
        params = {key: data[key] for key in ["alpha", "beta", "gamma", "delta"]}

        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "calculation_completed",
            duration_ms=round(duration_ms, 2),
            output_elements=pdsi_result.size,
        )

        return pdsi_result, phdi, wplm, z, params

    except Exception:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.error("calculation_failed", duration_ms=round(duration_ms, 2))
        raise


# ============================================================================
# SECTION 2: XARRAY ADAPTER LAYER
# ============================================================================
# Manual wrapper for Palmer multi-output Dataset return.
# Pattern C from research: required because @xarray_adapter only supports
# single-output functions returning np.ndarray, while Palmer returns a 5-tuple
# (pdsi, phdi, pmdi, z_index, params_dict) that must be assembled into xr.Dataset.

# variable names for the 4 Palmer output arrays
_PALMER_VARIABLE_NAMES = ("pdsi", "phdi", "pmdi", "z_index")


def palmer_xarray(
    precip_da: xr.DataArray,
    pet_da: xr.DataArray,
    awc: float | xr.DataArray,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    fitting_params: dict[str, Any] | None = None,
    time_dim: str = "time",
) -> xr.Dataset:
    """Compute Palmer drought indices with xarray Dataset return.

    This is a manual wrapper around the NumPy-based ``pdsi()`` function that
    accepts xarray DataArrays and returns an ``xr.Dataset`` containing all four
    Palmer indices (PDSI, PHDI, PMDI, Z-Index) with preserved coordinates and
    provenance metadata.

    A manual wrapper (Pattern C) is used instead of the ``@xarray_adapter``
    decorator because Palmer returns multiple output arrays plus a calibration
    parameters dictionary, which is incompatible with the decorator's
    single-output design.

    .. warning:: **Beta Feature** -- The xarray adapter layer is beta and may
       change in future minor releases. The NumPy ``pdsi()`` function is stable.

    Args:
        precip_da: Monthly precipitation values in inches as xr.DataArray
            with a time dimension.
        pet_da: Monthly potential evapotranspiration values in inches as
            xr.DataArray with a time dimension. Must align with precip_da.
        awc: Available water capacity (soil constant) in inches. Accepts
            a scalar float for uniform AWC or an ``xr.DataArray`` with
            spatial dimensions only (e.g. lat, lon). AWC is a soil property
            and must not contain a time dimension.
        data_start_year: Initial year of the input datasets. If None, inferred
            from the first timestamp in the time coordinate.
        calibration_year_initial: Initial year of the calibration period.
            If None, inferred from the first year of the time coordinate.
        calibration_year_final: Final year of the calibration period.
            If None, inferred from the last year of the time coordinate.
        fitting_params: Optional dictionary of pre-computed fitting parameters
            (alpha, beta, gamma, delta). If None, parameters are calibrated
            from the data.
        time_dim: Name of the time dimension (default: "time").

    Returns:
        xr.Dataset with data variables "pdsi", "phdi", "pmdi", "z_index",
        each preserving coordinates and dimensions from the input DataArrays.
        Dataset-level attrs include calibration parameters and provenance.

    Raises:
        CoordinateValidationError: If time dimension is missing, not monotonic,
            or inputs have no overlapping time steps.
        ValueError: If AWC has a time dimension or if precip and PET arrays
            are incompatible sizes.

    Examples:
        >>> import xarray as xr
        >>> import pandas as pd
        >>> import numpy as np
        >>> time_coord = pd.date_range("1980-01", periods=480, freq="MS")
        >>> precip = xr.DataArray(
        ...     np.random.uniform(0, 5, 480),
        ...     coords={"time": time_coord},
        ...     dims=["time"],
        ... )
        >>> pet = xr.DataArray(
        ...     np.random.uniform(1, 4, 480),
        ...     coords={"time": time_coord},
        ...     dims=["time"],
        ... )
        >>> ds = palmer_xarray(precip, pet, awc=5.0)
        >>> list(ds.data_vars)
        ['pdsi', 'phdi', 'pmdi', 'z_index']
    """
    # describe AWC shape for logging (scalar vs DataArray)
    awc_log_val = awc if isinstance(awc, (int, float)) else f"DataArray{awc.shape}"
    log = _logger.bind(
        index_type="palmer_xarray",
        awc=awc_log_val,
        time_dim=time_dim,
        precip_shape=precip_da.shape,
        pet_shape=pet_da.shape,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()

    try:
        # -- validate AWC parameter --
        if isinstance(awc, xr.DataArray) and time_dim in awc.dims:
            raise ValueError(
                f"AWC must not have time dimension '{time_dim}'. "
                f"AWC is a soil property (spatially varying only). "
                f"Got awc_dims={awc.dims}."
            )

        # -- validate time dimension on both inputs --
        _validate_time_dimension(precip_da, time_dim)
        _validate_time_dimension(pet_da, time_dim)

        precip_time = precip_da[time_dim]
        pet_time = pet_da[time_dim]
        _validate_time_monotonicity(precip_time)
        _validate_time_monotonicity(pet_time)

        # -- align inputs (inner join on time coordinate) --
        precip_aligned, pet_aligned = xr.align(precip_da, pet_da, join="inner")
        aligned_len = len(precip_aligned[time_dim])
        original_precip_len = len(precip_time)
        original_pet_len = len(pet_time)

        if aligned_len == 0:
            raise CoordinateValidationError(
                message=(f"No overlapping time steps between precip and PET along '{time_dim}' dimension."),
                coordinate_name=time_dim,
                reason="empty_intersection_after_alignment",
            )

        if aligned_len < original_precip_len or aligned_len < original_pet_len:
            dropped = max(original_precip_len, original_pet_len) - aligned_len
            log.warning(
                "input_alignment_dropped_data",
                original_precip_len=original_precip_len,
                original_pet_len=original_pet_len,
                aligned_len=aligned_len,
            )
            warnings.warn(
                InputAlignmentWarning(
                    message=(
                        f"Input alignment dropped time steps. "
                        f"precip had {original_precip_len}, PET had {original_pet_len}, "
                        f"aligned to {aligned_len}."
                    ),
                    original_size=max(original_precip_len, original_pet_len),
                    aligned_size=aligned_len,
                    dropped_count=dropped,
                ),
                stacklevel=2,
            )

        # -- infer temporal parameters from time coordinate --
        time_coord = precip_aligned[time_dim]

        if data_start_year is None:
            data_start_year = _infer_data_start_year(time_coord)

        if calibration_year_initial is None or calibration_year_final is None:
            cal_start, cal_end = _infer_calibration_period(time_coord)
            if calibration_year_initial is None:
                calibration_year_initial = cal_start
            if calibration_year_final is None:
                calibration_year_final = cal_end

        log.info(
            "parameters_resolved",
            data_start_year=data_start_year,
            calibration_year_initial=calibration_year_initial,
            calibration_year_final=calibration_year_final,
        )

        # -- determine spatial dimensions (everything except time) --
        spatial_dims = [d for d in precip_aligned.dims if d != time_dim]
        is_spatial = len(spatial_dims) > 0

        # -- compute Palmer indices --
        if not is_spatial or not isinstance(awc, xr.DataArray):
            # scalar AWC path: single call to pdsi() for 1D time series
            # (or multi-dim with uniform AWC treated pixel-by-pixel below)
            awc_scalar = float(awc) if not isinstance(awc, xr.DataArray) else float(awc.values)
            precip_values = precip_aligned.values
            pet_values = pet_aligned.values

            pdsi_arr, phdi_arr, pmdi_arr, z_arr, params_dict = pdsi(
                precips=precip_values,
                pet=pet_values,
                awc=awc_scalar,
                data_start_year=data_start_year,
                calibration_year_initial=calibration_year_initial,
                calibration_year_final=calibration_year_final,
                fitting_params=fitting_params,
            )

            output_arrays = {
                "pdsi": pdsi_arr,
                "phdi": phdi_arr,
                "pmdi": pmdi_arr,
                "z_index": z_arr,
            }
        else:
            # spatial DataArray AWC path: iterate over spatial pixels
            # pdsi() only accepts scalar AWC + 1D time series
            n_time = len(precip_aligned[time_dim])
            spatial_shape = tuple(precip_aligned.sizes[d] for d in spatial_dims)

            # pre-allocate output arrays: shape = (*spatial_dims, time)
            result_arrays = {name: np.full((*spatial_shape, n_time), np.nan) for name in _PALMER_VARIABLE_NAMES}
            params_dict = None

            # iterate over all spatial grid points
            for idx in np.ndindex(*spatial_shape):
                # build selection dict for spatial dims
                sel = {dim: precip_aligned[dim].values[i] for dim, i in zip(spatial_dims, idx, strict=True)}

                precip_pixel = precip_aligned.sel(sel).values
                pet_pixel = pet_aligned.sel(sel).values
                awc_pixel = float(awc.sel(sel).values)

                p_arr, h_arr, m_arr, z_arr_px, px_params = pdsi(
                    precips=precip_pixel,
                    pet=pet_pixel,
                    awc=awc_pixel,
                    data_start_year=data_start_year,
                    calibration_year_initial=calibration_year_initial,
                    calibration_year_final=calibration_year_final,
                    fitting_params=fitting_params,
                )

                result_arrays["pdsi"][idx] = p_arr
                result_arrays["phdi"][idx] = h_arr
                result_arrays["pmdi"][idx] = m_arr
                result_arrays["z_index"][idx] = z_arr_px

                # keep the last params_dict (all pixels get the same calibration)
                if params_dict is None:
                    params_dict = px_params

            output_arrays = result_arrays

        # -- build output Dataset --
        data_vars: dict[str, xr.DataArray] = {}
        for var_name in _PALMER_VARIABLE_NAMES:
            var_attrs = copy.deepcopy(CF_METADATA[var_name])
            da = xr.DataArray(
                output_arrays[var_name],
                coords=precip_aligned.coords,
                dims=precip_aligned.dims,
                attrs=var_attrs,
                name=var_name,
            )
            data_vars[var_name] = da

        ds = xr.Dataset(data_vars)

        # -- add Dataset-level attributes --
        from climate_indices import __version__

        ds.attrs["climate_indices_version"] = __version__

        # store calibration parameters with dual-access pattern:
        # 1. JSON string in ds.attrs["palmer_params"] for full dict round-trip
        # 2. Individual attrs ds.attrs["palmer_alpha"] etc. for convenience
        if params_dict is not None:
            serializable_params = {}
            for param_name in ("alpha", "beta", "gamma", "delta"):
                if param_name in params_dict:
                    param_list = params_dict[param_name].tolist()
                    ds.attrs[f"palmer_{param_name}"] = param_list
                    serializable_params[param_name] = param_list
            ds.attrs["palmer_params"] = json.dumps(serializable_params)

        # describe AWC in history (scalar value or "spatial DataArray")
        awc_history_val = awc if isinstance(awc, (int, float)) else "spatial_DataArray"
        history_entry = _build_history_entry(
            "Palmer",
            __version__,
            {
                "awc": awc_history_val,
                "calibration_year_initial": calibration_year_initial,
                "calibration_year_final": calibration_year_final,
            },
        )
        ds.attrs["history"] = _append_history(precip_aligned.attrs, history_entry)

        # deep-copy coordinate attrs to prevent mutation
        for coord_name in ds.coords:
            if coord_name in precip_aligned.coords:
                ds.coords[coord_name].attrs = copy.deepcopy(precip_aligned.coords[coord_name].attrs)

        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "calculation_completed",
            duration_ms=round(duration_ms, 2),
            output_variables=list(_PALMER_VARIABLE_NAMES),
            output_shape=list(ds.dims.values()),
        )

        return ds

    except Exception:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.error("calculation_failed", duration_ms=round(duration_ms, 2))
        raise
