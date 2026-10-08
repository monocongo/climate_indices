# Type stub for the optional Rust extension built from crates/climate-py.
# The extension is absent from pure-Python installs; callers import it inside
# try/except ImportError and fall back to the Python implementation. Every
# array argument must be a float64 ndarray; other dtypes raise TypeError.

import numpy as np
import numpy.typing as npt

__version__: str

def gamma_parameters(
    calibration: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def gamma_probabilities(
    values: npt.NDArray[np.float64],
    alphas: npt.NDArray[np.float64],
    betas: npt.NDArray[np.float64],
    probabilities_of_zero: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def pnp_normals(
    calibration: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def pnp_percentages(
    scale_sums: npt.NDArray[np.float64],
    normals: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def pci(rainfall: npt.NDArray[np.float64]) -> float: ...
def pearson_parameters(
    calibration: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
]: ...
def pearson_cdf(
    values: npt.NDArray[np.float64],
    skews: npt.NDArray[np.float64],
    locs: npt.NDArray[np.float64],
    scales: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def loglogistic_parameters(
    calibration: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
]: ...
def loglogistic_cdf(
    values: npt.NDArray[np.float64],
    locs: npt.NDArray[np.float64],
    scales: npt.NDArray[np.float64],
    shapes: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def norm_ppf(probabilities: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
def thornthwaite(
    monthly_temps_celsius: npt.NDArray[np.float64],
    latitude_radians: npt.NDArray[np.float64],
    leap_years: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]: ...
def hargreaves(
    daily_tmin_celsius: npt.NDArray[np.float64],
    daily_tmax_celsius: npt.NDArray[np.float64],
    daily_tmean_celsius: npt.NDArray[np.float64],
    latitude_radians: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def pm_eto(
    net_radiation: npt.NDArray[np.float64],
    soil_heat_flux: npt.NDArray[np.float64],
    temperature_celsius: npt.NDArray[np.float64],
    wind_speed_2m: npt.NDArray[np.float64],
    saturation_vp: npt.NDArray[np.float64],
    actual_vp: npt.NDArray[np.float64],
    delta: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def fao56_eto(
    daily_tmin_celsius: npt.NDArray[np.float64],
    daily_tmax_celsius: npt.NDArray[np.float64],
    latitude_degrees: npt.NDArray[np.float64],
    elevation_m: npt.NDArray[np.float64],
    wind_speed_m_s: npt.NDArray[np.float64],
    wind_speed_height_m: npt.NDArray[np.float64],
    day_of_year: npt.NDArray[np.float64],
    soil_heat_flux_mj_m2_day: npt.NDArray[np.float64],
    albedo: npt.NDArray[np.float64],
    humidity_variant: int,
    tdew_celsius: npt.NDArray[np.float64] | None,
    rh_min: npt.NDArray[np.float64] | None,
    rh_max: npt.NDArray[np.float64] | None,
    rh_mean: npt.NDArray[np.float64] | None,
    radiation_variant: int,
    solar_radiation_mj_m2_day: npt.NDArray[np.float64] | None,
    sunshine_hours: npt.NDArray[np.float64] | None,
    coastal: bool,
) -> npt.NDArray[np.float64]: ...
def tukey_probabilities(
    climatology: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    pads: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def hastings_inverse_normal(probabilities: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
