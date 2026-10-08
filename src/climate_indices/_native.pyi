# Type stub for the optional Rust extension built from crates/climate-py.
# The extension is absent from pure-Python installs; callers import it inside
# try/except ImportError and fall back to the Python implementation. Every
# float64 array must be a float64 ndarray, every validity mask a bool ndarray,
# and every index array an int64 ndarray; other dtypes raise TypeError.

import numpy as np
import numpy.typing as npt

__version__: str

class NonFiniteResultError(ValueError):
    """A recurrence step produced a non-finite value from finite inputs.

    The Python driver raises ``InvalidArgumentError`` for this; the dispatch
    translates it, so a caller never sees this type.
    """

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
def tukey_probabilities(
    climatology: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    pads: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def hastings_inverse_normal(probabilities: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
def ffmc(
    temperature_celsius: npt.NDArray[np.float64],
    relative_humidity_percent: npt.NDArray[np.float64],
    wind_speed_kilometers_per_hour: npt.NDArray[np.float64],
    precipitation_mm: npt.NDArray[np.float64],
    initial_ffmc: npt.NDArray[np.float64],
    weather_valid: npt.NDArray[np.bool_],
    static_valid: npt.NDArray[np.bool_],
    in_season: npt.NDArray[np.bool_] | None,
    trailing_gap_days: npt.NDArray[np.int64],
    spin_up: int,
    nan_policy: str,
    max_gap_days: int,
    record: bool,
) -> tuple[
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64],
    npt.NDArray[np.int64] | None,
]: ...
def duff_moisture_code(
    temperature_celsius: npt.NDArray[np.float64],
    relative_humidity_percent: npt.NDArray[np.float64],
    precipitation_mm: npt.NDArray[np.float64],
    day_length_table: npt.NDArray[np.float64],
    months: npt.NDArray[np.int64],
    day_length_band: npt.NDArray[np.int64],
    initial_dmc: npt.NDArray[np.float64],
    weather_valid: npt.NDArray[np.bool_],
    static_valid: npt.NDArray[np.bool_],
    in_season: npt.NDArray[np.bool_] | None,
    trailing_gap_days: npt.NDArray[np.int64],
    spin_up: int,
    nan_policy: str,
    max_gap_days: int,
    record: bool,
) -> tuple[
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64],
    npt.NDArray[np.int64] | None,
]: ...
def drought_code(
    temperature_celsius: npt.NDArray[np.float64],
    precipitation_mm: npt.NDArray[np.float64],
    day_length_table: npt.NDArray[np.float64],
    months: npt.NDArray[np.int64],
    day_length_band: npt.NDArray[np.int64],
    initial_dc: npt.NDArray[np.float64],
    weather_valid: npt.NDArray[np.bool_],
    static_valid: npt.NDArray[np.bool_],
    in_season: npt.NDArray[np.bool_] | None,
    trailing_gap_days: npt.NDArray[np.int64],
    spin_up: int,
    nan_policy: str,
    max_gap_days: int,
    record: bool,
) -> tuple[
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64],
    npt.NDArray[np.int64] | None,
]: ...
def kbdi(
    precipitation_mm: npt.NDArray[np.float64],
    maximum_temperature_celsius: npt.NDArray[np.float64],
    mean_annual_precipitation_mm: npt.NDArray[np.float64],
    initial_kbdi: npt.NDArray[np.float64],
    initial_wet_spell_precipitation: npt.NDArray[np.float64],
    weather_valid: npt.NDArray[np.bool_],
    static_valid: npt.NDArray[np.bool_],
    in_season: npt.NDArray[np.bool_] | None,
    trailing_gap_days: npt.NDArray[np.int64],
    spin_up: int,
    nan_policy: str,
    max_gap_days: int,
    record: bool,
) -> tuple[
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64] | None,
]: ...
def effective_precipitation(precipitation_mm: npt.NDArray[np.float64], duration: int) -> npt.NDArray[np.float64]: ...
def edi(
    years: npt.NDArray[np.float64],
    calibration_start: int,
    calibration_end: int,
) -> npt.NDArray[np.float64]: ...
def flood_index(
    pe: npt.NDArray[np.float64],
    first_start: int,
    calibration_years: int,
) -> npt.NDArray[np.float64]: ...
def antecedent_precipitation_index(
    precipitation_mm: npt.NDArray[np.float64],
    k: float,
    initial_api: npt.NDArray[np.float64],
    weather_valid: npt.NDArray[np.bool_],
    static_valid: npt.NDArray[np.bool_],
    trailing_gap_days: npt.NDArray[np.int64],
    spin_up: int,
    nan_policy: str,
    max_gap_days: int,
    record: bool,
) -> tuple[
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64],
    npt.NDArray[np.int64] | None,
]: ...
