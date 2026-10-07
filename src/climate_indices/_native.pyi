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
