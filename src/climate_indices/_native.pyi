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
def norm_ppf(probabilities: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
