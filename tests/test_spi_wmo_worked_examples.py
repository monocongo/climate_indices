"""Stage checks against the technical report cited by WMO-No. 1090, §6.

Edwards & McKee (1997), Climatology Report 97-2, chapter 3. These tests do not
use the Falls City software screenshots from the WMO guide as a numerical oracle.
"""

import numpy as np
import pytest

from climate_indices import compute
from climate_indices.drought import indices

pytestmark = pytest.mark.validation


def test_gamma_fit_from_report_equations() -> None:
    # Synthetic monthly totals of 1 and 4 mm in 2000–2001, no zeros or missing values.
    # Eq. 3.6–3.9: mean = 2.5, geometric mean = 2, A = ln(1.25). Evaluate the
    # equations with 45-digit Decimal arithmetic, then round alpha and beta to binary64.
    values = np.repeat([[1.0], [4.0]], 12, axis=1)
    alpha, beta = compute.gamma_parameters(values, 2000, 2000, 2001, compute.Periodicity.monthly)

    np.testing.assert_allclose(alpha, 2.3965396040318065, rtol=0, atol=1e-13)
    np.testing.assert_allclose(beta, 1.0431707432642204, rtol=0, atol=1e-13)


def test_report_gamma_to_normal_table() -> None:
    # Report Table 3.1 supplies the probability and expected SPI pairs. Invert the
    # unit-scale exponential (Gamma shape 1) to construct inputs with those CDFs.
    # The gamma parameters are supplied, so this is not a test of calibration.
    probabilities = np.array([0.1587, 0.5000, 0.8413])
    values = np.tile(-np.log1p(-probabilities), 4)
    params = {name: np.full(12, value) for name, value in (("alpha", 1.0), ("beta", 1.0), ("prob_zero", 0.0))}
    args = (values, 1, indices.Distribution.gamma, 2000, 2000, 2000, compute.Periodicity.monthly, params)

    actual_probabilities = indices.spi(*args, output_scale="probability")
    np.testing.assert_allclose(actual_probabilities, np.tile(probabilities, 4), rtol=0, atol=1e-15)
    # Four decimal places imply +/-0.00005 in p, or at most 0.00021 in z near +/-1.
    np.testing.assert_allclose(indices.spi(*args), np.tile([-1.0, 0.0, 1.0], 4), rtol=0, atol=0.00021)
