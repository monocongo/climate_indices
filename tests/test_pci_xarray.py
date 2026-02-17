"""Tests for PCI (Precipitation Concentration Index) xarray adapter integration.

Validates xarray DataArray support for pci() via the typed_public_api,
including numerical equivalence with the NumPy path, CF metadata application,
and input validation for daily rainfall lengths.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from climate_indices import pci
from climate_indices.indices import pci as numpy_pci


@pytest.fixture(scope="module")
def daily_rainfall_365() -> xr.DataArray:
    """Create a 1D daily rainfall DataArray for a non-leap year (365 days)."""
    import pandas as pd

    time = pd.date_range("2019-01-01", "2019-12-31", freq="D")
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=5.0, size=len(time))
    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={"long_name": "daily precipitation", "units": "mm"},
    )


@pytest.fixture(scope="module")
def daily_rainfall_366() -> xr.DataArray:
    """Create a 1D daily rainfall DataArray for a leap year (366 days)."""
    import pandas as pd

    time = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=5.0, size=len(time))
    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={"long_name": "daily precipitation", "units": "mm"},
    )


@pytest.fixture(scope="module")
def pci_xarray_result_365(daily_rainfall_365: xr.DataArray) -> xr.DataArray:
    """Cached PCI result for 365-day DataArray."""
    return pci(rainfall_mm=daily_rainfall_365)


@pytest.fixture(scope="module")
def pci_xarray_result_366(daily_rainfall_366: xr.DataArray) -> xr.DataArray:
    """Cached PCI result for 366-day DataArray."""
    return pci(rainfall_mm=daily_rainfall_366)


class TestPCIXarrayEquivalence:
    """Validate numerical equivalence between NumPy and xarray paths."""

    def test_values_match_numpy_365(
        self,
        daily_rainfall_365: xr.DataArray,
        pci_xarray_result_365: xr.DataArray,
    ) -> None:
        """xarray PCI values should match NumPy PCI values for 365-day input."""
        numpy_result = numpy_pci(daily_rainfall_365.values)
        np.testing.assert_allclose(
            pci_xarray_result_365.values,
            numpy_result[0],
            atol=1e-8,
            err_msg="PCI xarray vs NumPy equivalence failed (365 days)",
        )

    def test_values_match_numpy_366(
        self,
        daily_rainfall_366: xr.DataArray,
        pci_xarray_result_366: xr.DataArray,
    ) -> None:
        """xarray PCI values should match NumPy PCI values for 366-day input."""
        numpy_result = numpy_pci(daily_rainfall_366.values)
        np.testing.assert_allclose(
            pci_xarray_result_366.values,
            numpy_result[0],
            atol=1e-8,
            err_msg="PCI xarray vs NumPy equivalence failed (366 days)",
        )

    def test_output_is_dataarray(self, pci_xarray_result_365: xr.DataArray) -> None:
        """xarray input should produce xarray output."""
        assert isinstance(pci_xarray_result_365, xr.DataArray)

    def test_output_is_scalar(self, pci_xarray_result_365: xr.DataArray) -> None:
        """PCI output should be a scalar (0-D) DataArray."""
        assert pci_xarray_result_365.ndim == 0


class TestPCIXarrayCFMetadata:
    """Validate CF Convention metadata on PCI xarray output."""

    def test_has_long_name(self, pci_xarray_result_365: xr.DataArray) -> None:
        assert "long_name" in pci_xarray_result_365.attrs
        assert pci_xarray_result_365.attrs["long_name"] == "Precipitation Concentration Index"

    def test_has_units(self, pci_xarray_result_365: xr.DataArray) -> None:
        assert "units" in pci_xarray_result_365.attrs
        # PCI is dimensionless
        assert pci_xarray_result_365.attrs["units"] == ""

    def test_has_references(self, pci_xarray_result_365: xr.DataArray) -> None:
        assert "references" in pci_xarray_result_365.attrs
        assert "Oliver" in pci_xarray_result_365.attrs["references"]

    def test_has_version(self, pci_xarray_result_365: xr.DataArray) -> None:
        assert "climate_indices_version" in pci_xarray_result_365.attrs

    def test_has_history(self, pci_xarray_result_365: xr.DataArray) -> None:
        assert "history" in pci_xarray_result_365.attrs
        assert "PCI" in pci_xarray_result_365.attrs["history"]


class TestPCINumpyPassthrough:
    """Validate NumPy path is unchanged when called through typed API."""

    def test_numpy_returns_ndarray(self) -> None:
        """NumPy input should return ndarray, not DataArray."""
        rng = np.random.default_rng(123)
        values = rng.gamma(shape=2.0, scale=5.0, size=365)
        result = pci(rainfall_mm=values)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_numpy_366_returns_ndarray(self) -> None:
        """NumPy input with 366 days should return ndarray."""
        rng = np.random.default_rng(456)
        values = rng.gamma(shape=2.0, scale=5.0, size=366)
        result = pci(rainfall_mm=values)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_invalid_length_raises(self) -> None:
        """Input with wrong number of days should raise ValueError."""
        values = np.ones(300)
        with pytest.raises(ValueError, match="366 or 365"):
            pci(rainfall_mm=values)
