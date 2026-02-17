"""Tests for Penman-Monteith ETo xarray adapter (Story 2.6).

Validates:
- NumPy passthrough produces identical results to pm_eto.pm_eto
- xarray DataArray inputs produce DataArray outputs
- CF metadata is correctly applied
- Coordinate preservation
- Input alignment across DataArrays
- Dask compatibility
- Equivalence between NumPy and xarray paths (tolerance 1e-8)
- @overload signatures in typed_public_api
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices.pm_eto import pm_eto
from climate_indices.xarray_adapter import CF_METADATA, eto_penman_monteith

# ---------------------------------------------------------------------------
# Shared test data (FAO-56 Example 18 values)
# ---------------------------------------------------------------------------

# scalar values from FAO-56 Example 18 (pg 74)
FAO56_KWARGS = {
    "net_radiation": 13.28,
    "soil_heat_flux": 0.14,
    "temperature_celsius": 16.9,
    "wind_speed_2m": 2.078,
    "saturation_vp": 1.997,
    "actual_vp": 1.409,
    "delta": 0.122,
    "gamma": 0.0666,
}


def _make_numpy_kwargs(n: int = 10) -> dict[str, np.ndarray]:
    """Create numpy array kwargs with n copies of the FAO-56 example values."""
    return {k: np.full(n, v) for k, v in FAO56_KWARGS.items()}


def _make_xarray_kwargs(n: int = 10) -> dict[str, xr.DataArray]:
    """Create xarray DataArray kwargs with time coordinate."""
    time = pd.date_range("2020-01-01", periods=n, freq="D")
    return {k: xr.DataArray(np.full(n, v), coords={"time": time}, dims=["time"]) for k, v in FAO56_KWARGS.items()}


# ---------------------------------------------------------------------------
# Test: CF metadata registry entry
# ---------------------------------------------------------------------------


class TestCFMetadata:
    """Verify CF metadata entry for pet_penman_monteith."""

    def test_entry_exists(self) -> None:
        assert "pet_penman_monteith" in CF_METADATA

    def test_long_name(self) -> None:
        attrs = CF_METADATA["pet_penman_monteith"]
        assert attrs["long_name"] == "Reference Evapotranspiration (Penman-Monteith FAO56)"

    def test_units(self) -> None:
        attrs = CF_METADATA["pet_penman_monteith"]
        assert attrs["units"] == "mm day-1"

    def test_references(self) -> None:
        attrs = CF_METADATA["pet_penman_monteith"]
        assert "Allen" in attrs["references"]
        assert "FAO" in attrs["references"]


# ---------------------------------------------------------------------------
# Test: NumPy passthrough
# ---------------------------------------------------------------------------


class TestNumpyPassthrough:
    """NumPy inputs should pass through directly to pm_eto.pm_eto."""

    def test_returns_ndarray(self) -> None:
        kwargs = _make_numpy_kwargs()
        result = eto_penman_monteith(**kwargs)
        assert isinstance(result, np.ndarray)

    def test_matches_pm_eto_directly(self) -> None:
        kwargs = _make_numpy_kwargs()
        adapter_result = eto_penman_monteith(**kwargs)
        direct_result = pm_eto(**kwargs)
        np.testing.assert_array_equal(adapter_result, direct_result)

    def test_scalar_inputs(self) -> None:
        """Scalar floats should also work (numpy-coercible)."""
        result = eto_penman_monteith(**FAO56_KWARGS)
        assert isinstance(result, np.floating | float)

    def test_list_inputs(self) -> None:
        """List inputs should be numpy-coercible."""
        kwargs = {k: [v, v] for k, v in FAO56_KWARGS.items()}
        result = eto_penman_monteith(**kwargs)
        assert hasattr(result, "__len__")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Test: xarray path
# ---------------------------------------------------------------------------


class TestXarrayPath:
    """xarray DataArray inputs should be handled by the adapter."""

    def test_returns_dataarray(self) -> None:
        kwargs = _make_xarray_kwargs()
        result = eto_penman_monteith(**kwargs)
        assert isinstance(result, xr.DataArray)

    def test_preserves_time_coordinate(self) -> None:
        kwargs = _make_xarray_kwargs()
        result = eto_penman_monteith(**kwargs)
        assert "time" in result.coords
        xr.testing.assert_equal(result.coords["time"], kwargs["net_radiation"].coords["time"])

    def test_preserves_shape(self) -> None:
        n = 15
        kwargs = _make_xarray_kwargs(n=n)
        result = eto_penman_monteith(**kwargs)
        assert result.shape == (n,)

    def test_cf_metadata_applied(self) -> None:
        kwargs = _make_xarray_kwargs()
        result = eto_penman_monteith(**kwargs)
        assert result.attrs["long_name"] == "Reference Evapotranspiration (Penman-Monteith FAO56)"
        assert result.attrs["units"] == "mm day-1"
        assert "Allen" in result.attrs["references"]

    def test_version_attribute(self) -> None:
        kwargs = _make_xarray_kwargs()
        result = eto_penman_monteith(**kwargs)
        assert "climate_indices_version" in result.attrs

    def test_history_attribute(self) -> None:
        kwargs = _make_xarray_kwargs()
        result = eto_penman_monteith(**kwargs)
        assert "history" in result.attrs
        assert "ETo Penman-Monteith FAO56" in result.attrs["history"]


# ---------------------------------------------------------------------------
# Test: numpy vs xarray equivalence
# ---------------------------------------------------------------------------


class TestEquivalence:
    """NumPy and xarray paths must produce identical numerical results."""

    def test_equivalence_1d(self) -> None:
        n = 30
        np_kwargs = _make_numpy_kwargs(n=n)
        xa_kwargs = _make_xarray_kwargs(n=n)

        np_result = eto_penman_monteith(**np_kwargs)
        xa_result = eto_penman_monteith(**xa_kwargs)

        np.testing.assert_allclose(
            xa_result.values,
            np_result,
            atol=1e-8,
            err_msg="xarray and numpy results diverge",
        )

    def test_equivalence_varied_values(self) -> None:
        """Test with varying (non-constant) input values."""
        rng = np.random.default_rng(42)
        n = 50
        time = pd.date_range("2020-01-01", periods=n, freq="D")

        np_kwargs = {
            "net_radiation": rng.uniform(5, 20, n),
            "soil_heat_flux": rng.uniform(-0.5, 0.5, n),
            "temperature_celsius": rng.uniform(5, 35, n),
            "wind_speed_2m": rng.uniform(0.5, 5, n),
            "saturation_vp": rng.uniform(0.5, 5, n),
            "actual_vp": rng.uniform(0.3, 3, n),
            "delta": rng.uniform(0.05, 0.5, n),
            "gamma": rng.uniform(0.04, 0.08, n),
        }

        xa_kwargs = {k: xr.DataArray(v, coords={"time": time}, dims=["time"]) for k, v in np_kwargs.items()}

        np_result = eto_penman_monteith(**np_kwargs)
        xa_result = eto_penman_monteith(**xa_kwargs)

        np.testing.assert_allclose(xa_result.values, np_result, atol=1e-8)


# ---------------------------------------------------------------------------
# Test: input type validation
# ---------------------------------------------------------------------------


class TestInputTypeValidation:
    """Ensure mixed numpy/xarray inputs are rejected."""

    def test_mixed_types_raises(self) -> None:
        np_kwargs = _make_numpy_kwargs()
        xa_kwargs = _make_xarray_kwargs()

        # replace one numpy input with xarray
        mixed = dict(np_kwargs)
        mixed["temperature_celsius"] = xa_kwargs["temperature_celsius"]

        with pytest.raises(TypeError, match="same type"):
            eto_penman_monteith(**mixed)


# ---------------------------------------------------------------------------
# Test: input alignment
# ---------------------------------------------------------------------------


class TestInputAlignment:
    """Verify that xarray inputs with different time ranges are aligned."""

    def test_inner_join_alignment(self) -> None:
        """When inputs have different time ranges, inner join is used."""
        time_long = pd.date_range("2020-01-01", periods=20, freq="D")
        time_short = pd.date_range("2020-01-05", periods=10, freq="D")

        kwargs_long = {
            k: xr.DataArray(np.full(20, v), coords={"time": time_long}, dims=["time"]) for k, v in FAO56_KWARGS.items()
        }
        kwargs_short = {
            k: xr.DataArray(np.full(10, v), coords={"time": time_short}, dims=["time"]) for k, v in FAO56_KWARGS.items()
        }

        # mix long and short
        mixed = dict(kwargs_long)
        mixed["temperature_celsius"] = kwargs_short["temperature_celsius"]

        with pytest.warns(match="Input alignment"):
            result = eto_penman_monteith(**mixed)

        # the result should have 10 timesteps (inner join of 20 and 10 with overlap)
        # overlap is 2020-01-05 to 2020-01-14 = 10 days
        assert len(result.coords["time"]) == 10


# ---------------------------------------------------------------------------
# Test: multi-dimensional (gridded) inputs
# ---------------------------------------------------------------------------


class TestGriddedInput:
    """Test with multi-dimensional (time, lat, lon) DataArrays."""

    def test_2d_spatial(self) -> None:
        n_time = 10
        n_lat = 3
        n_lon = 2
        time = pd.date_range("2020-01-01", periods=n_time, freq="D")

        kwargs = {}
        for k, v in FAO56_KWARGS.items():
            data = np.full((n_time, n_lat, n_lon), v)
            kwargs[k] = xr.DataArray(
                data,
                coords={"time": time, "lat": [30, 35, 40], "lon": [-100, -90]},
                dims=["time", "lat", "lon"],
            )

        result = eto_penman_monteith(**kwargs)

        assert isinstance(result, xr.DataArray)
        assert result.shape == (n_time, n_lat, n_lon)
        assert set(result.dims) == {"time", "lat", "lon"}


# ---------------------------------------------------------------------------
# Test: typed_public_api overloads
# ---------------------------------------------------------------------------


class TestTypedPublicAPI:
    """Verify typed_public_api.eto_penman_monteith delegates correctly."""

    def test_numpy_through_typed_api(self) -> None:
        from climate_indices.typed_public_api import eto_penman_monteith as typed_pm

        kwargs = _make_numpy_kwargs()
        result = typed_pm(**kwargs)
        assert isinstance(result, np.ndarray)

    def test_xarray_through_typed_api(self) -> None:
        from climate_indices.typed_public_api import eto_penman_monteith as typed_pm

        kwargs = _make_xarray_kwargs()
        result = typed_pm(**kwargs)
        assert isinstance(result, xr.DataArray)

    def test_typed_api_matches_direct(self) -> None:
        from climate_indices.typed_public_api import eto_penman_monteith as typed_pm

        kwargs = _make_numpy_kwargs()
        typed_result = typed_pm(**kwargs)
        direct_result = eto_penman_monteith(**kwargs)
        np.testing.assert_array_equal(typed_result, direct_result)


# ---------------------------------------------------------------------------
# Test: Dask compatibility
# ---------------------------------------------------------------------------


class TestDaskCompatibility:
    """Verify Dask-backed DataArrays are handled correctly."""

    @pytest.fixture()
    def dask_kwargs(self) -> dict[str, xr.DataArray]:
        """Create Dask-backed xarray kwargs."""
        pytest.importorskip("dask.array")
        n = 10
        time = pd.date_range("2020-01-01", periods=n, freq="D")
        kwargs = {}
        for k, v in FAO56_KWARGS.items():
            da = xr.DataArray(np.full(n, v), coords={"time": time}, dims=["time"])
            kwargs[k] = da.chunk({"time": 5})
        return kwargs

    def test_dask_returns_dataarray(self, dask_kwargs: dict[str, xr.DataArray]) -> None:
        result = eto_penman_monteith(**dask_kwargs)
        assert isinstance(result, xr.DataArray)

    def test_dask_matches_eager(self, dask_kwargs: dict[str, xr.DataArray]) -> None:
        eager_kwargs = _make_xarray_kwargs(n=10)
        eager_result = eto_penman_monteith(**eager_kwargs)
        dask_result = eto_penman_monteith(**dask_kwargs)

        np.testing.assert_allclose(
            dask_result.values,
            eager_result.values,
            atol=1e-8,
        )
