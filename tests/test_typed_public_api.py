"""Tests for the typed public API with NumPy/xarray overloads."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import xarray as xr

if sys.version_info >= (3, 11):
    from typing import get_overloads
else:  # pragma: no cover - overload introspection is unavailable before 3.11
    get_overloads = None

from climate_indices import (
    eddi,
    indices,
    pci,
    percentage_of_normal,
    pet_hargreaves,
    pet_thornthwaite,
    spei,
    spi,
)
from climate_indices.compute import Periodicity
from climate_indices.indices import Distribution
from climate_indices.xarray_adapter import (
    pet_hargreaves as pet_hargreaves_impl,
)
from climate_indices.xarray_adapter import (
    pet_thornthwaite as pet_thornthwaite_impl,
)
from climate_indices.xarray_adapter import xarray_adapter

# fixtures now consolidated in conftest.py


def _verify_xarray_matches_manual_wrapping(
    typed_func: Callable[..., xr.DataArray],
    indices_func: Callable[..., np.ndarray],
    cf_metadata: dict[str, Any],
    index_display_name: str,
    calculation_metadata_keys: list[str],
    typed_call_kwargs: dict[str, Any],
    manual_call_args: tuple[Any, ...],
    manual_call_kwargs: dict[str, Any],
    additional_input_names: list[str] | None = None,
) -> None:
    """Helper to verify typed function matches manually-wrapped version.

    Args:
        typed_func: The typed public API function (e.g., spi, spei)
        indices_func: The original indices module function
        cf_metadata: CF metadata dict for the index
        index_display_name: Display name for the index (e.g., "SPI", "SPEI")
        calculation_metadata_keys: Keys to include in calculation metadata
        typed_call_kwargs: Kwargs to pass to typed function
        manual_call_args: Positional args to pass to manually wrapped function
        manual_call_kwargs: Kwargs to pass to manually wrapped function
        additional_input_names: Additional input parameter names for adapter
    """
    # manually wrap the function
    adapter_kwargs: dict[str, Any] = {
        "cf_metadata": cf_metadata,
        "index_display_name": index_display_name,
        "calculation_metadata_keys": calculation_metadata_keys,
    }
    if additional_input_names:
        adapter_kwargs["additional_input_names"] = additional_input_names

    manual_wrapped = xarray_adapter(**adapter_kwargs)(indices_func)

    # call both versions
    result_typed = typed_func(**typed_call_kwargs)
    result_manual = manual_wrapped(*manual_call_args, **manual_call_kwargs)

    # compare values and coordinates (ignoring timestamp in history attribute)
    xr.testing.assert_equal(result_typed, result_manual)

    # verify both have history attributes with matching content (ignoring timestamp prefix)
    assert "history" in result_typed.attrs
    assert "history" in result_manual.attrs
    # history format: "YYYY-MM-DD HH:MM:SS climate_indices <version> <INDEX>(...)"
    # extract content after timestamp (skip first 20 chars: "YYYY-MM-DD HH:MM:SS ")
    history_typed_content = result_typed.attrs["history"][20:]
    history_manual_content = result_manual.attrs["history"][20:]
    assert history_typed_content == history_manual_content

    # verify all non-history attributes are identical
    attrs_typed = {k: v for k, v in result_typed.attrs.items() if k != "history"}
    attrs_manual = {k: v for k, v in result_manual.attrs.items() if k != "history"}
    assert attrs_typed == attrs_manual


# public function -> (implementation it mirrors, parameters the public API does not expose)
_PUBLIC_IMPLEMENTATIONS: dict[Callable[..., Any], tuple[Callable[..., Any], tuple[str, ...]]] = {
    spi: (indices.spi, ("spatial_time_major",)),
    spei: (indices.spei, ("spatial_time_major",)),
    eddi: (indices.eddi, ("spatial_time_major",)),
    percentage_of_normal: (indices.percentage_of_normal, ("spatial_time_major",)),
    pci: (indices.pci, ()),
    pet_thornthwaite: (pet_thornthwaite_impl, ()),
    pet_hargreaves: (pet_hargreaves_impl, ()),
}

# the frozen published typing contract: rendered (NumPy, xarray) @overload signatures
_EXPECTED_OVERLOADS: dict[Callable[..., Any], tuple[str, str]] = {
    spi: (
        "(values: 'npt.NDArray[np.float64]', scale: 'int', distribution: 'Distribution', data_start_year: 'int', calibration_year_initial: 'int', calibration_year_final: 'int', periodicity: 'Periodicity', fitting_params: 'dict[str, Any] | None' = None) -> 'npt.NDArray[np.float64]'",
        "(values: 'xr.DataArray', scale: 'int', distribution: 'Distribution', data_start_year: 'int | None' = None, calibration_year_initial: 'int | None' = None, calibration_year_final: 'int | None' = None, periodicity: 'Periodicity | None' = None, fitting_params: 'dict[str, Any] | None' = None) -> 'xr.DataArray'",
    ),
    spei: (
        "(precips_mm: 'npt.NDArray[np.float64]', pet_mm: 'npt.NDArray[np.float64]', scale: 'int', distribution: 'Distribution', periodicity: 'Periodicity', data_start_year: 'int', calibration_year_initial: 'int', calibration_year_final: 'int', fitting_params: 'dict[str, Any] | None' = None) -> 'npt.NDArray[np.float64]'",
        "(precips_mm: 'xr.DataArray', pet_mm: 'xr.DataArray', scale: 'int', distribution: 'Distribution', periodicity: 'Periodicity | None' = None, data_start_year: 'int | None' = None, calibration_year_initial: 'int | None' = None, calibration_year_final: 'int | None' = None, fitting_params: 'dict[str, Any] | None' = None) -> 'xr.DataArray'",
    ),
    percentage_of_normal: (
        "(values: 'npt.NDArray[np.float64]', scale: 'int', data_start_year: 'int', calibration_start_year: 'int', calibration_end_year: 'int', periodicity: 'Periodicity') -> 'npt.NDArray[np.float64]'",
        "(values: 'xr.DataArray', scale: 'int', data_start_year: 'int | None' = None, calibration_start_year: 'int | None' = None, calibration_end_year: 'int | None' = None, periodicity: 'Periodicity | None' = None) -> 'xr.DataArray'",
    ),
    eddi: (
        "(pet_values: 'npt.NDArray[np.float64]', scale: 'int', data_start_year: 'int', calibration_year_initial: 'int', calibration_year_final: 'int', periodicity: 'Periodicity') -> 'npt.NDArray[np.float64]'",
        "(pet_values: 'xr.DataArray', scale: 'int', data_start_year: 'int | None' = None, calibration_year_initial: 'int | None' = None, calibration_year_final: 'int | None' = None, periodicity: 'Periodicity | None' = None) -> 'xr.DataArray'",
    ),
    pci: (
        "(rainfall_mm: 'npt.NDArray[np.float64]') -> 'npt.NDArray[np.float64]'",
        "(rainfall_mm: 'xr.DataArray') -> 'xr.DataArray'",
    ),
    pet_thornthwaite: (
        "(temperature: 'npt.NDArray[np.float64]', latitude: 'float', data_start_year: 'int', time_dim: 'str' = 'time') -> 'npt.NDArray[np.float64]'",
        "(temperature: 'xr.DataArray', latitude: 'float | np.floating | xr.DataArray', data_start_year: 'int | None' = None, time_dim: 'str' = 'time') -> 'xr.DataArray'",
    ),
    pet_hargreaves: (
        "(daily_tmin_celsius: 'npt.NDArray[np.float64]', daily_tmax_celsius: 'npt.NDArray[np.float64]', latitude: 'float', time_dim: 'str' = 'time') -> 'npt.NDArray[np.float64]'",
        "(daily_tmin_celsius: 'xr.DataArray', daily_tmax_celsius: 'xr.DataArray', latitude: 'float | np.floating | xr.DataArray', time_dim: 'str' = 'time') -> 'xr.DataArray'",
    ),
}


# only Python 3.11+ registers @overload stubs at runtime, so the introspection
# tests below cannot see them on 3.10 (mypy still checks the stubs statically)
_requires_overload_registry = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="typing.get_overloads requires the Python 3.11+ overload registry",
)


@_requires_overload_registry
def test_overloads_mirror_implementations() -> None:
    """Public @overload stubs must mirror the implementation they delegate to (issue #903)."""
    for public, (implementation, internal) in _PUBLIC_IMPLEMENTATIONS.items():
        implementation_params = [name for name in inspect.signature(implementation).parameters if name not in internal]
        overloads = get_overloads(public)
        assert len(overloads) == 2, f"{public.__name__} must keep its NumPy and xarray overloads"

        numpy_params = list(inspect.signature(overloads[0]).parameters)
        assert numpy_params == implementation_params, (
            f"{public.__name__} NumPy overload drifted from {implementation.__name__}: "
            f"{numpy_params} != {implementation_params}"
        )

        # both stubs expose every non-internal parameter of the implementation they
        # mirror; required-ness and input types are the overloads' own contract
        xarray_params = list(inspect.signature(overloads[1]).parameters)
        assert xarray_params == implementation_params, (
            f"{public.__name__} xarray overload drifted from {implementation.__name__}: "
            f"{xarray_params} != {implementation_params}"
        )


@_requires_overload_registry
def test_overload_signatures_are_frozen() -> None:
    """The overload signatures are the published typing contract; freeze them (issue #903)."""
    for public, expected in _EXPECTED_OVERLOADS.items():
        actual = tuple(str(inspect.signature(overload)) for overload in get_overloads(public))
        assert actual == expected, f"{public.__name__} overload signatures changed: {actual!r} != {expected!r}"


@_requires_overload_registry
def test_overload_tests_cover_every_public_overloaded_function() -> None:
    """A new overloaded public function must be added to the drift tests above."""
    import climate_indices

    overloaded = {
        member
        for name in climate_indices.__all__
        if callable(member := getattr(climate_indices, name)) and get_overloads(member)
    }
    assert overloaded == set(_PUBLIC_IMPLEMENTATIONS)


def _assert_same_result(left: xr.DataArray, right: xr.DataArray) -> None:
    """Assert equal values and metadata, ignoring the timestamped history attribute."""
    np.testing.assert_array_equal(left.values, right.values)
    assert {key: value for key, value in left.attrs.items() if key != "history"} == {
        key: value for key, value in right.attrs.items() if key != "history"
    }


class TestDelegateForwarding:
    """The generic implementations forward calls exactly as the explicit ones did (issue #903)."""

    def test_spi_xarray_positional_arguments_match_keywords(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """Positional scale/distribution keep the values and calculation metadata."""
        keyword_result = spi(values=sample_monthly_precip_da, scale=6, distribution=Distribution.gamma)
        positional_result = spi(sample_monthly_precip_da, 6, Distribution.gamma)

        assert positional_result.attrs["scale"] == 6
        assert positional_result.attrs["distribution"] == "gamma"
        assert positional_result.attrs["calibration_year_initial"] == keyword_result.attrs["calibration_year_initial"]
        _assert_same_result(positional_result, keyword_result)

    def test_spi_dask_positional_arguments_match_keywords(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """The Dask execution path also receives positionally passed parameters."""
        chunked = sample_monthly_precip_da.chunk({"time": -1})
        keyword_result = spi(values=chunked, scale=6, distribution=Distribution.gamma).compute()
        positional_result = spi(chunked, 6, Distribution.gamma).compute()

        assert positional_result.attrs["scale"] == 6
        _assert_same_result(positional_result, keyword_result)

    def test_unknown_keyword_argument_is_rejected(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """A misspelled optional parameter raises instead of being silently dropped."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            spi(
                values=sample_monthly_precip_da,
                scale=6,
                distribution=Distribution.gamma,
                calibraton_year_initial=1981,
            )

    def test_explicit_none_matches_omitted(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """Explicit None for an inferred parameter is dropped, positionally or by keyword."""
        omitted = spi(values=sample_monthly_precip_da, scale=6, distribution=Distribution.gamma)
        keyword_none = spi(
            values=sample_monthly_precip_da,
            scale=6,
            distribution=Distribution.gamma,
            data_start_year=None,
            calibration_year_initial=None,
            calibration_year_final=None,
            periodicity=None,
        )
        positional_none = spi(sample_monthly_precip_da, 6, Distribution.gamma, None)

        _assert_same_result(keyword_none, omitted)
        _assert_same_result(positional_none, omitted)


class TestSPIOverloads:
    """Test SPI function overloads for NumPy and xarray inputs."""

    def test_spi_numpy_returns_ndarray(self) -> None:
        """NumPy input should return numpy.ndarray."""
        # 40 years * 12 months = 480 values
        rng = np.random.default_rng(42)
        values = rng.gamma(shape=2.0, scale=50.0, size=480)

        result = spi(
            values=values,
            scale=6,
            distribution=Distribution.gamma,
            data_start_year=1980,
            calibration_year_initial=1980,
            calibration_year_final=2019,
            periodicity=Periodicity.monthly,
        )

        assert isinstance(result, np.ndarray)
        assert not isinstance(result, xr.DataArray)
        assert result.shape == values.shape

    def test_spi_xarray_returns_dataarray(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """xarray input should return xarray.DataArray."""
        result = spi(
            values=sample_monthly_precip_da,
            scale=6,
            distribution=Distribution.gamma,
            # temporal params are optional for xarray
        )

        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_monthly_precip_da.shape
        # verify CF metadata was applied
        assert "long_name" in result.attrs
        assert result.attrs["long_name"] == "Standardized Precipitation Index"

    def test_spi_xarray_temporal_params_optional(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """xarray inputs can omit temporal params (inferred from coordinates)."""
        result = spi(
            values=sample_monthly_precip_da,
            scale=3,
            distribution=Distribution.gamma,
        )

        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_monthly_precip_da.shape

    def test_spi_xarray_explicit_params(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """xarray inputs can provide explicit temporal params."""
        result = spi(
            values=sample_monthly_precip_da,
            scale=6,
            distribution=Distribution.gamma,
            data_start_year=1980,
            calibration_year_initial=1985,
            calibration_year_final=2015,
            periodicity=Periodicity.monthly,
        )

        assert isinstance(result, xr.DataArray)
        # verify metadata reflects explicit params
        assert result.attrs["calibration_year_initial"] == 1985
        assert result.attrs["calibration_year_final"] == 2015

    def test_spi_numpy_matches_indices_module(self) -> None:
        """Results for NumPy inputs should match climate_indices.indices.spi."""
        # import the original function for comparison
        from climate_indices.indices import spi as indices_spi

        rng = np.random.default_rng(42)
        values = rng.gamma(shape=2.0, scale=50.0, size=480)

        result_typed = spi(
            values=values,
            scale=6,
            distribution=Distribution.gamma,
            data_start_year=1980,
            calibration_year_initial=1980,
            calibration_year_final=2019,
            periodicity=Periodicity.monthly,
        )

        result_original = indices_spi(
            values=values,
            scale=6,
            distribution=Distribution.gamma,
            data_start_year=1980,
            calibration_year_initial=1980,
            calibration_year_final=2019,
            periodicity=Periodicity.monthly,
        )

        np.testing.assert_array_equal(result_typed, result_original)

    def test_spi_xarray_matches_manual_wrapping(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """Results for xarray inputs should match manually-wrapped function."""
        from climate_indices.indices import spi as indices_spi
        from climate_indices.xarray_adapter import CF_METADATA

        _verify_xarray_matches_manual_wrapping(
            typed_func=spi,
            indices_func=indices_spi,
            cf_metadata=CF_METADATA["spi"],
            index_display_name="SPI",
            calculation_metadata_keys=["scale", "distribution", "calibration_year_initial", "calibration_year_final"],
            typed_call_kwargs={
                "values": sample_monthly_precip_da,
                "scale": 6,
                "distribution": Distribution.gamma,
            },
            manual_call_args=(sample_monthly_precip_da,),
            manual_call_kwargs={
                "scale": 6,
                "distribution": Distribution.gamma,
            },
        )


class TestSPEIOverloads:
    """Test SPEI function overloads for NumPy and xarray inputs."""

    def test_spei_numpy_returns_ndarray(self) -> None:
        """NumPy input should return numpy.ndarray."""
        # 40 years * 12 months = 480 values
        rng = np.random.default_rng(42)
        precips = rng.gamma(shape=2.0, scale=50.0, size=480)
        pet = rng.gamma(shape=2.0, scale=30.0, size=480)

        result = spei(
            precips_mm=precips,
            pet_mm=pet,
            scale=6,
            distribution=Distribution.gamma,
            periodicity=Periodicity.monthly,
            data_start_year=1980,
            calibration_year_initial=1980,
            calibration_year_final=2019,
        )

        assert isinstance(result, np.ndarray)
        assert not isinstance(result, xr.DataArray)
        assert result.shape == precips.shape

    def test_spei_xarray_returns_dataarray(
        self, sample_monthly_precip_da: xr.DataArray, sample_monthly_pet_da: xr.DataArray
    ) -> None:
        """xarray input should return xarray.DataArray."""
        result = spei(
            precips_mm=sample_monthly_precip_da,
            pet_mm=sample_monthly_pet_da,
            scale=6,
            distribution=Distribution.gamma,
            # temporal params are optional for xarray
        )

        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_monthly_precip_da.shape
        # verify CF metadata was applied
        assert "long_name" in result.attrs
        assert result.attrs["long_name"] == "Standardized Precipitation Evapotranspiration Index"

    def test_spei_xarray_temporal_params_optional(
        self, sample_monthly_precip_da: xr.DataArray, sample_monthly_pet_da: xr.DataArray
    ) -> None:
        """xarray inputs can omit temporal params (inferred from coordinates)."""
        result = spei(
            precips_mm=sample_monthly_precip_da,
            pet_mm=sample_monthly_pet_da,
            scale=3,
            distribution=Distribution.gamma,
        )

        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_monthly_precip_da.shape

    def test_spei_xarray_explicit_params(
        self, sample_monthly_precip_da: xr.DataArray, sample_monthly_pet_da: xr.DataArray
    ) -> None:
        """xarray inputs can provide explicit temporal params."""
        result = spei(
            precips_mm=sample_monthly_precip_da,
            pet_mm=sample_monthly_pet_da,
            scale=6,
            distribution=Distribution.gamma,
            periodicity=Periodicity.monthly,
            data_start_year=1980,
            calibration_year_initial=1985,
            calibration_year_final=2015,
        )

        assert isinstance(result, xr.DataArray)
        # verify metadata reflects explicit params
        assert result.attrs["calibration_year_initial"] == 1985
        assert result.attrs["calibration_year_final"] == 2015

    def test_spei_numpy_matches_indices_module(self) -> None:
        """Results for NumPy inputs should match climate_indices.indices.spei."""
        # import the original function for comparison
        from climate_indices.indices import spei as indices_spei

        rng = np.random.default_rng(42)
        precips = rng.gamma(shape=2.0, scale=50.0, size=480)
        pet = rng.gamma(shape=2.0, scale=30.0, size=480)

        result_typed = spei(
            precips_mm=precips,
            pet_mm=pet,
            scale=6,
            distribution=Distribution.gamma,
            periodicity=Periodicity.monthly,
            data_start_year=1980,
            calibration_year_initial=1980,
            calibration_year_final=2019,
        )

        result_original = indices_spei(
            precips_mm=precips,
            pet_mm=pet,
            scale=6,
            distribution=Distribution.gamma,
            periodicity=Periodicity.monthly,
            data_start_year=1980,
            calibration_year_initial=1980,
            calibration_year_final=2019,
        )

        np.testing.assert_array_equal(result_typed, result_original)

    def test_spei_xarray_matches_manual_wrapping(
        self, sample_monthly_precip_da: xr.DataArray, sample_monthly_pet_da: xr.DataArray
    ) -> None:
        """Results for xarray inputs should match manually-wrapped function."""
        from climate_indices.indices import spei as indices_spei
        from climate_indices.xarray_adapter import CF_METADATA

        _verify_xarray_matches_manual_wrapping(
            typed_func=spei,
            indices_func=indices_spei,
            cf_metadata=CF_METADATA["spei"],
            index_display_name="SPEI",
            calculation_metadata_keys=["scale", "distribution", "calibration_year_initial", "calibration_year_final"],
            typed_call_kwargs={
                "precips_mm": sample_monthly_precip_da,
                "pet_mm": sample_monthly_pet_da,
                "scale": 6,
                "distribution": Distribution.gamma,
            },
            manual_call_args=(sample_monthly_precip_da, sample_monthly_pet_da),
            manual_call_kwargs={
                "scale": 6,
                "distribution": Distribution.gamma,
            },
            additional_input_names=["pet_mm"],
        )


class TestModuleExports:
    """Test that functions are properly exported from the main module."""

    def test_import_from_main_module(self) -> None:
        """Should be able to import spi and spei from climate_indices."""
        from climate_indices import spei, spi

        assert callable(spi)
        assert callable(spei)

    def test_module_all_contains_exports(self) -> None:
        """__all__ should contain spi and spei."""
        import climate_indices

        assert "spi" in climate_indices.__all__
        assert "spei" in climate_indices.__all__
