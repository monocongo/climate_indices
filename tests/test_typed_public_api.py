"""Tests for the typed public API with NumPy/xarray overloads."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import spei, spi
from climate_indices.compute import Periodicity
from climate_indices.indices import Distribution
from climate_indices.xarray_adapter import xarray_adapter


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
        # import the original function and wrap it manually
        from climate_indices.indices import spi as indices_spi
        from climate_indices.xarray_adapter import CF_METADATA

        manual_wrapped = xarray_adapter(
            cf_metadata=CF_METADATA["spi"],
            index_display_name="SPI",
            calculation_metadata_keys=["scale", "distribution", "calibration_year_initial", "calibration_year_final"],
        )(indices_spi)

        result_typed = spi(
            values=sample_monthly_precip_da,
            scale=6,
            distribution=Distribution.gamma,
        )

        result_manual = manual_wrapped(
            sample_monthly_precip_da,
            scale=6,
            distribution=Distribution.gamma,
        )

        # compare values and coordinates (ignoring timestamp in history attribute)
        xr.testing.assert_equal(result_typed, result_manual)

        # verify both have history attributes with matching content (ignoring timestamp prefix)
        assert "history" in result_typed.attrs
        assert "history" in result_manual.attrs
        # history format: "YYYY-MM-DD HH:MM:SS climate_indices <version> SPI(...)"
        # extract content after timestamp (skip first 20 chars: "YYYY-MM-DD HH:MM:SS ")
        history_typed_content = result_typed.attrs["history"][20:]
        history_manual_content = result_manual.attrs["history"][20:]
        assert history_typed_content == history_manual_content

        # verify all non-history attributes are identical
        attrs_typed = {k: v for k, v in result_typed.attrs.items() if k != "history"}
        attrs_manual = {k: v for k, v in result_manual.attrs.items() if k != "history"}
        assert attrs_typed == attrs_manual


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
        # import the original function and wrap it manually
        from climate_indices.indices import spei as indices_spei
        from climate_indices.xarray_adapter import CF_METADATA

        manual_wrapped = xarray_adapter(
            cf_metadata=CF_METADATA["spei"],
            index_display_name="SPEI",
            calculation_metadata_keys=["scale", "distribution", "calibration_year_initial", "calibration_year_final"],
            additional_input_names=["pet_mm"],
        )(indices_spei)

        result_typed = spei(
            precips_mm=sample_monthly_precip_da,
            pet_mm=sample_monthly_pet_da,
            scale=6,
            distribution=Distribution.gamma,
        )

        result_manual = manual_wrapped(
            sample_monthly_precip_da,
            sample_monthly_pet_da,
            scale=6,
            distribution=Distribution.gamma,
        )

        # compare values and coordinates (ignoring timestamp in history attribute)
        xr.testing.assert_equal(result_typed, result_manual)

        # verify both have history attributes with matching content (ignoring timestamp prefix)
        assert "history" in result_typed.attrs
        assert "history" in result_manual.attrs
        # history format: "YYYY-MM-DD HH:MM:SS climate_indices <version> SPEI(...)"
        # extract content after timestamp (skip first 20 chars: "YYYY-MM-DD HH:MM:SS ")
        history_typed_content = result_typed.attrs["history"][20:]
        history_manual_content = result_manual.attrs["history"][20:]
        assert history_typed_content == history_manual_content

        # verify all non-history attributes are identical
        attrs_typed = {k: v for k, v in result_typed.attrs.items() if k != "history"}
        attrs_manual = {k: v for k, v in result_manual.attrs.items() if k != "history"}
        assert attrs_typed == attrs_manual


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
