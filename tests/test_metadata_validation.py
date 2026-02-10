"""Integration-level metadata validation tests for xarray climate index outputs.

Story 4.2 (FR-TEST-002): Validates the full metadata contract when calling
the public API (spi(), spei()) with xarray DataArray inputs.

This file focuses on integration-level testing with real computations through
the typed_public_api.py entry points. Unit-level metadata tests (CF_METADATA dict
structure, _serialize_attr_value, _build_history_entry) are in test_xarray_adapter.py.

Test organization:
- Module-scoped fixtures cache expensive SPI/SPEI computation results
- Class-based structure groups related metadata validation assertions
- Tests use session-scoped fixtures from conftest.py for consistent test data
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import xarray as xr

from climate_indices import __version__, spei, spi
from climate_indices.indices import Distribution

# ==============================================================================
# module-scoped result fixtures: cache expensive computations
# ==============================================================================


@pytest.fixture(scope="module")
def spi_1d_result(sample_monthly_precip_da: xr.DataArray) -> xr.DataArray:
    """Cached SPI-6 gamma result for 1D monthly precipitation data."""
    return spi(
        values=sample_monthly_precip_da,
        scale=6,
        distribution=Distribution.gamma,
    )


@pytest.fixture(scope="module")
def spei_1d_result(
    sample_monthly_precip_da: xr.DataArray,
    sample_monthly_pet_da: xr.DataArray,
) -> xr.DataArray:
    """Cached SPEI-6 gamma result for 1D monthly precipitation and PET data."""
    return spei(
        precips_mm=sample_monthly_precip_da,
        pet_mm=sample_monthly_pet_da,
        scale=6,
        distribution=Distribution.gamma,
    )


@pytest.fixture(scope="module")
def spi_dask_1d_result(dask_monthly_precip_1d: xr.DataArray) -> xr.DataArray:
    """Cached SPI-6 gamma result for 1D Dask-backed precipitation data."""
    return spi(
        values=dask_monthly_precip_1d,
        scale=6,
        distribution=Distribution.gamma,
    )


# ==============================================================================
# test class 1: SPI coordinate preservation
# ==============================================================================


class TestSPICoordinatePreservation:
    """Validate SPI output coordinates exactly match input through real computation.

    Tests coordinate preservation across 1D time series, 3D gridded data, and
    Dask-backed arrays. Ensures no coordinate drift, missing coords, or dimension
    ordering changes during SPI computation.
    """

    def test_time_coords_match_input(
        self,
        sample_monthly_precip_da: xr.DataArray,
        spi_1d_result: xr.DataArray,
    ):
        """Time coordinates in SPI output should exactly match input."""
        xr.testing.assert_equal(
            spi_1d_result.coords["time"],
            sample_monthly_precip_da.coords["time"],
        )

    def test_dims_match_input(
        self,
        sample_monthly_precip_da: xr.DataArray,
        spi_1d_result: xr.DataArray,
    ):
        """Dimension names and order should be preserved."""
        assert spi_1d_result.dims == sample_monthly_precip_da.dims

    def test_coord_keys_exact(
        self,
        sample_monthly_precip_da: xr.DataArray,
        spi_1d_result: xr.DataArray,
    ):
        """No extra or missing coordinate keys should appear in output."""
        assert set(spi_1d_result.coords.keys()) == set(sample_monthly_precip_da.coords.keys())

    def test_output_shape_matches_input(
        self,
        sample_monthly_precip_da: xr.DataArray,
        spi_1d_result: xr.DataArray,
    ):
        """Output shape should exactly match input shape."""
        assert spi_1d_result.shape == sample_monthly_precip_da.shape


# ==============================================================================
# test class 2: SPEI coordinate preservation
# ==============================================================================


class TestSPEICoordinatePreservation:
    """Validate SPEI output coordinates match input through multi-input alignment.

    Tests coordinate preservation when SPEI aligns precipitation and PET inputs.
    Ensures the alignment path doesn't introduce coordinate drift or artifacts.
    """

    def test_time_coords_match_input(
        self,
        sample_monthly_precip_da: xr.DataArray,
        spei_1d_result: xr.DataArray,
    ):
        """Time coordinates in SPEI output should exactly match input."""
        xr.testing.assert_equal(
            spei_1d_result.coords["time"],
            sample_monthly_precip_da.coords["time"],
        )

    def test_dims_match_input(
        self,
        sample_monthly_precip_da: xr.DataArray,
        spei_1d_result: xr.DataArray,
    ):
        """Dimension names and order should be preserved."""
        assert spei_1d_result.dims == sample_monthly_precip_da.dims

    def test_coord_keys_exact(
        self,
        sample_monthly_precip_da: xr.DataArray,
        spei_1d_result: xr.DataArray,
    ):
        """No extra or missing coordinate keys should appear in output."""
        assert set(spei_1d_result.coords.keys()) == set(sample_monthly_precip_da.coords.keys())

    def test_output_shape_matches_input(
        self,
        sample_monthly_precip_da: xr.DataArray,
        spei_1d_result: xr.DataArray,
    ):
        """Output shape should exactly match input shape."""
        assert spei_1d_result.shape == sample_monthly_precip_da.shape


# ==============================================================================
# test class 3: SPI CF attributes
# ==============================================================================


class TestSPICFAttributes:
    """Validate required CF Convention attributes on actual SPI output.

    Tests CF-compliant metadata (long_name, units, references) and version
    attributes on real SPI computation results. Ensures types and values match
    the CF metadata contract.
    """

    def test_has_long_name(self, spi_1d_result: xr.DataArray):
        """SPI output should have long_name attribute."""
        assert "long_name" in spi_1d_result.attrs
        assert spi_1d_result.attrs["long_name"] == "Standardized Precipitation Index"

    def test_has_units(self, spi_1d_result: xr.DataArray):
        """SPI output should have units attribute."""
        assert "units" in spi_1d_result.attrs
        assert spi_1d_result.attrs["units"] == "dimensionless"

    def test_has_references(self, spi_1d_result: xr.DataArray):
        """SPI output should have references attribute."""
        assert "references" in spi_1d_result.attrs
        references = spi_1d_result.attrs["references"]
        # verify citation includes key elements
        assert "McKee" in references
        assert "1993" in references

    def test_long_name_is_string(self, spi_1d_result: xr.DataArray):
        """long_name attribute should be a string type."""
        assert isinstance(spi_1d_result.attrs["long_name"], str)

    def test_units_is_string(self, spi_1d_result: xr.DataArray):
        """units attribute should be a string type (not int or other)."""
        assert isinstance(spi_1d_result.attrs["units"], str)

    def test_references_is_string(self, spi_1d_result: xr.DataArray):
        """references attribute should be a string type."""
        assert isinstance(spi_1d_result.attrs["references"], str)

    def test_has_version_attribute(self, spi_1d_result: xr.DataArray):
        """SPI output should include climate_indices_version attribute."""
        assert "climate_indices_version" in spi_1d_result.attrs

    def test_version_matches_package(self, spi_1d_result: xr.DataArray):
        """climate_indices_version should match package __version__."""
        assert spi_1d_result.attrs["climate_indices_version"] == __version__

    def test_version_is_string(self, spi_1d_result: xr.DataArray):
        """climate_indices_version attribute should be a string type."""
        assert isinstance(spi_1d_result.attrs["climate_indices_version"], str)


# ==============================================================================
# test class 4: SPEI CF attributes
# ==============================================================================


class TestSPEICFAttributes:
    """Validate required CF Convention attributes on actual SPEI output.

    Tests CF-compliant metadata for SPEI computation results. Ensures SPEI-specific
    long_name and references are correctly applied.
    """

    def test_has_long_name(self, spei_1d_result: xr.DataArray):
        """SPEI output should have long_name attribute."""
        assert "long_name" in spei_1d_result.attrs
        assert spei_1d_result.attrs["long_name"] == "Standardized Precipitation Evapotranspiration Index"

    def test_has_units(self, spei_1d_result: xr.DataArray):
        """SPEI output should have units attribute."""
        assert "units" in spei_1d_result.attrs
        assert spei_1d_result.attrs["units"] == "dimensionless"

    def test_has_references(self, spei_1d_result: xr.DataArray):
        """SPEI output should have references attribute."""
        assert "references" in spei_1d_result.attrs
        references = spei_1d_result.attrs["references"]
        # verify citation includes key elements
        assert "Vicente-Serrano" in references
        assert "2010" in references

    def test_long_name_is_string(self, spei_1d_result: xr.DataArray):
        """long_name attribute should be a string type."""
        assert isinstance(spei_1d_result.attrs["long_name"], str)

    def test_units_is_string(self, spei_1d_result: xr.DataArray):
        """units attribute should be a string type (not int or other)."""
        assert isinstance(spei_1d_result.attrs["units"], str)

    def test_references_is_string(self, spei_1d_result: xr.DataArray):
        """references attribute should be a string type."""
        assert isinstance(spei_1d_result.attrs["references"], str)

    def test_has_version_attribute(self, spei_1d_result: xr.DataArray):
        """SPEI output should include climate_indices_version attribute."""
        assert "climate_indices_version" in spei_1d_result.attrs

    def test_version_matches_package(self, spei_1d_result: xr.DataArray):
        """climate_indices_version should match package __version__."""
        assert spei_1d_result.attrs["climate_indices_version"] == __version__

    def test_version_is_string(self, spei_1d_result: xr.DataArray):
        """climate_indices_version attribute should be a string type."""
        assert isinstance(spei_1d_result.attrs["climate_indices_version"], str)


# ==============================================================================
# test class 5: calculation metadata integration
# ==============================================================================


class TestCalculationMetadataIntegration:
    """Validate calculation parameters appear in output attrs with correct serialized types.

    Tests that scale, distribution, and calibration period parameters are recorded in
    output attributes with proper type serialization (e.g., Distribution enum → string).
    """

    def test_spi_scale_in_attrs(self, spi_1d_result: xr.DataArray):
        """Scale parameter should be recorded in output attributes."""
        assert "scale" in spi_1d_result.attrs
        assert spi_1d_result.attrs["scale"] == 6

    def test_spi_scale_is_int(self, spi_1d_result: xr.DataArray):
        """Scale attribute should be an integer type."""
        assert isinstance(spi_1d_result.attrs["scale"], int)

    def test_spi_distribution_in_attrs(self, spi_1d_result: xr.DataArray):
        """Distribution parameter should be serialized to string in output attributes."""
        assert "distribution" in spi_1d_result.attrs
        # Distribution enum should be serialized to .name string
        assert spi_1d_result.attrs["distribution"] == "gamma"

    def test_spi_distribution_is_string(self, spi_1d_result: xr.DataArray):
        """Distribution attribute should be a string (serialized from enum)."""
        assert isinstance(spi_1d_result.attrs["distribution"], str)

    def test_spei_scale_in_attrs(self, spei_1d_result: xr.DataArray):
        """SPEI scale parameter should be recorded in output attributes."""
        assert "scale" in spei_1d_result.attrs
        assert spei_1d_result.attrs["scale"] == 6

    def test_spei_distribution_in_attrs(self, spei_1d_result: xr.DataArray):
        """SPEI distribution parameter should be serialized to string in output attributes."""
        assert "distribution" in spei_1d_result.attrs
        assert spei_1d_result.attrs["distribution"] == "gamma"

    @pytest.mark.parametrize("scale", [1, 3, 6, 12])
    def test_spi_parametrized_scales(
        self,
        sample_monthly_precip_da: xr.DataArray,
        scale: int,
    ):
        """Each scale value should be correctly recorded in output attributes."""
        result = spi(
            values=sample_monthly_precip_da,
            scale=scale,
            distribution=Distribution.gamma,
        )
        assert result.attrs["scale"] == scale

    @pytest.mark.parametrize("distribution", [Distribution.gamma, Distribution.pearson])
    def test_spi_parametrized_distributions(
        self,
        sample_monthly_precip_da: xr.DataArray,
        distribution: Distribution,
    ):
        """Each distribution value should be serialized to its name string."""
        result = spi(
            values=sample_monthly_precip_da,
            scale=6,
            distribution=distribution,
        )
        # verify serialized as string name
        assert result.attrs["distribution"] == distribution.name
        assert isinstance(result.attrs["distribution"], str)


# ==============================================================================
# test class 6: history provenance integration
# ==============================================================================


class TestHistoryProvenanceIntegration:
    """Validate history attribute on actual computation results.

    Tests that history entries are created with correct timestamps, index labels,
    scale information, distribution details, and version tracking. Ensures existing
    history is preserved when present.
    """

    def test_history_present(self, spi_1d_result: xr.DataArray):
        """History attribute should be present in output."""
        assert "history" in spi_1d_result.attrs

    def test_history_is_string(self, spi_1d_result: xr.DataArray):
        """History attribute should be a string type."""
        assert isinstance(spi_1d_result.attrs["history"], str)

    def test_history_contains_iso_timestamp(self, spi_1d_result: xr.DataArray):
        """History should contain ISO 8601 UTC timestamp."""
        history = spi_1d_result.attrs["history"]
        # match pattern: YYYY-MM-DDTHH:MM:SSZ
        iso_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
        assert re.search(iso_pattern, history), f"No ISO timestamp found in history: {history}"

    def test_history_contains_index_and_scale(self, spi_1d_result: xr.DataArray):
        """History should contain index label with scale (e.g., 'SPI-6')."""
        history = spi_1d_result.attrs["history"]
        assert "SPI-6" in history

    def test_history_contains_distribution(self, spi_1d_result: xr.DataArray):
        """History should mention the distribution used."""
        history = spi_1d_result.attrs["history"]
        assert "gamma distribution" in history

    def test_history_contains_version(self, spi_1d_result: xr.DataArray):
        """History should include climate_indices version."""
        history = spi_1d_result.attrs["history"]
        assert "climate_indices v" in history
        # verify version string is present
        assert __version__ in history

    def test_spei_history_contains_label(self, spei_1d_result: xr.DataArray):
        """SPEI history should contain SPEI label (not SPI)."""
        history = spei_1d_result.attrs["history"]
        assert "SPEI-6" in history

    def test_history_preserves_existing(self, sample_monthly_precip_da: xr.DataArray):
        """Pre-existing history entries should be preserved in output."""
        # create input with pre-existing history
        input_with_history = sample_monthly_precip_da.copy(deep=True)
        existing_history = "2020-01-01T00:00:00Z: Data ingested from source"
        input_with_history.attrs["history"] = existing_history

        # compute SPI with input that has history
        result = spi(
            values=input_with_history,
            scale=6,
            distribution=Distribution.gamma,
        )

        # verify both entries present (newline-delimited)
        result_history = result.attrs["history"]
        assert existing_history in result_history
        assert "SPI-6" in result_history
        # verify newline separation
        assert "\n" in result_history


# ==============================================================================
# test class 7: input attribute preservation
# ==============================================================================


class TestInputAttributePreservation:
    """Validate custom input attributes survive (or are correctly overridden).

    Tests that non-CF attributes from input are preserved in output, while
    CF attributes are correctly overridden. Ensures coordinate attributes and
    DataArray names survive computation.
    """

    def test_preserves_custom_attrs(self, sample_monthly_precip_da: xr.DataArray):
        """Custom attributes not in CF metadata should be preserved."""
        # add custom attribute to input
        input_with_custom = sample_monthly_precip_da.copy(deep=True)
        input_with_custom.attrs["institution"] = "NOAA"
        input_with_custom.attrs["contact"] = "climate@noaa.gov"

        result = spi(
            values=input_with_custom,
            scale=6,
            distribution=Distribution.gamma,
        )

        # verify custom attributes preserved
        assert result.attrs["institution"] == "NOAA"
        assert result.attrs["contact"] == "climate@noaa.gov"

    def test_cf_overrides_conflicting_attrs(self, sample_monthly_precip_da: xr.DataArray):
        """CF metadata should override conflicting input attributes."""
        # create input with conflicting units
        input_with_conflict = sample_monthly_precip_da.copy(deep=True)
        input_with_conflict.attrs["units"] = "mm"
        input_with_conflict.attrs["long_name"] = "Precipitation"

        result = spi(
            values=input_with_conflict,
            scale=6,
            distribution=Distribution.gamma,
        )

        # verify CF metadata wins
        assert result.attrs["units"] == "dimensionless"
        assert result.attrs["long_name"] == "Standardized Precipitation Index"

    def test_preserves_coord_attrs(self, sample_monthly_precip_da: xr.DataArray):
        """Coordinate attributes should survive computation."""
        # add coordinate attribute
        input_with_coord_attrs = sample_monthly_precip_da.copy(deep=True)
        input_with_coord_attrs.coords["time"].attrs["axis"] = "T"
        input_with_coord_attrs.coords["time"].attrs["standard_name"] = "time"

        result = spi(
            values=input_with_coord_attrs,
            scale=6,
            distribution=Distribution.gamma,
        )

        # verify coordinate attributes preserved
        assert result.coords["time"].attrs["axis"] == "T"
        assert result.coords["time"].attrs["standard_name"] == "time"

    def test_preserves_dataarray_name(self, sample_monthly_precip_da: xr.DataArray):
        """DataArray .name attribute should be preserved."""
        # set name on input
        input_with_name = sample_monthly_precip_da.copy(deep=True)
        input_with_name.name = "precip_monthly"

        result = spi(
            values=input_with_name,
            scale=6,
            distribution=Distribution.gamma,
        )

        # verify name preserved
        assert result.name == "precip_monthly"


# ==============================================================================
# test class 8: dask chunk preservation
# ==============================================================================


class TestDaskChunkPreservation:
    """Validate Dask-backed arrays retain chunk structure and metadata.

    Tests that Dask-backed inputs produce Dask-backed outputs with matching chunk
    structure, preserved coordinates, and full CF metadata on lazy results.
    """

    def test_1d_output_is_dask_backed(self, spi_dask_1d_result: xr.DataArray):
        """Dask input should produce Dask output (lazy evaluation)."""
        assert spi_dask_1d_result.chunks is not None

    def test_1d_time_chunk_preserved(
        self,
        dask_monthly_precip_1d: xr.DataArray,
        spi_dask_1d_result: xr.DataArray,
    ):
        """Single time chunk should be preserved in output."""
        # verify input has single time chunk
        assert len(dask_monthly_precip_1d.chunks[0]) == 1
        # verify output has single time chunk
        assert len(spi_dask_1d_result.chunks[0]) == 1

    def test_dask_cf_attrs_present(self, spi_dask_1d_result: xr.DataArray):
        """CF metadata should be present on lazy Dask result."""
        # verify CF attributes without triggering computation
        assert spi_dask_1d_result.attrs["long_name"] == "Standardized Precipitation Index"
        assert spi_dask_1d_result.attrs["units"] == "dimensionless"
        assert "references" in spi_dask_1d_result.attrs

    def test_dask_history_present(self, spi_dask_1d_result: xr.DataArray):
        """History attribute should be present on lazy Dask result."""
        assert "history" in spi_dask_1d_result.attrs
        assert "SPI-6" in spi_dask_1d_result.attrs["history"]

    def test_dask_coords_match_input(
        self,
        dask_monthly_precip_1d: xr.DataArray,
        spi_dask_1d_result: xr.DataArray,
    ):
        """Coordinates should match input exactly on lazy Dask result."""
        xr.testing.assert_equal(
            spi_dask_1d_result.coords["time"],
            dask_monthly_precip_1d.coords["time"],
        )

    def test_dask_computed_values_finite(self, spi_dask_1d_result: xr.DataArray):
        """Computed Dask result should produce at least some non-NaN values."""
        # trigger computation
        computed = spi_dask_1d_result.compute()
        # verify we have some valid values (not all NaN)
        assert np.isfinite(computed.values).any()
