"""Tests for the CF metadata registry module.

Validates registry structure, required keys, and specific entry values
for all climate indices that produce xarray DataArray output.
"""

from __future__ import annotations

import pytest

from climate_indices.cf_metadata_registry import CF_METADATA

# expected registry keys
EXPECTED_KEYS = {
    "spi",
    "spei",
    "pet_thornthwaite",
    "pet_hargreaves",
    "percentage_of_normal",
    "pci",
    "pnp",
    "pdsi",
    "phdi",
    "pmdi",
    "z_index",
}

REQUIRED_FIELDS = {"long_name", "units", "references"}


class TestRegistryStructure:
    """Validate overall registry structure and completeness."""

    def test_registry_contains_all_expected_keys(self) -> None:
        """Registry has entries for all indices."""
        assert set(CF_METADATA.keys()) == EXPECTED_KEYS

    def test_registry_entry_count(self) -> None:
        """Registry has exactly 11 entries."""
        assert len(CF_METADATA) == 11

    @pytest.mark.parametrize("index_name", sorted(EXPECTED_KEYS))
    def test_entry_has_required_fields(self, index_name: str) -> None:
        """Each entry contains long_name, units, and references."""
        entry = CF_METADATA[index_name]
        actual_keys = set(entry.keys())
        assert REQUIRED_FIELDS.issubset(actual_keys), (
            f"Entry '{index_name}' missing required keys: {REQUIRED_FIELDS - actual_keys}"
        )

    @pytest.mark.parametrize("index_name", sorted(EXPECTED_KEYS))
    def test_all_values_are_strings(self, index_name: str) -> None:
        """All metadata values are strings (units may be empty for dimensionless indices)."""
        for key, value in CF_METADATA[index_name].items():
            assert isinstance(value, str), f"'{index_name}'.'{key}' is not a string"
            # units can be empty string for dimensionless indices like PCI
            if key != "units":
                assert value.strip(), f"'{index_name}'.'{key}' is empty or whitespace"


class TestPercentageOfNormalEntry:
    """Validate percentage_of_normal registry entry."""

    def test_long_name(self) -> None:
        assert CF_METADATA["percentage_of_normal"]["long_name"] == "Percent of Normal Precipitation"

    def test_units(self) -> None:
        assert CF_METADATA["percentage_of_normal"]["units"] == "%"

    def test_references_contains_willeke(self) -> None:
        references = CF_METADATA["percentage_of_normal"]["references"]
        assert "Willeke" in references
        assert "1994" in references


class TestPCIEntry:
    """Validate PCI registry entry."""

    def test_long_name(self) -> None:
        assert CF_METADATA["pci"]["long_name"] == "Precipitation Concentration Index"

    def test_units(self) -> None:
        assert CF_METADATA["pci"]["units"] == ""

    def test_references_contains_oliver(self) -> None:
        references = CF_METADATA["pci"]["references"]
        assert "Oliver" in references
        assert "1980" in references


class TestPNPEntry:
    """Validate PNP registry entry (alias for percentage_of_normal)."""

    def test_long_name(self) -> None:
        assert CF_METADATA["pnp"]["long_name"] == "Percent of Normal Precipitation"

    def test_units(self) -> None:
        assert CF_METADATA["pnp"]["units"] == "%"

    def test_pnp_matches_percentage_of_normal(self) -> None:
        """PNP and percentage_of_normal share identical metadata."""
        pnp = CF_METADATA["pnp"]
        pon = CF_METADATA["percentage_of_normal"]
        assert pnp["long_name"] == pon["long_name"]
        assert pnp["units"] == pon["units"]
        assert pnp["references"] == pon["references"]


class TestBackwardCompatibility:
    """Verify existing SPI/SPEI/PET entries unchanged after extraction."""

    def test_spi_entry_exists(self) -> None:
        assert "spi" in CF_METADATA

    def test_spi_long_name(self) -> None:
        assert CF_METADATA["spi"]["long_name"] == "Standardized Precipitation Index"

    def test_spi_units(self) -> None:
        assert CF_METADATA["spi"]["units"] == "dimensionless"

    def test_spi_references_contains_mckee(self) -> None:
        assert "McKee" in CF_METADATA["spi"]["references"]

    def test_spei_entry_exists(self) -> None:
        assert "spei" in CF_METADATA

    def test_spei_long_name(self) -> None:
        assert CF_METADATA["spei"]["long_name"] == "Standardized Precipitation Evapotranspiration Index"

    def test_pet_thornthwaite_entry_exists(self) -> None:
        assert "pet_thornthwaite" in CF_METADATA

    def test_pet_hargreaves_entry_exists(self) -> None:
        assert "pet_hargreaves" in CF_METADATA

    def test_importable_from_xarray_adapter(self) -> None:
        """CF_METADATA is still importable from xarray_adapter for backward compat."""
        from climate_indices.xarray_adapter import CF_METADATA as xa_cf_metadata

        assert xa_cf_metadata is CF_METADATA


class TestPDSIEntry:
    """Validate PDSI registry entry."""

    def test_long_name(self) -> None:
        assert CF_METADATA["pdsi"]["long_name"] == "Palmer Drought Severity Index"

    def test_units(self) -> None:
        assert CF_METADATA["pdsi"]["units"] == ""

    def test_references_contains_palmer(self) -> None:
        references = CF_METADATA["pdsi"]["references"]
        assert "Palmer" in references
        assert "1965" in references

    def test_references_contains_research_paper(self) -> None:
        references = CF_METADATA["pdsi"]["references"]
        assert "Meteorological Drought" in references


class TestPHDIEntry:
    """Validate PHDI registry entry."""

    def test_long_name(self) -> None:
        assert CF_METADATA["phdi"]["long_name"] == "Palmer Hydrological Drought Index"

    def test_units(self) -> None:
        assert CF_METADATA["phdi"]["units"] == ""

    def test_references_contains_palmer(self) -> None:
        references = CF_METADATA["phdi"]["references"]
        assert "Palmer" in references
        assert "1965" in references


class TestPMDIEntry:
    """Validate PMDI registry entry."""

    def test_long_name(self) -> None:
        assert CF_METADATA["pmdi"]["long_name"] == "Palmer Modified Drought Index"

    def test_units(self) -> None:
        assert CF_METADATA["pmdi"]["units"] == ""

    def test_references_contains_heddinghaus(self) -> None:
        """PMDI uses Heddinghaus & Sabol (1991) as primary reference."""
        references = CF_METADATA["pmdi"]["references"]
        assert "Heddinghaus" in references
        assert "1991" in references

    def test_references_contains_sabol(self) -> None:
        references = CF_METADATA["pmdi"]["references"]
        assert "Sabol" in references


class TestZIndexEntry:
    """Validate Z-Index registry entry."""

    def test_long_name(self) -> None:
        assert CF_METADATA["z_index"]["long_name"] == "Palmer Z-Index"

    def test_units(self) -> None:
        assert CF_METADATA["z_index"]["units"] == ""

    def test_references_contains_palmer(self) -> None:
        references = CF_METADATA["z_index"]["references"]
        assert "Palmer" in references
        assert "1965" in references


class TestPalmerEntriesCommon:
    """Cross-cutting tests for all Palmer registry entries."""

    @pytest.mark.parametrize("var_name", ["pdsi", "phdi", "pmdi", "z_index"])
    def test_palmer_entries_are_dimensionless(self, var_name: str) -> None:
        """All Palmer indices are dimensionless (empty string units)."""
        assert CF_METADATA[var_name]["units"] == ""

    @pytest.mark.parametrize("var_name", ["pdsi", "phdi", "z_index"])
    def test_palmer_1965_entries_share_reference(self, var_name: str) -> None:
        """PDSI, PHDI, and Z-Index all cite Palmer (1965)."""
        references = CF_METADATA[var_name]["references"]
        assert "Palmer, W. C. (1965)" in references

    def test_pmdi_has_different_reference(self) -> None:
        """PMDI cites Heddinghaus & Sabol, not Palmer (1965)."""
        references = CF_METADATA["pmdi"]["references"]
        assert "Palmer, W. C. (1965)" not in references
        assert "Heddinghaus" in references
