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
}

REQUIRED_FIELDS = {"long_name", "units", "references"}


class TestRegistryStructure:
    """Validate overall registry structure and completeness."""

    def test_registry_contains_all_expected_keys(self) -> None:
        """Registry has entries for all indices."""
        assert set(CF_METADATA.keys()) == EXPECTED_KEYS

    def test_registry_entry_count(self) -> None:
        """Registry has exactly 7 entries."""
        assert len(CF_METADATA) == 7

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
