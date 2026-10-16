"""Table-driven tests for the CF metadata registry.

Each table below is the single owner of one contract: the exact key set, the
per-entry structure, the documented field values, and the reference
attributions. Adding an index means adding table rows, not new test functions.
"""

from __future__ import annotations

import pytest

from climate_indices.cf_metadata_registry import CF_METADATA

# expected registry keys
EXPECTED_KEYS = {
    "spi",
    "spi_probability",
    "spi_bounded",
    "spei",
    "spei_probability",
    "spei_bounded",
    "standardized_index",
    "pet_thornthwaite",
    "pet_hargreaves",
    "pet_penman_monteith",
    "percentage_of_normal",
    "pci",
    "pnp",
    "eddi",
    "pdsi",
    "phdi",
    "pmdi",
    "z_index",
    "scpdsi",
    "effective_precipitation",
    "edi",
    "flood_index",
    "antecedent_precipitation_index",
    "kbdi",
    "kbdi_imperial",
    "ffwi",
    "hdw",
    "ffmc",
    "dmc",
    "dc",
    "isi",
    "bui",
    "fwi",
    "dsr",
    "haines_low",
    "haines_mid",
    "haines_high",
    "heating_degree_days",
    "cooling_degree_days",
    "energy_degree_days",
    "hub_height_wind_speed",
    "wind_capacity_factor",
    "pv_capacity_factor",
    "srepi",
    "standardized_demand_index",
    "srli",
}

REQUIRED_FIELDS = {"long_name", "units", "references"}

FLOOD_KEYS = (
    "effective_precipitation",
    "edi",
    "flood_index",
    "antecedent_precipitation_index",
)

FIRE_KEYS = (
    "kbdi",
    "kbdi_imperial",
    "ffwi",
    "hdw",
    "ffmc",
    "dmc",
    "dc",
    "isi",
    "bui",
    "fwi",
    "dsr",
    "haines_low",
    "haines_mid",
    "haines_high",
)

ENERGY_KEYS = (
    "heating_degree_days",
    "cooling_degree_days",
    "energy_degree_days",
    "hub_height_wind_speed",
    "wind_capacity_factor",
    "pv_capacity_factor",
    "srepi",
    "standardized_demand_index",
    "srli",
)

DEGREE_DAY_KEYS = ("heating_degree_days", "cooling_degree_days", "energy_degree_days")

# the only entries with an official CF standard name (CF standard name table v95);
# every other entry must omit the field rather than invent one
STANDARD_NAMES = {
    "heating_degree_days": "integral_wrt_time_of_air_temperature_deficit",
    "cooling_degree_days": "integral_wrt_time_of_air_temperature_excess",
    "hub_height_wind_speed": "wind_speed",
}

# the only entries carrying the CF 1.11 temperature-difference qualifier
UNITS_METADATA: dict[str, str] = dict.fromkeys(list(DEGREE_DAY_KEYS), "temperature: difference")

# (entry, field, expected value) rows: one row per literal registry assertion.
ENTRY_VALUE_ROWS = [
    ("spi", "long_name", "Standardized Precipitation Index"),
    ("spi", "units", "dimensionless"),
    ("spi_probability", "long_name", "Standardized Precipitation Index probability"),
    ("spi_probability", "units", "1"),
    ("spi_probability", "climate_indices_variant", "probability"),
    ("spi_bounded", "long_name", "Standardized Precipitation Index bounded probability"),
    ("spi_bounded", "units", "1"),
    ("spi_bounded", "climate_indices_variant", "bounded"),
    ("spei", "long_name", "Standardized Precipitation Evapotranspiration Index"),
    ("spei", "units", "dimensionless"),
    ("spei_probability", "long_name", "Standardized Precipitation Evapotranspiration Index probability"),
    ("spei_probability", "units", "1"),
    ("spei_probability", "climate_indices_variant", "probability"),
    ("spei_bounded", "long_name", "Standardized Precipitation Evapotranspiration Index bounded probability"),
    ("spei_bounded", "units", "1"),
    ("spei_bounded", "climate_indices_variant", "bounded"),
    ("standardized_index", "long_name", "Standardized index"),
    ("standardized_index", "units", "1"),
    ("pet_thornthwaite", "long_name", "Potential Evapotranspiration (Thornthwaite method)"),
    ("pet_thornthwaite", "units", "mm/month"),
    ("pet_hargreaves", "long_name", "Potential Evapotranspiration (Hargreaves method)"),
    ("pet_hargreaves", "units", "mm/day"),
    ("pet_penman_monteith", "long_name", "Potential Evapotranspiration (FAO-56 Penman-Monteith method)"),
    ("pet_penman_monteith", "units", "mm/day"),
    ("percentage_of_normal", "long_name", "Percent of Normal Precipitation"),
    ("percentage_of_normal", "units", "%"),
    ("pci", "long_name", "Precipitation Concentration Index"),
    ("pci", "units", ""),
    ("pnp", "long_name", "Percent of Normal Precipitation"),
    ("pnp", "units", "%"),
    ("eddi", "long_name", "Evaporative Demand Drought Index"),
    ("eddi", "units", "dimensionless"),
    ("pdsi", "long_name", "Palmer Drought Severity Index"),
    ("pdsi", "units", "dimensionless"),
    ("phdi", "long_name", "Palmer Hydrological Drought Index"),
    ("phdi", "units", "dimensionless"),
    ("pmdi", "long_name", "Palmer Modified Drought Index"),
    ("pmdi", "units", "dimensionless"),
    ("z_index", "long_name", "Palmer Z-Index"),
    ("z_index", "units", "dimensionless"),
    ("scpdsi", "long_name", "Self-calibrated Palmer Drought Severity Index"),
    ("scpdsi", "units", "dimensionless"),
    ("effective_precipitation", "long_name", "Effective Precipitation"),
    ("effective_precipitation", "units", "mm"),
    ("edi", "long_name", "Effective Drought Index"),
    ("edi", "units", "dimensionless"),
    ("flood_index", "long_name", "Flood Index"),
    ("flood_index", "units", "dimensionless"),
    ("antecedent_precipitation_index", "long_name", "Antecedent Precipitation Index"),
    ("antecedent_precipitation_index", "units", "mm"),
    ("kbdi", "long_name", "Keetch-Byram Drought Index"),
    ("kbdi", "units", "mm"),
    ("kbdi", "climate_indices_variant", "metric"),
    ("kbdi_imperial", "long_name", "Keetch-Byram Drought Index"),
    ("kbdi_imperial", "units", "0.01 in"),
    ("kbdi_imperial", "climate_indices_variant", "imperial"),
    ("ffwi", "long_name", "Fosberg Fire Weather Index"),
    ("ffwi", "units", "dimensionless"),
    ("hdw", "long_name", "Hot-Dry-Windy Index"),
    ("hdw", "units", "hPa m s-1"),
    ("ffmc", "long_name", "Fine Fuel Moisture Code"),
    ("ffmc", "units", "dimensionless"),
    ("ffmc", "climate_indices_variant", "cffwis_classic"),
    ("dmc", "long_name", "Duff Moisture Code"),
    ("dmc", "units", "dimensionless"),
    ("dmc", "climate_indices_variant", "cffwis_classic"),
    ("dc", "long_name", "Drought Code"),
    ("dc", "units", "dimensionless"),
    ("dc", "climate_indices_variant", "cffwis_classic"),
    ("isi", "long_name", "Initial Spread Index"),
    ("isi", "units", "dimensionless"),
    ("isi", "climate_indices_variant", "cffwis_classic"),
    ("bui", "long_name", "Buildup Index"),
    ("bui", "units", "dimensionless"),
    ("bui", "climate_indices_variant", "cffwis_classic"),
    ("fwi", "long_name", "Canadian Fire Weather Index"),
    ("fwi", "units", "dimensionless"),
    ("fwi", "climate_indices_variant", "cffwis_classic"),
    ("dsr", "long_name", "Daily Severity Rating"),
    ("dsr", "units", "dimensionless"),
    ("dsr", "climate_indices_variant", "cffwis_classic"),
    ("haines_low", "long_name", "Haines Index"),
    ("haines_low", "units", "dimensionless"),
    ("haines_low", "climate_indices_variant", "low"),
    ("haines_mid", "long_name", "Haines Index"),
    ("haines_mid", "units", "dimensionless"),
    ("haines_mid", "climate_indices_variant", "mid"),
    ("haines_high", "long_name", "Haines Index"),
    ("haines_high", "units", "dimensionless"),
    ("haines_high", "climate_indices_variant", "high"),
    ("heating_degree_days", "long_name", "Heating Degree Days"),
    ("heating_degree_days", "units", "K d"),
    ("cooling_degree_days", "long_name", "Cooling Degree Days"),
    ("cooling_degree_days", "units", "K d"),
    ("energy_degree_days", "long_name", "Energy Degree Days"),
    ("energy_degree_days", "units", "K d"),
    ("hub_height_wind_speed", "long_name", "Hub-Height Wind Speed"),
    ("hub_height_wind_speed", "units", "m s-1"),
    ("wind_capacity_factor", "long_name", "Wind Capacity Factor"),
    ("wind_capacity_factor", "units", "1"),
    ("pv_capacity_factor", "long_name", "Photovoltaic Capacity Factor"),
    ("pv_capacity_factor", "units", "1"),
    ("srepi", "long_name", "Standardized Renewable Energy Production Index"),
    ("srepi", "units", "1"),
    ("standardized_demand_index", "long_name", "Standardized Demand Index"),
    ("standardized_demand_index", "units", "1"),
    ("srli", "long_name", "Standardized Residual Load Index"),
    ("srli", "units", "1"),
]

ENTRY_VALUES = [
    pytest.param(entry, field, expected, id=f"{entry}-{field}") for entry, field, expected in ENTRY_VALUE_ROWS
]

# (entry, reference fragments) rows: every fragment must appear in the entry's references.
REFERENCE_ROWS = [
    ("spi", ("McKee", "1993")),
    ("spi_probability", ("McKee", "1993")),
    ("spi_bounded", ("McKee", "1993")),
    ("spei", ("Vicente-Serrano", "2010")),
    ("spei_probability", ("Vicente-Serrano", "2010")),
    ("spei_bounded", ("Vicente-Serrano", "2010")),
    ("standardized_index", ("McKee", "1993")),
    ("pet_thornthwaite", ("Thornthwaite", "1948")),
    ("pet_hargreaves", ("Hargreaves", "1985")),
    ("pet_penman_monteith", ("Allen", "1998")),
    ("percentage_of_normal", ("Willeke", "1994")),
    ("pci", ("Oliver", "1980")),
    ("pnp", ("Willeke", "1994")),
    ("eddi", ("Hobbins", "2016")),
    ("pdsi", ("Palmer", "1965")),
    ("phdi", ("Palmer", "1965")),
    ("pmdi", ("Heddinghaus", "1991")),
    ("z_index", ("Palmer", "1965")),
    ("scpdsi", ("Wells", "2004")),
    ("effective_precipitation", ("Byun", "Wilhite", "1999", "Lee", "2002")),
    ("edi", ("Byun", "Wilhite", "1999", "Lee", "2002")),
    ("flood_index", ("Deo", "2015", "Moishin", "2021")),
    ("antecedent_precipitation_index", ("Kohler", "Linsley", "1951")),
    ("kbdi", ("Keetch", "1968")),
    ("kbdi_imperial", ("Keetch", "1968")),
    ("ffwi", ("Fosberg", "1978")),
    ("hdw", ("Srock", "2018")),
    ("ffmc", ("Van Wagner", "1985")),
    ("dmc", ("Van Wagner", "1985")),
    ("dc", ("Van Wagner", "1985")),
    ("isi", ("Van Wagner", "1985")),
    ("bui", ("Van Wagner", "1985")),
    ("fwi", ("Van Wagner", "1985")),
    # DSR is Eq. 31 of the 1987 report, not the 1985 equations report
    ("dsr", ("Van Wagner", "1987")),
    ("haines_low", ("Haines", "1988")),
    ("haines_mid", ("Haines", "1988")),
    ("haines_high", ("Haines", "1988")),
    ("heating_degree_days", ("C3S", "10.24381/3bb607bd")),
    ("cooling_degree_days", ("C3S", "10.24381/3bb607bd")),
    ("energy_degree_days", ("C3S", "10.24381/3bb607bd")),
    ("hub_height_wind_speed", ("C3S", "10.24381/3bb607bd")),
    ("wind_capacity_factor", ("C3S", "10.24381/3bb607bd")),
    ("pv_capacity_factor", ("Jerez", "2015")),
    ("srepi", ("Allen", "Otero", "2023")),
    ("standardized_demand_index", ("Allen", "Otero", "2023")),
    ("srli", ("Allen", "Otero", "2023")),
]

REFERENCE_CASES = [pytest.param(entry, fragments, id=entry) for entry, fragments in REFERENCE_ROWS]


def test_registry_has_exactly_the_expected_keys() -> None:
    """An added, renamed, or removed index fails here rather than in twenty value tests."""
    assert set(CF_METADATA) == EXPECTED_KEYS
    assert set(STANDARD_NAMES) <= EXPECTED_KEYS, "a stale standard-name allowlist key survives here"
    assert set(UNITS_METADATA) <= EXPECTED_KEYS, "a stale units-metadata allowlist key survives here"
    documented = {(entry, field) for entry, field, _ in ENTRY_VALUE_ROWS}
    referenced = {entry for entry, _ in REFERENCE_ROWS}
    for index_name in EXPECTED_KEYS:
        assert (index_name, "long_name") in documented, f"'{index_name}' has no documented long_name row"
        assert (index_name, "units") in documented, f"'{index_name}' has no documented units row"
        assert index_name in referenced, f"'{index_name}' has no reference row"


@pytest.mark.parametrize("index_name", sorted(EXPECTED_KEYS))
def test_entry_structure(index_name: str) -> None:
    """Every entry declares the required fields as populated strings."""
    entry = CF_METADATA[index_name]
    missing = REQUIRED_FIELDS - set(entry)
    assert not missing, f"Entry '{index_name}' missing required keys: {missing}"
    for field, value in entry.items():
        assert isinstance(value, str), f"'{index_name}'.'{field}' is not a string"
        # units may be empty for dimensionless indices such as PCI
        if field != "units":
            assert value.strip(), f"'{index_name}'.'{field}' is empty or whitespace"


@pytest.mark.parametrize(("index_name", "field", "expected"), ENTRY_VALUES)
def test_entry_field_value(index_name: str, field: str, expected: str) -> None:
    """Each documented (entry, field) value matches the registry exactly."""
    assert CF_METADATA[index_name][field] == expected


@pytest.mark.parametrize(("index_name", "fragments"), REFERENCE_CASES)
def test_references_cite_their_source(index_name: str, fragments: tuple[str, ...]) -> None:
    """Each entry's references name the publication it derives from."""
    references = CF_METADATA[index_name]["references"]
    for fragment in fragments:
        assert fragment in references, f"'{index_name}' references lack '{fragment}'"


@pytest.mark.parametrize("index_name", FLOOD_KEYS)
def test_flood_entries_describe_potential_without_inventing_standard_names(index_name: str) -> None:
    """Flood-family metadata must not claim to represent observed flooding or CF standard names."""
    assert "flood potential, not flooding" in CF_METADATA[index_name]["description"]
    assert "standard_name" not in CF_METADATA[index_name]


@pytest.mark.parametrize(
    ("index_name", "phrases"),
    [
        ("edi", ("Fixed-window", "positive values", "negative values")),
        ("flood_index", ("harmonic", "unverified")),
    ],
)
def test_flood_descriptions_disclose_scientific_conventions(index_name: str, phrases: tuple[str, ...]) -> None:
    """Describe EDI's sign and the flood index's unverified kernel lineage."""
    for phrase in phrases:
        assert phrase in CF_METADATA[index_name]["description"]


@pytest.mark.parametrize("index_name", FIRE_KEYS)
def test_fire_entries_describe_themselves_without_inventing_standard_names(index_name: str) -> None:
    """Fire adapters take their description from the registry; no fire entry claims a CF standard_name."""
    assert CF_METADATA[index_name].get("description", "").strip()
    assert "standard_name" not in CF_METADATA[index_name]


@pytest.mark.parametrize("index_name", sorted(EXPECTED_KEYS))
def test_entries_claim_no_invented_standard_names(index_name: str) -> None:
    """Only officially defined CF standard names appear; every other entry omits the field."""
    assert CF_METADATA[index_name].get("standard_name") == STANDARD_NAMES.get(index_name)


@pytest.mark.parametrize("index_name", sorted(EXPECTED_KEYS))
def test_entries_qualify_only_temperature_differences(index_name: str) -> None:
    """Only a degree day is a temperature difference, so only it carries the CF 1.11 qualifier."""
    assert CF_METADATA[index_name].get("units_metadata") == UNITS_METADATA.get(index_name)


@pytest.mark.parametrize("index_name", ENERGY_KEYS)
def test_energy_entries_describe_themselves(index_name: str) -> None:
    """Energy adapters take their description from the registry."""
    assert CF_METADATA[index_name].get("description", "").strip()


@pytest.mark.parametrize("index_name", sorted(set(ENERGY_KEYS) - {"hub_height_wind_speed"}))
def test_energy_entries_disclaim_actual_generation_and_load(index_name: str) -> None:
    """ADR-0019 decision 4: energy indices describe the climate-driven component, not actual generation or load."""
    description = CF_METADATA[index_name].get("description", "")
    assert "climate-driven component" in description
    assert "not actual generation" in description or "not actual load" in description


@pytest.mark.parametrize("index_name", DEGREE_DAY_KEYS)
def test_degree_days_are_labelled_celsius_day(index_name: str) -> None:
    """ADR-0019 decision 6 labels a degree day °C day; K d is the same temperature difference."""
    assert "°C day" in CF_METADATA[index_name].get("description", "")


def test_registry_aliases_and_variants_stay_consistent() -> None:
    """PNP aliases percentage_of_normal; the two KBDI unit scales remain one index."""
    pnp = CF_METADATA["pnp"]
    percentage_of_normal = CF_METADATA["percentage_of_normal"]
    for field in REQUIRED_FIELDS:
        assert pnp[field] == percentage_of_normal[field]

    kbdi = CF_METADATA["kbdi"]
    kbdi_imperial = CF_METADATA["kbdi_imperial"]
    assert kbdi["long_name"] == kbdi_imperial["long_name"]
    assert kbdi["units"] != kbdi_imperial["units"]
    assert kbdi["climate_indices_variant"] != kbdi_imperial["climate_indices_variant"]


def test_registry_remains_importable_from_the_adapter() -> None:
    """CF_METADATA is still importable from xarray_adapter for backward compatibility."""
    from climate_indices.xarray_adapter import CF_METADATA as adapter_metadata

    assert adapter_metadata is CF_METADATA
