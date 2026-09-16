"""CF Convention metadata registry for climate indices.

Centralizes CF-compliant metadata (long_name, units, references) for
climate indices, including some without an xarray adapter yet: an entry may
land ahead of its adapter (see docs/design/fire-subsystem.md), but no
adapter ships before its entry. Each entry follows the CF Conventions
(https://cfconventions.org/) attribute model.

This module is a leaf dependency with no local imports, ensuring it can
be safely imported by any module without circular dependency risk.
"""

from __future__ import annotations

from typing import TypedDict


class _CFAttributesRequired(TypedDict):
    """Required CF Convention metadata attributes."""

    long_name: str
    units: str
    references: str


class CFAttributes(_CFAttributesRequired, total=False):
    """CF Convention metadata attributes for a climate index.

    Required keys: long_name, units, references.
    Optional keys: standard_name (only when officially defined in CF
    conventions), description (free-text detail such as value range or a
    resolved-ambiguity note), climate_indices_variant (distinguishes entries
    for one index that has more than one output convention, e.g. KBDI's
    metric and imperial unit scales, or reserves an entry against a future
    convention, e.g. the CFFWIS moisture codes' `cffwis_classic` against a
    future FWI2025 variant).

    .. note:: Part of the beta xarray adapter layer. See :doc:`xarray_migration`.
    """

    standard_name: str
    description: str
    climate_indices_variant: str


# The CFFWIS moisture codes (FFMC, DMC, DC) share one source publication; keep the
# rendered `references` text identical across their entries.
_VAN_WAGNER_PICKETT_1985 = (
    "Van Wagner, C. E., & Pickett, T. L. (1985). "
    "Equations and FORTRAN program for the Canadian Forest Fire Weather Index System. "
    "Canadian Forestry Service, Forestry Technical Report 33."
)


CF_METADATA: dict[str, CFAttributes] = {
    "spi": {
        "long_name": "Standardized Precipitation Index",
        "units": "dimensionless",
        "references": (
            "McKee, T. B., Doesken, N. J., & Kleist, J. (1993). "
            "The relationship of drought frequency and duration to time scales. "
            "Proceedings of the 8th Conference on Applied Climatology, "
            "17-22 January, Anaheim, CA. "
            "American Meteorological Society, Boston, MA, 179-184."
        ),
    },
    "spei": {
        "long_name": "Standardized Precipitation Evapotranspiration Index",
        "units": "dimensionless",
        "references": (
            "Vicente-Serrano, S. M., Begueria, S., & Lopez-Moreno, J. I. (2010). "
            "A Multiscalar Drought Index Sensitive to Global Warming: "
            "The Standardized Precipitation Evapotranspiration Index. "
            "Journal of Climate, 23(7), 1696-1718. "
            "https://doi.org/10.1175/2009JCLI2909.1"
        ),
    },
    "pet_thornthwaite": {
        "long_name": "Potential Evapotranspiration (Thornthwaite method)",
        "units": "mm/month",
        "references": (
            "Thornthwaite, C. W. (1948). "
            "An approach toward a rational classification of climate. "
            "Geographical Review, 38(1), 55-94. "
            "https://doi.org/10.2307/210739"
        ),
    },
    "pet_hargreaves": {
        "long_name": "Potential Evapotranspiration (Hargreaves method)",
        "units": "mm/day",
        "references": (
            "Hargreaves, G. H., & Samani, Z. A. (1985). "
            "Reference crop evapotranspiration from temperature. "
            "Applied Engineering in Agriculture, 1(2), 96-99. "
            "https://doi.org/10.13031/2013.26773"
        ),
    },
    "percentage_of_normal": {
        "long_name": "Percent of Normal Precipitation",
        "units": "%",
        "references": (
            "Willeke, G., Hosking, J. R. M., Wallis, J. R., & Guttman, N. B. (1994). "
            "The National Drought Atlas. Institute for Water Resources Report 94-NDS-4, "
            "U.S. Army Corps of Engineers."
        ),
    },
    "pci": {
        "long_name": "Precipitation Concentration Index",
        "units": "",
        "references": (
            "Oliver, J. E. (1980). "
            "Monthly precipitation distribution: A comparative index. "
            "The Professional Geographer, 32(3), 300-309. "
            "https://doi.org/10.1111/j.0033-0124.1980.00300.x"
        ),
    },
    "pnp": {
        "long_name": "Percent of Normal Precipitation",
        "units": "%",
        "references": (
            "Willeke, G., Hosking, J. R. M., Wallis, J. R., & Guttman, N. B. (1994). "
            "The National Drought Atlas. Institute for Water Resources Report 94-NDS-4, "
            "U.S. Army Corps of Engineers."
        ),
    },
    "eddi": {
        "long_name": "Evaporative Demand Drought Index",
        "units": "dimensionless",
        "references": (
            "Hobbins, M. T., Wood, A., McEvoy, D. J., Huntington, J. L., Morton, C., "
            "Anderson, M., & Hain, C. (2016). "
            "The Evaporative Demand Drought Index. Part I: Linking Drought Evolution "
            "to Variations in Evaporative Demand. "
            "Journal of Hydrometeorology, 17(6), 1745-1761. "
            "https://doi.org/10.1175/JHM-D-15-0121.1"
        ),
    },
    "pdsi": {
        "long_name": "Palmer Drought Severity Index",
        "units": "dimensionless",
        "references": (
            "Palmer, W. C. (1965). "
            "Meteorological Drought. Research Paper No. 45. "
            "U.S. Department of Commerce, Weather Bureau, Washington, D.C."
        ),
    },
    "phdi": {
        "long_name": "Palmer Hydrological Drought Index",
        "units": "dimensionless",
        "references": (
            "Palmer, W. C. (1965). "
            "Meteorological Drought. Research Paper No. 45. "
            "U.S. Department of Commerce, Weather Bureau, Washington, D.C."
        ),
    },
    "pmdi": {
        "long_name": "Palmer Modified Drought Index",
        "units": "dimensionless",
        "references": (
            "Heddinghaus, T. R., & Sabol, P. (1991). "
            "A review of the Palmer Drought Severity Index and where do we go from here? "
            "Preprints, 7th Conference on Applied Climatology, "
            "September 10-13, Salt Lake City, UT. "
            "American Meteorological Society, Boston, MA, 242-246."
        ),
    },
    "z_index": {
        "long_name": "Palmer Z-Index",
        "units": "dimensionless",
        "references": (
            "Palmer, W. C. (1965). "
            "Meteorological Drought. Research Paper No. 45. "
            "U.S. Department of Commerce, Weather Bureau, Washington, D.C."
        ),
    },
    # Fire-weather indices (#793). Only entries for indices implemented in
    # `climate_indices.fire` are added here; the CFFWIS behavior indices (isi,
    # bui, fwi, dsr, #804) and the Haines Index (#810) are deferred to their
    # own tickets, where unit and variant decisions can be validated against
    # real output rather than guessed ahead of implementation. None of the
    # fire indices has an official CF standard_name.
    "kbdi": {
        "long_name": "Keetch-Byram Drought Index",
        "units": "mm",
        "description": ("Cumulative soil moisture deficit, metric scale, range [0, 203.2]."),
        "climate_indices_variant": "metric",
        "references": (
            "Keetch, J. J., & Byram, G. M. (1968). "
            "A Drought Index for Forest Fire Control. "
            "USDA Forest Service Research Paper SE-38. "
            "https://research.fs.usda.gov/treesearch/40; "
            "Alexander, M. E. (1990). "
            "Computer calculation of the Keetch-Byram Drought Index - "
            "programmers beware! Fire Management Notes, 51(4), 23-25."
        ),
    },
    "kbdi_imperial": {
        "long_name": "Keetch-Byram Drought Index",
        "units": "0.01 in",
        "description": (
            "Cumulative soil moisture deficit, imperial scale, hundredths of "
            "an inch, range [0, 800] — the exact conversion of the metric "
            "[0, 203.2] mm scale."
        ),
        "climate_indices_variant": "imperial",
        "references": (
            "Keetch, J. J., & Byram, G. M. (1968). "
            "A Drought Index for Forest Fire Control. "
            "USDA Forest Service Research Paper SE-38. "
            "https://research.fs.usda.gov/treesearch/40; "
            "Alexander, M. E. (1990). "
            "Computer calculation of the Keetch-Byram Drought Index - "
            "programmers beware! Fire Management Notes, 51(4), 23-25."
        ),
    },
    "ffwi": {
        "long_name": "Fosberg Fire Weather Index",
        "units": "dimensionless",
        "description": "Weather-only fire-danger index, conventionally capped at 100.",
        "references": (
            "Fosberg, M. A. (1978). "
            "Weather in wildland fire management: the fire weather index. "
            "Conference on Sierra Nevada Meteorology, Lake Tahoe, CA, 1-4."
        ),
    },
    "hdw": {
        "long_name": "Hot-Dry-Windy Index",
        "units": "hPa m s-1",
        "description": ("Maximum over the lowest 500 m above ground level of vapor pressure deficit times wind speed."),
        "references": (
            "Srock, A. F., Charney, J. J., Potter, B. E., & Goodrick, S. L. (2018). "
            "The Hot-Dry-Windy Index: A New Fire Weather Index. "
            "Atmosphere, 9(7), 279. https://doi.org/10.3390/atmos9070279"
        ),
    },
    "ffmc": {
        "long_name": "Fine Fuel Moisture Code",
        "units": "dimensionless",
        "description": (
            "Moisture content of fine surface litter and other fine fuels, a dimensionless code in [0, 101]."
        ),
        "climate_indices_variant": "cffwis_classic",
        "references": _VAN_WAGNER_PICKETT_1985,
    },
    "dmc": {
        "long_name": "Duff Moisture Code",
        "units": "dimensionless",
        "description": (
            "Moisture content of loosely compacted organic layers of moderate depth, "
            "a dimensionless code floored at zero with no upper bound."
        ),
        "climate_indices_variant": "cffwis_classic",
        "references": _VAN_WAGNER_PICKETT_1985,
    },
    "dc": {
        "long_name": "Drought Code",
        "units": "dimensionless",
        "description": (
            "Moisture content of deep, compact organic layers, a dimensionless code "
            "floored at zero with no upper bound; the CFFWIS component only, distinct "
            "from the package's drought indices."
        ),
        "climate_indices_variant": "cffwis_classic",
        "references": _VAN_WAGNER_PICKETT_1985,
    },
}
