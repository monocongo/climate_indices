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


# The CFFWIS entries share one source publication; keep the rendered
# `references` text identical across the entries that cite it. DSR is the one
# exception: Van Wagner (1987) Eq. 31 defines the power transform itself.
_VAN_WAGNER_PICKETT_1985 = (
    "Van Wagner, C. E., & Pickett, T. L. (1985). "
    "Equations and FORTRAN program for the Canadian Forest Fire Weather Index System. "
    "Canadian Forestry Service, Forestry Technical Report 33."
)
_VAN_WAGNER_1987 = (
    "Van Wagner, C. E. (1987). "
    "Development and structure of the Canadian Forest Fire Weather Index System. "
    "Canadian Forestry Service, Forestry Technical Report 35."
)

# The three Haines elevation variants score different pressure layers of the
# same 1988 publication; keep the rendered `references` text identical across
# the three entries that cite it.
_HAINES_1988 = (
    "Haines, D. A. (1988). "
    "A lower atmospheric severity index for wildland fires. "
    "National Weather Digest, 13(2), 23-27."
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
    # `climate_indices.fire` are added here. One entry per output convention:
    # the Haines Index gets one per elevation variant (#810), because the
    # variant is chosen per call and decides which pressure levels the output
    # describes. None of the fire indices has an official CF standard_name.
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
    "isi": {
        "long_name": "Initial Spread Index",
        "units": "dimensionless",
        "description": (
            "Expected rate of fire spread immediately after ignition, dimensionless, "
            "from the Fine Fuel Moisture Code and the 10 m wind speed."
        ),
        "climate_indices_variant": "cffwis_classic",
        "references": _VAN_WAGNER_PICKETT_1985,
    },
    "bui": {
        "long_name": "Buildup Index",
        "units": "dimensionless",
        "description": (
            "Fuel available for spreading, dimensionless, from the Duff Moisture Code and the Drought Code."
        ),
        "climate_indices_variant": "cffwis_classic",
        "references": _VAN_WAGNER_PICKETT_1985,
    },
    "fwi": {
        "long_name": "Canadian Fire Weather Index",
        "units": "dimensionless",
        "description": (
            "The final CFFWIS output, dimensionless, from the Initial Spread Index "
            "and the Buildup Index; distinct from the Fosberg Fire Weather Index "
            "(``ffwi``)."
        ),
        "climate_indices_variant": "cffwis_classic",
        "references": _VAN_WAGNER_PICKETT_1985,
    },
    "dsr": {
        "long_name": "Daily Severity Rating",
        "units": "dimensionless",
        "description": (
            "``0.0272 * FWI ** 1.77`` power transform of the Canadian FWI that makes seasonal averaging meaningful."
        ),
        "climate_indices_variant": "cffwis_classic",
        "references": _VAN_WAGNER_1987,
    },
    "haines_low": {
        "long_name": "Haines Index",
        "units": "dimensionless",
        "description": (
            "Lower-atmosphere severity index, low elevation variant: stability from the "
            "950-850 hPa lapse rate, moisture from the 850 hPa dewpoint depression; integer in [2, 6]."
        ),
        "climate_indices_variant": "low",
        "references": _HAINES_1988,
    },
    "haines_mid": {
        "long_name": "Haines Index",
        "units": "dimensionless",
        "description": (
            "Lower-atmosphere severity index, mid elevation variant: stability from the "
            "850-700 hPa lapse rate, moisture from the 850 hPa dewpoint depression; integer in [2, 6]."
        ),
        "climate_indices_variant": "mid",
        "references": _HAINES_1988,
    },
    "haines_high": {
        "long_name": "Haines Index",
        "units": "dimensionless",
        "description": (
            "Lower-atmosphere severity index, high elevation variant: stability from the "
            "700-500 hPa lapse rate, moisture from the 700 hPa dewpoint depression; integer in [2, 6]."
        ),
        "climate_indices_variant": "high",
        "references": _HAINES_1988,
    },
}
