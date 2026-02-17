"""CF Convention metadata registry for climate indices.

Centralizes CF-compliant metadata (long_name, units, references) for all
climate indices that produce xarray DataArray output. Each entry follows
the CF Conventions (https://cfconventions.org/) attribute model.

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
    Optional keys: standard_name (only when officially defined in CF conventions).

    .. note:: Part of the beta xarray adapter layer. See :doc:`xarray_migration`.
    """

    standard_name: str


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
    "pdsi": {
        "long_name": "Palmer Drought Severity Index",
        "units": "",
        "references": (
            "Palmer, W. C. (1965). "
            "Meteorological Drought. Research Paper No. 45, "
            "U.S. Department of Commerce Weather Bureau, Washington, D.C."
        ),
    },
    "phdi": {
        "long_name": "Palmer Hydrological Drought Index",
        "units": "",
        "references": (
            "Palmer, W. C. (1965). "
            "Meteorological Drought. Research Paper No. 45, "
            "U.S. Department of Commerce Weather Bureau, Washington, D.C."
        ),
    },
    "pmdi": {
        "long_name": "Palmer Modified Drought Index",
        "units": "",
        "references": (
            "Heddinghaus, T. R., & Sabol, P. (1991). "
            "A review of the Palmer Drought Severity Index and where do we go from here? "
            "Proceedings of the 7th Conference on Applied Climatology, "
            "10-13 September, Salt Lake City, UT. "
            "American Meteorological Society, Boston, MA, 242-246."
        ),
    },
    "z_index": {
        "long_name": "Palmer Z-Index",
        "units": "",
        "references": (
            "Palmer, W. C. (1965). "
            "Meteorological Drought. Research Paper No. 45, "
            "U.S. Department of Commerce Weather Bureau, Washington, D.C."
        ),
    },
}
