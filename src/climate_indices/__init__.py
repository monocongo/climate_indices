"""Climate indices for drought monitoring"""

from importlib.metadata import PackageNotFoundError, version

from climate_indices.exceptions import (
    BetaFeatureWarning,
    ClimateIndicesDeprecationWarning,
    ClimateIndicesError,
    ClimateIndicesWarning,
    InputAlignmentWarning,
    emit_deprecation_warning,
)
from climate_indices.logging_config import configure_logging
from climate_indices.typed_public_api import spei, spi
from climate_indices.xarray_adapter import (
    CF_METADATA,
    CFAttributes,
    InputType,
    detect_input_type,
    eto_penman_monteith,
    pet_hargreaves,
    pet_thornthwaite,
    xarray_adapter,
)

try:
    __version__ = version("climate_indices")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "BetaFeatureWarning",
    "CF_METADATA",
    "CFAttributes",
    "ClimateIndicesDeprecationWarning",
    "ClimateIndicesError",
    "ClimateIndicesWarning",
    "configure_logging",
    "InputAlignmentWarning",
    "InputType",
    "detect_input_type",
    "emit_deprecation_warning",
    "eto_penman_monteith",
    "pet_hargreaves",
    "pet_thornthwaite",
    "spei",
    "spi",
    "xarray_adapter",
]
