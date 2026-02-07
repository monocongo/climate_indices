"""Climate indices for drought monitoring"""

from importlib.metadata import PackageNotFoundError, version

from climate_indices.exceptions import ClimateIndicesError, ClimateIndicesWarning
from climate_indices.logging_config import configure_logging
from climate_indices.xarray_adapter import (
    CF_METADATA,
    CFAttributes,
    InputType,
    detect_input_type,
    xarray_adapter,
)

try:
    __version__ = version("climate_indices")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "CF_METADATA",
    "CFAttributes",
    "ClimateIndicesError",
    "ClimateIndicesWarning",
    "configure_logging",
    "InputType",
    "detect_input_type",
    "xarray_adapter",
]
