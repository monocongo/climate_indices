"""Climate indices for drought monitoring"""

from importlib.metadata import PackageNotFoundError, version

from climate_indices.exceptions import ClimateIndicesError, ClimateIndicesWarning
from climate_indices.logging_config import configure_logging

try:
    __version__ = version("climate_indices")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["__version__", "ClimateIndicesError", "ClimateIndicesWarning", "configure_logging"]
