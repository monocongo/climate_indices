"""Climate indices for drought monitoring"""

from importlib.metadata import PackageNotFoundError, version

from climate_indices.exceptions import ClimateIndicesError

try:
    __version__ = version("climate_indices")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["__version__", "ClimateIndicesError"]
