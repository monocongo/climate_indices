"""Climate indices for drought monitoring"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("climate_indices")
except PackageNotFoundError:
    __version__ = "unknown"

# Register xarray accessors.
from climate_indices import accessors as _accessors  # noqa: F401
