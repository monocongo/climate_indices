"""Climate indices for drought monitoring"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("climate_indices")
except PackageNotFoundError:
    __version__ = "unknown"
