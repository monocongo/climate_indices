"""Deprecated alias for :mod:`climate_indices.drought.palmer`.

Import from ``climate_indices.drought.palmer`` instead. Removed in 4.0.0.
"""

from climate_indices.drought import palmer as _palmer
from climate_indices.exceptions import alias_deprecated_module

alias_deprecated_module(old_name=__name__, canonical=_palmer)
