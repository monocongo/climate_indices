"""Deprecated alias for :mod:`climate_indices.drought.eto`.

Import from ``climate_indices.drought.eto`` instead. Removed in 4.0.0.
"""

from climate_indices.drought import eto as _eto
from climate_indices.exceptions import alias_deprecated_module

alias_deprecated_module(old_name=__name__, canonical=_eto)
