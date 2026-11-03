"""Deprecated alias for :mod:`climate_indices.drought.pm_eto`.

Import from ``climate_indices.drought.pm_eto`` instead. Removed in 4.0.0.
"""

from climate_indices.drought import pm_eto as _pm_eto
from climate_indices.exceptions import alias_deprecated_module

alias_deprecated_module(old_name=__name__, canonical=_pm_eto)
