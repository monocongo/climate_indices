"""Climate-driven energy indices: demand proxies and resource transforms.

Degree days describe climate-driven heating and cooling demand, not actual load.
Wind profiles describe climate-driven resources, not actual generation. The NumPy
reference functions are stable; degree-day and wind-profile DataArray dispatch
is beta, preserves CF metadata, and supports lazy Dask chunks on every dimension.
SRLI standardizes residual load, the climate-driven component of residual load
and not actual load; its NumPy function is stable. Standardized energy
adapters remain deferred to #1174.
Import this package as ``from climate_indices import energy``; no energy
function is re-exported from the package root (ADR-0019).
"""

from __future__ import annotations

from climate_indices.energy._degree_days import (
    cooling_degree_days,
    energy_degree_days,
    heating_degree_days,
)
from climate_indices.energy._residual_load import residual_load, srli
from climate_indices.energy._wind import hub_height_wind_speed, wind_shear_exponent

__all__ = [
    "cooling_degree_days",
    "energy_degree_days",
    "heating_degree_days",
    "hub_height_wind_speed",
    "residual_load",
    "srli",
    "wind_shear_exponent",
]
