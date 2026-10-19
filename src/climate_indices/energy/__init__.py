"""Climate-driven energy indices: demand proxies and resource transforms.

Degree days describe climate-driven heating and cooling demand, not actual load.
Wind profiles and capacity factors describe climate-driven resources, not actual generation.
Event statistics describe index runs, not recorded supply shortfalls.
The wind capacity-factor transform is NumPy-only. The NumPy
reference functions are stable; degree-day and wind-profile DataArray dispatch
is beta, preserves CF metadata, and supports lazy Dask chunks on every dimension.
SREPI, the standardized demand index, and SRLI describe the climate-driven component of
supply and demand, not actual generation or load; their NumPy functions are stable
and their xarray adapters remain deferred to #1174.
Import this package as ``from climate_indices import energy``; no energy
function is re-exported from the package root (ADR-0019).
"""

from __future__ import annotations

from climate_indices.energy._degree_days import (
    cooling_degree_days,
    energy_degree_days,
    heating_degree_days,
)
from climate_indices.energy._events import energy_drought_events
from climate_indices.energy._residual_load import residual_load, srli
from climate_indices.energy._standardized import SREPI_DROUGHT_THRESHOLDS, srepi, standardized_demand_index
from climate_indices.energy._wind import hub_height_wind_speed, wind_capacity_factor, wind_shear_exponent

__all__ = [
    "SREPI_DROUGHT_THRESHOLDS",
    "cooling_degree_days",
    "energy_degree_days",
    "energy_drought_events",
    "heating_degree_days",
    "hub_height_wind_speed",
    "residual_load",
    "srepi",
    "srli",
    "standardized_demand_index",
    "wind_capacity_factor",
    "wind_shear_exponent",
]
