"""Climate-driven energy indices: demand proxies and resource transforms.

Degree days describe climate-driven heating and cooling demand, not actual load.
Wind profiles and PV capacity factors describe climate-driven resources, not actual generation. Event
statistics describe index runs, not recorded supply shortfalls. The NumPy
reference functions are stable; degree-day and wind-profile DataArray dispatch
is beta, preserves CF metadata, and supports lazy Dask chunks on every dimension.
SREPI, the standardized demand index, and SRLI describe the climate-driven component of
supply and demand, not actual generation or load; their NumPy functions are stable,
their xarray adapters are beta, and event statistics report a per-cell run structure
instead of a gridded index.
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
from climate_indices.energy._solar import pv_capacity_factor
from climate_indices.energy._standardized import SREPI_DROUGHT_THRESHOLDS, srepi, standardized_demand_index
from climate_indices.energy._wind import hub_height_wind_speed, wind_shear_exponent

__all__ = [
    "SREPI_DROUGHT_THRESHOLDS",
    "cooling_degree_days",
    "energy_degree_days",
    "energy_drought_events",
    "heating_degree_days",
    "hub_height_wind_speed",
    "pv_capacity_factor",
    "residual_load",
    "srepi",
    "srli",
    "standardized_demand_index",
    "wind_shear_exponent",
]
