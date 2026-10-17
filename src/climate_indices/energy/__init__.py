"""Climate-driven energy indices: demand proxies and resource transforms.

Degree days are the climate-driven component of heating and cooling demand, not
actual load. The NumPy reference functions are the stable tier; xarray adapters
are beta and land separately (#1166, #1174). Imported as
``from climate_indices import energy``; no energy name is re-exported from the
package root (ADR-0019).
"""

from __future__ import annotations

from climate_indices.energy._degree_days import (
    cooling_degree_days,
    energy_degree_days,
    heating_degree_days,
)

__all__ = [
    "cooling_degree_days",
    "energy_degree_days",
    "heating_degree_days",
]
