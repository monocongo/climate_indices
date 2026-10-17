"""Climate-driven energy resource proxies, not actual generation or load.

The NumPy wind-profile API is stable. xarray/Dask adapters and other energy
transforms are tracked separately by the energy-family roadmap (#1157).
"""

from climate_indices.energy._wind import hub_height_wind_speed, wind_shear_exponent

__all__ = ["hub_height_wind_speed", "wind_shear_exponent"]
