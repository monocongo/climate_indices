"""Daily indices of flood potential, not observed flooding.

Effective precipitation is the shared input for the planned EDI and Flood Index.
"""

from climate_indices.flood._pe import effective_precipitation

__all__ = ["effective_precipitation"]
