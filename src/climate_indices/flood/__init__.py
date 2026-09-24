"""Daily indices of flood potential, not observed flooding.

Effective precipitation is the shared input for EDI and the Flood Index.
"""

from climate_indices.flood._edi import edi
from climate_indices.flood._if import flood_index
from climate_indices.flood._pe import effective_precipitation

__all__ = ["edi", "effective_precipitation", "flood_index"]
