"""Daily indices of flood potential, not observed flooding.

Effective precipitation is the shared input for EDI and the planned Flood Index.
"""

from climate_indices.flood._antecedent import APIResult, APIState, antecedent_precipitation_index
from climate_indices.flood._edi import edi
from climate_indices.flood._pe import effective_precipitation

__all__ = ["APIResult", "APIState", "antecedent_precipitation_index", "edi", "effective_precipitation"]
