"""Power-distribution reliability event records and indices (ADR-0026).

Import this package with ``from climate_indices import reliability``.
``classify_interruptions()`` validates and classifies event records.
``reliability_indices()`` computes SAIFI and the shared customer-interruption
accumulator (``CI``) from sustained events. Duration metrics, momentary
metrics, availability, and simulation are not implemented yet.
"""

from climate_indices.reliability._events import classify_interruptions
from climate_indices.reliability._indices import ReliabilityResult, reliability_indices

__all__ = ["ReliabilityResult", "classify_interruptions", "reliability_indices"]
