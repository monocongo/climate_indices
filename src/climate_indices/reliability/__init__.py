"""Power-distribution reliability event records (ADR-0026).

Import this package with ``from climate_indices import reliability``.
Metric calculations and simulation are not implemented yet.
"""

from climate_indices.reliability._events import classify_interruptions

__all__ = ["classify_interruptions"]
