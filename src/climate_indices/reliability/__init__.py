"""Power-distribution reliability event records and indices (ADR-0026).

Import this package with ``from climate_indices import reliability``.
``classify_interruptions()`` validates and classifies event records.
``reliability_indices()`` computes SAIFI, SAIDI, CAIDI, MAIFI_E, MAIFI, and the
shared customer-interruption (``CI``) and customer-minute (``CMI``) accumulators
from the event records. CAIDI is NaN when no sustained customer interruption
occurs, and MAIFI is None without a ``device_operations`` column; the result
names each undefined metric in ``undefined_metrics``. ``group_momentary_events()``
collapses raw momentary operations into events. ``simulate_reliability()`` runs a
seeded Monte Carlo series or parallel component simulation into the deterministic
metric path. Availability is not implemented yet.
"""

from climate_indices.reliability._events import classify_interruptions, group_momentary_events
from climate_indices.reliability._indices import ReliabilityResult, reliability_indices
from climate_indices.reliability._simulate import simulate_reliability

__all__ = [
    "ReliabilityResult",
    "classify_interruptions",
    "group_momentary_events",
    "reliability_indices",
    "simulate_reliability",
]
