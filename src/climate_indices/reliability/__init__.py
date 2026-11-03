"""Power-distribution reliability event records and indices (ADR-0026).

Import this package with ``from climate_indices import reliability``.
``classify_interruptions()`` validates and classifies event records.
``reliability_indices()`` computes SAIFI, SAIDI, CAIDI, MAIFI_E, MAIFI, ASAI, and
the shared customer-interruption (``CI``) and customer-minute (``CMI``)
accumulators. It counts sustained and momentary rows separately. CAIDI is NaN
when no sustained customer interruption occurs, and MAIFI is None without a
``device_operations`` column. ASAI is None without reporting hours or explicit
period bounds. It is NaN when interruption time exceeds exposure. The result
names each undefined metric in ``undefined_metrics``. ``group_momentary_events()``
collapses raw momentary operations into events. ``simulate_reliability()`` runs a
seeded Monte Carlo series or parallel component simulation into the deterministic
metric path.
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
