"""Compound-hazard analysis on declared, observed hazard occurrence.

Prepare two or three index-defined hazards on one observed time axis, without
inventing observations, and match their runs without combining severity measures.
The NumPy API is stable; this module is imported explicitly
(``import climate_indices.compound as compound``) and is not re-exported from the
package root.

Implemented so far: :func:`harmonize` with :class:`AlignedHazards`, and
:func:`compound_events` with :class:`CompoundEventSet`.
"""

from climate_indices.compound._events import CompoundEventSet, compound_events
from climate_indices.compound._harmonize import AlignedHazards, harmonize

__all__ = ["AlignedHazards", "CompoundEventSet", "compound_events", "harmonize"]
