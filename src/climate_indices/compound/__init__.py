"""Compound-hazard analysis on declared, observed hazard occurrence.

Prepare two or three index-defined hazards on one observed time axis, without
inventing observations, then count their joint and conditional occurrences and match
their runs without combining severity measures:
:func:`harmonize`
returns :class:`AlignedHazards`, :func:`joint_statistics` returns a
:class:`JointResult` of marginal and joint frequencies, the independence product,
and the likelihood multiplication factor, :func:`conditional_probability`
returns a :class:`ConditionalResult` of anchor-conditioned and response-window
frequencies, and :func:`compound_events` returns a :class:`CompoundEventSet` of
paired concurrent or sequential events. The NumPy API is stable; this module is
imported explicitly (``import climate_indices.compound as compound``) and is not
re-exported from the package root.
"""

from climate_indices.compound._events import CompoundEventSet, compound_events
from climate_indices.compound._harmonize import AlignedHazards, harmonize
from climate_indices.compound._statistics import (
    ConditionalResult,
    JointResult,
    conditional_probability,
    joint_statistics,
)

__all__ = [
    "AlignedHazards",
    "CompoundEventSet",
    "ConditionalResult",
    "JointResult",
    "compound_events",
    "conditional_probability",
    "harmonize",
    "joint_statistics",
]
