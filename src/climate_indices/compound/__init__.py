"""Compound-hazard analysis on declared, observed hazard occurrence.

Prepare two or three index-defined hazards on one observed time axis, without
inventing observations, then count their joint occurrences: :func:`harmonize`
returns :class:`AlignedHazards`, and :func:`joint_statistics` returns a
:class:`JointResult` of marginal and joint frequencies, the independence product,
and the likelihood multiplication factor. The NumPy API is stable; this module is
imported explicitly (``import climate_indices.compound as compound``) and is not
re-exported from the package root.
"""

from climate_indices.compound._harmonize import AlignedHazards, harmonize
from climate_indices.compound._statistics import JointResult, joint_statistics

__all__ = ["AlignedHazards", "JointResult", "harmonize", "joint_statistics"]
