"""Compound-hazard analysis on declared, observed hazard occurrence.

Prepare two or three index-defined hazards on one observed time axis, without
inventing observations. The NumPy API is stable; this module is imported
explicitly (``import climate_indices.compound as compound``) and is not
re-exported from the package root.

So far only harmonization exists: :func:`harmonize` and :class:`AlignedHazards`.
"""

from climate_indices.compound._harmonize import AlignedHazards, harmonize

__all__ = ["AlignedHazards", "harmonize"]
