"""Drought indices: SPI, SPEI, SPAI, PNP, EDDI, PCI, and the Palmer family.

The NumPy implementations live in :mod:`climate_indices.drought.indices` and
:mod:`climate_indices.drought.palmer`, and the PET implementations in
:mod:`climate_indices.drought.eto` and :mod:`climate_indices.drought.pm_eto`.
The typed NumPy/DataArray entry points are re-exported here from
:mod:`climate_indices.typed_public_api`, so they are the same objects as their
package-root names: ``climate_indices.drought.spi is climate_indices.spi``.

Two indices of other families are drought indices but stay where their
application named them, following ADR-0013: KBDI lives in
:mod:`climate_indices.fire` and EDI in :mod:`climate_indices.flood`. EDDI, the
evaporative-demand drought index, lives here; EDI is the distinct Effective
Drought Index in ``flood``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from climate_indices.drought import eto, indices, palmer, pm_eto, self_calibration
from climate_indices.drought.pm_eto import HumidityInputs, RadiationInputs

if TYPE_CHECKING:
    # Type checkers see the overloaded signatures that __getattr__ resolves at runtime.
    from climate_indices.typed_public_api import (
        eddi,
        fit_diagnostics,
        pci,
        pdsi,
        percentage_of_normal,
        pet_hargreaves,
        pet_penman_monteith,
        pet_thornthwaite,
        spai,
        spei,
        spi,
        standardized_index,
    )

# The typed public API re-exports. They are resolved lazily because
# typed_public_api imports the drought implementation modules, so importing it
# eagerly here would close an import cycle (#1238).
_TYPED_PUBLIC_API_NAMES = frozenset(
    {
        "eddi",
        "fit_diagnostics",
        "pci",
        "pdsi",
        "percentage_of_normal",
        "pet_hargreaves",
        "pet_penman_monteith",
        "pet_thornthwaite",
        "spai",
        "spei",
        "spi",
        "standardized_index",
    }
)

__all__ = [
    "HumidityInputs",
    "RadiationInputs",
    "eddi",
    "eto",
    "fit_diagnostics",
    "indices",
    "palmer",
    "pci",
    "pdsi",
    "percentage_of_normal",
    "pet_hargreaves",
    "pet_penman_monteith",
    "pet_thornthwaite",
    "pm_eto",
    "self_calibration",
    "spai",
    "spei",
    "spi",
    "standardized_index",
]


def __getattr__(name: str) -> object:
    """Resolve the typed public API re-exports without an eager import."""
    if name in _TYPED_PUBLIC_API_NAMES:
        from climate_indices import typed_public_api

        return getattr(typed_public_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
