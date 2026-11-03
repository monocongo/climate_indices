"""The drought namespace facade and its deprecated module aliases (#1555, ADR-0027)."""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings
from types import ModuleType

import pytest

import climate_indices
from climate_indices import drought
from climate_indices.exceptions import ClimateIndicesDeprecationWarning

# old public module path -> canonical module path
_DEPRECATED_MODULES = {
    "climate_indices.indices": "climate_indices.drought.indices",
    "climate_indices.palmer": "climate_indices.drought.palmer",
    "climate_indices.eto": "climate_indices.drought.eto",
    "climate_indices.pm_eto": "climate_indices.drought.pm_eto",
    "climate_indices.self_calibration": "climate_indices.drought.self_calibration",
}

_PRIVATE_MODULES = (
    "climate_indices._palmer_common",
    "climate_indices._palmer_duration",
    "climate_indices._palmer_pdi",
    "climate_indices._palmer_wells",
)

# Names the drought facade re-exports from typed_public_api, which must be the
# same objects as their package-root names (ADR-0027).
_FACADE_NAMES = (
    "spi",
    "spei",
    "spai",
    "pdsi",
    "eddi",
    "percentage_of_normal",
    "pet_thornthwaite",
    "pet_hargreaves",
    "pet_penman_monteith",
    "pci",
    "standardized_index",
    "fit_diagnostics",
    "HumidityInputs",
    "RadiationInputs",
)


@pytest.mark.parametrize(("old_path", "canonical_path"), sorted(_DEPRECATED_MODULES.items()))
def test_deprecated_module_path_warns_and_aliases(
    old_path: str, canonical_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each old public module path warns and installs the canonical module object."""
    canonical = importlib.import_module(canonical_path)
    # Drop the cached alias so the stub module executes and warns again.
    monkeypatch.delitem(sys.modules, old_path, raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = importlib.import_module(old_path)

    assert any(issubclass(warning.category, ClimateIndicesDeprecationWarning) for warning in caught), caught
    assert isinstance(old, ModuleType)
    assert old is canonical
    assert sys.modules[old_path] is canonical


@pytest.mark.parametrize("name", _FACADE_NAMES)
def test_drought_facade_exports_the_same_objects_as_the_root(name: str) -> None:
    """The drought facade re-exports, rather than wraps, the root typed API objects."""
    assert getattr(drought, name) is getattr(climate_indices, name)


def test_deprecated_alias_keeps_string_patch_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patching through the old module path reaches the canonical module."""
    canonical = importlib.import_module("climate_indices.drought.palmer")
    importlib.import_module("climate_indices.palmer")

    monkeypatch.setattr("climate_indices.palmer.pdsi", None)

    assert vars(canonical)["pdsi"] is None


@pytest.mark.parametrize("private_path", _PRIVATE_MODULES)
def test_private_module_paths_have_no_alias(private_path: str) -> None:
    """The old private _palmer_* paths are removed, not aliased."""
    assert private_path not in sys.modules
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(private_path)


def test_drought_package_imports_without_a_cycle() -> None:
    """Importing the drought package first does not close an import cycle."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import climate_indices.drought as drought; "
            "import climate_indices as root; "
            "assert drought.spi is root.spi, 'facade stopped re-exporting root names'",
        ],
        check=True,
    )


def test_root_import_does_not_reach_the_deprecated_paths() -> None:
    """Importing the library root emits no migration warning for its own imports."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, climate_indices; "
            "old = [name for name in ('indices', 'palmer', 'eto', 'pm_eto', 'self_calibration') "
            "if f'climate_indices.{name}' in sys.modules]; "
            "assert not old, old",
        ],
        check=True,
    )
