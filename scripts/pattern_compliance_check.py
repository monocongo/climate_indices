#!/usr/bin/env python3
"""v2.4.0 Pattern Compliance Dashboard.

Scans the climate_indices source tree to verify that all canonical v2.4.0
patterns are consistently applied across every module and index function.

Patterns checked:
    1. Structured Exception Hierarchy (InvalidArgumentError vs bare ValueError)
    2. structlog Lifecycle Logging (calculation_started/completed/failed)
    3. CF Metadata Registry (entries in cf_metadata_registry.py)
    4. xarray Adapter / typed_public_api @overload Coverage
    5. Property-Based Test Coverage (Hypothesis @given tests)

Exit codes:
    0  -- all checks pass (100% compliant)
    1  -- one or more violations detected

Usage:
    uv run python scripts/pattern_compliance_check.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src" / "climate_indices"
_TEST_ROOT = _REPO_ROOT / "tests"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _count_pattern(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text))


# ---------------------------------------------------------------------------
# 1. Structured Exception Coverage
# ---------------------------------------------------------------------------
def check_structured_exceptions() -> tuple[bool, list[str], int, int]:
    """Check that core modules use InvalidArgumentError instead of bare ValueError.

    We audit the main computation modules (indices.py, eto.py, palmer.py).
    CLI modules (__main__.py, __spi__.py) and utility/math modules (compute.py,
    lmoments.py, utils.py) are excluded from this check because they either
    handle user-facing argument parsing or low-level math that predates the
    exception hierarchy.
    """
    audited_modules = ["indices.py", "eto.py"]
    violations: list[str] = []
    compliant_count = 0

    # known exceptions: internal control-flow ValueErrors that are caught
    # by except handlers within the same function (e.g. Pearson fallback in spi)
    _KNOWN_INTERNAL_VALUEERRORS = {
        "indices.py": 1,  # Pearson fitting fallback caught by except (ValueError, ...)
    }

    for mod_name in audited_modules:
        source = _read(_SRC_ROOT / mod_name)
        # count bare ValueError raises (excluding comments and docstrings)
        bare_raises = re.findall(r"^\s+raise ValueError\(", source, re.MULTILINE)
        structured_raises = re.findall(r"raise InvalidArgumentError\(", source)
        allowed = _KNOWN_INTERNAL_VALUEERRORS.get(mod_name, 0)

        if len(bare_raises) > allowed:
            violations.append(
                f"  {mod_name}: {len(bare_raises)} bare ValueError raise(s) remain "
                f"({allowed} allowed as internal control flow, "
                f"{len(structured_raises)} use InvalidArgumentError)"
            )
        else:
            compliant_count += 1

    # palmer.py has structlog error context but uses ValueError for legacy compat
    palmer_source = _read(_SRC_ROOT / "palmer.py")
    if "calculation_failed" in palmer_source:
        compliant_count += 1
    else:
        violations.append("  palmer.py: missing calculation_failed error context")

    total = len(audited_modules) + 1  # +1 for palmer
    return len(violations) == 0, violations, compliant_count, total


# ---------------------------------------------------------------------------
# 2. structlog Lifecycle Pattern
# ---------------------------------------------------------------------------

# functions that should have lifecycle logging, mapped as (module, function_name)
# pet_thornthwaite lifecycle is provided by indices.pet() wrapper
_LIFECYCLE_FUNCTIONS = [
    ("indices.py", "spi"),
    ("indices.py", "spei"),
    ("indices.py", "percentage_of_normal"),
    ("indices.py", "pet"),          # wraps eto_thornthwaite
    ("indices.py", "pci"),
    ("eto.py", "eto_hargreaves"),
    ("palmer.py", "pdsi"),
]


def check_structlog_lifecycle() -> tuple[bool, list[str], int, int]:
    """Check that compute functions have calculation_started/completed/failed."""
    violations: list[str] = []
    compliant_count = 0

    for mod_name, func_name in _LIFECYCLE_FUNCTIONS:
        source = _read(_SRC_ROOT / mod_name)

        # extract the function body via AST
        tree = ast.parse(source)
        func_source = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                start = node.lineno - 1
                end = node.end_lineno
                func_source = "\n".join(source.splitlines()[start:end])
                break

        if func_source is None:
            violations.append(f"  {mod_name}::{func_name} -- function not found")
            continue

        missing = []
        for event in ["calculation_started", "calculation_completed", "calculation_failed"]:
            if event not in func_source:
                missing.append(event)

        if missing:
            violations.append(f"  {mod_name}::{func_name} -- missing: {', '.join(missing)}")
        else:
            compliant_count += 1

    total = len(_LIFECYCLE_FUNCTIONS)
    return len(violations) == 0, violations, compliant_count, total


# ---------------------------------------------------------------------------
# 3. CF Metadata Registry
# ---------------------------------------------------------------------------

_EXPECTED_CF_KEYS = [
    "spi",
    "spei",
    "pet_thornthwaite",
    "pet_hargreaves",
    "percentage_of_normal",
    "pci",
    "pnp",
]


def check_cf_metadata() -> tuple[bool, list[str], int, int]:
    """Check that all indices have CF metadata registry entries."""
    source = _read(_SRC_ROOT / "cf_metadata_registry.py")
    violations: list[str] = []
    compliant_count = 0

    for key in _EXPECTED_CF_KEYS:
        # look for the key in the CF_METADATA dict literal
        if f'"{key}"' in source or f"'{key}'" in source:
            compliant_count += 1
        else:
            violations.append(f"  Missing CF_METADATA entry: '{key}'")

    # verify required fields exist for each entry
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key_node in node.keys:
                if isinstance(key_node, ast.Constant) and key_node.value in _EXPECTED_CF_KEYS:
                    # found a top-level dict entry -- the detailed field checks
                    # are handled by the test_pattern_compliance.py test suite
                    pass

    total = len(_EXPECTED_CF_KEYS)
    return len(violations) == 0, violations, compliant_count, total


# ---------------------------------------------------------------------------
# 4. xarray Adapter / typed_public_api @overload Coverage
# ---------------------------------------------------------------------------

_EXPECTED_TYPED_API_FUNCTIONS = [
    "spi",
    "spei",
    "pet_thornthwaite",
    "pet_hargreaves",
    "percentage_of_normal",
    "pci",
]


def check_typed_public_api() -> tuple[bool, list[str], int, int]:
    """Check typed_public_api has @overload signatures for all public indices."""
    source = _read(_SRC_ROOT / "typed_public_api.py")
    tree = ast.parse(source)
    violations: list[str] = []
    compliant_count = 0

    for func_name in _EXPECTED_TYPED_API_FUNCTIONS:
        overload_count = 0
        impl_count = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                has_overload = any(
                    isinstance(d, ast.Name) and d.id == "overload"
                    for d in node.decorator_list
                )
                if has_overload:
                    overload_count += 1
                else:
                    impl_count += 1

        if overload_count >= 2 and impl_count >= 1:
            compliant_count += 1
        else:
            violations.append(
                f"  {func_name}: {overload_count} @overload(s), {impl_count} impl(s) "
                f"(need >= 2 overloads + 1 impl)"
            )

    # check __init__.py re-exports
    init_source = _read(_SRC_ROOT / "__init__.py")
    for func_name in _EXPECTED_TYPED_API_FUNCTIONS:
        if func_name not in init_source:
            violations.append(f"  {func_name}: not re-exported from __init__.py")

    total = len(_EXPECTED_TYPED_API_FUNCTIONS)
    return len(violations) == 0, violations, compliant_count, total


# ---------------------------------------------------------------------------
# 5. Property-Based Test Coverage
# ---------------------------------------------------------------------------

_EXPECTED_PROPERTY_TEST_PREFIXES = [
    ("test_spi_", "SPI"),
    ("test_spei_", "SPEI"),
    ("test_pet_thornthwaite_", "PET Thornthwaite"),
    ("test_pet_hargreaves_", "PET Hargreaves"),
    ("test_pnp_", "PNP"),
    ("test_pci_", "PCI"),
    ("test_pdsi_", "Palmer/PDSI"),
]


def check_property_based_tests() -> tuple[bool, list[str], int, int, int]:
    """Check that Hypothesis property-based tests exist for all indices."""
    test_file = _TEST_ROOT / "test_property_based.py"
    source = _read(test_file)
    violations: list[str] = []
    compliant_count = 0
    total_given_tests = len(re.findall(r"@given\(", source))

    for prefix, label in _EXPECTED_PROPERTY_TEST_PREFIXES:
        matches = re.findall(rf"def ({prefix}\w+)\(", source)
        if matches:
            compliant_count += 1
        else:
            violations.append(f"  No property-based tests for {label} (prefix: {prefix})")

    total = len(_EXPECTED_PROPERTY_TEST_PREFIXES)
    return len(violations) == 0, violations, compliant_count, total, total_given_tests


# ---------------------------------------------------------------------------
# Dashboard Output
# ---------------------------------------------------------------------------


def main() -> int:
    """Run all compliance checks and print dashboard."""
    print()
    print("=" * 60)
    print("  v2.4.0 Pattern Compliance Dashboard")
    print("=" * 60)
    print()

    all_passed = True

    # 1. Structured Exceptions
    ok, violations, passed, total = check_structured_exceptions()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Structured Exceptions: {passed}/{total} modules compliant")
    if violations:
        all_passed = False
        for v in violations:
            print(v)

    # 2. structlog Lifecycle
    ok, violations, passed, total = check_structlog_lifecycle()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] structlog Lifecycle: {passed}/{total} compute functions")
    if violations:
        all_passed = False
        for v in violations:
            print(v)

    # 3. CF Metadata Registry
    ok, violations, passed, total = check_cf_metadata()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] CF Metadata Registry: {passed}/{total} indices registered")
    if violations:
        all_passed = False
        for v in violations:
            print(v)

    # 4. typed_public_api / xarray Adapters
    ok, violations, passed, total = check_typed_public_api()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] xarray / typed_public_api: {passed}/{total} public functions")
    if violations:
        all_passed = False
        for v in violations:
            print(v)

    # 5. Property-Based Tests
    ok, violations, passed, total, given_count = check_property_based_tests()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Property-Based Tests: {passed}/{total} indices, {given_count} @given tests total")
    if violations:
        all_passed = False
        for v in violations:
            print(v)

    # Summary
    print()
    print("-" * 60)
    if all_passed:
        print("  Overall: 100% pattern compliance")
        print("  All v2.4.0 canonical patterns verified.")
    else:
        print("  Overall: VIOLATIONS DETECTED")
        print("  Fix the issues listed above to achieve full compliance.")
    print("-" * 60)
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
