"""Tests for the reference dataset provenance protocol.

Validates that all reference datasets in tests/fixture/ have valid
provenance.json files conforming to the project's provenance schema.
This ensures scientific reproducibility and guards against silent
data corruption in reference datasets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

# path constants
FIXTURE_DIR = Path(__file__).parent / "fixture"
SCHEMA_PATH = FIXTURE_DIR / "provenance_schema.json"


def _load_schema() -> dict[str, Any]:
    """Load the provenance JSON Schema."""
    with SCHEMA_PATH.open() as f:
        return json.load(f)


def _find_provenance_files() -> list[Path]:
    """Find all provenance.json files under the fixture directory."""
    return sorted(FIXTURE_DIR.rglob("provenance.json"))


def _compute_checksum_for_division(directory: Path) -> str:
    """Compute SHA-256 checksum for .npy files in a single directory.

    Concatenates sorted .npy file contents and returns the hex digest.

    Args:
        directory: Path to the directory containing .npy files.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    hasher = hashlib.sha256()
    npy_files = sorted(directory.glob("*.npy"))
    for npy_file in npy_files:
        hasher.update(npy_file.read_bytes())
    return hasher.hexdigest()


class TestProvenanceSchema:
    """Validate provenance.json files against the JSON Schema."""

    def test_schema_file_exists(self) -> None:
        """Verify the provenance schema file exists."""
        assert SCHEMA_PATH.exists(), f"Provenance schema not found at {SCHEMA_PATH}. Run story 3.2 to create it."

    def test_schema_is_valid_json(self) -> None:
        """Verify the schema file contains valid JSON."""
        schema = _load_schema()
        assert isinstance(schema, dict)
        assert "required" in schema

    @pytest.mark.parametrize(
        "provenance_path",
        _find_provenance_files(),
        ids=[str(p.parent.relative_to(FIXTURE_DIR)) for p in _find_provenance_files()],
    )
    def test_provenance_conforms_to_schema(self, provenance_path: Path) -> None:
        """Validate each provenance.json file against the schema."""
        schema = _load_schema()
        with provenance_path.open() as f:
            provenance = json.load(f)

        jsonschema.validate(instance=provenance, schema=schema)

    @pytest.mark.parametrize(
        "provenance_path",
        _find_provenance_files(),
        ids=[str(p.parent.relative_to(FIXTURE_DIR)) for p in _find_provenance_files()],
    )
    def test_provenance_has_required_fields(self, provenance_path: Path) -> None:
        """Verify all required fields are present and non-empty."""
        with provenance_path.open() as f:
            provenance = json.load(f)

        required_fields = [
            "source",
            "url",
            "download_date",
            "subset_description",
            "checksum_sha256",
            "fixture_version",
            "validation_tolerance",
        ]
        for field in required_fields:
            assert field in provenance, f"Missing required field: {field}"
            assert provenance[field], f"Field '{field}' is empty"


class TestProvenanceChecksum:
    """Validate that checksums match actual data files."""

    def test_palmer_checksum_matches(self) -> None:
        """Verify Palmer fixture checksum matches provenance.json.

        The checksum in the Palmer provenance file is computed from
        division 0101 only (the reference subset for validation).
        """
        provenance_path = FIXTURE_DIR / "palmer" / "provenance.json"
        if not provenance_path.exists():
            pytest.skip("Palmer provenance.json not yet created")

        with provenance_path.open() as f:
            provenance = json.load(f)

        expected_checksum = provenance["checksum_sha256"]

        # compute actual checksum from division 0101
        division_dir = FIXTURE_DIR / "palmer" / "0101"
        assert division_dir.exists(), "Palmer division 0101 fixture missing"

        actual_checksum = _compute_checksum_for_division(division_dir)
        assert actual_checksum == expected_checksum, (
            f"Palmer fixture checksum mismatch. "
            f"Expected: {expected_checksum}, "
            f"Actual: {actual_checksum}. "
            f"If the fixture data was intentionally updated, "
            f"regenerate the checksum and update provenance.json."
        )


class TestProvenanceProtocolCoverage:
    """Ensure provenance protocol is being followed for reference datasets."""

    def test_at_least_one_provenance_exists(self) -> None:
        """Verify that at least one provenance.json has been created."""
        provenance_files = _find_provenance_files()
        assert len(provenance_files) > 0, (
            "No provenance.json files found under tests/fixture/. Reference datasets must have provenance metadata."
        )

    def test_fixture_readme_exists(self) -> None:
        """Verify the fixture README documenting the protocol exists."""
        readme = FIXTURE_DIR / "README.md"
        assert readme.exists(), "tests/fixture/README.md not found. The provenance protocol documentation is required."
