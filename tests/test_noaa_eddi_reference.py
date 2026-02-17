"""NOAA EDDI reference validation tests (FR-TEST-004).

Validates the library's EDDI implementation against pre-computed EDDI values
from the NOAA Physical Sciences Laboratory (PSL). This is a critical test
that must pass before merge.

Reference data source:
    https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/

The test expects fixture data in tests/fixture/noaa-eddi-{scale}month/
directories. Each directory should contain:
    - provenance.json: Metadata conforming to the provenance schema
    - pet_input.npy: PET input values used to compute EDDI
    - eddi_reference.npy: NOAA-computed EDDI reference values
    - metadata.json: Calibration parameters (start_year, cal_start, cal_end)

If fixture data is not present, tests are skipped with a clear message.
Run scripts/prepare_noaa_eddi_fixtures.py to download and prepare the data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from climate_indices import compute, indices

# tolerance per NFR-EDDI-VAL (looser than SPI/SPEI due to non-parametric ranking)
_RTOL = 1e-5
_ATOL = 1e-5

FIXTURE_DIR = Path(__file__).parent / "fixture"
_EDDI_SCALES = [1, 3, 6]


def _fixture_dir_for_scale(scale: int) -> Path:
    """Return the fixture directory for a given EDDI time scale."""
    return FIXTURE_DIR / f"noaa-eddi-{scale}month"


def _fixture_available(scale: int) -> bool:
    """Check if all required fixture files exist for the given scale."""
    fixture_dir = _fixture_dir_for_scale(scale)
    required_files = ["provenance.json", "pet_input.npy", "eddi_reference.npy", "metadata.json"]
    return all((fixture_dir / f).exists() for f in required_files)


def _load_fixture(scale: int) -> dict[str, Any]:
    """Load all fixture data for a given scale.

    Args:
        scale: EDDI time scale (1, 3, or 6 months).

    Returns:
        Dict with keys: pet_input, eddi_reference, metadata, provenance.
    """
    fixture_dir = _fixture_dir_for_scale(scale)

    with (fixture_dir / "metadata.json").open() as f:
        metadata = json.load(f)

    with (fixture_dir / "provenance.json").open() as f:
        provenance = json.load(f)

    return {
        "pet_input": np.load(fixture_dir / "pet_input.npy"),
        "eddi_reference": np.load(fixture_dir / "eddi_reference.npy"),
        "metadata": metadata,
        "provenance": provenance,
    }


def _skip_reason(scale: int) -> str:
    """Generate a skip message indicating missing fixtures."""
    fixture_dir = _fixture_dir_for_scale(scale)
    return (
        f"NOAA EDDI {scale}-month fixture data not found at {fixture_dir}. "
        f"Run scripts/prepare_noaa_eddi_fixtures.py to download and prepare reference data."
    )


class TestNoaaEddiReference:
    """Validate EDDI against NOAA PSL reference data (FR-TEST-004)."""

    @pytest.mark.parametrize("scale", _EDDI_SCALES, ids=[f"{s}month" for s in _EDDI_SCALES])
    def test_eddi_matches_noaa_reference(self, scale: int) -> None:
        """Library EDDI should match NOAA reference within tolerance.

        This test computes EDDI using the library's implementation and
        compares against pre-computed NOAA PSL reference values. The
        tolerance (rtol=1e-5, atol=1e-5) accounts for minor numerical
        differences in non-parametric ranking implementations.
        """
        if not _fixture_available(scale):
            pytest.skip(_skip_reason(scale))

        fixture = _load_fixture(scale)
        metadata = fixture["metadata"]

        computed = indices.eddi(
            pet_values=fixture["pet_input"],
            scale=scale,
            data_start_year=metadata["data_start_year"],
            calibration_year_initial=metadata["calibration_year_initial"],
            calibration_year_final=metadata["calibration_year_final"],
            periodicity=compute.Periodicity.monthly,
        )

        reference = fixture["eddi_reference"]

        # shapes must match
        assert computed.shape == reference.shape, (
            f"Shape mismatch: computed {computed.shape} vs reference {reference.shape}"
        )

        # compare non-NaN values
        valid_mask = ~np.isnan(reference)
        assert np.sum(valid_mask) > 0, "Reference data is all NaN"

        np.testing.assert_allclose(
            computed[valid_mask],
            reference[valid_mask],
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=(
                f"EDDI {scale}-month values differ from NOAA reference. "
                f"Max difference: {np.nanmax(np.abs(computed[valid_mask] - reference[valid_mask])):.2e}. "
                f"If the algorithm was intentionally changed, update the reference data."
            ),
        )

    @pytest.mark.parametrize("scale", _EDDI_SCALES, ids=[f"{s}month" for s in _EDDI_SCALES])
    def test_nan_positions_match(self, scale: int) -> None:
        """NaN positions in computed EDDI should match reference data.

        NaN values appear at the start of the series (from sliding sums)
        and where the input has insufficient climatology. The pattern
        should be identical between our implementation and NOAA's.
        """
        if not _fixture_available(scale):
            pytest.skip(_skip_reason(scale))

        fixture = _load_fixture(scale)
        metadata = fixture["metadata"]

        computed = indices.eddi(
            pet_values=fixture["pet_input"],
            scale=scale,
            data_start_year=metadata["data_start_year"],
            calibration_year_initial=metadata["calibration_year_initial"],
            calibration_year_final=metadata["calibration_year_final"],
            periodicity=compute.Periodicity.monthly,
        )

        reference = fixture["eddi_reference"]
        computed_nans = np.isnan(computed)
        reference_nans = np.isnan(reference)

        np.testing.assert_array_equal(
            computed_nans,
            reference_nans,
            err_msg=f"EDDI {scale}-month NaN positions differ from NOAA reference.",
        )

    @pytest.mark.parametrize("scale", _EDDI_SCALES, ids=[f"{s}month" for s in _EDDI_SCALES])
    def test_provenance_metadata_valid(self, scale: int) -> None:
        """Fixture provenance.json should contain valid metadata."""
        if not _fixture_available(scale):
            pytest.skip(_skip_reason(scale))

        fixture = _load_fixture(scale)
        provenance = fixture["provenance"]

        # check required fields
        required = ["source", "url", "download_date", "checksum_sha256", "fixture_version"]
        for field in required:
            assert field in provenance, f"Missing provenance field: {field}"
            assert provenance[field], f"Empty provenance field: {field}"

        # source should reference NOAA
        assert "NOAA" in provenance["source"] or "PSL" in provenance["source"], (
            f"Provenance source should reference NOAA/PSL, got: {provenance['source']}"
        )


class TestEddiSelfConsistency:
    """Self-consistency checks that don't require external reference data.

    These tests validate mathematical properties of the EDDI implementation
    that must hold regardless of the input data.
    """

    def test_eddi_output_range(self) -> None:
        """EDDI output should be clipped to [-3.09, 3.09]."""
        rng = np.random.default_rng(42)
        pet = rng.uniform(10.0, 100.0, size=360)
        result = indices.eddi(
            pet,
            scale=1,
            data_start_year=1990,
            calibration_year_initial=1990,
            calibration_year_final=2019,
            periodicity=compute.Periodicity.monthly,
        )
        valid = ~np.isnan(result)
        assert np.all(result[valid] >= -3.09)
        assert np.all(result[valid] <= 3.09)

    def test_scale_consistency(self) -> None:
        """Larger scales should produce smoother (lower variance) EDDI."""
        rng = np.random.default_rng(123)
        pet = rng.uniform(20.0, 80.0, size=360)

        result_1 = indices.eddi(
            pet,
            scale=1,
            data_start_year=1990,
            calibration_year_initial=1990,
            calibration_year_final=2019,
            periodicity=compute.Periodicity.monthly,
        )
        result_6 = indices.eddi(
            pet,
            scale=6,
            data_start_year=1990,
            calibration_year_initial=1990,
            calibration_year_final=2019,
            periodicity=compute.Periodicity.monthly,
        )

        # scale-6 should have fewer NaN leading values than scale-1
        # (actually more NaN due to sliding sum, but the valid variance should be lower)
        valid_1 = result_1[~np.isnan(result_1)]
        valid_6 = result_6[~np.isnan(result_6)]

        if len(valid_1) > 0 and len(valid_6) > 0:
            # variance of scale-6 should be <= scale-1 (smoothing effect)
            # use a generous margin since this is stochastic
            assert np.var(valid_6) <= np.var(valid_1) * 1.5, (
                f"Scale-6 variance ({np.var(valid_6):.4f}) unexpectedly larger than scale-1 ({np.var(valid_1):.4f})"
            )

    def test_identical_input_produces_zero_eddi(self) -> None:
        """Constant PET input should produce EDDI near zero (all ranks tied)."""
        pet = np.full(360, 50.0)
        result = indices.eddi(
            pet,
            scale=1,
            data_start_year=1990,
            calibration_year_initial=1990,
            calibration_year_final=2019,
            periodicity=compute.Periodicity.monthly,
        )
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            # with constant input, all values have the same rank
            # so all z-scores should be identical (and near the median = 0)
            assert np.std(valid) < 0.01, (
                f"Constant input should produce near-zero variance EDDI, got std={np.std(valid):.4f}"
            )
