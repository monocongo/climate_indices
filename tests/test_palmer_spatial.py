"""Equivalence tests for palmer.pdsi()'s spatial block path (#937, ADR-0009).

Palmer's spell recursion has genuine per-cell control flow (unlike SPI/SPEI's
fitting-based kernels, which are pure arithmetic per calendar period), so the
spatial path is a masked vectorization of the recursion itself rather than a
broadcast. Equivalence here is bit-for-bit (`assert_array_equal`), matching
the decision recorded on #937 and in ADR-0009: the vectorized form keeps the
same operation order as the per-location path, so identical float results are
achievable and any divergence is a real bug, not expected rounding drift.

scPDSI is out of scope for the spatial path (see ADR-0009) and is asserted to
reject a spatial block explicitly.
"""

from pathlib import Path

import numpy as np
import pytest

from climate_indices import palmer

_DATA_START_YEAR = 1895
_CALIBRATION_START = 1931
_CALIBRATION_END = 1990
_FIXTURE_ROOT = Path(__file__).parent / "fixture" / "palmer"


def _stack_divisions(
    divisions: list[str], awcs: dict, rows: int, cols: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack real fixture divisions into a (time, rows, cols) time-major block."""
    precips = [np.load(_FIXTURE_ROOT / d / "precips.npy") for d in divisions]
    pet = [np.load(_FIXTURE_ROOT / d / "pet.npy") for d in divisions]
    awc = [awcs[d] for d in divisions]
    n = precips[0].shape[0]
    precips_block = np.stack(precips).reshape(rows, cols, n).transpose(2, 0, 1)
    pet_block = np.stack(pet).reshape(rows, cols, n).transpose(2, 0, 1)
    awc_block = np.array(awc).reshape(rows, cols)
    return precips_block, pet_block, awc_block


class TestSpatialBlockMatchesPerLocation:
    """The spatial block path must match calling palmer.pdsi() once per cell.

    Uses real fixture divisions (not synthetic data): the recursion's control
    flow depends on genuine drought/wet-spell dynamics that only show up over
    a real, decades-long climate record, and a prior run of this same check
    against synthetic data missed a bug (a dropped state reset in the
    "spell just ended" dispatch branch) that only 7 of 48 real divisions'
    worth of K-factor rounding happened to expose.
    """

    def test_grid_block_matches_sweep_results(self, palmer_awcs, palmer_pdsi_results):
        divisions = sorted(d for d in palmer_awcs if (_FIXTURE_ROOT / d).is_dir())[:12]
        rows, cols = 3, 4
        precips_block, pet_block, awc_block = _stack_divisions(divisions, palmer_awcs, rows, cols)

        pdsi, phdi, pmdi, zindex, params = palmer.pdsi(
            precips_block,
            pet_block,
            awc_block,
            _DATA_START_YEAR,
            _CALIBRATION_START,
            _CALIBRATION_END,
            spatial_time_major=True,
        )

        assert pdsi.shape == precips_block.shape
        assert params is not None
        assert params["alpha"].shape == (12, rows, cols)

        for idx, division in enumerate(divisions):
            r, c = divmod(idx, cols)
            expected_pdsi, expected_phdi, expected_pmdi, expected_zindex, expected_params = palmer_pdsi_results[
                division
            ]
            np.testing.assert_array_equal(pdsi[:, r, c], expected_pdsi, err_msg=f"{division}: PDSI")
            np.testing.assert_array_equal(phdi[:, r, c], expected_phdi, err_msg=f"{division}: PHDI")
            np.testing.assert_array_equal(pmdi[:, r, c], expected_pmdi, err_msg=f"{division}: PMDI")
            np.testing.assert_array_equal(zindex[:, r, c], expected_zindex, err_msg=f"{division}: Z-Index")
            np.testing.assert_array_equal(
                params["alpha"][:, r, c], expected_params["alpha"], err_msg=f"{division}: alpha"
            )

    def test_a_single_cell_block_matches_the_per_location_call(self, palmer_awcs):
        """ndim == 3 with a single cell still takes the spatial path (n_cells == 1
        either way), and must agree with the un-declared 1-D call."""
        division = "0101"
        precips = np.load(_FIXTURE_ROOT / division / "precips.npy")
        pet = np.load(_FIXTURE_ROOT / division / "pet.npy")
        awc = palmer_awcs[division]

        block_pdsi, block_phdi, block_pmdi, block_z, _ = palmer.pdsi(
            precips.reshape(-1, 1, 1),
            pet.reshape(-1, 1, 1),
            np.array([[awc]]),
            _DATA_START_YEAR,
            _CALIBRATION_START,
            _CALIBRATION_END,
            spatial_time_major=True,
        )
        single_pdsi, single_phdi, single_pmdi, single_z, _ = palmer.pdsi(
            precips, pet, awc, _DATA_START_YEAR, _CALIBRATION_START, _CALIBRATION_END
        )

        np.testing.assert_array_equal(block_pdsi[:, 0, 0], single_pdsi)
        np.testing.assert_array_equal(block_phdi[:, 0, 0], single_phdi)
        np.testing.assert_array_equal(block_pmdi[:, 0, 0], single_pmdi)
        np.testing.assert_array_equal(block_z[:, 0, 0], single_z)


def test_undeclared_ambiguous_block_raises():
    """A (time, 12, *cells) block whose first cell axis is a calendar period
    length is ambiguous with (years, periods, *cells) per ADR-0008, and must
    be rejected unless declared."""
    precips_block = np.zeros((10, 12, 3))
    pet_block = np.zeros((10, 12, 3))

    with pytest.raises(ValueError, match="ambiguous"):
        palmer.pdsi(precips_block, pet_block, 5.0, 2000, 2000, 2000)


def test_all_nan_cell_matches_per_location_all_missing_and_does_not_poison_neighbours(palmer_awcs):
    division = "0101"
    precips = np.load(_FIXTURE_ROOT / division / "precips.npy")
    pet = np.load(_FIXTURE_ROOT / division / "pet.npy")
    awc = palmer_awcs[division]

    precips_block = np.stack([precips, np.full_like(precips, np.nan)], axis=-1).reshape(-1, 1, 2)
    pet_block = np.stack([pet, np.full_like(pet, np.nan)], axis=-1).reshape(-1, 1, 2)
    awc_block = np.array([[awc, awc]])

    pdsi, phdi, pmdi, zindex, params = palmer.pdsi(
        precips_block,
        pet_block,
        awc_block,
        _DATA_START_YEAR,
        _CALIBRATION_START,
        _CALIBRATION_END,
        spatial_time_major=True,
    )

    single_pdsi, single_phdi, single_pmdi, single_z, _ = palmer.pdsi(
        precips, pet, awc, _DATA_START_YEAR, _CALIBRATION_START, _CALIBRATION_END
    )
    np.testing.assert_array_equal(pdsi[:, 0, 0], single_pdsi)
    np.testing.assert_array_equal(phdi[:, 0, 0], single_phdi)
    np.testing.assert_array_equal(pmdi[:, 0, 0], single_pmdi)
    np.testing.assert_array_equal(zindex[:, 0, 0], single_z)

    assert np.isnan(pdsi[:, 0, 1]).all()
    assert np.isnan(phdi[:, 0, 1]).all()
    assert np.isnan(pmdi[:, 0, 1]).all()
    assert np.isnan(zindex[:, 0, 1]).all()


def test_per_cell_awc_actually_varies_the_result(palmer_awcs):
    """Guards against a broadcasting bug that applies one AWC to every cell."""
    division = "0101"
    precips = np.load(_FIXTURE_ROOT / division / "precips.npy")
    pet = np.load(_FIXTURE_ROOT / division / "pet.npy")

    precips_block = np.stack([precips, precips], axis=-1).reshape(-1, 1, 2)
    pet_block = np.stack([pet, pet], axis=-1).reshape(-1, 1, 2)
    awc_block = np.array([[3.0, 9.0]])

    pdsi, *_ = palmer.pdsi(
        precips_block,
        pet_block,
        awc_block,
        _DATA_START_YEAR,
        _CALIBRATION_START,
        _CALIBRATION_END,
        spatial_time_major=True,
    )
    assert not np.array_equal(pdsi[:, 0, 0], pdsi[:, 0, 1], equal_nan=True)


def test_legacy_1d_and_2d_input_unaffected_by_spatial_time_major_default(palmer_awcs):
    """spatial_time_major defaults to False, so unmodified callers see no change."""
    division = "0101"
    precips = np.load(_FIXTURE_ROOT / division / "precips.npy")
    pet = np.load(_FIXTURE_ROOT / division / "pet.npy")
    awc = palmer_awcs[division]

    flat_result = palmer.pdsi(precips, pet, awc, _DATA_START_YEAR, _CALIBRATION_START, _CALIBRATION_END)
    reshaped_result = palmer.pdsi(
        precips.reshape(-1, 12), pet.reshape(-1, 12), awc, _DATA_START_YEAR, _CALIBRATION_START, _CALIBRATION_END
    )
    for a, b in zip(flat_result[:4], reshaped_result[:4], strict=True):
        np.testing.assert_array_equal(a, b)
        assert a.ndim == 1


def test_scpdsi_rejects_spatial_block(palmer_awcs):
    division = "0101"
    precips = np.load(_FIXTURE_ROOT / division / "precips.npy")
    pet = np.load(_FIXTURE_ROOT / division / "pet.npy")
    awc = palmer_awcs[division]
    precips_block = precips.reshape(-1, 1, 1)
    pet_block = pet.reshape(-1, 1, 1)

    with pytest.raises(ValueError, match="spatial block"):
        palmer.scpdsi(precips_block, pet_block, awc, _DATA_START_YEAR, _CALIBRATION_START, _CALIBRATION_END)
