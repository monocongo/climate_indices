"""Tests for ``suitability.drought_category`` and ``compare_categories`` (#1587, ADR-0028 section 7)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from climate_indices import suitability
from climate_indices.exceptions import BetaFeatureWarning, InvalidArgumentError

pytestmark = pytest.mark.filterwarnings("ignore::climate_indices.exceptions.BetaFeatureWarning")

# USDM "Drought Classification" table, SPI/SPEI column, one literal per class boundary
# (https://droughtmonitor.unl.edu/About/AbouttheData/DroughtClassification.aspx).
# Code: -1 missing, 0 none, 1..5 = D0..D4. The page leaves the exact edges to ADR-0028: each
# class includes its upper edge, so -0.5 is D0 and -2.0 is D4.
USDM_CASES = [
    (3.09, 0),
    (0.0, 0),
    (-0.49, 0),
    (-0.5, 1),
    (-0.79, 1),
    (-0.8, 2),
    (-1.29, 2),
    (-1.3, 3),
    (-1.59, 3),
    (-1.6, 4),
    (-1.99, 4),
    (-2.0, 5),
    (-3.09, 5),
    (-10.0, 5),
]


def _categories(values, **kwargs):
    kwargs.setdefault("output_scale", "normal")
    return suitability.drought_category(values, **kwargs)


class TestDroughtCategory:
    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    @pytest.mark.parametrize(("value", "expected"), USDM_CASES)
    def test_usdm_table_edges(self, value, expected, dtype):
        # the literal is stored in the input dtype, so a float32 edge lands on its own edge
        assert _categories(np.array([value], dtype=dtype))[0] == expected

    def test_adjacent_floats_either_side_of_each_edge(self):
        for edge, upper in zip((-2.0, -1.6, -1.3, -0.8, -0.5), (5, 4, 3, 2, 1), strict=True):
            below, above = np.nextafter(edge, -np.inf), np.nextafter(edge, np.inf)
            assert _categories(np.array([edge]))[0] == upper
            assert _categories(np.array([below]))[0] == upper  # a class spans (lower edge, upper edge]
            assert _categories(np.array([above]))[0] == upper - 1

    def test_nan_is_minus_one(self):
        assert _categories(np.array([np.nan, -1.0, np.nan])).tolist() == [-1, 2, -1]

    def test_masked_values_are_missing(self):
        values = np.ma.masked_array([-1.0, -1.0, 0.5], mask=[False, True, False])
        assert _categories(values).tolist() == [2, -1, 0]

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_masked_edges_keep_the_input_width(self, dtype):
        # a masked float32 or float16 must not be widened to float64 before the edges are compared
        edges = np.array([-0.5, -0.8, -1.3, -1.6, -2.0], dtype=dtype)
        plain = _categories(edges)
        assert plain.tolist() == [1, 2, 3, 4, 5]
        assert _categories(np.ma.masked_array(edges, mask=False)).tolist() == plain.tolist()

    def test_infinity_in_a_lazy_input_raises_when_computed(self):
        dask_array = pytest.importorskip("dask.array")
        da = xr.DataArray(dask_array.from_array(np.array([np.inf, 1.0]), chunks=1), dims="x")
        lazy = _categories(da)  # the data is not read at call time
        with pytest.raises(InvalidArgumentError, match="infinite"):
            lazy.compute()

    def test_int8_output_and_shape_preserved(self):
        out = _categories(np.zeros((4, 3, 2)))
        assert out.dtype == np.int8
        assert out.shape == (4, 3, 2)

    def test_integer_input(self):
        assert _categories(np.array([0, -1, -2, -3])).tolist() == [0, 2, 5, 5]

    @pytest.mark.parametrize("scale", ["probability", "bounded"])
    def test_non_normal_scale_is_rejected(self, scale):
        with pytest.raises(InvalidArgumentError, match="normal"):
            _categories(np.array([0.5]), output_scale=scale)

    @pytest.mark.parametrize("scale", ["Normal", "", None, 1])
    def test_unknown_scale_is_rejected(self, scale):
        with pytest.raises(InvalidArgumentError):
            _categories(np.array([0.5]), output_scale=scale)

    def test_output_scale_is_required(self):
        with pytest.raises(TypeError):
            suitability.drought_category(np.array([0.5]))  # type: ignore[call-arg]

    def test_infinity_is_rejected(self):
        with pytest.raises(InvalidArgumentError, match="infinite"):
            _categories(np.array([np.inf]))

    def test_non_numeric_is_rejected(self):
        with pytest.raises(InvalidArgumentError):
            _categories(np.array(["a"]))

    def test_beta_warning(self):
        with pytest.warns(BetaFeatureWarning):
            _categories(np.array([0.0]))

    def test_dataarray_keeps_coords_and_records_edges(self):
        da = xr.DataArray(
            [[-0.5, np.nan], [-2.5, 1.0]], dims=("y", "x"), coords={"y": [10, 20]}, attrs={"long_name": "SPI-3"}
        )
        out = _categories(da)
        assert isinstance(out, xr.DataArray)
        assert out.dtype == np.int8
        assert out.values.tolist() == [[1, -1], [5, 0]]
        assert out.dims == da.dims
        assert out["y"].values.tolist() == [10, 20]
        assert out.attrs["long_name"] != "SPI-3"  # the input's description does not describe categories
        assert out.attrs["flag_values"] == [-1, 0, 1, 2, 3, 4, 5]
        assert out.attrs["flag_meanings"] == "missing none D0 D1 D2 D3 D4"
        assert json.loads(out.attrs["climate_indices_category_edges"]) == [-2.0, -1.6, -1.3, -0.8, -0.5]
        assert "droughtmonitor.unl.edu" in out.attrs["climate_indices_category_source"]
        assert "climate_indices_provenance" in out.attrs

    def test_dask_is_lazy_and_matches_numpy(self):
        dask_array = pytest.importorskip("dask.array")
        data = np.random.default_rng(0).normal(size=(30, 4, 3))
        da = xr.DataArray(dask_array.from_array(data, chunks=(10, 2, 3)), dims=("time", "y", "x"))
        out = _categories(da)
        assert out.chunks is not None
        np.testing.assert_array_equal(out.values, _categories(data))


# reference and alternative categories for the hand-computed fixture
#   idx:  0     1     2     3     4      5     6      7     8     9
REF = np.array([0, 1, 1, 2, 3, 5, -1, 2, 0, 4], dtype=np.int8)
ALT = np.array([0, 1, 2, 2, -1, 4, 1, 0, 1, 4], dtype=np.int8)
# idx 4 and 6 hold a missing value and drop out, leaving 8 valid pairs:
#   (0,0) (1,1) (1,2) (2,2) (5,4) (2,0) (0,1) (4,4)
# rows are the reference class, columns the alternative class
CONFUSION = np.zeros((6, 6), dtype=np.int64)
for _r, _a in [(0, 0), (1, 1), (1, 2), (2, 2), (5, 4), (2, 0), (0, 1), (4, 4)]:
    CONFUSION[_r, _a] += 1


class TestCompareCategoriesNumpy:
    def test_hand_computed_fixture(self):
        result = suitability.compare_categories(REF, ALT)
        np.testing.assert_array_equal(result.confusion, CONFUSION)
        assert result.confusion.dtype == np.int64
        assert result.n_valid == 8
        # 4 of the 8 pairs differ: (1,2) (5,4) (2,0) (0,1)
        assert result.share_changed == pytest.approx(4 / 8)
        # the one (0,0) pair is outside drought, so the 4 changes are of 7 pairs
        assert result.share_changed_in_drought == pytest.approx(4 / 7)
        np.testing.assert_array_equal(result.reference_counts, [2, 2, 2, 0, 1, 1])
        np.testing.assert_array_equal(result.alternative_counts, [2, 2, 2, 0, 2, 0])
        assert result.confusion.sum() == result.n_valid

    def test_symmetric_roles_for_in_drought_denominator(self):
        # (0, 3) and (3, 0) change into and out of drought, (3, 3) is unchanged drought, (0, 0) is outside it.
        # Counting only reference-in-drought or only alternative-in-drought pairs would give 1/2, not 2/3.
        result = suitability.compare_categories(np.array([0, 3, 3, 0]), np.array([3, 0, 3, 0]))
        assert result.share_changed == pytest.approx(2 / 4)
        assert result.share_changed_in_drought == pytest.approx(2 / 3)

    def test_masked_code_counts_as_missing(self):
        # the value under the mask (2) must not be counted as a valid pair
        masked = np.ma.masked_array([1, 2, 3], mask=[False, True, False])
        result = suitability.compare_categories(masked, np.array([1, 2, 3]))
        assert result.n_valid == 2
        assert result.confusion[2, 2] == 0

    def test_unmasked_out_of_range_uint64_is_rejected_like_a_plain_array(self):
        # the int64 cast would wrap UINT64_MAX to -1 and silently drop it as missing
        bad = np.array([1, np.iinfo(np.uint64).max], dtype=np.uint64)
        masked = np.ma.masked_array(bad, mask=[False, False])
        with pytest.raises(InvalidArgumentError, match="outside"):
            suitability.compare_categories(masked, np.array([1, 2]))
        # a bad code under the mask is still ignored
        hidden = np.ma.masked_array(bad, mask=[False, True])
        assert suitability.compare_categories(hidden, np.array([1, 2])).n_valid == 1

    def test_masked_non_integer_codes_are_rejected(self):
        with pytest.raises(InvalidArgumentError, match="integer"):
            suitability.compare_categories(np.ma.masked_array([1.0, 2.0], mask=[False, True]), np.array([1, 2]))

    def test_identical_categories_have_no_change(self):
        result = suitability.compare_categories(REF, REF)
        assert result.share_changed == 0.0
        assert result.share_changed_in_drought == 0.0

    def test_no_drought_pairs_leaves_in_drought_share_undefined(self):
        zeros = np.zeros(5, dtype=np.int8)
        result = suitability.compare_categories(zeros, zeros)
        assert result.share_changed == 0.0
        assert np.isnan(result.share_changed_in_drought)

    def test_no_valid_pairs_leaves_shares_undefined(self):
        missing = np.full(4, -1, dtype=np.int8)
        result = suitability.compare_categories(missing, missing)
        assert result.n_valid == 0
        assert np.isnan(result.share_changed)
        assert np.isnan(result.share_changed_in_drought)
        assert not result.confusion.any()

    def test_multidimensional_inputs_reduce_over_everything(self):
        result = suitability.compare_categories(REF.reshape(2, 5), ALT.reshape(2, 5))
        np.testing.assert_array_equal(result.confusion, CONFUSION)

    def test_end_to_end_from_index_values(self):
        values = np.array([-0.5, -0.8, -1.3, np.nan, 0.2])
        shifted = values - 0.01  # each value stays in its class: D0, D1, D2 and none
        result = suitability.compare_categories(_categories(values), _categories(shifted))
        assert result.n_valid == 4
        assert result.share_changed == 0.0

    def test_metadata_records_edges_source_and_provenance(self):
        meta = suitability.compare_categories(REF, ALT).metadata
        assert json.loads(meta["climate_indices_category_edges"]) == [-2.0, -1.6, -1.3, -0.8, -0.5]
        assert "droughtmonitor.unl.edu" in meta["climate_indices_category_source"]
        assert "climate_indices_provenance" in meta

    def test_dims_are_xarray_only(self):
        with pytest.raises(InvalidArgumentError, match="dims"):
            suitability.compare_categories(REF, ALT, dims=["x"])

    @pytest.mark.parametrize("bad", [np.array([0, 6]), np.array([0, -2]), np.array([0, 1000])])
    def test_out_of_range_code_is_rejected(self, bad):
        # a code of 7 would otherwise land in another cell of the flattened confusion matrix
        with pytest.raises(InvalidArgumentError, match="-1"):
            suitability.compare_categories(np.array([0, 0]), bad)
        with pytest.raises(InvalidArgumentError, match="-1"):
            suitability.compare_categories(bad, np.array([0, 0]))

    @pytest.mark.parametrize("bad", [np.array([0.0, 1.0]), np.array([True, False]), np.array(["a", "b"])])
    def test_non_integer_codes_are_rejected(self, bad):
        with pytest.raises(InvalidArgumentError, match="integer"):
            suitability.compare_categories(bad, bad)

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(InvalidArgumentError, match="shape"):
            suitability.compare_categories(REF, ALT[:-1])

    def test_mixing_numpy_and_xarray_is_rejected(self):
        with pytest.raises(InvalidArgumentError, match="both"):
            suitability.compare_categories(REF, xr.DataArray(ALT, dims="time"))

    def test_beta_warning(self):
        with pytest.warns(BetaFeatureWarning):
            suitability.compare_categories(REF, ALT)


def _grid(seed: int = 0, shape: tuple[int, int, int] = (40, 3, 2)):
    """Reference/alternative categories on a (time, y, x) grid; the alternative is a perturbed copy."""
    rng = np.random.default_rng(seed)
    ref_values = rng.normal(size=shape)
    alt_values = ref_values + rng.normal(scale=0.3, size=shape)
    ref_values[rng.random(shape) < 0.05] = np.nan
    alt_values[rng.random(shape) < 0.05] = np.nan  # a different set, so a pair can miss on either side
    return _categories(ref_values), _categories(alt_values)


class TestCompareCategoriesXarray:
    def test_default_reduces_every_dimension(self):
        ref, alt = _grid()
        out = suitability.compare_categories(
            xr.DataArray(ref, dims=("time", "y", "x")), xr.DataArray(alt, dims=("time", "y", "x"))
        )
        expected = suitability.compare_categories(ref, alt)
        assert isinstance(out, xr.Dataset)
        assert out["confusion"].dims == ("reference_category", "alternative_category")
        np.testing.assert_array_equal(out["confusion"].values, expected.confusion)
        assert out["confusion"].dtype == np.int64
        assert out["n_valid"].item() == expected.n_valid
        assert out["share_changed"].item() == pytest.approx(expected.share_changed)
        assert out["share_changed_in_drought"].item() == pytest.approx(expected.share_changed_in_drought)
        np.testing.assert_array_equal(out["reference_counts"].values, expected.reference_counts)
        np.testing.assert_array_equal(out["alternative_counts"].values, expected.alternative_counts)

    def test_hand_computed_fixture(self):
        # the NumPy fixture's hand-derived values, so this route is checked against them and not only the NumPy route
        out = suitability.compare_categories(xr.DataArray(REF, dims="time"), xr.DataArray(ALT, dims="time"))
        np.testing.assert_array_equal(out["confusion"].values, CONFUSION)
        assert out["n_valid"].item() == 8
        assert out["share_changed"].item() == pytest.approx(4 / 8)
        assert out["share_changed_in_drought"].item() == pytest.approx(4 / 7)
        np.testing.assert_array_equal(out["reference_counts"].values, [2, 2, 2, 0, 1, 1])
        np.testing.assert_array_equal(out["alternative_counts"].values, [2, 2, 2, 0, 2, 0])

    def test_pair_missing_only_in_the_alternative_is_excluded(self):
        ref = xr.DataArray(np.array([0, 1, -1, 2], dtype=np.int8), dims="time")
        alt = xr.DataArray(np.array([0, -1, 0, 2], dtype=np.int8), dims="time")
        out = suitability.compare_categories(ref, alt)
        assert out["n_valid"].item() == 2
        assert out["confusion"].sum().item() == 2

    def test_category_coordinates_are_labels(self):
        ref, alt = _grid()
        out = suitability.compare_categories(
            xr.DataArray(ref, dims=("time", "y", "x")), xr.DataArray(alt, dims=("time", "y", "x"))
        )
        labels = ["none", "D0", "D1", "D2", "D3", "D4"]
        assert out["reference_category"].values.tolist() == labels
        assert out["alternative_category"].values.tolist() == labels
        assert out["reference_counts"].dims == ("reference_category",)
        assert out["alternative_counts"].dims == ("alternative_category",)

    def test_reduce_over_time_only_keeps_the_cells(self):
        ref, alt = _grid()
        out = suitability.compare_categories(
            xr.DataArray(ref, dims=("time", "y", "x")), xr.DataArray(alt, dims=("time", "y", "x")), dims="time"
        )
        assert set(out["confusion"].dims) == {"y", "x", "reference_category", "alternative_category"}
        assert out["share_changed"].dims == ("y", "x")
        for j in range(3):
            for i in range(2):
                cell = suitability.compare_categories(ref[:, j, i], alt[:, j, i])
                got = out.isel(y=j, x=i)
                np.testing.assert_array_equal(got["confusion"].values, cell.confusion)
                assert got["n_valid"].item() == cell.n_valid
                np.testing.assert_allclose(got["share_changed"].item(), cell.share_changed, equal_nan=True)
                np.testing.assert_allclose(
                    got["share_changed_in_drought"].item(), cell.share_changed_in_drought, equal_nan=True
                )

    def test_dims_accepts_a_sequence(self):
        ref, alt = _grid()
        out = suitability.compare_categories(
            xr.DataArray(ref, dims=("time", "y", "x")),
            xr.DataArray(alt, dims=("time", "y", "x")),
            dims=["time", "y"],
        )
        assert out["share_changed"].dims == ("x",)
        expected = suitability.compare_categories(ref[:, :, 1], alt[:, :, 1])
        np.testing.assert_array_equal(out.isel(x=1)["confusion"].values, expected.confusion)

    def test_unknown_dim_is_rejected(self):
        ref, alt = _grid()
        da = xr.DataArray(ref, dims=("time", "y", "x"))
        with pytest.raises(InvalidArgumentError, match="lat"):
            suitability.compare_categories(da, da, dims="lat")

    def test_mismatched_dims_or_coordinates_are_rejected(self):
        a = xr.DataArray(np.zeros(3, dtype=np.int8), dims="time", coords={"time": [1, 2, 3]})
        with pytest.raises(InvalidArgumentError):
            suitability.compare_categories(a, a.rename(time="t"))
        with pytest.raises(InvalidArgumentError):
            suitability.compare_categories(a, a.assign_coords(time=[1, 2, 4]))

    @pytest.mark.parametrize("bad", [["time", "time"], 0, ..., []])
    def test_invalid_dims_raise_invalid_argument_error(self, bad):
        da = xr.DataArray(np.zeros((3, 2), dtype=np.int8), dims=("time", "x"))
        with pytest.raises(InvalidArgumentError, match="dim"):
            suitability.compare_categories(da, da, dims=bad)

    def test_kept_dimension_named_like_a_confusion_axis_is_rejected(self):
        da = xr.DataArray(np.zeros((3, 2), dtype=np.int8), dims=("time", "reference_category"))
        with pytest.raises(InvalidArgumentError, match="reserved"):
            suitability.compare_categories(da, da, dims="time")
        # reducing over it is fine
        assert suitability.compare_categories(da, da, dims="reference_category")["n_valid"].dims == ("time",)

    def test_stray_code_in_a_lazy_input_is_excluded_from_every_field(self):
        dask_array = pytest.importorskip("dask.array")
        ref = xr.DataArray(dask_array.from_array(np.array([0, 7, -5, 1, 2], dtype=np.int8), chunks=2), dims="t")
        alt = xr.DataArray(dask_array.from_array(np.array([0, 1, 1, 2, 2], dtype=np.int8), chunks=2), dims="t")
        out = suitability.compare_categories(ref, alt).compute()
        # pairs (0,0), (1,2), (2,2) remain; the 7 and the -5 are not counted anywhere
        assert out["n_valid"].item() == out["confusion"].sum().item() == 3
        assert out["share_changed"].item() == pytest.approx(1 / 3)

    def test_non_integer_dtype_is_rejected(self):
        da = xr.DataArray(np.array([0.0, 1.0]), dims="time")
        with pytest.raises(InvalidArgumentError, match="integer"):
            suitability.compare_categories(da, da)

    def test_out_of_range_code_is_rejected_for_in_memory_data(self):
        good = xr.DataArray(np.zeros(2, dtype=np.int8), dims="time")
        bad = xr.DataArray(np.array([0, 7], dtype=np.int8), dims="time")
        with pytest.raises(InvalidArgumentError, match="-1"):
            suitability.compare_categories(good, bad)

    def test_all_missing_cell_is_nan(self):
        ref = xr.DataArray(np.array([[0, 1], [-1, -1]], dtype=np.int8), dims=("cell", "time"))
        out = suitability.compare_categories(ref, ref, dims="time")
        assert out["n_valid"].values.tolist() == [2, 0]
        assert np.isnan(out["share_changed"].values[1])
        assert out["share_changed"].values[0] == 0.0

    def test_attributes_record_edges_source_and_provenance(self):
        ref, alt = _grid()
        out = suitability.compare_categories(
            xr.DataArray(ref, dims=("time", "y", "x")), xr.DataArray(alt, dims=("time", "y", "x"))
        )
        assert json.loads(out.attrs["climate_indices_category_edges"]) == [-2.0, -1.6, -1.3, -0.8, -0.5]
        assert "droughtmonitor.unl.edu" in out.attrs["climate_indices_category_source"]
        assert "climate_indices_provenance" in out.attrs
        assert out.attrs["climate_indices_version"]

    def test_dask_matches_numpy_and_stays_lazy(self):
        dask_array = pytest.importorskip("dask.array")
        ref, alt = _grid(seed=3)
        # time is split across chunks: the reduction needs no single-chunk time axis
        ref_da = xr.DataArray(dask_array.from_array(ref, chunks=(7, 3, 1)), dims=("time", "y", "x"))
        alt_da = xr.DataArray(dask_array.from_array(alt, chunks=(11, 3, 2)), dims=("time", "y", "x"))
        out = suitability.compare_categories(ref_da, alt_da)
        for name in ("confusion", "reference_counts", "alternative_counts", "n_valid", "share_changed"):
            assert out[name].chunks is not None, name
        expected = suitability.compare_categories(ref, alt)
        np.testing.assert_array_equal(out["confusion"].values, expected.confusion)
        assert float(out["share_changed"]) == pytest.approx(expected.share_changed)

    def test_drought_category_output_feeds_compare_categories(self):
        rng = np.random.default_rng(5)
        spi = xr.DataArray(rng.normal(size=(60, 2)), dims=("time", "x"))
        other = spi + 0.2
        out = suitability.compare_categories(_categories(spi), _categories(other), dims="time")
        assert out["share_changed"].dims == ("x",)
        assert (out["share_changed"] > 0).all()
