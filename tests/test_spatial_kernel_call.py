"""Contract tests for the manual adapters' shared Spatial Kernel call."""

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import utils
from climate_indices.xarray_adapter import _spatial_kernel_call, _SpatialOutput


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("cell_count", [2, 12])
def test_four_outputs_preserve_layout_metadata_and_cell_parameters(chunked, cell_count, caplog):
    """A four-output kernel receives declared blocks, including an ambiguous cell axis."""
    time = pd.date_range("2001-01-01", periods=24, freq="MS")
    source = xr.DataArray(
        np.arange(cell_count * 24 * 3, dtype=float).reshape(cell_count, 24, 3),
        dims=("lat", "date", "lon"),
        coords={"date": time, "lat": np.arange(cell_count), "lon": [1, 2, 3]},
        attrs={"history": "source history", "standard_name": "air_temperature", "valid_range": [-1, 1]},
    )
    source.date.attrs["axis"] = "T"
    cell_param = xr.DataArray(np.arange(cell_count * 3).reshape(3, cell_count), dims=("lon", "lat")).T
    if chunked:
        source = source.chunk({"date": -1, "lat": 2, "lon": 1})
    calls = []

    def kernel(first, second, cells, *, spatial_time_major):
        assert spatial_time_major is True
        assert first.shape[0] == 24
        calls.append(first.shape)
        return tuple(first + second + cells + offset for offset in range(4))

    declarations = tuple(_SpatialOutput("date", {"units": "1"}, name) for name in ("PDSI", "PHDI", "PMDI", "Z-Index"))
    with caplog.at_level(logging.INFO):
        results = _spatial_kernel_call(
            kernel,
            (source, source * 2),
            cell_params=(cell_param,),
            spatial_kernel=True,
            outputs=declarations,
            calculation_metadata={"data_start_year": 2001},
            completion_event="packing_completed",
            time_dim="date",
        )
    assert bool(calls) is not chunked
    for offset, result in enumerate(results):
        assert result.dims == source.dims
        xr.testing.assert_equal(result.coords.to_dataset(), source.coords.to_dataset())
        # Integer arithmetic in f64 is exact for these small values.
        np.testing.assert_array_equal(result.compute(), source * 3 + cell_param + offset)
        assert result.attrs["valid_range"] == [-1, 1]
        assert result.attrs["units"] == "1"
        assert result.attrs["data_start_year"] == 2001
        assert "standard_name" not in result.attrs
        assert result.attrs["history"].startswith("source history\n")
        assert declarations[offset].index_name in result.attrs["history"]
    assert any("packing_completed" in record.message for record in caplog.records)


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("spatial_kernel", [False, True])
@pytest.mark.parametrize("cell_shape", [(), (2, 2), (2, 3)])
def test_single_reduced_output_preserves_spatial_values(chunked, spatial_kernel, cell_shape):
    """A sole reduced output keeps its cell axes, including a scalar result."""
    dims = ("time", "lat", "lon")[: 1 + len(cell_shape)]
    shape = (24, *cell_shape)
    source = xr.DataArray(
        np.arange(np.prod(shape), dtype=float).reshape(shape),
        dims=dims,
        coords={dim: np.arange(size) for dim, size in zip(dims, shape, strict=True)},
    )
    expected = source.mean("time")
    if chunked:
        source = source.chunk({"time": -1})
    calls = []

    def kernel(series, **kwargs):
        assert kwargs.get("spatial_time_major", False) is spatial_kernel
        calls.append(series.shape)
        return series.mean(axis=0 if spatial_kernel else -1)

    (result,) = _spatial_kernel_call(
        kernel,
        (source,),
        spatial_kernel=spatial_kernel,
        outputs=(_SpatialOutput(None, {}, "Mean"),),
        calculation_metadata={},
        completion_event="reduction_completed",
    )
    assert bool(calls) is not chunked
    xr.testing.assert_equal(result.compute(), expected)


@pytest.mark.parametrize("spatial_kernel", [False, True])
def test_calendar_conversion_applies_only_to_time_series_outputs(spatial_kernel):
    """Static inputs and reduced outputs do not receive Gregorian restoration."""
    time = pd.date_range("2001-01-01", "2002-03-02", freq="D")
    values = np.arange(len(time), dtype=float)
    source = xr.DataArray(values, dims="time", coords={"time": time})
    plan = utils.CalendarPlan.from_year_span(2001, 2, len(time))

    def kernel(series, cell, **kwargs):
        assert series.shape == (732,)
        assert cell == 7
        assert series[59] == (values[58] + values[59]) / 2
        assert kwargs.get("spatial_time_major", False) is spatial_kernel
        return series + cell, np.arange(366, dtype=float), np.asarray(1.0)

    results = _spatial_kernel_call(
        kernel,
        (source,),
        cell_params=(7,),
        spatial_kernel=spatial_kernel,
        calendar_plan=plan,
        outputs=(
            _SpatialOutput("time", {}, "Series"),
            _SpatialOutput("dayofyear", {}, "Fit", coordinate=np.arange(1, 367)),
            _SpatialOutput(None, {}, "Code", drop_attrs=("units",)),
        ),
        calculation_metadata={},
        completion_event="calendar_completed",
    )
    np.testing.assert_array_equal(results[0], values + 7)
    np.testing.assert_array_equal(results[1], np.arange(366))
    np.testing.assert_array_equal(results[1].dayofyear, np.arange(1, 367))
    assert results[2].item() == 1.0
    assert "units" not in results[2].attrs
