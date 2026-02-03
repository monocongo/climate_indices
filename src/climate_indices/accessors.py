"""Xarray accessors for climate indices.

This module provides xarray DataArray accessors for computing climate indices.
The accessors support both 1-D time series and N-dimensional arrays (e.g., time, lat, lon),
using xr.apply_ufunc for efficient broadcasting and Dask parallelization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from climate_indices import compute, indices, utils

if TYPE_CHECKING:
    pass

# Valid range for fitted indices (SPI, SPEI)
_FITTED_INDEX_VALID_MIN = -3.09
_FITTED_INDEX_VALID_MAX = 3.09

_logger = utils.get_logger(__name__, logging.DEBUG)


@xr.register_dataarray_accessor("indices")
class IndicesAccessor:
    """Accessor for computing climate indices on xarray DataArray objects.

    Supports both 1-D time series and N-dimensional arrays (e.g., time, lat, lon).
    For multi-dimensional arrays, computation is vectorized over non-time dimensions
    using xr.apply_ufunc with Dask parallelization support.

    Example
    -------
    >>> import xarray as xr
    >>> from climate_indices.compute import Periodicity
    >>> # Load a 3-D precipitation dataset (time, lat, lon)
    >>> ds = xr.open_dataset("precip.nc", chunks={"time": -1})
    >>> # Compute SPI using the accessor
    >>> spi = ds.precip.indices.spi(
    ...     scale=3,
    ...     distribution="gamma",
    ...     data_start_year=1981,
    ...     calibration_year_initial=1981,
    ...     calibration_year_final=2010,
    ...     periodicity=Periodicity.monthly
    ... )
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def _detect_time_dim(self) -> str:
        """Detect the time dimension name from the DataArray.

        Returns
        -------
        str
            The name of the time dimension.

        Raises
        ------
        ValueError
            If no time dimension can be detected.
        """
        # common time dimension names
        time_names = {"time", "t", "Time", "TIME", "datetime", "date"}

        for dim in self._obj.dims:
            if dim in time_names:
                return dim

        # fallback: assume first dimension is time
        if self._obj.ndim >= 1:
            _logger.warning(
                "Could not detect time dimension by name; assuming first dimension '%s' is time",
                self._obj.dims[0],
            )
            return self._obj.dims[0]

        msg = "Cannot detect time dimension from DataArray with no dimensions"
        raise ValueError(msg)

    def _validate_time_chunks(self, time_dim: str) -> None:
        """Validate that the time dimension is not chunked into multiple pieces.

        SPI/SPEI require the full time history for rolling sums and distribution
        fitting, so the time dimension must be a single contiguous chunk.

        Parameters
        ----------
        time_dim : str
            Name of the time dimension.

        Raises
        ------
        ValueError
            If the time dimension is split across multiple chunks.
        """
        if self._obj.chunks is None:
            return

        time_dim_idx = self._obj.dims.index(time_dim)
        time_chunks = self._obj.chunks[time_dim_idx]

        if len(time_chunks) > 1:
            msg = (
                f"Time dimension '{time_dim}' has multiple chunks {time_chunks}. "
                "SPI/SPEI require the full time history in a single chunk. "
                "Re-chunk with time=-1 before calling this method."
            )
            raise ValueError(msg)

    def _set_spi_attrs(self, result: xr.DataArray, scale: int) -> xr.DataArray:
        """Set CF-compliant attributes on the SPI result.

        Parameters
        ----------
        result : xr.DataArray
            The computed SPI DataArray.
        scale : int
            The time scale used for SPI computation.

        Returns
        -------
        xr.DataArray
            The DataArray with updated attributes.
        """
        result.attrs["standard_name"] = "spi"
        result.attrs["long_name"] = f"Standardized Precipitation Index (Scale: {scale})"
        result.attrs["units"] = "1"
        result.attrs["valid_min"] = _FITTED_INDEX_VALID_MIN
        result.attrs["valid_max"] = _FITTED_INDEX_VALID_MAX
        result.name = "spi"
        return result

    def spi(
        self,
        scale: int,
        distribution: indices.Distribution | str,
        data_start_year: int,
        calibration_year_initial: int,
        calibration_year_final: int,
        periodicity: compute.Periodicity,
        fitting_params: dict[str, Any] | None = None,
    ) -> xr.DataArray:
        """Compute SPI (Standardized Precipitation Index) for a DataArray.

        Supports both 1-D time series and N-dimensional arrays (e.g., time, lat, lon).
        For multi-dimensional arrays, computation is vectorized over non-time dimensions
        using xr.apply_ufunc with optional Dask parallelization.

        Parameters
        ----------
        scale : int
            Number of time steps over which the values should be scaled
            before the index is computed (e.g., 3 for 3-month SPI).
        distribution : Distribution or str
            Distribution type for fitting: 'gamma' or 'pearson'.
        data_start_year : int
            The initial year of the input precipitation dataset.
        calibration_year_initial : int
            Initial year of the calibration period.
        calibration_year_final : int
            Final year of the calibration period.
        periodicity : Periodicity
            The periodicity of the time series: Periodicity.monthly or Periodicity.daily.
        fitting_params : dict, optional
            Pre-computed distribution fitting parameters. For gamma distribution,
            should contain 'alpha' and 'beta' arrays. For Pearson distribution,
            should contain 'prob_zero', 'loc', 'scale', and 'skew' arrays.

        Returns
        -------
        xr.DataArray
            SPI values with the same dimensions and coordinates as the input.
            Values are clipped to [-3.09, 3.09].

        Notes
        -----
        For Dask-backed arrays, the time dimension must be a single chunk
        (use ``da.chunk({"time": -1})``). The function will raise a ValueError
        if the time dimension is split across multiple chunks.

        Example
        -------
        >>> # 1-D time series
        >>> spi = precip_1d.indices.spi(
        ...     scale=3, distribution="gamma", data_start_year=2000,
        ...     calibration_year_initial=2000, calibration_year_final=2020,
        ...     periodicity=Periodicity.monthly
        ... )
        >>> # 3-D gridded data with Dask
        >>> precip_3d = xr.open_dataset("precip.nc", chunks={"time": -1}).precip
        >>> spi = precip_3d.indices.spi(
        ...     scale=6, distribution="gamma", data_start_year=1981,
        ...     calibration_year_initial=1981, calibration_year_final=2010,
        ...     periodicity=Periodicity.monthly
        ... )
        """
        # normalize distribution argument
        if isinstance(distribution, str):
            try:
                distribution = indices.Distribution[distribution]
            except KeyError:
                distribution = indices.Distribution(distribution)

        # detect time dimension
        time_dim = self._detect_time_dim()

        # validate Dask chunking
        self._validate_time_chunks(time_dim)

        # for 1-D arrays, use the fast legacy path
        if self._obj.ndim == 1:
            return self._spi_1d(
                scale=scale,
                distribution=distribution,
                data_start_year=data_start_year,
                calibration_year_initial=calibration_year_initial,
                calibration_year_final=calibration_year_final,
                periodicity=periodicity,
                fitting_params=fitting_params,
            )

        # for multi-dimensional arrays, use spi_xarray with apply_ufunc
        result = indices.spi_xarray(
            self._obj,
            scale=scale,
            distribution=distribution,
            data_start_year=data_start_year,
            calibration_year_initial=calibration_year_initial,
            calibration_year_final=calibration_year_final,
            periodicity=periodicity,
            fitting_params=fitting_params,
        )

        # apply_ufunc with output_core_dims moves time to the end
        # restore original dimension order
        if result.dims != self._obj.dims:
            result = result.transpose(*self._obj.dims)

        # set CF-compliant attributes
        return self._set_spi_attrs(result, scale)

    def _spi_1d(
        self,
        scale: int,
        distribution: indices.Distribution,
        data_start_year: int,
        calibration_year_initial: int,
        calibration_year_final: int,
        periodicity: compute.Periodicity,
        fitting_params: dict[str, Any] | None = None,
    ) -> xr.DataArray:
        """Compute SPI for a 1-D DataArray using the NumPy-based implementation.

        This is the fast path for simple time series that don't require
        broadcasting over spatial dimensions.
        """
        values = np.asarray(self._obj.values)
        spi_values = indices.spi(
            values,
            scale,
            distribution,
            data_start_year,
            calibration_year_initial,
            calibration_year_final,
            periodicity,
            fitting_params=fitting_params,
        )

        # preserve input attributes and add SPI-specific ones
        attrs = dict(self._obj.attrs)
        result = xr.DataArray(
            spi_values,
            coords=self._obj.coords,
            dims=self._obj.dims,
            name="spi",
            attrs=attrs,
        )

        return self._set_spi_attrs(result, scale)
