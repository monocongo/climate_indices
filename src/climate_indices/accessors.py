"""Xarray accessors for climate indices."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from climate_indices import compute, indices


@xr.register_dataarray_accessor("indices")
class IndicesAccessor:
    """Accessor for computing climate indices on xarray DataArray objects."""

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

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
        """
        Compute SPI for a 1-D DataArray and return a DataArray with matching coords.
        """
        if self._obj.ndim != 1:
            raise ValueError("SPI accessor currently supports 1-D DataArray inputs only")

        if isinstance(distribution, str):
            try:
                distribution = indices.Distribution[distribution]
            except KeyError:
                distribution = indices.Distribution(distribution)

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

        attrs = dict(self._obj.attrs)
        attrs.update({"long_name": "Standardized Precipitation Index", "units": "unitless"})

        return xr.DataArray(
            spi_values,
            coords=self._obj.coords,
            dims=self._obj.dims,
            name="spi",
            attrs=attrs,
        )
