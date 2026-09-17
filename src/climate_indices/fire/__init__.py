"""Fire-weather indices computed from standard meteorological inputs.

This module is the NumPy layer of the fire-weather family tracked in #793. It
provides the Fosberg Fire Weather Index and the Hot-Dry-Windy Index, both
weather-only and elementwise, the Haines Index, which scores an atmospheric
layer's stability and moisture, the Keetch-Byram Drought Index, and the three
Canadian Forest Fire Weather Index System (CFFWIS) moisture codes: the Fine
Fuel Moisture Code, the Duff Moisture Code, and the Drought Code. Stateful
functions follow the execution, state-ownership, and append/resume contract
recorded in ``docs/adr/0006-fire-recursive-state-and-execution.md`` and the
missing-data policy recorded in ``docs/adr/0007-fire-missing-data-policy.md``;
the CFFWIS behavior indices (#804) use the same contract, and the
:func:`cffwis` orchestrator threads all three moisture codes through a single
time loop. :func:`overwinter_drought_code` carries the DC across the fire
season's off-season shutdown.

References
----------
Srock, A.F., Charney, J.J., Potter, B.E., Goodrick, S.L. (2018) The
Hot-Dry-Windy Index: A New Fire Weather Index. Atmosphere, 9(7), 279.
doi:10.3390/atmos9070279.

Fosberg, M.A. (1978) Weather in wildland fire management: the fire weather
index. Conference on Sierra Nevada Meteorology, Lake Tahoe, CA, 1-4.

Simard, A.J. (1968) The moisture content of forest fuels - I. A review of the
basic concepts. Canadian Department of Forest and Rural Development, Forest
Fire Research Institute, Information Report FF-X-14.

Goodrick, S.L. (2002) Modification of the Fosberg fire weather index to include
drought. International Journal of Wildland Fire, 11, 205-211.
NCEP GEMPAK, ``pd_fosb`` / ``pr_fosb`` (T. Lee, 2003): the operational
implementation behind the ``FOSINDX`` GRIB2 parameter.
https://github.com/Unidata/gempak

Keetch, J.J. and Byram, G.M. (1968) A Drought Index for Forest Fire Control.
USDA Forest Service Research Paper SE-38.
https://research.fs.usda.gov/treesearch/40

Alexander, M.E. (1990) Computer calculation of the Keetch-Byram Drought
Index - programmers beware! Fire Management Notes, 51(4), 23-25.

Van Wagner, C.E. and Pickett, T.L. (1985) Equations and FORTRAN program for
the Canadian Forest Fire Weather Index System. Canadian Forestry Service,
Forestry Technical Report 33.

Haines, D.A. (1988) A lower atmospheric severity index for wildland fires.
National Weather Digest, 13(2), 23-27.
"""

from __future__ import annotations

from climate_indices.fire._cffwis import (
    CFFWISResult,
    CFFWISState,
    DCResult,
    DCState,
    DMCResult,
    DMCState,
    FFMCResult,
    FFMCState,
    buildup_index,
    cffwis,
    cffwis_fwi,
    daily_severity_rating,
    drought_code,
    duff_moisture_code,
    ffmc,
    initial_spread_index,
    overwinter_drought_code,
)
from climate_indices.fire._fosberg import fosberg_ffwi
from climate_indices.fire._haines import haines_index, haines_index_from_profile
from climate_indices.fire._hdw import hot_dry_windy
from climate_indices.fire._kbdi import KBDIResult, KBDIState, kbdi

# declare the function names that should be included in the public API for this package
__all__ = [
    "CFFWISResult",
    "CFFWISState",
    "DCResult",
    "DCState",
    "DMCResult",
    "DMCState",
    "FFMCResult",
    "FFMCState",
    "KBDIResult",
    "KBDIState",
    "buildup_index",
    "cffwis",
    "cffwis_fwi",
    "daily_severity_rating",
    "drought_code",
    "duff_moisture_code",
    "ffmc",
    "fosberg_ffwi",
    "haines_index",
    "haines_index_from_profile",
    "hot_dry_windy",
    "initial_spread_index",
    "kbdi",
    "overwinter_drought_code",
]
