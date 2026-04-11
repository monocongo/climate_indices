---
path: src/models/**
---

# Database / ORM Conventions

`src/models/` does not exist in climate_indices v2.4.0. This is a structural placeholder.

climate_indices uses xarray/dask/NetCDF for data, not a relational database. If ORM models
are added in the future, encode SQLAlchemy or similar conventions here:
- Model naming, relationship declarations
- Migration strategy
- Session management patterns
