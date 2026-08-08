# nClimGrid acquisition and redistribution facts

**Research date:** 2026-08-08  
**Scope:** The source supply chain and rights for the Explorer's Demonstration Sample only. This finding does not revisit the interactive-explorer landscape. Every external source below is an NCEI/NOAA primary record or an official NOAA-managed NODD dataset entry.

## Answer

NCEI supplies nClimGrid through a direct archive and designates an NOAA-managed public AWS S3 bucket as an additional download route. The readily discoverable monthly NetCDF inputs are **mutable period-of-record objects**, not immutable per-month source objects: NCEI says the latest month is appended, and it reprocesses preliminary files after two years. The current `.pnt` files are explicitly snapshots for the most recent month, not a historical immutable monthly archive.

A fixed Demonstration Sample can still be provenance-verified, but its Source Manifest must freeze a **retrieval** of the selected period-of-record object(s), rather than claim that NCEI published immutable monthly objects. Record the chosen official endpoint, object key, retrieval time, size, `Last-Modified`, opaque ETag, any source-published checksum, a project-computed cryptographic digest of the downloaded bytes, selected time coordinates, and the reduction recipe. The project digest is necessary because the official checksum observed here is object-level, not per-month.

The NOAA-managed NODD entry says its data are public and may be used as desired. It requests attribution for unaltered dissemination; it prohibits implying NOAA endorsement or affiliation; and it prohibits presenting a modified dataset as original, unaltered NOAA data. That supports project or third-party redistribution of a clearly identified **reduced/modified** Demonstration Sample, subject to the selected host's own terms. It is not legal advice.

## Sourced evidence

### 1. Authoritative acquisition and object discovery

- NCEI is the distributor in the [NClimGrid ISO metadata record][metadata]. It labels [`/data/nclimgrid-monthly/access/`][archive] as **NCEI Direct Download (archive)** and names the [monthly NOAA S3 bucket][s3-explorer] as **AWS S3 Explorer (Region: us-east-1)**. Therefore both are official acquisition routes; the AWS path is not an unaffiliated mirror.
- NCEI's [gridded-data README][readme] names the four archive NetCDF objects as `nclimgrid_{tavg,tmax,tmin,prcp}.nc`. It says they contain monthly data from 1895 to present and that "the latest month is appended to these period of record NetCDF files." Fixed names therefore do **not** identify immutable objects.
- The same README calls the `.pnt` inputs snapshots of the data used for a single month's nClimDiv processing and says they cover the "most recent single month." On 2026-08-08, the [NCEI archive directory][archive] and the [official S3 listing][s3-listing] exposed four period-of-record NetCDF objects and eight `202606.*.pnt` objects; the S3 listing reported `KeyCount=13` including `index.html`. This observation is evidence that those public directories did not offer an historical collection of per-month source objects at that time. It is not evidence that no other NCEI archival mechanism exists.
- The NCEI metadata also names a separate [real-time download location][realtime]. That does not establish a versioned historical monthly-object service or an immutable revision identifier.

### 2. Revision and correction behavior

- The metadata lineage says that files are initially released as **preliminary** because of collection delays and that, at the end of two years, preliminary files are reprocessed with all available data so the most complete record is archived. A sample must therefore treat recent and preliminary periods as revisable. [metadata]
- The NOAA-managed [NODD dataset entry][nodd-entry] says that approximately one year of final nClimGrid is submitted annually to replace initially supplied preliminary data. The entry does not explain how this approximate annual statement maps to NCEI's two-year reprocessing statement. This research does not reconcile those descriptions into a revision calendar.
- During this research, the direct NCEI and S3 `nclimgrid_prcp.nc` responses had the same content length (1,465,377,174 bytes) but different serving metadata: the NCEI response reported `Last-Modified: 2026-07-15T20:11:56Z`, ETag `"5757e196-656abef647b00"`; S3 reported `Last-Modified: 2026-08-06T00:01:06Z`, ETag `"b04fd07f359af44b042589882a2cded8"`. This does **not** establish different scientific content. It does establish that a manifest must name its chosen endpoint and record retrieval-time evidence, rather than compare ETags across endpoints.

### 3. Checksums and verification

- The official S3 ListObjectsV2 response supplies an object-level `ChecksumAlgorithm` of `CRC32`, `ChecksumType` of `FULL_OBJECT`, ETag, size, and last-modified time for each listed object. It does not supply a checksum per monthly time coordinate. [s3-listing]
- A `HEAD` request to the official S3 `nclimgrid_prcp.nc` object on 2026-08-08 with `x-amz-checksum-mode: ENABLED` returned `x-amz-checksum-crc32: 3NxyXg==` and `x-amz-checksum-type: FULL_OBJECT`. The equivalent direct NCEI archive response did not expose an `x-amz-checksum-*` header. These are source-object verification values, not a manifest for the individual months used in a reduction. [s3-prcp]
- An ETag must be retained as server metadata, not represented as a portable cryptographic digest. The evidence above does not establish an NCEI-published SHA-256 (or another cryptographic digest) for the archive object or for each monthly slice.

### 4. Citation, reduction, and redistribution

- NCEI specifies this citation in its metadata: "Vose, Russell S., Applequist, Scott, Squires, Mike, Durre, Imke, Menne, Matthew J., Williams, Claude N. Jr., Fenimore, Chris, Gleason, Karin, and Arndt, Derek (2014): NOAA Monthly U.S. Climate Gridded Dataset (NClimGrid), Version 1. **[indicate subset used]**. NOAA National Centers for Environmental Information. DOI:10.7289/V5SX6B56 **[access date]**." A Demonstration Sample must fill in the subset/reduction and access date rather than cite it as the whole unmodified dataset. [metadata]
- The official [NODD dataset entry][nodd-entry] is marked **Managed By NOAA** and states: NOAA data disseminated through NODD are open to the public and "can be used as desired." It requests attribution for use or dissemination of **unaltered** NOAA data, prohibits a claim or implication of NOAA endorsement or affiliation, and says modified data may not be presented as original, unaltered NOAA data.
- Consequently, the primary-source policy supports a project-hosted or third-party-hosted reduced sample when it is labeled as reduced/modified, preserves the NCEI citation and source provenance, and makes no NOAA endorsement or unaltered-data claim. The policy does not select a host or override that host's terms, retention limits, or release-asset size limits.

### 5. Demonstration Sample hosting

- NCEI's metadata establishes the NCEI archive, real-time location, and the official AWS S3 explorer as **source-data** distribution routes. [metadata]
- The NODD entry identifies `noaa-nclimgrid-monthly-pds` as **Monthly NClimGrid Data**, offers anonymous read access, and is managed by NOAA. [nodd-entry]
- No reviewed NCEI/NOAA source establishes a project-facing write/upload service for a `climate_indices`-reduced derivative, a required host for that derivative, or a NCEI approval workflow for it. The decision between a repository release asset, another third-party host, or reproducible local acquisition is therefore not answered by the source-data policy.

## Unknowns requiring maintainer confirmation

1. **Source versioning:** Is a retrieval-pinned period-of-record NetCDF object acceptable as the Demonstration Sample input, or must the project first obtain immutable historical monthly objects/revision identifiers from NCEI? The public sources reviewed here do not establish the latter.
2. **Revision policy:** Which months are admitted, and how will the project classify preliminary versus final data, given NCEI's two-year reprocessing statement and NODD's approximate one-year replacement statement?
3. **Manifest contract:** Which project-computed cryptographic algorithm and exact retrieval fields are required in the Demonstration Sample Source Manifest? NCEI/NODD exposes only the observed object-level CRC32, not a per-month immutable checksum manifest.
4. **Distribution:** Will `--demo` acquire official data and reduce it reproducibly, download a project-hosted derived asset, or support both? If a derived asset is hosted, which host's terms, retention policy, and size limits are acceptable?
5. **Attribution presentation:** Where will the required source citation, subset statement, access date, modification notice, and no-endorsement notice be shown—in the asset, manifest, documentation, or all three?

## No new capability gate

This research adds no ticket or dependency edge. The existing blocked decision, [Resolve the Demonstration Sample version and distribution contract](https://github.com/monocongo/climate_indices/issues/726), already owns the newly actionable maintainer choices: exact period and crop, reduction recipe, manifest fields, asset versioning/caching, and acquisition-versus-project-hosted distribution.

## Sources

[metadata]: https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00332;view=xml;responseType=text/xml
[archive]: https://www.ncei.noaa.gov/data/nclimgrid-monthly/access/
[realtime]: https://www.ncei.noaa.gov/pub/data/cirs/climgrid/
[readme]: https://www.ncei.noaa.gov/data/nclimgrid-monthly/doc/gridded-readme.txt
[s3-explorer]: https://noaa-nclimgrid-monthly-pds.s3.amazonaws.com/index.html
[s3-listing]: https://noaa-nclimgrid-monthly-pds.s3.amazonaws.com/?list-type=2
[s3-prcp]: https://noaa-nclimgrid-monthly-pds.s3.amazonaws.com/nclimgrid_prcp.nc
[nodd-entry]: https://registry.opendata.aws/noaa-nclimgrid/
