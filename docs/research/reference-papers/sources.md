# Palmer reference sources

These are the sources behind `docs/algorithm_refs/palmer-provenance.md`. They are recorded here
instead of being committed: none of them is redistributed with the repository. Copyright and
licensing questions belong to REFIMPL-27. To check a copy against this record, fetch it from the
URL and compare its SHA-256. "Rows" are rows of the provenance record, and D entries are its
"Differences found while tracing".

| Source | URL | Accessed | SHA-256 | Rows |
| --- | --- | --- | --- | --- |
| Palmer, W. C. (1965), *Meteorological Drought*, U.S. Weather Bureau Research Paper No. 45 (NCEI scan) | `https://www.ncei.noaa.gov/monitoring-content/temp-and-precip/drought/docs/palmer.pdf` | 2026-11-19 | `9b52259f8732de597f4c8c34982b196a03a5dc2668fdca0d0a5b22cf60d1e65f` | 1-20; D1, D3, D5, D8 |
| Alley, W. M. (1984), *J. Climate Appl. Meteor.* 23, 1100-1109 | `https://doi.org/10.1175/1520-0450(1984)023<1100:TPDSIL>2.0.CO;2` | 2026-11-19 (maintainer's copy) | `f366353c8e6d2dfe018d7c9e8e05cfb8facebdf74f0b9bb3d0e60f3bcf21b15e` | 1-9, 11, 12, 14, 16-20, 22; D1, D2, D3, D5 |
| Karl, T. R. (1986), *J. Climate Appl. Meteor.* 25, 77-86 | `https://doi.org/10.1175/1520-0450(1986)025<0077:TSOTPD>2.0.CO;2` | 2026-11-19 (maintainer's copy) | `d7f44584d865e3a0adfdc0eb0ceff0b7172874a66622a35e153435872b0c9939` | 1, 7-12, 18, 21; D8 |
| Wells, N., Goddard, S., and Hayes, M. J. (2004), *J. Climate* 17, 2335-2351 | `https://doi.org/10.1175/1520-0442(2004)017<2335:ASPDSI>2.0.CO;2` | 2026-11-19 (maintainer's copy) | `bbdd152d4fbdb129dd72aa0b9053d570696fd39c96619787e29d3548f1fb8389` | 10-17, 20, 23-26; D2, D3, D5-D8 |
| Heim, R. R., Jr. (2002), *Bull. Amer. Meteor. Soc.* 83, 1149-1165 | `https://doi.org/10.1175/1520-0477-83.8.1149` | 2026-11-19 (maintainer's copy) | `5de646f75bc01cdfa358e2af8b2572ac494073a69fc677576f4503522e4629cf` | 21, 22 |
| NCEI nClimDiv `divisional-readme.txt` (changes with each monthly release) | `https://www.ncei.noaa.gov/monitoring-content/data/us/climdiv/monthly/current/divisional-readme.txt` | 2026-11-19 | `5d3578d278a46b0cb20461e8bca5735623160e69e9d0764a7f12fa8caf38cd20` | 20-22 |
| NCEI `pdi.f` (2013-04-03, 899 lines; "as is", no licence stated) | `https://www.ncei.noaa.gov/pub/data/nidis/software/ncei/palmer/pdi.f` | 2026-11-19 | `e0c58ccbdc5fe694e9218e317065cf14cb6095de329e1dd43087fab67f20cb9e` | 1-14, 16-22; D1, D2, D9 |
| NCEI `soilcnst.asc` (the soil and Thornthwaite constants `pdi.f` reads) | `https://www.ncei.noaa.gov/pub/data/nidis/software/ncei/palmer/soilcnst.asc` | 2026-11-19 | `40fc3a55a0c91aa1f83ce753b45950b2dbebad7c2fb7d23c9e829416cae588c2` | 1, 2; D1, D9 |
| drought.gov page for `pdi.f` (availability and disclaimer) | `https://www.drought.gov/data-maps-tools/drought-indices-fortran-spi-pdsi` | 2026-11-19 | not recorded (HTML) | Sources table |
| Wells, Goddard and Hayes C++ v2.0, `scPDSI.zip` (holds `src/scpdsi.cpp`, 5,179 lines, and a Windows binary that was not run; no licence stated) | `https://greenleaf.unl.edu/downloads/scPDSI.zip` | 2026-11-19 | zip `52ceeb4287396e969556548bd8c2c12baa95bf14f80a92e366f6aeaef24e30ba`; `scpdsi.cpp` `2eae4a06ec07bccab5e5f81169a14d5d877d61bfc4437ed9bbcde0a9fa9e0f71` | 25, 26; D2, D5-D7, D9 |
| Its manual | `https://greenleaf.unl.edu/downloads/scPDSI_Manual.pdf` | 2026-11-19 | `2da8b858163d205d8d54c4ef5e30807cfad73477c9db3b7d400f5df38a8662c9` | D9 |
| `Sibada/scPDSI` 0.1.3 (GPL-3), a port of the Wells C++ | `https://github.com/Sibada/scPDSI` | 2026-11-19 | commit `10f8c3b` | 2-11, 13-19, 21-26; D1, D5-D7 |

The maintainer supplied the copies of the four journal papers. The SHA-256 identifies those
copies, which can differ from a copy downloaded from the publisher.
