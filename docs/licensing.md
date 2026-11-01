# Licensing

## Primary license and history

The project's primary license is
[BSD-3-Clause](https://github.com/monocongo/climate_indices/blob/main/LICENSE),
with the existing copyright notice: Copyright 2017, James Adams.
The original code came from NCEI/NOAA through
[drought.gov](https://www.drought.gov/data-maps-tools/climate-and-drought-indices-python-spi-spei-pet).
The original public-domain notice appears below without changes to its wording.
This document organization does not change the project's licensing terms.

## Third-party exceptions and distribution

The Rust routines include code derived from Cephes, SciPy/xsf, NumPy, and Boost.Math.
Their terms remain separate from the project's primary license.
In particular, the Lanczos coefficients retain the Boost Software License 1.0.
The [third-party notices](https://github.com/monocongo/climate_indices/blob/main/THIRD_PARTY_NOTICES.md)
identify the components and contain their complete license texts and provenance URLs.
Existing source-file legal notices remain in place.

Python wheels and source distributions include `LICENSE`, `THIRD_PARTY_NOTICES.md`,
and this document.
The standalone Rust crate retains its
[combined LICENSE](https://github.com/monocongo/climate_indices/blob/main/crates/climate-indices/LICENSE),
which includes the primary license, the original NOAA notice, and all these third-party notices.
This keeps the crate's legal documents complete without requiring the repository or Python package.

## Original NOAA public-domain notice

```text
Previous Public Domain License (from code originally downloaded from drought.gov):

NOAA's National Centers for Environmental Information
NOAA/NESDIS/NCEI
151 Patton Ave, Asheville, NC 28801

THIS SOFTWARE AND ITS DOCUMENTATION ARE CONSIDERED TO BE IN THE PUBLIC DOMAIN AND THUS ARE AVAILABLE
FOR UNRESTRICTED PUBLIC USE.

THEY ARE FURNISHED "AS IS." THE AUTHORS, THE UNITED STATES GOVERNMENT, ITS INSTRUMENTALITIES, OFFICERS, EMPLOYEES,
AND AGENTS MAKE NO WARRANTY, EXPRESS OR IMPLIED, AS TO THE USEFULNESS OF THE SOFTWARE AND DOCUMENTATION FOR ANY PURPOSE.
THEY ASSUME NO RESPONSIBILITY (1) FOR THE USE OF THE SOFTWARE AND DOCUMENTATION; OR (2) TO PROVIDE TECHNICAL SUPPORT
TO USERS.
```
