# SEI energy references

These fixtures support [ENERGY-14](https://github.com/monocongo/climate_indices/issues/1170).
They compare statistical implementations on identical inputs.
They describe the climate-driven component of energy supply and demand, not actual generation or load.
They do not validate a wind-power model, a demand model, or beta fitting.

## Sources and rights

The reference implementation is **SEI 0.2.0**, at commit
`4d4a6d02d0fbfefdb48da63580c6be1bca3d5f14`.
The generator verifies the source archive checksum before installation into a temporary R library.
The system's installed SEI package cannot replace that source.
SEI uses GPL (>= 2); these fixtures contain numerical outputs, not its implementation.

The real input comes from Bloomfield and Brayshaw (2021),
*ERA5 derived time series of European aggregated surface weather variables, wind power, and solar power capacity factors: hourly data from 1950-2020*.
The University of Reading publishes it at
[DOI 10.17864/1947.000321](https://doi.org/10.17864/1947.000321).
Its landing page and archive README specify [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/), Copyright 2021 University of Reading.
The underlying meteorological source is Copernicus Climate Change Service ERA5.
Attribution identifies the creators, DOI, publisher, licence, and the monthly subset derived here.

The selected archive member is
`ERA5_data_1950-2020/wp_onshore/NUTS_0_wp_ons_sim_1_historical_loc_weighted.nc`.
The source README interchanges “onshore” and “offshore” in some file descriptions.
This fixture identifies the exact member and simulation, rather than inferring a different physical model.
The selected country is `NUTS_keys == "DE"`.
The generator converts its hourly values to float64, then calculates complete monthly means without missing-value removal.
The 852 monthly means cover January 1950 through December 2020.
The calibration period is 1981–2010.
This is an attributed monthly subset; it does not redistribute turbine-location records or claim measured generation.

Reading provides a public, rights-verified real energy series without a CDS account.
This substitutes for the survey's suggested C3S product subset, not for any statistical acceptance criterion.
The generator verifies the original NetCDF checksum and every hourly time coordinate.
`metadata.json` records the archive URL, ETag, member checksum, README checksum, selection, conversion, units, and calendar.

Maintainer confirmation of extraction, generation, and redistribution remains required before acceptance.

## Statistical contract

Synthetic monthly inputs cover 1900–2003, with a separate 1902–2001 calibration period.
They include bounded capacity factors, production, demand, and signed `demand - production` in compatible MWh units.
They also include ties, constant groups, all-zero/all-one groups, missing values, and held-out values outside the calibration range.
Four calendar groups contain exactly 9, 10, 99, and 100 calibration observations at Timescale 1.
The synthetic seed is 1170.

The generator applies each complete rolling window before calendar-month grouping.
A window includes the current observation; an incomplete or missing window produces NaN.
The 12 cases cover Timescales 1 and 3, means for capacity factors, sums for production/demand, and both aggregations for residual load.
SEI receives separate `x_ref` and `x_new` vectors for each calendar month.
Its grouped recursion and missing-tolerant rolling helper are not used.

The exact empirical calls are in `scripts/energy_sei_reference.R`:

```r
SEI::std_index(
  x_new = scaled[ind], x_ref = reference, dist = "empirical", method = "mle",
  index_type = "prob01", lower = -Inf, upper = Inf, cens = "none", n_thres = 10
)
SEI::std_index(
  x_new = scaled[ind], x_ref = reference, dist = "empirical", method = "mle",
  index_type = "normal", lower = -Inf, upper = Inf, cens = "none", n_thres = 10
)
```

SEI uses `(1 + count(x_ref <= x)) / (n + 2)` and maximum ranks for ties.
Ten observations permit a fit; fewer than 100 produce a recommendation warning, not an automatic distribution change.
`warnings.txt` preserves these warnings.
Constant groups therefore do not map to zero normal scores.

The parametric control uses gamma maximum likelihood estimation (MLE) on strictly positive synthetic production.
Its saved shape/rate parameters permit comparison without conflating MLE with the library's different gamma estimator.
It is not a gamma-default recommendation for energy wrappers and does not validate beta.
[ENERGY-13](https://github.com/monocongo/climate_indices/issues/1169) owns beta validation.

A separate finite-bound control records endpoint behavior with and without endpoint observations in calibration.
It calls `SEI::get_pit()` directly because this pinned `std_index()` rejects a numeric censoring pair.
When calibration contains no endpoint observations, SEI leaves endpoint evaluations missing.
This records reference behavior; it does not establish an energy endpoint policy.

## Files and checks

- `inputs.npz` contains the five input series.
- `reference.npz` contains separate scaled inputs, probabilities, normal indices, calibration counts, and control outputs.
- `metadata.json` records cases and conventions, source pins, and Python package versions.
- `sessionInfo.txt` records R 4.6.1 and its loaded dependencies, including SEI 0.2.0 and fitdistrplus 1.2-6.
- `provenance.json` follows `tests/fixture/provenance_schema.json` and records measured statistical agreement.

The checksum covers `inputs.npz`, `metadata.json`, `reference.npz`, `sessionInfo.txt`, and `warnings.txt`, in that order.
The tests also verify the R script checksum.
R writes 17 significant digits; its default CSV conversion would retain only 15.
Missing masks and calibration counts must match exactly.
The absolute tolerances are `1e-12` for scaled values/normal indices and `1e-15` for empirical probabilities, with `rtol=0`.
The measured maximum errors are `6.67e-16` for normal indices and `1.12e-16` for probabilities, rounded upwards.
The generator compares scaled values, probabilities, and normal indices against SEI and refuses comparisons outside these tolerances.
It then stages the complete set and swaps it in as one directory, restoring the previous set if the swap fails.
A failed write therefore cannot leave truncated payloads beside the old provenance.
The library supplies measurements only; it never supplies reference values.

The offline tests exercise the shipped standardization function now.
Only absent energy wrappers xfail, with specific references to
[SREPI and standardized demand](https://github.com/monocongo/climate_indices/issues/1171)
and [SRLI](https://github.com/monocongo/climate_indices/issues/1172).
A present but incorrect implementation fails normally.
[ENERGY-24](https://github.com/monocongo/climate_indices/issues/1180) owns the family-wide validation tables and published API documentation.

## Reproduction

1. Download the pinned SEI source archive.

   ```bash
   curl -L --fail -o /path/to/SEI-source.tar.gz \
     https://codeload.github.com/noeliaof/SEI/tar.gz/4d4a6d02d0fbfefdb48da63580c6be1bca3d5f14
   ```

2. Download the Reading archive from its recorded URL.

   The archive is approximately 1.83 GB.
   Download it only when sufficient storage is available.

3. Extract the selected NetCDF member.
4. Install the R dependency versions recorded in `sessionInfo.txt` into a local R library.
5. Run the generator with the two local source paths.

   ```bash
   uv run scripts/prepare_energy_sei_fixtures.py /path/to/SEI-source.tar.gz \
     /path/to/NUTS_0_wp_ons_sim_1_historical_loc_weighted.nc
   ```

6. Run the offline checks, including the validation marker.

   ```bash
   uv run pytest tests/test_energy_sei_reference.py tests/test_provenance_protocol.py \
     -m 'validation or not validation'
   ```

The generator does not download sources or install R dependencies.
It checks SEI and fitdistrplus versions before calculation.
It wrote the committed payloads with NumPy 2.4.2; it avoids an `allow_pickle` keyword that NumPy before 2.2 would store as an extra archive member.
CI loads committed outputs without R or network access.
