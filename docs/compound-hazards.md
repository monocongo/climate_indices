# Analyze declared compound hazards

Use this guide to count joint occurrences, inspect forward-window associations, and retain the source choices.
The NumPy API is stable. The xarray Dataset routes and eager event catalogs are beta.
See {doc}`reference` for every parameter and result field.

:::{warning}
These results describe relationships between index-defined hazards, not impacts, causality, return periods, or operational predictions.
The public-study comparison below reproduces a same-author calculation on three cells.
It does not independently validate the method or reproduce the full published map.
:::

## Declare probability inputs and physical indicators

Supply fitted cumulative probabilities directly when an upstream function supports `output_scale="probability"`.
Do not convert clipped normal scores back into probabilities.
The probability declaration checks the range, not the distribution fit.

For physical thresholds, construct Boolean indicators and keep a separate validity mask.
A comparison with NaN produces `False`, which alone would incorrectly count an unknown observation as a non-occurrence.
Record the original physical threshold and comparator in `source_metadata`.
The indicator declaration itself always uses threshold `0.5` and direction `"above"`.

This synthetic monthly example combines probability values with a physical heat threshold.
Its values illustrate the interface; they are not observations or a validated heatwave definition.

```{testcode}
import json
from pathlib import Path

import numpy as np
import xarray as xr

import climate_indices.compound as compound

rng = np.random.default_rng(1374)
months = np.arange("1800-01", "2000-01", dtype="datetime64[M]")
drought_probability = rng.uniform(size=len(months))
temperature = rng.normal(30.0, 5.0, size=len(months))
heat = temperature >= 30.0
hazards = compound.harmonize(
    {"drought": drought_probability, "heat": heat},
    time={"drought": months, "heat": months},
    cadence={"drought": "monthly", "heat": "monthly"},
    input_scale={"drought": "probability", "heat": "indicator"},
    thresholds={"drought": 0.1, "heat": 0.5},
    directions={"drought": "below", "heat": "above"},
    validity={"heat": np.isfinite(temperature)},
    target_cadence="monthly",
    source_metadata={
        "drought": {
            "calibration_year_initial": 1800,
            "calibration_year_final": 1999,
            "scale": 1,
            "source": "synthetic uniform probabilities, seed 1374",
        },
        "heat": {
            "physical_threshold": 30.0,
            "comparator": ">=",
            "units": "degree_Celsius",
            "source": "synthetic monthly temperatures, seed 1374",
        },
    },
)
joint = compound.joint_statistics(hazards)
assert joint.n_valid.shape == (12,)
assert np.all(joint.n_valid == 200)
assert np.all(joint.n_joint <= joint.n_marginal.min(axis=0))
choices = json.loads(json.dumps(joint.metadata))
assert choices["harmonization"]["sources"]["heat"]["comparator"] == ">="
```

`n_valid` counts dates where every selected source is observed.
Every marginal uses that same support.
`joint_probability` is `n_joint / n_valid`.
The likelihood multiplication factor (LMF) divides this frequency by the product of the observed marginal frequencies.
A zero marginal makes LMF undefined (`NaN`); positive marginals with no joint occurrence give zero.
`LMF = 1` does not establish independence.

Inspect `quality_flags` and `metadata["quality_flags"]` before interpreting estimates.
The default `min_samples=30` is a product guardrail, not evidence that a study has enough observations.
Use `groupby="pooled"` only when a mixture across seasons answers the intended question.
A pooled LMF is not an average of monthly LMFs.

## Reduce daily indicators to calendar months

Threshold daily inputs before reduction.
Choose `"any"` for at least one hazard day, or `"all"` for every day meeting the condition.
A month is invalid if any civil day is unknown, including a real February 29.
Monthly inputs are never expanded to days by the library.

```{testcode}
days = np.arange("2000-01-01", "2000-03-01", dtype="datetime64[D]")
daily_heat = np.zeros(len(days), dtype=bool)
daily_heat[[0, -1]] = True
reduced = compound.harmonize(
    {"heat": daily_heat, "drought": np.array([True, False])},
    time={"heat": days, "drought": np.array(["2000-01", "2000-02"], dtype="datetime64[M]")},
    cadence={"heat": "daily", "drought": "monthly"},
    input_scale={"heat": "indicator", "drought": "indicator"},
    thresholds={"heat": 0.5, "drought": 0.5},
    directions={"heat": "above", "drought": "above"},
    target_cadence="monthly",
    aggregation={"heat": "any"},
)
np.testing.assert_array_equal(reduced.exceedance, [[True, True], [True, False]])
assert reduced.valid.all()
```

Use matching date coverage when comparing sources.
The default `join="exact"` rejects mismatched coverage.
An explicit `join="inner"` warns about removed boundaries.
Interior missing dates retain their positions and become invalid.

## Count forward-window responses

Use inclusive cadence-step offsets.
For monthly inputs, `(1, 3)` asks whether the target occurs in any of the next three calendar months.
Each eligible positive anchor contributes at most one success, even when several target occurrences follow it.
An incomplete horizon or an unknown target step excludes the anchor, even if another step contains a response.

```{testcode}
conditional = compound.conditional_probability(
    hazards, given="drought", target="heat", lag_window=(1, 3),
)
assert np.all(conditional.n_success <= conditional.n_positive)
assert np.all(conditional.n_positive <= conditional.n_anchors)
assert conditional.n_anchors.sum() == len(months) - 3
```

Compare `conditional_probability` with `response_window_frequency`, which uses the same eligible anchors.
This is a time-step association, not an event-onset conversion probability.
For concurrent or sequential run pairs, use `compound_events()` or the {ref}`eager numeric catalog <compound-event-catalog>`.
Sequential lag is `B.start - A.end`; adjacent runs have lag one.
Unknown bridges cannot establish a sequential match.
Pair counts are not independent observations and must not enter a binomial significance test.

## Run a seeded monthly bootstrap example

:::{warning}
The bootstrap has synthetic screening evidence only inside a measured scope, condition
by condition rather than as one configuration. At the default 1,999 resamples: 200-year
records with the seasonal monthly probabilities (0.35 to 0.55), source overlap of 1, 6,
or 36 months, and independent hazards. At 399 resamples: 200-year records with 1 to 30
months of overlap, conditional lag windows of `(0, 0)`, `(0, 1)`, `(0, 3)`, and `(1, 3)`
at 1- and 6-month overlap, monthly probabilities of 0.30 to 0.60 (1- and 6-month
overlap) or 0.70 to 0.80 (1-, 6-, and 36-month overlap), hazards that correlate 0.5 at
the same time with 1, 6, or 12 months of overlap, and 100-year records with the seasonal
probabilities. The plan crossed no two of those conditions. 36-month persistence at 399
resamples with independent hazards, correlated 36-month persistence at the default
resamples, monthly probabilities near 0.20, and 60-year and 30-year records each failed
a predeclared gate. No independent-hazard case exceeded its per-month rejection band;
the failures are interval under-coverage and suppressed p-values where few effective
blocks or occurrences remain.
See the [evidence record](https://github.com/monocongo/climate_indices/blob/main/VALIDATION.md#compound-bootstrap-characterization).
:::

Use a separate balanced-frequency synthetic example within those measured conditions.
It demonstrates execution, not new calibration evidence.

```{testcode}
independent = rng.uniform(size=(2, len(months))) < 0.5
monthly = compound.harmonize(
    {"A": independent[0], "B": independent[1]},
    time={"A": months, "B": months},
    cadence={"A": "monthly", "B": "monthly"},
    input_scale={"A": "indicator", "B": "indicator"},
    thresholds={"A": 0.5, "B": 0.5},
    directions={"A": "above", "B": "above"},
    target_cadence="monthly",
)
inference = compound.bootstrap_dependence(
    monthly, statistic="likelihood_multiplication_factor",
    seed=1374,
)
assert inference.n_years == 200
assert np.isfinite(inference.estimate).all()
assert np.all(inference.n_null_resamples == 1999)
assert np.all(inference.quality_flags == 0)
```

Inspect `inference_start`, `inference_end`, and `full_record_estimate` beside the inference-support `estimate`.
Missing months split complete-year segments; inference uses the longest segment and selects the earliest tie.
Check block-length sensitivity as described in {ref}`the API reference <compound-bootstrap-inference>`.
Do not choose a block length by its smallest p-value.
Local p-values are unadjusted and do not establish field significance.
Issue [#1398](https://github.com/monocongo/climate_indices/issues/1398) owns further bootstrap calibration.

## Reproduce the public drought–heatwave reference

Run the following example from the repository root after installation.
It uses committed files without downloads or optional external packages.
The fixture attributes the public inputs to Vieira Passos (2024), Figshare version 2, CC BY 4.0.
It pins the same-author notebook at commit `6c88f255ee3034279dcebe2a2fc40c954dbe9575`.
See the [fixture README](https://github.com/monocongo/climate_indices/blob/main/tests/fixture/compound/README.md) for coordinates, checksums, and notices.

:::{important}
The author's preparation expands month-end SPEI to daily dates, then interpolates gaps left by source NaNs.
This is an explicit reproduction convention outside `harmonize()`, not recommended harmonization or independently observed daily drought resolution.
The library receives the resulting Boolean indicators, not interpolated probability inputs.
The original Calibration Period is unknown.
:::

```{testcode}
fixture = Path("tests/fixture/compound")
reference = json.loads((fixture / "reference.json").read_text())
with np.load(fixture / "inputs.npz", allow_pickle=False) as raw:
    spei = xr.DataArray(
        raw["monthly_spei"], dims=("time", "cell"),
        coords={"time": raw["monthly_time"]},
    ).resample(time="D").ffill().interpolate_na(dim="time", method="linear")
    dates = spei.time.values.astype("datetime64[D]")
    hwi = raw["daily_hwi"][np.searchsorted(raw["daily_time"], dates)]
    drought = spei.values
published = compound.harmonize(
    {"heat": hwi >= 0, "drought": drought <= -1},
    time={"heat": dates, "drought": dates},
    cadence={"heat": "daily", "drought": "daily"},
    input_scale={"heat": "indicator", "drought": "indicator"},
    thresholds={"heat": 0.5, "drought": 0.5},
    directions={"heat": "above", "drought": "above"},
    validity={"heat": np.isfinite(hwi), "drought": np.isfinite(drought)},
    target_cadence="daily",
    spatial_time_major=True,
    source_metadata={
        "heat": {
            "physical_threshold": 0, "comparator": ">=",
            "dataset_doi": reference["dataset_doi"],
            "source": reference["sources"]["heatwave_SMHI.nc"],
        },
        "drought": {
            "physical_threshold": -1, "comparator": "<=", "scale": 12,
            "dataset_doi": reference["dataset_doi"],
            "source": reference["sources"]["SPEI12.nc"],
            "preparation": reference["preparation"],
            "author_commit": reference["author_commit"],
        },
    },
)
reproduced = compound.joint_statistics(published)
for case in reference["cases"]:
    month, cell = case["month"] - 1, case["cell"]
    assert reproduced.n_valid[month, cell] == case["n_days"]
    assert reproduced.n_joint[month, cell] == case["n_joint"]
    np.testing.assert_array_equal(
        reproduced.n_marginal[:, month, cell],
        [case["n_heat"], case["n_drought"]],
    )
    np.testing.assert_allclose(
        reproduced.likelihood_multiplication_factor[month, cell],
        case["lmf"], rtol=0, atol=1e-12,
    )
```

The comparison covers June, July, and August separately at southern, central, and northern Swedish cells, over 1922–2021.
June has 3,000 eligible days per cell; July and August each have 3,100.
Non-summer groups can emit sparse-occurrence and zero-marginal warnings; they are outside the reproduction target.
The [validation record](https://github.com/monocongo/climate_indices/blob/main/VALIDATION.md#compound-published-method-reproduction) reports measured errors and evidence limits.

The source thresholds are inclusive: `HWI >= 0` and `SPEI <= -1`.
Probability inputs use strict comparisons, so the reproduction deliberately declares Boolean indicators instead.
The author's zero-joint LMF is undefined; the library returns zero when both marginals are positive.
All nine reference cases have positive joint counts, so this intentional difference does not affect them.
Neither the author's IID significance procedure nor its random tie noise is reproduced.
Daily inference remains unsupported; this daily reproduction makes no bootstrap-significance claim.

## Use labeled or Dask-backed inputs

Pass a mapping of DataArrays to `harmonize()` to receive a prepared Dataset.
Do not supply the NumPy-only `time` or `spatial_time_major` arguments.
Use the DataArray time coordinates instead.
Statistics return a Dataset with numeric counts, flags, dimensionless units, and a JSON `compound_manifest` attribute.
They do not invent CF `standard_name` values or inherit source attributes as result definitions.

For Dask inputs, keep the complete `time` and `hazard` axes in single chunks.
Use identical spatial chunks for every input.
The library does not compute or rechunk inputs implicitly.
Select a bounded subset and call `.compute()` explicitly before building an eager event catalog.
See the {ref}`beta route reference <compound-xarray-route>` for memory measurements and serialization contracts.
