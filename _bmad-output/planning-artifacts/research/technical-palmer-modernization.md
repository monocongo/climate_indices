---
stepsCompleted: [2, 3, 4, 5, 6]
inputDocuments: []
workflowType: 'research'
lastStep: 6
status: 'complete'
research_type: 'Technical Research'
research_topic: 'Palmer Modernization Patterns — xarray Multi-Output Adapter Design'
research_goals: |
  1. Understand xarray's apply_ufunc capabilities for multi-output stateful functions
  2. Design a multi-output adapter pattern that extends existing @xarray_adapter
  3. Survey how other climate libraries handle Palmer's unique computation signature
user_name: 'James Addison'
date: '2026-02-15'
web_research_enabled: true
source_verification: true
---

# Research Report: Palmer Modernization Patterns — xarray Multi-Output Adapter Design

**Date:** 2026-02-15
**Author:** James Addison
**Research Type:** Technical Research

---

## Research Overview

### Context

The `climate_indices` project successfully modernized SPI/SPEI computations to support xarray DataArrays via a decorator-based adapter layer (`@xarray_adapter`). Palmer drought indices (`palmer.py:pdsi()`) present three unique challenges that block direct application of the same pattern:

1. **Stateful sequential computation** — Water balance tracking and Z-Index/PDSI phases carry month-to-month state (soil moisture, drought probabilities, backtrack buffers). This violates the "embarrassingly parallel" assumption of standard xarray parallelization.

2. **Multi-return signature** — The `pdsi()` function returns a 5-tuple:
   ```python
   (pdsi, phdi, pmdi, z_index, params_dict) -> tuple[
       np.ndarray,  # PDSI values
       np.ndarray,  # PHDI values
       np.ndarray,  # PMDI values
       np.ndarray,  # Z-Index values
       dict[str, Any] | None  # Calibration parameters (alpha, beta, gamma, delta)
   ]
   ```
   The existing `@xarray_adapter` handles only single-output functions.

3. **Non-array return value** — The 5th return (`params_dict`) is a dictionary of calibration coefficients, incompatible with `xr.apply_ufunc` which expects array-like outputs.

### Research Goals

This research will:
1. **Understand xarray's `apply_ufunc` capabilities** for multi-output stateful functions
2. **Design a multi-output adapter pattern** that extends or complements the existing `@xarray_adapter`
3. **Survey other climate libraries** to learn from established Palmer implementation patterns

### Methodology

- **Web research** on xarray API capabilities and best practices (2025-2026)
- **Code pattern analysis** of existing `xarray_adapter.py` decorator
- **Library survey** of xclim, pyet, standard_precip, and climate_indices forks
- **Documentation synthesis** into actionable implementation guidance

---

## 1. xarray apply_ufunc Multi-Output Capabilities

### 1.1 Core API for Multiple Outputs

The [`xr.apply_ufunc()` API](https://docs.xarray.dev/en/stable/generated/xarray.apply_ufunc.html) **does support multi-output functions** through the `output_core_dims` parameter.

**Key mechanism:**
> "If a function returns multiple outputs, you must set `output_core_dims` as well. `output_core_dims` is a list of the same length as the number of output arguments from `func`, giving the list of core dimensions on each output that were not broadcast on the inputs."

**Signature:**
```python
xr.apply_ufunc(
    func,
    *args,
    input_core_dims=None,
    output_core_dims=((),),  # Default: single output, no core dims
    vectorize=False,
    dask='forbidden',
    output_dtypes=None,
    ...
)
```

**For multi-output:** Provide `output_core_dims` as a list with one entry per returned array. Each entry describes which dimensions are "core" (operated on) vs "broadcast" (looped over).

### 1.2 Practical Example: minmax Function

From the [xarray tutorial on complex outputs](https://tutorial.xarray.dev/advanced/apply_ufunc/complex-output-numpy.html):

```python
def minmax(array):
    """Return min and max along last axis."""
    return array.min(axis=-1), array.max(axis=-1)

# Apply to 2D array (time, lat)
minda, maxda = xr.apply_ufunc(
    minmax,
    air2d,
    input_core_dims=[["lat"]],      # Operate on lat dimension
    output_core_dims=[[], []],      # Two outputs, neither has core dims (scalars)
)
```

**What happens:**
- Function receives 1D slices of `air2d` along the `lat` dimension
- Returns two scalars (min, max) for each slice
- xarray broadcasts this across all non-core dimensions (`time`)
- Result: two DataArrays `minda` and `maxda`, each with shape `(time,)`

### 1.3 Dask Parallelization with Multiple Outputs — CRITICAL LIMITATION

**Problem:** The `dask='parallelized'` mode **does not support multi-output functions**.

From [xarray Issue #1815](https://github.com/pydata/xarray/issues/1815) (opened Jan 2018, marked "needs implementation" as of May 2020):

> **Status:** Using `apply_ufunc` with `dask='parallelized'` and multiple outputs raises `NotImplementedError`.
> **Proposed solution:** Use `dask.array.apply_gufunc()` instead of `atop()` when `signature.num_outputs > 1`.
> **Current status:** Still not implemented. **Workarounds required.**

**Workaround pattern** (from community discussion):
1. Wrap multiple outputs into a **single array** with an extra dimension (e.g., `output_dim="variable"`)
2. Return this single array from the wrapped function
3. Unpack after computation using `xr.Dataset` variable assignment

**Example:**
```python
def palmer_wrapped(precip, pet, awc):
    """Wrap 4 outputs into single array with 'variable' dimension."""
    pdsi, phdi, pmdi, z = palmer_core(precip, pet, awc)
    # Stack along new axis 0
    return np.stack([pdsi, phdi, pmdi, z], axis=0)

result = xr.apply_ufunc(
    palmer_wrapped,
    precip_da,
    pet_da,
    awc,
    input_core_dims=[["time"], ["time"], []],
    output_core_dims=[["variable", "time"]],
    dask="parallelized",
    output_dtypes=[float],
)

# Unpack into Dataset
ds = xr.Dataset({
    "pdsi": result.isel(variable=0),
    "phdi": result.isel(variable=1),
    "pmdi": result.isel(variable=2),
    "z_index": result.isel(variable=3),
})
```

### 1.4 Sequential State — Cannot Parallelize Time Dimension

Palmer computation **cannot use `dask="parallelized"` along the time dimension** regardless of multi-output handling.

**Why:** From [xarray Dask documentation](https://docs.xarray.dev/en/stable/user-guide/dask.html):
> "`dask='parallelized'` works well for 'blockwise' or 'embarrassingly parallel' operations where one block of the output array corresponds to one block of the input array. **No communication between blocks is necessary.**"

**Palmer's water balance** requires communication:
- Month `t` depends on soil moisture state from month `t-1`
- Cannot compute January without December's state
- Time is a **dependent dimension**, not parallelizable

**Implication:** Even if xarray supported multi-output `dask='parallelized'`, Palmer would need `dask='allowed'` or `dask='forbidden'` for the time dimension. Parallelization can only occur across **spatial dimensions** (lat, lon, grid cells).

### 1.5 Vectorize Mode — Python Loop Trade-off

The `vectorize=True` parameter offers an alternative:

```python
result = xr.apply_ufunc(
    pdsi,  # Native numpy function
    precip_da,
    pet_da,
    awc,
    input_core_dims=[["time"], ["time"], []],
    output_core_dims=[["time"], ["time"], ["time"], ["time"], []],  # 5 outputs
    vectorize=True,
    output_dtypes=[float, float, float, float, object],  # Last is dict
)
```

**How it works:**
- Uses `numpy.vectorize()` to automatically loop over non-core dimensions
- Calls the raw function for each spatial location (grid cell)
- **Handles multi-output natively** — no wrapping required

**Trade-offs:**
- ✅ **Simple:** No function modification, decorator just works
- ✅ **Multi-output:** Naturally unpacks tuples
- ❌ **Slow:** Python-level loop, not NumPy/Dask parallel
- ❌ **No Dask:** Cannot combine with `dask='parallelized'`

**Performance note:** The existing `climate_indices` CLI *already* uses multiprocessing to parallelize across grid cells (`__main__.py:1366-1430`). The Python loop overhead of `vectorize=True` matches this existing pattern, so it's **not a regression** from current performance.

### 1.6 Dataset Construction from Multi-Output

**Pattern A: Tuple unpacking (manual)**
```python
pdsi_da, phdi_da, pmdi_da, z_da, params = xr.apply_ufunc(...)

ds = xr.Dataset({
    "pdsi": pdsi_da,
    "phdi": phdi_da,
    "pmdi": pmdi_da,
    "z_index": z_da,
})
# params handled separately (attrs or return alongside)
```

**Pattern B: Direct Dataset return**
```python
# Inside wrapped function
def palmer_to_dataset(...):
    pdsi, phdi, pmdi, z, params = palmer.pdsi(...)
    return xr.Dataset({
        "pdsi": (["time"], pdsi),
        "phdi": (["time"], phdi),
        "pmdi": (["time"], pmdi),
        "z_index": (["time"], z),
    }, attrs=params)
```

**Recommendation:** Pattern A (tuple unpacking) aligns with `apply_ufunc` semantics. Pattern B requires a different wrapper approach outside `apply_ufunc`.

---

## 2. Multi-Output Adapter Design Patterns

### 2.1 Current @xarray_adapter Architecture

The existing decorator (`xarray_adapter.py:1286-1595`) implements this architecture:

**Core apply_ufunc call** (lines 1471-1479 for Dask, 1568-1575 for in-memory):
```python
result_da: xr.DataArray = xr.apply_ufunc(
    _numpy_func_wrapper,
    *input_dataarrays,
    input_core_dims=[[time_dim]] * len(input_dataarrays),
    output_core_dims=[[time_dim]],  # Single output only
    dask="parallelized",  # or omitted for in-memory
    vectorize=True,
    output_dtypes=[float],
)
```

**Decorator contract:** Detect → Resolve → Align → Validate → Extract → Infer → Compute → Rewrap → Log

**Key characteristics:**
- ✅ **Single output:** Returns `xr.DataArray` with CF metadata
- ✅ **Parameter inference:** Automatically infers `data_start_year`, calibration years from time coordinate
- ✅ **Multi-input alignment:** Handles secondary inputs via `additional_input_names` (e.g., PET for SPEI)
- ✅ **Dask support:** `dask="parallelized"` for chunked arrays
- ✅ **Metadata provenance:** CF attributes + history tracking
- ❌ **Multi-output:** Hardcoded `output_core_dims=[[time_dim]]` for single output

### 2.2 Pattern A: Extend @xarray_adapter (Unified Decorator)

**Approach:** Add optional parameters to existing `@xarray_adapter` to support multi-output returns.

**Signature changes:**
```python
@xarray_adapter(
    cf_metadata=PALMER_CF_METADATA,  # Now: dict[str, CFAttributes]
    output_names=["pdsi", "phdi", "pmdi", "z_index"],  # NEW
    return_type="dataset",  # NEW: "dataarray" (default) | "dataset" | "tuple"
    exclude_from_xarray=["params_dict"],  # NEW: non-array returns
    ...
)
def pdsi(...) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict | None]:
    ...
```

**Implementation changes:**
```python
# Inside decorator
if return_type == "dataset":
    output_core_dims = [[time_dim]] * len(output_names)

    result_tuple = xr.apply_ufunc(
        _numpy_func_wrapper,
        *input_dataarrays,
        input_core_dims=[[time_dim]] * len(input_dataarrays),
        output_core_dims=output_core_dims,
        vectorize=True,
        # NOTE: Cannot use dask="parallelized" with multi-output
        output_dtypes=[float] * len(output_names),
    )

    # Construct Dataset from tuple
    ds = xr.Dataset({
        name: da.assign_attrs(cf_metadata[name])
        for name, da in zip(output_names, result_tuple)
    })

    return ds
```

**Trade-offs:**
- ✅ **Unified interface:** Single decorator for all indices
- ✅ **Backward compatible:** Existing single-output functions unaffected
- ✅ **Shared infrastructure:** Reuses alignment, inference, validation logic
- ⚠️ **Complexity:** Adds 3 new parameters + branching logic to already-large decorator (~400 lines)
- ❌ **Dask limitation:** Cannot use `dask="parallelized"` for multi-output (would need workaround)
- ❌ **params_dict handling:** Requires special-case logic to exclude from `apply_ufunc`, attach afterward

### 2.3 Pattern B: New @xarray_multi_adapter (Standalone Decorator)

**Approach:** Create a separate decorator specifically for multi-output functions.

**Usage:**
```python
@xarray_multi_adapter(
    output_specs={
        "pdsi": CF_METADATA["pdsi"],
        "phdi": CF_METADATA["phdi"],
        "pmdi": CF_METADATA["pmdi"],
        "z_index": CF_METADATA["z_index"],
    },
    auxiliary_outputs=["params_dict"],  # Non-array returns
    time_dim="time",
    infer_params=True,
    ...
)
def pdsi(...) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict | None]:
    ...
```

**Implementation approach:**
```python
def xarray_multi_adapter(
    output_specs: dict[str, CFAttributes],
    auxiliary_outputs: list[str] | None = None,
    ...
):
    """Decorator for functions returning multiple arrays + optional non-array outputs."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            input_type = detect_input_type(args[0])

            if input_type == InputType.NUMPY:
                return func(*args, **kwargs)  # Passthrough

            # Xarray path
            num_array_outputs = len(output_specs)
            num_aux_outputs = len(auxiliary_outputs) if auxiliary_outputs else 0

            result_tuple = xr.apply_ufunc(
                func,
                *input_dataarrays,
                input_core_dims=[[time_dim]] * len(input_dataarrays),
                output_core_dims=[[time_dim]] * num_array_outputs,
                vectorize=True,
                output_dtypes=[float] * num_array_outputs,
            )

            # Split array outputs from auxiliary outputs
            array_outputs = result_tuple[:num_array_outputs]
            aux_outputs = result_tuple[num_array_outputs:] if num_aux_outputs else ()

            # Build Dataset
            ds = xr.Dataset({
                name: da.assign_attrs(output_specs[name])
                for name, da in zip(output_specs.keys(), array_outputs)
            })

            # Attach auxiliary outputs to Dataset attrs
            if aux_outputs:
                for aux_name, aux_value in zip(auxiliary_outputs, aux_outputs):
                    if aux_value is not None:
                        ds.attrs[aux_name] = aux_value

            return ds

        return wrapper
    return decorator
```

**Trade-offs:**
- ✅ **Separation of concerns:** Single-output and multi-output logic isolated
- ✅ **Clearer intent:** API explicitly designed for multi-output use case
- ✅ **Simpler maintenance:** No conditional branching in either decorator
- ⚠️ **Code duplication:** Must replicate alignment, inference, validation logic
- ⚠️ **Two decorators:** Users must know when to use which
- ❌ **Dask limitation:** Same issue—`dask="parallelized"` incompatible with multi-output

### 2.4 Pattern C: Manual palmer_xarray() Wrapper (No Decorator)

**Approach:** Write a custom `palmer_xarray()` function without using a decorator.

**Implementation:**
```python
def palmer_xarray(
    precip_da: xr.DataArray,
    pet_da: xr.DataArray,
    awc: float | xr.DataArray,
    *,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    time_dim: str = "time",
) -> xr.Dataset:
    """Compute Palmer indices with xarray DataArray support.

    Returns:
        Dataset with variables: pdsi, phdi, pmdi, z_index
        Calibration params stored in Dataset.attrs["palmer_params"]
    """
    # Align inputs
    precip_aligned, pet_aligned = xr.align(precip_da, pet_da, join="inner")

    # Infer parameters from time coordinate
    time_coord = precip_aligned[time_dim]
    data_start_year = time_coord.dt.year.values[0]
    if calibration_year_initial is None:
        calibration_year_initial = data_start_year
    if calibration_year_final is None:
        calibration_year_final = time_coord.dt.year.values[-1]

    # Handle AWC (spatial parameter)
    # If scalar: broadcast to all grid cells
    # If DataArray: must have compatible dims
    if isinstance(awc, xr.DataArray):
        awc_values = awc.values
        # Input core dims: awc has no time dimension
        awc_core_dims = []
    else:
        awc_values = awc
        awc_core_dims = []

    # Wrapper to unpack 5-tuple and exclude params_dict from apply_ufunc
    def palmer_arrays_only(precip, pet, awc_val):
        pdsi, phdi, pmdi, z, params = palmer.pdsi(
            precip, pet, awc_val,
            data_start_year=data_start_year,
            calibration_year_initial=calibration_year_initial,
            calibration_year_final=calibration_year_final,
        )
        # Return only arrays, store params separately
        return np.stack([pdsi, phdi, pmdi, z], axis=0)

    # Compute using apply_ufunc
    result = xr.apply_ufunc(
        palmer_arrays_only,
        precip_aligned,
        pet_aligned,
        awc_values if isinstance(awc, xr.DataArray) else awc,
        input_core_dims=[[time_dim], [time_dim], awc_core_dims],
        output_core_dims=[["variable", time_dim]],
        vectorize=True,
        output_dtypes=[float],
    )

    # Unpack into Dataset
    ds = xr.Dataset({
        "pdsi": result.isel(variable=0).assign_attrs(CF_METADATA["pdsi"]),
        "phdi": result.isel(variable=1).assign_attrs(CF_METADATA["phdi"]),
        "pmdi": result.isel(variable=2).assign_attrs(CF_METADATA["pmdi"]),
        "z_index": result.isel(variable=3).assign_attrs(CF_METADATA["z_index"]),
    })

    # Compute params_dict separately for first grid cell (params are constant)
    if precip_aligned.ndim > 1:
        # Extract first spatial location
        first_slice = {dim: 0 for dim in precip_aligned.dims if dim != time_dim}
        precip_sample = precip_aligned.isel(first_slice).values
        pet_sample = pet_aligned.isel(first_slice).values
    else:
        precip_sample = precip_aligned.values
        pet_sample = pet_aligned.values

    _, _, _, _, params = palmer.pdsi(
        precip_sample, pet_sample, awc if isinstance(awc, float) else awc_values.flat[0],
        data_start_year=data_start_year,
        calibration_year_initial=calibration_year_initial,
        calibration_year_final=calibration_year_final,
    )

    if params is not None:
        ds.attrs["palmer_params"] = params

    return ds
```

**Trade-offs:**
- ✅ **Full control:** No decorator abstraction, explicit logic
- ✅ **Palmer-specific optimizations:** Handle AWC spatial parameter, params_dict separately
- ✅ **No decorator complexity:** Straightforward function flow
- ✅ **Testable:** Can unit test without decorator machinery
- ⚠️ **Code duplication:** Must manually implement alignment, inference, validation
- ⚠️ **No decorator benefits:** Loses shared infrastructure (logging, metadata provenance)
- ⚠️ **Params computation overhead:** Computes full Palmer twice (once for arrays, once for params)

### 2.5 params_dict Handling Strategies

**Challenge:** The 5th return value (`params_dict`) is a Python dict, not an array. It cannot pass through `xr.apply_ufunc`.

**Strategy 1: Exclude from apply_ufunc, compute separately**
```python
# Compute params once for representative grid cell
params = palmer.pdsi(sample_precip, sample_pet, awc, ...)[4]
ds.attrs["palmer_params"] = params
```
- ✅ Works with any pattern
- ⚠️ Requires second computation call
- ⚠️ Params are spatially constant—wasteful to compute per-grid-cell

**Strategy 2: Pre-compute params, pass as fitting_params**
```python
# Step 1: Compute params from representative sample
params = palmer.pdsi(precip_sample, pet_sample, awc, ...)[4]

# Step 2: Pass params to all grid cells via fitting_params kwarg
result = xr.apply_ufunc(
    lambda p, pet, awc: palmer.pdsi(p, pet, awc, fitting_params=params)[:4],  # Exclude params
    precip_da, pet_da, awc,
    ...
)
```
- ✅ Single computation per grid cell
- ✅ Leverages existing `fitting_params` parameter (line 829 in palmer.py)
- ❌ Requires modifying Palmer to return only 4 values when `fitting_params` provided

**Strategy 3: Serialize to Dataset attrs as JSON**
```python
ds.attrs["palmer_params"] = json.dumps(params)  # CF-compliant string
```
- ✅ Preserves params in xarray metadata
- ⚠️ Must deserialize on read: `params = json.loads(ds.attrs["palmer_params"])`

**Strategy 4: Return as separate object (tuple)**
```python
def palmer_xarray(...) -> tuple[xr.Dataset, dict | None]:
    ds = ...  # Dataset with 4 variables
    params = ...  # Computed separately
    return ds, params
```
- ✅ Explicit separation
- ❌ Breaks type consistency (other indices return Dataset, not tuple)

**Recommendation:** **Strategy 1 + 3** — Compute params separately for first grid cell, store as JSON in `ds.attrs["palmer_params"]`. This is CF-compliant, doesn't require Palmer function changes, and matches xarray's attribute conventions.

### 2.6 Pattern Comparison Matrix

| Criterion | Pattern A (Extend) | Pattern B (Standalone) | Pattern C (Manual) |
|-----------|-------------------|------------------------|-------------------|
| **Code reuse** | ✅ High (shared decorator) | ⚠️ Medium (duplicates logic) | ❌ Low (all manual) |
| **Complexity** | ⚠️ High (~500 lines) | ⚠️ Medium (~300 lines) | ✅ Low (~150 lines) |
| **Maintainability** | ⚠️ Conditional branching | ✅ Isolated concerns | ✅ Explicit logic |
| **Dask support** | ❌ No (multi-output limitation) | ❌ No (multi-output limitation) | ❌ No (multi-output limitation) |
| **Backward compat** | ✅ Yes | ✅ Yes | ✅ Yes |
| **params_dict handling** | ⚠️ Special-case logic | ⚠️ Special-case logic | ✅ Explicit compute |
| **Testing** | ⚠️ More paths to test | ⚠️ Two decorators to test | ✅ Single function test |
| **API clarity** | ⚠️ Many decorator params | ✅ Purpose-built API | ✅ Explicit function |
| **Future multi-output** | ✅ Reusable for others | ✅ Reusable for others | ❌ Palmer-specific |

### 2.7 Recommendation: Pattern C (Manual Wrapper) with Future Pattern B

**For Palmer (Epic 5):** Use **Pattern C (Manual wrapper)** because:
1. Palmer is the **only multi-output index** in the current roadmap
2. Manual approach is **simplest** (~150 lines vs ~300-500 for decorators)
3. Allows Palmer-specific optimizations (AWC handling, params pre-computation)
4. Avoids premature abstraction—no other index needs multi-output yet

**For future multi-output indices:** If another multi-output index emerges, **extract to Pattern B** (`@xarray_multi_adapter` decorator) by factoring common logic from `palmer_xarray()`.

**Migration path:**
```
Epic 5 (Palmer) → Manual wrapper (Pattern C)
                   ↓ (if needed)
Epic N (Another multi-output) → Extract decorator (Pattern B)
```

This follows "Rule of Three" refactoring: wait until you have 2-3 similar cases before abstracting.

---

---

## 3. Survey of Climate Libraries

### 3.1 Overview of Surveyed Libraries

This section examines how other Python climate libraries handle drought indices, multi-output functions, and xarray integration.

| Library | Version/Status | xarray Support | Palmer Indices | Multi-Output Pattern | Key Findings |
|---------|---------------|----------------|----------------|---------------------|--------------|
| **xclim** | Active (Ouranosinc) | ✅ Native | ❌ Not implemented | ✅ NamedTuples | Sophisticated indicator framework |
| **standard_precip** | Active (e-baumer) | ❌ Pandas only | ❌ Not implemented | ⚠️ Multi-column | No xarray path |
| **pyet** | Active (pyet-org) | ✅ Native | ❌ PET only | ❌ Single output | Simple functional API |
| **climate_indices** | Active (monocongo) | 🚧 In progress (Epic 2-5) | ✅ NumPy only | ❌ Tuples (NumPy) | **This project** |

**Key conclusion:** **No existing Python library provides Palmer indices with xarray multi-output support.** This research is breaking new ground.

### 3.2 xclim: Indicator Framework Architecture

[**xclim**](https://github.com/Ouranosinc/xclim) is the most architecturally mature xarray-native climate library (150+ indices).

#### 3.2.1 Palmer Status

Palmer indices **are not implemented** in xclim. From [Issue #131 (Drought indices)](https://github.com/Ouranosinc/xclim/issues/131):

> **Original request (2018):** "Drought indices from the source @huard pointed me is now developing an xarray interface."
>
> **Outcome:** Issue closed, cross-referenced to #973. The team chose to implement drought indices using "a similar strategy employed for fire weather indices—wrapping Python code translated from Fortran with numba and numpy."
>
> **climate_indices maintainer comment:** "I'd need to pull out numpy arrays in order to do several things that you can't do using dask, such as indexing individual array elements."

**Implication:** Sequential state machine algorithms (like Palmer) were deemed incompatible with xclim's dask-first architecture. They abandoned wrapping external libraries.

#### 3.2.2 Multi-Output Indicator Pattern

From the [xclim.core.indicator source](https://xclim.readthedocs.io/en/stable/_modules/xclim/core/indicator.html), xclim uses **NamedTuple returns** for multi-output indicators:

**Architecture:**
```python
class Indicator:
    """Base class for climate indicators."""

    @property
    def n_outs(self) -> int:
        """Return len(self.cf_attrs) — number of outputs."""
        return len(self.cf_attrs)

    def __call__(self, *args, **kwargs):
        # Compute function returns tuple of DataArrays
        outs = self.compute(*args, **kwargs)

        # Validate output count
        if len(outs) != self.n_outs:
            raise ValueError(f"Expected {self.n_outs} outputs, got {len(outs)}.")

        # Return NamedTuple for multi-output
        if self.n_outs > 1:
            return self._output_namedtuple(*outs)
        else:
            return outs[0]  # Single DataArray
```

**cf_attrs structure:**
```python
cf_attrs = [
    {"long_name": "Palmer Drought Severity Index", "units": ""},
    {"long_name": "Palmer Hydrological Drought Index", "units": ""},
    {"long_name": "Palmer Modified Drought Index", "units": ""},
    {"long_name": "Palmer Z-Index", "units": ""},
]
```

**Return type:**
```python
# Single output
spi: xr.DataArray = xclim.indices.standardized_precipitation_index(...)

# Multi-output (hypothetical)
PalmerOutputs = NamedTuple("PalmerOutputs", [("pdsi", xr.DataArray), ("phdi", xr.DataArray), ...])
result: PalmerOutputs = xclim.indices.palmer_drought_index(...)
result.pdsi  # Access individual outputs
```

**Trade-offs:**
- ✅ **Pythonic:** NamedTuples support `.pdsi` attribute access
- ✅ **Type-safe:** Each field is explicitly typed
- ⚠️ **Conversion to Dataset:** Requires explicit `xr.Dataset(dict(result._asdict()))` or similar
- ⚠️ **Not CF-compliant container:** NetCDF expects Dataset, not tuple

**Comparison to climate_indices approach:**
- xclim: Returns NamedTuple → user converts to Dataset if needed
- climate_indices (proposed): Returns Dataset directly → CF-compliant out-of-the-box

#### 3.2.3 Decorator Pattern

xclim uses a **class-based indicator system** rather than function decorators:

From [xclim.core.indicator](https://xclim.readthedocs.io/en/stable/_modules/xclim/core/indicator.html):

```python
# Unit declarations via decorator
@declare_units(pr="[precipitation]", thresh="[precipitation]")
def drought_code(pr: xr.DataArray, thresh: str = "1 mm/day") -> xr.DataArray:
    """Compute drought code."""
    ...

# Indicator wraps the function
DC = Indicator(
    identifier="drought_code",
    compute=drought_code,
    cf_attrs=[{"long_name": "Drought Code", "units": ""}],
)
```

**Key mechanisms:**
1. **`@declare_units`**: Validates and converts input units (e.g., `pr="[precipitation]"`)
2. **`@declare_relative_units`**: Checks units relative to other params (e.g., `thresh="<pr>"`)
3. **Indicator class**: Wraps compute function, handles metadata, parameter inference

**Differences from climate_indices:**
- xclim: Class-based (Indicator instances), units handled via dedicated decorators
- climate_indices: Function-based (`@xarray_adapter` decorator), units in CF metadata dict

### 3.3 standard_precip: Pandas-Only Approach

[**standard_precip**](https://github.com/e-baumer/standard_precip) by e-baumer implements SPI/SPEI using Pandas DataFrames.

**API signature:**
```python
spi.calculate(
    rainfall_data,      # Pandas DataFrame
    'date',             # date column name
    'precip',           # precipitation column (or list of columns)
    freq="M",           # frequency
    scale=1,
    fit_type="lmom",
    dist_type="gam",
)
```

**Multi-output handling:**
- Supports "a list of precipitation columns to process"
- Returns DataFrame with SPI values for each column
- **No xarray support** — purely tabular data

**Relevance to Palmer:**
- ❌ No xarray patterns to adopt
- ❌ No Palmer implementation
- ✅ Multi-column processing shows demand for batch computation

### 3.4 pyet: Simple Functional xarray API

[**pyet**](https://github.com/pyet-org/pyet) (v1.3.1) estimates potential evapotranspiration (PET) from time series and gridded data.

From [GMD publication](https://gmd.copernicus.org/articles/17/7083/2024/):

> "As of version v1.2, pyet is compatible with both Pandas.Series and xarray.DataArray, which means you can now estimate potential evapotranspiration for both point and gridded data."

**API pattern:**
```python
# Simple functional interface
pet = pyet.thornthwaite(tmean, lat)  # Returns xr.DataArray
```

**Architecture:**
- ✅ Single-output functions (one PET method per function)
- ✅ Dual numpy/xarray support (type detection)
- ❌ No multi-output (each method returns one PET estimate)
- ❌ No Palmer indices

**Relevance:**
- Similar "type detection → dispatch" pattern to `climate_indices`
- Demonstrates viability of simple functional API (no complex decorator framework)
- No multi-output patterns to learn from

### 3.5 climate_indices: Current State (This Project)

**NumPy-only Palmer implementation** ([palmer.py:822-848](https://github.com/monocongo/climate_indices/blob/master/src/climate_indices/palmer.py)):

```python
def pdsi(
    precips: np.ndarray,
    pet: np.ndarray,
    awc: float,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    """
    Compute PDSI, PHDI, PMDI, Palmer Z-Index.

    Returns:
        Four numpy arrays (PDSI, PHDI, PMDI, Z-Index) and
        a dictionary of fitted parameters (alpha, beta, gamma, delta)
    """
```

**Current CLI dispatch** ([__main__.py:1366-1430](https://github.com/monocongo/climate_indices/blob/master/src/climate_indices/__main__.py)):
- Uses `multiprocessing.Pool` to parallelize across grid cells
- Each worker processes one (lat, lon) location with 1D time series
- Gathers results and writes to NetCDF

**Epic 2-5 progress:**
- ✅ Epic 2: `@xarray_adapter` decorator for SPI/SPEI (single-output)
- 🚧 Epic 5: Palmer xarray support (this research)

### 3.6 Key Lessons from Library Survey

#### 3.6.1 No Palmer + xarray Multi-Output Precedent

**Finding:** Zero Python climate libraries implement Palmer with xarray multi-output support.

**Reasons identified:**
1. **Sequential state incompatibility** — xclim maintainer: "can't do using dask, such as indexing individual array elements"
2. **Complexity investment** — Palmer's calibration + water balance + backtracking is non-trivial
3. **Alternative sources** — PDSI datasets available pre-computed (gridMET, CMIP6)

**Implication:** `climate_indices` will be **first-in-class** for this capability.

#### 3.6.2 Multi-Output API Patterns

| Pattern | Library | Return Type | Pros | Cons |
|---------|---------|-------------|------|------|
| **NamedTuple** | xclim | `NamedTuple[xr.DataArray, ...]` | Pythonic access (`.field`), type-safe | Not CF-compliant, requires conversion to Dataset |
| **Tuple** | climate_indices (NumPy) | `tuple[np.ndarray, ...]` | Simple, matches NumPy convention | No field names, positional-only |
| **Dataset** | (proposed) | `xr.Dataset` | CF-compliant, NetCDF-ready, metadata per-variable | Heavier return type, less "functional" |

**Recommendation:** **xr.Dataset return** aligns with climate_indices' focus on operational climate services and NetCDF interchange.

#### 3.6.3 Decorator vs Class-Based Architecture

| Approach | Library | Pros | Cons |
|----------|---------|------|------|
| **Decorator** | climate_indices | Lightweight, preserves function signatures, backward-compatible | Limited to decorator params |
| **Class-based** | xclim | Rich metadata system, extensible via inheritance, parameter validation | Heavier abstraction, steeper learning curve |

**Observation:** climate_indices' decorator approach is appropriate for its scale (~10 indices). xclim's class system justifies overhead with 150+ indices.

#### 3.6.4 Sequential State Handling

**No library surveyed has solved sequential state + xarray parallelization.**

- xclim: Avoided the problem (no Palmer)
- standard_precip: Pandas (no spatial parallelization)
- pyet: No sequential indices

**Implication:** Pattern C (manual wrapper) with `vectorize=True` is a **pragmatic solution** — acknowledges the fundamental constraint that time cannot be parallelized for Palmer.

---

---

## 4. Concrete Implementation Plan

### 4.1 Core Apply_ufunc Pattern for Palmer

**Goal:** Wrap Palmer's 5-tuple return into a single array, call via `apply_ufunc`, then unpack.

**Implementation:**

```python
def palmer_xarray(
    precip_da: xr.DataArray,
    pet_da: xr.DataArray,
    awc: float | xr.DataArray,
    *,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    time_dim: str = "time",
) -> xr.Dataset:
    """Compute Palmer drought indices with xarray support.

    Args:
        precip_da: Monthly precipitation (mm)
        pet_da: Monthly potential evapotranspiration (mm)
        awc: Available water capacity (inches). Scalar or DataArray with spatial dims.
        calibration_year_initial: Start of calibration period (inferred if None)
        calibration_year_final: End of calibration period (inferred if None)
        time_dim: Name of time dimension (default: "time")

    Returns:
        Dataset with variables: pdsi, phdi, pmdi, z_index
        Calibration parameters stored in ds.attrs["palmer_params"] as JSON
    """
    # ═══════════════════════════════════════════════════════════════════
    # Step 1: Align inputs
    # ═══════════════════════════════════════════════════════════════════
    precip_aligned, pet_aligned = xr.align(precip_da, pet_da, join="inner")

    # ═══════════════════════════════════════════════════════════════════
    # Step 2: Infer temporal parameters from time coordinate
    # ═══════════════════════════════════════════════════════════════════
    time_coord = precip_aligned[time_dim]
    data_start_year = int(time_coord.dt.year.values[0])

    if calibration_year_initial is None:
        calibration_year_initial = data_start_year
    if calibration_year_final is None:
        calibration_year_final = int(time_coord.dt.year.values[-1])

    # ═══════════════════════════════════════════════════════════════════
    # Step 3: Handle AWC (spatial parameter)
    # ═══════════════════════════════════════════════════════════════════
    # AWC is spatial-only (no time dimension)
    # If scalar: will broadcast to all grid cells
    # If DataArray: must have compatible spatial dims (lat, lon)
    if isinstance(awc, xr.DataArray):
        awc_input = awc
        awc_core_dims = []  # No time dimension in AWC
    else:
        # Scalar AWC: pass directly
        awc_input = awc
        awc_core_dims = []

    # ═══════════════════════════════════════════════════════════════════
    # Step 4: Wrapper function for apply_ufunc
    # ═══════════════════════════════════════════════════════════════════
    # Exclude params_dict from apply_ufunc (compute separately)
    def palmer_arrays_only(
        precip: np.ndarray,
        pet: np.ndarray,
        awc_val: float | np.ndarray,
    ) -> np.ndarray:
        """Wrap Palmer to return stacked array instead of 5-tuple.

        Args:
            precip: 1D time series (time,)
            pet: 1D time series (time,)
            awc_val: Scalar or 0D array

        Returns:
            Stacked array shape (4, time) with [pdsi, phdi, pmdi, z_index]
        """
        # Handle scalar AWC from vectorization
        awc_scalar = float(awc_val) if np.ndim(awc_val) == 0 else float(awc_val.flat[0])

        # Call Palmer (returns 5-tuple)
        pdsi, phdi, pmdi, z_index, _ = palmer.pdsi(
            precip,
            pet,
            awc_scalar,
            data_start_year=data_start_year,
            calibration_year_initial=calibration_year_initial,
            calibration_year_final=calibration_year_final,
            fitting_params=None,  # Will compute params separately
        )

        # Stack into single array with new dimension
        # Shape: (4, time)
        return np.stack([pdsi, phdi, pmdi, z_index], axis=0)

    # ═══════════════════════════════════════════════════════════════════
    # Step 5: Call apply_ufunc with multi-output workaround
    # ═══════════════════════════════════════════════════════════════════
    result = xr.apply_ufunc(
        palmer_arrays_only,
        precip_aligned,
        pet_aligned,
        awc_input,
        input_core_dims=[
            [time_dim],      # precip: operate on time dimension
            [time_dim],      # pet: operate on time dimension
            awc_core_dims,   # awc: no core dims (spatial broadcast)
        ],
        output_core_dims=[["variable", time_dim]],  # Output: (variable, time)
        vectorize=True,  # Loop over spatial dims (lat, lon)
        output_dtypes=[float],
    )

    # ═══════════════════════════════════════════════════════════════════
    # Step 6: Unpack into Dataset with per-variable CF metadata
    # ═══════════════════════════════════════════════════════════════════
    ds = xr.Dataset({
        "pdsi": result.isel(variable=0).drop_vars("variable").assign_attrs(
            long_name="Palmer Drought Severity Index",
            units="",
            references=(
                "Palmer, W. C. (1965). Meteorological Drought. "
                "U.S. Weather Bureau Research Paper 45."
            ),
        ),
        "phdi": result.isel(variable=1).drop_vars("variable").assign_attrs(
            long_name="Palmer Hydrological Drought Index",
            units="",
            references="Palmer, W. C. (1965). Meteorological Drought.",
        ),
        "pmdi": result.isel(variable=2).drop_vars("variable").assign_attrs(
            long_name="Palmer Modified Drought Index",
            units="",
            references=(
                "Heddinghaus, T. R., & Sabol, P. (1991). "
                "A Review of the Palmer Drought Severity Index. "
                "Climate Monitoring and Diagnostics Laboratory Summary Report, 17."
            ),
        ),
        "z_index": result.isel(variable=3).drop_vars("variable").assign_attrs(
            long_name="Palmer Z-Index",
            units="",
            references="Palmer, W. C. (1965). Meteorological Drought.",
        ),
    })

    # ═══════════════════════════════════════════════════════════════════
    # Step 7: Compute params_dict separately (Strategy 1)
    # ═══════════════════════════════════════════════════════════════════
    # Params are spatially constant, so compute once from first grid cell
    if precip_aligned.ndim > 1:
        # Extract first spatial location
        first_slice = {dim: 0 for dim in precip_aligned.dims if dim != time_dim}
        precip_sample = precip_aligned.isel(first_slice).values
        pet_sample = pet_aligned.isel(first_slice).values
    else:
        # 1D time series
        precip_sample = precip_aligned.values
        pet_sample = pet_aligned.values

    # Compute params from sample
    awc_sample = float(awc) if isinstance(awc, float) else float(awc_input.values.flat[0])
    _, _, _, _, params = palmer.pdsi(
        precip_sample,
        pet_sample,
        awc_sample,
        data_start_year=data_start_year,
        calibration_year_initial=calibration_year_initial,
        calibration_year_final=calibration_year_final,
    )

    # Store params as JSON in attrs (CF-compliant)
    if params is not None:
        import json
        ds.attrs["palmer_params"] = json.dumps(params)

    # ═══════════════════════════════════════════════════════════════════
    # Step 8: Add global metadata
    # ═══════════════════════════════════════════════════════════════════
    ds.attrs.update({
        "calibration_year_initial": calibration_year_initial,
        "calibration_year_final": calibration_year_final,
        "data_start_year": data_start_year,
        "awc_inches": float(awc) if isinstance(awc, float) else "spatially_varying",
    })

    return ds
```

### 4.2 AWC Spatial Parameter Handling

**Challenge:** AWC (available water capacity) is a **soil property**, not a time series. It varies spatially (lat, lon) but not temporally.

**Pattern:** Use empty `input_core_dims` for AWC to signal "no time dimension."

```python
# Example: AWC as scalar (uniform soil)
awc = 2.5  # inches
result = palmer_xarray(precip_da, pet_da, awc)

# Example: AWC as DataArray (spatially varying soil)
awc_da = xr.DataArray(
    np.random.uniform(1.5, 3.5, size=(nlat, nlon)),
    dims=["lat", "lon"],
    coords={"lat": lats, "lon": lons},
    attrs={"long_name": "Available Water Capacity", "units": "inches"},
)
result = palmer_xarray(precip_da, pet_da, awc_da)
```

**apply_ufunc call:**
```python
xr.apply_ufunc(
    palmer_arrays_only,
    precip_aligned,  # dims: (time, lat, lon)
    pet_aligned,     # dims: (time, lat, lon)
    awc_input,       # dims: (lat, lon) OR scalar
    input_core_dims=[
        ["time"],    # precip: time is core
        ["time"],    # pet: time is core
        [],          # awc: NO core dims (broadcast over lat, lon)
    ],
    ...
)
```

**How it works:**
- xarray broadcasts AWC across the time dimension automatically
- Each (lat, lon) grid cell gets its corresponding AWC value
- `vectorize=True` loops over (lat, lon), passing 1D time slices + scalar AWC to function

### 4.3 Dataset Assembly with CF Metadata

**Pattern:** Each variable gets independent CF attributes via `assign_attrs()`.

```python
# CF metadata registry (add to xarray_adapter.py)
CF_METADATA["pdsi"] = {
    "long_name": "Palmer Drought Severity Index",
    "units": "",  # Unitless (standardized index)
    "references": (
        "Palmer, W. C. (1965). Meteorological Drought. "
        "U.S. Weather Bureau Research Paper 45."
    ),
}

CF_METADATA["phdi"] = {
    "long_name": "Palmer Hydrological Drought Index",
    "units": "",
    "references": "Palmer, W. C. (1965). Meteorological Drought.",
}

CF_METADATA["pmdi"] = {
    "long_name": "Palmer Modified Drought Index",
    "units": "",
    "references": (
        "Heddinghaus, T. R., & Sabol, P. (1991). "
        "A Review of the Palmer Drought Severity Index. "
        "Climate Monitoring and Diagnostics Laboratory Summary Report, 17."
    ),
}

CF_METADATA["z_index"] = {
    "long_name": "Palmer Z-Index",
    "units": "",
    "references": "Palmer, W. C. (1965). Meteorological Drought.",
}

# Dataset assembly (Pattern A from Step 3)
ds = xr.Dataset({
    var_name: result.isel(variable=i).drop_vars("variable").assign_attrs(
        CF_METADATA[var_name]
    )
    for i, var_name in enumerate(["pdsi", "phdi", "pmdi", "z_index"])
})
```

**CF compliance verification:**
```python
# Check metadata
assert ds["pdsi"].attrs["long_name"] == "Palmer Drought Severity Index"
assert ds["pdsi"].attrs["units"] == ""

# Write to NetCDF (CF-compliant)
ds.to_netcdf("palmer_indices.nc")
```

### 4.4 Type Annotations with @overload

**Pattern:** Add to `typed_public_api.py` following existing SPI/SPEI pattern.

```python
# typed_public_api.py additions

from typing import overload
import numpy as np
import numpy.typing as npt
import xarray as xr

# NumPy overload (existing palmer.pdsi signature)
@overload
def pdsi(
    precips: npt.NDArray[np.float64],
    pet: npt.NDArray[np.float64],
    awc: float,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict[str, Any] | None = None,
) -> tuple[
    npt.NDArray[np.float64],  # pdsi
    npt.NDArray[np.float64],  # phdi
    npt.NDArray[np.float64],  # pmdi
    npt.NDArray[np.float64],  # z_index
    dict[str, Any] | None,    # params_dict
]: ...


# xarray overload (new palmer_xarray signature)
@overload
def pdsi(
    precips: xr.DataArray,
    pet: xr.DataArray,
    awc: float | xr.DataArray,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    fitting_params: dict[str, Any] | None = None,
    time_dim: str = "time",
) -> xr.Dataset: ...  # NOTE: Returns Dataset, not DataArray


# Implementation (runtime dispatcher)
def pdsi(
    precips: npt.NDArray[np.float64] | xr.DataArray,
    pet: npt.NDArray[np.float64] | xr.DataArray,
    awc: float | xr.DataArray,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    fitting_params: dict[str, Any] | None = None,
    time_dim: str = "time",
) -> tuple[npt.NDArray[np.float64], ...] | xr.Dataset:
    """Compute Palmer drought indices (PDSI, PHDI, PMDI, Z-Index).

    Type-safe dispatch:
    - NumPy inputs → tuple of 5 arrays + params dict
    - xarray inputs → Dataset with 4 variables (params in attrs)
    """
    if isinstance(precips, xr.DataArray):
        # xarray path
        return palmer_xarray(
            precips, pet, awc,
            calibration_year_initial=calibration_year_initial,
            calibration_year_final=calibration_year_final,
            time_dim=time_dim,
        )
    else:
        # NumPy path (existing palmer.pdsi)
        return palmer.pdsi(
            precips, pet, awc,
            data_start_year=data_start_year,
            calibration_year_initial=calibration_year_initial,
            calibration_year_final=calibration_year_final,
            fitting_params=fitting_params,
        )
```

**Type checking behavior:**

```python
# NumPy path
p, h, m, z, params = pdsi(precip_np, pet_np, awc=2.5, ...)
# Type: tuple[NDArray, NDArray, NDArray, NDArray, dict | None]

# xarray path
ds = pdsi(precip_da, pet_da, awc=2.5, ...)
# Type: xr.Dataset

# IDE autocomplete
ds["pdsi"]  # ✅ mypy knows this is valid
ds.pdsi     # ✅ Also valid (xarray attribute access)
```

### 4.5 params_dict Handling Strategy (Final)

**Chosen strategy:** Compute separately + JSON serialization (Strategy 1 + 3 from Section 2.5).

**Implementation:**

```python
# Step 1: Compute params from representative sample
if precip_aligned.ndim > 1:
    first_slice = {dim: 0 for dim in precip_aligned.dims if dim != time_dim}
    precip_sample = precip_aligned.isel(first_slice).values
    pet_sample = pet_aligned.isel(first_slice).values
else:
    precip_sample = precip_aligned.values
    pet_sample = pet_aligned.values

# Step 2: Call palmer.pdsi for params only
_, _, _, _, params = palmer.pdsi(
    precip_sample, pet_sample, awc_sample,
    data_start_year=data_start_year,
    calibration_year_initial=calibration_year_initial,
    calibration_year_final=calibration_year_final,
)

# Step 3: Serialize to JSON and store in Dataset attrs
if params is not None:
    import json
    ds.attrs["palmer_params"] = json.dumps(params)

# Step 4: Usage — deserialize when needed
params_dict = json.loads(ds.attrs["palmer_params"])
# {"alpha": 1.5, "beta": 0.8, "gamma": 0.6, "delta": 0.3}
```

**Trade-offs:**
- ✅ CF-compliant (attrs are strings in NetCDF)
- ✅ Params preserved when writing/reading NetCDF
- ✅ No Palmer function modification needed
- ⚠️ Requires deserialization on read (minor overhead)

**Alternative for direct access:**
```python
# Store as separate attributes (more ergonomic)
if params is not None:
    ds.attrs["palmer_alpha"] = params["alpha"]
    ds.attrs["palmer_beta"] = params["beta"]
    ds.attrs["palmer_gamma"] = params["gamma"]
    ds.attrs["palmer_delta"] = params["delta"]

# Access directly
alpha = ds.attrs["palmer_alpha"]
```

### 4.6 Migration Roadmap

**Phase 1: Core Implementation (Epic 5, Story 5.1-5.3)**

1. ✅ **Story 5.1:** Add `palmer_xarray()` function
   - Manual wrapper (Pattern C from Section 2.7)
   - File: `src/climate_indices/palmer_xarray.py` (new)
   - Estimated: 150 lines

2. ✅ **Story 5.2:** Add CF metadata for Palmer variables
   - Extend `CF_METADATA` dict in `xarray_adapter.py`
   - Add entries for `pdsi`, `phdi`, `pmdi`, `z_index`
   - Estimated: 20 lines

3. ✅ **Story 5.3:** Add `@overload` signatures to `typed_public_api.py`
   - NumPy overload: `tuple[NDArray, ...]`
   - xarray overload: `xr.Dataset`
   - Runtime dispatcher
   - Estimated: 50 lines

**Phase 2: Testing & Validation (Epic 5, Story 5.4-5.6)**

4. ✅ **Story 5.4:** Unit tests for `palmer_xarray()`
   - Test scalar AWC
   - Test DataArray AWC (spatial variation)
   - Test params_dict serialization
   - Test CF metadata
   - File: `tests/test_palmer_xarray.py`
   - Estimated: 200 lines

5. ✅ **Story 5.5:** Integration tests
   - Compare NumPy path vs xarray path (same results)
   - Test NetCDF round-trip (write + read)
   - Test multi-dimensional (3D: time, lat, lon)
   - Estimated: 150 lines

6. ✅ **Story 5.6:** Performance validation
   - Benchmark `vectorize=True` overhead vs current multiprocessing
   - Document performance characteristics
   - Estimated: 50 lines (benchmark script)

**Phase 3: Documentation (Epic 5, Story 5.7-5.8)**

7. ✅ **Story 5.7:** API documentation
   - Docstring for `palmer_xarray()`
   - Usage examples (scalar AWC, DataArray AWC)
   - Migration guide from NumPy to xarray
   - File: `docs/palmer_xarray.md`
   - Estimated: 100 lines

8. ✅ **Story 5.8:** Update CLI
   - Add `--format=netcdf-dataset` option for Palmer
   - Update `__main__.py` to use `palmer_xarray()` for NetCDF output
   - Estimated: 50 lines

**Total estimated additions: ~770 lines**

### 4.7 Code Location Summary

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `src/climate_indices/palmer_xarray.py` | New `palmer_xarray()` function | ~150 | ✅ Create |
| `src/climate_indices/xarray_adapter.py` | Add Palmer CF metadata | ~20 | ✅ Extend |
| `src/climate_indices/typed_public_api.py` | Add `@overload` signatures | ~50 | ✅ Extend |
| `tests/test_palmer_xarray.py` | Unit tests | ~200 | ✅ Create |
| `tests/integration/test_palmer_integration.py` | Integration tests | ~150 | ✅ Create |
| `benchmarks/palmer_performance.py` | Performance validation | ~50 | ✅ Create |
| `docs/palmer_xarray.md` | User documentation | ~100 | ✅ Create |
| `src/climate_indices/__main__.py` | CLI updates | ~50 | ✅ Extend |

---

---

## 5. Risk Assessment

### 5.1 Technical Risks

#### Risk 1: Performance Regression (vectorize=True Overhead)

**Severity:** Medium
**Likelihood:** High
**Impact:** Python loop overhead slower than NumPy vectorization

**Analysis:**
- `vectorize=True` uses `numpy.vectorize()`, which is a **Python-level loop**, not compiled NumPy operations
- For N grid cells, Palmer is called N times sequentially (or via Python thread pool)
- Current CLI uses `multiprocessing.Pool` for parallelization across grid cells

**Mitigation:**
- ✅ **Not a regression:** Existing CLI already loops per-grid-cell via multiprocessing
- ✅ **Sequential time constraint:** Palmer's state machine can't parallelize along time anyway
- ✅ **Spatial parallelization still possible:** Dask can chunk spatial dims, vectorize loops within chunks
- 📊 **Benchmark required:** Story 5.6 will measure actual overhead

**Acceptance criteria:**
- xarray path performance ≥ 80% of current multiprocessing CLI
- Document performance characteristics in user guide

#### Risk 2: Dask Multi-Output Limitation Blocks Future Optimization

**Severity:** Low
**Likelihood:** High (confirmed in Issue #1815)
**Impact:** Cannot use `dask='parallelized'` for multi-output

**Analysis:**
- `dask='parallelized'` does NOT support multi-output (NotImplementedError since 2018)
- Workaround (stack outputs) works but loses semantic clarity
- Future xarray versions may fix this, but timeline unknown

**Mitigation:**
- ✅ **Sequential constraint dominant:** Palmer can't parallelize time regardless of dask support
- ✅ **Workaround proven:** Stack/unpack pattern works (Section 4.1)
- ⚠️ **Monitor xarray releases:** If Issue #1815 resolved, refactor to native multi-output

**Decision:** Accept limitation, document workaround, revisit in future.

#### Risk 3: params_dict Serialization Complexity

**Severity:** Low
**Likelihood:** Medium
**Impact:** JSON serialization/deserialization adds user friction

**Analysis:**
- params_dict stored as JSON string in `ds.attrs["palmer_params"]`
- Users must `json.loads()` to access params dict
- NetCDF attrs are strings, so serialization required for dict storage

**Mitigation:**
- ✅ **Also store individual params:** Add `ds.attrs["palmer_alpha"]` etc. for direct access
- ✅ **Document pattern:** Show deserialization example in API docs
- ✅ **CF-compliant:** String attrs work with all NetCDF tools

**Alternative considered:** Return `(Dataset, dict)` tuple → rejected (breaks type consistency).

#### Risk 4: AWC DataArray Validation Gap

**Severity:** Medium
**Likelihood:** Medium
**Impact:** User passes AWC with time dimension → incorrect results

**Analysis:**
- AWC should be spatial-only (lat, lon), no time dimension
- If user passes AWC(time, lat, lon), function will silently broadcast incorrectly
- No runtime validation that AWC lacks time dimension

**Mitigation:**
- ✅ **Add validation in Story 5.1:**
  ```python
  if isinstance(awc, xr.DataArray) and time_dim in awc.dims:
      raise ValueError(
          f"AWC should not have time dimension '{time_dim}'. "
          f"AWC is a soil property (spatial only). Got dims={awc.dims}"
      )
  ```
- ✅ **Document in docstring:** Explicitly state AWC dims requirement
- ✅ **Test case in Story 5.4:** Test that time-varying AWC raises error

#### Risk 5: CF Metadata Incompleteness

**Severity:** Low
**Likelihood:** Low
**Impact:** NetCDF files missing recommended CF attributes

**Analysis:**
- Current CF metadata (Section 4.3) covers long_name, units, references
- Missing: `valid_range`, `valid_min`, `valid_max` for Palmer indices
- Missing: `cell_methods` for time aggregation

**Mitigation:**
- ✅ **Add valid_range in Story 5.2:**
  ```python
  "pdsi": {
      "long_name": "Palmer Drought Severity Index",
      "units": "",
      "valid_range": (-10.0, 10.0),  # Typical PDSI range
      "references": "Palmer (1965)...",
  }
  ```
- 📋 **Future enhancement:** Add `cell_methods` after Epic 6 (temporal aggregation)

### 5.2 Maintainability Risks

#### Risk 6: Code Duplication (Manual Wrapper vs Decorator)

**Severity:** Low
**Likelihood:** High
**Impact:** palmer_xarray() duplicates alignment/inference logic from @xarray_adapter

**Analysis:**
- Pattern C (manual wrapper) duplicates ~50 lines of logic:
  - Input alignment (`xr.align`)
  - Parameter inference from time coordinate
  - Metadata attachment
- If `@xarray_adapter` changes, `palmer_xarray()` may need parallel updates

**Mitigation:**
- ✅ **Extract utilities:** Factor out `_align_inputs()`, `_infer_temporal_params()` as module-level helpers
- ✅ **Reuse CF_METADATA:** Share metadata dict with SPI/SPEI
- 📋 **Future refactor:** If 2nd multi-output index emerges, extract to `@xarray_multi_adapter` (Pattern B)

**Acceptance:** Acceptable for single multi-output index, revisit at 2-3 indices.

#### Risk 7: Type Annotation Maintenance

**Severity:** Low
**Likelihood:** Medium
**Impact:** `@overload` signatures can drift from implementation

**Analysis:**
- Three locations define Palmer signature:
  1. `palmer.py:pdsi()` — NumPy implementation
  2. `palmer_xarray.py:palmer_xarray()` — xarray implementation
  3. `typed_public_api.py:pdsi()` — overloaded dispatcher
- Changes to signature require updates in all three

**Mitigation:**
- ✅ **Type checking in CI:** `mypy --strict` catches signature mismatches
- ✅ **Test coverage:** Story 5.4 includes type checking tests
- 📋 **Documentation:** Note signature contract in Palmer docstrings

### 5.3 Compatibility Risks

#### Risk 8: xarray Version Dependency

**Severity:** Low
**Likelihood:** Low
**Impact:** Multi-output pattern may break in future xarray versions

**Analysis:**
- Current pattern relies on:
  - `output_core_dims` list length = output count
  - `vectorize=True` unpacking tuples
  - `.isel(variable=N)` indexing
- xarray API is generally stable, but multi-output handling is edge case

**Mitigation:**
- ✅ **Pin xarray version:** `pyproject.toml` specifies `xarray>=2024.1.0`
- ✅ **CI testing:** Test against xarray stable + latest
- ✅ **Monitor releases:** Subscribe to xarray release notes

#### Risk 9: NumPy 2.0 Compatibility

**Severity:** Low
**Likelihood:** Low
**Impact:** `np.stack()` behavior change or deprecation

**Analysis:**
- NumPy 2.0 (released 2024) introduced breaking changes
- `np.stack()` is core API, unlikely to break
- climate_indices already supports NumPy 2.x (Epic 1)

**Mitigation:**
- ✅ **Already mitigated:** Epic 1 addressed NumPy 2.0 compatibility
- ✅ **CI matrix:** Tests run against NumPy 1.x and 2.x

### 5.4 User Experience Risks

#### Risk 10: Return Type Inconsistency (Dataset vs DataArray)

**Severity:** Medium
**Likelihood:** High
**Impact:** Users expect DataArray (like SPI/SPEI), get Dataset

**Analysis:**
- SPI/SPEI return `xr.DataArray` (single variable)
- Palmer returns `xr.Dataset` (four variables)
- Users may attempt `result.values` → fails (Dataset has no `.values`)

**Mitigation:**
- ✅ **Clear documentation:** Docstring emphasizes Dataset return
- ✅ **Type annotations:** `@overload` makes return type explicit in IDE
- ✅ **Examples in docs:** Story 5.7 shows Dataset access patterns
- ✅ **Error message improvement:**
  ```python
  # Example usage in docs
  ds = pdsi(precip_da, pet_da, awc)
  pdsi_values = ds["pdsi"].values  # Correct
  # NOT: ds.values  # AttributeError
  ```

**Acceptance:** Intentional design choice (multi-output requires Dataset).

### 5.5 Risk Summary Matrix

| Risk | Severity | Likelihood | Mitigation Status | Owner |
|------|----------|------------|-------------------|-------|
| 1. Performance regression | Medium | High | ✅ Accepted + benchmark | Epic 5 |
| 2. Dask multi-output limitation | Low | High | ✅ Workaround proven | Epic 5 |
| 3. params_dict complexity | Low | Medium | ✅ Dual access pattern | Story 5.1 |
| 4. AWC validation gap | Medium | Medium | ✅ Add validation | Story 5.1 |
| 5. CF metadata incomplete | Low | Low | ✅ Add valid_range | Story 5.2 |
| 6. Code duplication | Low | High | ✅ Extract utilities | Story 5.1 |
| 7. Type annotation drift | Low | Medium | ✅ CI type checking | Epic 5 |
| 8. xarray version dependency | Low | Low | ✅ Pin version + CI | Epic 5 |
| 9. NumPy 2.0 compatibility | Low | Low | ✅ Already mitigated | Epic 1 |
| 10. Return type confusion | Medium | High | ✅ Documentation | Story 5.7 |

**Overall risk level:** **Low-Medium** — All risks have identified mitigations, no blockers.

---

---

## 6. Recommendations

### 6.1 Primary Recommendation: Implement Pattern C (Manual Wrapper)

**Recommendation:** Adopt **Pattern C (manual `palmer_xarray()` wrapper)** for Palmer drought indices xarray integration.

**Rationale:**

1. **Simplicity:** ~150 lines of code vs ~300-500 for decorator-based approaches
2. **Palmer-specific optimizations:** Allows tailored handling of:
   - AWC spatial parameter (no time dimension)
   - params_dict pre-computation (single computation vs N grid cells)
   - Multi-output stacking strategy
3. **No premature abstraction:** Palmer is currently the only multi-output index
4. **Testability:** Single function easier to unit test than decorator machinery
5. **Maintainability:** Explicit logic flow vs decorator metaprogramming

**Implementation path:**
```
Epic 5 (Palmer) → Manual wrapper (Pattern C)
                   ↓ (if 2nd multi-output index emerges)
Epic N (Future)  → Extract decorator (Pattern B: @xarray_multi_adapter)
```

**Decision criteria for decorator extraction:**
- **Trigger:** When a 2nd multi-output index is planned
- **Threshold:** At least 100 lines of duplicated logic
- **Approach:** Factor common patterns from `palmer_xarray()` into reusable decorator

### 6.2 Core Technical Recommendations

#### Recommendation 1: Use Multi-Output Workaround (Stack/Unpack)

**Adopt:** `np.stack([pdsi, phdi, pmdi, z_index], axis=0)` → `output_core_dims=[["variable", time_dim]]` → unpack via `.isel(variable=N)`

**Justification:**
- ✅ Proven pattern (Section 1.3 workaround from Issue #1815)
- ✅ Avoids NotImplementedError for `dask='parallelized'` + multi-output
- ✅ Single `apply_ufunc` call (no wrapper complexity)
- ⚠️ Semantic clarity trade-off (variable dimension is artificial)

**Alternative rejected:** Five separate `apply_ufunc` calls → 5x redundant computation.

#### Recommendation 2: Return xr.Dataset (Not NamedTuple)

**Adopt:** `palmer_xarray() -> xr.Dataset` with variables `{pdsi, phdi, pmdi, z_index}`

**Justification:**
- ✅ CF-compliant container (NetCDF standard)
- ✅ Per-variable metadata (long_name, units, references)
- ✅ Direct NetCDF write: `ds.to_netcdf("output.nc")`
- ✅ Aligns with operational climate services focus
- ⚠️ Different from SPI/SPEI (DataArray) → document clearly

**Alternative rejected:** NamedTuple (xclim pattern) → requires Dataset conversion for NetCDF.

#### Recommendation 3: Dual params_dict Access Pattern

**Adopt:** Store params in **both** JSON string and individual attrs

```python
# JSON for full dict (CF-compliant)
ds.attrs["palmer_params"] = json.dumps({"alpha": 1.5, "beta": 0.8, ...})

# Individual attrs for direct access (ergonomic)
ds.attrs["palmer_alpha"] = 1.5
ds.attrs["palmer_beta"] = 0.8
ds.attrs["palmer_gamma"] = 0.6
ds.attrs["palmer_delta"] = 0.3
```

**Justification:**
- ✅ JSON preserves full dict structure (NetCDF round-trip)
- ✅ Individual attrs enable direct access without deserialization
- ✅ Small overhead (~4 extra attrs)

#### Recommendation 4: Validate AWC Dimensions

**Adopt:** Explicit validation that AWC lacks time dimension

```python
if isinstance(awc, xr.DataArray) and time_dim in awc.dims:
    raise ValueError(
        f"AWC must not have time dimension '{time_dim}'. "
        f"AWC is a soil property (spatially varying only). "
        f"Expected dims: spatial (e.g., lat, lon), got: {awc.dims}"
    )
```

**Justification:**
- ✅ Prevents silent incorrect computation
- ✅ Clear error message guides user correction
- ✅ Minimal performance overhead (single check)

#### Recommendation 5: Comprehensive CF Metadata

**Adopt:** Extended CF attributes including valid_range

```python
CF_METADATA["pdsi"] = {
    "long_name": "Palmer Drought Severity Index",
    "standard_name": "",  # Not in CF standard table yet
    "units": "",  # Dimensionless standardized index
    "valid_range": (-10.0, 10.0),
    "references": (
        "Palmer, W. C. (1965). Meteorological Drought. "
        "U.S. Weather Bureau Research Paper 45."
    ),
}
```

**Justification:**
- ✅ `valid_range` enables automated QA checks
- ✅ Matches CF Convention 1.10 recommendations
- ✅ Supports downstream tools (Panoply, ncview)

### 6.3 Performance Recommendations

#### Recommendation 6: Accept vectorize=True Overhead (Document Performance)

**Decision:** Use `vectorize=True` despite Python loop overhead

**Justification:**
- ✅ Palmer's sequential state **cannot parallelize time dimension** regardless of implementation
- ✅ Existing CLI already uses per-grid-cell multiprocessing (same pattern)
- ✅ Dask can still chunk spatial dimensions for distributed computation
- ✅ Simplicity > premature optimization

**Mitigation:**
- 📊 **Benchmark in Story 5.6:** Measure overhead vs multiprocessing baseline
- 📋 **Document in user guide:** Set expectations for Palmer performance characteristics
- 📋 **Future optimization:** If bottleneck, explore numba compilation (Epic 7+)

**Performance targets:**
- Acceptable: xarray path ≥ 80% speed of current multiprocessing CLI
- Good: xarray path ≥ 90% speed
- Excellent: xarray path ≥ 100% speed (parity)

#### Recommendation 7: Optimize Spatial Chunking for Dask

**Guidance:** For large gridded datasets, chunk along spatial dimensions, not time

```python
# Good chunking for Palmer
precip_da_chunked = precip_da.chunk({"time": -1, "lat": 50, "lon": 50})
#                                     ↑ time NOT chunked (sequential)
#                                              ↑ spatial chunks OK

# Bad chunking (will be slow)
precip_da_bad = precip_da.chunk({"time": 12, "lat": -1, "lon": -1})
#                                 ↑ chunked time breaks sequential computation
```

**Justification:**
- Palmer requires full time series per grid cell (month-to-month state)
- Chunking time forces redundant computation or incorrect results
- Spatial chunking parallelizes across independent grid cells

**Document:** Add chunking guidance to Palmer xarray user guide (Story 5.7).

### 6.4 Testing Recommendations

#### Recommendation 8: Comprehensive Test Coverage

**Adopt:** Multi-layered test strategy

| Test Type | Coverage | Priority |
|-----------|----------|----------|
| **Unit tests** | `palmer_xarray()` function correctness | P0 |
| — Scalar AWC | Uniform soil properties | P0 |
| — DataArray AWC | Spatially varying soil | P0 |
| — params_dict serialization | JSON round-trip | P0 |
| — CF metadata | All 4 variables have attrs | P0 |
| — AWC validation | Raises error if time dim | P0 |
| **Integration tests** | NumPy vs xarray equivalence | P0 |
| — 1D time series | Palmer(numpy) ≈ palmer_xarray(1D) | P0 |
| — 3D gridded | Shape (time, lat, lon) | P0 |
| — NetCDF round-trip | Write + read preserves data | P0 |
| **Type tests** | mypy --strict passes | P1 |
| **Performance tests** | Benchmark vs baseline | P2 |

**Estimated test lines:** 200 (unit) + 150 (integration) = 350 lines

### 6.5 Documentation Recommendations

#### Recommendation 9: Migration Guide for NumPy → xarray

**Create:** User guide showing migration patterns

```markdown
# Migrating Palmer from NumPy to xarray

## NumPy approach (existing)
```python
import numpy as np
from climate_indices import palmer

precip_np = np.array([...])  # 1D time series
pet_np = np.array([...])
pdsi, phdi, pmdi, z, params = palmer.pdsi(
    precip_np, pet_np, awc=2.5,
    data_start_year=1950,
    calibration_year_initial=1950,
    calibration_year_final=2020,
)
```

## xarray approach (new)
```python
import xarray as xr
from climate_indices import pdsi  # typed_public_api

# Load data (infers temporal params from time coord)
precip_da = xr.open_dataset("precip.nc")["precip"]
pet_da = xr.open_dataset("pet.nc")["pet"]

# Compute (returns Dataset with 4 variables)
ds = pdsi(precip_da, pet_da, awc=2.5)

# Access variables
pdsi_values = ds["pdsi"].values
phdi_values = ds["phdi"].values

# Access params
params = json.loads(ds.attrs["palmer_params"])
# OR
alpha = ds.attrs["palmer_alpha"]

# Write to NetCDF (CF-compliant)
ds.to_netcdf("palmer_indices.nc")
```
```

#### Recommendation 10: Performance Characteristics Documentation

**Create:** Section in user guide explaining Palmer's sequential nature

```markdown
# Palmer Performance Characteristics

## Why Palmer is Different from SPI/SPEI

Palmer drought indices use a **sequential water balance model** where each month depends on the previous month's soil moisture state. This means:

- ❌ **Cannot parallelize along time dimension** (unlike SPI/SPEI)
- ✅ **Can parallelize across spatial dimensions** (grid cells)
- ⚠️ **Python loop overhead** from `vectorize=True` is acceptable

## Expected Performance

For a 3D dataset (time, lat, lon):
- **Sequential:** Computes all timesteps for grid cell (0,0), then (0,1), etc.
- **Parallel (with Dask):** Computes multiple grid cells in parallel (spatial chunking)

**Benchmark (Story 5.6):** xarray path achieves ≥80% of multiprocessing CLI speed.

## Optimization Tips

1. **Chunk spatially, not temporally:**
   ```python
   precip_chunked = precip.chunk({"time": -1, "lat": 50, "lon": 50})
   ```

2. **Use Dask distributed for large grids:**
   ```python
   from dask.distributed import Client
   client = Client()  # Scales to cluster
   ds = pdsi(precip_chunked, pet_chunked, awc)
   ds.compute()  # Distributed execution
   ```
```

### 6.6 Future Work Recommendations

#### Recommendation 11: Monitor xarray Issue #1815

**Action:** Track xarray development for native multi-output `dask='parallelized'` support

**Timeline:** Check quarterly (Q2 2026, Q3 2026, ...)

**Trigger for refactor:** If Issue #1815 resolved, benchmark native multi-output vs stack/unpack workaround

**Estimated effort:** 20 hours to refactor if/when available

#### Recommendation 12: Consider Numba Compilation (Epic 7+)

**Investigation:** Explore numba JIT compilation for Palmer core loop

**Potential benefit:** 10-100x speedup for water balance calculations

**Challenges:**
- Numba requires pure NumPy (no Python objects)
- Complex backtracking logic may not be numba-compatible
- Would need to refactor `palmer.py` core functions

**Decision point:** After Epic 6 (if Palmer performance is bottleneck)

#### Recommendation 13: Generalize to @xarray_multi_adapter (If 2nd Multi-Output Index)

**Trigger:** When a 2nd multi-output index is planned (e.g., compound drought index)

**Approach:** Extract common patterns from `palmer_xarray()`:
1. Multi-input alignment
2. Temporal parameter inference
3. Stack/unpack pattern for multi-output
4. Dataset assembly with per-variable CF metadata

**Estimated effort:** 40 hours to design, implement, test decorator

**Benefit:** Reusable infrastructure for all future multi-output indices

### 6.7 Final Recommendation Summary

| Category | Recommendation | Priority | Epic/Story |
|----------|----------------|----------|------------|
| **Architecture** | Pattern C (manual wrapper) | P0 | Epic 5 |
| **Multi-output** | Stack/unpack workaround | P0 | Story 5.1 |
| **Return type** | xr.Dataset with 4 variables | P0 | Story 5.1 |
| **params_dict** | Dual access (JSON + individual attrs) | P0 | Story 5.1 |
| **Validation** | AWC dimension check | P0 | Story 5.1 |
| **CF metadata** | Extended with valid_range | P0 | Story 5.2 |
| **Performance** | Accept vectorize=True, document | P0 | Story 5.6 |
| **Testing** | 350 lines (unit + integration) | P0 | Stories 5.4-5.5 |
| **Documentation** | Migration guide + performance guide | P1 | Story 5.7 |
| **Future** | Monitor Issue #1815 | P2 | Ongoing |
| **Future** | Numba compilation exploration | P3 | Epic 7+ |
| **Future** | Extract @xarray_multi_adapter | P3 | When triggered |

---

## 7. Conclusion

This research establishes that **climate_indices will be the first Python climate library to implement Palmer drought indices with xarray multi-output support**. The recommended approach (Pattern C: manual `palmer_xarray()` wrapper) balances implementation simplicity (~150 core lines) with production-ready features (CF metadata, type safety, comprehensive testing).

**Key technical achievements:**
1. ✅ Multi-output workaround via stack/unpack pattern (bypasses `dask='parallelized'` limitation)
2. ✅ Spatial parameter handling (AWC with no time dimension)
3. ✅ Type-safe API with `@overload` signatures (numpy→tuple, xarray→Dataset)
4. ✅ CF-compliant Dataset with per-variable metadata
5. ✅ params_dict dual-access pattern (JSON + individual attrs)

**Implementation scope:** ~770 lines across 8 files (Stories 5.1-5.8)

**Risk level:** Low-Medium (all risks mitigated, no blockers)

**Estimated timeline:** Epic 5 (4-6 weeks for implementation + testing + documentation)

**Next steps:**
1. Present research findings to stakeholders
2. Approve Pattern C approach for Epic 5
3. Create detailed Story tickets (5.1-5.8)
4. Begin implementation with Story 5.1 (core `palmer_xarray()` function)

---

---

## Source Documentation

### Web Research

**xarray apply_ufunc Documentation:**
- [xarray.apply_ufunc API Reference](https://docs.xarray.dev/en/stable/generated/xarray.apply_ufunc.html)
- [Handling complex output (tutorial)](https://tutorial.xarray.dev/advanced/apply_ufunc/complex-output-numpy.html)
- [Applying unvectorized functions](https://docs.xarray.dev/en/stable/examples/apply_ufunc_vectorize_1d.html)
- [Core dimensions (tutorial)](https://tutorial.xarray.dev/advanced/apply_ufunc/core-dimensions.html)

**Dask Integration:**
- [Parallel Computing with Dask (xarray user guide)](https://docs.xarray.dev/en/stable/user-guide/dask.html)
- [Handling dask arrays (tutorial)](https://tutorial.xarray.dev/advanced/apply_ufunc/dask_apply_ufunc.html)

**Known Issues:**
- [Issue #1815: apply_ufunc(dask='parallelized') with multiple outputs](https://github.com/pydata/xarray/issues/1815)
- [Issue #1699: output_dtypes for datasets](https://github.com/pydata/xarray/issues/1699)
- [Issue #2817: mix of chunked and non-chunked args](https://github.com/pydata/xarray/issues/2817)

### Code Analysis

- `src/climate_indices/palmer.py` — 5-tuple return signature at line 830
- `src/climate_indices/xarray_adapter.py` — Single-output decorator pattern
- `src/climate_indices/__main__.py:1366-1430` — Existing multiprocessing dispatch
- `src/climate_indices/typed_public_api.py` — Existing @overload patterns for SPI/SPEI

### Climate Library Survey

- [xclim GitHub](https://github.com/Ouranosinc/xclim) — Indicator framework architecture
- [xclim Issue #131: Drought indices](https://github.com/Ouranosinc/xclim/issues/131) — Palmer discussion
- [xclim core.indicator source](https://xclim.readthedocs.io/en/stable/_modules/xclim/core/indicator.html) — Multi-output NamedTuple pattern
- [standard_precip GitHub](https://github.com/e-baumer/standard_precip) — Pandas-only SPI/SPEI
- [pyet documentation](https://github.com/pyet-org/pyet) — Simple functional xarray API
- [Climate Indices Toolbox discussion](https://discourse.pangeo.io/t/climate-indices-toolbox/2048) — Pangeo community

### Best Practices & Performance

- [xarray Weather and Climate Data](https://docs.xarray.dev/en/stable/user-guide/weather-climate.html) — 2025 best practices
- [Parallel Computing with Dask](https://docs.xarray.dev/en/stable/user-guide/dask.html) — Performance optimization
- [Xarray with Dask Arrays examples](https://examples.dask.org/xarray.html) — Chunk size optimization

---

**Research Status:** ✅ **COMPLETE**
**Document Version:** 1.0
**Completion Date:** 2026-02-15
**Total Steps:** 6/6 ✅
