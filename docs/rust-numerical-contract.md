# Rust numerical contract

Each public Rust kernel documents its algorithm, units, shapes, calendar assumptions,
and missing-value behavior in its `Numerical contract` section.
Build the API reference with `cargo doc -p climate-core --no-deps`.
The crate documentation includes the same generated accuracy page as this reference.

```{include} ../crates/climate-core/NUMERICS.md
:heading-offset: 1
```
