//! Trailing Timescale sums and means over borrowed `(steps, cells)` float64 views.
//!
//! This ports the Spatial Block reduction in Python's `compute.sum_to_scale`.
//! Callers retain calendar preparation, Timescale selection, clipping, and fitting.
//! Each window is reduced independently: no running sum or prefix subtraction can
//! carry cancellation, NaN, infinity, or overflow into a later complete window.

use ndarray::{Array2, ArrayView2, Zip};

use crate::{ClimateError, reduction::pairwise_sum};

/// Sum each trailing `scale`-step window independently for every cell.
///
/// # Numerical contract
///
/// Inputs borrow `(steps, cells)` values in any units and may be strided.
/// The owned output retains that shape and those units. The first `scale - 1`
/// positions are NaN; any window containing NaN is NaN. Infinities, overflow,
/// subnormals, and cancellation follow ordinary float64 arithmetic without
/// skipping values. A window longer than the record gives an all-NaN result;
/// empty axes stay empty. `scale == 1` copies the input, including signed zeros.
/// Zero `scale` returns [`ClimateError::EmptyPeriod`].
///
/// The operation order matches NumPy's C-contiguous Spatial Block oracle:
/// multiple cells add from oldest to newest, starting at positive zero; a
/// singleton cell uses NumPy's pairwise reduction, which starts at `-0.0`, and
/// the kernel adds that result to positive zero.
/// This order does not depend on the Rust input's strides. NumPy's 1-D convolution
/// has a different, potentially BLAS-dependent order for larger windows; it is
/// not this kernel's oracle. Work is O(steps * cells * scale), with only the output
/// allocated. See the `Measured accuracy` section in [`crate`], entries
/// `aggregation_monthly`, `aggregation_spatial_block`, `aggregation_signed_monthly`,
/// and `aggregation_signed_spatial_block`, for isolated sum measurements.
///
/// ```
/// use climate_indices::aggregation::trailing_sum;
/// use ndarray::array;
/// let values = array![[3.0], [4.0], [6.0], [2.0]];
/// let sums = trailing_sum(values.view(), 3)?;
/// assert!(sums[[1, 0]].is_nan());
/// assert_eq!(sums[[2, 0]], 13.0);
/// assert_eq!(sums[[3, 0]], 12.0);
/// # Ok::<(), climate_indices::ClimateError>(())
/// ```
pub fn trailing_sum(
    values: ArrayView2<'_, f64>,
    scale: usize,
) -> Result<Array2<f64>, ClimateError> {
    if scale == 0 {
        return Err(ClimateError::EmptyPeriod { argument: "scale" });
    }
    if scale == 1 {
        return Ok(values.to_owned());
    }
    let (steps, cells) = values.dim();
    let mut output = Array2::from_elem((steps, cells), f64::NAN);
    if scale > steps || cells == 0 {
        return Ok(output);
    }
    for end in scale..=steps {
        let start = end - scale;
        if cells == 1 {
            output[[end - 1, 0]] = 0.0 + pairwise_sum(start..end, |step| values[[step, 0]]);
        } else {
            let mut sums = output.row_mut(end - 1);
            sums.fill(0.0);
            for step in start..end {
                Zip::from(&mut sums)
                    .and(values.row(step))
                    .for_each(|sum, &value| *sum += value);
            }
        }
    }
    Ok(output)
}

/// Average each trailing window as its [`trailing_sum`] divided by `scale`.
///
/// # Numerical contract
///
/// Shape, errors, operation order, missing values, and calendar responsibilities
/// are those of [`trailing_sum`]. Output units match input units. Division follows
/// summation, so an overflowing sum remains infinite even if a mathematically
/// exact mean would be finite. `scale == 1` preserves signed zeros without division.
/// Python's `compute.prepare_scaled` retains its own division and does not call
/// this function; it must not divide a mean twice.
/// The `Measured accuracy` entry `aggregation_spatial_block` in [`crate`] measures
/// the sum, not the subsequent division. Independent Rust tests check the mean.
pub fn trailing_mean(
    values: ArrayView2<'_, f64>,
    scale: usize,
) -> Result<Array2<f64>, ClimateError> {
    let mut output = trailing_sum(values, scale)?;
    if scale > 1 {
        output.mapv_inplace(|value| value / scale as f64);
    }
    Ok(output)
}
