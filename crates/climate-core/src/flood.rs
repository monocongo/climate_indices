//! Flood-family kernels, starting with effective precipitation.
//!
//! Ports of `climate_indices.flood._pe`, which stays the parity oracle. Python
//! keeps the validation and the layout; the kernel takes the prepared
//! time-first float64 block.
//!
//! The operation order is the Python one: effective precipitation follows
//! `scipy.ndimage.correlate1d`'s general (non-symmetric) loop, which starts from
//! the newest day's term and then adds the window from its oldest day.

/// The correlation filter of Byun and Wilhite (1999), Eq. 2, oldest day first.
///
/// `w_m = sum(n=m..D, 1/n)` weights the day `m - 1` days before the window's
/// last one, so the oldest day carries `1 / D` and the newest `H_D`. Built
/// exactly as `np.cumsum((1.0 / np.arange(1, D + 1))[::-1])` (the Python weights
/// reversed for `correlate1d`): a running sum from `1 / D` up to `1`.
#[allow(dead_code)]
fn harmonic_filter(duration: usize) -> Vec<f64> {
    let mut running = 0.0;
    (1..=duration)
        .rev()
        .map(|n| {
            running += 1.0 / n as f64;
            running
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative closeness for closed-form references, scaled by `tol`.
    fn close(actual: f64, expected: f64, tol: f64) -> bool {
        (actual - expected).abs() <= tol * expected.abs().max(1.0)
    }

    #[test]
    fn the_filter_has_the_harmonic_endpoint_weights() {
        let filter = harmonic_filter(4);
        // oldest day 1/D, newest day H_D
        assert_eq!(filter[0], 0.25);
        assert!(close(filter[3], 1.0 + 0.5 + 1.0 / 3.0 + 0.25, 1e-15));
        assert!(filter.windows(2).all(|pair| pair[0] < pair[1]));
    }
}
