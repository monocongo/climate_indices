//! Sample L-moments, the input to the Pearson Type III and generalized logistic fits.
//!
//! Reference: Hosking (1990), "L-moments: analysis and estimation of
//! distributions using linear combinations of order statistics", J. R. Statist.
//! Soc. B 52(1), 105-124; the algorithm is `SAMLMR` from Hosking's IBM Research
//! Report RC20525 (1996), cut down to the first three L-moments.
//!
//! The Python implementation in `climate_indices.lmoments` stays the reference
//! oracle; `tests/test_native_parity.py` checks these kernels against it at
//! `rtol = atol = 1e-10`.

/// Fewest non-missing values a sample needs for L-moments, `MIN_VALUES_FOR_LMOMENTS`.
pub(crate) const MIN_VALUES: usize = 4;

/// The first two sample L-moments and the L-skewness of one column, or `None`
/// when the column cannot supply them.
///
/// - Python source: `lmoments._estimate_lmoments_spatial`, the cell-axis form of
///   `_estimate_lmoments`.
/// - Inputs: one calibration column, zeros and negatives included.
/// - Outputs: `[lambda_1, lambda_2, tau_3]`, with `tau_3 = lambda_3 / lambda_2`.
/// - NaN semantics: NaN is missing and excluded. Fewer than [`MIN_VALUES`]
///   non-missing values, or a zero second sum, gives `None`; Python marks those
///   cells invalid and zeroes their L-moments.
/// - Numerics: the values are sorted ascending and the weighted sums
///   accumulate in rank order, as NumPy does, with the same operation order.
///   An infinite value makes the L-moments NaN, which no validity test accepts;
///   the single-series Python fit would instead let the NaN through, so Python
///   keeps those inputs.
pub(crate) fn sample_lmoments<'a>(column: impl IntoIterator<Item = &'a f64>) -> Option<[f64; 3]> {
    let mut sorted: Vec<f64> = column
        .into_iter()
        .copied()
        .filter(|x| !x.is_nan())
        .collect();
    if sorted.len() < MIN_VALUES {
        return None;
    }
    sorted.sort_by(f64::total_cmp);

    let mut sums = [0.0_f64; 3];
    for (index, &value) in sorted.iter().enumerate() {
        let rank = index as f64;
        sums[0] += value;
        let first_term = value * rank;
        sums[1] += first_term;
        sums[2] += first_term * (rank - 1.0);
    }

    let count = sorted.len() as f64;
    let mut z = count;
    sums[0] /= z;
    let mut y = count - 1.0;
    z *= y;
    sums[1] /= z;
    y -= 1.0;
    z *= y;
    sums[2] /= z;

    // unbiased probability-weighted moments -> L-moments (Hosking's SAMLMR)
    let mut k = 3_usize;
    let mut p0 = -1.0_f64;
    for _ in 0..2 {
        let ak = k as f64;
        p0 = -p0;
        let mut p = p0;
        let mut temp = p * sums[0];
        for (i, &sum) in sums.iter().enumerate().take(k).skip(1) {
            let ai = i as f64;
            p = -p * (ak + ai - 1.0) * (ak - ai) / (ai * ai);
            temp += p * sum;
        }
        sums[k - 1] = temp;
        k -= 1;
    }

    (sums[1] != 0.0).then(|| [sums[0], sums[1], sums[2] / sums[1]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_a_hand_computed_sample() {
        // 1..=5: lambda_1 = 3, lambda_2 = 1, lambda_3 = 0
        let [l1, l2, t3] = sample_lmoments(&[3.0, 1.0, 5.0, 2.0, 4.0]).unwrap();
        assert!((l1 - 3.0).abs() < 1e-15);
        assert!((l2 - 1.0).abs() < 1e-15);
        assert!(t3.abs() < 1e-15);
    }

    #[test]
    fn missing_values_are_excluded_and_short_samples_have_none() {
        let with_gaps = [f64::NAN, 2.0, 1.0, f64::NAN, 3.0, 4.0];
        assert_eq!(
            sample_lmoments(&with_gaps),
            sample_lmoments(&[1.0, 2.0, 3.0, 4.0])
        );
        assert!(sample_lmoments(&[1.0, 2.0, 3.0, f64::NAN]).is_none());
    }

    #[test]
    fn a_constant_sample_has_none() {
        assert!(sample_lmoments(&[2.0; 6]).is_none());
    }
}
