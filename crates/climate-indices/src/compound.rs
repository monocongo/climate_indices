//! Deterministic metric evaluation for monthly compound bootstrap resamples.
//!
//! This ports the two `_metrics` calls in Python's `compound._bootstrap._resample`.
//! Python retains all random draws, support selection, point estimates, and inference.
//! Counts use exact `i64` arithmetic. Rates divide counts before multiplying source
//! marginals in source order, as NumPy does. There are no random numbers or special functions.

use ndarray::{Array2, ArrayView2, ArrayView3};

use crate::ClimateError;

/// The metric evaluated on each resampled occurrence table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Statistic {
    /// The frequency of simultaneous occurrences in every source.
    JointProbability,
    /// Joint probability divided by the product of marginal probabilities.
    LikelihoodMultiplicationFactor,
    /// The second source's occurrence probability given the first source.
    ConditionalProbability,
}

/// Calendar grouping for monthly occurrence tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GroupBy {
    /// Evaluate the twelve calendar months separately.
    Month,
    /// Evaluate all calendar months together.
    Pooled,
}

fn rate(numerator: i64, denominator: i64) -> f64 {
    if denominator == 0 {
        f64::NAN
    } else {
        numerator as f64 / denominator as f64
    }
}

fn metrics(
    rows: ArrayView3<'_, bool>,
    index: impl Fn(usize, usize) -> usize,
    statistic: Statistic,
    groupby: GroupBy,
) -> ([f64; 12], [f64; 12]) {
    let (sources, years, _) = rows.dim();
    let mut marginal = [[0_i64; 12]; 3];
    let mut joint = [0_i64; 12];
    for year in 0..years {
        for month in 0..12 {
            let group = if groupby == GroupBy::Month { month } else { 0 };
            let mut all = true;
            for source in 0..sources {
                let occurs = rows[[source, index(source, year), month]];
                marginal[source][group] += i64::from(occurs);
                all &= occurs;
            }
            joint[group] += i64::from(all);
        }
    }
    let groups = if groupby == GroupBy::Month { 12 } else { 1 };
    let n = if groupby == GroupBy::Month {
        years
    } else {
        years * 12
    } as i64;
    let mut estimates = [f64::NAN; 12];
    let mut excesses = [f64::NAN; 12];
    for group in 0..groups {
        let joint_rate = rate(joint[group], n);
        let mut independent = 1.0;
        for counts in marginal.iter().take(sources) {
            independent *= rate(counts[group], n);
        }
        let (estimate, excess) = match statistic {
            Statistic::ConditionalProbability => {
                let conditional = rate(joint[group], marginal[0][group]);
                (conditional, conditional - rate(marginal[1][group], n))
            }
            Statistic::JointProbability => (joint_rate, joint_rate - independent),
            Statistic::LikelihoodMultiplicationFactor => (
                if independent > 0.0 {
                    joint_rate / independent
                } else {
                    f64::NAN
                },
                joint_rate - independent,
            ),
        };
        estimates[group] = estimate;
        excesses[group] = excess;
    }
    (estimates, excesses)
}

/// Evaluate paired estimates and independence-null excesses from supplied year indices.
///
/// # Numerical contract
///
/// `rows` has shape `(sources, years, 12)` with two or three Boolean occurrence
/// channels. Conditional probability requires exactly two channels, given then target.
/// `paired_years` has shape `(resamples, years)`; `null_years` has shape
/// `(resamples, sources, years)`. Each index selects a whole year without changing
/// calendar-month phase. Inputs can be strided. No input is modified.
///
/// Both outputs have shape `(resamples, 12)` for [`GroupBy::Month`] or
/// `(resamples, 1)` for [`GroupBy::Pooled`]. The first contains paired estimates.
/// The second contains null excesses: joint minus the product of marginals, or
/// conditional minus target frequency. An undefined metric gives NaN, including
/// a null LMF with a zero marginal product. An empty year axis gives NaNs; an
/// empty resample axis gives empty outputs.
///
/// Counts are exact `i64` sums. Each rate divides a count by its denominator
/// before the product over sources, in source order, as Python's `_safe_rate` and
/// `np.prod(axis=0)` do, so results match Python bit for bit. See the measured
/// workflow row `compound_bootstrap` in the crate's `Measured accuracy` section.
///
/// # Errors
///
/// Returns [`ClimateError::ShapeMismatch`] for an invalid source count, month
/// count, or index-array axis, and [`ClimateError::IndexOutOfRange`] for a year
/// index outside `[0, years)`. All indices are checked before evaluation.
pub fn resample_metrics(
    rows: ArrayView3<'_, bool>,
    paired_years: ArrayView2<'_, i64>,
    null_years: ArrayView3<'_, i64>,
    statistic: Statistic,
    groupby: GroupBy,
) -> Result<(Array2<f64>, Array2<f64>), ClimateError> {
    let (sources, years, months) = rows.dim();
    let resamples = paired_years.nrows();
    let expected_sources = if statistic == Statistic::ConditionalProbability || sources != 3 {
        2
    } else {
        3
    };
    for (argument, expected, actual) in [
        ("rows sources", expected_sources, sources),
        ("rows months", 12, months),
        ("paired_years years", years, paired_years.ncols()),
        ("null_years resamples", resamples, null_years.dim().0),
        ("null_years sources", sources, null_years.dim().1),
        ("null_years years", years, null_years.dim().2),
    ] {
        if actual != expected {
            return Err(ClimateError::ShapeMismatch {
                argument,
                expected,
                actual,
            });
        }
    }
    for (argument, indices) in [
        ("paired_years", paired_years.into_dyn()),
        ("null_years", null_years.into_dyn()),
    ] {
        for &value in indices {
            if value < 0 || value as usize >= years {
                return Err(ClimateError::IndexOutOfRange {
                    argument,
                    value,
                    minimum: 0,
                    maximum: years as i64 - 1,
                });
            }
        }
    }
    let groups = if groupby == GroupBy::Month { 12 } else { 1 };
    let mut paired = Array2::from_elem((resamples, groups), f64::NAN);
    let mut null = paired.clone();
    for resample in 0..resamples {
        let (estimates, _) = metrics(
            rows,
            |_, year| paired_years[[resample, year]] as usize,
            statistic,
            groupby,
        );
        let (null_metrics, excesses) = metrics(
            rows,
            |source, year| null_years[[resample, source, year]] as usize,
            statistic,
            groupby,
        );
        for group in 0..groups {
            paired[[resample, group]] = estimates[group];
            if null_metrics[group].is_finite() {
                null[[resample, group]] = excesses[group];
            }
        }
    }
    Ok((paired, null))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, Axis, Slice, array};

    #[test]
    fn hand_computed_contingency_table() {
        // Four years: A=3, B=2, A&B=2. Independence is (3/4)*(2/4)=3/8.
        let rows = Array3::from_shape_fn(
            (2, 4, 12),
            |(source, year, _)| {
                if source == 0 { year < 3 } else { year < 2 }
            },
        );
        let paired = array![[0, 1, 2, 3]];
        let null = Array3::from_shape_fn((1, 2, 4), |(_, _, year)| year as i64);
        for groupby in [GroupBy::Month, GroupBy::Pooled] {
            for (statistic, estimate, excess) in [
                (Statistic::JointProbability, 0.5, 0.125),
                (
                    Statistic::LikelihoodMultiplicationFactor,
                    0.5 / 0.375,
                    0.125,
                ),
                (
                    Statistic::ConditionalProbability,
                    2.0 / 3.0,
                    2.0 / 3.0 - 0.5,
                ),
            ] {
                let (p, n) =
                    resample_metrics(rows.view(), paired.view(), null.view(), statistic, groupby)
                        .unwrap();
                assert!(p.iter().all(|&v| v == estimate));
                assert!(n.iter().all(|&v| v == excess));
            }
        }
    }

    #[test]
    fn three_sources_and_month_phase_with_strided_inputs() {
        let rows = Array3::from_shape_fn((3, 4, 24), |(source, year, month)| {
            year <= source && month == 0
        });
        let paired = array![[3, 0, 2, 1]];
        let null =
            Array3::from_shape_fn((1, 3, 4), |(_, source, year)| ((year + source) % 4) as i64);
        let (p, n) = resample_metrics(
            rows.slice_axis(Axis(2), Slice::new(0, None, 2)),
            paired.view(),
            null.view(),
            Statistic::JointProbability,
            GroupBy::Month,
        )
        .unwrap();
        assert_eq!(p[[0, 0]], 0.25);
        assert_eq!(n[[0, 0]], 0.25 - 0.25 * 0.5 * 0.75);
        assert!(
            p.slice_axis(Axis(1), Slice::from(1..))
                .iter()
                .all(|&v| v == 0.0)
        );
    }

    #[test]
    fn zero_denominators_and_empty_axes() {
        for years in [0, 3] {
            let rows = Array3::from_elem((2, years, 12), false);
            let paired = Array2::zeros((2, years));
            let null = Array3::zeros((2, 2, years));
            for statistic in [
                Statistic::ConditionalProbability,
                Statistic::LikelihoodMultiplicationFactor,
            ] {
                let (p, n) = resample_metrics(
                    rows.view(),
                    paired.view(),
                    null.view(),
                    statistic,
                    GroupBy::Month,
                )
                .unwrap();
                assert!(p.iter().chain(n.iter()).all(|v| v.is_nan()));
            }
            let (p, n) = resample_metrics(
                rows.view(),
                paired.slice_axis(Axis(0), Slice::from(..0)),
                null.slice_axis(Axis(0), Slice::from(..0)),
                Statistic::JointProbability,
                GroupBy::Pooled,
            )
            .unwrap();
            assert_eq!(p.dim(), (0, 1));
            assert_eq!(n.dim(), (0, 1));
        }
    }

    #[test]
    fn rejects_invalid_shapes_and_indices() {
        let rows = Array3::from_elem((2, 3, 12), false);
        let paired = Array2::zeros((1, 3));
        let null = Array3::zeros((1, 2, 3));
        fn evaluate(
            r: ArrayView3<'_, bool>,
            p: ArrayView2<'_, i64>,
            n: ArrayView3<'_, i64>,
        ) -> Result<(Array2<f64>, Array2<f64>), ClimateError> {
            resample_metrics(r, p, n, Statistic::JointProbability, GroupBy::Month)
        }
        for bad in [
            Array3::from_elem((1, 3, 12), false),
            Array3::from_elem((4, 3, 12), false),
            Array3::from_elem((2, 3, 11), false),
        ] {
            assert!(matches!(
                evaluate(bad.view(), paired.view(), null.view()),
                Err(ClimateError::ShapeMismatch { .. })
            ));
        }
        assert!(matches!(
            evaluate(
                rows.view(),
                paired.slice_axis(Axis(1), Slice::from(..2)),
                null.view()
            ),
            Err(ClimateError::ShapeMismatch { .. })
        ));
        for bad in [
            Array3::zeros((2, 2, 3)),
            Array3::zeros((1, 3, 3)),
            Array3::zeros((1, 2, 2)),
        ] {
            assert!(matches!(
                evaluate(rows.view(), paired.view(), bad.view()),
                Err(ClimateError::ShapeMismatch { .. })
            ));
        }
        for value in [-1, 3, i64::MAX] {
            let mut bad_paired = paired.clone();
            bad_paired[[0, 0]] = value;
            assert!(matches!(
                evaluate(rows.view(), bad_paired.view(), null.view()),
                Err(ClimateError::IndexOutOfRange { .. })
            ));
            let mut bad_null = null.clone();
            bad_null[[0, 1, 2]] = value;
            assert!(matches!(
                evaluate(rows.view(), paired.view(), bad_null.view()),
                Err(ClimateError::IndexOutOfRange { .. })
            ));
        }
    }
}
