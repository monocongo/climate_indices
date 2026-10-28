//! Shared sorting and two-sided Kolmogorov-Smirnov D statistic.

pub(crate) fn d_statistic(sample: &mut [f64], cdf: impl Fn(f64) -> f64) -> f64 {
    if sample.is_empty() {
        return f64::NAN;
    }
    sample.sort_by(f64::total_cmp);
    let size = sample.len() as f64;
    let mut upper = f64::NEG_INFINITY;
    let mut lower = f64::NEG_INFINITY;
    for (index, &value) in sample.iter().enumerate() {
        let probability = cdf(value);
        if probability.is_nan() {
            return f64::NAN;
        }
        let rank = (index + 1) as f64;
        upper = upper.max(rank / size - probability);
        lower = lower.max(probability - (rank - 1.0) / size);
    }
    upper.max(lower)
}
