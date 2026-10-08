//! Precipitation Concentration Index (PCI) of one year of daily rainfall: the
//! year's monthly totals and their concentration ratio.
//!
//! Reference: Oliver, J. E. (1980), "Monthly precipitation distribution: a
//! comparative index", The Professional Geographer 32(3), 300-309
//! (doi:10.1111/j.0033-0124.1980.00300.x), as `docs/algorithm-reference.md`
//! cites it, and `docs/algorithms.md` § Precipitation Concentration Index. The
//! Python source is the `np.add.reduceat` and ratio lines of `indices.pci`,
//! which stays the reference oracle. The year-length and missing-value
//! validation, the masked-input early return, and logging all stay in Python,
//! which dispatches only a complete year of float64 values with no missing day.

use ndarray::ArrayView1;

use crate::reduction::pairwise_sum;

/// Day-of-year start of each calendar month, keyed by the number of days in the
/// year, exactly as `indices._PCI_MONTH_STARTS` holds them.
const MONTH_STARTS_365: [usize; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
const MONTH_STARTS_366: [usize; 12] = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335];

/// The precipitation concentration index of one year of daily rainfall, or
/// `None` when the year is neither 365 nor 366 days long.
///
/// - Python source: the `np.add.reduceat(rainfall, month_starts)` and
///   `(np.sum(monthly_totals**2) / (np.sum(monthly_totals) ** 2)) * 100` lines of
///   `indices.pci`.
/// - Inputs: `rainfall`, one value per day of the year in calendar order, with
///   no missing day.
/// - Outputs: `100 * Σ P_m² / (Σ P_m)²`, the NaN a year without rain gives
///   (`0 / 0`, as NumPy computes it).
/// - Numerics: `reduceat` seeds each month's total with its first day and adds
///   NumPy's pairwise sum of the remaining days. Both annual sums use the same
///   pairwise grouping as `np.sum`.
pub fn pci(rainfall: ArrayView1<'_, f64>) -> Option<f64> {
    let month_starts: &[usize; 12] = match rainfall.len() {
        365 => &MONTH_STARTS_365,
        366 => &MONTH_STARTS_366,
        _ => return None,
    };

    let mut monthly_totals = [0.0_f64; 12];
    for (month, &start) in month_starts.iter().enumerate() {
        let end = month_starts
            .get(month + 1)
            .copied()
            .unwrap_or(rainfall.len());
        monthly_totals[month] = rainfall[start] + pairwise_sum(start + 1..end, |day| rainfall[day]);
    }

    let total = pairwise_sum(0..12, |month| monthly_totals[month]);
    let squared = pairwise_sum(0..12, |month| monthly_totals[month] * monthly_totals[month]);
    Some((squared / (total * total)) * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array, s};

    #[test]
    fn a_year_of_rain_in_one_month_is_maximally_concentrated() {
        let mut january_only = Array1::<f64>::zeros(365);
        for day in &mut january_only.slice_mut(s![..31]) {
            *day = 1.0;
        }
        assert_eq!(pci(january_only.view()), Some(100.0));
    }

    #[test]
    fn february_29_belongs_to_february_of_a_366_day_year() {
        let mut february_29_only = Array1::<f64>::zeros(366);
        february_29_only[58] = 1.0;
        february_29_only[59] = 1.0;
        assert_eq!(pci(february_29_only.view()), Some(100.0));
        february_29_only[58] = 0.0;
        february_29_only[60] = 1.0;
        assert_eq!(pci(february_29_only.view()), Some(50.0));
    }

    #[test]
    fn uniform_rainfall_over_the_year_concentrates_only_by_month_length() {
        let uniform = Array1::<f64>::ones(365);
        let value = pci(uniform.view()).unwrap();
        assert!(
            value > 8.0 && value < 9.0,
            "PCI of uniform rain was {value}"
        );
    }

    #[test]
    fn a_year_without_rain_is_nan() {
        assert!(pci(Array1::<f64>::zeros(366).view()).unwrap().is_nan());
    }

    #[test]
    fn a_year_of_the_wrong_length_has_no_index() {
        assert_eq!(pci(array![1.0, 2.0, 3.0].view()), None);
    }
}
