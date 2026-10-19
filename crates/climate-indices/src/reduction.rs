//! NumPy's float64 pairwise reduction, shared by PNP and PCI.
//!
//! Port of DOUBLE_pairwise_sum in NumPy 2.4.2's
//! numpy/_core/src/umath/loops_utils.h.src (PW_BLOCKSIZE = 128).
//! Copyright (c) 2005-2025, NumPy Developers. BSD-3-Clause; see LICENSE.

use std::ops::Range;

/// Preserve NumPy's eight-lane grouping and recursive block boundaries.
/// The indexed accessor lets PNP replace NaNs with zero without dropping slots.
pub(crate) fn pairwise_sum(range: Range<usize>, value: impl Fn(usize) -> f64 + Copy) -> f64 {
    let length = range.len();
    if length < 8 {
        return range.fold(-0.0, |sum, index| sum + value(index));
    }
    if length <= 128 {
        let mut lanes: [f64; 8] = std::array::from_fn(|lane| value(range.start + lane));
        let full_end = range.end - length % 8;
        for start in (range.start + 8..full_end).step_by(8) {
            for (lane, sum) in lanes.iter_mut().enumerate() {
                *sum += value(start + lane);
            }
        }
        let sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        return (full_end..range.end).fold(sum, |sum, index| sum + value(index));
    }
    let half = length / 2;
    let middle = range.start + half - half % 8;
    pairwise_sum(range.start..middle, value) + pairwise_sum(middle..range.end, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouping_preserves_small_terms_around_cancellation() {
        let values = [1e16, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1e16];
        assert_eq!(pairwise_sum(0..values.len(), |index| values[index]), 4.0);
        assert_eq!(pairwise_sum(0..0, |_| unreachable!()), -0.0);
    }

    #[test]
    fn recursive_blocks_and_their_remainders_cover_every_value() {
        for length in [7, 8, 12, 128, 129, 256, 1025] {
            assert_eq!(pairwise_sum(0..length, |_| 1.0), length as f64);
        }
    }
}
