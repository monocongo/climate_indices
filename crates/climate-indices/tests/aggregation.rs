use climate_indices::{
    ClimateError,
    aggregation::{trailing_mean, trailing_sum},
};
use ndarray::{Array2, Axis, Slice, array};

#[test]
fn independent_sum_mean_and_calendar_boundary() {
    let input = array![[3., 30.], [4., 40.], [6., 60.], [2., 20.], [1., 10.]];
    let sums = trailing_sum(input.view(), 3).unwrap();
    let means = trailing_mean(input.view(), 3).unwrap();
    assert_eq!(sums.dim(), input.dim());
    for row in 0..2 {
        assert!(sums.row(row).iter().all(|x| x.is_nan()));
    }
    for (row, expected) in [13., 12., 9.].into_iter().enumerate() {
        assert_eq!(sums[[row + 2, 0]], expected);
        assert_eq!(sums[[row + 2, 1]], expected * 10.);
        assert_eq!(means[[row + 2, 0]], expected / 3.);
    }
}

#[test]
fn missing_infinite_and_overflow_windows_recover() {
    for marker in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let input = array![[marker, marker], [1., 1.], [2., 2.], [3., 3.]];
        let result = trailing_sum(input.view(), 2).unwrap();
        assert!(result[[0, 0]].is_nan());
        if marker.is_nan() {
            assert!(result[[1, 0]].is_nan());
        } else {
            assert_eq!(result[[1, 0]], marker);
        }
        assert_eq!(result[[2, 0]], 3.);
        assert_eq!(result[[3, 0]], 5.);
    }
    let input = array![[f64::MAX], [f64::MAX], [1.], [2.]];
    let result = trailing_sum(input.view(), 2).unwrap();
    assert_eq!(result[[1, 0]], f64::INFINITY);
    assert_eq!(result[[3, 0]], 3.);
}

#[test]
fn grouping_matches_numpy_block_reductions() {
    let input = [1e16, 1., 1., 1., 1., 1., 1., -1e16];
    let singleton = Array2::from_shape_vec((8, 1), input.to_vec()).unwrap();
    assert_eq!(trailing_sum(singleton.view(), 8).unwrap()[[7, 0]], 4.);
    let block = Array2::from_shape_fn((8, 2), |(row, _)| input[row]);
    assert_eq!(trailing_sum(block.view(), 8).unwrap()[[7, 0]], 0.);
}

#[test]
fn shapes_identity_invalid_and_strided_views() {
    for shape in [(0, 2), (3, 0), (1, 1)] {
        let input = Array2::from_elem(shape, -0.0);
        assert_eq!(trailing_sum(input.view(), 1).unwrap(), input);
        assert_eq!(trailing_mean(input.view(), 1).unwrap(), input);
        let result = trailing_sum(input.view(), usize::MAX).unwrap();
        assert_eq!(result.dim(), shape);
        assert!(result.iter().all(|x| x.is_nan()));
        assert_eq!(
            trailing_sum(input.view(), 0),
            Err(ClimateError::EmptyPeriod { argument: "scale" })
        );
    }
    let input = Array2::from_shape_fn((20, 4), |(row, cell)| (row + cell) as f64);
    let view = input.slice_axis(Axis(0), Slice::new(0, None, 2));
    let view = view.slice_axis(Axis(1), Slice::new(0, None, -2));
    assert_eq!(trailing_sum(view, 3).unwrap()[[2, 0]], 15.);
    let tiny = Array2::from_elem((9, 2), f64::from_bits(1));
    assert_eq!(trailing_sum(tiny.view(), 8).unwrap()[[7, 0]].to_bits(), 8);
    let zero = Array2::from_elem((9, 1), -0.0);
    assert_eq!(trailing_sum(zero.view(), 8).unwrap()[[7, 0]].to_bits(), 0);
}
