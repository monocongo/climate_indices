//! Standalone distribution regression checks; Cargo reads checked-in text only.
//! See fixtures/distribution_reference.md for sources, shapes, units, and coverage.

use std::collections::BTreeMap;
use std::path::Path;

use climate_indices::{eddi, gamma, loglogistic, pci, pearson, pnp, recurrence, special};
use ndarray::{Array2, array, s};

fn fixture(name: &str) -> BTreeMap<String, Array2<f64>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let text = std::fs::read_to_string(path).expect("checked-in distribution fixture");
    let mut tokens = text
        .lines()
        .filter(|line| !line.starts_with('#'))
        .flat_map(str::split_whitespace);
    let mut arrays = BTreeMap::new();
    while let Some(name) = tokens.next() {
        let rows: usize = tokens.next().unwrap().parse().unwrap();
        let columns: usize = tokens.next().unwrap().parse().unwrap();
        let values = (0..rows * columns)
            .map(|_| tokens.next().unwrap().parse::<f64>().unwrap())
            .collect();
        assert!(
            arrays
                .insert(
                    name.to_owned(),
                    Array2::from_shape_vec((rows, columns), values).unwrap(),
                )
                .is_none(),
            "duplicate fixture field {name}"
        );
    }
    assert!(!arrays.is_empty());
    arrays
}

fn close(label: &str, actual: f64, expected: f64) {
    assert_eq!(actual.is_nan(), expected.is_nan(), "{label}: NaN mask");
    if expected.is_nan() {
        return;
    }
    if expected.is_infinite() {
        assert_eq!(actual, expected, "{label}: signed infinity");
    } else {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= 1e-10 + 1e-10 * expected.abs(),
            "{label}: {actual} != {expected}"
        );
    }
}

fn vector(label: &str, actual: impl IntoIterator<Item = f64>, expected: &Array2<f64>) {
    let actual: Vec<_> = actual.into_iter().collect();
    assert_eq!(expected.nrows(), 1, "{label}: expected vector");
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        close(&format!("{label}[{index}]"), actual, expected);
    }
}

fn matrix(label: &str, actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim(), "{label}: shape");
    for ((index, &actual), &expected) in actual.indexed_iter().zip(expected) {
        close(&format!("{label}{index:?}"), actual, expected);
    }
}

#[test]
fn deterministic_positive_parameters_preserve_cdf_invariants_and_nan_masks() {
    for seed in 1..=256 {
        let alpha = (-7.0 + 14.6 * ((seed * 73) % 257) as f64 / 256.0).exp();
        let scale = (-3.0 + 6.0 * ((seed * 131) % 257) as f64 / 256.0).exp();
        let loc = seed as f64 / 13.0 - 10.0;
        let skew = 0.02 + 4.0 * ((seed * 31) % 257) as f64 / 256.0;
        let shape = 0.05 + 0.9 * ((seed * 17) % 257) as f64 / 256.0;
        let (mut previous_cdf, mut previous_sf) = (0.0, 1.0);
        for factor in [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let x = alpha * factor;
            let cdf = gamma::gamma_cdf(x * scale, alpha, scale);
            let sf = gamma::gamma_sf(x, alpha);
            assert!((0.0..=1.0).contains(&cdf) && cdf >= previous_cdf);
            assert!((0.0..=1.0).contains(&sf) && sf <= previous_sf);
            close("gamma complement", cdf + sf, 1.0);
            close("lower incomplete gamma", special::igam(alpha, x), cdf);
            close("upper incomplete gamma", special::igamc(alpha, x), sf);
            (previous_cdf, previous_sf) = (cdf, sf);
        }
        for signed_skew in [-skew, 0.0, skew] {
            let mut previous = 0.0;
            for z in [-40.0, -8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0, 40.0] {
                let p = pearson::pearson_cdf(loc + scale * z, signed_skew, loc, scale);
                assert!((0.0..=1.0).contains(&p) && p >= previous);
                previous = p;
            }
        }
        for signed_shape in [-shape, 0.0, shape] {
            let mut previous = 0.0;
            for z in [-40.0, -8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0, 40.0] {
                let p = loglogistic::loglogistic_cdf(loc + scale * z, loc, scale, signed_shape);
                assert!((0.0..=1.0).contains(&p) && p >= previous);
                previous = p;
            }
        }
        let q = (seed as f64 - 0.5) / 256.0;
        close("normal inverse", special::ndtr(special::norm_ppf(q)), q);
        close(
            "normal symmetry",
            special::ndtr(-loc) + special::ndtr(loc),
            1.0,
        );
        for (x, a, b) in [
            (f64::NAN, alpha, scale),
            (1.0, f64::NAN, scale),
            (1.0, alpha, f64::NAN),
        ] {
            assert!(gamma::gamma_cdf(x, a, b).is_nan());
        }
        for (a, x) in [(f64::NAN, alpha), (alpha, f64::NAN)] {
            assert!(special::igam(a, x).is_nan());
            assert!(special::igamc(a, x).is_nan());
            assert!(gamma::gamma_sf(x, a).is_nan());
        }
        for missing in 0..4 {
            let mut parameters = [loc, skew, loc, scale];
            parameters[missing] = f64::NAN;
            assert!(
                pearson::pearson_cdf(parameters[0], parameters[1], parameters[2], parameters[3])
                    .is_nan()
            );
            let mut parameters = [loc, loc, scale, shape];
            parameters[missing] = f64::NAN;
            assert!(
                loglogistic::loglogistic_cdf(
                    parameters[0],
                    parameters[1],
                    parameters[2],
                    parameters[3]
                )
                .is_nan()
            );
        }
    }
    assert!(special::ndtr(f64::NAN).is_nan());
    assert!(special::ndtri(f64::NAN).is_nan());
    assert!(special::norm_ppf(f64::NAN).is_nan());
    assert!(eddi::hastings_inverse_normal(f64::NAN).is_nan());
}

#[test]
fn comparison_rejects_nan_mask_errors_wrong_infinity_signs_and_nonfinite_substitutions() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        close("same", value, value);
    }
    for (actual, expected) in [
        (0.0, f64::NAN),
        (f64::NAN, 0.0),
        (f64::INFINITY, f64::NEG_INFINITY),
        (f64::INFINITY, 1.0),
        (1.0, f64::INFINITY),
        (1.01, 1.0),
    ] {
        assert!(std::panic::catch_unwind(|| close("must reject", actual, expected)).is_err());
    }
}

#[test]
fn recurrence_builder_and_runner_match_explicit_state_and_gap_expectations() {
    let weather = array![
        [false, true, true, true],
        [true, false, true, true],
        [false, true, true, false],
        [true, false, true, false],
        [true, false, true, true],
        [false, true, true, true],
        [true, false, true, false],
    ];
    let mut season = Array2::from_elem((7, 4), true);
    season.slice_mut(s![4..6, 0]).fill(false);
    let static_valid = array![true, true, false, true];
    let gaps = array![-1, 1, -1, 0];
    let initial = [0.0, 10.0, 20.0, f64::NAN];
    let nan = f64::NAN;
    for policy in ["propagate", "bridge"] {
        let mut expected = array![
            [1.0, nan, nan, nan],
            [nan, 12.0, nan, nan],
            [2.0, nan, nan, nan],
            [2.0, nan, nan, nan],
            [2.0, nan, nan, nan],
            [3.0, nan, nan, nan],
        ];
        let mut final_state = array![[3.0, nan, 20.0, nan]];
        if policy == "propagate" {
            expected.slice_mut(s![1.., ..]).fill(nan);
            final_state[[0, 0]] = nan;
        }
        let final_gaps = [0, 1, -1, 1];
        for columns in [0..4, 0..1, 1..2, 2..3, 3..4] {
            for record in [true, false] {
                let inputs = recurrence::recurrence_inputs(
                    weather.slice(s![.., columns.clone()]),
                    static_valid.slice(s![columns.clone()]),
                    Some(season.slice(s![.., columns.clone()])),
                    gaps.slice(s![columns.clone()]),
                    1,
                    policy,
                    1,
                )
                .unwrap();
                assert_eq!(inputs.spin_up, 1);
                assert_eq!(
                    inputs.policy,
                    if policy == "bridge" {
                        recurrence::MissingDayPolicy::Bridge { max_gap_days: 1 }
                    } else {
                        recurrence::MissingDayPolicy::Propagate
                    }
                );
                let result = recurrence::run(
                    "explicit_increment",
                    &initial[columns.clone()],
                    &inputs,
                    record,
                    |state| *state,
                    |state| *state = f64::NAN,
                    |state, day, cell| {
                        assert!(day < 7 && cell < columns.len());
                        state + 1.0
                    },
                )
                .unwrap();
                if record {
                    matrix(
                        "recurrence history",
                        &result.values.unwrap(),
                        &expected.slice(s![.., columns.clone()]).to_owned(),
                    );
                } else {
                    assert!(result.values.is_none());
                }
                vector(
                    "recurrence state",
                    result.state,
                    &final_state.slice(s![.., columns.clone()]).to_owned(),
                );
                if columns == (2..3) {
                    assert!(result.trailing_gap_days.is_none());
                } else {
                    assert_eq!(
                        result.trailing_gap_days.unwrap().to_vec(),
                        final_gaps[columns.clone()]
                    );
                }
            }
        }
    }
    assert!(matches!(
        recurrence::recurrence_inputs(
            weather.view(),
            static_valid.view(),
            None,
            gaps.view(),
            0,
            "unknown",
            1
        ),
        Err(climate_indices::ClimateError::UnknownNanPolicy { .. })
    ));
}

#[test]
fn special_functions_match_scipy_including_normal_tails_and_signed_infinities() {
    let f = fixture("distribution_special.txt");
    for (index, case) in f["igamma_cases"].rows().into_iter().enumerate() {
        close(
            "igam",
            special::igam(case[0], case[1]),
            f["igam"][[0, index]],
        );
        close(
            "igamc",
            special::igamc(case[0], case[1]),
            f["igamc"][[0, index]],
        );
    }
    vector(
        "ndtr",
        f["normal_x"].iter().map(|&x| special::ndtr(x)),
        &f["ndtr"],
    );
    vector(
        "ndtri",
        f["quantiles"].iter().map(|&p| special::ndtri(p)),
        &f["ndtri"],
    );
    vector(
        "norm_ppf",
        f["quantiles"].iter().map(|&p| special::norm_ppf(p)),
        &f["norm_ppf"],
    );
}

#[test]
fn eddi_strict_ranks_pads_and_hastings_match_python() {
    let f = fixture("distribution_indices.txt");
    let result = eddi::tukey_probabilities(
        f["eddi_climatology"].view(),
        f["eddi_values"].view(),
        f["eddi_pads"].row(0),
    )
    .unwrap();
    matrix("Tukey probabilities", &result, &f["eddi_probabilities"]);
    vector(
        "Hastings",
        f["hastings_p"]
            .iter()
            .map(|&p| eddi::hastings_inverse_normal(p)),
        &f["hastings_z"],
    );
}

#[test]
fn pnp_normals_and_ratios_match_python_with_partial_years() {
    let f = fixture("distribution_indices.txt");
    let calibration = &f["pnp_calibration"];
    let normals = pnp::pnp_normals(calibration.view());
    vector("PNP normals", normals.iter().copied(), &f["pnp_normals"]);
    // A contiguous single column takes NumPy's pairwise reduction path.
    for column in 0..calibration.ncols() {
        let contiguous = calibration.slice(s![.., column..column + 1]).to_owned();
        close(
            "PNP single column",
            pnp::pnp_normals(contiguous.view())[0],
            normals[column],
        );
    }
    let normals = normals.into_shape_with_order((12, 4)).unwrap();
    let result = pnp::pnp_percentages(f["pnp_values"].view(), normals.view()).unwrap();
    matrix("PNP ratios", &result, &f["pnp_ratios"]);
}

#[test]
fn pci_matches_python_for_both_year_lengths_and_zero_rain() {
    let f = fixture("distribution_indices.txt");
    for days in [365, 366] {
        for mode in ["varied", "uniform", "zero", "february"] {
            let key = format!("pci_{days}_{mode}");
            close(
                &key,
                pci::pci(f[&key].row(0)).unwrap(),
                f[&format!("{key}_expected")][[0, 0]],
            );
        }
    }
    assert_eq!(pci::pci(array![1.0, 2.0].view()), None);
}

#[test]
fn gamma_cdfs_and_zero_mixture_match_scipy() {
    let f = fixture("distribution_cdfs.txt");
    for (index, case) in f["gamma_cases"].rows().into_iter().enumerate() {
        close(
            "gamma cdf",
            gamma::gamma_cdf(case[0], case[1], case[2]),
            f["gamma_cdf"][[0, index]],
        );
        close(
            "gamma sf",
            gamma::gamma_sf(case[0], case[1]),
            f["gamma_sf"][[0, index]],
        );
    }
    let result = gamma::gamma_probabilities(
        f["gamma_values"].view(),
        f["gamma_alpha"].row(0),
        f["gamma_beta"].row(0),
        f["gamma_p0"].row(0),
    )
    .unwrap();
    matrix("gamma mixture", &result, &f["gamma_probabilities"]);
}

#[test]
fn pearson_scalar_block_and_ks_match_python_and_scipy() {
    let f = fixture("distribution_cdfs.txt");
    let values = &f["pearson_values"];
    let skews = f["pearson_skew"].row(0);
    let locs = f["pearson_loc"].row(0);
    let scales = f["pearson_scale"].row(0);
    let result = pearson::pearson_cdf_block(values.view(), skews, locs, scales).unwrap();
    matrix("Pearson block", &result, &f["pearson_cdf"]);
    for ((row, column), &value) in values.indexed_iter() {
        close(
            "Pearson scalar",
            pearson::pearson_cdf(value, skews[column], locs[column], scales[column]),
            f["pearson_cdf"][[row, column]],
        );
    }
    let statistics = pearson::pearson_ks_statistics(
        f["ks_values"].view(),
        f["ks_skew"].row(0),
        f["ks_loc"].row(0),
        f["ks_scale"].row(0),
    )
    .unwrap();
    vector("Pearson KS", statistics, &f["pearson_ks"]);
}

#[test]
fn loglogistic_scalar_and_block_match_python_at_support_limits() {
    let f = fixture("distribution_cdfs.txt");
    let values = &f["glo_values"];
    let locs = f["glo_loc"].row(0);
    let scales = f["glo_scale"].row(0);
    let shapes = f["glo_shape"].row(0);
    let result = loglogistic::loglogistic_cdf_block(values.view(), locs, scales, shapes).unwrap();
    matrix("GLO block", &result, &f["glo_cdf"]);
    for ((row, column), &value) in values.indexed_iter() {
        close(
            "GLO scalar",
            loglogistic::loglogistic_cdf(value, locs[column], scales[column], shapes[column]),
            f["glo_cdf"][[row, column]],
        );
    }
    // `loglogistic_cdf` is the bare formula: no parameter validation, so a
    // non-positive scale or |shape| >= 1 must return a finite value, not NaN.
    let bare_values = f["glo_bare_values"].row(0);
    let bare_locs = f["glo_bare_loc"].row(0);
    let bare_scales = f["glo_bare_scale"].row(0);
    let bare_shapes = f["glo_bare_shape"].row(0);
    for column in 0..bare_values.len() {
        close(
            "GLO bare formula",
            loglogistic::loglogistic_cdf(
                bare_values[column],
                bare_locs[column],
                bare_scales[column],
                bare_shapes[column],
            ),
            f["glo_bare_cdf"][[0, column]],
        );
    }
}

#[test]
fn fitting_matches_python_including_validity_masks_and_degenerate_columns() {
    let f = fixture("distribution_fits.txt");
    let calibration = f["calibration"].view();
    let (alphas, betas) = gamma::gamma_parameters(calibration);
    vector("gamma alpha", alphas, &f["gamma_alpha"]);
    vector("gamma beta", betas, &f["gamma_beta"]);

    let fit = pearson::pearson_parameters(calibration);
    vector("Pearson p0", fit.probabilities_of_zero, &f["pearson_p0"]);
    vector("Pearson loc", fit.locs, &f["pearson_loc"]);
    vector("Pearson scale", fit.scales, &f["pearson_scale"]);
    vector("Pearson skew", fit.skews, &f["pearson_skew"]);
    assert_eq!(
        fit.valid.to_vec(),
        f["pearson_valid"]
            .iter()
            .map(|&v| v == 1.0)
            .collect::<Vec<_>>()
    );

    let fit = loglogistic::loglogistic_parameters(calibration);
    vector("GLO loc", fit.locs, &f["glo_loc"]);
    vector("GLO scale", fit.scales, &f["glo_scale"]);
    vector("GLO shape", fit.shapes, &f["glo_shape"]);
    assert_eq!(
        fit.valid.to_vec(),
        f["glo_valid"].iter().map(|&v| v == 1.0).collect::<Vec<_>>()
    );
}
