//! Python regression fixtures, not independent scientific validation.
//! See `fixtures/palmer_reference.txt` for source revision, units, and regeneration.

use std::collections::BTreeMap;

use climate_indices::{palmer, palmer_pdi, palmer_wells, palmer_zindex, self_calibration};
use ndarray::{Array1, Array2, Array3};

const ATOL: f64 = 1e-10;
const RTOL: f64 = 1e-10;
const WATER_SHAPE: (usize, usize, usize) = (8, 12, 4);

struct Fixture(BTreeMap<String, Vec<f64>>);

impl Fixture {
    fn load() -> Self {
        let mut fields = BTreeMap::new();
        let mut current = None;
        for line in include_str!("fixtures/palmer_reference.txt").lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(name) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                assert!(fields.insert(name.to_owned(), Vec::new()).is_none());
                current = Some(name.to_owned());
            } else {
                let values = fields
                    .get_mut(current.as_ref().expect("fixture section"))
                    .unwrap();
                values.extend(line.split_whitespace().map(|v| v.parse::<f64>().unwrap()));
            }
        }
        Self(fields)
    }

    fn values(&self, name: &str) -> &[f64] {
        self.0
            .get(name)
            .unwrap_or_else(|| panic!("missing fixture {name}"))
            .as_slice()
    }

    fn water(&self, name: &str) -> Array3<f64> {
        Array3::from_shape_vec(WATER_SHAPE, self.values(name).to_vec()).unwrap()
    }

    fn calendar(&self, name: &str) -> Array2<f64> {
        Array2::from_shape_vec((12, WATER_SHAPE.2), self.values(name).to_vec()).unwrap()
    }

    fn recursion(&self, name: &str) -> Array2<f64> {
        Array2::from_shape_vec((120, 3), self.values(name).to_vec()).unwrap()
    }

    fn factors(&self, label: &str) -> palmer_wells::WellsFactors {
        let value = |name| {
            let key = format!("factors.{label}.{name}");
            let values = self.values(&key);
            assert_eq!(values.len(), 1, "{key}: expected one value");
            values[0]
        };
        palmer_wells::WellsFactors {
            wetm: value("wetm"),
            wetb: value("wetb"),
            drym: value("drym"),
            dryb: value("dryb"),
            wet_denominator: value("wet_denominator"),
            dry_denominator: value("dry_denominator"),
            wetc: value("wetc"),
            dryc: value("dryc"),
            dry_spell_c: value("dry_spell_c"),
        }
    }
}

fn assert_close(label: &str, actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.is_nan(), e.is_nan(), "{label}[{i}]: NaN mask");
        if e.is_nan() {
            continue;
        }
        if e.is_infinite() {
            assert_eq!(a, e, "{label}[{i}]: infinity");
        } else {
            assert!(
                a.is_finite() && (a - e).abs() <= ATOL + RTOL * e.abs(),
                "{label}[{i}]: actual {a}, expected {e}"
            );
        }
    }
}

#[test]
fn water_balance_matches_python_monthly_values_and_calibration_sums() {
    let fixture = Fixture::load();
    let p = fixture.water("water.precips");
    let pet = fixture.water("water.pet");
    let awc = Array1::from_vec(fixture.values("water.awc").to_vec());
    let result = palmer::water_balance(p.view(), pet.view(), awc.view(), 1, 6).unwrap();
    for (name, values) in [
        ("spdat", &result.spdat),
        ("pldat", &result.pldat),
        ("prdat", &result.prdat),
        ("rdat", &result.rdat),
        ("tldat", &result.tldat),
        ("etdat", &result.etdat),
        ("rodat", &result.rodat),
        ("sssdat", &result.sssdat),
        ("ssudat", &result.ssudat),
    ] {
        let key = format!("water.{name}");
        assert_eq!(values.dim(), WATER_SHAPE, "{key}: shape");
        assert_close(&key, values.as_slice().unwrap(), fixture.values(&key));
    }
    for (name, values) in [
        ("psum", &result.psum),
        ("spsum", &result.spsum),
        ("petsum", &result.petsum),
        ("plsum", &result.plsum),
        ("prsum", &result.prsum),
        ("rsum", &result.rsum),
        ("tlsum", &result.tlsum),
        ("etsum", &result.etsum),
        ("rosum", &result.rosum),
    ] {
        let key = format!("water.{name}");
        assert_eq!(values.dim(), (12, WATER_SHAPE.2), "{key}: shape");
        assert_close(&key, values.as_slice().unwrap(), fixture.values(&key));
    }

    // Each finite month conserves water, including months outside calibration.
    for ((year, month, cell), &precip) in p.indexed_iter() {
        let index = [year, month, cell];
        let supply = precip + result.spdat[index];
        let disposition =
            result.etdat[index] + result.rodat[index] + result.sssdat[index] + result.ssudat[index];
        if supply.is_finite() && disposition.is_finite() {
            assert_close("monthly water conservation", &[supply], &[disposition]);
        }
    }
}

#[test]
fn cafec_departures_and_both_zindex_weightings_match_python() {
    let fixture = Fixture::load();
    // Golden operands isolate these kernels from changes in the Rust water balance.
    let p = fixture.water("water.precips");
    let pet = fixture.water("water.pet");
    let pr = fixture.water("water.prdat");
    let sp = fixture.water("water.spdat");
    let pl = fixture.water("water.pldat");
    let alpha = fixture.calendar("cafec.alpha");
    let beta = fixture.calendar("cafec.beta");
    let gamma = fixture.calendar("cafec.gamma");
    let delta = fixture.calendar("cafec.delta");
    let inputs = palmer_zindex::CafecInputs {
        precips: p.view(),
        pet: pet.view(),
        prdat: pr.view(),
        spdat: sp.view(),
        pldat: pl.view(),
        alpha: alpha.view(),
        beta: beta.view(),
        gamma: gamma.view(),
        delta: delta.view(),
    };
    let trat = fixture.calendar("cafec.trat");
    let (dbar, k_prime) = palmer_zindex::k_prime_and_dbar(&inputs, trat.view(), 1, 6).unwrap();
    for (key, values) in [("cafec.dbar", dbar), ("cafec.k_prime", k_prime)] {
        assert_eq!(values.dim(), (12, WATER_SHAPE.2), "{key}: shape");
        assert_close(key, values.as_slice().unwrap(), fixture.values(key));
    }
    for (weight, key) in [
        ("cafec.ak", "cafec.standard_z"),
        ("cafec.k_prime", "cafec.unnormalized_z"),
    ] {
        let ak = fixture.calendar(weight);
        let z = palmer_zindex::raw_zindex(&inputs, ak.view()).unwrap();
        assert_eq!(z.dim(), WATER_SHAPE, "{key}: shape");
        assert_close(key, z.as_slice().unwrap(), fixture.values(key));
    }
}

fn assert_recursion(label: &str, actual: &[f64], expected: &[f64]) {
    assert_close(label, actual, expected);
    // Backtracking makes categorical choices at zero; a tolerance must not hide them.
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if !e.is_nan() {
            assert_eq!(a == 0.0, e == 0.0, "{label}[{i}]: exact zero");
            if e != 0.0 {
                assert_eq!(
                    a.is_sign_positive(),
                    e.is_sign_positive(),
                    "{label}[{i}]: sign"
                );
            }
        }
    }
}

#[test]
fn pdi_recursion_matches_python_for_default_and_asymmetric_factors() {
    let fixture = Fixture::load();
    let z = fixture.recursion("recursion.z");
    for label in ["default", "asymmetric"] {
        let f = fixture.factors(label);
        let result = palmer_pdi::calculate(
            z.view(),
            palmer_pdi::PdiDurationFactors {
                wetm: f.wetm,
                wetb: f.wetb,
                drym: f.drym,
                dryb: f.dryb,
            },
        );
        for (name, values) in [
            ("pdsi", result.pdsi),
            ("phdi", result.phdi),
            ("pmdi", result.pmdi),
        ] {
            let key = format!("pdi.{label}.{name}");
            assert_eq!(values.dim(), z.dim(), "{key}: shape");
            assert_recursion(&key, values.as_slice().unwrap(), fixture.values(&key));
        }
    }
}

#[test]
fn wells_recursion_matches_python_for_default_and_asymmetric_factors() {
    let fixture = Fixture::load();
    let z = fixture.recursion("recursion.z");
    for label in ["default", "asymmetric"] {
        let factors = fixture.factors(label);
        for cell in 0..z.ncols() {
            let result = palmer_wells::calculate(z.column(cell), &factors).unwrap();
            for (name, values) in [
                ("pdsi", result.pdsi),
                ("phdi", result.phdi),
                ("pmdi", result.pmdi),
            ] {
                let key = format!("wells.{label}.{name}");
                let expected = fixture.recursion(&key).column(cell).to_vec();
                assert_recursion(&key, values.as_slice().unwrap(), &expected);
            }
        }
    }
}

#[test]
fn fitted_wet_and_dry_duration_factors_match_python() {
    let fixture = Fixture::load();
    let z = fixture.recursion("recursion.z");
    for cell in 0..2 {
        let series = z.column(cell).to_vec();
        for (label, spell) in [
            ("wet", self_calibration::Spell::Wet),
            ("dry", self_calibration::Spell::Dry),
        ] {
            let (m, b) = self_calibration::duration_factors(&series, spell).unwrap();
            let key = format!("duration.{cell}.{label}");
            assert_close(&key, &[m, b], fixture.values(&key));
        }
    }
}

#[test]
fn truncated_rank_percentiles_match_python_with_missing_values() {
    let fixture = Fixture::load();
    let samples =
        Array2::from_shape_vec((3, 9), fixture.values("percentile.samples").to_vec()).unwrap();
    let fractions = fixture.values("percentile.fractions");
    let actual: Vec<f64> = samples
        .rows()
        .into_iter()
        .flat_map(|row| {
            fractions.iter().map(move |&fraction| {
                self_calibration::nan_safe_percentile(row.as_slice().unwrap(), fraction)
            })
        })
        .collect();
    assert_close(
        "percentiles",
        &actual,
        fixture.values("percentile.expected"),
    );
    // An empty slice has the same missing result as the all-NaN Python case.
    assert_close(
        "empty percentile",
        &[self_calibration::nan_safe_percentile(&[], 0.5)],
        &[f64::NAN],
    );
}

#[test]
fn wells_and_duration_fits_ignore_gaps_without_advancing_the_state() {
    let fixture = Fixture::load();
    let z = fixture.recursion("recursion.z");
    let factors = fixture.factors("asymmetric");
    for cell in 0..2 {
        let series = z.column(cell).to_vec();
        let present: Vec<f64> = series.iter().copied().filter(|v| !v.is_nan()).collect();
        let with_gaps = palmer_wells::calculate(z.column(cell), &factors).unwrap();
        let compressed =
            palmer_wells::calculate(ndarray::ArrayView1::from(&present), &factors).unwrap();
        for (gapped, compact) in [
            (with_gaps.pdsi, compressed.pdsi),
            (with_gaps.phdi, compressed.phdi),
            (with_gaps.pmdi, compressed.pmdi),
        ] {
            let retained: Vec<f64> = gapped
                .iter()
                .zip(&series)
                .filter_map(
                    |(&value, input)| {
                        if input.is_nan() { None } else { Some(value) }
                    },
                )
                .collect();
            assert_recursion(
                "Wells compressed gaps",
                &retained,
                compact.as_slice().unwrap(),
            );
        }
        for spell in [self_calibration::Spell::Wet, self_calibration::Spell::Dry] {
            assert_eq!(
                self_calibration::duration_factors(&series, spell).unwrap(),
                self_calibration::duration_factors(&present, spell).unwrap()
            );
        }
    }
}
