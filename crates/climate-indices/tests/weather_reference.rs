//! Standalone weather-kernel checks. Cargo reads committed text fixtures only.
//! See fixtures/weather_reference.md for sources, shapes, units, and tolerances.

use std::collections::BTreeMap;
use std::path::Path;

use climate_indices::pm_eto::{HumidityPathway, MetInputs, Operand, PmEtoInputs, RadiationPathway};
use climate_indices::recurrence::{MissingDayPolicy, RecurrenceInputs};
use climate_indices::{eto, fire, flood, pm_eto};
use ndarray::{Array1, Array2, Array3, s};

fn fixture(name: &str) -> BTreeMap<String, Array2<f64>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let text = std::fs::read_to_string(path).expect("committed weather fixture");
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

fn close(
    label: &str,
    actual: impl IntoIterator<Item = f64>,
    expected: &Array2<f64>,
    atol: f64,
    rtol: f64,
) {
    let actual: Vec<_> = actual.into_iter().collect();
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.is_nan(),
            expected.is_nan(),
            "{label}[{index}]: NaN mask"
        );
        if !expected.is_nan() {
            assert!(
                actual.is_finite() && expected.is_finite(),
                "{label}[{index}]: non-finite value"
            );
            assert!(
                (actual - expected).abs() <= atol + rtol * expected.abs(),
                "{label}[{index}]: {actual} != {expected}"
            );
        }
    }
}

fn parity(label: &str, actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim(), "{label}: shape");
    close(label, actual.iter().copied(), expected, 1e-10, 1e-10);
}

#[test]
fn temperature_pet_matches_python_regression() {
    let f = fixture("weather_eto.txt");
    let temperatures = Array3::from_shape_vec(
        (2, 12, 4),
        f["monthly_temperature"].iter().copied().collect(),
    )
    .unwrap();
    let leaps = f["leap_years"].row(0).mapv(|v| v != 0.0);
    let result = eto::thornthwaite(
        temperatures.view(),
        f["latitude_radians"].row(0),
        leaps.view(),
    )
    .unwrap();
    assert_eq!(result.dim(), (2, 12, 4));
    close(
        "thornthwaite",
        result.iter().copied(),
        &f["thornthwaite"],
        1e-10,
        1e-10,
    );
    assert!(result.iter().filter(|v| v.is_finite()).all(|v| *v >= 0.0));
    let result = eto::hargreaves(
        f["tmin"].view(),
        f["tmax"].view(),
        f["tmean"].view(),
        f["latitude_radians"].row(0),
    )
    .unwrap();
    parity("hargreaves", &result, &f["hargreaves"]);
    assert!(result[[3, 0]].is_nan()); // An inverted temperature range has no real square root.
    assert_eq!(result[[4, 1]], 0.0);
}

#[test]
fn operand_at_reads_scalars_and_strided_arrays() {
    let f = fixture("weather_pm.txt");
    let scalar = Operand::Scalar(f["net_radiation"][[0, 0]]);
    assert_eq!(scalar.at(usize::MAX), f["net_radiation"][[0, 0]]);
    let row = f["net_radiation"].row(0);
    let array = Operand::Array(row.slice(s![..;2]));
    for (index, value) in row.iter().step_by(2).enumerate() {
        assert_eq!(array.at(index).to_bits(), value.to_bits());
    }
}

#[test]
fn penman_monteith_core_matches_python_for_array_scalar_and_mixed_operands() {
    let f = fixture("weather_pm.txt");
    for mode in 0..3 {
        let operand = |name: &str| {
            if mode == 2 || (mode == 1 && name != "net_radiation") {
                Operand::Scalar(f[name][[0, 0]])
            } else {
                Operand::Array(f[name].row(0))
            }
        };
        let result = pm_eto::pm_eto(&PmEtoInputs {
            net_radiation: operand("net_radiation"),
            soil_heat_flux: operand("soil_heat_flux"),
            temperature_celsius: operand("temperature"),
            wind_speed_2m: operand("wind_2m"),
            saturation_vp: operand("saturation_vp"),
            actual_vp: operand("actual_vp"),
            delta: operand("delta"),
            gamma: operand("gamma"),
        })
        .unwrap();
        close(
            "pm_eto",
            result.iter().copied(),
            &f[&format!("core_{mode}")],
            1e-10,
            1e-10,
        );
    }
}

fn meteorology<'a>(
    f: &'a BTreeMap<String, Array2<f64>>,
    humidity: usize,
    radiation: usize,
) -> MetInputs<'a> {
    MetInputs {
        daily_tmin_celsius: Operand::Array(f["met_tmin"].row(0)),
        daily_tmax_celsius: Operand::Array(f["met_tmax"].row(0)),
        latitude_degrees: Operand::Array(f["latitude_degrees"].row(0)),
        elevation_m: Operand::Scalar(100.0),
        wind_speed_m_s: Operand::Array(f["wind"].row(0)),
        wind_speed_height_m: Operand::Scalar(10.0),
        day_of_year: Operand::Array(f["day"].row(0)),
        soil_heat_flux_mj_m2_day: Operand::Scalar(0.0),
        albedo: Operand::Scalar(0.23),
        humidity: match humidity {
            0 => HumidityPathway::Dewpoint {
                tdew_celsius: Operand::Scalar(12.05),
            },
            1 => HumidityPathway::MinimumAndMaximumRelativeHumidity {
                rh_min: Operand::Scalar(63.0),
                rh_max: Operand::Scalar(84.0),
            },
            2 => HumidityPathway::MaximumRelativeHumidity {
                rh_max: Operand::Scalar(84.0),
            },
            3 => HumidityPathway::MeanRelativeHumidity {
                rh_mean: Operand::Scalar(70.0),
            },
            _ => HumidityPathway::MinimumTemperature,
        },
        radiation: match radiation {
            0 => RadiationPathway::Supplied {
                solar_radiation_mj_m2_day: Operand::Scalar(22.0),
            },
            1 => RadiationPathway::SunshineHours {
                sunshine_hours: Operand::Scalar(9.25),
            },
            _ => RadiationPathway::TemperatureRange {
                coastal: radiation == 2,
            },
        },
    }
}

#[test]
fn derived_penman_monteith_matches_python_for_every_pathway() {
    let f = fixture("weather_pm.txt");
    for humidity in 0..5 {
        for radiation in 0..4 {
            let name = format!("met_{humidity}_{radiation}");
            let result =
                pm_eto::penman_monteith_eto(&meteorology(&f, humidity, radiation)).unwrap();
            close(&name, result.iter().copied(), &f[&name], 1e-10, 1e-10);
        }
    }
}

#[test]
fn fire_recurrences_match_python_histories_and_final_states() {
    let f = fixture("weather_fire.txt");
    let months = f["months"].mapv(|v| v as i64);
    let dmc_band = f["dmc_band"].row(0).mapv(|v| v as i64);
    let dc_band = f["dc_band"].row(0).mapv(|v| v as i64);
    let dmc_day = fire::DayLength {
        table: f["dmc_table"].view(),
        band: dmc_band.view(),
        months: months.view(),
    };
    let dc_day = fire::DayLength {
        table: f["dc_table"].view(),
        band: dc_band.view(),
        months: months.view(),
    };
    let valid = f["temperature"].mapv(f64::is_finite);
    let static_valid = Array1::from_elem(valid.ncols(), true);
    let gaps = Array1::from_elem(valid.ncols(), -1);
    let season = f["season"].mapv(|v| v != 0.0);
    for (policy_name, policy) in [
        ("propagate", MissingDayPolicy::Propagate),
        ("bridge", MissingDayPolicy::Bridge { max_gap_days: 2 }),
    ] {
        let inputs = RecurrenceInputs {
            weather_valid: valid.view(),
            static_valid: static_valid.view(),
            in_season: None,
            trailing_gap_days: gaps.view(),
            spin_up: 2,
            policy,
        };
        let ffmc = fire::ffmc(
            f["temperature"].view(),
            f["humidity"].view(),
            f["wind_kmh"].view(),
            f["rain"].view(),
            f["initial_ffmc"].row(0),
            &inputs,
            true,
        )
        .unwrap();
        let dmc = fire::dmc(
            f["temperature"].view(),
            f["humidity"].view(),
            f["rain"].view(),
            &dmc_day,
            f["initial_dmc"].row(0),
            &inputs,
            true,
        )
        .unwrap();
        let dc = fire::dc(
            f["temperature"].view(),
            f["rain"].view(),
            &dc_day,
            f["initial_dc"].row(0),
            &RecurrenceInputs {
                in_season: Some(season.view()),
                ..inputs
            },
            true,
        )
        .unwrap();
        for (name, run) in [("ffmc", ffmc), ("dmc", dmc), ("dc", dc)] {
            let key = format!("{name}_{policy_name}");
            let values = run.values.unwrap();
            parity(&key, &values, &f[&key]);
            close(
                &key,
                run.code.iter().copied(),
                &f[&format!("{key}_state")],
                1e-10,
                1e-10,
            );
            close(
                &key,
                run.trailing_gap_days.unwrap().iter().map(|&v| v as f64),
                &f[&format!("{key}_gaps")],
                0.0,
                0.0,
            );
            assert!(values.iter().filter(|v| v.is_finite()).all(|v| *v >= 0.0));
            if name == "ffmc" {
                assert!(values.iter().filter(|v| v.is_finite()).all(|v| *v <= 101.0));
            }
            if name == "dc" {
                assert_eq!(values[[6, 0]], values[[5, 0]]);
            } // Off-season days carry the code.
        }
        let initial: Vec<_> = f["initial_kbdi"]
            .iter()
            .map(|&kbdi| fire::KbdiCell {
                kbdi,
                wet_spell_precipitation: 0.0,
            })
            .collect();
        let run = fire::kbdi(
            f["rain"].view(),
            f["temperature"].view(),
            f["annual_rain"].row(0),
            &initial,
            &inputs,
            true,
        )
        .unwrap();
        let key = format!("kbdi_{policy_name}");
        let values = run.values.unwrap();
        parity(&key, &values, &f[&key]);
        close(
            &key,
            run.state.iter().map(|v| v.kbdi),
            &f[&format!("{key}_state")],
            1e-10,
            1e-10,
        );
        close(
            &key,
            run.state.iter().map(|v| v.wet_spell_precipitation),
            &f[&format!("{key}_wet_spell")],
            1e-10,
            1e-10,
        );
        close(
            &key,
            run.trailing_gap_days.unwrap().iter().map(|&v| v as f64),
            &f[&format!("{key}_gaps")],
            0.0,
            0.0,
        );
        assert!(
            values
                .iter()
                .filter(|v| v.is_finite())
                .all(|v| (0.0..=203.2).contains(v))
        );
    }
}

#[test]
fn fire_behavior_indices_and_overwinter_match_python_regression() {
    let f = fixture("weather_fire.txt");
    // the histories are recorded after the two-day spin-up, so the wind aligns from day two
    let wind_kmh = f["wind_kmh"].slice(s![2.., ..]);
    let overwinter_precipitation = f["overwinter_precipitation"].view();
    for policy in ["propagate", "bridge"] {
        let ffmc = &f[&format!("ffmc_{policy}")];
        let dmc = &f[&format!("dmc_{policy}")];
        let dc = &f[&format!("dc_{policy}")];
        let isi = fire::initial_spread_index(ffmc.view(), wind_kmh.view()).unwrap();
        parity(&format!("isi_{policy}"), &isi, &f[&format!("isi_{policy}")]);
        let bui = fire::buildup_index(dmc.view(), dc.view()).unwrap();
        parity(&format!("bui_{policy}"), &bui, &f[&format!("bui_{policy}")]);
        let fwi = fire::fwi(isi.view(), bui.view()).unwrap();
        parity(&format!("fwi_{policy}"), &fwi, &f[&format!("fwi_{policy}")]);
        let dsr = fire::daily_severity_rating(fwi.view());
        parity(&format!("dsr_{policy}"), &dsr, &f[&format!("dsr_{policy}")]);
        let overwinter = fire::overwinter_dc(
            f[&format!("dc_{policy}_state")].view(),
            overwinter_precipitation,
            0.75,
            0.75,
        )
        .unwrap();
        parity(
            &format!("overwinter_dc_{policy}"),
            &overwinter,
            &f[&format!("overwinter_dc_{policy}")],
        );
        // the start-up code never falls below the published seed where it is defined
        assert!(
            overwinter
                .iter()
                .filter(|v| v.is_finite())
                .all(|v| *v >= 15.0 - 1e-10)
        );
    }
}

#[test]
fn flood_kernels_match_python_regression() {
    let f = fixture("weather_flood.txt");
    for duration in [1, 2, 7] {
        let result = flood::effective_precipitation(f["rain"].view(), duration).unwrap();
        parity(
            "effective_precipitation",
            &result,
            &f[&format!("pe_{duration}")],
        );
        assert!(
            result
                .slice(s![..duration - 1, ..])
                .iter()
                .all(|v| v.is_nan())
        );
    }
    let pe = &f["pe_7"];
    let padded = &f["pe_padded"];
    let result = flood::edi(padded.view(), 0..3).unwrap();
    parity("edi", &result, &f["edi"]);
    let result = flood::flood_index(pe.view(), 0, 3).unwrap();
    parity("flood_index", &result, &f["flood_index"]);
    assert!(result.column(1).iter().all(|v| v.is_nan())); // Constant calibration maxima cannot define an SD.

    let rain = &f["api_rain"];
    let valid = rain.mapv(f64::is_finite);
    let static_valid = Array1::from_elem(rain.ncols(), true);
    let gaps = Array1::from_elem(rain.ncols(), -1);
    let initial = Array1::zeros(rain.ncols());
    for (name, policy) in [
        ("propagate", MissingDayPolicy::Propagate),
        ("bridge", MissingDayPolicy::Bridge { max_gap_days: 2 }),
    ] {
        let inputs = RecurrenceInputs {
            weather_valid: valid.view(),
            static_valid: static_valid.view(),
            in_season: None,
            trailing_gap_days: gaps.view(),
            spin_up: 2,
            policy,
        };
        let run =
            flood::antecedent_precipitation_index(rain.view(), 0.9, initial.view(), &inputs, true)
                .unwrap();
        let key = format!("api_{name}");
        parity(&key, &run.values.unwrap(), &f[&key]);
        close(
            &key,
            run.state.iter().copied(),
            &f[&format!("{key}_state")],
            1e-10,
            1e-10,
        );
        close(
            &key,
            run.trailing_gap_days.unwrap().iter().map(|&v| v as f64),
            &f[&format!("{key}_gaps")],
            0.0,
            0.0,
        );
    }
}

#[test]
fn literature_pet_examples_keep_their_documented_tolerances() {
    let f = fixture("weather_literature.txt");
    let temps = Array3::from_shape_vec(
        (1, 12, 1),
        f["watson_temperature"].iter().copied().collect(),
    )
    .unwrap();
    let latitude = Array1::from_vec(vec![43.0_f64.to_radians()]);
    let leaps = Array1::from_vec(vec![false]);
    let result = eto::thornthwaite(temps.view(), latitude.view(), leaps.view()).unwrap();
    close(
        "Watson 19-2",
        result.iter().copied(),
        &f["watson_pet"],
        4.0,
        0.0,
    );
    let t = &f["mehta_temperature"];
    let tmin = Array2::from_elem((58, 1), t[[0, 0]]);
    let tmax = Array2::from_elem((58, 1), t[[0, 1]]);
    let tmean = Array2::from_elem((58, 1), t[[0, 2]]);
    let latitude = Array1::from_vec(vec![(-12.3_f64).to_radians()]);
    let result = eto::hargreaves(tmin.view(), tmax.view(), tmean.view(), latitude.view()).unwrap();
    close(
        "Mehta day 58",
        [result[[57, 0]]],
        &f["mehta_pet"],
        0.05,
        0.0,
    );
    let met = meteorology(&f, 1, 1);
    let result = pm_eto::penman_monteith_eto(&met).unwrap();
    close(
        "FAO-56 Example 18",
        result.iter().copied(),
        &f["fao56_pet"],
        0.05,
        0.0,
    );
}

#[test]
fn cffwis_matches_the_committed_nrcan_reference() {
    let f = fixture("weather_nrcan.txt");
    let valid = Array2::from_elem(f["temperature"].dim(), true);
    let static_valid = Array1::from_vec(vec![true]);
    let gaps = Array1::from_vec(vec![-1]);
    let months = f["months"].mapv(|v| v as i64);
    let band = Array1::from_vec(vec![0]);
    let inputs = RecurrenceInputs {
        weather_valid: valid.view(),
        static_valid: static_valid.view(),
        in_season: None,
        trailing_gap_days: gaps.view(),
        spin_up: 0,
        policy: MissingDayPolicy::Propagate,
    };
    let ffmc = fire::ffmc(
        f["temperature"].view(),
        f["humidity"].view(),
        f["wind_kmh"].view(),
        f["rain"].view(),
        f["initial_ffmc"].row(0),
        &inputs,
        true,
    )
    .unwrap();
    let dmc = fire::dmc(
        f["temperature"].view(),
        f["humidity"].view(),
        f["rain"].view(),
        &fire::DayLength {
            table: f["dmc_table"].view(),
            band: band.view(),
            months: months.view(),
        },
        f["initial_dmc"].row(0),
        &inputs,
        true,
    )
    .unwrap();
    let dc = fire::dc(
        f["temperature"].view(),
        f["rain"].view(),
        &fire::DayLength {
            table: f["dc_table"].view(),
            band: band.view(),
            months: months.view(),
        },
        f["initial_dc"].row(0),
        &inputs,
        true,
    )
    .unwrap();
    for (name, run) in [("ffmc", ffmc), ("dmc", dmc), ("dc", dc)] {
        close(
            name,
            run.values.unwrap().iter().copied(),
            &f[name],
            1e-9,
            0.0,
        );
    }
}
