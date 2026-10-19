use std::hint::black_box;

use climate_indices::recurrence::{MissingDayPolicy, RecurrenceInputs};
use climate_indices::{eddi, eto, fire, flood, gamma, loglogistic, palmer};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ndarray::{Array1, Array2, Array3};

const YEARS: usize = 40;
const MONTHS: usize = 12;
const DAYS: usize = 5 * 366;
const DURATION: usize = 30;

// Integer arithmetic fixes the fixture across platforms without a random-number dependency.
fn sample(step: usize, cell: usize) -> f64 {
    ((step * 37 + cell * 17 + 11) % 101) as f64 / 100.0
}

fn monthly(c: &mut Criterion) {
    let mut group = c.benchmark_group("monthly");
    for cells in [1, 32] {
        let rain_mm = Array3::from_shape_fn((YEARS, MONTHS, cells), |(year, month, cell)| {
            let step = year * MONTHS + month;
            if (step + cell) % 13 == 0 {
                0.0
            } else {
                5.0 + 95.0 * sample(step, cell)
            }
        });
        let temperatures_c = Array3::from_shape_fn(rain_mm.dim(), |(year, month, cell)| {
            5.0 + 25.0 * sample(year * MONTHS + month + 7, cell)
        });
        let water_balance_mm = Array2::from_shape_fn((YEARS, MONTHS * cells), |(year, column)| {
            100.0 * (sample(year * MONTHS + column / cells, column % cells) - 0.5)
        });
        let rain_columns = rain_mm
            .view()
            .into_shape_with_order((YEARS, MONTHS * cells))
            .unwrap();
        let (alphas, betas) = gamma::gamma_parameters(rain_columns);
        let p0 = Array1::from_shape_fn(MONTHS * cells, |column| {
            rain_columns
                .column(column)
                .iter()
                .filter(|&&value| value == 0.0)
                .count() as f64
                / YEARS as f64
        });
        let glo = loglogistic::loglogistic_parameters(water_balance_mm.view());
        let probabilities =
            gamma::gamma_probabilities(rain_columns, alphas.view(), betas.view(), p0.view())
                .unwrap();
        assert!(
            alphas
                .iter()
                .chain(betas.iter())
                .all(|value| value.is_finite() && *value > 0.0)
        );
        assert!(
            probabilities
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
        );
        assert!(glo.valid.iter().all(|&valid| valid));
        group.throughput(Throughput::Elements(rain_mm.len() as u64));

        group.bench_function(BenchmarkId::new("gamma_parameters", cells), |b| {
            b.iter(|| black_box(gamma::gamma_parameters(black_box(rain_columns))))
        });
        group.bench_function(BenchmarkId::new("gamma_probabilities", cells), |b| {
            b.iter(|| {
                black_box(
                    gamma::gamma_probabilities(
                        black_box(rain_columns),
                        alphas.view(),
                        betas.view(),
                        p0.view(),
                    )
                    .unwrap(),
                )
            })
        });
        group.bench_function(BenchmarkId::new("loglogistic_parameters", cells), |b| {
            b.iter(|| {
                black_box(loglogistic::loglogistic_parameters(black_box(
                    water_balance_mm.view(),
                )))
            })
        });

        let latitudes_radians = Array1::from_elem(cells, 40.0_f64.to_radians());
        let leap_years = Array1::from_shape_fn(YEARS, |year| {
            let year = 1980 + year;
            year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
        });
        let pet = eto::thornthwaite(
            temperatures_c.view(),
            latitudes_radians.view(),
            leap_years.view(),
        )
        .unwrap();
        assert!(pet.iter().all(|value| value.is_finite() && *value >= 0.0));
        group.throughput(Throughput::Elements(rain_mm.len() as u64));
        group.bench_function(BenchmarkId::new("thornthwaite", cells), |b| {
            b.iter(|| {
                black_box(
                    eto::thornthwaite(
                        black_box(temperatures_c.view()),
                        latitudes_radians.view(),
                        leap_years.view(),
                    )
                    .unwrap(),
                )
            })
        });

        // EDDI receives one calendar step at a time, with one column per cell.
        let january = pet.index_axis(ndarray::Axis(1), 0);
        let pads = Array1::zeros(cells);
        let ranks = eddi::tukey_probabilities(january, january, pads.view()).unwrap();
        assert!(ranks.iter().all(|value| (0.0..1.0).contains(value)));
        group.throughput(Throughput::Elements(january.len() as u64));
        group.bench_function(BenchmarkId::new("tukey_probabilities", cells), |b| {
            b.iter(|| {
                black_box(
                    eddi::tukey_probabilities(black_box(january), january, pads.view()).unwrap(),
                )
            })
        });

        group.throughput(Throughput::Elements(rain_mm.len() as u64));
        let rain_inches = rain_mm.mapv(|value| value / 25.4);
        let pet_inches = pet.mapv(|value| value / 25.4);
        let awc_inches = Array1::from_elem(cells, 5.0);
        let balance = palmer::water_balance(
            rain_inches.view(),
            pet_inches.view(),
            awc_inches.view(),
            0,
            YEARS - 1,
        )
        .unwrap();
        assert_eq!(balance.etdat.dim(), rain_mm.dim());
        assert!(
            balance
                .etdat
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
        group.bench_function(BenchmarkId::new("water_balance", cells), |b| {
            b.iter(|| {
                black_box(
                    palmer::water_balance(
                        black_box(rain_inches.view()),
                        pet_inches.view(),
                        awc_inches.view(),
                        0,
                        YEARS - 1,
                    )
                    .unwrap(),
                )
            })
        });
    }
    group.finish();
}

fn daily(c: &mut Criterion) {
    let mut group = c.benchmark_group("daily");
    for cells in [1, 32] {
        let rain_mm = Array2::from_shape_fn((DAYS, cells), |(day, cell)| {
            if (day + cell) % 7 < 3 {
                0.0
            } else {
                20.0 * sample(day, cell)
            }
        });
        let temperatures_c =
            Array2::from_shape_fn(rain_mm.dim(), |(day, cell)| 20.0 + 10.0 * sample(day, cell));
        let humidity_percent = Array2::from_shape_fn(rain_mm.dim(), |(day, cell)| {
            30.0 + 50.0 * sample(day + 3, cell)
        });
        let wind_kmh = Array2::from_shape_fn(rain_mm.dim(), |(day, cell)| {
            5.0 + 20.0 * sample(day + 11, cell)
        });
        let valid = Array2::from_elem(rain_mm.dim(), true);
        let static_valid = Array1::from_elem(cells, true);
        let gaps = Array1::from_elem(cells, -1);
        let inputs = RecurrenceInputs {
            weather_valid: valid.view(),
            static_valid: static_valid.view(),
            in_season: None,
            trailing_gap_days: gaps.view(),
            spin_up: 0,
            policy: MissingDayPolicy::Propagate,
        };
        let initial_ffmc = Array1::from_elem(cells, 85.0);
        let codes = fire::ffmc(
            temperatures_c.view(),
            humidity_percent.view(),
            wind_kmh.view(),
            rain_mm.view(),
            initial_ffmc.view(),
            &inputs,
            true,
        )
        .unwrap()
        .values
        .unwrap();
        assert_eq!(codes.dim(), rain_mm.dim());
        assert!(codes.iter().all(|value| (0.0..=101.0).contains(value)));
        group.throughput(Throughput::Elements(rain_mm.len() as u64));
        group.bench_function(BenchmarkId::new("ffmc", cells), |b| {
            b.iter(|| {
                black_box(
                    fire::ffmc(
                        black_box(temperatures_c.view()),
                        humidity_percent.view(),
                        wind_kmh.view(),
                        rain_mm.view(),
                        initial_ffmc.view(),
                        &inputs,
                        true,
                    )
                    .unwrap(),
                )
            })
        });

        let effective = flood::effective_precipitation(rain_mm.view(), DURATION).unwrap();
        assert!(
            effective
                .rows()
                .into_iter()
                .take(DURATION - 1)
                .all(|row| row.iter().all(|v| v.is_nan()))
        );
        assert!(
            effective
                .rows()
                .into_iter()
                .skip(DURATION - 1)
                .all(|row| row.iter().all(|v| v.is_finite()))
        );
        group.bench_function(BenchmarkId::new("effective_precipitation", cells), |b| {
            b.iter(|| {
                black_box(
                    flood::effective_precipitation(black_box(rain_mm.view()), DURATION).unwrap(),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, monthly, daily);
criterion_main!(benches);
