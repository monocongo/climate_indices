//! Print deterministic scalar probes for `scripts/native_parity_maxima.py --write-contract`.
//! The reporting script compares these values directly with SciPy, including non-finite cases.

use climate_core::special::{igam, igamc, ndtr, ndtri, norm_ppf};

fn main() {
    let gamma_points = [
        0.0,
        1e-12,
        0.1,
        0.75,
        1.0,
        2.5,
        10.0,
        30.0,
        200.0,
        1000.0,
        1e4,
        1e6,
        f64::INFINITY,
        f64::NAN,
    ];
    for a in gamma_points {
        for x in gamma_points {
            println!("special_igam,{a:e},{x:e},{:e}", igam(a, x));
            println!("special_igamc,{a:e},{x:e},{:e}", igamc(a, x));
        }
    }
    for p in [
        -1.0,
        0.0,
        1e-300,
        1e-10,
        0.025,
        0.5,
        0.975,
        1.0 - 1e-10,
        1.0,
        2.0,
        f64::INFINITY,
        f64::NAN,
    ] {
        println!("special_ndtri,{p:e},0,{:e}", ndtri(p));
        println!("special_norm_ppf,{p:e},0,{:e}", norm_ppf(p));
    }
    for x in [
        f64::NEG_INFINITY,
        -40.0,
        -8.0,
        -1.0,
        0.0,
        0.1,
        1.0,
        8.0,
        40.0,
        f64::INFINITY,
        f64::NAN,
    ] {
        println!("special_ndtr,{x:e},0,{:e}", ndtr(x));
    }
}
