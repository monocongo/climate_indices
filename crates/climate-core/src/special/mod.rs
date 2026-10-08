//! Special functions ported from the Cephes library as SciPy 1.17.0 ships it.
//!
//! `scipy.stats.gamma.cdf` evaluates `scipy.special.gammainc`, i.e. Cephes
//! `igam`, `scipy.stats.norm.ppf` evaluates `scipy.special.ndtri`,
//! `scipy.stats.pearson3.cdf` evaluates `scipy.special.ndtr` for a near-zero
//! skew, and `scipy.special.gammaln` is Cephes `lgam`. A generic
//! special-function crate uses different series and continued fractions, and
//! the transformed tails of a standardized index are ill-conditioned enough
//! (`ndtri(p)` with `p` within a few ulps of 1) for that to show above the 1e-10
//! parity contract. These are therefore line-by-line ports of
//! `subprojects/xsf/include/xsf/cephes/{igam,ndtri,ndtr,gamma,unity,zeta,
//! lanczos,polevl}.h` from the SciPy 1.17.0 sdist: same branches, constants,
//! and operation order. Only the branches reachable from `igam`, `igamc`,
//! `ndtri`, `ndtr`, and `lgam` are ported; SciPy's `set_error` reporting is
//! dropped, keeping its return values.
//!
//! Licensing: the original Cephes notices and the full xsf BSD-3-Clause and
//! Boost Software License 1.0 texts are retained in the repository's LICENSE.
//! xsf is Copyright (c) 2024, SciPy. The Lanczos coefficients come from
//! Boost.Math (`lanczos.hpp`, (C) Copyright John Maddock 2006).

// constants are copied digit for digit from the C headers, beyond f64 precision
#![allow(clippy::excessive_precision, clippy::approx_constant)]

mod igam_asymp_coeff;

use std::f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2};

/// 2**-53, Cephes `MACHEP`.
const MACHEP: f64 = 1.110_223_024_625_156_540_42E-16;
/// log(DBL_MAX), Cephes `MAXLOG`.
const MAXLOG: f64 = 7.097_827_128_933_839_967_32E2;
/// sqrt(2 pi), named `SQRTPI` in Cephes.
const SQRT_2PI: f64 = 2.506_628_274_631_000_502_42E0;
/// Euler-Mascheroni constant, xsf `SCIPY_EULER`.
const EULER: f64 = 0.577_215_664_901_532_860_606_512_090_082_402_431;
/// Iteration cap of `log1pmx`, Cephes `MAXITER`.
const MAXITER: u64 = 500;

/// `a * b + c`, fused on aarch64.
///
/// SciPy's aarch64 builds compile the Cephes `a * b + c` patterns to fused
/// multiply-adds (the C compilers contract them by default there), and its
/// x86-64 wheels do not. A 1-ulp difference in `lgam` is amplified by the
/// Pearson Type III fit's `exp(lgam(a) - lgam(a + 0.5))` for near-symmetric
/// samples, so this ports the contraction `lgam` and the polynomial helpers
/// get on each target; Rust never contracts on its own. Other expressions in
/// these ports stay unfused, which their well-conditioned callers tolerate.
#[inline(always)]
fn mul_add(a: f64, b: f64, c: f64) -> f64 {
    if cfg!(target_arch = "aarch64") {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

// polevl.h

/// Evaluate the polynomial `coef[0] x^n + ... + coef[n]`.
fn polevl(x: f64, coef: &[f64], n: usize) -> f64 {
    let mut ans = coef[0];
    for &c in &coef[1..=n] {
        ans = mul_add(ans, x, c);
    }
    ans
}

/// Evaluate `x^n + coef[0] x^(n-1) + ... + coef[n-1]`, the leading coefficient 1 implied.
fn p1evl(x: f64, coef: &[f64], n: usize) -> f64 {
    let mut ans = x + coef[0];
    for &c in &coef[1..n] {
        ans = mul_add(ans, x, c);
    }
    ans
}

/// Evaluate the rational function `num(x) / denom(x)`, reversing both
/// polynomials in `1/x` when `|x| > 1`.
fn ratevl(x: f64, num: &[f64], m: usize, denom: &[f64], n: usize) -> f64 {
    let absx = x.abs();
    let evaluate = |coef: &[f64], degree: usize| {
        if absx > 1.0 {
            let y = 1.0 / x;
            let mut ans = coef[degree];
            for c in coef[..degree].iter().rev() {
                ans = ans * y + c;
            }
            ans
        } else {
            let mut ans = coef[0];
            for c in &coef[1..=degree] {
                ans = ans * x + c;
            }
            ans
        }
    };
    let num_ans = evaluate(num, m);
    let denom_ans = evaluate(denom, n);
    if absx > 1.0 {
        x.powi(m as i32 - n as i32) * num_ans / denom_ans
    } else {
        num_ans / denom_ans
    }
}

// lanczos.h (Boost.Math, N=13, G=6.024680040776729583740234375)

const LANCZOS_G: f64 = 6.024_680_040_776_729_583_740_234_375;

const LANCZOS_SUM_EXPG_SCALED_NUM: [f64; 13] = [
    0.006_061_842_346_248_906_525_783_753_964_555_936_883_222,
    0.509_841_665_565_667_618_812_517_864_480_450_950_999_3,
    19.519_927_882_476_174_828_478_609_662_356_521_362_08,
    449.944_556_906_316_811_944_685_860_765_098_840_962_3,
    6_955.999_602_515_376_140_356_310_115_515_198_987_526,
    75_999.293_040_145_426_498_753_034_431_389_091_370_92,
    601_859.617_168_109_878_667_022_653_369_930_235_250_7,
    3_481_712.154_980_645_908_820_710_189_647_745_564_68,
    14_605_578.087_685_068_084_141_699_827_913_592_185_71,
    43_338_889.324_676_138_347_737_237_405_905_333_160_85,
    86_363_131.288_138_591_455_469_272_889_778_684_223_42,
    103_794_043.116_344_545_190_627_105_361_607_023_855_4,
    56_906_521.913_471_563_880_907_910_335_591_226_868_59,
];

const LANCZOS_SUM_EXPG_SCALED_DENOM: [f64; 13] = [
    1.0,
    66.0,
    1925.0,
    32670.0,
    357423.0,
    2637558.0,
    13339535.0,
    45995730.0,
    105258076.0,
    150917976.0,
    120543840.0,
    39916800.0,
    0.0,
];

fn lanczos_sum_expg_scaled(x: f64) -> f64 {
    ratevl(
        x,
        &LANCZOS_SUM_EXPG_SCALED_NUM,
        12,
        &LANCZOS_SUM_EXPG_SCALED_DENOM,
        12,
    )
}

// gamma.h: lgam

const GAMMA_A: [f64; 5] = [
    8.116_141_674_705_084_503_00E-4,
    -5.950_619_042_843_014_383_24E-4,
    7.936_503_404_577_169_439_45E-4,
    -2.777_777_777_300_996_872_05E-3,
    8.333_333_333_333_319_277_22E-2,
];
const GAMMA_B: [f64; 6] = [
    -1.378_251_525_691_208_591_00E3,
    -3.880_163_151_346_378_409_24E4,
    -3.316_129_927_388_711_847_44E5,
    -1.162_370_974_927_623_073_83E6,
    -1.721_737_008_208_396_621_46E6,
    -8.535_556_642_457_654_656_27E5,
];
const GAMMA_C: [f64; 6] = [
    -3.518_157_014_365_234_705_49E2,
    -1.706_421_066_518_811_592_23E4,
    -2.205_285_905_538_544_548_39E5,
    -1.139_334_443_679_825_072_07E6,
    -2.532_523_071_775_829_512_85E6,
    -2.018_891_414_335_327_732_31E6,
];
/// log(sqrt(2 pi))
const LS2PI: f64 = 0.918_938_533_204_672_741_78;
const MAXLGM: f64 = 2.556_348e305;

fn lgam_large_x(x: f64) -> f64 {
    let q = mul_add(x - 0.5, x.ln(), -x) + LS2PI;
    if x > 1.0e8 {
        return q;
    }
    let p = 1.0 / (x * x);
    let p = mul_add(
        mul_add(
            7.936_507_936_507_936_507_936_5e-4,
            p,
            -2.777_777_777_777_777_777_777_8e-3,
        ),
        p,
        0.083_333_333_333_333_333_333_3,
    ) / x;
    q + p
}

/// Natural log of the absolute value of the gamma function, Cephes `lgam`.
///
/// Every caller passes a positive argument, so the reflection branch Cephes
/// takes below -34 is not ported.
pub(crate) fn lgam(x: f64) -> f64 {
    debug_assert!(
        x.is_nan() || x >= -34.0,
        "lgam reflection branch is not ported"
    );
    if !x.is_finite() {
        return x;
    }
    if x < 13.0 {
        let mut z = 1.0;
        let mut p = 0.0;
        let mut u = x;
        while u >= 3.0 {
            p -= 1.0;
            u = x + p;
            z *= u;
        }
        while u < 2.0 {
            if u == 0.0 {
                return f64::INFINITY;
            }
            z /= u;
            p += 1.0;
            u = x + p;
        }
        if z < 0.0 {
            z = -z;
        }
        if u == 2.0 {
            return z.ln();
        }
        p -= 2.0;
        let x = x + p;
        let p = x * polevl(x, &GAMMA_B, 5) / p1evl(x, &GAMMA_C, 6);
        return z.ln() + p;
    }
    if x > MAXLGM {
        return f64::INFINITY;
    }
    if x >= 1000.0 {
        return lgam_large_x(x);
    }
    let q = mul_add(x - 0.5, x.ln(), -x) + LS2PI;
    let p = 1.0 / (x * x);
    q + polevl(p, &GAMMA_A, 4) / x
}

// zeta.h: Hurwitz zeta, only needed at q = 1 by lgam1p_taylor

/// (2k)! / B2k, the Euler-Maclaurin expansion coefficients.
const ZETA_A: [f64; 12] = [
    12.0,
    -720.0,
    30240.0,
    -1209600.0,
    47900160.0,
    -1.892_437_580_318_379_160_6e9,
    7.47242496e10,
    -2.950_130_727_918_164_224e12,
    1.164_678_281_435_006_724_9e14,
    -4.597_978_722_407_472_610_5e15,
    1.815_210_540_194_354_677_3e17,
    -7.166_165_256_175_667_011_3e18,
];

/// Hurwitz zeta function for `q > 0`, Cephes `zeta`.
fn zeta(x: f64, q: f64) -> f64 {
    debug_assert!(q > 0.0, "zeta is only ported for q > 0");
    if x == 1.0 {
        return f64::INFINITY;
    }
    if x < 1.0 {
        return f64::NAN;
    }
    if q > 1e8 {
        return (1.0 / (x - 1.0) + 1.0 / (2.0 * q)) * q.powf(1.0 - x);
    }

    let mut s = q.powf(-x);
    let mut a = q;
    let mut i = 0;
    let mut b = 0.0;
    while i < 9 || a <= 9.0 {
        i += 1;
        a += 1.0;
        b = a.powf(-x);
        s += b;
        if (b / s).abs() < MACHEP {
            return s;
        }
    }
    let w = a;
    s += b * w / (x - 1.0);
    s -= 0.5 * b;
    let mut a = 1.0;
    let mut k = 0.0;
    for coefficient in ZETA_A {
        a *= x + k;
        b /= w;
        let t = a * b / coefficient;
        s += t;
        if (t / s).abs() < MACHEP {
            return s;
        }
        k += 1.0;
        a *= x + k;
        b /= w;
        k += 1.0;
    }
    s
}

// unity.h

const UNITY_LP: [f64; 7] = [
    4.527_000_086_244_519_963_521_5E-5,
    4.985_410_282_319_337_597_221_2E-1,
    6.578_732_594_206_104_484_696_9E0,
    2.991_191_932_855_307_327_737_5E1,
    6.094_966_798_098_778_705_755_6E1,
    5.711_296_359_058_553_810_333_6E1,
    2.003_955_349_920_128_125_964_8E1,
];
const UNITY_LQ: [f64; 6] = [
    1.506_290_908_346_919_204_316_7E1,
    8.304_756_596_796_720_946_943_4E1,
    2.217_623_982_373_285_646_539_4E2,
    3.090_987_222_531_205_977_493_8E2,
    2.164_278_861_449_594_768_500_3E2,
    6.011_866_049_760_384_391_930_6E1,
];

/// log(1 + x), Cephes `log1p`.
fn log1p(x: f64) -> f64 {
    let z = 1.0 + x;
    if !(FRAC_1_SQRT_2..=SQRT_2).contains(&z) {
        return z.ln();
    }
    let z = x * x;
    let z = -0.5 * z + x * (z * polevl(x, &UNITY_LP, 6) / p1evl(x, &UNITY_LQ, 6));
    x + z
}

/// log(1 + x) - x, xsf `log1pmx`.
fn log1pmx(x: f64) -> f64 {
    if x.abs() < 0.5 {
        let mut xfac = x;
        let mut res = 0.0;
        for n in 2..MAXITER {
            xfac *= -x;
            let term = xfac / n as f64;
            res += term;
            if term.abs() < MACHEP * res.abs() {
                break;
            }
        }
        res
    } else {
        log1p(x) - x
    }
}

const UNITY_EP: [f64; 3] = [
    1.261_771_930_748_105_908_779_8E-4,
    3.029_944_077_074_419_612_995_6E-2,
    9.999_999_999_999_999_999_102_5E-1,
];
const UNITY_EQ: [f64; 4] = [
    3.001_985_051_386_644_550_415_9E-6,
    2.524_483_403_496_841_041_922_4E-3,
    2.272_655_482_081_550_287_659_3E-1,
    2.000_000_000_000_000_000_089_7E0,
];

/// exp(x) - 1, Cephes `expm1`.
fn expm1(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_nan() || x > 0.0 {
            return x;
        }
        return -1.0;
    }
    if !(-0.5..=0.5).contains(&x) {
        return x.exp() - 1.0;
    }
    let xx = x * x;
    let r = x * polevl(xx, &UNITY_EP, 2);
    let r = r / (polevl(xx, &UNITY_EQ, 3) - r);
    r + r
}

/// lgam(x + 1) around x = 0 from its Taylor series.
fn lgam1p_taylor(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let mut res = -EULER * x;
    let mut xfac = -x;
    for n in 2..42 {
        xfac *= -x;
        let coeff = zeta(n as f64, 1.0) * xfac / n as f64;
        res += coeff;
        if coeff.abs() < MACHEP * res.abs() {
            break;
        }
    }
    res
}

/// lgam(x + 1), xsf `lgam1p`.
fn lgam1p(x: f64) -> f64 {
    if x.abs() <= 0.5 {
        lgam1p_taylor(x)
    } else if (x - 1.0).abs() < 0.5 {
        x.ln() + lgam1p_taylor(x - 1.0)
    } else {
        lgam(x + 1.0)
    }
}

// ndtr.h: erf and erfc

const NDTR_P: [f64; 9] = [
    2.461_969_814_735_305_125_24E-10,
    5.641_895_648_310_688_219_77E-1,
    7.463_210_564_422_699_126_87E0,
    4.863_719_709_856_813_666_14E1,
    1.965_208_329_560_770_982_42E2,
    5.264_451_949_954_773_586_31E2,
    9.345_285_271_719_576_075_40E2,
    1.027_551_886_895_157_102_72E3,
    5.575_353_353_693_993_275_26E2,
];
const NDTR_Q: [f64; 8] = [
    1.322_819_511_547_449_925_08E1,
    8.670_721_408_859_897_423_29E1,
    3.549_377_788_878_198_910_62E2,
    9.757_085_017_432_054_897_53E2,
    1.823_909_166_879_097_362_89E3,
    2.246_337_608_187_109_817_92E3,
    1.656_663_091_941_613_501_82E3,
    5.575_353_408_177_276_755_46E2,
];
const NDTR_R: [f64; 6] = [
    5.641_895_835_477_550_739_84E-1,
    1.275_366_707_599_781_044_16E0,
    5.019_050_422_511_804_774_14E0,
    6.160_210_979_930_535_851_95E0,
    7.409_742_699_504_489_391_60E0,
    2.978_866_653_721_002_406_70E0,
];
const NDTR_S: [f64; 6] = [
    2.260_528_632_201_172_765_90E0,
    9.396_035_249_380_014_346_73E0,
    1.204_895_398_080_966_566_05E1,
    1.708_144_507_475_658_972_22E1,
    9.608_968_090_632_858_781_98E0,
    3.369_076_451_000_815_160_50E0,
];
const NDTR_T: [f64; 5] = [
    9.604_973_739_870_516_387_49E0,
    9.002_601_972_038_426_892_17E1,
    2.232_005_345_946_843_192_26E3,
    7.003_325_141_128_050_754_73E3,
    5.559_230_130_103_949_627_68E4,
];
const NDTR_U: [f64; 5] = [
    3.356_171_416_475_030_996_47E1,
    5.213_579_497_801_526_797_95E2,
    4.594_323_829_709_801_279_87E3,
    2.262_900_006_138_909_342_46E4,
    4.926_739_426_086_359_210_86E4,
];

/// Complementary error function, Cephes `erfc`.
fn erfc(a: f64) -> f64 {
    if a.is_nan() {
        return f64::NAN;
    }
    let x = a.abs();
    if x < 1.0 {
        return 1.0 - erf(a);
    }
    let z = -a * a;
    if z >= -MAXLOG {
        let z = z.exp();
        let (p, q) = if x < 8.0 {
            (polevl(x, &NDTR_P, 8), p1evl(x, &NDTR_Q, 8))
        } else {
            (polevl(x, &NDTR_R, 5), p1evl(x, &NDTR_S, 6))
        };
        let mut y = (z * p) / q;
        if a < 0.0 {
            y = 2.0 - y;
        }
        if y != 0.0 {
            return y;
        }
    }
    // underflow
    if a < 0.0 { 2.0 } else { 0.0 }
}

/// Error function, Cephes `erf`.
fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return -erf(-x);
    }
    if x.abs() > 1.0 {
        return 1.0 - erfc(x);
    }
    let z = x * x;
    x * polevl(z, &NDTR_T, 4) / p1evl(z, &NDTR_U, 5)
}

// igam.h

const IGAM_MAXITER: usize = 2000;
const IGAM_SMALL: f64 = 20.0;
const IGAM_LARGE: f64 = 200.0;
const IGAM_SMALLRATIO: f64 = 0.3;
const IGAM_LARGERATIO: f64 = 4.5;
const IGAM_BIG: f64 = 4.503_599_627_370_496e15;
const IGAM_BIGINV: f64 = 2.220_446_049_250_313_080_85e-16;

/// Which tail `asymptotic_series` evaluates.
#[derive(Clone, Copy)]
enum Tail {
    Lower,
    Upper,
}

/// x^a exp(-x) / gamma(a), equations (15) and (16) of Maddock et al. with
/// exp(x - a) corrected to exp(a - x).
fn igam_fac(a: f64, x: f64) -> f64 {
    if (a - x).abs() > 0.4 * a.abs() {
        let ax = a * x.ln() - x - lgam(a);
        if ax < -MAXLOG {
            return 0.0;
        }
        return ax.exp();
    }

    let fac = a + LANCZOS_G - 0.5;
    let mut res = (fac / 1.0_f64.exp()).sqrt() / lanczos_sum_expg_scaled(a);

    if a < 200.0 && x < 200.0 {
        res *= (a - x).exp() * (x / fac).powf(a);
    } else {
        let num = x - a - LANCZOS_G + 0.5;
        res *= (a * log1pmx(num / fac) + x * (0.5 - LANCZOS_G) / fac).exp();
    }
    res
}

/// igamc from the continued fraction DLMF 8.9.2.
fn igamc_continued_fraction(a: f64, x: f64) -> f64 {
    let ax = igam_fac(a, x);
    if ax == 0.0 {
        return 0.0;
    }

    let mut y = 1.0 - a;
    let mut z = x + y + 1.0;
    let mut c = 0.0;
    let mut pkm2 = 1.0;
    let mut qkm2 = x;
    let mut pkm1 = x + 1.0;
    let mut qkm1 = z * x;
    let mut ans = pkm1 / qkm1;

    for _ in 0..IGAM_MAXITER {
        c += 1.0;
        y += 1.0;
        z += 2.0;
        let yc = y * c;
        let pk = pkm1 * z - pkm2 * yc;
        let qk = qkm1 * z - qkm2 * yc;
        let t = if qk != 0.0 {
            let r = pk / qk;
            let t = ((ans - r) / r).abs();
            ans = r;
            t
        } else {
            1.0
        };
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;
        if pk.abs() > IGAM_BIG {
            pkm2 *= IGAM_BIGINV;
            pkm1 *= IGAM_BIGINV;
            qkm2 *= IGAM_BIGINV;
            qkm1 *= IGAM_BIGINV;
        }
        if t <= MACHEP {
            break;
        }
    }
    ans * ax
}

/// igam from the power series DLMF 8.11.4.
fn igam_series(a: f64, x: f64) -> f64 {
    let ax = igam_fac(a, x);
    if ax == 0.0 {
        return 0.0;
    }

    let mut r = a;
    let mut c = 1.0;
    let mut ans = 1.0;
    for _ in 0..IGAM_MAXITER {
        r += 1.0;
        c *= x / r;
        ans += c;
        if c <= MACHEP * ans {
            break;
        }
    }
    ans * ax / a
}

/// igamc from DLMF 8.7.3, avoiding the cancellation of `1 - igam_series`.
fn igamc_series(a: f64, x: f64) -> f64 {
    let mut fac = 1.0;
    let mut sum = 0.0;
    for n in 1..IGAM_MAXITER {
        let n = n as f64;
        fac *= -x / n;
        let term = fac / (a + n);
        sum += term;
        if term.abs() <= MACHEP * sum.abs() {
            break;
        }
    }

    let logx = x.ln();
    let term = -expm1(a * logx - lgam1p(a));
    term - (a * logx - lgam(a)).exp() * sum
}

/// igam or igamc from the asymptotic series DLMF 8.12.3/8.12.4 (Temme).
fn asymptotic_series(a: f64, x: f64, tail: Tail) -> f64 {
    use igam_asymp_coeff::{D, K, N};

    let sgn = match tail {
        Tail::Lower => -1.0,
        Tail::Upper => 1.0,
    };
    let lambda = x / a;
    let sigma = (x - a) / a;
    let eta = if lambda > 1.0 {
        (-2.0 * log1pmx(sigma)).sqrt()
    } else if lambda < 1.0 {
        -(-2.0 * log1pmx(sigma)).sqrt()
    } else {
        0.0
    };
    let mut res = 0.5 * erfc(sgn * eta * (a / 2.0).sqrt());

    let mut etapow = [0.0; N];
    etapow[0] = 1.0;
    let mut maxpow = 0;
    let mut sum = 0.0;
    let mut afac = 1.0;
    let mut absoldterm = f64::INFINITY;
    for row in D.iter().take(K) {
        let mut ck = row[0];
        for n in 1..N {
            if n > maxpow {
                etapow[n] = eta * etapow[n - 1];
                maxpow += 1;
            }
            let ckterm = row[n] * etapow[n];
            ck += ckterm;
            if ckterm.abs() < MACHEP * ck.abs() {
                break;
            }
        }
        let term = ck * afac;
        let absterm = term.abs();
        if absterm > absoldterm {
            break;
        }
        sum += term;
        if absterm < MACHEP * sum.abs() {
            break;
        }
        absoldterm = absterm;
        afac /= a;
    }
    res += sgn * (-0.5 * a * eta * eta).exp() * sum / (2.0 * PI * a).sqrt();
    res
}

/// Regularized lower incomplete gamma function P(a, x), Cephes `igam`, i.e.
/// `scipy.special.gammainc`.
pub fn igam(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 || a < 0.0 {
        return f64::NAN;
    } else if a == 0.0 {
        return if x > 0.0 { 1.0 } else { f64::NAN };
    } else if x == 0.0 {
        return 0.0;
    } else if a.is_infinite() {
        return if x.is_infinite() { f64::NAN } else { 0.0 };
    } else if x.is_infinite() {
        return 1.0;
    }

    // asymptotic regime where a ~ x
    let absxma_a = (x - a).abs() / a;
    if (a > IGAM_SMALL && a < IGAM_LARGE && absxma_a < IGAM_SMALLRATIO)
        || (a > IGAM_LARGE && absxma_a < IGAM_LARGERATIO / a.sqrt())
    {
        return asymptotic_series(a, x, Tail::Lower);
    }

    if x > 1.0 && x > a {
        return 1.0 - igamc(a, x);
    }
    igam_series(a, x)
}

/// Regularized upper incomplete gamma function Q(a, x), Cephes `igamc`, i.e.
/// `scipy.special.gammaincc`.
pub fn igamc(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 || a < 0.0 {
        return f64::NAN;
    } else if a == 0.0 {
        return if x > 0.0 { 0.0 } else { f64::NAN };
    } else if x == 0.0 {
        return 1.0;
    } else if a.is_infinite() {
        return if x.is_infinite() { f64::NAN } else { 1.0 };
    } else if x.is_infinite() {
        return 0.0;
    }

    // asymptotic regime where a ~ x
    let absxma_a = (x - a).abs() / a;
    if (a > IGAM_SMALL && a < IGAM_LARGE && absxma_a < IGAM_SMALLRATIO)
        || (a > IGAM_LARGE && absxma_a < IGAM_LARGERATIO / a.sqrt())
    {
        return asymptotic_series(a, x, Tail::Upper);
    }

    if x > 1.1 {
        if x < a {
            1.0 - igam_series(a, x)
        } else {
            igamc_continued_fraction(a, x)
        }
    } else if x <= 0.5 {
        if -0.4 / x.ln() < a {
            1.0 - igam_series(a, x)
        } else {
            igamc_series(a, x)
        }
    } else if x * 1.1 < a {
        1.0 - igam_series(a, x)
    } else {
        igamc_series(a, x)
    }
}

// ndtri.h

/// approximation for 0 <= |y - 0.5| <= 3/8
const NDTRI_P0: [f64; 5] = [
    -5.996_335_010_141_078_952_67E1,
    9.800_107_541_859_996_615_36E1,
    -5.667_628_574_690_702_934_39E1,
    1.393_126_093_872_796_795_03E1,
    -1.239_165_838_673_812_580_16E0,
];
const NDTRI_Q0: [f64; 8] = [
    1.954_488_583_381_417_598_34E0,
    4.676_279_128_988_815_384_53E0,
    8.636_024_213_908_905_905_75E1,
    -2.254_626_878_541_193_705_27E2,
    2.002_602_123_800_606_603_59E2,
    -8.203_722_561_683_333_399_12E1,
    1.590_562_251_262_116_955_15E1,
    -1.183_316_211_213_300_031_42E0,
];
/// approximation for z = sqrt(-2 log y) between 2 and 8
const NDTRI_P1: [f64; 9] = [
    4.055_448_923_059_624_199_23E0,
    3.152_510_945_998_938_661_54E1,
    5.716_281_922_464_212_881_62E1,
    4.408_050_738_932_008_347_00E1,
    1.468_495_619_288_580_240_14E1,
    2.186_633_068_507_902_675_39E0,
    -1.402_560_791_713_544_958_75E-1,
    -3.504_246_268_278_482_034_18E-2,
    -8.574_567_851_546_854_136_11E-4,
];
const NDTRI_Q1: [f64; 8] = [
    1.577_998_832_564_667_497_31E1,
    4.539_076_351_288_792_105_84E1,
    4.131_720_382_546_720_304_40E1,
    1.504_253_856_929_075_034_08E1,
    2.504_649_462_083_094_159_79E0,
    -1.421_829_228_547_877_885_74E-1,
    -3.808_064_076_915_782_771_94E-2,
    -9.332_594_808_954_574_273_72E-4,
];
/// approximation for z = sqrt(-2 log y) between 8 and 64
const NDTRI_P2: [f64; 9] = [
    3.237_748_917_769_460_359_70E0,
    6.915_228_890_689_842_116_95E0,
    3.938_810_252_924_744_434_15E0,
    1.333_034_608_158_075_423_89E0,
    2.014_853_895_491_790_815_38E-1,
    1.237_166_348_178_200_213_58E-2,
    3.015_815_535_082_354_160_07E-4,
    2.658_069_746_867_375_508_32E-6,
    6.239_745_391_849_832_937_30E-9,
];
const NDTRI_Q2: [f64; 8] = [
    6.024_270_393_647_420_142_55E0,
    3.679_835_638_561_608_594_03E0,
    1.377_020_994_890_813_302_71E0,
    2.162_369_935_944_966_358_90E-1,
    1.342_040_060_885_431_890_37E-2,
    3.280_144_646_821_277_391_04E-4,
    2.892_478_647_453_806_839_36E-6,
    6.790_194_080_099_812_744_25E-9,
];

/// exp(-2)
const EXP_MINUS_2: f64 = 0.135_335_283_236_612_691_89;

/// Inverse of the standard normal CDF, Cephes `ndtri`, i.e. `scipy.special.ndtri`.
pub fn ndtri(y0: f64) -> f64 {
    if y0 == 0.0 {
        return f64::NEG_INFINITY;
    }
    if y0 == 1.0 {
        return f64::INFINITY;
    }
    if y0 < 0.0 || y0 > 1.0 {
        return f64::NAN;
    }
    let mut negate = true;
    let mut y = y0;
    if y > 1.0 - EXP_MINUS_2 {
        y = 1.0 - y;
        negate = false;
    }

    if y > EXP_MINUS_2 {
        let y = y - 0.5;
        let y2 = y * y;
        let x = y + y * (y2 * polevl(y2, &NDTRI_P0, 4) / p1evl(y2, &NDTRI_Q0, 8));
        return x * SQRT_2PI;
    }

    let x = (-2.0 * y.ln()).sqrt();
    let x0 = x - x.ln() / x;
    let z = 1.0 / x;
    let x1 = if x < 8.0 {
        z * polevl(z, &NDTRI_P1, 8) / p1evl(z, &NDTRI_Q1, 8)
    } else {
        z * polevl(z, &NDTRI_P2, 8) / p1evl(z, &NDTRI_Q2, 8)
    };
    let x = x0 - x1;
    if negate { -x } else { x }
}

/// `scipy.stats.norm.ppf(q)`: `ndtri` inside (0, 1), -inf at 0, inf at 1, and
/// NaN for a NaN or a value outside [0, 1].
pub fn norm_ppf(q: f64) -> f64 {
    if q > 0.0 && q < 1.0 {
        ndtri(q)
    } else if q == 0.0 {
        f64::NEG_INFINITY
    } else if q == 1.0 {
        f64::INFINITY
    } else {
        f64::NAN
    }
}

/// Standard normal CDF, Cephes `ndtr`, i.e. `scipy.special.ndtr`.
pub fn ndtr(a: f64) -> f64 {
    if a.is_nan() {
        return f64::NAN;
    }
    let x = a * FRAC_1_SQRT_2;
    let z = x.abs();
    if z < FRAC_1_SQRT_2 {
        0.5 + 0.5 * erf(x)
    } else {
        let y = 0.5 * erfc(z);
        if x > 0.0 { 1.0 - y } else { y }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // reference values from scipy.special 1.17.0 (gammainc, gammaincc, ndtri)
    #[test]
    fn igam_matches_scipy_on_each_branch() {
        let cases = [
            (0.5, 0.2, 0.472_910_743_134_461_96),    // igam_series
            (2.0, 5.0, 0.959_572_318_005_487_3),     // 1 - igamc continued fraction
            (0.5, 1.05, 0.852_700_861_377_328_3),    // 1 - igamc_series (lgam1p Taylor, zeta)
            (50.0, 55.0, 0.767_795_219_499_143_6),   // asymptotic series
            (500.0, 520.0, 0.815_308_850_901_256_3), // asymptotic series, large a
            (250.0, 330.0, 0.999_998_110_612_559_6), // igam_fac Lanczos/log1pmx branch
            (180.0, 240.0, 0.999_977_303_637_979_4), // igam_fac Lanczos/log1pmx branch
        ];
        for (a, x, expected) in cases {
            let actual = igam(a, x);
            assert!(
                (actual - expected).abs() <= 1e-15 * expected.abs(),
                "igam({a}, {x}) = {actual:e}, expected {expected:e}"
            );
        }
    }

    #[test]
    fn igam_edge_values() {
        assert!(igam(f64::NAN, 1.0).is_nan());
        assert!(igam(1.0, -1.0).is_nan());
        assert_eq!(igam(1.0, 0.0), 0.0);
        assert_eq!(igam(0.0, 1.0), 1.0);
        assert_eq!(igam(f64::INFINITY, 1.0), 0.0);
        assert_eq!(igam(1.0, f64::INFINITY), 1.0);
    }

    // reference values from scipy.special.gammaln 1.17.0; a fused and an unfused build
    // differ by an ulp, so the tolerance is a few ulps rather than bit equality
    #[test]
    fn lgam_matches_scipy_on_each_branch() {
        let cases = [
            (0.75, 0.203_280_951_431_295_26), // rational approximation, x < 3
            (4.5, 2.453_736_570_842_442_3),   // rational approximation, 3 <= x < 13
            (14.5, 23.862_765_841_689_086),   // Stirling with polevl, 13 <= x < 1000
            (1234.5, 7_550.550_901_077_895),  // lgam_large_x with the correction
            (250_000.5, 2_857_304.968_149_462_7), // lgam_large_x with the correction
            (3.5e9, 73_416_100_808.977_16),   // lgam_large_x, x > 1e8
        ];
        for (x, expected) in cases {
            let actual = lgam(x);
            assert!(
                (actual - expected).abs() <= 4e-16 * expected.abs(),
                "lgam({x}) = {actual:e}, expected {expected:e}"
            );
        }
    }

    #[test]
    fn ndtri_matches_scipy() {
        let cases = [
            (0.5, 0.0),
            (0.975, 1.959_963_984_540_054),
            (1e-10, -6.361_340_902_404_056),
            (1e-300, -37.047_096_299_361_2),
        ];
        for (p, expected) in cases {
            let actual = ndtri(p);
            assert!(
                (actual - expected).abs() <= 1e-15 * expected.abs().max(1.0),
                "ndtri({p}) = {actual:e}, expected {expected:e}"
            );
        }
        assert_eq!(norm_ppf(0.0), f64::NEG_INFINITY);
        assert_eq!(norm_ppf(1.0), f64::INFINITY);
        assert!(norm_ppf(1.5).is_nan());
        assert!(norm_ppf(f64::NAN).is_nan());
    }
}
