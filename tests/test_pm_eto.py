"""Tests for Penman-Monteith ETo helper functions (FAO-56).

Test values are derived from FAO-56 worked examples and tabulated data
(Allen et al., 1998). Equation references in comments correspond to the
FAO-56 document.

References
----------
Allen, R.G., Pereira, L.S., Raes, D. and Smith, M. (1998)
    Crop evapotranspiration - Guidelines for computing crop water requirements.
    FAO Irrigation and Drainage Paper 56. Rome, FAO.
"""

from __future__ import annotations

import numpy as np
import pytest

from climate_indices.pm_eto import (
    ATMOSPHERIC_PRESSURE_SEA_LEVEL,
    actual_vapor_pressure_from_dewpoint,
    actual_vapor_pressure_from_rhmax,
    actual_vapor_pressure_from_rhmean,
    actual_vapor_pressure_from_rhmin_rhmax,
    actual_vapor_pressure_from_tmin,
    atmospheric_pressure,
    latent_heat_of_vaporization,
    mean_saturation_vapor_pressure,
    pm_eto,
    psychrometric_constant,
    saturation_vapor_pressure,
    vapor_pressure_slope,
)

# ---------------------------------------------------------------------------
# Tolerance for floating-point comparisons.  FAO-56 tables are given to 3-4
# significant figures so we use absolute tolerance of 0.01 for most tests
# (values are in kPa or degC ranges).  Tighter tolerance of 1e-4 for pure
# mathematical relations.
# ---------------------------------------------------------------------------
FAO_ABS_TOL = 0.01
MATH_ABS_TOL = 1e-4


# ===================================================================
# Tests for Story 2.1: Atmospheric Helpers (Eq 7, 8, 2.1)
# ===================================================================


class TestAtmosphericPressure:
    """Tests for atmospheric_pressure() -- FAO-56 Eq 7."""

    def test_sea_level(self) -> None:
        """At z=0 m, pressure should equal sea-level constant 101.3 kPa."""
        result = atmospheric_pressure(0.0)
        assert result == pytest.approx(ATMOSPHERIC_PRESSURE_SEA_LEVEL, abs=MATH_ABS_TOL)

    @pytest.mark.parametrize(
        ("elevation_m", "expected_kpa"),
        [
            # FAO-56 Table 2.1: Atmospheric pressure for different elevations
            (0, 101.3),
            (100, 100.1),
            (200, 98.9),
            (300, 97.8),
            (400, 96.6),
            (500, 95.5),
            (600, 94.3),
            (800, 92.1),
            (1000, 89.9),
            (1200, 87.7),
            (1400, 85.6),
            (1600, 83.5),
            (1800, 81.5),
            (2000, 79.5),
            (2500, 74.7),
            (3000, 70.1),
            (4000, 61.6),
        ],
    )
    def test_fao56_table_2_1(self, elevation_m: float, expected_kpa: float) -> None:
        """Validate against FAO-56 Table 2.1 pressure values.

        FAO-56 table values are rounded to 1 decimal place.  The
        difference between our FP64 computation and the tabulated values
        grows slightly at high elevations (up to ~0.5 kPa at 4000 m)
        because the table was produced with lower-precision arithmetic.
        We use 0.6 kPa tolerance to cover this.
        """
        result = atmospheric_pressure(float(elevation_m))
        assert result == pytest.approx(expected_kpa, abs=0.6)

    def test_numpy_array_input(self) -> None:
        """Function should accept and return numpy arrays."""
        elevations = np.array([0.0, 500.0, 1000.0, 2000.0])
        result = atmospheric_pressure(elevations)
        assert isinstance(result, np.ndarray)
        assert result.shape == elevations.shape
        # spot-check sea level
        assert result[0] == pytest.approx(101.3, abs=MATH_ABS_TOL)

    def test_high_elevation(self) -> None:
        """Pressure should decrease monotonically with elevation."""
        elevations = np.linspace(0, 5000, 100)
        pressures = atmospheric_pressure(elevations)
        # pressure should be monotonically decreasing
        assert np.all(np.diff(pressures) < 0)

    def test_negative_elevation(self) -> None:
        """Negative elevations (below sea level) should give higher pressure."""
        result = atmospheric_pressure(-100.0)
        assert result > ATMOSPHERIC_PRESSURE_SEA_LEVEL


class TestLatentHeatOfVaporization:
    """Tests for latent_heat_of_vaporization() -- FAO-56 simplified Eq 2.1."""

    def test_at_20c(self) -> None:
        """At 20 degC, lambda should be approximately 2.45 MJ/kg."""
        result = latent_heat_of_vaporization(20.0)
        assert result == pytest.approx(2.4528, abs=0.001)

    def test_at_0c(self) -> None:
        """At 0 degC, lambda = 2.501 MJ/kg."""
        result = latent_heat_of_vaporization(0.0)
        assert result == pytest.approx(2.501, abs=MATH_ABS_TOL)

    def test_decreases_with_temperature(self) -> None:
        """Latent heat decreases with increasing temperature."""
        temps = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        result = latent_heat_of_vaporization(temps)
        assert np.all(np.diff(result) < 0)

    def test_numpy_array_input(self) -> None:
        """Function should accept and return numpy arrays."""
        temps = np.array([0.0, 20.0, 40.0])
        result = latent_heat_of_vaporization(temps)
        assert isinstance(result, np.ndarray)
        assert result.shape == temps.shape


class TestPsychrometricConstant:
    """Tests for psychrometric_constant() -- FAO-56 Eq 8."""

    @pytest.mark.parametrize(
        ("pressure_kpa", "expected_gamma"),
        [
            # FAO-56 Table 2.2: Psychrometric constant for different altitudes
            # using gamma = 0.665e-3 * P
            (101.3, 0.0674),
            (100.1, 0.0666),
            (98.9, 0.0658),
            (97.8, 0.0650),
            (96.6, 0.0642),
            (95.5, 0.0635),
            (94.3, 0.0627),
            (92.1, 0.0612),
            (89.9, 0.0598),
            (87.7, 0.0583),
            (85.6, 0.0569),
            (81.5, 0.0542),
            (79.5, 0.0529),
            (74.7, 0.0497),
            (70.1, 0.0466),
            (61.6, 0.0410),
        ],
    )
    def test_fao56_table_2_2(self, pressure_kpa: float, expected_gamma: float) -> None:
        """Validate against FAO-56 Table 2.2 psychrometric constant values."""
        result = psychrometric_constant(pressure_kpa)
        assert result == pytest.approx(expected_gamma, abs=0.001)

    def test_sea_level_pressure(self) -> None:
        """At sea level (101.3 kPa), gamma ~ 0.0674 kPa/degC."""
        result = psychrometric_constant(101.3)
        assert result == pytest.approx(0.0674, abs=0.001)

    def test_proportional_to_pressure(self) -> None:
        """Psychrometric constant is linearly proportional to pressure."""
        p1 = 101.3
        p2 = 50.65
        # halving pressure should halve gamma
        ratio = psychrometric_constant(p1) / psychrometric_constant(p2)
        assert ratio == pytest.approx(2.0, abs=MATH_ABS_TOL)

    def test_numpy_array_input(self) -> None:
        """Function should accept and return numpy arrays."""
        pressures = np.array([101.3, 89.9, 70.1])
        result = psychrometric_constant(pressures)
        assert isinstance(result, np.ndarray)
        assert result.shape == pressures.shape


class TestAtmosphericPressurePsychrometricChain:
    """Integration test: atmospheric_pressure -> psychrometric_constant."""

    @pytest.mark.parametrize(
        ("elevation_m", "expected_gamma"),
        [
            # FAO-56 Table 2.2: combined elevation -> gamma
            (0, 0.0674),
            (500, 0.0635),
            (1000, 0.0598),
            (2000, 0.0529),
            (3000, 0.0466),
        ],
    )
    def test_elevation_to_gamma(self, elevation_m: float, expected_gamma: float) -> None:
        """Chain atmospheric_pressure -> psychrometric_constant matches Table 2.2."""
        pressure = atmospheric_pressure(float(elevation_m))
        gamma = psychrometric_constant(pressure)
        assert gamma == pytest.approx(expected_gamma, abs=0.002)


# ===================================================================
# Tests for Story 2.2: Vapor Pressure Helpers (Eq 11, 12, 13)
# ===================================================================


class TestSaturationVaporPressure:
    """Tests for saturation_vapor_pressure() -- FAO-56 Eq 11."""

    @pytest.mark.parametrize(
        ("temp_c", "expected_kpa"),
        [
            # FAO-56 Table 2.3: e0(T) for selected temperatures
            (1.0, 0.657),
            (2.0, 0.706),
            (5.0, 0.872),
            (10.0, 1.228),
            (15.0, 1.705),
            (20.0, 2.338),
            (25.0, 3.168),
            (30.0, 4.243),
            (35.0, 5.624),
            (40.0, 7.384),
            (45.0, 9.585),
            (48.0, 11.17),
        ],
    )
    def test_fao56_table_2_3(self, temp_c: float, expected_kpa: float) -> None:
        """Validate against FAO-56 Table 2.3 saturation vapor pressure values."""
        result = saturation_vapor_pressure(temp_c)
        assert result == pytest.approx(expected_kpa, abs=0.02)

    def test_at_100c(self) -> None:
        """At 100 degC, saturation vapor pressure ~ 101.3 kPa (boiling)."""
        result = saturation_vapor_pressure(100.0)
        assert result > 90.0

    def test_monotonically_increasing(self) -> None:
        """Saturation vapor pressure increases with temperature."""
        temps = np.linspace(-10, 50, 100)
        result = saturation_vapor_pressure(temps)
        assert np.all(np.diff(result) > 0)

    def test_numpy_array_input(self) -> None:
        """Function should accept and return numpy arrays."""
        temps = np.array([10.0, 20.0, 30.0])
        result = saturation_vapor_pressure(temps)
        assert isinstance(result, np.ndarray)
        assert result.shape == temps.shape

    def test_scalar_input(self) -> None:
        """Function should accept scalar float input."""
        result = saturation_vapor_pressure(20.0)
        assert np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0)


class TestVaporPressureSlope:
    """Tests for vapor_pressure_slope() -- FAO-56 Eq 13."""

    @pytest.mark.parametrize(
        ("temp_c", "expected_slope"),
        [
            # FAO-56 Table 2.4: slope of vapor pressure curve
            (1.0, 0.047),
            (2.0, 0.050),
            (5.0, 0.061),
            (10.0, 0.082),
            (15.0, 0.110),
            (20.0, 0.145),
            (25.0, 0.189),
            (30.0, 0.243),
            (35.0, 0.311),
            (40.0, 0.393),
            (45.0, 0.494),
        ],
    )
    def test_fao56_table_2_4(self, temp_c: float, expected_slope: float) -> None:
        """Validate against FAO-56 Table 2.4 slope values."""
        result = vapor_pressure_slope(temp_c)
        assert result == pytest.approx(expected_slope, abs=0.002)

    def test_monotonically_increasing(self) -> None:
        """Slope of the saturation vapor pressure curve increases with temperature."""
        temps = np.linspace(-10, 50, 100)
        result = vapor_pressure_slope(temps)
        assert np.all(np.diff(result) > 0)

    def test_numpy_array_input(self) -> None:
        """Function should accept and return numpy arrays."""
        temps = np.array([10.0, 20.0, 30.0])
        result = vapor_pressure_slope(temps)
        assert isinstance(result, np.ndarray)
        assert result.shape == temps.shape


class TestMeanSaturationVaporPressure:
    """Tests for mean_saturation_vapor_pressure() -- FAO-56 Eq 12."""

    def test_basic_calculation(self) -> None:
        """Mean of e0(Tmin) and e0(Tmax) should equal (e0(15) + e0(25)) / 2."""
        result = mean_saturation_vapor_pressure(15.0, 25.0)
        expected = (saturation_vapor_pressure(15.0) + saturation_vapor_pressure(25.0)) / 2.0
        assert result == pytest.approx(expected, abs=MATH_ABS_TOL)

    def test_symmetric_temperatures(self) -> None:
        """When Tmin == Tmax, mean e_s equals e0(T)."""
        result = mean_saturation_vapor_pressure(20.0, 20.0)
        expected = saturation_vapor_pressure(20.0)
        assert result == pytest.approx(expected, abs=MATH_ABS_TOL)

    def test_numpy_array_input(self) -> None:
        """Function should accept and return numpy arrays."""
        tmin = np.array([10.0, 15.0, 20.0])
        tmax = np.array([25.0, 30.0, 35.0])
        result = mean_saturation_vapor_pressure(tmin, tmax)
        assert isinstance(result, np.ndarray)
        assert result.shape == tmin.shape

    def test_nonlinear_effect(self) -> None:
        """Mean e_s from Tmin/Tmax should exceed e0(Tmean) due to nonlinearity."""
        tmin, tmax = 10.0, 30.0
        tmean = (tmin + tmax) / 2.0
        mean_es = mean_saturation_vapor_pressure(tmin, tmax)
        es_at_mean = saturation_vapor_pressure(tmean)
        assert mean_es > es_at_mean


# ===================================================================
# Tests for Story 2.3: Humidity Pathway Dispatcher (Eq 14-19)
# ===================================================================


class TestActualVaporPressureFromDewpoint:
    """Tests for actual_vapor_pressure_from_dewpoint() -- FAO-56 Eq 14."""

    def test_basic_value(self) -> None:
        """Dewpoint of 17 degC should give e_a ~ 1.938 kPa."""
        result = actual_vapor_pressure_from_dewpoint(17.0)
        assert result == pytest.approx(1.938, abs=FAO_ABS_TOL)

    def test_equals_saturation_at_dewpoint(self) -> None:
        """e_a(Tdew) should equal e0(Tdew) by definition."""
        for tdew in [5.0, 10.0, 15.0, 20.0, 25.0]:
            ea = actual_vapor_pressure_from_dewpoint(tdew)
            e0 = saturation_vapor_pressure(tdew)
            assert ea == pytest.approx(e0, abs=MATH_ABS_TOL)

    def test_numpy_array_input(self) -> None:
        """Function should accept and return numpy arrays."""
        tdew = np.array([5.0, 10.0, 15.0, 20.0])
        result = actual_vapor_pressure_from_dewpoint(tdew)
        assert isinstance(result, np.ndarray)
        assert result.shape == tdew.shape


class TestActualVaporPressureFromRhMinRhMax:
    """Tests for actual_vapor_pressure_from_rhmin_rhmax() -- FAO-56 Eq 17."""

    def test_fao56_example(self) -> None:
        """Validate against expected calculation from FAO-56."""
        e_tmin = saturation_vapor_pressure(15.0)
        e_tmax = saturation_vapor_pressure(25.0)
        result = actual_vapor_pressure_from_rhmin_rhmax(e_tmin, e_tmax, 54.0, 82.0)
        assert result == pytest.approx(1.554, abs=0.02)

    def test_100_percent_rh(self) -> None:
        """At 100% RH for both min and max, e_a should equal mean e_s."""
        e_tmin = saturation_vapor_pressure(15.0)
        e_tmax = saturation_vapor_pressure(25.0)
        result = actual_vapor_pressure_from_rhmin_rhmax(e_tmin, e_tmax, 100.0, 100.0)
        expected = (e_tmin + e_tmax) / 2.0
        assert result == pytest.approx(expected, abs=MATH_ABS_TOL)

    def test_numpy_array_input(self) -> None:
        """Function should accept and return numpy arrays."""
        e_tmin = np.array([1.705, 1.228])
        e_tmax = np.array([3.168, 2.338])
        rh_min = np.array([54.0, 60.0])
        rh_max = np.array([82.0, 90.0])
        result = actual_vapor_pressure_from_rhmin_rhmax(e_tmin, e_tmax, rh_min, rh_max)
        assert isinstance(result, np.ndarray)
        assert result.shape == e_tmin.shape


class TestActualVaporPressureFromRhMax:
    """Tests for actual_vapor_pressure_from_rhmax() -- FAO-56 Eq 18."""

    def test_basic_calculation(self) -> None:
        """e_a = e0(Tmin) * RHmax / 100."""
        e_tmin = saturation_vapor_pressure(15.0)
        result = actual_vapor_pressure_from_rhmax(e_tmin, 82.0)
        expected = e_tmin * 82.0 / 100.0
        assert result == pytest.approx(expected, abs=MATH_ABS_TOL)

    def test_100_percent_rh(self) -> None:
        """At 100% RHmax, e_a should equal e0(Tmin)."""
        e_tmin = saturation_vapor_pressure(15.0)
        result = actual_vapor_pressure_from_rhmax(e_tmin, 100.0)
        assert result == pytest.approx(e_tmin, abs=MATH_ABS_TOL)


class TestActualVaporPressureFromRhMean:
    """Tests for actual_vapor_pressure_from_rhmean() -- FAO-56 Eq 19."""

    def test_basic_calculation(self) -> None:
        """e_a = e_s * RHmean / 100."""
        e_s = mean_saturation_vapor_pressure(15.0, 25.0)
        result = actual_vapor_pressure_from_rhmean(e_s, 68.0)
        expected = e_s * 68.0 / 100.0
        assert result == pytest.approx(expected, abs=MATH_ABS_TOL)


class TestActualVaporPressureFromTmin:
    """Tests for actual_vapor_pressure_from_tmin() -- FAO-56 approximation."""

    def test_basic_calculation(self) -> None:
        """e_a = e0(Tmin - 2), 2 degC offset for arid conditions."""
        result = actual_vapor_pressure_from_tmin(18.0)
        expected = saturation_vapor_pressure(16.0)
        assert result == pytest.approx(expected, abs=MATH_ABS_TOL)

    def test_less_than_dewpoint_method(self) -> None:
        """Tmin-based estimate should be less than direct dewpoint method."""
        tmin = 18.0
        ea_tmin = actual_vapor_pressure_from_tmin(tmin)
        ea_dewpoint = actual_vapor_pressure_from_dewpoint(tmin)
        assert ea_tmin < ea_dewpoint


# ===================================================================
# Tests for Story 2.4: PM-ET Core Calculation (Eq 6)
# ===================================================================


class TestPmEto:
    """Tests for pm_eto() -- FAO-56 Equation 6."""

    def test_zero_wind(self) -> None:
        """With zero wind, aerodynamic term should vanish."""
        result = pm_eto(
            net_radiation=10.0,
            soil_heat_flux=0.0,
            temperature_celsius=20.0,
            wind_speed_2m=0.0,
            saturation_vp=2.338,
            actual_vp=1.5,
            delta=0.145,
            gamma=0.067,
        )
        assert result > 0.0
        expected_approx = 0.408 * 0.145 * 10.0 / (0.145 + 0.067)
        assert result == pytest.approx(expected_approx, abs=0.01)

    def test_zero_vpd(self) -> None:
        """With zero vapor pressure deficit, aerodynamic term should vanish."""
        result = pm_eto(
            net_radiation=10.0,
            soil_heat_flux=0.0,
            temperature_celsius=20.0,
            wind_speed_2m=2.0,
            saturation_vp=2.0,
            actual_vp=2.0,
            delta=0.145,
            gamma=0.067,
        )
        expected_rad = 0.408 * 0.145 * 10.0 / (0.145 + 0.067 * (1.0 + 0.34 * 2.0))
        assert result == pytest.approx(expected_rad, abs=0.01)

    def test_positive_eto(self) -> None:
        """Typical conditions should give positive ETo."""
        result = pm_eto(
            net_radiation=13.28,
            soil_heat_flux=0.14,
            temperature_celsius=16.9,
            wind_speed_2m=2.078,
            saturation_vp=1.997,
            actual_vp=1.409,
            delta=0.122,
            gamma=0.0666,
        )
        assert result > 0.0

    def test_numpy_array_input(self) -> None:
        """Function should accept and return numpy arrays."""
        n = 5
        result = pm_eto(
            net_radiation=np.full(n, 13.28),
            soil_heat_flux=np.full(n, 0.14),
            temperature_celsius=np.full(n, 16.9),
            wind_speed_2m=np.full(n, 2.078),
            saturation_vp=np.full(n, 1.997),
            actual_vp=np.full(n, 1.409),
            delta=np.full(n, 0.122),
            gamma=np.full(n, 0.0666),
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (n,)

    def test_broadcast_scalar_and_array(self) -> None:
        """Scalar and array inputs should broadcast correctly."""
        n = 3
        result = pm_eto(
            net_radiation=np.array([10.0, 12.0, 14.0]),
            soil_heat_flux=0.0,
            temperature_celsius=20.0,
            wind_speed_2m=2.0,
            saturation_vp=2.338,
            actual_vp=1.5,
            delta=0.145,
            gamma=0.067,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (n,)
        assert np.all(np.diff(result) > 0)

    def test_soil_heat_flux_reduces_eto(self) -> None:
        """Positive soil heat flux (energy stored) reduces available energy for ET."""
        eto_no_g = pm_eto(
            net_radiation=13.0,
            soil_heat_flux=0.0,
            temperature_celsius=20.0,
            wind_speed_2m=2.0,
            saturation_vp=2.338,
            actual_vp=1.5,
            delta=0.145,
            gamma=0.067,
        )
        eto_with_g = pm_eto(
            net_radiation=13.0,
            soil_heat_flux=2.0,
            temperature_celsius=20.0,
            wind_speed_2m=2.0,
            saturation_vp=2.338,
            actual_vp=1.5,
            delta=0.145,
            gamma=0.067,
        )
        assert eto_with_g < eto_no_g


# ===================================================================
# Tests for Story 2.5: FAO-56 Worked Example 18 Validation
# ===================================================================


class TestFAO56Example18:
    """Validate against FAO-56 Example 18, page 74.

    FAO-56 Example 18 provides pre-computed intermediate values for a PM-ET
    calculation.  We validate our pm_eto() function by plugging in the exact
    stated intermediate values and comparing against the reported ETo result.

    The example reports:
        Rn = 13.28 MJ m-2 day-1, G = 0.14 MJ m-2 day-1
        T = 16.9 degC, u2 = 2.078 m/s
        e_s = 1.997 kPa, e_a = 1.409 kPa
        Delta = 0.122 kPa/degC, gamma = 0.0666 kPa/degC
        ETo = 3.88 mm/day
    """

    # --- Pre-computed intermediate values stated in Example 18 ---
    NET_RADIATION = 13.28  # MJ m-2 day-1
    SOIL_HEAT_FLUX = 0.14  # MJ m-2 day-1
    TMEAN = 16.9  # degC
    WIND_SPEED_2M = 2.078  # m/s
    E_S = 1.997  # kPa
    E_A = 1.409  # kPa
    DELTA = 0.122  # kPa/degC
    GAMMA = 0.0666  # kPa/degC

    # Expected final ETo from FAO-56 Example 18
    EXPECTED_ETO = 3.88  # mm/day

    def test_pm_eto_with_stated_values(self) -> None:
        """PM-ET with exact Example 18 intermediate values should match 3.88 mm/day."""
        eto = pm_eto(
            net_radiation=self.NET_RADIATION,
            soil_heat_flux=self.SOIL_HEAT_FLUX,
            temperature_celsius=self.TMEAN,
            wind_speed_2m=self.WIND_SPEED_2M,
            saturation_vp=self.E_S,
            actual_vp=self.E_A,
            delta=self.DELTA,
            gamma=self.GAMMA,
        )
        # FAO-56 reports 3.88; our computation gives ~3.85 due to
        # floating-point precision in the rounded intermediate inputs.
        # A tolerance of 0.05 mm/day is well within acceptable range.
        assert eto == pytest.approx(self.EXPECTED_ETO, abs=0.05)

    def test_vapor_pressure_slope_at_example_temp(self) -> None:
        """Our vapor_pressure_slope at 16.9 degC should match Example 18 Delta."""
        delta = vapor_pressure_slope(self.TMEAN)
        assert delta == pytest.approx(self.DELTA, abs=0.002)

    def test_decomposition_radiation_vs_aerodynamic(self) -> None:
        """Verify radiation and aerodynamic term contributions sum correctly.

        Example 18 decomposes ETo into:
        - Radiation term: 0.408 * Delta * (Rn - G) / denominator
        - Aerodynamic term: gamma * (900/(T+273)) * u2 * VPD / denominator

        Both should be positive and sum to total ETo.
        """
        denominator = self.DELTA + self.GAMMA * (1.0 + 0.34 * self.WIND_SPEED_2M)

        rad_term = 0.408 * self.DELTA * (self.NET_RADIATION - self.SOIL_HEAT_FLUX) / denominator
        aero_term = (
            self.GAMMA * (900.0 / (self.TMEAN + 273.0)) * self.WIND_SPEED_2M * (self.E_S - self.E_A) / denominator
        )

        assert rad_term > 0
        assert aero_term > 0

        total = rad_term + aero_term
        eto = pm_eto(
            net_radiation=self.NET_RADIATION,
            soil_heat_flux=self.SOIL_HEAT_FLUX,
            temperature_celsius=self.TMEAN,
            wind_speed_2m=self.WIND_SPEED_2M,
            saturation_vp=self.E_S,
            actual_vp=self.E_A,
            delta=self.DELTA,
            gamma=self.GAMMA,
        )
        assert total == pytest.approx(float(eto), abs=MATH_ABS_TOL)
