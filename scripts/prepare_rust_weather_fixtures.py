"""Freeze Python weather regressions and copy existing external references for Rust.

Run from the repository root with PYTHONPATH=$PWD/src. No extension is required.
Cargo never runs this script. See crates/climate-core/tests/fixtures/weather_reference.md.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Literal

# Block extension imports before importing any package module, even if it is installed.
sys.modules["climate_indices._native"] = None  # type: ignore[assignment]

import numpy as np  # noqa: E402

from climate_indices import eto, fire, flood, pm_eto  # noqa: E402
from climate_indices.fire import _cffwis_codes as codes  # noqa: E402
from climate_indices.fire import _native as fire_native  # noqa: E402
from climate_indices.flood import _native as flood_native  # noqa: E402

SOURCE_REVISION = "f4cb938eb622a3ddf8b96c129337e0ebecd4121e"
POLICIES: tuple[tuple[Literal["propagate", "bridge"], int], ...] = (("propagate", 0), ("bridge", 2))
ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "crates/climate-core/tests/fixtures"
COMMAND = "PYTHONPATH=src uv run python scripts/prepare_rust_weather_fixtures.py"


def save(name: str, arrays: dict[str, np.ndarray], units: str, *, external: str | None = None) -> None:
    """Write named row-major matrices with round-trip float64 precision."""
    classification = external or "Python regression; climate_indices pure-Python output, not an independent reference"
    lines = [
        f"# Source revision: {SOURCE_REVISION}",
        f"# Generation command: {COMMAND}",
        f"# Classification: {classification}",
        f"# Units: {units}",
        "# Shapes: each record is NAME ROWS COLUMNS followed by ROWS numeric lines; C-order, NaN spelled nan.",
        "# Details and citations: weather_reference.md. No Rust output is used.",
        "# Tolerance: rtol=atol=1e-10 and exact NaN masks for Python regressions; external bounds are in the documentation.",
        f"# Generator runtime: Python {sys.version.split()[0]}, NumPy {np.__version__}",
    ]
    for key, values in arrays.items():
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim < 2:
            matrix = matrix.reshape(1, -1)
        assert matrix.ndim == 2, (key, matrix.shape)
        assert not np.isinf(matrix).any(), key
        lines.append(f"{key} {matrix.shape[0]} {matrix.shape[1]}")
        lines.extend(" ".join(format(value, ".17g") for value in row) for row in matrix)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / name).write_text("\n".join(lines) + "\n")


def temperature_pet() -> None:
    """Cover leap years, hemispheres, poles, gaps, and a partial 366-day year."""
    latitude = np.array([35.0, -35.0, 90.0, -90.0])
    months = np.arange(24)[:, None]
    temperatures = -4.0 + (months * 7 + np.arange(4) * 3) % 33
    temperatures[5, 0] = np.nan
    monthly_pet = eto.eto_thornthwaite(
        temperatures.reshape(24, 2, 2).copy(), latitude.reshape(2, 2), 1999, spatial_time_major=True
    )
    days = np.arange(370)[:, None]
    tmin = -3.0 + (days * 3 + np.arange(4) * 5) % 25
    tmax = tmin + 10.0
    tmax[3, 0] = tmin[3, 0] - 1.0
    tmax[4, 1] = tmin[4, 1]
    tmin[8, 2] = np.nan
    tmean = (tmin + tmax) / 2.0
    daily_pet = eto.eto_hargreaves(
        tmin.reshape(370, 2, 2),
        tmax.reshape(370, 2, 2),
        tmean.reshape(370, 2, 2),
        latitude.reshape(2, 2),
        spatial_time_major=True,
    )
    save(
        "weather_eto.txt",
        {
            "monthly_temperature": np.maximum(temperatures, 0.0),
            "latitude_radians": np.deg2rad(latitude),
            "leap_years": np.array([0, 1]),
            "thornthwaite": monthly_pet.reshape(24, 4),
            "tmin": tmin,
            "tmax": tmax,
            "tmean": tmean,
            "hargreaves": daily_pet.reshape(370, 4),
        },
        "temperature: degrees C (monthly input clamped at zero); latitude: radians; leap flags: 0/1; Thornthwaite: mm/month; Hargreaves: mm/day",
    )


def penman_monteith() -> None:
    """Exercise every humidity/radiation combination, including polar night."""
    arrays = {
        "net_radiation": np.array([13.28, 12.5, 14.0, 0.0, np.nan, 8.0]),
        "soil_heat_flux": np.array([0.14, 0.0, 0.3, 0.0, 0.0, -0.5]),
        "temperature": np.array([16.9, 20.0, 5.0, -5.0, 15.0, np.nan]),
        "wind_2m": np.array([2.078, 1.5, 3.0, 0.0, 1.0, 2.0]),
        "saturation_vp": np.array([1.997, 2.3, 0.9, 0.5, 2.0, 1.0]),
        "actual_vp": np.array([1.409, 1.1, 0.7, 0.5, 1.0, 0.8]),
        "delta": np.array([0.122, 0.145, 0.06, 0.03, 0.1, 0.1]),
        "gamma": np.array([0.0666, 0.067, 0.05, 0.06, 0.06, 0.06]),
    }
    arguments = tuple(arrays.values())
    arrays["core_0"] = np.asarray(pm_eto.pm_eto(*arguments))
    arrays["core_1"] = np.asarray(pm_eto.pm_eto(arguments[0], *(value[0] for value in arguments[1:])))
    arrays["core_2"] = np.asarray(pm_eto.pm_eto(*(value[0] for value in arguments)))
    arrays.update(
        {
            "met_tmin": np.array([12.3, 8.0, -4.0, np.nan, 12.3, 12.3, 12.3, 10.0]),
            "met_tmax": np.array([21.5, 20.0, 8.0, 20.0, 21.5, 21.5, 21.5, 10.0]),
            "latitude_degrees": np.array([50.8, 35.0, -35.0, 0.0, 90.0, -90.0, 50.8, 0.0]),
            "wind": np.array([2.78, 0.0, 5.0, 2.0, 2.78, 2.78, np.nan, 2.0]),
            "day": np.array([187.0, 58.0, 360.0, 187.0, 187.0, 187.0, 187.0, 1.0]),
        }
    )
    humidity = [
        pm_eto.HumidityInputs(tdew_celsius=12.05),
        pm_eto.HumidityInputs(rh_min=63.0, rh_max=84.0),
        pm_eto.HumidityInputs(rh_max=84.0),
        pm_eto.HumidityInputs(rh_mean=70.0),
        None,
    ]
    radiation = [
        pm_eto.RadiationInputs(solar_radiation_mj_m2_day=22.0),
        pm_eto.RadiationInputs(sunshine_hours=9.25),
        pm_eto.RadiationInputs(coastal=True),
        pm_eto.RadiationInputs(),
    ]
    for h, humidity_input in enumerate(humidity):
        for r, radiation_input in enumerate(radiation):
            arrays[f"met_{h}_{r}"] = np.asarray(
                pm_eto.penman_monteith_eto(
                    arrays["met_tmin"],
                    arrays["met_tmax"],
                    arrays["latitude_degrees"],
                    100.0,
                    arrays["wind"],
                    arrays["day"],
                    wind_speed_height_m=10.0,
                    humidity=humidity_input,
                    radiation=radiation_input,
                )
            )
    save(
        "weather_pm.txt",
        arrays,
        "temperature: degrees C; radiation/heat flux: MJ m-2 day-1; wind: m/s; vapor pressures: kPa; delta/gamma: kPa/degree C; latitude: degrees north; day: 1-based; outputs: mm/day",
    )


def fire_recurrences() -> None:
    """Freeze histories, final states, and gap counts across all latitude bands."""
    day, cell = np.arange(18)[:, None], np.arange(5)
    temperature = (-5.0 + (day * 7 + cell * 11) % 46).astype(float)
    temperature[0, 1] = temperature[4, 2] = temperature[17, 4] = np.nan
    temperature[11:14, 3] = np.nan
    humidity = (day * 13.0 + cell * 17.0) % 101
    wind = (day * 2.0 + cell) % 12
    rain = np.array([0.0, 0.5, 1.5, 2.8, 5.08, 8.0, 30.0, 0.0])[(day + cell) % 8]
    months = (day + cell * 2) % 12 + 1
    latitude = np.array([55.0, 20.0, 0.0, -20.0, -40.0])
    season = np.ones((18, 5), dtype=bool)
    season[8:10] = False
    arrays = {
        "temperature": temperature,
        "humidity": humidity,
        "wind_kmh": wind * 3.6,
        "rain": rain,
        "months": months,
        "latitude_degrees": latitude,
        "season": season,
        "dmc_table": codes._DMC_EFFECTIVE_DAY_LENGTH_HOURS,
        "dc_table": codes._DC_DAY_LENGTH_ADJUSTMENT,
        "dmc_band": codes._dmc_day_length_band(latitude),
        "dc_band": codes._dc_day_length_band(latitude),
        "initial_ffmc": np.array([85.0, 20.0, 50.0, 90.0, 95.0]),
        "initial_dmc": np.array([6.0, 34.0, 66.0, 120.0, 0.0]),
        "initial_dc": np.array([15.0, 100.0, 400.0, 800.0, 0.0]),
        "initial_kbdi": np.array([50.0, 100.0, 150.0, 200.0, 0.0]),
        "annual_rain": np.array([250.0, 500.0, 1000.0, 1500.0, 2000.0]),
    }
    for policy, max_gap in POLICIES:
        results = {
            "ffmc": fire.ffmc(
                temperature,
                humidity,
                wind,
                rain,
                initial_ffmc=arrays["initial_ffmc"],
                return_state=True,
                spin_up=2,
                nan_policy=policy,
                max_gap_days=max_gap,
            ),
            "dmc": fire.duff_moisture_code(
                temperature,
                humidity,
                rain,
                latitude,
                months,
                initial_dmc=arrays["initial_dmc"],
                return_state=True,
                spin_up=2,
                nan_policy=policy,
                max_gap_days=max_gap,
            ),
            "dc": fire.drought_code(
                temperature,
                rain,
                latitude,
                months,
                in_season=season,
                initial_dc=arrays["initial_dc"],
                return_state=True,
                spin_up=2,
                nan_policy=policy,
                max_gap_days=max_gap,
            ),
            "kbdi": fire.kbdi(
                rain,
                temperature,
                arrays["annual_rain"],
                initial_kbdi=arrays["initial_kbdi"],
                return_state=True,
                spin_up=2,
                nan_policy=policy,
                max_gap_days=max_gap,
            ),
        }
        for name, result in results.items():
            key = f"{name}_{policy}"
            arrays[key] = result.values
            arrays[f"{key}_state"] = getattr(result.state, name)
            arrays[f"{key}_gaps"] = result.state.trailing_gap_days
            if name == "kbdi":
                arrays[f"{key}_wet_spell"] = result.state.wet_spell_precipitation
    save(
        "weather_fire.txt",
        arrays,
        "temperature: degrees C; humidity: percent; wind: km/h; rain/annual rain/KBDI/wet spell: mm; DMC table: hours; DC table: equation adjustment; codes: dimensionless; month: 1-based; bands/gaps: integer; season: 0/1",
    )


def flood_indices() -> None:
    """Include warm-up, gap recovery, constant/sparse cells, and a partial year."""
    days = 3 * 366 + 17
    time = np.arange(days)
    rain = np.empty((days, 3))
    rain[:, 0] = ((time * 7) % 19) * (1.0 + time // 366) / 3.0
    rain[400, 0] = np.nan
    rain[:, 1] = 2.0
    rain[:, 2] = np.where(time < 366, (time % 11).astype(float), np.nan)
    block = rain.reshape(days, 3, 1)
    arrays = {"rain": rain}
    for duration in [1, 2, 7]:
        arrays[f"pe_{duration}"] = flood.effective_precipitation(block, duration=duration).reshape(days, 3)
    pe = arrays["pe_7"].reshape(days, 3, 1)
    padded = np.full((4 * 366, 3), np.nan)
    padded[:days] = arrays["pe_7"]
    arrays["pe_padded"] = padded.reshape(4, 366 * 3)
    edi = flood.edi(pe, 2000, 2000, 2002).reshape(days, 3)
    expected = np.full_like(padded, np.nan)
    expected[:days] = edi
    arrays["edi"] = expected.reshape(4, 366 * 3)
    arrays["flood_index"] = flood.flood_index(pe, 2000, 2000, 2002, year_start_month=1).reshape(days, 3)
    api_rain = rain[:18].copy()
    api_rain[0, 1] = api_rain[4, 1] = api_rain[-1, 1] = np.nan
    api_rain[10:13, 2] = np.nan
    arrays["api_rain"] = api_rain
    for policy, max_gap in POLICIES:
        result = flood.antecedent_precipitation_index(
            api_rain.reshape(18, 3, 1),
            0.9,
            return_state=True,
            spin_up=2,
            nan_policy=policy,
            max_gap_days=max_gap,
        )
        key = f"api_{policy}"
        arrays[key] = result.values.reshape(16, 3)
        arrays[f"{key}_state"] = result.state.api.ravel()
        assert result.state.trailing_gap_days is not None
        arrays[f"{key}_gaps"] = result.state.trailing_gap_days.ravel()
    save("weather_flood.txt", arrays, "rain/PE/API: mm; EDI/Flood Index: dimensionless; gaps: integer days")


def external_references() -> None:
    """Copy external values unchanged; do not recompute them with this package."""
    pet = ROOT / "tests/fixture/pet_literature"
    save(
        "weather_literature.txt",
        {
            "watson_temperature": np.load(pet / "thornthwaite_monthly_temps_celsius.npy"),
            "watson_pet": np.load(pet / "thornthwaite_expected_pet_mm.npy"),
            "mehta_temperature": np.load(pet / "hargreaves_daily_tmin_tmax_tmean_celsius.npy"),
            "mehta_pet": np.load(pet / "hargreaves_expected_eto_mm_per_day.npy"),
            "met_tmin": np.array([12.3]),
            "met_tmax": np.array([21.5]),
            "latitude_degrees": np.array([50.8]),
            "wind": np.array([2.78]),
            "day": np.array([187.0]),
            "fao56_pet": np.array([3.88]),
        },
        "temperature: degrees C; Thornthwaite: mm/month; daily PET: mm/day; latitude: degrees north; wind: m/s; day: 1-based",
        external="External literature worked examples; see weather_reference.md for reconstructed-coordinate limitations",
    )
    with (ROOT / "tests/fixture/cffwis_vwp1985/reference.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = {key: np.array([float(row[key]) for row in rows])[:, None] for key in rows[0]}
    save(
        "weather_nrcan.txt",
        {
            "temperature": columns["temperature_celsius"],
            "humidity": columns["relative_humidity_percent"],
            "wind_kmh": columns["wind_speed_kmh"],
            "rain": columns["precipitation_mm"],
            "months": columns["month"],
            "ffmc": columns["ffmc"],
            "dmc": columns["dmc"],
            "dc": columns["dc"],
            "initial_ffmc": np.array([85.0]),
            "initial_dmc": np.array([6.0]),
            "initial_dc": np.array([15.0]),
            "dmc_table": codes._DMC_EFFECTIVE_DAY_LENGTH_HOURS,
            "dc_table": codes._DC_DAY_LENGTH_ADJUSTMENT,
        },
        "temperature: degrees C; humidity: percent; wind: km/h; rain: mm; codes: dimensionless; DMC table: hours; DC table: equation adjustment; month: 1-based",
        external="External NRCan cffdrs_py reference, commit 0f57fcca2a6a84b69fe8d50f29d947f0be34d5f6; copied, not recalculated",
    )


def main() -> None:
    """Generate all weather fixtures with extension dispatch explicitly disabled."""
    subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            SOURCE_REVISION,
            "--",
            "src/climate_indices",
            "tests/fixture/pet_literature",
            "tests/fixture/cffwis_vwp1985",
            "tests/test_pm_eto.py",
        ],
        cwd=ROOT,
        check=True,
    )
    # Expected values must come from this checkout, not an installed copy.
    assert all(
        Path(module.__file__).resolve().is_relative_to(ROOT / "src")
        for module in (eto, pm_eto, fire, flood, codes, fire_native, flood_native)
    )
    eto._native = None
    pm_eto._native = None
    fire_native._native = None
    flood_native._native = None
    assert all(value is None for value in (eto._native, pm_eto._native, fire_native._native, flood_native._native))
    assert sys.modules.get("climate_indices._native") is None
    with np.errstate(all="ignore"):
        temperature_pet()
        penman_monteith()
        fire_recurrences()
        flood_indices()
        external_references()


if __name__ == "__main__":
    main()
