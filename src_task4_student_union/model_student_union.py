#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Student Union model: decision variables and evaluation chain.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from . import ensure_src_on_path
from .config_task4 import BUILDING, ECONOMICS, SCHEDULE, DGP_CONFIG

ensure_src_on_path()

from economics import energy_cost, life_cycle_cost, NPV, payback_period
from fenestration import shgc_effective, vt_effective, glass_cost_per_m2
from irradiance_perez import tilted_irradiance
from schedule import student_union_schedule
from shading_geometry import compute_shading_factors
from solar_geometry import calculate_solar_position, incidence_cos_window
from thermal_loads import (
    simulate_2R1C,
    split_solar_to_air_and_mass,
    cooling_load_calculation,
    heating_load_calculation,
)
from visual_glare import ASE_1000, dgp_proxy


VAR_DEFS = [
    {"name": "gamma_build", "type": "float", "bounds": (0.0, 360.0)},
    {"name": "wwr_N", "type": "float", "bounds": (0.20, 0.70)},
    {"name": "wwr_E", "type": "float", "bounds": (0.20, 0.70)},
    {"name": "wwr_S", "type": "float", "bounds": (0.20, 0.70)},
    {"name": "wwr_W", "type": "float", "bounds": (0.20, 0.70)},
    {"name": "veg_cover_S", "type": "float", "bounds": (0.0, 0.9)},
    {"name": "veg_cover_W", "type": "float", "bounds": (0.0, 0.9)},
    {"name": "jJ_SU", "type": "int", "bounds": (1, 3)},
    {"name": "SHGC", "type": "float", "bounds": (0.20, 0.70)},
    {"name": "VT", "type": "float", "bounds": (0.30, 0.80)},
    {"name": "d_oh", "type": "float", "bounds": (0.0, 1.50)},
    {"name": "h_fin", "type": "float", "bounds": (0.0, 1.00)},
    {"name": "E_dir_in_th", "type": "float", "bounds": (500.0, 1200.0)},
    {"name": "alpha_summer", "type": "float", "bounds": (0.0, 60.0)},
    {"name": "alpha_winter", "type": "float", "bounds": (0.0, 60.0)},
    {"name": "m_mass", "type": "float", "bounds": (0.80, 3.00)},
]

VEG_COST_PER_M2 = 120.0


def decode_u(x: np.ndarray, var_defs: List[Dict] = VAR_DEFS) -> Dict[str, Any]:
    u = {}
    for val, spec in zip(x, var_defs):
        if spec["type"] == "int":
            u[spec["name"]] = int(round(val))
        else:
            u[spec["name"]] = float(val)
    return repair_u(u, var_defs)


def repair_u(u: Dict[str, Any], var_defs: List[Dict] = VAR_DEFS) -> Dict[str, Any]:
    for spec in var_defs:
        name = spec["name"]
        lo, hi = spec["bounds"]
        if spec["type"] == "int":
            u[name] = int(np.clip(int(round(u[name])), lo, hi))
        else:
            u[name] = float(np.clip(float(u[name]), lo, hi))
    if u["VT"] < 0.25:
        u["VT"] = 0.25
    return u


def serialize_u(u: Dict[str, Any]) -> str:
    return json.dumps(u, sort_keys=True)


def pretty_table(u: Dict[str, Any], output_csv: str | None = None) -> pd.DataFrame:
    df = pd.DataFrame([{"variable": k, "value": v} for k, v in u.items()])
    if output_csv:
        df.to_csv(output_csv, index=False)
    return df


def _building_geometry() -> Tuple[Dict[str, float], float, float]:
    length = BUILDING["length_m"]
    width = BUILDING["width_m"]
    floors = BUILDING["floors"]
    height = floors * BUILDING["story_height_m"]
    wall_areas = {
        "N": length * height,
        "S": length * height,
        "E": width * height,
        "W": width * height,
    }
    floor_area = length * width * floors
    envelope_area = 2 * (length + width) * height + (length * width)
    return wall_areas, floor_area, envelope_area


def evaluate_student_union(
    u: Dict[str, Any],
    epw_df: pd.DataFrame,
    meta: Dict,
    scenario: str = "now",
    stride: int = 1,
    return_timeseries: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate Student Union performance for one scenario.
    """
    if "datetime" in epw_df.columns:
        times = pd.DatetimeIndex(epw_df["datetime"])
    else:
        times = pd.DatetimeIndex(epw_df.index)

    if stride > 1:
        epw_df = epw_df.iloc[::stride].reset_index(drop=True)
        times = times[::stride]

    dni = epw_df["DNI"].astype(float).to_numpy()
    dhi = epw_df["DHI"].astype(float).to_numpy()
    ghi = epw_df["GHI"].astype(float).to_numpy()
    temp_out = epw_df["Temp"].astype(float).to_numpy()

    solar_pos = calculate_solar_position(
        times=times,
        latitude=meta.get("latitude", 0.0),
        longitude=meta.get("longitude", 0.0),
        altitude=meta.get("altitude", 0.0),
    )
    zen = solar_pos["zenith"].to_numpy()
    az = solar_pos["azimuth"].to_numpy()

    occ = student_union_schedule(
        times,
        weekday_start=SCHEDULE["weekday_start"],
        weekday_end=SCHEDULE["weekday_end"],
        weekend_start=SCHEDULE["weekend_start"],
        weekend_end=SCHEDULE["weekend_end"],
    ).astype(float)

    wall_areas, floor_area, envelope_area = _building_geometry()

    gamma = u["gamma_build"]
    base_azimuth = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}

    q_solar_total = np.zeros(len(times))
    illum_in = np.zeros(len(times))
    ase_weighted = 0.0
    total_window_area = 0.0
    direct_weighted = np.zeros(len(times))
    veg_area_total = 0.0

    jJ = max(1, int(u["jJ_SU"]))
    band_factor = max(0.85, 1.0 - 0.05 * (jJ - 1))

    summer_mask = (times.month >= 5) & (times.month <= 9)
    season_factor = np.where(summer_mask, 1.0, 0.3)
    alpha = np.where(summer_mask, u["alpha_summer"], u["alpha_winter"])
    louver_factor = np.cos(np.radians(alpha))
    louver_factor = np.clip(louver_factor, 0.2, 1.0)

    for facade in ["N", "E", "S", "W"]:
        wwr = u[f"wwr_{facade}"]
        wall_area = wall_areas[facade]
        window_area = wall_area * wwr
        total_window_area += window_area
        veg_cover = float(u.get(f"veg_cover_{facade}", 0.0))
        veg_area_total += window_area * veg_cover

        surf_az = (base_azimuth[facade] + gamma) % 360.0

        direct, diffuse, reflected = tilted_irradiance(
            surface_tilt=90.0,
            surface_azimuth=surf_az,
            solar_zenith=zen,
            solar_azimuth=az,
            dni=dni,
            dhi=dhi,
            ghi=ghi,
            albedo=BUILDING["albedo"],
        )

        cos_theta = incidence_cos_window(az, zen, surf_az, 90.0)
        shgc_eff = shgc_effective(u["SHGC"], cos_theta)
        vt_eff = vt_effective(u["VT"], cos_theta)

        S_b, F_sunlit = compute_shading_factors(
            solar_zenith=zen,
            solar_azimuth=az,
            window_azimuth=surf_az,
            overhang_depth=u["d_oh"],
            left_fin_depth=u["h_fin"],
            right_fin_depth=u["h_fin"],
            window_height=BUILDING["window_height_m"],
            window_width=BUILDING["window_width_m"],
        )
        S_b = S_b * band_factor
        F_sunlit = F_sunlit * band_factor

        E_dir_in = direct * vt_eff
        control_mask = E_dir_in > u["E_dir_in_th"]
        if np.any(control_mask):
            S_b = S_b.copy()
            F_sunlit = F_sunlit.copy()
            S_b[control_mask] *= louver_factor[control_mask]
            F_sunlit[control_mask] *= louver_factor[control_mask]

        S_b = np.clip(S_b, 0.0, 1.0)
        F_sunlit = np.clip(F_sunlit, 0.0, 1.0)

        tau_leaf = 0.2
        S_veg = 1.0 - (veg_cover * season_factor * (1.0 - tau_leaf))
        S_veg = np.clip(S_veg, 0.0, 1.0)

        direct_shaded = direct * S_b * S_veg
        diffuse_shaded = diffuse * 0.5 * S_veg
        reflected_shaded = reflected * 1.0

        q_solar_f = shgc_eff * window_area * (direct_shaded + diffuse_shaded + reflected_shaded)
        q_solar_total += q_solar_f
        direct_weighted += direct_shaded * window_area

        illum_window = (direct_shaded + diffuse_shaded) * vt_eff * BUILDING["luminous_efficacy"]
        illum_in += illum_window * (window_area / max(floor_area, 1.0)) * BUILDING["daylight_factor"]

        ase_hours = ASE_1000(direct_shaded, F_sunlit, threshold=1000.0, VT=u["VT"])
        ase_weighted += ase_hours * window_area

    ase_hours_total = 0.0 if total_window_area <= 0 else ase_weighted / total_window_area

    q_internal = BUILDING["internal_gain_W_m2"] * floor_area * occ
    q_solar_air, q_solar_mass = split_solar_to_air_and_mass(
        q_solar_total,
        fraction_to_air=BUILDING["fraction_solar_to_air"],
    )

    c_mass = BUILDING["c_mass_base"] * u["m_mass"]
    temp_air, temp_mass = simulate_2R1C(
        temp_outdoor=temp_out,
        q_solar_a=q_solar_air,
        q_solar_m=q_solar_mass,
        q_internal=q_internal,
        h_in=BUILDING["h_in"],
        h_out=BUILDING["h_out"],
        c_mass=c_mass,
        r_mass=BUILDING["r_mass"],
        r_wall=BUILDING["r_wall"],
        area=envelope_area,
    )

    cooling_load = cooling_load_calculation(
        temp_air,
        temp_setpoint=BUILDING["cooling_setpoint_C"],
        area=envelope_area,
        h_in=BUILDING["h_in"],
    )
    heating_load = heating_load_calculation(
        temp_air,
        temp_setpoint=BUILDING["heating_setpoint_C"],
        area=envelope_area,
        h_in=BUILDING["h_in"],
    )

    dt_hours = float(stride)
    occ_mask = occ > 0.5
    cooling_kwh = np.sum(cooling_load[occ_mask]) * dt_hours / 1000.0 / BUILDING["cop_cool"]
    heating_kwh = np.sum(heating_load[occ_mask]) * dt_hours / 1000.0 / BUILDING["cop_heat"]

    x1 = cooling_kwh / max(floor_area, 1.0)
    x2 = heating_kwh / max(floor_area, 1.0)
    x3 = float(np.max(cooling_load) / max(floor_area, 1.0))
    x4 = float(np.sum(np.maximum(0.0, temp_air - BUILDING["comfort_temp_C"])[occ_mask]) * dt_hours)
    x5 = float(np.std(temp_air[occ_mask])) if np.any(occ_mask) else 0.0
    x6 = float(ase_hours_total)
    x7 = float(np.sum((illum_in[occ_mask] < 500.0).astype(int)) * dt_hours)
    x8 = 0.0

    window_area_total = total_window_area
    window_cost = window_area_total * glass_cost_per_m2(u["SHGC"], u["VT"])
    overhang_length = window_area_total / max(BUILDING["window_height_m"], 0.1)
    fin_height_total = window_area_total / max(BUILDING["window_width_m"], 0.1)
    overhang_cost = u["d_oh"] * overhang_length * ECONOMICS["overhang_cost_per_m"]
    fin_cost = u["h_fin"] * fin_height_total * ECONOMICS["fin_cost_per_m"]
    mass_cost = max(0.0, u["m_mass"] - 1.0) * floor_area * ECONOMICS["mass_cost_per_m2"]
    veg_cost = veg_area_total * VEG_COST_PER_M2
    capex = window_cost + overhang_cost + fin_cost + mass_cost + veg_cost

    annual_energy_kwh = cooling_kwh + heating_kwh
    annual_energy_cost = energy_cost(annual_energy_kwh, ECONOMICS["energy_price"])
    lcc = life_cycle_cost(
        initial_investment=capex,
        annual_operating_cost=annual_energy_cost,
        discount_rate=ECONOMICS["discount_rate"],
        life_years=ECONOMICS["life_years"],
    )

    baseline_energy = ECONOMICS["baseline_energy_kwh_m2"] * floor_area
    annual_savings = max(0.0, baseline_energy - annual_energy_kwh) * ECONOMICS["energy_price"]
    pbp = payback_period(capex, annual_savings)
    npv = NPV(annual_savings, ECONOMICS["discount_rate"], years=ECONOMICS["life_years"]) - capex
    x9 = lcc / max(floor_area, 1.0)

    # DGP sampling proxy (use average window as dominant glare source)
    omega_sr = 0.0
    if total_window_area > 0.0:
        omega_sr = (total_window_area / max(DGP_CONFIG["view_distance_m"] ** 2, 0.1)) * DGP_CONFIG["cos_view_angle"]
    omega_sr = float(np.clip(omega_sr, 1e-6, 2.0 * np.pi))
    ev_lux = np.maximum(illum_in, 1.0)
    dgp_series = dgp_proxy(ev_lux, omega_sr, position_index=DGP_CONFIG["position_index"])
    dgp_hours = float(np.sum((dgp_series > DGP_CONFIG["threshold"]) & occ_mask) * dt_hours)
    x6 = float(max(x6, dgp_hours))

    timeseries = None
    if return_timeseries:
        timeseries = pd.DataFrame({
            "time": times,
            "T_out": temp_out,
            "T_in": temp_air,
            "Q_solar_W": q_solar_total,
            "Cooling_Load_W": cooling_load,
            "Heating_Load_W": heating_load,
            "Illuminance_in_lux": illum_in,
            "Occupied": occ_mask.astype(int),
            "DGP": dgp_series,
        })

    return {
        "scenario": scenario,
        "x": {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "x5": x5,
            "x6": x6,
            "x7": x7,
            "x8": x8,
            "x9": x9,
        },
        "constraints": {
            "ase_hours": ase_hours_total,
            "ld_hours": x7,
            "pbp_years": pbp,
            "npv": npv,
            "dgp_hours": dgp_hours,
        },
        "costs": {
            "capex": capex,
            "lcc": lcc,
            "annual_energy_kwh": annual_energy_kwh,
            "annual_energy_cost": annual_energy_cost,
            "pbp_years": pbp,
            "npv": npv,
        },
        "kpi_summary": {
            "cooling_kwh": cooling_kwh,
            "heating_kwh": heating_kwh,
            "floor_area_m2": floor_area,
        },
        "diagnostics": {
            "window_area_total": window_area_total,
            "band_factor": band_factor,
            "stride": stride,
            "dgp_hours": dgp_hours,
        },
        "timeseries": timeseries,
    }
