#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for Task 4 (Student Union).

Keep all thresholds, weights, and scenario parameters centralized
for traceability and audit.
"""

from __future__ import annotations

from typing import Dict, List


def _month_list(value) -> List[float]:
    if isinstance(value, (list, tuple)) and len(value) == 12:
        return list(value)
    return [float(value)] * 12


SCENARIO_ORDER = ["now", "future2030", "future2050", "future2080"]

SCENARIOS: Dict[str, Dict] = {
    "now": {
        "label": "Now",
        "year": 2023,
        "deltas": {},
        "scales": {},
    },
    "future2030": {
        "label": "2030",
        "year": 2030,
        "deltas": {
            "Temp": _month_list(0.8),
            "DNI": _month_list(0.0),
            "DHI": _month_list(0.0),
            "GHI": _month_list(0.0),
        },
        "scales": {
            "DNI": _month_list(0.01),
            "DHI": _month_list(0.01),
            "GHI": _month_list(0.01),
        },
    },
    "future2050": {
        "label": "2050",
        "year": 2050,
        "deltas": {
            "Temp": _month_list(1.8),
            "DNI": _month_list(0.0),
            "DHI": _month_list(0.0),
            "GHI": _month_list(0.0),
        },
        "scales": {
            "DNI": _month_list(0.02),
            "DHI": _month_list(0.02),
            "GHI": _month_list(0.02),
        },
    },
    "future2080": {
        "label": "2080",
        "year": 2080,
        "deltas": {
            "Temp": _month_list(3.0),
            "DNI": _month_list(0.0),
            "DHI": _month_list(0.0),
            "GHI": _month_list(0.0),
        },
        "scales": {
            "DNI": _month_list(0.03),
            "DHI": _month_list(0.03),
            "GHI": _month_list(0.03),
        },
    },
}

MORPH_COLS = ["Temp", "DNI", "DHI", "GHI"]

SCENARIO_WEIGHTS = {
    "now": 0.25,
    "future2030": 0.25,
    "future2050": 0.25,
    "future2080": 0.25,
}


BUILDING = {
    "length_m": 60.0,
    "width_m": 30.0,
    "floors": 3,
    "story_height_m": 4.0,
    "window_height_m": 2.4,
    "window_width_m": 1.8,
    "albedo": 0.2,
    "cooling_setpoint_C": 24.0,
    "heating_setpoint_C": 20.0,
    "comfort_temp_C": 26.0,
    "internal_gain_W_m2": 12.0,
    "h_in": 3.0,
    "h_out": 25.0,
    "c_mass_base": 4.0e8,
    "r_mass": 0.15,
    "r_wall": 0.30,
    "fraction_solar_to_air": 0.35,
    "cop_cool": 3.0,
    "cop_heat": 3.0,
    "daylight_factor": 0.12,
    "luminous_efficacy": 120.0,
}

SCHEDULE = {
    "weekday_start": 8,
    "weekday_end": 22,
    "weekend_start": 10,
    "weekend_end": 20,
}

ECONOMICS = {
    "discount_rate": 0.05,
    "life_years": 20,
    "energy_price": 0.15,
    "baseline_energy_kwh_m2": 180.0,
    "mass_cost_per_m2": 20.0,
    "overhang_cost_per_m": 180.0,
    "fin_cost_per_m": 160.0,
}

DEFAULT_THRESHOLDS = {
    "x1": {"good": 60.0, "bad": 200.0},
    "x2": {"good": 10.0, "bad": 80.0},
    "x3": {"good": 30.0, "bad": 120.0},
    "x4": {"good": 0.0, "bad": 800.0},
    "x5": {"good": 0.5, "bad": 3.0},
    "x6": {"good": 0.0, "bad": 250.0},
    "x7": {"good": 0.0, "bad": 1500.0},
    "x8": {"good": 0.0, "bad": 200.0},
    "x9": {"good": 200.0, "bad": 1200.0},
}

THRESHOLD_RATIOS = {
    "x1": {"good": 0.8, "bad": 1.0},
    "x2": {"good": 0.8, "bad": 1.0},
    "x3": {"good": 0.85, "bad": 1.0},
    "x4": {"good": 0.8, "bad": 1.0},
    "x5": {"good": 0.8, "bad": 1.0},
    "x7": {"good": 0.8, "bad": 1.0},
    "x8": {"good": 0.8, "bad": 1.0},
    "x9": {"good": 0.85, "bad": 1.0},
}

BASELINE_U0 = {
    "gamma_build": 0.0,
    "wwr_N": 0.4,
    "wwr_E": 0.4,
    "wwr_S": 0.4,
    "wwr_W": 0.4,
    "veg_cover_S": 0.0,
    "veg_cover_W": 0.0,
    "jJ_SU": 1,
    "SHGC": 0.6,
    "VT": 0.6,
    "d_oh": 0.0,
    "h_fin": 0.0,
    "E_dir_in_th": 1000.0,
    "alpha_summer": 20.0,
    "alpha_winter": 0.0,
    "m_mass": 1.0,
}

RISK_WEIGHTS = {
    "x1": 0.18,
    "x2": 0.08,
    "x3": 0.12,
    "x4": 0.12,
    "x5": 0.08,
    "x6": 0.12,
    "x7": 0.10,
    "x8": 0.10,
    "x9": 0.10,
}

RISK_CONFIG = {
    "lambda1": 0.6,
    "lambda2": 0.4,
    "lambda_w": 0.2,
    "var_ddof": 0,
    "include_now_in_rob": False,
}

CONSTRAINTS = {
    "ase_hours_max": 250.0,
    "ld_hours_max": 1200.0,
    "pbp_max_years": 12.0,
    "npv_min": 0.0,
    "use_npv": False,
}

DGP_CONFIG = {
    "threshold": 0.45,
    "view_distance_m": 4.0,
    "cos_view_angle": 0.8,
    "position_index": 1.0,
}

NSGA2_CONFIG = {
    "pop_size": 60,
    "n_gen": 40,
    "seed": 42,
    "stage_a_stride": 3,
    "stage_b_topk": 10,
    "stage_b_stride": 1,
}
