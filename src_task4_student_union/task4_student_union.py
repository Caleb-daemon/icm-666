#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 4 runner for Student Union robust optimization.
"""

from __future__ import annotations

import argparse
import os

import json
from typing import Dict

from .config_task4 import (
    SCENARIOS,
    DEFAULT_THRESHOLDS,
    THRESHOLD_RATIOS,
    BASELINE_U0,
    RISK_WEIGHTS,
    RISK_CONFIG,
    CONSTRAINTS,
)
from .morphing import generate_future_epws
from .nsga2_task4 import run_nsga2_task4
from .plots_task4 import run_all_plots
from .model_student_union import evaluate_student_union, repair_u
from .risk import compute_robust_risk

from . import ensure_src_on_path
ensure_src_on_path()
from io_epw import read_epw
from .config_task4 import SCENARIO_ORDER, SCENARIO_WEIGHTS


def run_task4(
    epw_path: str,
    results_dir: str,
    pop_size: int | None = None,
    n_gen: int | None = None,
    seed: int | None = None,
    no_plots: bool = False,
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    _, epw_paths = generate_future_epws(
        epw_path=epw_path,
        scenarios=SCENARIOS,
        results_dir=results_dir,
        recompute_ghi=True,
    )

    thresholds = build_thresholds(epw_path, results_dir)
    print("=== Task 4 Config: NORM-1 thresholds ===")
    for key, vals in thresholds.items():
        print(f"  {key}: good={vals['good']}, bad={vals['bad']}")
    print("=== Task 4 Config: R-1 weights ===")
    for key, val in RISK_WEIGHTS.items():
        print(f"  {key}: {val}")
    print("=== Task 4 Config: Robust risk ===")
    print(RISK_CONFIG)
    print("=== Task 4 Config: Constraints ===")
    print(CONSTRAINTS)

    run_nsga2_task4(
        epw_paths=epw_paths,
        results_dir=results_dir,
        thresholds=thresholds,
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed,
    )

    if not no_plots:
        run_all_plots(results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="ICM 2026 Problem E Task 4 (Student Union)")
    parser.add_argument(
        "--epw",
        type=str,
        default=os.path.join("data", "epw", "HKG_NT_Lau.Fau.Shan.450350_TMYx.2009-2023.epw"),
        help="Base EPW file path.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join("results", "task4"),
        help="Results directory.",
    )
    parser.add_argument("--pop", type=int, default=None, help="NSGA-II population size.")
    parser.add_argument("--gen", type=int, default=None, help="NSGA-II generations.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--no_plots", action="store_true", help="Disable plot generation.")
    args = parser.parse_args()

    run_task4(
        epw_path=args.epw,
        results_dir=args.results_dir,
        pop_size=args.pop,
        n_gen=args.gen,
        seed=args.seed,
        no_plots=args.no_plots,
    )


def build_thresholds(epw_path: str, results_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Build thresholds from baseline u0 (v3 anchor) with fallback defaults.
    """
    os.makedirs(results_dir, exist_ok=True)
    df_now, meta_now = read_epw(epw_path)
    u0 = repair_u(dict(BASELINE_U0))
    baseline_now = evaluate_student_union(u0, df_now, meta_now, scenario="now", stride=1, return_timeseries=False)
    metrics_now = baseline_now["x"]

    thresholds: Dict[str, Dict[str, float]] = {}
    eps = 1e-6

    # Hard standard for glare (ASE_1000,250)
    thresholds["x6"] = {"good": 0.0, "bad": 250.0}

    for key, ratios in THRESHOLD_RATIOS.items():
        if key == "x6":
            continue
        base_val = metrics_now.get(key, 0.0)
        if base_val <= eps:
            thresholds[key] = DEFAULT_THRESHOLDS.get(key, {"good": 0.0, "bad": 1.0})
            continue
        thresholds[key] = {
            "good": ratios["good"] * base_val,
            "bad": ratios["bad"] * base_val,
        }

    # Compute Rob baseline to anchor x8 thresholds (requires scenarios)
    metrics_by_s = {"now": metrics_now}
    for scenario, cfg in SCENARIOS.items():
        if scenario == "now":
            continue
        path = os.path.join(results_dir, "epw_future", f"EPW_future_{cfg['year']}.epw")
        if not os.path.exists(path):
            continue
        df_s, meta_s = read_epw(path)
        res_s = evaluate_student_union(u0, df_s, meta_s, scenario=scenario, stride=1, return_timeseries=False)
        metrics_by_s[scenario] = res_s["x"]

    risk = compute_robust_risk(
        metrics_by_scenario=metrics_by_s,
        thresholds=thresholds,
        weights=RISK_WEIGHTS,
        scenario_weights=SCENARIO_WEIGHTS,
        risk_config=RISK_CONFIG,
    )
    rob0 = risk["Rob"]
    if "x8" in THRESHOLD_RATIOS and rob0 > eps:
        ratios = THRESHOLD_RATIOS["x8"]
        thresholds["x8"] = {"good": ratios["good"] * rob0, "bad": ratios["bad"] * rob0}
    else:
        thresholds["x8"] = DEFAULT_THRESHOLDS.get("x8", {"good": 0.0, "bad": 1.0})

    thresholds_path = os.path.join(results_dir, "thresholds_used.json")
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    return thresholds


if __name__ == "__main__":
    main()
