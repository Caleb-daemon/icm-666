#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSGA-II wrapper for Task 4 (Student Union).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import ensure_src_on_path
from .config_task4 import (
    SCENARIO_ORDER,
    SCENARIO_WEIGHTS,
    DEFAULT_THRESHOLDS,
    RISK_WEIGHTS,
    RISK_CONFIG,
    CONSTRAINTS,
    NSGA2_CONFIG,
)
from .model_student_union import VAR_DEFS, decode_u, serialize_u, evaluate_student_union, pretty_table
from .risk import compute_robust_risk

ensure_src_on_path()

from io_epw import read_epw

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination


def _bounds_from_vars() -> Tuple[np.ndarray, np.ndarray]:
    xl = []
    xu = []
    for spec in VAR_DEFS:
        lo, hi = spec["bounds"]
        xl.append(lo)
        xu.append(hi)
    return np.array(xl, dtype=float), np.array(xu, dtype=float)


def _eval_cached(
    u: Dict,
    scenario: str,
    epw_df: pd.DataFrame,
    meta: Dict,
    cache: Dict,
    stride: int,
) -> Dict:
    key = (serialize_u(u), scenario, stride)
    if key in cache:
        return cache[key]
    result = evaluate_student_union(u, epw_df, meta, scenario=scenario, stride=stride, return_timeseries=False)
    cache[key] = result
    return result


class StudentUnionOptimization(ElementwiseProblem):
    def __init__(
        self,
        epw_by_scenario: Dict[str, pd.DataFrame],
        meta_by_scenario: Dict[str, Dict],
        stride: int = 1,
        thresholds: Dict[str, Dict[str, float]] | None = None,
    ):
        xl, xu = _bounds_from_vars()
        super().__init__(n_var=len(VAR_DEFS), n_obj=2, n_constr=3, xl=xl, xu=xu)
        self.epw_by_scenario = epw_by_scenario
        self.meta_by_scenario = meta_by_scenario
        self.stride = stride
        self.cache: Dict = {}
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    def _evaluate(self, x, out, *args, **kwargs):
        u = decode_u(np.array(x, dtype=float))
        metrics_by_s = {}
        constraints_by_s = {}
        for scenario in SCENARIO_ORDER:
            epw_df = self.epw_by_scenario[scenario]
            meta = self.meta_by_scenario[scenario]
            result = _eval_cached(u, scenario, epw_df, meta, self.cache, self.stride)
            metrics_by_s[scenario] = result["x"]
            constraints_by_s[scenario] = result["constraints"]

        risk = compute_robust_risk(
            metrics_by_scenario=metrics_by_s,
            thresholds=self.thresholds,
            weights=RISK_WEIGHTS,
            scenario_weights=SCENARIO_WEIGHTS,
            risk_config=RISK_CONFIG,
        )

        cost_value = metrics_by_s["now"]["x9"]
        out["F"] = [risk["R_rob"], cost_value]

        max_ase = max(c["ase_hours"] for c in constraints_by_s.values())
        max_ld = max(c["ld_hours"] for c in constraints_by_s.values())

        if CONSTRAINTS["use_npv"]:
            min_npv = min(c["npv"] for c in constraints_by_s.values())
            g3 = CONSTRAINTS["npv_min"] - min_npv
        else:
            max_pbp = max(c["pbp_years"] for c in constraints_by_s.values())
            g3 = max_pbp - CONSTRAINTS["pbp_max_years"]

        g1 = max_ase - CONSTRAINTS["ase_hours_max"]
        g2 = max_ld - CONSTRAINTS["ld_hours_max"]
        out["G"] = [g1, g2, g3]


def select_knee(F: np.ndarray) -> int:
    if len(F) == 0:
        return 0
    f_min = np.min(F, axis=0)
    f_max = np.max(F, axis=0)
    ranges = np.where(f_max - f_min == 0.0, 1.0, f_max - f_min)
    F_norm = (F - f_min) / ranges
    distances = np.sqrt(np.sum(F_norm ** 2, axis=1))
    return int(np.argmin(distances))


def run_nsga2_task4(
    epw_paths: Dict[str, str],
    results_dir: str,
    thresholds: Dict[str, Dict[str, float]] | None = None,
    pop_size: int | None = None,
    n_gen: int | None = None,
    seed: int | None = None,
) -> Dict[str, object]:
    os.makedirs(results_dir, exist_ok=True)

    epw_by_scenario = {}
    meta_by_scenario = {}
    for scenario, path in epw_paths.items():
        df, meta = read_epw(path)
        epw_by_scenario[scenario] = df
        meta_by_scenario[scenario] = meta

    pop_size = pop_size or NSGA2_CONFIG["pop_size"]
    n_gen = n_gen or NSGA2_CONFIG["n_gen"]
    seed = seed if seed is not None else NSGA2_CONFIG["seed"]
    stride = NSGA2_CONFIG["stage_a_stride"]

    problem = StudentUnionOptimization(
        epw_by_scenario,
        meta_by_scenario,
        stride=stride,
        thresholds=thresholds or DEFAULT_THRESHOLDS,
    )
    algorithm = NSGA2(pop_size=pop_size, n_offsprings=max(1, pop_size // 2), eliminate_duplicates=True)
    termination = MaximumGenerationTermination(n_max_gen=n_gen)

    res = minimize(problem, algorithm, termination, seed=seed, verbose=True)

    if res.X is None:
        pop = res.pop
        X = pop.get("X")
        F = pop.get("F")
        G = pop.get("G")
    else:
        X = res.X
        F = res.F
        G = res.G
    if G is None:
        G = np.zeros((len(X), 3))

    results_rows = []
    for x, f, g in zip(X, F, G):
        u = decode_u(np.array(x, dtype=float))
        row = {spec["name"]: u[spec["name"]] for spec in VAR_DEFS}
        row.update({
            "R_rob": float(f[0]),
            "C_u": float(f[1]),
            "g_ase": float(g[0]),
            "g_ld": float(g[1]),
            "g_fin": float(g[2]),
        })
        results_rows.append(row)

    pareto_df = pd.DataFrame(results_rows)
    pareto_csv = os.path.join(results_dir, "pareto_solutions.csv")
    pareto_df.to_csv(pareto_csv, index=False)

    knee_idx = select_knee(F)
    u_star = decode_u(np.array(X[knee_idx], dtype=float))

    # High-fidelity evaluation for u* and kpi outputs
    metrics_by_s = {}
    constraints_by_s = {}
    costs_by_s = {}
    timeseries_by_s = {}
    for scenario in SCENARIO_ORDER:
        df = epw_by_scenario[scenario]
        meta = meta_by_scenario[scenario]
        result = evaluate_student_union(
            u_star,
            df,
            meta,
            scenario=scenario,
            stride=NSGA2_CONFIG["stage_b_stride"],
            return_timeseries=True,
        )
        metrics_by_s[scenario] = result["x"]
        constraints_by_s[scenario] = result["constraints"]
        costs_by_s[scenario] = result["costs"]
        timeseries_by_s[scenario] = result["timeseries"]

    risk = compute_robust_risk(
        metrics_by_scenario=metrics_by_s,
        thresholds=thresholds or DEFAULT_THRESHOLDS,
        weights=RISK_WEIGHTS,
        scenario_weights=SCENARIO_WEIGHTS,
        risk_config=RISK_CONFIG,
    )

    u_star_path = os.path.join(results_dir, "u_SU_star.json")
    with open(u_star_path, "w", encoding="utf-8") as f:
        json.dump(u_star, f, indent=2)

    pretty_table(u_star, output_csv=os.path.join(results_dir, "u_SU_star.csv"))

    kpi_rows = []
    for scenario in SCENARIO_ORDER:
        row = {"scenario": scenario}
        row.update(metrics_by_s[scenario])
        row.update({
            "R_s": risk["R_by_scenario"].get(scenario, 0.0),
            "R0_s": risk["R0_by_scenario"].get(scenario, 0.0),
        })
        row.update(constraints_by_s[scenario])
        row.update({
            "capex": costs_by_s[scenario]["capex"],
            "lcc": costs_by_s[scenario]["lcc"],
            "annual_energy_kwh": costs_by_s[scenario]["annual_energy_kwh"],
            "annual_energy_cost": costs_by_s[scenario]["annual_energy_cost"],
            "pbp_years": costs_by_s[scenario]["pbp_years"],
            "npv": costs_by_s[scenario]["npv"],
        })
        kpi_rows.append(row)
    kpi_df = pd.DataFrame(kpi_rows)
    kpi_csv = os.path.join(results_dir, "kpi_by_scenario.csv")
    kpi_df.to_csv(kpi_csv, index=False)

    contrib_rows = []
    for scenario, contrib in risk["contributions_by_scenario"].items():
        for key, val in contrib.items():
            contrib_rows.append({"scenario": scenario, "metric": key, "contribution": val})
    contrib_df = pd.DataFrame(contrib_rows)
    contrib_csv = os.path.join(results_dir, "risk_contributions.csv")
    contrib_df.to_csv(contrib_csv, index=False)

    diag = {
        "seed": seed,
        "pop_size": pop_size,
        "n_gen": n_gen,
        "stride_stage_a": stride,
        "stride_stage_b": NSGA2_CONFIG["stage_b_stride"],
        "R_rob": risk["R_rob"],
        "Rob": risk["Rob"],
        "scenario_weights": SCENARIO_WEIGHTS,
        "risk_weights": RISK_WEIGHTS,
        "risk_config": RISK_CONFIG,
        "constraints": CONSTRAINTS,
        "thresholds": thresholds or DEFAULT_THRESHOLDS,
    }
    diag_path = os.path.join(results_dir, "diagnostics.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    timeseries_dir = os.path.join(results_dir, "timeseries")
    os.makedirs(timeseries_dir, exist_ok=True)
    for scenario, ts in timeseries_by_s.items():
        if ts is None:
            continue
        ts.to_csv(os.path.join(timeseries_dir, f"timeseries_u_star_{scenario}.csv"), index=False)

    return {
        "pareto_csv": pareto_csv,
        "u_star_path": u_star_path,
        "kpi_csv": kpi_csv,
        "contrib_csv": contrib_csv,
        "diagnostics_path": diag_path,
        "timeseries_dir": timeseries_dir,
    }
