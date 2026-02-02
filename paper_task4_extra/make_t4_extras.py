#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate paper figures for Task 4:
- T4-2 decision variable table
- T4-5 risk decomposition waterfall
- T4-8 normalization threshold schematic

This script is intentionally isolated from the main pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src_task4_student_union.config_task4 import (
    BASELINE_U0,
    SCENARIOS,
    SCENARIO_ORDER,
    SCENARIO_WEIGHTS,
    RISK_WEIGHTS,
    RISK_CONFIG,
)
from src_task4_student_union.model_student_union import VAR_DEFS, repair_u, evaluate_student_union
from src_task4_student_union.risk import compute_robust_risk
from src.io_epw import read_epw


VAR_INFO = {
    "gamma_build": {
        "meaning": "Building orientation azimuth (deg)",
        "direction": "Tune solar exposure; cooling vs daylight tradeoff",
    },
    "wwr_N": {
        "meaning": "Window-to-wall ratio (North)",
        "direction": "WWR up: x1/x3/x4 up, x6 up, x7 down",
    },
    "wwr_E": {
        "meaning": "Window-to-wall ratio (East)",
        "direction": "WWR up: x1/x3/x4 up, x6 up, x7 down",
    },
    "wwr_S": {
        "meaning": "Window-to-wall ratio (South)",
        "direction": "WWR up: x1/x3/x4 up, x6 up, x7 down",
    },
    "wwr_W": {
        "meaning": "Window-to-wall ratio (West)",
        "direction": "WWR up: x1/x3/x4 up, x6 up, x7 down",
    },
    "veg_cover_S": {
        "meaning": "Vertical greening cover ratio (South)",
        "direction": "Cover up: x1/x3/x4 down, x6 down, x7 up, x9 up",
    },
    "veg_cover_W": {
        "meaning": "Vertical greening cover ratio (West)",
        "direction": "Cover up: x1/x3/x4 down, x6 down, x7 up, x9 up",
    },
    "jJ_SU": {
        "meaning": "Shading band count (facade)",
        "direction": "Band count up: more shading, x1/x3/x4 down, x7 up",
    },
    "SHGC": {
        "meaning": "Solar heat gain coefficient",
        "direction": "SHGC up: x1/x3/x4 up, x2 down",
    },
    "VT": {
        "meaning": "Visible transmittance",
        "direction": "VT up: x7 down, x6 up",
    },
    "d_oh": {
        "meaning": "Overhang depth (m)",
        "direction": "Depth up: x1/x3/x4 down, x6 down, x7 up, x9 up",
    },
    "h_fin": {
        "meaning": "Fin depth (m)",
        "direction": "Depth up: x1/x3/x4 down, x6 down, x7 up, x9 up",
    },
    "E_dir_in_th": {
        "meaning": "Direct illuminance control threshold",
        "direction": "Threshold up: less shading, x1/x3/x4 up, x6 up, x7 down",
    },
    "alpha_summer": {
        "meaning": "Summer louver angle (deg)",
        "direction": "Angle up: more shading, x1/x3/x4 down, x6 down, x7 up",
    },
    "alpha_winter": {
        "meaning": "Winter louver angle (deg)",
        "direction": "Angle up: more shading, x2 up, x7 up",
    },
    "m_mass": {
        "meaning": "Thermal mass multiplier",
        "direction": "Mass up: x3 down, x5 down, x9 up",
    },
}


def _format_bounds(bounds: Tuple[float, float]) -> str:
    lo, hi = bounds
    return f"[{lo}, {hi}]"


def build_var_table() -> pd.DataFrame:
    rows = []
    for spec in VAR_DEFS:
        name = spec["name"]
        info = VAR_INFO.get(name, {})
        rows.append(
            {
                "Variable": name,
                "Meaning": info.get("meaning", ""),
                "Type": spec["type"],
                "Range": _format_bounds(spec["bounds"]),
                "Effect (direction)": info.get("direction", ""),
            }
        )
    return pd.DataFrame(rows)


def plot_var_table(df: pd.DataFrame, out_path: Path) -> None:
    fig_h = max(4.0, 0.35 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _load_thresholds(results_dir: Path) -> Dict[str, Dict[str, float]]:
    thresholds_path = results_dir / "thresholds_used.json"
    with thresholds_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_u_star(results_dir: Path) -> Dict[str, float]:
    u_star_path = results_dir / "u_SU_star.json"
    with u_star_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _scenario_epw_paths(results_dir: Path, epw_path: Path) -> Dict[str, Path]:
    paths = {"now": epw_path}
    future_dir = results_dir / "epw_future"
    for scenario, cfg in SCENARIOS.items():
        if scenario == "now":
            continue
        paths[scenario] = future_dir / f"EPW_future_{cfg['year']}.epw"
    return paths


def _metrics_by_scenario(u: Dict[str, float], epw_paths: Dict[str, Path]) -> Dict[str, Dict[str, float]]:
    metrics_by_s = {}
    for scenario in SCENARIO_ORDER:
        path = epw_paths.get(scenario)
        if path is None or not path.exists():
            continue
        df, meta = read_epw(str(path))
        res = evaluate_student_union(u, df, meta, scenario=scenario, stride=1, return_timeseries=False)
        metrics_by_s[scenario] = res["x"]
    return metrics_by_s


def _risk_contrib_total(
    metrics_by_s: Dict[str, Dict[str, float]],
    thresholds: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
    scenario_weights: Dict[str, float],
    risk_config: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    risk = compute_robust_risk(
        metrics_by_scenario=metrics_by_s,
        thresholds=thresholds,
        weights=weights,
        scenario_weights=scenario_weights,
        risk_config=risk_config,
    )
    r_by_s = risk["R_by_scenario"]
    contrib_by_s = risk["contributions_by_scenario"]
    contrib_total: Dict[str, float] = {}

    for scenario, contrib in contrib_by_s.items():
        w = scenario_weights.get(scenario, 0.0)
        for key, val in contrib.items():
            contrib_total[key] = contrib_total.get(key, 0.0) + w * val

    lambda_w = float(risk_config.get("lambda_w", 0.0))
    if lambda_w > 0.0 and r_by_s:
        max_s = max(r_by_s, key=r_by_s.get)
        for key, val in contrib_by_s[max_s].items():
            contrib_total[key] = contrib_total.get(key, 0.0) + lambda_w * val

    return float(risk["R_rob"]), contrib_total


def plot_risk_waterfall(
    base_r: float,
    base_contrib: Dict[str, float],
    star_r: float,
    star_contrib: Dict[str, float],
    out_path: Path,
) -> None:
    metrics = [f"x{i}" for i in range(1, 10)]
    deltas = []
    for m in metrics:
        deltas.append(star_contrib.get(m, 0.0) - base_contrib.get(m, 0.0))

    labels = ["R(u0)"] + metrics + ["R(u*)"]
    values = [base_r] + deltas + [star_r]

    fig, ax = plt.subplots(figsize=(12, 5))

    cum = base_r
    ax.bar(0, base_r, color="#bdbdbd", edgecolor="black", linewidth=0.6)
    for i, delta in enumerate(deltas, start=1):
        color = "#d95f02" if delta > 0 else "#1b9e77"
        bottom = cum if delta >= 0 else cum + delta
        ax.bar(i, delta, bottom=bottom, color=color, edgecolor="black", linewidth=0.6)
        ax.text(i, bottom + delta, f"{delta:+.2f}", ha="center", va="bottom", fontsize=8)
        cum += delta

    ax.bar(len(values) - 1, star_r, color="#3182bd", edgecolor="black", linewidth=0.6)

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("R_rob")
    ax.set_title("T4-5 Risk Decomposition Waterfall")
    ax.axhline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_norm_schematic(
    thresholds: Dict[str, Dict[str, float]],
    out_path: Path,
    metric: str = "x1",
) -> None:
    if metric not in thresholds:
        metric = next(iter(thresholds))
    good = thresholds[metric]["good"]
    bad = thresholds[metric]["bad"]

    span = max(1e-6, bad - good)
    x = np.linspace(good - 0.6 * span, bad + 0.6 * span, 200)
    y = (x - good) / span
    y = np.clip(y, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, color="#2c7fb8", linewidth=2)
    ax.axvline(good, color="#238b45", linestyle="--", linewidth=1)
    ax.axvline(bad, color="#cb181d", linestyle="--", linewidth=1)
    ax.text(good, 0.02, "good", color="#238b45", ha="right", va="bottom", fontsize=9)
    ax.text(bad, 0.98, "bad", color="#cb181d", ha="left", va="top", fontsize=9)
    ax.text(x.min(), 0.05, "clip to 0", color="#555555", ha="left", va="bottom", fontsize=8)
    ax.text(x.max(), 0.95, "clip to 1", color="#555555", ha="right", va="top", fontsize=8)
    ax.set_xlabel(f"{metric} value")
    ax.set_ylabel("normalized value")
    ax.set_title("T4-8 NORM-1 Threshold Schematic (good/bad/clip)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Task 4 extra paper figures.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(ROOT / "results" / "task4"),
        help="Task 4 results directory.",
    )
    parser.add_argument(
        "--epw",
        type=str,
        default=str(ROOT / "data" / "epw" / "HKG_NT_Lau.Fau.Shan.450350_TMYx.2009-2023.epw"),
        help="Base EPW path for scenario 'now'.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT / "paper_task4_extra" / "figs"),
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_vars = build_var_table()
    plot_var_table(df_vars, out_dir / "T4-2_u_SU_table.png")

    thresholds = _load_thresholds(results_dir)
    plot_norm_schematic(thresholds, out_dir / "T4-8_norm_thresholds.png", metric="x1")

    u_star = repair_u(_load_u_star(results_dir))
    u0 = repair_u(dict(BASELINE_U0))

    epw_paths = _scenario_epw_paths(results_dir, Path(args.epw))
    metrics_u0 = _metrics_by_scenario(u0, epw_paths)
    metrics_us = _metrics_by_scenario(u_star, epw_paths)

    base_r, base_contrib = _risk_contrib_total(
        metrics_u0, thresholds, RISK_WEIGHTS, SCENARIO_WEIGHTS, RISK_CONFIG
    )
    star_r, star_contrib = _risk_contrib_total(
        metrics_us, thresholds, RISK_WEIGHTS, SCENARIO_WEIGHTS, RISK_CONFIG
    )

    plot_risk_waterfall(
        base_r, base_contrib, star_r, star_contrib, out_dir / "T4-5_risk_waterfall.png"
    )


if __name__ == "__main__":
    main()
