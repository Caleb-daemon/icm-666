#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots for Task 4 outputs (T4-1 ... T4-7).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config_task4 import SCENARIO_ORDER, SCENARIO_WEIGHTS


def plot_climate_stats(stats_csv: str, outdir: str) -> str:
    df = pd.read_csv(stats_csv, index_col=0)
    df = df.reindex(SCENARIO_ORDER)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    df["temp_mean_C"].plot(kind="bar", ax=axes[0, 0], color="#2c7fb8")
    axes[0, 0].set_title("Mean Dry Bulb (C)")
    axes[0, 0].set_xlabel("")

    df["CDD_base18"].plot(kind="bar", ax=axes[0, 1], color="#f03b20")
    axes[0, 1].set_title("Cooling Degree Days (base 18C)")
    axes[0, 1].set_xlabel("")

    df["temp_peak_C"].plot(kind="bar", ax=axes[1, 0], color="#feb24c")
    axes[1, 0].set_title("Peak Dry Bulb (C)")
    axes[1, 0].set_xlabel("")

    axes[1, 1].plot(df.index, df["DNI_mean"], marker="o", label="DNI")
    axes[1, 1].plot(df.index, df["DHI_mean"], marker="s", label="DHI")
    axes[1, 1].set_title("Mean Irradiance (W/m2)")
    axes[1, 1].legend()

    fig.tight_layout()
    out_path = os.path.join(outdir, "T4-1_climate_stats.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_pareto(pareto_csv: str, u_star_json: str, outdir: str) -> str:
    df = pd.read_csv(pareto_csv)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["C_u"], df["R_rob"], s=30, alpha=0.7, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("C(u) (LCC per m2)")
    ax.set_ylabel("R_rob")
    ax.set_title("T4-4 Pareto Frontier")

    try:
        import json
        with open(u_star_json, "r", encoding="utf-8") as f:
            u_star = json.load(f)
        mask = np.ones(len(df), dtype=bool)
        for key, val in u_star.items():
            if key in df.columns:
                mask &= np.isclose(df[key].astype(float), float(val), rtol=1e-3, atol=1e-3)
        if np.any(mask):
            row = df[mask].iloc[0]
            ax.scatter([row["C_u"]], [row["R_rob"]], c="red", s=90, marker="*", label="u*")
            ax.legend()
    except Exception:
        pass

    fig.tight_layout()
    out_path = os.path.join(outdir, "T4-4_pareto.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_risk_contributions(contrib_csv: str, outdir: str) -> str:
    df = pd.read_csv(contrib_csv)
    metrics = sorted(df["metric"].unique())
    contrib_weighted = {m: 0.0 for m in metrics}

    for scenario in SCENARIO_ORDER:
        w = SCENARIO_WEIGHTS.get(scenario, 0.0)
        sub = df[df["scenario"] == scenario]
        for _, row in sub.iterrows():
            contrib_weighted[row["metric"]] += w * row["contribution"]

    values = [contrib_weighted[m] for m in metrics]
    max_val = max(values) if values else 1.0
    epsilon = max(0.5, 0.01 * max_val)
    plot_vals = [v if v > 0 else epsilon for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(metrics, plot_vals, color="#3182bd", edgecolor="black", linewidth=0.6)
    for i, v in enumerate(values):
        if v <= 0:
            bars[i].set_facecolor("none")
            bars[i].set_hatch("//")
        ax.text(i, plot_vals[i] + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("T4-3 Risk Contributions (Weighted)")
    ax.set_ylabel("Contribution to R")
    ax.set_xlabel("Indicator")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out_path = os.path.join(outdir, "T4-3_risk_contrib.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_robustness(kpi_csv: str, outdir: str) -> str:
    df = pd.read_csv(kpi_csv)
    df = df.set_index("scenario").reindex(SCENARIO_ORDER)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(df.index, df["R_s"], color="#74c476")
    vals = df["R_s"].to_numpy()
    max_val = float(np.max(vals)) if len(vals) else 1.0
    y_off = max(0.2, 0.01 * max_val)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_off,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title("T4-6 Scenario Risks for u*")
    ax.set_ylabel("R_s")
    fig.tight_layout()
    out_path = os.path.join(outdir, "T4-6_robustness.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_heatwave_week(timeseries_dir: str, outdir: str, scenario: str = "future2080") -> Optional[str]:
    path = os.path.join(timeseries_dir, f"timeseries_u_star_{scenario}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    temps = df["T_out"].to_numpy()
    window = 24 * 7
    if len(temps) < window:
        idx_start = 0
    else:
        rolling = np.convolve(temps, np.ones(window), "valid") / window
        idx_start = int(np.argmax(rolling))

    df_week = df.iloc[idx_start: idx_start + window]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_week["time"], df_week["T_out"], label="T_out", color="#f03b20")
    ax.plot(df_week["time"], df_week["T_in"], label="T_in", color="#2b8cbe")
    ax.set_title("T4-7 Heatwave Week (2080)")
    ax.set_ylabel("Temperature (C)")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(outdir, "T4-7_heatwave_week.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def run_all_plots(results_dir: str) -> Dict[str, str]:
    figs_dir = os.path.join(results_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    outputs = {}
    stats_csv = os.path.join(results_dir, "climate_stats.csv")
    pareto_csv = os.path.join(results_dir, "pareto_solutions.csv")
    u_star_json = os.path.join(results_dir, "u_SU_star.json")
    contrib_csv = os.path.join(results_dir, "risk_contributions.csv")
    kpi_csv = os.path.join(results_dir, "kpi_by_scenario.csv")
    timeseries_dir = os.path.join(results_dir, "timeseries")

    if os.path.exists(stats_csv):
        outputs["T4-1"] = plot_climate_stats(stats_csv, figs_dir)
    if os.path.exists(pareto_csv) and os.path.exists(u_star_json):
        outputs["T4-4"] = plot_pareto(pareto_csv, u_star_json, figs_dir)
    if os.path.exists(contrib_csv):
        outputs["T4-3"] = plot_risk_contributions(contrib_csv, figs_dir)
    if os.path.exists(kpi_csv):
        outputs["T4-6"] = plot_robustness(kpi_csv, figs_dir)
    if os.path.exists(timeseries_dir):
        path = plot_heatwave_week(timeseries_dir, figs_dir, scenario="future2080")
        if path:
            outputs["T4-7"] = path

    return outputs
