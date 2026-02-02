#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 1 visualization generator (Fig1-3 + extra)."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (12, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def _load_csv(tables_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(tables_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    return df


def plot_monthly_load_compare(df_baseline: pd.DataFrame, df_best: pd.DataFrame, figs_dir: str) -> str:
    df_baseline['month'] = df_baseline['time'].dt.month
    df_best['month'] = df_best['time'].dt.month

    monthly_baseline = df_baseline.groupby('month')['Cooling_Load_W'].sum() / 1000.0
    monthly_best = df_best.groupby('month')['Cooling_Load_W'].sum() / 1000.0

    fig, ax = plt.subplots(figsize=(12, 8))
    months = list(range(1, 13))
    x = np.arange(len(months))
    width = 0.35

    ax.bar(x - width / 2, monthly_baseline.values, width, label='Baseline', color='skyblue')
    ax.bar(x + width / 2, monthly_best.values, width, label='Retrofit', color='lightgreen')

    ax.set_xlabel('Month')
    ax.set_ylabel('Monthly Cooling Load (kWh)')
    ax.set_title('Monthly Cooling Load: Baseline vs Retrofit')
    ax.set_xticks(x)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend()

    output_path = os.path.join(figs_dir, 'Fig1_Monthly_Load_Compare.png')
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_load_duration(df_baseline: pd.DataFrame, df_best: pd.DataFrame, figs_dir: str) -> str:
    baseline_load = np.sort(df_baseline['Cooling_Load_W'].values)[::-1]
    best_load = np.sort(df_best['Cooling_Load_W'].values)[::-1]

    hours = np.arange(len(baseline_load))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(hours, baseline_load, label='Baseline', color='skyblue', linewidth=2)
    ax.plot(hours, best_load, label='Retrofit', color='lightgreen', linewidth=2)
    ax.set_xlabel('Hour (sorted by load)')
    ax.set_ylabel('Cooling Load (W)')
    ax.set_title('Cooling Load Duration Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(figs_dir, 'Fig2_Cooling_Load_Duration.png')
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_summer_week(df_baseline: pd.DataFrame, df_best: pd.DataFrame, figs_dir: str) -> str:
    start_date = '2023-07-10'
    end_date = '2023-07-17'

    mask_baseline = (df_baseline['time'] >= start_date) & (df_baseline['time'] < end_date)
    mask_best = (df_best['time'] >= start_date) & (df_best['time'] < end_date)

    df_b = df_baseline.loc[mask_baseline].copy()
    df_r = df_best.loc[mask_best].copy()

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    ax1.plot(df_b['time'], df_b['Cooling_Load_W'], label='Baseline Cooling Load', color='skyblue')
    ax1.plot(df_r['time'], df_r['Cooling_Load_W'], label='Retrofit Cooling Load', color='lightgreen')
    ax2.plot(df_b['time'], df_b['T_out'], label='Outdoor Temp', color='orange', alpha=0.7)

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cooling Load (W)')
    ax2.set_ylabel('Outdoor Temperature (°C)')
    ax1.set_title('Summer Week Profile (July 10–17)')
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    output_path = os.path.join(figs_dir, 'Fig3_Summer_Week_Profile.png')
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_sdi_hist(df_best: pd.DataFrame, figs_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_best['SDI'], bins=40, kde=False, ax=ax, color='slateblue')
    ax.set_title('SDI Distribution (Retrofit)')
    ax.set_xlabel('SDI')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    output_path = os.path.join(figs_dir, 'Fig_T1_4_SDI_hist.png')
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_load_vs_tout(df_best: pd.DataFrame, figs_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_best['T_out'], df_best['Cooling_Load_W'], s=8, alpha=0.4, color='teal')
    ax.set_title('Cooling Load vs Outdoor Temperature')
    ax.set_xlabel('Outdoor Temperature (°C)')
    ax.set_ylabel('Cooling Load (W)')
    ax.grid(True, alpha=0.3)
    output_path = os.path.join(figs_dir, 'Fig_T1_5_Load_vs_Tout.png')
    plt.savefig(output_path)
    plt.close()
    return output_path


def main():
    results_dir = os.path.join('results', 'task1')
    tables_dir = os.path.join(results_dir, 'tables')
    figs_dir = os.path.join(results_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)

    df_baseline = _load_csv(tables_dir, 'task1_hourly_baseline.csv')
    df_best = _load_csv(tables_dir, 'task1_hourly_best.csv')

    plot_monthly_load_compare(df_baseline, df_best, figs_dir)
    plot_load_duration(df_baseline, df_best, figs_dir)
    plot_summer_week(df_baseline, df_best, figs_dir)
    plot_sdi_hist(df_best, figs_dir)
    plot_load_vs_tout(df_best, figs_dir)

    print(f"Task1 figures saved to: {figs_dir}")


if __name__ == '__main__':
    main()
