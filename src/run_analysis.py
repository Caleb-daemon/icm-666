#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run analysis script for ICM E Problem.

This script generates baseline and retrofit hourly data, then creates
paper-quality visualization plots for the mathematical modeling paper.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on sys.path for task imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src_task1_sungrove.task1_sungrove import SimulationEngine, BuildingConfig

# Set matplotlib style for academic plots
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

# Create results directories if they don't exist
results_dir = os.path.join('results', 'task1')
tables_dir = os.path.join(results_dir, 'tables')
figs_dir = os.path.join(results_dir, 'figs')
os.makedirs(tables_dir, exist_ok=True)
os.makedirs(figs_dir, exist_ok=True)

def run_baseline_simulation():
    """
    Run baseline simulation and save hourly data.
    """
    print("Running baseline simulation...")
    # Create simulation engine with synthetic weather data for consistency
    engine = SimulationEngine(
        epw_path=None,
        use_synthetic=True,
        building_cfg=BuildingConfig()
    )
    
    # Run baseline simulation with detailed hourly data
    baseline = engine.simulate(
        wwr=0.4,
        overhang_depths_m={"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0},
        label="baseline",
        detailed=True
    )
    
    # Save hourly data to CSV
    df = pd.DataFrame(baseline['hourly_data'])
    output_path = os.path.join(tables_dir, 'task1_hourly_baseline.csv')
    df.to_csv(output_path, index=False)
    print(f"Baseline hourly data saved to: {output_path}")
    return output_path

def run_retrofit_simulation():
    """
    Run best retrofit simulation with specified parameters and save hourly data.
    """
    print("Running best retrofit simulation...")
    # Create simulation engine with synthetic weather data for consistency
    engine = SimulationEngine(
        epw_path=None,
        use_synthetic=True,
        building_cfg=BuildingConfig()
    )
    
    # Run retrofit simulation with detailed hourly data
    retrofit = engine.simulate(
        wwr=0.218,
        overhang_depths_m={"N": 0.98, "E": 0.65, "S": 0.31, "W": 0.89},
        label="retrofit_best",
        detailed=True
    )
    
    # Save hourly data to CSV
    df = pd.DataFrame(retrofit['hourly_data'])
    output_path = os.path.join(tables_dir, 'task1_hourly_retrofit.csv')
    df.to_csv(output_path, index=False)
    print(f"Retrofit hourly data saved to: {output_path}")
    return output_path

def create_monthly_load_compare():
    """
    Create monthly load comparison bar chart.
    """
    print("Creating monthly load comparison chart...")
    
    # Read CSV files
    df_baseline = pd.read_csv(os.path.join(tables_dir, 'task1_hourly_baseline.csv'))
    df_retrofit = pd.read_csv(os.path.join(tables_dir, 'task1_hourly_retrofit.csv'))
    
    # Convert time column to datetime
    df_baseline['time'] = pd.to_datetime(df_baseline['time'])
    df_retrofit['time'] = pd.to_datetime(df_retrofit['time'])
    
    # Add month column
    df_baseline['month'] = df_baseline['time'].dt.month
    df_retrofit['month'] = df_retrofit['time'].dt.month
    
    # Calculate monthly cooling load (kWh)
    # Assuming Cooling_Load_W is in Watts, convert to kWh by dividing by 1000
    monthly_baseline = df_baseline.groupby('month')['Cooling_Load_W'].sum() / 1000
    monthly_retrofit = df_retrofit.groupby('month')['Cooling_Load_W'].sum() / 1000
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    months = list(range(1, 13))
    x = np.arange(len(months))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, monthly_baseline.values, width, label='Baseline', color='skyblue')
    bars2 = ax.bar(x + width/2, monthly_retrofit.values, width, label='Retrofit', color='lightgreen')
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Monthly Cooling Load (kWh)')
    ax.set_title('Monthly Cooling Load Comparison: Baseline vs Retrofit')
    ax.set_xticks(x)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend()
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.0f}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.0f}', ha='center', va='bottom')
    
    # Save figure
    output_path = os.path.join(figs_dir, 'Fig1_Monthly_Load_Compare.png')
    plt.savefig(output_path)
    print(f"Monthly load comparison chart saved to: {output_path}")

def create_cooling_load_duration_curve():
    """
    Create cooling load duration curve.
    """
    print("Creating cooling load duration curve...")
    
    # Read CSV files
    df_baseline = pd.read_csv(os.path.join(tables_dir, 'task1_hourly_baseline.csv'))
    df_retrofit = pd.read_csv(os.path.join(tables_dir, 'task1_hourly_retrofit.csv'))
    
    # Extract cooling load data
    baseline_load = df_baseline['Cooling_Load_W'].values
    retrofit_load = df_retrofit['Cooling_Load_W'].values
    
    # Sort from highest to lowest
    baseline_load_sorted = np.sort(baseline_load)[::-1]
    retrofit_load_sorted = np.sort(retrofit_load)[::-1]
    
    # Create hour array (0-8760)
    hours = np.arange(len(baseline_load_sorted))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(hours, baseline_load_sorted, label='Baseline', color='skyblue', linewidth=2)
    ax.plot(hours, retrofit_load_sorted, label='Retrofit', color='lightgreen', linewidth=2)
    
    ax.set_xlabel('Hour (sorted by load)')
    ax.set_ylabel('Cooling Load (W)')
    ax.set_title('Cooling Load Duration Curve: Baseline vs Retrofit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save figure
    output_path = os.path.join(figs_dir, 'Fig2_Cooling_Load_Duration.png')
    plt.savefig(output_path)
    print(f"Cooling load duration curve saved to: {output_path}")

def create_summer_week_profile():
    """
    Create summer week profile plot with dual Y-axis.
    """
    print("Creating summer week profile chart...")
    
    # Read CSV files
    df_baseline = pd.read_csv(os.path.join(tables_dir, 'task1_hourly_baseline.csv'))
    df_retrofit = pd.read_csv(os.path.join(tables_dir, 'task1_hourly_retrofit.csv'))
    
    # Convert time column to datetime
    df_baseline['time'] = pd.to_datetime(df_baseline['time'])
    df_retrofit['time'] = pd.to_datetime(df_retrofit['time'])
    
    # Select summer week (July 10-17)
    start_date = '2023-07-10'
    end_date = '2023-07-17'
    
    mask_baseline = (df_baseline['time'] >= start_date) & (df_baseline['time'] < end_date)
    mask_retrofit = (df_retrofit['time'] >= start_date) & (df_retrofit['time'] < end_date)
    
    week_baseline = df_baseline.loc[mask_baseline]
    week_retrofit = df_retrofit.loc[mask_retrofit]
    
    # Create plot with dual Y-axis
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # First Y-axis: Temperature (°C)
    ax1.plot(week_baseline['time'], week_baseline['T_out'], label='Outdoor Temp', color='red', linestyle='--')
    ax1.plot(week_baseline['time'], week_baseline['T_in'], label='Baseline Indoor Temp', color='skyblue')
    ax1.plot(week_retrofit['time'], week_retrofit['T_in'], label='Retrofit Indoor Temp', color='lightgreen')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Temperature (°C)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(20, 35)
    
    # Second Y-axis: Solar Radiation (W/m²)
    ax2 = ax1.twinx()
    # Calculate solar radiation from Q_solar_total and area
    # Assuming area is 420 m² (south facade) * 0.4 (WWr) = 168 m²
    area = 168  # m²
    solar_radiation = week_baseline['Q_solar_total'] / area
    ax2.plot(week_baseline['time'], solar_radiation, label='Solar Radiation', color='orange', alpha=0.7)
    
    ax2.set_ylabel('Solar Radiation (W/m²)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim(0, 1000)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.set_title('Summer Week Profile: Temperature and Solar Radiation')
    ax1.grid(True, alpha=0.3)
    
    # Save figure
    output_path = os.path.join(figs_dir, 'Fig3_Summer_Week_Profile.png')
    plt.savefig(output_path)
    print(f"Summer week profile chart saved to: {output_path}")

def main():
    """
    Main function to run all tasks.
    """
    print("=== ICM E Problem Analysis Script ===")
    
    # Task 1: Generate baseline and retrofit data
    print("\nTask 1: Generating simulation data...")
    run_baseline_simulation()
    run_retrofit_simulation()
    
    # Task 2: Create visualization plots
    print("\nTask 2: Creating visualization plots...")
    create_monthly_load_compare()
    create_cooling_load_duration_curve()
    create_summer_week_profile()
    
    print("\n=== All tasks completed successfully! ===")

if __name__ == "__main__":
    main()
