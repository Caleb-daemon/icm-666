#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 2 visualization generator (2D only)."""

import os
import numpy as np
import pandas as pd
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


def main():
    results_dir = os.path.join('results', 'task2')
    tables_dir = os.path.join(results_dir, 'tables')
    figs_dir = os.path.join(results_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)

    csv_path = os.path.join(tables_dir, 'task2_pareto_results.csv')
    best_path = os.path.join(tables_dir, 'task2_best_solution.csv')

    df = pd.read_csv(csv_path)
    best = pd.read_csv(best_path) if os.path.exists(best_path) else None

    # Energy vs Cost
    plt.figure(figsize=(10, 8))
    plt.scatter(df['Annual_Energy_kWh'], df['Lifecycle_Cost_USD'], c='blue', s=40, alpha=0.7, edgecolors='k')
    if best is not None:
        plt.scatter(best['Annual_Energy_kWh'], best['Lifecycle_Cost_USD'], c='red', s=120, marker='*', edgecolors='k', label='Best')
        plt.legend()
    plt.xlabel('Annual Energy (kWh)')
    plt.ylabel('Lifecycle Cost (USD)')
    plt.title('Energy vs Cost')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figs_dir, 'Fig_T2_1_energy_vs_cost_2d.png'))
    plt.close()

    # Energy vs Discomfort
    plt.figure(figsize=(10, 8))
    plt.scatter(df['Annual_Energy_kWh'], df['ASE_Hours'], c='green', s=40, alpha=0.7, edgecolors='k')
    if best is not None:
        plt.scatter(best['Annual_Energy_kWh'], best['ASE_Hours'], c='red', s=120, marker='*', edgecolors='k', label='Best')
        plt.legend()
    plt.xlabel('Annual Energy (kWh)')
    plt.ylabel('ASE Hours')
    plt.title('Energy vs Discomfort')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figs_dir, 'Fig_T2_2_energy_vs_discomfort_2d.png'))
    plt.close()

    # Cost vs Discomfort
    plt.figure(figsize=(10, 8))
    plt.scatter(df['Lifecycle_Cost_USD'], df['ASE_Hours'], c='purple', s=40, alpha=0.7, edgecolors='k')
    if best is not None:
        plt.scatter(best['Lifecycle_Cost_USD'], best['ASE_Hours'], c='red', s=120, marker='*', edgecolors='k', label='Best')
        plt.legend()
    plt.xlabel('Lifecycle Cost (USD)')
    plt.ylabel('ASE Hours')
    plt.title('Cost vs Discomfort')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figs_dir, 'Fig_T2_3_cost_vs_discomfort_2d.png'))
    plt.close()

    # Objective correlations
    corr = df[['Annual_Energy_kWh', 'Lifecycle_Cost_USD', 'ASE_Hours']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Objective Correlations')
    plt.savefig(os.path.join(figs_dir, 'Fig_T2_4_objective_correlations.png'))
    plt.close()

    # Decision variables distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['WWR', 'dN', 'dE', 'dS', 'dW']])
    plt.ylabel('Value')
    plt.title('Decision Variables Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'Fig_T2_5_decision_distribution.png'))
    plt.close()

    print(f"Task2 figures saved to: {figs_dir}")


if __name__ == '__main__':
    main()
