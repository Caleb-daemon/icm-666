#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pareto frontier visualization for Borealis University optimization results.

This script generates 3D scatter plots and other visualizations for the
NSGA-II optimization results of Borealis University.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

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


def plot_3d_pareto_frontier(csv_path, output_dir):
    """
    Plot 3D Pareto frontier from CSV results.
    
    Args:
        csv_path: Path to the Pareto results CSV file.
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Extract objectives
    energy = df['Annual_Energy_kWh'].values
    cost = df['Lifecycle_Cost_USD'].values
    discomfort = df['ASE_Hours'].values
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Pareto frontier
    scatter = ax.scatter(
        energy, cost, discomfort,
        c='blue',
        s=50,
        alpha=0.7,
        edgecolors='k'
    )
    
    # Add labels
    ax.set_xlabel('Annual Energy (kWh)')
    ax.set_ylabel('Lifecycle Cost (USD)')
    ax.set_zlabel('ASE Hours')
    
    # Add title
    ax.set_title('Pareto Frontier for Borealis University Retrofit')
    
    # Adjust viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Save figure
    output_path = os.path.join(output_dir, 'Fig_T2_0_pareto_3d.png')
    plt.savefig(output_path)
    print(f"3D Pareto frontier plot saved to: {output_path}")
    plt.close()


def plot_2d_projections(csv_path, output_dir):
    """
    Plot 2D projections of the Pareto frontier.
    
    Args:
        csv_path: Path to the Pareto results CSV file.
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Read best solution if available
    best_solution_path = os.path.join(os.path.dirname(output_dir), 'tables', 'task2_best_solution.csv')
    best_solution = None
    if os.path.exists(best_solution_path):
        best_solution = pd.read_csv(best_solution_path)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Energy vs Cost
    ax = axes[0, 0]
    ax.scatter(df['Annual_Energy_kWh'], df['Lifecycle_Cost_USD'],
               c='blue', s=50, alpha=0.7, edgecolors='k')
    if best_solution is not None:
        ax.scatter(best_solution['Annual_Energy_kWh'], best_solution['Lifecycle_Cost_USD'],
                   c='red', s=100, marker='*', edgecolors='k', label='Best Solution')
        ax.legend()
    ax.set_xlabel('Annual Energy (kWh)')
    ax.set_ylabel('Lifecycle Cost (USD)')
    ax.set_title('Energy vs Cost')
    ax.grid(True, alpha=0.3)
    
    # Energy vs Discomfort
    ax = axes[0, 1]
    ax.scatter(df['Annual_Energy_kWh'], df['ASE_Hours'],
               c='green', s=50, alpha=0.7, edgecolors='k')
    if best_solution is not None:
        ax.scatter(best_solution['Annual_Energy_kWh'], best_solution['ASE_Hours'],
                   c='red', s=100, marker='*', edgecolors='k', label='Best Solution')
        ax.legend()
    ax.set_xlabel('Annual Energy (kWh)')
    ax.set_ylabel('ASE Hours')
    ax.set_title('Energy vs Discomfort')
    ax.grid(True, alpha=0.3)
    
    # Cost vs Discomfort
    ax = axes[1, 0]
    ax.scatter(df['Lifecycle_Cost_USD'], df['ASE_Hours'],
               c='purple', s=50, alpha=0.7, edgecolors='k')
    if best_solution is not None:
        ax.scatter(best_solution['Lifecycle_Cost_USD'], best_solution['ASE_Hours'],
                   c='red', s=100, marker='*', edgecolors='k', label='Best Solution')
        ax.legend()
    ax.set_xlabel('Lifecycle Cost (USD)')
    ax.set_ylabel('ASE Hours')
    ax.set_title('Cost vs Discomfort')
    ax.grid(True, alpha=0.3)
    
    # Decision variables distribution
    ax = axes[1, 1]
    sns.boxplot(data=df[['WWR', 'dN', 'dE', 'dS', 'dW']], ax=ax)
    ax.set_ylabel('Value')
    ax.set_title('Decision Variables Distribution')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'Fig_T2_6_pareto_2d_projections.png')
    plt.savefig(output_path)
    print(f"2D projections plot saved to: {output_path}")
    plt.close()


def plot_objective_correlations(csv_path, output_dir):
    """
    Plot correlations between objectives.
    
    Args:
        csv_path: Path to the Pareto results CSV file.
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Extract objectives
    objectives = df[['Annual_Energy_kWh', 'Lifecycle_Cost_USD', 'ASE_Hours']]
    
    # Calculate correlation matrix
    corr_matrix = objectives.corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True, vmin=-1, vmax=1)
    plt.title('Objective Correlations')
    
    # Save figure
    output_path = os.path.join(output_dir, 'Fig_T2_4_objective_correlations.png')
    plt.savefig(output_path)
    print(f"Objective correlations plot saved to: {output_path}")
    plt.close()


def plot_decision_distribution(csv_path, output_dir):
    """
    Plot decision variable distributions.
    """
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['WWR', 'dN', 'dE', 'dS', 'dW']])
    plt.ylabel('Value')
    plt.title('Decision Variables Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'Fig_T2_5_decision_distribution.png')
    plt.savefig(output_path)
    print(f"Decision distribution plot saved to: {output_path}")
    plt.close()


def main():
    """
    Main function to generate all visualization plots.
    """
    print("=== Generating Pareto frontier visualizations ===")
    
    # Check if results directory exists
    if not os.path.exists('results'):
        print("Error: Results directory not found. Please run run_nsga2.py first.")
        return
    
    # Path to Pareto results
    csv_path = os.path.join('results', 'task2', 'tables', 'task2_pareto_results.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run run_nsga2.py first.")
        return

    output_dir = os.path.join('results', 'task2', 'figs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    plot_3d_pareto_frontier(csv_path, output_dir)
    plot_2d_projections(csv_path, output_dir)
    plot_objective_correlations(csv_path, output_dir)
    plot_decision_distribution(csv_path, output_dir)
    
    print("\nAll visualization plots generated successfully!")


if __name__ == "__main__":
    main()
