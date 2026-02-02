#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSGA-II optimization runner for Borealis University.

This script runs the NSGA-II algorithm to find the Pareto frontier
for Borealis University building retrofit options.
"""

import sys
import os
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.termination import Termination
import argparse
from plot_pareto import plot_objective_correlations, plot_decision_distribution

# Add the current directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the optimization problem
from problem_borealis import BorealisOptimization


def normalize_objectives(F, bounds=None):
    """
    Normalize objectives to [0, 1] range.
    
    Args:
        F: Objective values array.
        bounds: Optional bounds array. If None, calculate from F.
        
    Returns:
        Normalized objectives array.
    """
    if bounds is None:
        min_vals = np.min(F, axis=0)
        max_vals = np.max(F, axis=0)
    else:
        min_vals = bounds[:, 0]
        max_vals = bounds[:, 1]
    
    # Avoid division by zero
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1
    
    return (F - min_vals) / ranges


def topsis_selection(F, weights=None):
    """
    Select the best balanced solution using TOPSIS method.
    
    Args:
        F: Objective values array (minimization).
        weights: Optional weights for objectives.
        
    Returns:
        Index of the best solution.
    """
    # Normalize objectives
    F_norm = normalize_objectives(F)
    
    # Use weights if provided, otherwise equal weights
    if weights is None:
        weights = np.ones(F.shape[1])
    
    # Weighted normalized matrix
    F_weighted = F_norm * weights
    
    # Ideal and negative ideal solutions
    ideal = np.min(F_weighted, axis=0)
    negative_ideal = np.max(F_weighted, axis=0)
    
    # Calculate distances
    distance_to_ideal = np.sqrt(np.sum((F_weighted - ideal) ** 2, axis=1))
    distance_to_negative_ideal = np.sqrt(np.sum((F_weighted - negative_ideal) ** 2, axis=1))
    
    # Calculate relative closeness
    closeness = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)
    
    # Select the solution with highest closeness
    best_idx = np.argmax(closeness)
    
    return best_idx


def main():
    """
    Main function to run NSGA-II optimization.
    """
    print("=== NSGA-II Optimization for Borealis University ===")

    parser = argparse.ArgumentParser(description="NSGA-II for Borealis (EPW-driven)")
    parser.add_argument("--epw", type=str, default=os.path.join("data", "epw", "NOR_OS_Oslo.Blindern.014920_TMYx.2009-2023.epw"),
                        help="EPW file path.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic weather data instead of EPW.")
    parser.add_argument("--pop", type=int, default=50, help="NSGA-II population size.")
    parser.add_argument("--gen", type=int, default=50, help="NSGA-II max generations.")
    args = parser.parse_args()

    # Initialize the optimization problem
    problem = BorealisOptimization(epw_path=args.epw, use_synthetic=args.synthetic)
    
    # Configure NSGA-II algorithm
    # For demonstration purposes, using small population and generations
    # For real-world application, increase pop_size to 100-200 and n_gen to 100-200
    algorithm = NSGA2(
        pop_size=args.pop,  # Population size
        n_offsprings=max(1, args.pop // 2),  # Offspring size per generation
        eliminate_duplicates=True
    )
    
    # Set termination criteria (simplified for compatibility)
    from pymoo.termination.max_gen import MaximumGenerationTermination
    termination = MaximumGenerationTermination(n_max_gen=args.gen)
    
    # Run optimization
    print("Running NSGA-II optimization...")
    print(f"Population size: {algorithm.pop_size}")
    print(f"Maximum generations: {termination.n_max_gen}")
    
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=True
    )
    
    # Extract results
    X = res.X  # Decision variables
    F = res.F  # Objective values
    
    print(f"\nOptimization completed!")
    print(f"Number of non-dominated solutions: {len(X)}")
    
    # Create results directories if they don't exist
    results_dir = os.path.join('results', 'task2')
    tables_dir = os.path.join(results_dir, 'tables')
    figs_dir = os.path.join(results_dir, 'figs')
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    
    # Save Pareto results to CSV
    results_df = pd.DataFrame(columns=[
        'WWR', 'dN', 'dE', 'dS', 'dW',
        'Annual_Energy_kWh', 'Lifecycle_Cost_USD', 'ASE_Hours'
    ])
    
    for i, (x, f) in enumerate(zip(X, F)):
        results_df.loc[i] = [
            x[0], x[1], x[2], x[3], x[4],
            f[0], f[1], f[2]
        ]
    
    output_csv = os.path.join(tables_dir, 'task2_pareto_results.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"Pareto results saved to: {output_csv}")
    
    # Select best balanced solution using TOPSIS
    print("\nSelecting best balanced solution using TOPSIS...")
    best_idx = topsis_selection(F)
    best_x = X[best_idx]
    best_f = F[best_idx]
    
    print(f"Best balanced solution:")
    print(f"  WWR: {best_x[0]:.4f}")
    print(f"  dN: {best_x[1]:.4f}")
    print(f"  dE: {best_x[2]:.4f}")
    print(f"  dS: {best_x[3]:.4f}")
    print(f"  dW: {best_x[4]:.4f}")
    print(f"  Annual Energy: {best_f[0]:.2f} kWh")
    print(f"  Lifecycle Cost: ${best_f[1]:.2f}")
    print(f"  ASE Hours: {best_f[2]:.2f}")
    
    # Save best solution
    best_df = pd.DataFrame([[
        best_x[0], best_x[1], best_x[2], best_x[3], best_x[4],
        best_f[0], best_f[1], best_f[2]
    ]], columns=[
        'WWR', 'dN', 'dE', 'dS', 'dW',
        'Annual_Energy_kWh', 'Lifecycle_Cost_USD', 'ASE_Hours'
    ])
    best_solution_path = os.path.join(tables_dir, 'task2_best_solution.csv')
    best_df.to_csv(best_solution_path, index=False)
    print(f"Best solution saved to: {best_solution_path}")
    
    # Plot Pareto frontier (2D projections) - simplified for compatibility
    print("\nGenerating Pareto frontier plots...")
    
    # Use matplotlib directly for plotting
    import matplotlib.pyplot as plt
    
    # Energy vs Cost
    plt.figure(figsize=(10, 8))
    plt.scatter(F[:, 0], F[:, 1], c='blue', s=50, alpha=0.7, edgecolors='k')
    plt.scatter(best_f[0], best_f[1], c='red', s=100, marker='*', edgecolors='k', label='Best Solution')
    plt.xlabel('Annual Energy (kWh)')
    plt.ylabel('Lifecycle Cost (USD)')
    plt.title('Energy vs Cost')
    plt.legend()
    plt.grid(True, alpha=0.3)
    energy_cost_path = os.path.join(figs_dir, 'Fig_T2_1_energy_vs_cost_2d.png')
    plt.savefig(energy_cost_path)
    print(f"Energy vs Cost plot saved to: {energy_cost_path}")
    plt.close()
    
    # Energy vs Discomfort
    plt.figure(figsize=(10, 8))
    plt.scatter(F[:, 0], F[:, 2], c='green', s=50, alpha=0.7, edgecolors='k')
    plt.scatter(best_f[0], best_f[2], c='red', s=100, marker='*', edgecolors='k', label='Best Solution')
    plt.xlabel('Annual Energy (kWh)')
    plt.ylabel('ASE Hours')
    plt.title('Energy vs Discomfort')
    plt.legend()
    plt.grid(True, alpha=0.3)
    energy_discomfort_path = os.path.join(figs_dir, 'Fig_T2_2_energy_vs_discomfort_2d.png')
    plt.savefig(energy_discomfort_path)
    print(f"Energy vs Discomfort plot saved to: {energy_discomfort_path}")
    plt.close()
    
    # Cost vs Discomfort
    plt.figure(figsize=(10, 8))
    plt.scatter(F[:, 1], F[:, 2], c='purple', s=50, alpha=0.7, edgecolors='k')
    plt.scatter(best_f[1], best_f[2], c='red', s=100, marker='*', edgecolors='k', label='Best Solution')
    plt.xlabel('Lifecycle Cost (USD)')
    plt.ylabel('ASE Hours')
    plt.title('Cost vs Discomfort')
    plt.legend()
    plt.grid(True, alpha=0.3)
    cost_discomfort_path = os.path.join(figs_dir, 'Fig_T2_3_cost_vs_discomfort_2d.png')
    plt.savefig(cost_discomfort_path)
    print(f"Cost vs Discomfort plot saved to: {cost_discomfort_path}")
    plt.close()

    # Additional visuals (2D only per requirement)
    plot_objective_correlations(output_csv, figs_dir)
    plot_decision_distribution(output_csv, figs_dir)
    
    print("\n=== Optimization complete! ===")


if __name__ == "__main__":
    main()
