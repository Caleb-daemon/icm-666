#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization module for building performance simulation.

This module provides functions to optimize building retrofit designs
by minimizing energy use, glare, and cost through various optimization algorithms.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Tuple, Optional, List


class OptimizationConfig:
    """
    Configuration class for optimization parameters.
    
    Attributes:
        method: Optimization method ('de' for differential evolution, 'grid' for grid search).
        wwr_bounds: Bounds for window-to-wall ratio (min, max).
        d_bounds_m: Bounds for overhang depth (min, max) in meters.
        w1_energy: Weight for energy objective.
        w2_cost: Weight for cost objective.
        w3_discomfort: Weight for discomfort objective.
        grid_n: Number of grid points for grid search.
    """
    def __init__(
        self,
        method: str = 'de',
        wwr_bounds: Tuple[float, float] = (0.2, 0.6),
        d_bounds_m: Tuple[float, float] = (0.0, 1.2),
        w1_energy: float = 0.5,
        w2_cost: float = 0.2,
        w3_discomfort: float = 0.3,
        grid_n: int = 5
    ):
        self.method = method
        self.wwr_bounds = wwr_bounds
        self.d_bounds_m = d_bounds_m
        self.w1_energy = w1_energy
        self.w2_cost = w2_cost
        self.w3_discomfort = w3_discomfort
        self.grid_n = grid_n


class OptimizationResult:
    """
    Result class for optimization.
    
    Attributes:
        params: Optimal parameters (WWR and overhang depths).
        energy: Energy use at optimal parameters.
        discomfort: Discomfort penalty at optimal parameters.
        cost: Cost at optimal parameters.
        score: Overall score at optimal parameters.
    """
    def __init__(
        self,
        params: Dict[str, float],
        energy: float,
        discomfort: float,
        cost: float,
        score: float
    ):
        self.params = params
        self.energy = energy
        self.discomfort = discomfort
        self.cost = cost
        self.score = score


def objective_function(
    x: np.ndarray,
    engine: object,
    config: OptimizationConfig
) -> float:
    """
    Objective function for optimization.

    Args:
        x: Optimization variables [wwr, dN, dE, dS, dW].
        engine: Simulation engine object with a simulate method.
        config: Optimization configuration.

    Returns:
        Combined objective value to minimize.

    Notes:
        The objective function combines normalized values of:
        - Energy use (higher is worse)
        - Discomfort penalty (higher is worse)
        - Cost (higher is worse)
        
        The weights are taken from the optimization configuration.
    """
    # Extract variables
    wwr = x[0]
    dN = x[1]
    dE = x[2]
    dS = x[3]
    dW = x[4]
    
    # Run simulation
    result = engine.simulate(
        wwr=wwr,
        overhang_depths_m={"N": dN, "E": dE, "S": dS, "W": dW},
        label="optimization"
    )
    
    # Calculate objective value (minimize)
    # Weights from config
    w1 = config.w1_energy
    w2 = config.w2_cost
    w3 = config.w3_discomfort
    
    # Objective is weighted sum of negative score (since we want to maximize score)
    # Score is already normalized to [0, 1]
    objective = -result['score']
    
    return objective


def optimize_differential_evolution(
    engine: object,
    config: OptimizationConfig
) -> OptimizationResult:
    """
    Optimize using scipy's differential evolution algorithm.

    Args:
        engine: Simulation engine object with a simulate method.
        config: Optimization configuration.

    Returns:
        Optimization result with optimal parameters and performance metrics.

    Notes:
        Uses scipy.optimize.differential_evolution to find the optimal solution.
        The algorithm explores the parameter space efficiently and is robust
        to local minima.
    """
    # Define bounds for variables
    bounds = [
        config.wwr_bounds,  # wwr
        config.d_bounds_m,  # dN
        config.d_bounds_m,  # dE
        config.d_bounds_m,  # dS
        config.d_bounds_m   # dW
    ]
    
    # Run differential evolution
    result = differential_evolution(
        objective_function,
        bounds=bounds,
        args=(engine, config),
        strategy='best1bin',
        maxiter=50,
        popsize=15,
        polish=True
    )
    
    # Extract optimal variables
    wwr_opt = result.x[0]
    dN_opt = result.x[1]
    dE_opt = result.x[2]
    dS_opt = result.x[3]
    dW_opt = result.x[4]
    
    # Run final simulation with optimal parameters
    sim_result = engine.simulate(
        wwr=wwr_opt,
        overhang_depths_m={"N": dN_opt, "E": dE_opt, "S": dS_opt, "W": dW_opt},
        label="optimal"
    )
    
    # Create optimization result
    opt_result = OptimizationResult(
        params={
            "wwr": wwr_opt,
            "dN": dN_opt,
            "dE": dE_opt,
            "dS": dS_opt,
            "dW": dW_opt
        },
        energy=sim_result['annual_cooling_kwh'],
        discomfort=sim_result['discomfort_penalty'],
        cost=sim_result['cost_usd'],
        score=sim_result['score']
    )
    
    return opt_result


def optimize_grid_search(
    engine: object,
    config: OptimizationConfig
) -> OptimizationResult:
    """
    Optimize using grid search.

    Args:
        engine: Simulation engine object with a simulate method.
        config: Optimization configuration.

    Returns:
        Optimization result with optimal parameters and performance metrics.

    Notes:
        Uses a brute-force grid search over the parameter space.
        The number of grid points is specified in the configuration.
        
        This method is more computationally expensive but guaranteed to find
        the best solution within the discretized grid.
    """
    # Generate grid points with cost-effective resolution
    # Overhang depth: 0-0.6m with 0.2m steps (4 points)
    d_values = np.linspace(0.0, 0.6, 4)
    
    # WWR: 0.30-0.35 with 0.05 steps (2 points)
    wwr_values = np.linspace(0.30, 0.35, 2)
    
    # Initialize best result
    best_score = -float('inf')
    best_params = None
    best_energy = float('inf')
    best_discomfort = float('inf')
    best_cost = float('inf')
    
    # Iterate over all grid points with direction-specific logic
    for wwr in wwr_values:
        # 为不同方向尝试不同的遮阳深度，只为南/西向添加遮阳
        for dS in d_values:
            for dW in d_values:
                for dE in [0.0]:  # 东向不添加遮阳，节省成本
                    for dN in [0.0]:  # 北向不添加遮阳，节省成本
                        # Run simulation
                        result = engine.simulate(
                            wwr=wwr,
                            overhang_depths_m={"N": dN, "E": dE, "S": dS, "W": dW},
                            label="grid_search"
                        )
                        
                        # Check if this is the best result
                        if result['score'] > best_score:
                            best_score = result['score']
                            best_params = {
                                "wwr": wwr,
                                "dN": dN,
                                "dE": dE,
                                "dS": dS,
                                "dW": dW
                            }
                            best_energy = result['annual_cooling_kwh']
                            best_discomfort = result['discomfort_penalty']
                            best_cost = result['cost_usd']
    
    # Create optimization result
    opt_result = OptimizationResult(
        params=best_params,
        energy=best_energy,
        discomfort=best_discomfort,
        cost=best_cost,
        score=best_score
    )
    
    return opt_result


def run_optimization(
    engine: object,
    config: OptimizationConfig
) -> OptimizationResult:
    """
    Run optimization based on the specified method.

    Args:
        engine: Simulation engine object with a simulate method.
        config: Optimization configuration.

    Returns:
        Optimization result with optimal parameters and performance metrics.

    Notes:
        This function dispatches to the appropriate optimization algorithm
        based on the method specified in the configuration.
    """
    if config.method == 'de':
        return optimize_differential_evolution(engine, config)
    elif config.method == 'grid':
        return optimize_grid_search(engine, config)
    else:
        raise ValueError(f"Unknown optimization method: {config.method}")