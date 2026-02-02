#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics module for building performance simulation.

This module provides functions to calculate and normalize various performance
metrics for building design, including energy use, glare, and cost.
"""

import numpy as np
from typing import Tuple, Optional


def normalize(
    value: float,
    min_value: float,
    max_value: float
) -> float:
    """
    Normalize a value to the range [0, 1].

    Args:
        value: Value to normalize.
        min_value: Minimum value of the range.
        max_value: Maximum value of the range.

    Returns:
        Normalized value (0-1).

    Notes:
        Implements Formula 29 from the v3 document:
        normalized = (value - min_value) / (max_value - min_value)
        
        For values outside the range, clamps to 0 or 1.
    """
    # Calculate normalized value
    if max_value == min_value:
        # Avoid division by zero
        return 0.5
    
    normalized = (value - min_value) / (max_value - min_value)
    
    # Clamp to [0, 1]
    normalized = max(0.0, min(1.0, normalized))
    
    return normalized


def score_R(
    energy_use: float,
    glare_hours: float,
    cost: float,
    energy_weight: float = 0.5,
    glare_weight: float = 0.3,
    cost_weight: float = 0.2,
    energy_reference: float = 100000.0,
    glare_reference: float = 1000.0,
    cost_reference: float = 100000.0
) -> float:
    """
    Calculate the overall performance score R.

    Args:
        energy_use: Annual energy use (kWh).
        glare_hours: Annual glare hours (hours/year).
        cost: Total cost (USD).
        energy_weight: Weight for energy use (default 0.5).
        glare_weight: Weight for glare hours (default 0.3).
        cost_weight: Weight for cost (default 0.2).
        energy_reference: Reference value for energy use normalization (default 100000.0).
        glare_reference: Reference value for glare hours normalization (default 1000.0).
        cost_reference: Reference value for cost normalization (default 100000.0).

    Returns:
        Overall performance score R (0-1).
        Higher scores indicate better performance.

    Notes:
        Implements Formula 32 from the v3 document:
        R = energy_weight * (1 - energy_normalized) + \
            glare_weight * (1 - glare_normalized) + \
            cost_weight * (1 - cost_normalized)
        
        The score is designed to be higher for lower energy use, fewer glare hours,
        and lower cost.
    """
    # Normalize each metric
    # Note: Lower values are better, so we normalize and subtract from 1
    energy_normalized = normalize(energy_use, 0.0, energy_reference)
    glare_normalized = normalize(glare_hours, 0.0, glare_reference)
    cost_normalized = normalize(cost, 0.0, cost_reference)
    
    # Calculate weighted score
    # Higher scores are better, so we use (1 - normalized) for each metric
    score = energy_weight * (1 - energy_normalized) + \
            glare_weight * (1 - glare_normalized) + \
            cost_weight * (1 - cost_normalized)
    
    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))
    
    return score


def calculate_energy_savings(
    baseline_energy: float,
    retrofit_energy: float
) -> float:
    """
    Calculate energy savings from baseline to retrofit.

    Args:
        baseline_energy: Baseline energy use (kWh).
        retrofit_energy: Retrofit energy use (kWh).

    Returns:
        Energy savings percentage (%).
        Positive values indicate energy savings.

    Notes:
        Calculates the percentage reduction in energy use:
        savings = ((baseline_energy - retrofit_energy) / baseline_energy) * 100
    """
    if baseline_energy <= 0:
        return 0.0
    
    savings = ((baseline_energy - retrofit_energy) / baseline_energy) * 100
    
    return savings


def calculate_discomfort_penalty(
    glare_hours: float,
    max_penalty: float = 50000.0
) -> float:
    """
    Calculate discomfort penalty based on glare hours.

    Args:
        glare_hours: Annual glare hours (hours/year).
        max_penalty: Maximum penalty value (default 50000.0).

    Returns:
        Discomfort penalty value.

    Notes:
        This penalty is used in the optimization process to discourage
        designs with excessive glare hours.
    """
    # Calculate penalty based on glare hours
    # Penalty increases linearly with glare hours up to max_penalty
    penalty = min(max_penalty, glare_hours * 50.0)
    
    return penalty