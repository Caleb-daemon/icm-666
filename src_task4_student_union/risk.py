#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk normalization, aggregation, and robust risk (Task 4).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def normalize_metrics(
    metrics: Dict[str, float],
    thresholds: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    NORM-1 normalization with clip to [0, 1].
    """
    normed: Dict[str, float] = {}
    for key, value in metrics.items():
        if key not in thresholds:
            continue
        good = thresholds[key]["good"]
        bad = thresholds[key]["bad"]
        if bad == good:
            raise ValueError(f"NORM-1 invalid threshold for {key}: good == bad")
        x = (value - good) / (bad - good)
        normed[key] = float(np.clip(x, 0.0, 1.0))
    return normed


def scenario_risk(
    normed: Dict[str, float],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """
    R-1 risk aggregation for a single scenario.
    """
    contributions = {}
    total = 0.0
    for key, weight in weights.items():
        val = normed.get(key, 0.0)
        contrib = 100.0 * weight * val
        contributions[key] = contrib
        total += contrib
    return total, contributions


def compute_robust_risk(
    metrics_by_scenario: Dict[str, Dict[str, float]],
    thresholds: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
    scenario_weights: Dict[str, float],
    risk_config: Dict[str, float],
) -> Dict[str, object]:
    """
    Compute Rob and robust risk across scenarios.
    """
    lambda1 = risk_config.get("lambda1", 0.6)
    lambda2 = risk_config.get("lambda2", 0.4)
    lambda_w = risk_config.get("lambda_w", 0.0)
    var_ddof = int(risk_config.get("var_ddof", 0))
    include_now = bool(risk_config.get("include_now_in_rob", False))

    # Normalize without Rob (x8 placeholder)
    normed_by_s = {}
    r0_by_s = {}
    for scenario, metrics in metrics_by_scenario.items():
        metrics_no_rob = dict(metrics)
        metrics_no_rob["x8"] = 0.0
        normed = normalize_metrics(metrics_no_rob, thresholds)
        normed_by_s[scenario] = normed
        weights_no_x8 = dict(weights)
        weights_no_x8["x8"] = 0.0
        r0, _ = scenario_risk(normed, weights_no_x8)
        r0_by_s[scenario] = r0

    if include_now:
        r0_values = list(r0_by_s.values())
    else:
        r0_values = [v for k, v in r0_by_s.items() if k != "now"]
    if len(r0_values) == 0:
        r0_values = list(r0_by_s.values())

    var_r0 = float(np.var(r0_values, ddof=var_ddof))
    max_r0 = float(np.max(r0_values))
    rob = lambda1 * var_r0 + lambda2 * max_r0

    # Recompute with Rob injected
    r_by_s = {}
    contributions_by_s = {}
    normed_final_by_s = {}
    for scenario, metrics in metrics_by_scenario.items():
        metrics_with_rob = dict(metrics)
        metrics_with_rob["x8"] = rob
        normed = normalize_metrics(metrics_with_rob, thresholds)
        normed_final_by_s[scenario] = normed
        r, contrib = scenario_risk(normed, weights)
        r_by_s[scenario] = r
        contributions_by_s[scenario] = contrib

    weighted_sum = 0.0
    for scenario, r_val in r_by_s.items():
        weight = scenario_weights.get(scenario, 0.0)
        weighted_sum += weight * r_val
    max_r = float(np.max(list(r_by_s.values()))) if r_by_s else 0.0
    r_rob = weighted_sum + lambda_w * max_r

    # Average contributions for waterfall charts
    avg_contrib = {}
    for scenario, contrib in contributions_by_s.items():
        w = scenario_weights.get(scenario, 0.0)
        for key, val in contrib.items():
            avg_contrib[key] = avg_contrib.get(key, 0.0) + w * val

    return {
        "R_by_scenario": r_by_s,
        "R0_by_scenario": r0_by_s,
        "Rob": rob,
        "R_rob": r_rob,
        "contributions_by_scenario": contributions_by_s,
        "contributions_weighted": avg_contrib,
        "normed_by_scenario": normed_final_by_s,
    }

