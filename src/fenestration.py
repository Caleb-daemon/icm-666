#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fenestration helpers for glazing properties.

These utilities provide simple SHGC/VT angle modifiers and cost proxies
used by Task 4 without changing existing task behavior.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def shgc_effective(shgc: float, cos_incidence: np.ndarray, k: float = 3.5) -> np.ndarray:
    """
    Compute angle-adjusted SHGC (simple empirical modifier).

    Args:
        shgc: Nominal SHGC at normal incidence.
        cos_incidence: Cosine of incidence angle (0-1).
        k: Shape parameter for angle modifier.

    Returns:
        Effective SHGC array.
    """
    cos_incidence = np.clip(cos_incidence, 0.0, 1.0)
    modifier = 1.0 - np.power(1.0 - cos_incidence, k)
    modifier = np.clip(modifier, 0.2, 1.0)
    return shgc * modifier


def vt_effective(vt: float, cos_incidence: np.ndarray, k: float = 2.5) -> np.ndarray:
    """
    Compute angle-adjusted visible transmittance.

    Args:
        vt: Nominal visible transmittance.
        cos_incidence: Cosine of incidence angle (0-1).
        k: Shape parameter for angle modifier.

    Returns:
        Effective VT array.
    """
    cos_incidence = np.clip(cos_incidence, 0.0, 1.0)
    modifier = 1.0 - np.power(1.0 - cos_incidence, k)
    modifier = np.clip(modifier, 0.2, 1.0)
    return vt * modifier


def glass_cost_per_m2(shgc: float, vt: float) -> float:
    """
    Rough cost proxy for glazing based on performance.

    Args:
        shgc: Solar heat gain coefficient.
        vt: Visible transmittance.

    Returns:
        Cost per m2 (USD).
    """
    base_cost = 350.0
    performance_premium = 450.0 * (0.6 - shgc) + 200.0 * (vt - 0.5)
    return max(200.0, base_cost + performance_premium)
