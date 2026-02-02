#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shading geometry module for building performance simulation.

This module provides functions to calculate shading factors for building
fenestrations (windows) based on overhangs, fins, and other shading devices.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def compute_shading_factors(
    solar_zenith: np.ndarray,
    solar_azimuth: np.ndarray,
    window_azimuth: float,
    overhang_depth: float,
    left_fin_depth: float = 0.0,
    right_fin_depth: float = 0.0,
    window_height: float = 2.1,
    window_width: float = 1.8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate shading factors for a window with overhang and fins.

    Args:
        solar_zenith: Solar zenith angle (degrees, from vertical).
        solar_azimuth: Solar azimuth angle (degrees, clockwise from north).
        window_azimuth: Window azimuth angle (degrees, clockwise from north).
        overhang_depth: Overhang depth (meters).
        left_fin_depth: Left fin depth (meters, default 0.0).
        right_fin_depth: Right fin depth (meters, default 0.0).
        window_height: Window height (meters, default 2.1).
        window_width: Window width (meters, default 1.8).

    Returns:
        A tuple containing:
        - S_b: Direct solar blocking factor (0 = fully shaded, 1 = fully sunlit).
        - F_sunlit: Fraction of window area that is sunlit (0-1).

    Notes:
        Implements a simplified geometric approximation for overhangs and fins
        based on the logic from Formula 15 in the v3 document.
        Uses vectorized calculations for efficiency.
    """
    # Convert angles to radians
    solar_zenith_rad = np.radians(solar_zenith)
    solar_azimuth_rad = np.radians(solar_azimuth)
    window_azimuth_rad = np.radians(window_azimuth)
    
    # Calculate solar elevation angle
    solar_elevation_rad = np.pi/2 - solar_zenith_rad
    
    # Calculate angle difference between solar and window azimuth
    azimuth_diff_rad = solar_azimuth_rad - window_azimuth_rad
    
    # 添加保护措施，防止未来可能的除零错误
    # 当太阳与墙面平行时，azimuth_diff_rad 接近 ±π/2，cos 值接近 0
    
    # 背阳面判定：太阳是否在墙面这一侧
    is_sun_in_front = np.cos(azimuth_diff_rad) > 0
    
    # Initialize shading factors as float arrays
    S_b = np.ones_like(solar_zenith, dtype=np.float64)
    F_sunlit = np.ones_like(solar_zenith, dtype=np.float64)
    
    # Only calculate shading for sun above horizon
    sun_above_horizon = solar_elevation_rad > 0
    
    # 只有太阳在前面且在水平面以上，才需要计算遮阳
    sun_in_front_and_above = sun_above_horizon & is_sun_in_front
    
    if np.any(sun_in_front_and_above):
        # Calculate shading for sun in front and above horizon
        # Overhang shading calculation
        if overhang_depth > 0:
            # Calculate shadow length from overhang
            shadow_length = overhang_depth * np.tan(solar_elevation_rad[sun_in_front_and_above])
            
            # Calculate fraction of window shaded by overhang
            overhang_shading = np.clip(shadow_length / window_height, 0, 1)
            
            # Update shading factors
            S_b[sun_in_front_and_above] *= (1 - overhang_shading)
            F_sunlit[sun_in_front_and_above] *= (1 - overhang_shading)
        
        # Fin shading calculation
        if left_fin_depth > 0 or right_fin_depth > 0:
            # 角度截断保护，防止 tan 函数爆炸
            diff_abs = np.abs(azimuth_diff_rad[sun_in_front_and_above])
            # 强制限制在 89.9 度以内，防止 tan(90) 爆炸
            azimuth_diff_rad_safe = np.sign(azimuth_diff_rad[sun_in_front_and_above]) * np.clip(diff_abs, 0, np.radians(89.9))
            
            # Calculate horizontal shadow components
            # For left fin
            if left_fin_depth > 0:
                # Calculate shadow from left fin
                left_shadow = left_fin_depth * np.tan(np.abs(azimuth_diff_rad_safe))
                left_shading = np.clip(left_shadow / window_width, 0, 1)
                
                # Update shading factors for left side
                S_b[sun_in_front_and_above] *= (1 - left_shading)
                F_sunlit[sun_in_front_and_above] *= (1 - left_shading)
            
            # For right fin
            if right_fin_depth > 0:
                # Calculate shadow from right fin
                right_shadow = right_fin_depth * np.tan(np.abs(azimuth_diff_rad_safe))
                right_shading = np.clip(right_shadow / window_width, 0, 1)
                
                # Update shading factors for right side
                S_b[sun_in_front_and_above] *= (1 - right_shading)
                F_sunlit[sun_in_front_and_above] *= (1 - right_shading)
    
    # For sun below horizon, no direct sunlight
    S_b[~sun_above_horizon] = 0.0
    F_sunlit[~sun_above_horizon] = 0.0
    
    # Ensure values are within [0, 1]
    S_b = np.clip(S_b, 0, 1)
    F_sunlit = np.clip(F_sunlit, 0, 1)
    
    return S_b, F_sunlit


def shading_factor_vectorized(
    solar_zenith_deg: np.ndarray,
    solar_azimuth_deg: np.ndarray,
    facade: str,
    overhang_depth_m: float
) -> np.ndarray:
    """
    Vectorized shading factor calculation for a building facade.

    Args:
        solar_zenith_deg: Solar zenith angle (degrees).
        solar_azimuth_deg: Solar azimuth angle (degrees).
        facade: Facade direction ('N', 'E', 'S', 'W').
        overhang_depth_m: Overhang depth in meters.

    Returns:
        Shading factor (0 = fully shaded, 1 = fully sunlit) for each time step.

    Notes:
        This function is designed to be used in vectorized calculations
        for building performance simulation.
    """
    # Define facade azimuth angles
    FACADE_AZIMUTH = {
        'N': 0.0,
        'E': 90.0,
        'S': 180.0,
        'W': 270.0
    }
    
    # Get azimuth for the specified facade
    facade_azimuth = FACADE_AZIMUTH.get(facade, 0.0)
    
    # Calculate shading factors
    S_b, _ = compute_shading_factors(
        solar_zenith=solar_zenith_deg,
        solar_azimuth=solar_azimuth_deg,
        window_azimuth=facade_azimuth,
        overhang_depth=overhang_depth_m
    )
    
    return S_b