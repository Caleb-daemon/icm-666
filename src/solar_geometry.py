#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solar geometry module for building performance simulation.

This module provides functions to calculate solar position and incidence angles
for building surfaces and windows, which are essential for solar irradiance
and thermal load calculations.
"""

import pandas as pd
import numpy as np
import pvlib
from typing import Tuple, Optional


def calculate_solar_position(
    times: pd.DatetimeIndex, 
    latitude: float, 
    longitude: float, 
    altitude: float = 0.0,
    timezone: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate solar position for a given location and time range.

    Args:
        times: Datetime index for which to calculate solar position.
        latitude: Latitude of the location (degrees).
        longitude: Longitude of the location (degrees).
        altitude: Altitude of the location (meters).
        timezone: Timezone string (e.g., 'Asia/Hong_Kong').

    Returns:
        A pandas DataFrame with solar position data, including:
        - 'azimuth': Solar azimuth angle (degrees, clockwise from north).
        - 'zenith': Solar zenith angle (degrees, from vertical).
        - 'elevation': Solar elevation angle (degrees, from horizontal).
    """
    # Use pvlib's solar position calculator
    # Fix parameter name: 'times' should be 'time'
    solar_position = pvlib.solarposition.get_solarposition(
        time=times,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        method='nrel_numpy'
    )
    
    # Return only the relevant columns
    return solar_position[['azimuth', 'zenith', 'elevation']]


def incidence_cos_window(
    solar_azimuth: np.ndarray, 
    solar_zenith: np.ndarray, 
    window_azimuth: float, 
    window_tilt: float
) -> np.ndarray:
    """
    Calculate the cosine of the angle of incidence for a window surface.

    Args:
        solar_azimuth: Solar azimuth angle (degrees, clockwise from north).
        solar_zenith: Solar zenith angle (degrees, from vertical).
        window_azimuth: Window azimuth angle (degrees, clockwise from north).
        window_tilt: Window tilt angle (degrees, from horizontal, 90 for vertical).

    Returns:
        The cosine of the angle of incidence (dimensionless).
        Values are clamped to a minimum of 0 (no negative irradiance).

    Notes:
        Implements Formula 6 from the v3 document:
        cos(θ) = sin(β) * cos(φ) + cos(β) * sin(φ) * cos(α - γ)
        where:
        - θ is the angle of incidence
        - β is the solar elevation angle (90 - zenith)
        - φ is the window tilt angle (from horizontal)
        - α is the solar azimuth angle
        - γ is the window azimuth angle
    """
    # Convert angles to radians
    solar_azimuth_rad = np.radians(solar_azimuth)
    solar_zenith_rad = np.radians(solar_zenith)
    window_azimuth_rad = np.radians(window_azimuth)
    window_tilt_rad = np.radians(window_tilt)
    
    # Calculate solar elevation angle (β)
    solar_elevation_rad = np.pi/2 - solar_zenith_rad
    
    # Calculate angle difference between solar and window azimuth
    azimuth_diff_rad = solar_azimuth_rad - window_azimuth_rad
    
    # Implement Formula 6
    term1 = np.sin(solar_elevation_rad) * np.cos(window_tilt_rad)
    term2 = np.cos(solar_elevation_rad) * np.sin(window_tilt_rad) * np.cos(azimuth_diff_rad)
    cos_theta = term1 + term2
    
    # Clamp to minimum 0
    cos_theta = np.maximum(0, cos_theta)
    
    return cos_theta


def aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth):
    """
    Calculate angle of incidence (AOI) for a surface.

    Args:
        surface_tilt: Surface tilt angle (degrees, from horizontal).
        surface_azimuth: Surface azimuth angle (degrees, clockwise from north).
        solar_zenith: Solar zenith angle (degrees, from vertical).
        solar_azimuth: Solar azimuth angle (degrees, clockwise from north).

    Returns:
        Angle of incidence (degrees).
    """
    return pvlib.irradiance.aoi(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth
    )