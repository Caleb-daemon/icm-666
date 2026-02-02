#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Irradiance calculation module using the Perez anisotropic sky model.

This module provides functions to calculate diffuse solar irradiance on
tilted surfaces using the Perez model, which accounts for sky anisotropy.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def perez_D(zenith: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Perez model coefficients D for anisotropic sky diffuse irradiance.

    Args:
        zenith: Solar zenith angle (degrees, from vertical).

    Returns:
        A tuple containing the Perez coefficients:
        - DNI beam irradiance factor (dimensionless)
        - Sky diffuse factor (dimensionless)
        - Ground diffuse factor (dimensionless)

    Notes:
        Implements Formula 7 from the v3 document for calculating the Perez D coefficients.
        These coefficients are used to account for the anisotropy of the sky.
    """
    # Convert zenith to radians
    zenith_rad = np.radians(zenith)
    
    # Calculate cosine of zenith
    cos_zenith = np.cos(zenith_rad)
    
    # Initialize coefficients
    D_beam = np.zeros_like(zenith)
    D_sky = np.zeros_like(zenith)
    D_ground = np.zeros_like(zenith)
    
    # Calculate coefficients based on zenith angle
    # For zenith < 90 degrees (sun above horizon)
    mask = zenith < 90
    
    # Formula 7 implementation
    D_beam[mask] = cos_zenith[mask]
    D_sky[mask] = 0.5 * (1 - cos_zenith[mask])
    D_ground[mask] = 0.5 * (1 + cos_zenith[mask])
    
    # For zenith >= 90 degrees (sun below horizon)
    D_beam[~mask] = 0.0
    D_sky[~mask] = 1.0
    D_ground[~mask] = 0.0
    
    return D_beam, D_sky, D_ground


def tilted_irradiance(
    surface_tilt: float,
    surface_azimuth: float,
    solar_zenith: np.ndarray,
    solar_azimuth: np.ndarray,
    dni: np.ndarray,
    dhi: np.ndarray,
    ghi: np.ndarray,
    albedo: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate total irradiance on a tilted surface using the Perez model.

    Args:
        surface_tilt: Surface tilt angle (degrees, from horizontal).
        surface_azimuth: Surface azimuth angle (degrees, clockwise from north).
        solar_zenith: Solar zenith angle (degrees, from vertical).
        solar_azimuth: Solar azimuth angle (degrees, clockwise from north).
        dni: Direct Normal Irradiance (W/m²).
        dhi: Diffuse Horizontal Irradiance (W/m²).
        ghi: Global Horizontal Irradiance (W/m²).
        albedo: Ground albedo (dimensionless, default 0.2).

    Returns:
        A tuple containing:
        - Direct irradiance on tilted surface (W/m²)
        - Diffuse irradiance on tilted surface (W/m²)
        - Reflected irradiance on tilted surface (W/m²)

    Notes:
        Implements Formulas 8-9 from the v3 document for calculating
        irradiance on tilted surfaces using the Perez anisotropic sky model.
    """
    # Convert angles to radians
    surface_tilt_rad = np.radians(surface_tilt)
    surface_azimuth_rad = np.radians(surface_azimuth)
    solar_zenith_rad = np.radians(solar_zenith)
    solar_azimuth_rad = np.radians(solar_azimuth)
    
    # Calculate cosine of surface tilt
    cos_tilt = np.cos(surface_tilt_rad)
    sin_tilt = np.sin(surface_tilt_rad)
    
    # Calculate angle difference between solar and surface azimuth
    azimuth_diff_rad = solar_azimuth_rad - surface_azimuth_rad
    
    # Calculate cosine of angle of incidence for beam irradiance
    cos_theta = np.cos(solar_zenith_rad) * cos_tilt + \
                np.sin(solar_zenith_rad) * sin_tilt * np.cos(azimuth_diff_rad)
    
    # Clamp to minimum 0
    cos_theta = np.maximum(0, cos_theta)
    
    # Calculate direct irradiance on tilted surface
    direct_tilted = dni * cos_theta
    
    # Calculate Perez D coefficients
    D_beam, D_sky, D_ground = perez_D(solar_zenith)
    
    # Calculate diffuse irradiance on tilted surface (Formula 8)
    diffuse_tilted = dhi * D_sky
    
    # Calculate reflected irradiance on tilted surface (Formula 9)
    reflected_tilted = ghi * albedo * D_ground
    
    return direct_tilted, diffuse_tilted, reflected_tilted


def total_irradiance(
    surface_tilt: float,
    surface_azimuth: float,
    solar_zenith: np.ndarray,
    solar_azimuth: np.ndarray,
    dni: np.ndarray,
    dhi: np.ndarray,
    ghi: np.ndarray,
    albedo: float = 0.2
) -> np.ndarray:
    """
    Calculate total irradiance on a tilted surface.

    Args:
        surface_tilt: Surface tilt angle (degrees, from horizontal).
        surface_azimuth: Surface azimuth angle (degrees, clockwise from north).
        solar_zenith: Solar zenith angle (degrees, from vertical).
        solar_azimuth: Solar azimuth angle (degrees, clockwise from north).
        dni: Direct Normal Irradiance (W/m²).
        dhi: Diffuse Horizontal Irradiance (W/m²).
        ghi: Global Horizontal Irradiance (W/m²).
        albedo: Ground albedo (dimensionless, default 0.2).

    Returns:
        Total irradiance on the tilted surface (W/m²).
    """
    direct, diffuse, reflected = tilted_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        dni=dni,
        dhi=dhi,
        ghi=ghi,
        albedo=albedo
    )
    
    return direct + diffuse + reflected