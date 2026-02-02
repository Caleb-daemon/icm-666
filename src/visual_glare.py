#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual glare module for building performance simulation.

This module provides functions to calculate visual glare metrics such as
Annual Sunlight Exposure (ASE), which is important for building daylighting
and occupant comfort analysis.
"""

import numpy as np
from typing import Tuple, Optional


def ASE_1000(
    direct_irradiance: np.ndarray,
    F_sunlit: np.ndarray,
    threshold: float = 1000.0,
    VT: float = 0.7
) -> float:
    """
    Calculate Annual Sunlight Exposure (ASE) for a window.

    Args:
        direct_irradiance: Direct solar irradiance on the window (W/m²).
        F_sunlit: Fraction of window area that is sunlit (0-1).
        threshold: Threshold for direct irradiance (W/m², default 1000.0).
        VT: Visible transmittance of the glazing (default 0.7).

    Returns:
        Annual Sunlight Exposure (hours/year) where direct irradiance
        exceeds the threshold and the window is sunlit.

    Notes:
        Implements Formula 25 from the v3 document with glazing transmittance:
        E_dir_in = direct_irradiance * VT
        ASE = Σ (1 for each hour where E_dir_in > threshold and F_sunlit > 0.5)
        
        ASE is defined as the number of hours per year when:
        - Direct sunlight falls on more than 50% of the window area
        - The transmitted direct irradiance exceeds 1000 W/m²
        
        This metric is used to assess visual comfort and glare potential.
    """
    # 计算透射后的直射照度
    E_dir_in = direct_irradiance * VT
    
    # Calculate ASE hours
    # Count hours where transmitted direct irradiance exceeds threshold and window is sunlit
    ase_hours = np.sum((E_dir_in > threshold) & (F_sunlit > 0.5))
    
    return float(ase_hours)


def glare_index(
    direct_irradiance: np.ndarray,
    view_angle: float = 60.0
) -> np.ndarray:
    """
    Calculate a simple glare index based on direct irradiance.

    Args:
        direct_irradiance: Direct solar irradiance on the window (W/m²).
        view_angle: Angle of view towards the window (degrees, default 60.0).

    Returns:
        Glare index for each time step.
        Higher values indicate more severe glare.

    Notes:
        This is a simplified glare index calculation that takes into account
        both the intensity of direct sunlight and the angle at which it is viewed.
    """
    # Convert view angle to radians
    view_angle_rad = np.radians(view_angle)
    
    # Calculate cosine of view angle (affects glare perception)
    cos_view_angle = np.cos(view_angle_rad)
    
    # Calculate glare index
    # Simplified formula: glare increases with irradiance and view angle
    glare = direct_irradiance * (1.0 / (cos_view_angle + 0.1))
    
    return glare


def discomfort_glare_probability(
    glare_index: np.ndarray
) -> np.ndarray:
    """
    Calculate Discomfort Glare Probability (DGP) based on glare index.

    Args:
        glare_index: Glare index values.

    Returns:
        Discomfort Glare Probability (0-1) for each time step.
        Higher values indicate higher probability of discomfort.

    Notes:
        This is a simplified implementation of DGP based on the glare index.
        In practice, DGP calculations are more complex and take into account
        additional factors such as room geometry and background luminance.
    """
    # Calculate DGP using a sigmoid function
    # This is a simplified model that maps glare index to discomfort probability
    dgp = 1.0 / (1.0 + np.exp(-0.001 * (glare_index - 500.0)))
    
    return dgp


def dgp_proxy(
    ev_lux: np.ndarray,
    omega_sr: np.ndarray,
    position_index: float = 1.0
) -> np.ndarray:
    """
    Approximate Daylight Glare Probability (DGP) using a simplified source model.

    Args:
        ev_lux: Vertical eye illuminance (lux).
        omega_sr: Solid angle of the glare source (sr).
        position_index: Position index P_i (dimensionless).

    Returns:
        DGP values (0-1).

    Notes:
        This is a coarse proxy following the DGP structure. It assumes a
        single dominant glare source and uses an approximate luminance term.
    """
    ev_lux = np.maximum(ev_lux, 1.0)
    omega_sr = np.maximum(omega_sr, 1e-6)
    position_index = max(position_index, 1e-3)

    # Approximate source luminance using Ev ~= L * omega * cos(theta).
    cos_view = 0.8
    luminance = ev_lux / (omega_sr * cos_view)

    term = (luminance ** 2) * omega_sr / (ev_lux ** 1.87 * position_index ** 2)
    dgp = 5.87e-5 * ev_lux + 0.0918 * np.log1p(term) + 0.16
    return np.clip(dgp, 0.0, 1.0)
