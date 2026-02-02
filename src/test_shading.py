#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify shading effectiveness.
"""

import numpy as np
import pandas as pd
from shading_geometry import compute_shading_factors
from solar_geometry import incidence_cos_window

# Test case 1: Sun in front of the wall
print("Test 1: Sun in front of the wall")
solar_zenith = np.array([30])  # 60 degrees elevation
solar_azimuth = np.array([90])  # East
window_azimuth = 90  # East-facing wall
overhang_depth = 1.0

S_b, F_sunlit = compute_shading_factors(
    solar_zenith=solar_zenith,
    solar_azimuth=solar_azimuth,
    window_azimuth=window_azimuth,
    overhang_depth=overhang_depth
)

print(f"Shading factor (S_b): {S_b[0]}")
print(f"Sunlit fraction (F_sunlit): {F_sunlit[0]}")

# Calculate beam POA
cos_theta = incidence_cos_window(
    solar_azimuth=solar_azimuth,
    solar_zenith=solar_zenith,
    window_azimuth=window_azimuth,
    window_tilt=90
)
dni = 800.0  # Typical DNI
beam_poa = dni * cos_theta
print(f"Beam POA: {beam_poa[0]} W/m²")
print(f"Shaded beam POA: {beam_poa[0] * (1 - S_b[0])} W/m²")

# Test case 2: Sun behind the wall
print("\nTest 2: Sun behind the wall")
solar_azimuth = np.array([270])  # West, behind east-facing wall

S_b, F_sunlit = compute_shading_factors(
    solar_zenith=solar_zenith,
    solar_azimuth=solar_azimuth,
    window_azimuth=window_azimuth,
    overhang_depth=overhang_depth
)

print(f"Shading factor (S_b): {S_b[0]}")
print(f"Sunlit fraction (F_sunlit): {F_sunlit[0]}")

# Calculate beam POA
cos_theta = incidence_cos_window(
    solar_azimuth=solar_azimuth,
    solar_zenith=solar_zenith,
    window_azimuth=window_azimuth,
    window_tilt=90
)
beam_poa = dni * cos_theta
print(f"Beam POA: {beam_poa[0]} W/m²")
print(f"Shaded beam POA: {beam_poa[0] * (1 - S_b[0])} W/m²")
