#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify ASE reduction with shading devices.
"""

import numpy as np
import pandas as pd
from shading_geometry import compute_shading_factors
from solar_geometry import incidence_cos_window
from visual_glare import ASE_1000

# Test parameters
window_azimuth = 90  # East-facing wall
window_tilt = 90  # Vertical
threshold = 50.0  # 50 W/m²

# Generate test data for a day
n_hours = 24
times = pd.date_range('2023-06-21', periods=n_hours, freq='H')

# Simulate solar positions throughout the day
# Simplified solar positions for summer solstice in Hong Kong
solar_zenith = np.array([80, 70, 60, 50, 40, 30, 25, 30, 40, 50, 60, 70, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80])
solar_azimuth = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Calculate beam POA
cos_theta = incidence_cos_window(
    solar_azimuth=solar_azimuth,
    solar_zenith=solar_zenith,
    window_azimuth=window_azimuth,
    window_tilt=window_tilt
)
dni = 800.0  # Typical DNI
beam_poa = dni * cos_theta
print(f"Beam POA range: {beam_poa.min():.1f} - {beam_poa.max():.1f} W/m²")

# Test with no overhang
print("\nTest 1: No overhang")
overhang_depth = 0.0
S_b, F_sunlit = compute_shading_factors(
    solar_zenith=solar_zenith,
    solar_azimuth=solar_azimuth,
    window_azimuth=window_azimuth,
    overhang_depth=overhang_depth
)
# S_b 是 Direct solar blocking factor (0 = fully shaded, 1 = fully sunlit)
beam_poa_shaded = S_b * beam_poa
ase_hours = ASE_1000(beam_poa_shaded, F_sunlit, threshold=threshold)
print(f"ASE hours: {ase_hours}")
print(f"Shading factor range: {S_b.min():.3f} - {S_b.max():.3f}")
print(f"Sunlit fraction range: {F_sunlit.min():.3f} - {F_sunlit.max():.3f}")

# Test with 1m overhang
print("\nTest 2: 1m overhang")
overhang_depth = 1.0
S_b, F_sunlit = compute_shading_factors(
    solar_zenith=solar_zenith,
    solar_azimuth=solar_azimuth,
    window_azimuth=window_azimuth,
    overhang_depth=overhang_depth
)
# S_b 是 Direct solar blocking factor (0 = fully shaded, 1 = fully sunlit)
beam_poa_shaded = S_b * beam_poa
ase_hours = ASE_1000(beam_poa_shaded, F_sunlit, threshold=threshold)
print(f"ASE hours: {ase_hours}")
print(f"Shading factor range: {S_b.min():.3f} - {S_b.max():.3f}")
print(f"Sunlit fraction range: {F_sunlit.min():.3f} - {F_sunlit.max():.3f}")
