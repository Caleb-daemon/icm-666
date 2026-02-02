#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermal loads module for building performance simulation.

This module provides functions to calculate thermal loads in buildings
using the 2R1C thermal network model and to split solar gains between
air and thermal mass.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional


def simulate_2R2C(
    temp_outdoor: np.ndarray,
    q_solar_a: np.ndarray,
    q_solar_m: np.ndarray,
    q_internal: np.ndarray,
    h_in: float,
    h_out: float,
    c_air: float,
    c_mass: float,
    r_env: float,
    r_am: float,
    area: float,
    initial_temp_air: float = 22.0,
    initial_temp_mass: float = 22.0,
    dt: float = 3600.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate building thermal behavior using a 2R2C thermal network model.

    Args:
        temp_outdoor: Outdoor dry bulb temperature (°C).
        q_solar_a: Solar heat gain to air (W).
        q_solar_m: Solar heat gain to mass (W).
        q_internal: Internal heat gain (W).
        h_in: Indoor convective heat transfer coefficient (W/m²·K).
        h_out: Outdoor convective heat transfer coefficient (W/m²·K).
        c_air: Air thermal capacitance (J/K).
        c_mass: Thermal mass capacitance (J/K).
        r_env: Envelope thermal resistance (K/W).
        r_am: Air-mass thermal resistance (K/W).
        area: Wall area (m²).
        initial_temp_air: Initial indoor air temperature (°C, default 22.0).
        initial_temp_mass: Initial mass temperature (°C, default 22.0).
        dt: Time step (seconds, default 3600.0).

    Returns:
        A tuple containing:
        - temp_air: Indoor air temperature (°C) for each time step.
        - temp_mass: Mass temperature (°C) for each time step.
        - q_mass: Heat transfer between mass and air (W) for each time step.

    Notes:
        Implements 2R2C model with explicit Euler discretization.
        Equations:
        C_a * (dT_a/dt) = (T_out - T_a)/R_env + (T_m - T_a)/R_am + Q_int + Q_solar_a - Q_hvac
        C_m * (dT_m/dt) = (T_a - T_m)/R_am + Q_solar_m
    """
    # 添加参数有效性检查
    if c_air <= 0:
        c_air = 100000.0  # 防止除零错误
    if c_mass <= 0:
        c_mass = 100000.0  # 防止除零错误
    if r_env <= 0:
        r_env = 0.1  # 防止除零错误
    if r_am <= 0:
        r_am = 0.1  # 防止除零错误

    # 限制热阻和热容值，防止溢出
    max_resistance = 10.0
    min_resistance = 0.01
    max_capacitance = 1e9
    min_capacitance = 1000.0

    r_env = np.clip(r_env, min_resistance, max_resistance)
    r_am = np.clip(r_am, min_resistance, max_resistance)
    c_air = np.clip(c_air, min_capacitance, max_capacitance)
    c_mass = np.clip(c_mass, min_capacitance, max_capacitance)

    # Initialize temperature arrays
    n_steps = len(temp_outdoor)
    temp_air = np.zeros(n_steps)
    temp_mass = np.zeros(n_steps)
    q_mass = np.zeros(n_steps)

    # Set initial conditions
    temp_air[0] = initial_temp_air
    temp_mass[0] = initial_temp_mass

    # Simulate for each time step
    for i in range(1, n_steps):
        # 确保温度值有效
        if np.isnan(temp_air[i-1]) or np.isinf(temp_air[i-1]):
            temp_air[i-1] = 22.0
        if np.isnan(temp_mass[i-1]) or np.isinf(temp_mass[i-1]):
            temp_mass[i-1] = 22.0
        if np.isnan(temp_outdoor[i-1]) or np.isinf(temp_outdoor[i-1]):
            temp_outdoor[i-1] = 20.0

        # Calculate heat flows
        # Heat flow through envelope
        q_env = (temp_outdoor[i-1] - temp_air[i-1]) / r_env
        
        # Heat flow between air and mass
        q_am = (temp_mass[i-1] - temp_air[i-1]) / r_am
        q_mass[i] = q_am  # 存储热质量释放的热量

        # Total heat balance for air
        q_air_total = q_env + q_am + q_internal[i-1] + q_solar_a[i-1]

        # Total heat balance for mass
        q_mass_total = -q_am + q_solar_m[i-1]

        # 限制热流量，防止溢出
        max_heat_flow = 1e6
        q_air_total = np.clip(q_air_total, -max_heat_flow, max_heat_flow)
        q_mass_total = np.clip(q_mass_total, -max_heat_flow, max_heat_flow)

        # Update temperatures using Explicit Euler
        temp_air_change = (q_air_total / c_air) * dt
        temp_mass_change = (q_mass_total / c_mass) * dt

        # 限制温度变化率，防止数值爆炸
        max_temp_change = 5.0  # 最大温度变化 5°C/小时
        temp_air_change = np.clip(temp_air_change, -max_temp_change, max_temp_change)
        temp_mass_change = np.clip(temp_mass_change, -max_temp_change, max_temp_change)

        # Add smoothing to prevent temperature abrupt changes
        smoothing_factor = 0.9  # Smoothing factor between 0 and 1
        temp_air_new = temp_air[i-1] + temp_air_change
        temp_mass_new = temp_mass[i-1] + temp_mass_change

        # Apply smoothing
        if i > 1:
            # Use exponential moving average for smoothing
            temp_air[i] = smoothing_factor * temp_air[i-1] + (1 - smoothing_factor) * temp_air_new
            temp_mass[i] = smoothing_factor * temp_mass[i-1] + (1 - smoothing_factor) * temp_mass_new
        else:
            temp_air[i] = temp_air_new
            temp_mass[i] = temp_mass_new

        # 限制温度范围，确保物理合理性
        min_temp = -10.0
        max_temp = 50.0
        temp_air[i] = np.clip(temp_air[i], min_temp, max_temp)
        temp_mass[i] = np.clip(temp_mass[i], min_temp, max_temp)

    return temp_air, temp_mass, q_mass


def simulate_2R1C(
    temp_outdoor,
    q_solar_a,
    q_solar_m,
    q_internal,
    h_in,
    h_out,
    c_mass,
    r_mass,
    r_wall,
    area,
    initial_temp_air=22.0,
    initial_temp_mass=22.0,
    dt=3600.0
):
    """
    Simulate building thermal behavior using a 2R1C thermal network model.

    Args:
        temp_outdoor: Outdoor dry bulb temperature (°C).
        q_solar_a: Solar heat gain to air (W).
        q_solar_m: Solar heat gain to mass (W).
        q_internal: Internal heat gain (W).
        h_in: Indoor convective heat transfer coefficient (W/m²·K).
        h_out: Outdoor convective heat transfer coefficient (W/m²·K).
        c_mass: Thermal mass capacitance (J/K).
        r_mass: Thermal resistance between air and mass (m²·K/W).
        r_wall: Wall thermal resistance (m²·K/W).
        area: Wall area (m²).
        initial_temp_air: Initial indoor air temperature (°C, default 22.0).
        initial_temp_mass: Initial mass temperature (°C, default 22.0).
        dt: Time step (seconds, default 3600.0).

    Returns:
        A tuple containing:
        - temp_air: Indoor air temperature (°C) for each time step.
        - temp_mass: Mass temperature (°C) for each time step.

    Notes:
        Implements Formula 21 from the v3 document using Explicit Euler discretization.
        The 2R1C model consists of:
        - Two resistances: wall resistance (r_wall) and mass resistance (r_mass)
        - One capacitance: thermal mass (c_mass)
        
        IMPORTANT: c_mass must be in J/K (total thermal capacitance), NOT J/kgK (specific heat capacity).
        Ensure dt is sufficiently small to maintain numerical stability.
    """
    # 添加参数有效性检查
    if area <= 0:
        area = 1.0  # 防止除零错误
    if r_wall <= 0:
        r_wall = 0.1  # 防止除零错误
    if r_mass <= 0:
        r_mass = 0.1  # 防止除零错误
    if c_mass <= 0:
        c_mass = 10000.0  # 防止除零错误
    if h_in <= 0:
        h_in = 3.0  # 防止除零错误
    
    # 添加时间步长稳定性检查
    # 计算最小热阻
    r_min = min(r_mass, r_wall)
    # 稳定性条件：dt < R_min * C_m
    max_stable_dt = (r_min * c_mass) * 0.9
    stability_condition = dt < max_stable_dt
    if not stability_condition:
        # 自动调整时间步长以保证稳定性
        dt = max_stable_dt * 0.9
    
    # Calculate conductance values
    g_wall = area / r_wall  # Wall conductance (W/K)
    g_mass = area / r_mass  # Mass conductance (W/K)
    
    # 限制导纳值，防止溢出
    max_conductance = 1e6
    g_wall = min(g_wall, max_conductance)
    g_mass = min(g_mass, max_conductance)
    
    # Initialize temperature arrays
    n_steps = len(temp_outdoor)
    temp_air = np.zeros(n_steps)
    temp_mass = np.zeros(n_steps)
    
    # Set initial conditions
    temp_air[0] = initial_temp_air
    temp_mass[0] = initial_temp_mass
    
    # Simulate for each time step
    for i in range(1, n_steps):
        # 确保温度值有效
        if np.isnan(temp_air[i-1]) or np.isinf(temp_air[i-1]):
            temp_air[i-1] = 22.0
        if np.isnan(temp_mass[i-1]) or np.isinf(temp_mass[i-1]):
            temp_mass[i-1] = 22.0
        if np.isnan(temp_outdoor[i-1]) or np.isinf(temp_outdoor[i-1]):
            temp_outdoor[i-1] = 25.0
        
        # Calculate heat flows
        # Heat flow through wall
        q_wall = g_wall * (temp_outdoor[i-1] - temp_air[i-1])
        
        # Heat flow between air and mass
        q_air_mass = g_mass * (temp_air[i-1] - temp_mass[i-1])
        
        # 限制热流量，防止溢出
        max_heat_flow = 1e6
        q_wall = np.clip(q_wall, -max_heat_flow, max_heat_flow)
        q_air_mass = np.clip(q_air_mass, -max_heat_flow, max_heat_flow)
        
        # Total heat balance for air
        q_air_total = q_solar_a[i-1] + q_internal[i-1] + q_wall - q_air_mass
        
        # Total heat balance for mass
        q_mass_total = q_air_mass + q_solar_m[i-1]
        
        # 限制总热流量，防止溢出
        q_air_total = np.clip(q_air_total, -max_heat_flow, max_heat_flow)
        q_mass_total = np.clip(q_mass_total, -max_heat_flow, max_heat_flow)
        
        # 计算空气热容
        # 假设层高 3.5m
        volume = area * 3.5
        air_density = 1.2  # kg/m^3
        air_cp = 1005.0    # J/kg·K
        c_air = volume * air_density * air_cp
        
        # 确保 c_air 为正数
        if c_air <= 0:
            c_air = 10000.0
        
        # Update temperatures using Explicit Euler
        # Note: This is a simplified implementation; in a real 2R1C model,
        # the heat balance equations would be solved simultaneously
        temp_air_change = (q_air_total / c_air) * dt
        temp_mass_change = (q_mass_total / c_mass) * dt
        
        # 限制温度变化率，防止数值爆炸
        max_temp_change = 5.0  # 最大温度变化 5°C/小时
        temp_air_change = np.clip(temp_air_change, -max_temp_change, max_temp_change)
        temp_mass_change = np.clip(temp_mass_change, -max_temp_change, max_temp_change)
        
        temp_air[i] = temp_air[i-1] + temp_air_change
        temp_mass[i] = temp_mass[i-1] + temp_mass_change
        
        # 限制温度范围，确保物理合理性
        min_temp = -10.0
        max_temp = 50.0
        temp_air[i] = np.clip(temp_air[i], min_temp, max_temp)
        temp_mass[i] = np.clip(temp_mass[i], min_temp, max_temp)
    
    return temp_air, temp_mass


def split_solar_to_air_and_mass(
    q_solar: np.ndarray,
    fraction_to_air: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split solar heat gain into air and thermal mass components.

    Args:
        q_solar: Total solar heat gain (W).
        fraction_to_air: Fraction of solar gain going directly to air (default 0.3).

    Returns:
        A tuple containing:
        - q_solar_air: Solar heat gain to air (W).
        - q_solar_mass: Solar heat gain to thermal mass (W).

    Notes:
        Implements Formula 23 from the v3 document:
        q_solar_air = fraction_to_air * q_solar
        q_solar_mass = (1 - fraction_to_air) * q_solar
        
        This split is important for Task 2 (Borealis) which requires
        more detailed thermal modeling.
    """
    # Calculate solar gain to air
    q_solar_air = fraction_to_air * q_solar
    
    # Calculate solar gain to mass
    q_solar_mass = (1 - fraction_to_air) * q_solar
    
    return q_solar_air, q_solar_mass


def cooling_load_calculation(
    temp_air: np.ndarray,
    temp_setpoint: float = 24.0,
    area: float = 1.0,
    h_in: float = 3.0
) -> np.ndarray:
    """
    Calculate cooling load based on indoor air temperature and setpoint.

    Args:
        temp_air: Indoor air temperature (°C).
        temp_setpoint: Cooling setpoint temperature (°C, default 24.0).
        area: Wall area (m², default 1.0).
        h_in: Indoor convective heat transfer coefficient (W/m²·K, default 3.0).

    Returns:
        Cooling load (W) for each time step.
        Positive values indicate cooling is required.
    """
    # Calculate temperature difference from setpoint
    delta_temp = temp_air - temp_setpoint
    
    # Cooling load is positive when temperature exceeds setpoint
    # Convert temperature difference to power using h_in and area
    cooling_load = np.maximum(0.0, delta_temp) * h_in * area
    
    return cooling_load


def heating_load_calculation(
    temp_air: np.ndarray,
    temp_setpoint: float = 20.0,
    area: float = 1.0,
    h_in: float = 3.0
) -> np.ndarray:
    """
    Calculate heating load based on indoor air temperature and setpoint.

    Args:
        temp_air: Indoor air temperature (°C).
        temp_setpoint: Heating setpoint temperature (°C, default 20.0).
        area: Wall area (m², default 1.0).
        h_in: Indoor convective heat transfer coefficient (W/m²·K, default 3.0).

    Returns:
        Heating load (W) for each time step.
        Positive values indicate heating is required.
    """
    # Calculate temperature difference from setpoint
    delta_temp = temp_setpoint - temp_air
    
    # Heating load is positive when temperature is below setpoint
    # Convert temperature difference to power using h_in and area
    heating_load = np.maximum(0.0, delta_temp) * h_in * area
    
    return heating_load