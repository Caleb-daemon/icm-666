#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Economics module for building performance simulation.

This module provides functions to calculate economic metrics for building
retrofits, including Return on Investment (ROI) and Net Present Value (NPV).
"""

import numpy as np
from typing import Tuple, Optional


def ROI_20(
    initial_investment: float,
    annual_savings: float,
    discount_rate: float = 0.05
) -> float:
    """
    Calculate Return on Investment (ROI) over a 20-year period.

    Args:
        initial_investment: Initial investment cost (USD).
        annual_savings: Annual energy savings (USD).
        discount_rate: Discount rate for future cash flows (default 0.05).

    Returns:
        Return on Investment (ROI) as a decimal (e.g., 0.15 for 15%).

    Notes:
        Calculates ROI based on the net present value of 20 years of savings
        minus the initial investment, divided by the initial investment.
        
        Formula:
        ROI = (NPV of savings - initial_investment) / initial_investment
    """
    if initial_investment <= 0:
        return 0.0
    
    # Calculate NPV of savings over 20 years
    npv_savings = NPV(annual_savings, discount_rate, years=20)
    
    # Calculate ROI
    roi = (npv_savings - initial_investment) / initial_investment
    
    return roi


def NPV(
    cash_flow: float,
    discount_rate: float,
    years: int = 20
) -> float:
    """
    Calculate Net Present Value (NPV) of a series of cash flows.

    Args:
        cash_flow: Annual cash flow (USD).
        discount_rate: Discount rate (decimal).
        years: Number of years (default 20).

    Returns:
        Net Present Value (NPV) of the cash flows (USD).

    Notes:
        Implements the standard NPV formula:
        NPV = Σ (cash_flow / (1 + discount_rate)^t) for t from 1 to years
    """
    npv = 0.0
    
    for t in range(1, years + 1):
        # Calculate present value of cash flow for year t
        present_value = cash_flow / ((1 + discount_rate) ** t)
        npv += present_value
    
    return npv


def payback_period(
    initial_investment: float,
    annual_savings: float
) -> float:
    """
    Calculate simple payback period for an investment.

    Args:
        initial_investment: Initial investment cost (USD).
        annual_savings: Annual energy savings (USD).

    Returns:
        Payback period (years).

    Notes:
        Simple payback period is the time required to recover the initial investment
        through annual savings, without considering the time value of money.
        
        Formula:
        payback_period = initial_investment / annual_savings
    """
    if annual_savings <= 0:
        return float('inf')
    
    payback = initial_investment / annual_savings
    
    return payback


def life_cycle_cost(
    initial_investment: float,
    annual_operating_cost: float,
    discount_rate: float = 0.05,
    life_years: int = 20
) -> float:
    """
    Calculate Life Cycle Cost (LCC) of a building or retrofit.

    Args:
        initial_investment: Initial investment cost (USD).
        annual_operating_cost: Annual operating cost (USD).
        discount_rate: Discount rate for future cash flows (default 0.05).
        life_years: Expected life of the system (years, default 20).

    Returns:
        Life Cycle Cost (LCC) over the specified period (USD).

    Notes:
        LCC includes both the initial investment and the present value of
        all future operating costs over the life of the system.
    """
    # Calculate present value of operating costs
    npv_operating = NPV(annual_operating_cost, discount_rate, years=life_years)
    
    # Calculate total life cycle cost
    lcc = initial_investment + npv_operating
    
    return lcc


def energy_cost(
    energy_use: float,
    energy_price: float = 0.15
) -> float:
    """
    Calculate annual energy cost based on energy use and price.

    Args:
        energy_use: Annual energy use (kWh).
        energy_price: Energy price (USD/kWh, default 0.15).

    Returns:
        Annual energy cost (USD).

    Notes:
        Simple calculation of annual energy cost:
        cost = energy_use * energy_price
    """
    cost = energy_use * energy_price
    
    return cost