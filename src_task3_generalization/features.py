#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Climate feature extraction module."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .adapters import read_epw

# Season definitions
SEASONS = {
    'spring': {'months': [3, 4, 5]},
    'summer': {'months': [6, 7, 8]},
    'autumn': {'months': [9, 10, 11]},
    'winter': {'months': [12, 1, 2]}
}

def get_season(month: int) -> str:
    """Get season from month."""
    for season, info in SEASONS.items():
        if month in info['months']:
            return season
    return 'unknown'

def compute_hdd(temperatures: np.ndarray, base_temp: float = 18.0) -> float:
    """Calculate Heating Degree Days."""
    return np.sum(np.maximum(0, base_temp - temperatures))

def compute_cdd(temperatures: np.ndarray, base_temp: float = 24.0) -> float:
    """Calculate Cooling Degree Days."""
    return np.sum(np.maximum(0, temperatures - base_temp))

def compute_solar_potential(dni: np.ndarray, cos_theta: Optional[np.ndarray] = None) -> float:
    """Calculate solar potential."""
    if cos_theta is not None:
        return np.sum(dni * np.maximum(0, cos_theta))
    return np.sum(dni)

def compute_cloud_proxy(dhi: np.ndarray, ghi: np.ndarray) -> float:
    """Calculate cloudiness proxy."""
    valid_indices = ghi > 0
    if not np.any(valid_indices):
        return 0.0
    return np.mean(dhi[valid_indices] / ghi[valid_indices])

def compute_humid_proxy(rh: Optional[np.ndarray] = None) -> float:
    """Calculate humidity proxy."""
    if rh is None:
        return 0.0
    return np.mean(rh)

def compute_t_range(temp_max: Optional[np.ndarray] = None, temp_min: Optional[np.ndarray] = None, temp: Optional[np.ndarray] = None) -> float:
    """Calculate temperature range."""
    if temp_max is not None and temp_min is not None:
        return np.mean(temp_max - temp_min)
    elif temp is not None:
        # Calculate daily range from hourly data
        # Reshape to days x hours
        if len(temp) % 24 == 0:
            temp_daily = temp.reshape(-1, 24)
            return np.mean(np.max(temp_daily, axis=1) - np.min(temp_daily, axis=1))
    return 0.0

def compute_climate_features(epw_df: pd.DataFrame, meta: Dict[str, Any], masks: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    """Calculate climate features from EPW data.
    
    Args:
        epw_df: EPW dataframe.
        meta: Metadata containing location information.
        masks: Optional masks for different seasons.
        
    Returns:
        Dictionary of climate features.
    """
    features = {}
    
    # Extract temperature data
    if 'temp_air' in epw_df.columns:
        temp_air = epw_df['temp_air'].values
    elif 'Temp' in epw_df.columns:
        temp_air = epw_df['Temp'].values
    else:
        temp_air = np.zeros(len(epw_df))
    
    # Extract solar data
    dni = epw_df.get('dni', epw_df.get('DNI', np.zeros(len(epw_df)))).values
    dhi = epw_df.get('dhi', epw_df.get('DHI', np.zeros(len(epw_df)))).values
    ghi = epw_df.get('ghi', epw_df.get('GHI', np.zeros(len(epw_df)))).values
    
    # Extract humidity data
    rh = epw_df.get('relative_humidity', epw_df.get('RH', None))
    if rh is not None:
        rh = rh.values
    
    # Extract temperature extremes
    temp_max = epw_df.get('temp_max', None)
    temp_min = epw_df.get('temp_min', None)
    if temp_max is not None:
        temp_max = temp_max.values
    if temp_min is not None:
        temp_min = temp_min.values
    
    # Calculate annual features
    features['phi'] = meta.get('latitude', 0.0)
    features['HDD18'] = compute_hdd(temp_air, 18.0)
    features['CDD24'] = compute_cdd(temp_air, 24.0)
    features['Cloud_proxy'] = compute_cloud_proxy(dhi, ghi)
    features['Humid_proxy'] = compute_humid_proxy(rh)
    features['T_range'] = compute_t_range(temp_max, temp_min, temp_air)
    
    # Calculate seasonal features
    for season in SEASONS.keys():
        # Create season mask
        season_mask = epw_df.index.month.isin(SEASONS[season]['months'])
        
        # Calculate solar potential for season
        season_dni = dni[season_mask]
        features[f'S_{season}'] = compute_solar_potential(season_dni)
    
    # Calculate summer and winter solar potential (for backward compatibility)
    summer_mask = epw_df.index.month.isin(SEASONS['summer']['months'])
    winter_mask = epw_df.index.month.isin(SEASONS['winter']['months'])
    
    features['S_summer'] = compute_solar_potential(dni[summer_mask])
    features['S_winter'] = compute_solar_potential(dni[winter_mask])
    
    return features

def compute_seasonal_features(epw_df: pd.DataFrame, meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Calculate features for each season.
    
    Args:
        epw_df: EPW dataframe.
        meta: Metadata containing location information.
        
    Returns:
        Dictionary of seasonal features.
    """
    seasonal_features = {}
    
    for season in SEASONS.keys():
        # Create season mask
        season_mask = epw_df.index.month.isin(SEASONS[season]['months'])
        season_df = epw_df[season_mask]
        
        if len(season_df) > 0:
            seasonal_features[season] = compute_climate_features(season_df, meta)
    
    return seasonal_features

def infer_params_from_results(results_dir: str) -> Dict[str, Any]:
    """Infer parameters from results files.
    
    Args:
        results_dir: Directory containing results files.
        
    Returns:
        Dictionary with 'params' and 'logs' keys.
    """
    import os
    import re
    
    params = {}
    logs = []
    
    # Files to check
    files_to_check = [
        os.path.join('task1', 'tables', 'task1_hourly_best.csv'),
        os.path.join('task1', 'tables', 'task1_best_solution.csv'),
        os.path.join('task2', 'tables', 'task2_best_solution.csv'),
        os.path.join('task2', 'tables', 'task2_pareto_results.csv'),
    ]
    
    for file_name in files_to_check:
        file_path = os.path.join(results_dir, file_name)
        if not os.path.exists(file_path):
            logs.append(f"File not found: {file_path}")
            continue
        
        try:
            df = pd.read_csv(file_path)
            logs.append(f"Reading file: {file_path}")
            
            # Analyze columns
            for col in df.columns:
                # Skip obvious metric columns
                metric_patterns = [
                    r'E_cool', r'E_heat', r'ASE', r'GR', r'R', r'cost', r'fitness',
                    r'cooling', r'heating', r'energy', r'savings', r'penalty',
                    r'roi', r'lifecycle', r'capex', r'opex'
                ]
                
                is_metric = any(re.search(pattern, col, re.IGNORECASE) for pattern in metric_patterns)
                if is_metric:
                    logs.append(f"Skipping metric column: {col}")
                    continue
                
                # Check if column contains numerical data
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Score column as potential parameter
                    score = 0
                    
                    # Check for parameter-like names
                    param_patterns = [
                        r'wwr', r'shade', r'overhang', r'depth',
                        r'mass', r'c_m', r'thermal', r'insulation',
                        r'glass', r'window', r'u_value', r'shgc'
                    ]
                    
                    for pattern in param_patterns:
                        if re.search(pattern, col, re.IGNORECASE):
                            score += 2
                            logs.append(f"Column {col} matched pattern {pattern}, score +2")
                    
                    # Check for reasonable values
                    col_min = df[col].min()
                    col_max = df[col].max()
                    if 0 <= col_min <= col_max <= 10:
                        score += 1
                        logs.append(f"Column {col} has reasonable range [{col_min:.2f}, {col_max:.2f}], score +1")
                    
                    if score > 0:
                        # Use the median value as the inferred parameter
                        params[col] = float(df[col].median())
                        logs.append(f"Selected parameter {col} with value {params[col]:.4f}, score {score}")
        
        except Exception as e:
            logs.append(f"Error reading {file_path}: {e}")
    
    # Fallback: if no parameters found, try to use any numerical column
    if not params:
        logs.append("No parameters found, trying fallback")
        for file_name in files_to_check:
            file_path = os.path.join(results_dir, file_name)
            if not os.path.exists(file_path):
                continue
            
            try:
                df = pd.read_csv(file_path)
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # Check if column has reasonable variation
                        if df[col].std() > 0.01:
                            params[col] = float(df[col].median())
                            logs.append(f"Fallback: selected parameter {col} with value {params[col]:.4f}")
                            break
            except:
                pass
    
    return {'params': params, 'logs': logs}

def classify_climate(features: Dict[str, Any]) -> Dict[str, str]:
    """Classify climate based on features.
    
    Args:
        features: Climate features.
        
    Returns:
        Dictionary of climate classifications.
    """
    classification = {}
    
    # Cooling/Heating/Mixed
    hdd = features.get('HDD18', 0)
    cdd = features.get('CDD24', 0)
    
    if hdd > 2 * cdd:
        classification['load_type'] = 'heating'
    elif cdd > 2 * hdd:
        classification['load_type'] = 'cooling'
    else:
        classification['load_type'] = 'mixed'
    
    # Solar potential
    s_summer = features.get('S_summer', 0)
    if s_summer > 1e9:
        classification['solar'] = 'high'
    else:
        classification['solar'] = 'low'
    
    # Humidity
    humid = features.get('Humid_proxy', 0)
    if humid > 60:
        classification['humidity'] = 'humid'
    else:
        classification['humidity'] = 'dry'
    
    # Temperature swing
    t_range = features.get('T_range', 0)
    if t_range > 10:
        classification['swing'] = 'large'
    else:
        classification['swing'] = 'small'
    
    return classification
